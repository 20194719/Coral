"""Train independent stratum models and prepare reusable relation probabilities."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from src.data.storage import Stages, atomic_json
import src.offline.model as M
import src.offline.records as R
from src.offline.labels import selected_seed_labels, seed_pair_count
from src.offline.expansion import expand_pool
import src.offline.preparation as P
from src.online.calibration import CalibratedGrid
from src.prompts import selection_prompt, parse_selection


@dataclass
class Cards:
    pairs: np.ndarray
    stratum: np.ndarray
    incl_prob: np.ndarray

    def __len__(self):
        """Return the number of records."""
        return len(self.pairs)


class PairScores:
    def __init__(
        self,
        cfg,
        view,
        paths,
        tables,
        encoder,
        llm,
        output,
        positives=None,
        resume=False,
    ):
        """Initialize model state."""
        self.cfg, self.view, self.encoder, self.llm = cfg, view, encoder, llm
        self.oracle = positives is not None
        self.output = Path(output)
        prior_audit = self.output / "offline-audit.json"
        self.previous_selections = (
            json.loads(prior_audit.read_text()).get("selections", {})
            if resume and prior_audit.exists()
            else {}
        )
        stages = Stages(self.output / "stages", resume=resume)
        self.emb = [
            np.load(self.output / f"{table.name}-embeddings.npy")
            if resume and (self.output / f"{table.name}-embeddings.npy").exists()
            else encoder(table.texts)
            for table in tables
        ]
        for table, values in zip(tables, self.emb):
            np.save(self.output / f"{table.name}-embeddings.npy", values)
        params = P.parameters(cfg, view, encoder, self.output)
        llm.entities = view["entities"]
        self.pool = P.candidate_pool(params, view, stages, tables, self.emb, llm)
        self.sample = P.seeds(params, stages, self.pool)
        if (self.pool.n_R, self.pool.n_S) != tuple(map(len, tables)):
            raise ValueError("Offline candidate pool does not match table dimensions")
        self.strata = R._assign_C_strata(params, self.pool)
        self.labeled, self.types = self._labels(
            params, view, paths, tables, stages, positives
        )
        stages.build("training_labels", lambda: self.labeled)
        self.grid = M.build_grid(
            self.pool,
            *self.emb,
            self.strata,
            tower_dim=cfg["tower_dimension"],
            seed=cfg["seed_global"],
        )
        self.spec = M.Spec(**view["model_spec"])
        fingerprint = R.label_fingerprint(self.labeled)
        rows = M.build_rows(self.labeled, pool=self.pool, strata_of_C=self.strata)
        if len(rows) != len(self.labeled.label):
            raise ValueError("All offline labels must enter model training")
        qt = np.asarray(self.types.q_types, np.float32)
        qt = qt / np.maximum(np.linalg.norm(qt, axis=1, keepdims=True), 1e-12)
        target = self.output / "pair-models.pt"
        if resume and target.exists():
            checkpoint = torch.load(target, map_location="cpu", weights_only=False)
            if checkpoint["label_fingerprint"] != fingerprint or checkpoint[
                "spec"
            ] != vars(self.spec):
                raise ValueError(
                    "Saved join model does not match the current labels or settings"
                )
            self.model = M.StratifiedModel(
                {
                    str(int(group)): M.PairModel(self.spec, self.grid.D)
                    for group in np.unique(self.grid.strat)
                }
            )
            self.model.load_state_dict(checkpoint["state"])
            self.model.eval()
            print("[resume] trained join models", flush=True)
        else:
            self.model = M.train(
                rows, self.grid, qt, self.spec, device="cpu", verbose=True
            )
        if not (resume and target.exists()):
            torch.save(
                dict(
                    state=self.model.state_dict(),
                    spec=vars(self.spec),
                    tower_U=self.grid.U,
                    label_fingerprint=fingerprint,
                ),
                target,
            )
        self.baselines, self.precomputed = {}, {}
        self.audit = dict(
            checkpoint=str(target),
            training_fingerprint=fingerprint,
            training_rows=len(self.labeled.label),
            seed_distinct_pairs=seed_pair_count(self.labeled),
            labeled_distinct_pairs=int(np.unique(self.labeled.pairs, axis=0).shape[0]),
            stratum_training_rows={
                name: int((rows.strat == j).sum()) for j, name in enumerate(M.STRATA)
            },
            spec=vars(self.spec),
            relations={},
            selections=dict(self.previous_selections),
        )
        for index, (label, q) in enumerate(zip(self.types.labels, qt)):
            score = M.score_grid(self.model, self.grid, q, device="cpu")
            baseline = CalibratedGrid.build(
                score, self.grid.strat, selected_seed_labels(self.labeled, index)
            )
            self.precomputed[index] = self._save_grid(f"relation-{index}", baseline)
            self.audit["relations"][index] = dict(
                condition=label,
                calibration=baseline.calibration.audit(),
                M_hat=baseline.M_hat,
                fingerprint=baseline.fingerprint,
            )
        atomic_json(self.output / "offline-audit.json", self.audit)

    def _labels(self, params, view, paths, tables, stages, positives):
        """Labels."""
        if positives is not None:
            from src.offline.oracle_training import build

            condition = paths["condition"]
            vector = self.encoder([condition])[0]
            return stages.build(
                "oracle_training",
                lambda: build(
                    params,
                    self.pool,
                    self.sample,
                    positives,
                    vector,
                    condition,
                    e_R=self.emb[0],
                    e_S=self.emb[1],
                ),
            )
        types = P.relation_types(params, view, stages, tables, self.sample, self.llm)
        cards = Cards(
            np.column_stack((self.sample.r, self.sample.s)),
            self.sample.stratum,
            self.sample.incl_prob,
        )
        texts = [table.texts for table in tables]
        seed = R.label_pairs(
            self.llm, stages, "seed_relation_labels", *texts, cards.pairs, types
        )
        expansion = stages.build(
            "expansion",
            lambda: expand_pool(
                params, self.pool, cards, types, *self.emb, seed, per_anchor=20
            ),
        )
        labeled = R.harvest(
            params,
            self.llm,
            stages,
            *texts,
            self.pool,
            self.sample,
            cards,
            types,
            expansion,
            seed_labels=seed,
        )
        return labeled, types

    def _save_grid(self, name, grid):
        """Save grid."""
        score_path, pi_path = (
            self.output / f"{name}-scores.npy",
            self.output / f"{name}-probabilities.npy",
        )
        np.save(score_path, grid.score)
        np.save(pi_path, grid.pi)
        return CalibratedGrid(
            np.load(score_path, mmap_mode="r"),
            self.grid.strat,
            np.load(pi_path, mmap_mode="r"),
            grid.calibration,
            grid.fingerprint,
        )

    def _match(self, predicate):
        """Match."""
        if self.oracle:
            return 0
        if predicate in self.previous_selections:
            return self.previous_selections[predicate]["condition_id"]
        exact = [
            i
            for i, label in enumerate(self.types.labels)
            if label.strip().casefold() == predicate.strip().casefold()
        ]
        if exact:
            return exact[0]
        prompt = selection_prompt(predicate, list(self.types.labels))
        return parse_selection(self.llm.complete([prompt], 128)[0], self.types.T)

    def baseline(self, predicate):
        """Baseline."""
        if predicate not in self.baselines:
            selected = self._match(predicate)
            if selected is None:
                q = self.encoder([predicate])[0]
                q = (q / np.linalg.norm(q)).astype(np.float32)
                score = M.score_grid(self.model, self.grid, q, device="cpu")
                baseline = CalibratedGrid.uncalibrated(score, self.grid.strat)
            else:
                baseline = self.precomputed[selected]
            self.baselines[predicate] = baseline
            self.audit["selections"][predicate] = dict(
                condition_id=selected,
                condition=self.types.labels[selected] if selected is not None else None,
                source="precomputed" if selected is not None else "query-model",
                calibration=baseline.calibration.audit()
                if baseline.calibration is not None
                else None,
                M_hat=baseline.M_hat,
                fingerprint=baseline.fingerprint,
            )
            atomic_json(self.output / "offline-audit.json", self.audit)
            print(
                f'[pair] {self.view["name"]}: source={self.audit["selections"][predicate]["source"]} M={baseline.M_hat:.6f}',
                flush=True,
            )
        return self.baselines[predicate]
