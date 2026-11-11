"""Offline data records."""

from __future__ import annotations
import hashlib
import json
from src.online.sampling import STRATA
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class CandidatePool:
    n_R: int
    n_S: int
    pairs: np.ndarray
    bm25_score: np.ndarray
    dense_score: np.ndarray
    mention_flag: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        """Return the number of records."""
        return self.pairs.shape[0]

    def key(self) -> np.ndarray:
        """Return encoded pair identifiers."""
        return self.pairs[:, 0].astype(np.int64) * self.n_S + self.pairs[:, 1]


@dataclass
class StratifiedSample:
    r: np.ndarray
    s: np.ndarray
    stratum: np.ndarray
    incl_prob: np.ndarray
    mass: Dict[str, int]
    N: int


def _assign_C_strata(cfg, pool: CandidatePool) -> np.ndarray:
    """Build strata."""
    has_b = pool.bm25_score > 0
    has_d = pool.dense_score > -1.0
    bm_hi_thr = (
        np.quantile(pool.bm25_score[has_b], cfg.strat_hi_q) if has_b.any() else np.inf
    )
    dn_hi_thr = (
        np.quantile(pool.dense_score[has_d], cfg.strat_hi_q) if has_d.any() else np.inf
    )
    high_b = has_b & (pool.bm25_score >= bm_hi_thr)
    high_d = has_d & (pool.dense_score >= dn_hi_thr)
    out = np.empty(len(pool), dtype=object)
    out[:] = "medium"
    out[high_b & ~high_d] = "bm25_high"
    out[high_d & ~high_b] = "dense_high"
    out[high_b & high_d] = "both_high"
    return out


C_STRATA = set(STRATA[:-1])


@dataclass
class LabeledSet:
    n_R: int
    n_S: int
    N: int
    type_labels: List[str]
    pairs: np.ndarray
    rel_type: np.ndarray
    label: np.ndarray
    incl_prob: np.ndarray
    stratum: np.ndarray
    source: np.ndarray
    strata_mass: Dict[str, int]

    @property
    def T(self) -> int:
        """Return the relation count."""
        return len(self.type_labels)


def label_fingerprint(labeled):
    """Label fingerprint."""
    digest = hashlib.sha256(
        json.dumps(list(labeled.type_labels), ensure_ascii=False).encode()
    )
    for values, dtype in (
        (labeled.pairs, np.int64),
        (labeled.rel_type, np.int64),
        (labeled.label, bool),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    digest.update(json.dumps(labeled.stratum.tolist()).encode())
    return digest.hexdigest()


def label_pairs(llm, runner, stage, texts_R, texts_S, pairs, types):
    """Label pairs."""
    if not types.T:
        raise ValueError("No discovered relation types to judge")
    if not len(pairs):
        return np.zeros((0, types.T), dtype=bool)

    def _label():
        """Collect relation labels."""
        replies = llm.relation_spec(
            types.labels, [(texts_R[int(r)], texts_S[int(s)]) for r, s in pairs]
        )
        if len(replies) != len(pairs):
            raise ValueError(f"{stage}: missing pair judgments")
        labels = np.empty((len(pairs), types.T), dtype=bool)
        for i, reply in enumerate(replies):
            for t, name in enumerate(types.labels):
                entry = reply.get(name) if isinstance(reply, dict) else None
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("holds"), bool
                ):
                    raise ValueError(
                        f"{stage}: invalid judgment for pair {i}, type {name!r}"
                    )
                labels[i, t] = entry["holds"]
        return labels

    labels = runner.build(stage, _label)
    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (len(pairs), types.T)
        or labels.dtype != bool
    ):
        raise ValueError(f"{stage}: invalid offline label matrix")
    return labels


def _seed_rows(cards, types, seed_labels):
    """Seed rows."""
    rows = []
    if seed_labels.shape != (len(cards), types.T):
        raise ValueError("Seed labels do not match the seed pairs and frozen types")
    for i in range(len(cards)):
        st = cards.stratum[i]
        incl = 1.0 if st in C_STRATA else float(cards.incl_prob[i])
        for t in range(types.T):
            holds = bool(seed_labels[i, t])
            rows.append(
                (
                    int(cards.pairs[i, 0]),
                    int(cards.pairs[i, 1]),
                    t,
                    holds,
                    incl,
                    st,
                    "seed",
                )
            )
    return rows


def harvest(
    cfg,
    llm,
    runner,
    texts_R,
    texts_S,
    pool,
    sample,
    cards,
    types,
    expansion,
    seed_labels=None,
) -> LabeledSet:
    """Harvest."""
    n_R, n_S = (pool.n_R, pool.n_S)
    if seed_labels is None:
        seed_labels = label_pairs(
            llm, runner, "seed_relation_labels", texts_R, texts_S, cards.pairs, types
        )
    from src.offline.expansion import second_round
    from src.data.storage import atomic_json

    selected, source, audit = second_round(cfg, pool, cards.pairs, expansion)
    harv_pairs = [tuple(map(int, p)) for p in selected]
    harv_src = source.tolist()
    runner.build("second_round_pairs", lambda: dict(pairs=selected, source=source))
    atomic_json(cfg.results_dir / "offline-sampling.json", audit)
    harv_labels = label_pairs(
        llm, runner, "harvest_labels", texts_R, texts_S, harv_pairs, types
    )
    rows = _seed_rows(cards, types, seed_labels)
    strata_of_C = _assign_C_strata(cfg, pool)
    stratum_by_key = dict(zip(pool.key().tolist(), strata_of_C.tolist()))
    for (r, s), src, lab in zip(harv_pairs, harv_src, harv_labels):
        st = stratum_by_key.get(int(r) * n_S + int(s), "background")
        for t, name in enumerate(types.labels):
            holds = bool(lab[t])
            rows.append((r, s, t, holds, 1.0, st, src))
    if not rows:
        raise RuntimeError("harvest produced no rows; check mining stages")
    pairs = np.array([[r[0], r[1]] for r in rows], dtype=np.int64)
    rel_type = np.array([r[2] for r in rows], dtype=np.int64)
    label = np.array([r[3] for r in rows], dtype=bool)
    incl = np.array([r[4] for r in rows], dtype=np.float64)
    stratum = np.array([r[5] for r in rows], dtype=object)
    source = np.array([r[6] for r in rows], dtype=object)
    return LabeledSet(
        n_R=n_R,
        n_S=n_S,
        N=sample.N,
        type_labels=list(types.labels),
        pairs=pairs,
        rel_type=rel_type,
        label=label,
        incl_prob=incl,
        stratum=stratum,
        source=source,
        strata_mass=dict(sample.mass),
    )


@dataclass
class DiscoveredTypes:
    labels: List[str]
    q_types: np.ndarray
    representatives: List[List[str]] = field(default_factory=list)

    @property
    def T(self):
        """Return the relation count."""
        return len(self.labels)


@dataclass
class ExpansionCandidates:
    pairs: np.ndarray

    def __len__(self):
        """Return the number of records."""
        return len(self.pairs)
