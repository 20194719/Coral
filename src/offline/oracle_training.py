"""Label first-round and expansion samples using relation annotations."""

from dataclasses import dataclass
import numpy as np
from src.data.storage import atomic_json
from src.offline.expansion import expand_pool, second_round
from src.offline.records import LabeledSet, DiscoveredTypes, _assign_C_strata


@dataclass
class SeedPairs:
    pairs: np.ndarray

    def __len__(self):
        """Return the number of records."""
        return len(self.pairs)


def build(cfg, pool, sample, positives, q_theta, theta_name, *, e_R, e_S):
    """Build reusable method artifacts."""
    types = DiscoveredTypes(
        labels=[theta_name], q_types=np.asarray(q_theta, np.float32).reshape(1, -1)
    )
    seed_pairs = np.column_stack((sample.r, sample.s)).astype(np.int64)
    seed_labels = np.array([tuple(p) in positives for p in seed_pairs], dtype=bool)
    expansion = expand_pool(
        cfg,
        pool,
        SeedPairs(seed_pairs),
        types,
        e_R,
        e_S,
        seed_labels[:, None],
        per_anchor=20,
    )
    selected, selected_source, audit = second_round(cfg, pool, seed_pairs, expansion)
    pairs = np.concatenate((seed_pairs, selected))
    source = np.concatenate(
        (np.full(len(seed_pairs), "seed", dtype=object), selected_source)
    )
    strata_by_key = dict(zip(pool.key().tolist(), _assign_C_strata(cfg, pool).tolist()))
    strata = np.concatenate(
        (
            sample.stratum,
            np.array(
                [
                    strata_by_key.get(int(r) * pool.n_S + int(s), "background")
                    for r, s in selected
                ],
                dtype=object,
            ),
        )
    )
    labels = np.array([tuple(p) in positives for p in pairs], dtype=bool)
    np.savez_compressed(
        cfg.results_dir / "offline-sampling.npz",
        pairs=pairs,
        labels=labels,
        source=source.astype(str),
        expansion_candidates=expansion.pairs,
        expansion_selected=selected[selected_source == "expand"],
        supplemental_selected=selected[selected_source == "supplement"],
    )
    atomic_json(
        cfg.results_dir / "offline-sampling.json",
        dict(audit, distinct_labeled_pairs=len(pairs)),
    )
    return (
        LabeledSet(
            n_R=pool.n_R,
            n_S=pool.n_S,
            N=sample.N,
            type_labels=[theta_name],
            pairs=pairs,
            rel_type=np.zeros(len(pairs), dtype=np.int64),
            label=labels,
            incl_prob=np.concatenate((sample.incl_prob, np.ones(len(selected)))),
            stratum=strata,
            source=source,
            strata_mass=dict(sample.mass),
        ),
        types,
    )
