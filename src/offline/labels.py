"""Extract seed observations."""

from dataclasses import dataclass
import numpy as np
from src.online.sampling import STRATA

SEED_CALIBRATION_POLICY = "seed_sample_sum"


def seed_row_mask(labeled):
    """Seed row mask."""
    sources = np.asarray(labeled.source)
    if sources.shape != (len(labeled.pairs),):
        raise ValueError(
            "offline labels need one source per row to identify seed samples"
        )
    return sources == "seed"


def seed_pair_count(labeled):
    """Seed pair count."""
    pairs = labeled.pairs[seed_row_mask(labeled)]
    return int(np.unique(pairs[:, 0] * int(labeled.n_S) + pairs[:, 1]).size)


@dataclass
class SeedLabels:
    pairs: np.ndarray
    labels: np.ndarray
    strata: np.ndarray


def selected_seed_labels(labeled, selected_type: int) -> SeedLabels:
    """Selected seed labels."""
    if type(selected_type) is not int or selected_type < 0:
        raise ValueError("selected_type must be one nonnegative integer")
    seed = seed_row_mask(labeled)
    expected = {tuple(map(int, p)) for p in labeled.pairs[seed]}
    observations = {}
    for i in np.flatnonzero(seed & (np.asarray(labeled.rel_type) == selected_type)):
        pair = tuple(map(int, labeled.pairs[i]))
        st = labeled.stratum[i]
        y = labeled.label[i]
        if st not in STRATA or y not in (0, 1):
            raise ValueError("seed observations need a valid stratum and binary label")
        value = (STRATA.index(st), int(y))
        if pair in observations and observations[pair] != value:
            raise ValueError("conflicting duplicate seed labels for selected condition")
        observations[pair] = value
    if not expected or set(observations) != expected:
        raise ValueError(
            "selected condition must label every distinct first-batch seed pair"
        )
    pairs = sorted(observations)
    return SeedLabels(
        np.asarray(pairs, np.int64),
        np.array([observations[p][1] for p in pairs], np.float64),
        np.array([observations[p][0] for p in pairs], np.int64),
    )
