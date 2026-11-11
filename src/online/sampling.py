"""Offline strata and fixed document probability bands."""

import numpy as np

STRATA = ("both_high", "bm25_high", "dense_high", "medium", "background")


def quantile_strata(probabilities, groups):
    """Quantile strata."""
    if groups < 1 or len(probabilities) == 0:
        raise ValueError("Probability bands need documents and a positive group count")
    groups = min(groups, len(probabilities))
    order = np.argsort(probabilities, kind="stable")
    result = np.empty(len(order), dtype=np.int32)
    for group, indices in enumerate(np.array_split(order, groups)):
        result[indices] = group
    return result
