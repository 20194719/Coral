"""Build filter similarity features."""

import numpy as np
from scipy.stats import rankdata


def context_features(values, mode):
    """Context features."""
    if mode == "fields":
        return values
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 0.025)
    normalized = (values - mean) / scale
    if mode == "field-z":
        return np.column_stack([values, normalized])
    base = values[:, :2]
    statistics = np.concatenate(
        [
            base.mean(axis=0),
            base.std(axis=0),
            np.quantile(base, [0.1, 0.5, 0.9], axis=0).ravel(),
        ]
    )
    context = np.broadcast_to(statistics, (len(values), len(statistics)))
    if mode == "context":
        return np.column_stack([values, context])
    ranks = np.column_stack(
        [rankdata(base[:, i], method="average") / len(base) for i in range(2)]
    )
    parts = [values, normalized, ranks]
    if mode == "interactions":
        parts.append(
            (values[:, :, None] * mean[None, None, :]).reshape(len(values), -1)
        )
    elif mode != "wide":
        raise ValueError(f"Unknown filter features: {mode}")
    return np.column_stack([*parts, context])
