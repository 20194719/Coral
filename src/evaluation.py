"""Evaluate estimates after inference using per-query median seed Q-error."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def qerror(estimate: float, truth: int) -> float:
    """Qerror."""
    if not math.isfinite(estimate) or estimate < 0 or truth < 0:
        raise ValueError("Cardinalities must be finite and nonnegative")
    estimate, truth = max(float(estimate), 1.0), max(float(truth), 1.0)
    return max(estimate / truth, truth / estimate)


def percentile(values: list[float], fraction: float) -> float:
    """Percentile."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Percentiles require nonempty values and a fraction in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows: list[dict], truth: dict[int, int], seeds: tuple[int, ...]) -> dict:
    """Summarize."""
    expected = {(qid, seed) for qid in truth for seed in seeds}
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty, unique sequence")
    seen, errors, calls = set(), defaultdict(list), []
    for row in rows:
        key = (row["qid"], row["seed"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate query/seed {key}")
        seen.add(key)
        errors[row["qid"]].append(qerror(float(row["estimate"]), truth[row["qid"]]))
        count = row["judgments"]
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid judgment count for {key}")
        calls.append(count)
    if seen != expected:
        raise ValueError(f"Missing query/seed results: {sorted(expected - seen)}")
    medians = {qid: statistics.median(values) for qid, values in errors.items()}
    values = list(medians.values())
    return {
        "queries": len(truth),
        "seeds": list(seeds),
        **{
            label: percentile(values, level)
            for label, level in (
                ("median", 0.5),
                ("p90", 0.9),
                ("p99", 0.99),
                ("max", 1.0),
            )
        },
        "mean_online_judgments": statistics.mean(calls),
        "per_query_median_qerror": medians,
        "aggregation": "Median Q-error across seeds for each query, then linear percentiles across queries.",
    }
