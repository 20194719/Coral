"""Probability calibration."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from src.online.sampling import STRATA
from src.offline.labels import SEED_CALIBRATION_POLICY, SeedLabels


def sigmoid(z):
    """Sigmoid."""
    z = np.asarray(z, np.float64)
    return np.exp(-np.logaddexp(0.0, -z))


def probabilities(score, offset):
    """Probabilities."""
    if np.isnan(offset):
        raise ValueError("NaN calibration offset")
    return sigmoid(np.asarray(score, np.float64) + offset)


def fit_offset(scores, target):
    """Fit offset."""
    scores = np.asarray(scores, np.float64).ravel()
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("calibration needs nonempty, finite labelled scores")
    if not np.isfinite(target) or not 0 <= target <= len(scores):
        raise ValueError("calibration target must lie between zero and sample size")
    if target == 0:
        return -np.inf
    if target == len(scores):
        return np.inf
    lo, hi = (-float(scores.max()) - 40.0, -float(scores.min()) + 40.0)
    while probabilities(scores, lo).sum() > target:
        lo -= 40.0
    while probabilities(scores, hi).sum() < target:
        hi += 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if probabilities(scores, mid).sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def offset_json(value):
    """Offset json."""
    return "-inf" if value == -np.inf else "+inf" if value == np.inf else float(value)


@dataclass
class SampleCalibration:
    offsets: np.ndarray
    sample_n: np.ndarray
    sample_targets: np.ndarray
    sample_achieved: np.ndarray

    def audit(self):
        """Return calibration diagnostics."""
        return dict(
            policy=SEED_CALIBRATION_POLICY,
            offsets=[offset_json(x) for x in self.offsets],
            sample_n=self.sample_n.tolist(),
            sample_label_sums=self.sample_targets.tolist(),
            sample_probability_sums=self.sample_achieved.tolist(),
        )


def calibrate(score, strat, samples: SeedLabels):
    """Calibrate."""
    score, strat = (np.asarray(score), np.asarray(strat))
    pairs = np.asarray(samples.pairs)
    if score.ndim != 2 or strat.shape != score.shape or (not np.isfinite(score).all()):
        raise ValueError("invalid score/stratum grid")
    if np.any(strat < 0) or np.any(strat >= len(STRATA)):
        raise ValueError("invalid grid stratum")
    if not np.issubdtype(strat.dtype, np.integer) or not np.issubdtype(
        pairs.dtype, np.integer
    ):
        raise ValueError("strata and pair coordinates must be integers")
    if pairs.ndim != 2 or pairs.shape != (len(samples.labels), 2) or (not len(pairs)):
        raise ValueError("calibration needs labelled pair coordinates")
    if (
        (pairs < 0).any()
        or (pairs[:, 0] >= score.shape[0]).any()
        or (pairs[:, 1] >= score.shape[1]).any()
    ):
        raise ValueError("calibration pair outside grid")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("calibration pairs must be deduplicated")
    if not np.isin(samples.labels, [0, 1]).all():
        raise ValueError("calibration labels must be binary")
    ts = strat[pairs[:, 0], pairs[:, 1]]
    if not np.array_equal(ts, samples.strata):
        raise ValueError("seed strata disagree with pair-score strata")
    offsets = np.zeros(len(STRATA))
    ns = np.zeros(len(STRATA), np.int64)
    targets = np.zeros(len(STRATA))
    achieved = np.zeros(len(STRATA))
    for t in range(len(STRATA)):
        mask = ts == t
        ns[t] = int(mask.sum())
        if not ns[t]:
            continue
        xy = pairs[mask]
        logits = score[xy[:, 0], xy[:, 1]]
        targets[t] = float(samples.labels[mask].sum())
        offsets[t] = fit_offset(logits, targets[t])
        achieved[t] = float(probabilities(logits, offsets[t]).sum())
    return SampleCalibration(offsets, ns, targets, achieved)


def apply_calibration(score, strat, cal):
    """Apply calibration."""
    return sigmoid(
        np.asarray(score, np.float64) + cal.offsets[np.asarray(strat, np.int64)]
    )


@dataclass(frozen=True)
class CalibratedGrid:
    score: np.ndarray
    strata: np.ndarray
    pi: np.ndarray
    calibration: SampleCalibration
    fingerprint: str

    @classmethod
    def build(cls, score, strata, samples):
        """Build reusable method artifacts."""
        score = np.array(score, np.float64, copy=True)
        strata = np.array(strata, copy=True)
        cal = calibrate(score, strata, samples)
        strata = strata.astype(np.int8)
        pi = apply_calibration(score, strata, cal)
        digest = hashlib.sha256()
        for arr in (score, strata, pi):
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
            arr.setflags(write=False)
        return cls(score, strata, pi, cal, digest.hexdigest())

    @classmethod
    def uncalibrated(cls, score, strata):
        """Uncalibrated."""
        score = np.array(score, np.float64, copy=True)
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite query scores")
        strata = np.asarray(strata, dtype=np.int8)
        pi = sigmoid(score)
        fingerprint = hashlib.sha256(pi.tobytes()).hexdigest()
        score.setflags(write=False)
        pi.setflags(write=False)
        return cls(score, strata, pi, None, fingerprint)

    @property
    def M_hat(self):
        """Return the estimated join size."""
        return float(self.pi.sum())
