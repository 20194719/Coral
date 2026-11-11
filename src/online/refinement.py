"""Fixed-group additive probability correction and label priorities."""

from __future__ import annotations
import numpy as np


class Feedback:
    def __init__(self, initial, groups, strength=1.0):
        """Initialize model state."""
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.groups = np.asarray(groups, dtype=np.int32).copy()
        if self.initial.shape != self.groups.shape or not self.initial.size:
            raise ValueError("Feedback arrays must have equal, nonempty shapes")
        if not np.isfinite(self.initial).all() or np.any(
            (self.initial < 0) | (self.initial > 1)
        ):
            raise ValueError("Initial probabilities must lie in [0, 1]")
        if np.any(self.groups < 0) or not np.isfinite(strength) or strength <= 0:
            raise ValueError(
                "Group IDs must be nonnegative and prior strength positive"
            )
        self.strength = float(strength)
        self.size = int(self.groups.max()) + 1
        self.population = np.bincount(self.groups.ravel(), minlength=self.size)
        total = np.bincount(
            self.groups.ravel(), weights=self.initial.ravel(), minlength=self.size
        )
        self.mean = total / np.maximum(self.population, 1)
        self.values = self.initial.copy()
        self.known = np.zeros(self.initial.shape, dtype=bool)
        self.labels = {}
        self.count = np.zeros(self.size, dtype=np.int64)
        self.positive = np.zeros(self.size)
        self.residual = np.zeros(self.size)

    def observe(self, index, label):
        """Record a binary label."""
        index, label = int(index), float(label)
        if not 0 <= index < self.initial.size or label not in (0.0, 1.0):
            raise ValueError("Invalid feedback index or label")
        if index in self.labels:
            if self.labels[index] != label:
                raise ValueError("Conflicting labels for one prediction")
            return
        group = self.groups.flat[index]
        self.labels[index] = label
        self.known.flat[index] = True
        self.count[group] += 1
        self.positive[group] += label
        self.residual[group] += label - self.initial.flat[index]

    def recompute(self):
        """Update calibrated probabilities."""
        offset = self.residual / (self.strength + self.count)
        self.values = np.clip(self.initial + offset[self.groups], 0.0, 1.0)
        for index, label in self.labels.items():
            self.values.flat[index] = label

    def priorities(self, influence):
        """Compute sampling priorities."""
        influence = np.broadcast_to(influence, self.initial.shape)
        remaining = np.where(self.known, 0.0, influence)
        weight = np.bincount(
            self.groups.ravel(), weights=remaining.ravel(), minlength=self.size
        )
        n = self.strength + self.count
        mean = (self.strength * self.mean + self.positive) / n
        variance = weight**2 * mean * (1.0 - mean) / n
        gain = variance / (n + 1.0)
        return gain, variance

    def state_dict(self):
        """State dict."""
        return dict(
            prior_strength=self.strength,
            observations=[[int(i), float(y)] for i, y in sorted(self.labels.items())],
            group_counts=self.count.tolist(),
            group_residuals=self.residual.tolist(),
            group_offsets=(self.residual / (self.strength + self.count)).tolist(),
        )
