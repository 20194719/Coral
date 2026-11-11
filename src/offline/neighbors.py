"""Exact inner-product neighbors for positive-pair expansion."""

from __future__ import annotations
import numpy as np

try:
    import faiss

    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


class NearestIndex:
    def __init__(self, vecs: np.ndarray):
        """Initialize model state."""
        self.vecs = np.ascontiguousarray(vecs.astype(np.float32))
        self.n, self.d = self.vecs.shape
        self._faiss = None
        if _HAVE_FAISS and self.n > 0:
            idx = faiss.IndexFlatIP(self.d)
            idx.add(self.vecs)
            self._faiss = idx

    def search(self, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve nearest embedding neighbors."""
        q = np.ascontiguousarray(queries.astype(np.float32))
        if q.ndim == 1:
            q = q[None, :]
        k = min(k, self.n)
        if k <= 0:
            return (
                np.zeros((q.shape[0], 0), np.float32),
                np.zeros((q.shape[0], 0), np.int64),
            )
        if self._faiss is not None:
            return self._faiss.search(q, k)
        sims = q @ self.vecs.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(q.shape[0])[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, order]
        return (sims[rows, idx], idx)
