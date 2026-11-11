"""Match entity names in documents."""

from __future__ import annotations
import re
from typing import List

STOP = {"and", "the", "of", "for", "with", "in", "to", "a", "an"}


def _norm(t: str) -> str:
    """Normalize dictionary names."""
    return re.sub("[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def scan(texts: List[str], names: List[str], min_len: int = 5) -> List[List[str]]:
    """Find dictionary mentions in documents."""
    pats = []
    for n in names:
        c = _norm(n)
        if len(c) < min_len or c in STOP:
            continue
        pats.append((c, re.compile("\\b" + re.escape(c) + "\\b")))
    out: List[List[str]] = []
    for t in texts:
        low = _norm(t)
        out.append([c for c, p in pats if p.search(low)])
    return out
