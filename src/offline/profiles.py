"""Profile features and prediction."""

from __future__ import annotations
import re
import src.inference.embeddings as sbert
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


def attr_items(profile: Optional[dict], fields: List[dict]):
    """Attr items."""
    if not profile:
        return []
    out = []
    for index, f in enumerate(f for f in fields if f["type"] != "number"):
        name = f["name"].replace("_", " ")
        v = profile.get(f["name"])
        if isinstance(v, list):
            out += [(index, f"{name}: {x}") for x in v[:12] if x]
        elif isinstance(v, str) and v.strip():
            out.append((index, f"{name}: {v.strip()}"))
    return out[:24]


def attr_texts(profile: Optional[dict], fields: List[dict]) -> List[str]:
    """Attr texts."""
    return [text for _, text in attr_items(profile, fields)]


def numbers(profile: Optional[dict], fields: List[dict]) -> Dict[str, Optional[float]]:
    """Numbers."""
    if not profile:
        return {}
    return {f["name"]: profile.get(f["name"]) for f in fields if f["type"] == "number"}


@dataclass
class ProfileStore:
    attr_mat: np.ndarray
    attr_doc: np.ndarray
    doc_emb: np.ndarray
    n_docs: int
    nums: List[Dict[str, Optional[float]]] = field(default_factory=list)
    attr_field: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    field_names: List[str] = field(default_factory=list)

    texts: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, cfg, profiles, fields, doc_emb, texts=None):
        """Build reusable method artifacts."""
        lists = [attr_items(p, fields) for p in profiles]
        flat, owner, field_ids = ([], [], [])
        for d, xs in enumerate(lists):
            flat += [text for _, text in xs]
            field_ids += [index for index, _ in xs]
            owner += [d] * len(xs)
        emb = (
            sbert.encode(cfg, flat)
            if flat
            else np.zeros((0, cfg.sbert_dim), np.float32)
        )
        return cls(
            attr_mat=emb.astype(np.float32),
            attr_doc=np.asarray(owner, np.int64),
            doc_emb=doc_emb.astype(np.float32),
            n_docs=len(profiles),
            nums=[numbers(p, fields) for p in profiles],
            texts=[] if texts is None else list(texts),
            attr_field=np.asarray(field_ids, np.int64),
            field_names=[f["name"] for f in fields if f["type"] != "number"],
        )

    def feats(self, femb: np.ndarray) -> np.ndarray:
        """Compute document similarity features."""
        g_attr = np.full(self.n_docs, -np.inf, np.float32)
        if len(self.attr_mat):
            sims = self.attr_mat @ femb.astype(np.float32)
            np.maximum.at(g_attr, self.attr_doc, sims)
        g_doc = self.doc_emb @ femb.astype(np.float32)
        miss = ~np.isfinite(g_attr)
        g_attr[miss] = g_doc[miss]
        return np.stack([g_attr, g_doc], axis=1)

    def field_feats(self, femb: np.ndarray) -> np.ndarray:
        """Compute profile field similarities."""
        if not getattr(self, "field_names", None):
            raise ValueError("Field features require profile field metadata")
        if len(self.attr_field) != len(self.attr_doc):
            raise ValueError("Profile attribute field metadata is incomplete")
        values = np.full((self.n_docs, len(self.field_names)), -1.0, np.float32)
        if len(self.attr_mat):
            similarities = self.attr_mat @ femb.astype(np.float32)
            np.maximum.at(values, (self.attr_doc, self.attr_field), similarities)
        return np.column_stack([self.feats(femb), values])


class ProfileHead:
    def __init__(self, store: ProfileStore, model_options=None):
        """Initialize model state."""
        self.store = store
        self.model_options = dict(model_options or {})
        self._m = None

    def proba(self, femb: np.ndarray) -> np.ndarray:
        """Predict filter probabilities."""
        if self._m is None:
            F = self.store.feats(femb)
            return 1.0 / (1.0 + np.exp(-6.0 * (F[:, 0] - 0.5)))
        return self._m.predict(self.store.field_feats(femb), self.store.doc_emb, femb)


_NUM = re.compile("^-?\\d+(?:\\.\\d+)?$")


def parse_batch(reply: str, fields: List[dict], n: int) -> List[Optional[dict]]:
    """Parse batch."""
    from src.prompts import _extract_json_array

    arr = _extract_json_array(reply) or []
    out: List[Optional[dict]] = [None] * n
    names = [f["name"] for f in fields]
    types = {f["name"]: f["type"] for f in fields}
    for i in range(min(n, len(arr))):
        item = arr[i]
        if not isinstance(item, dict):
            continue
        norm = {}
        for k in names:
            v = item.get(k)
            t = types[k]
            if t == "number":
                if isinstance(v, (int, float)):
                    norm[k] = float(v)
                elif isinstance(v, str) and _NUM.match(v.strip()):
                    norm[k] = float(v)
                else:
                    norm[k] = None
            elif t == "list":
                norm[k] = (
                    [str(x).strip() for x in v if str(x).strip()]
                    if isinstance(v, list)
                    else []
                )
            else:
                norm[k] = "" if v is None else str(v)
        out[i] = norm
    return out
