"""Numeric filter prediction."""

from __future__ import annotations
import json
import re
import numpy as np
from src.offline.profiles import ProfileHead, ProfileStore
from src.prompts import NUMERIC_FILTER_PARSING_PROMPT

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}
_EXISTS = {"is_present": True, "is_absent": False}
MIN_PRESENT = 10


def parse_spec(reply: str, allowed) -> dict | None:
    """Parse spec."""
    txt = re.sub("```(?:json)?", "", reply or "").strip()
    if txt.lower().startswith("null"):
        return None
    a, b = (txt.find("{"), txt.rfind("}"))
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(txt[a : b + 1])
    except Exception:
        return None
    if d.get("field") not in allowed or d.get("op") not in {**_OPS, **_EXISTS}:
        return None
    res = d.get("residual")
    res = (
        res.strip() if isinstance(res, str) and res.strip().lower() != "null" else None
    )
    if d["op"] in _EXISTS:
        return {"field": d["field"], "op": d["op"], "residual": res}
    try:
        d["value"] = float(d["value"])
    except Exception:
        return None
    return {"field": d["field"], "op": d["op"], "value": d["value"], "residual": res}


class NumericHead(ProfileHead):
    def __init__(
        self,
        store: ProfileStore,
        cfg,
        llm,
        fields,
        what: str,
        min_coverage: float = 0.5,
        model_options=None,
    ):
        """Initialize model state."""
        super().__init__(store, model_options)
        self.cfg, self.llm, self.what = (cfg, llm, what)
        self.min_coverage = float(min_coverage)
        self.num_fields = [f["name"] for f in fields if f["type"] == "number"]
        self._spec_cache: dict = {}

    def spec_for(self, phi: str):
        """Spec for."""
        if phi not in self._spec_cache:
            if not self.num_fields:
                self._spec_cache[phi] = None
            else:
                desc = "\n".join((f"  - {n}" for n in self.num_fields))
                template = NUMERIC_FILTER_PARSING_PROMPT.replace(
                    "male actors", self.cfg.numeric_example_subject
                )
                rep = self.llm.complete(
                    [template.format(what=self.what, fields=desc, phi=phi)],
                    max_tokens=80,
                )[0]
                self._spec_cache[phi] = parse_spec(rep, set(self.num_fields))
        return self._spec_cache[phi]

    def proba_pushdown(self, phi: str, femb: np.ndarray, encode=None):
        """Proba pushdown."""
        base = super().proba(femb).astype(np.float64)
        spec = self.spec_for(phi)
        if spec is None:
            return (base, None)
        vals = np.array(
            [d.get(spec["field"]) if d else None for d in self.store.nums], dtype=object
        )
        known = np.array([isinstance(v, (int, float)) for v in vals])
        cov = float(known.mean())
        if spec["op"] in _EXISTS:
            if int(known.sum()) < MIN_PRESENT:
                return (base, dict(spec=spec, coverage=cov, applied=False))
            out = (known == _EXISTS[spec["op"]]).astype(np.float64)
        else:
            if cov < self.min_coverage:
                return (base, dict(spec=spec, coverage=cov, applied=False))
            out = base.copy()
            if known.any():
                v = np.array([float(x) for x in vals[known]])
                out[known] = _OPS[spec["op"]](v, spec["value"]).astype(np.float64)
        resid = spec.get("residual")
        if resid and encode is not None:
            out = out * super().proba(encode(resid)).astype(np.float64)
        return (
            out,
            dict(
                spec=spec, coverage=cov, applied=True, residual=resid if resid else None
            ),
        )
