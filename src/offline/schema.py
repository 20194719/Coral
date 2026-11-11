"""Profile schema induction."""

from __future__ import annotations
from src.prompts import PROFILE_SCHEMA_INDUCTION_PROMPT
import json
import re
import numpy as np


def induce(cfg, llm, texts, what, ex1, ex2, n, seed):
    """Induce."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    docs = "\n\n".join((f"[DOC {i}]\n{texts[int(d)]}" for i, d in enumerate(idx)))
    prompt = PROFILE_SCHEMA_INDUCTION_PROMPT.format(
        what=what, n=len(idx), ex1=ex1, ex2=ex2, docs=docs
    )
    reply = llm.complete([prompt], max_tokens=1200)[0]
    txt = re.sub("```(?:json)?", "", reply or "")
    start, end = (txt.find("["), txt.rfind("]"))
    try:
        fields = json.loads(txt[start : end + 1])
    except Exception:
        fields = None
    return (fields, reply)


def validate_schema(schema):
    """Validate schema."""
    if not isinstance(schema, dict):
        raise ValueError("Profile schema must be an object with R and S fields")
    for side in ("R", "S"):
        fields = schema.get(side)
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"Profile schema has no fields for {side}")
        names = set()
        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(f"Invalid schema field for {side}")
            name = field.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Invalid or duplicate field name for {side}: {name}")
            if field.get("type") not in ("list", "string", "number"):
                raise ValueError(f"Unsupported field type for {side}.{name}")
            names.add(name)
    return schema
