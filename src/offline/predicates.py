"""Discover predicates from corpus profiles and document excerpts."""

from collections import Counter
import hashlib
import json
import numpy as np
from src.prompts import FILTER_SIMPLE_PREDICATES as FILTER_PREDICATES
from src.prompts import TREX_FILTER_SIMPLE_PREDICATES
from src.data.storage import atomic_json


def profile_statistics(profiles, fields):
    """Profile statistics."""
    summaries = []
    for field in fields:
        values = [p.get(field["name"]) if p else None for p in profiles]
        summary = dict(name=field["name"], type=field["type"])
        if field["type"] == "number":
            numbers = [
                float(v)
                for v in values
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and np.isfinite(v)
            ]
            present = len(numbers)
            if numbers:
                summary["quantiles"] = dict(
                    zip(
                        ("min", "p25", "median", "p75", "max"),
                        np.quantile(numbers, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                    )
                )
        else:
            counts, present = Counter(), 0
            for value in values:
                items = value if isinstance(value, list) else [value]
                terms = {
                    " ".join(v.split()).casefold()
                    for v in items
                    if isinstance(v, str) and v.strip()
                }
                present += bool(terms)
                counts.update(terms)
            summary["distinct_values"] = len(counts)
            summary["top_values"] = [
                dict(value=value[:160], documents=count)
                for value, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[
                    :10
                ]
            ]
        summary.update(
            present_documents=present,
            missing_documents=len(profiles) - present,
        )
        summaries.append(summary)
    return dict(documents=len(profiles), fields=summaries)


def generate(cfg, stage, table, role, side, llm, output, profiles, fields):
    """Generate training filter predicates."""
    if len(profiles) != len(table):
        raise ValueError("Profile count differs from table size")
    count = cfg["filter_predicate_count"]
    ids = np.random.default_rng(cfg["seed_global"]).choice(
        len(table), min(cfg["filter_predicate_sample"], len(table)), replace=False
    )

    docs = "\n\n".join(f"[DOCUMENT {i}]\n{table.texts[int(i)][:1800]}" for i in ids)
    evidence = {}
    if cfg["dataset"] == "trex":
        prompt = TREX_FILTER_SIMPLE_PREDICATES.format(
            what=role, count=count, docs=docs
        )
    else:
        statistics = profile_statistics(profiles, fields)
        evidence["profile_statistics"] = statistics
        prompt = FILTER_PREDICATES.format(
            what=role,
            count=count,
            docs=docs,
            statistics=json.dumps(statistics, ensure_ascii=False),
        )
    signature = hashlib.sha256(prompt.encode()).hexdigest()

    def discover():
        """Discover corpus filter predicates."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "filter_predicates",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "predicates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": count,
                            "maxItems": count,
                        }
                    },
                    "required": ["predicates"],
                    "additionalProperties": False,
                },
            },
        }
        for attempt in range(3):
            reply = llm.complete(
                [prompt],
                max_tokens=1600 * (attempt + 1),
                response_format=response_format,
            )[0]
            try:
                values = json.loads(reply)["predicates"]
                if (
                    isinstance(values, list)
                    and len(values) == count
                    and all(isinstance(v, str) and v.strip() for v in values)
                    and len({" ".join(v.casefold().split()) for v in values}) == count
                ):
                    return [v.strip() for v in values]
            except (ValueError, KeyError, TypeError):
                pass
        raise ValueError("Invalid generated filter predicates after retries")

    predicates = stage.build(f"filter_predicates_{side}_{signature[:12]}", discover)
    atomic_json(
        output / f"filter-predicates-{table.name}.json",
        dict(
            table=table.name,
            document_ids=ids.tolist(),
            document_excerpt_chars=1800,
            source="corpus-llm",
            **evidence,
            prompt_sha256=signature,
            predicates=predicates,
        ),
    )
    return predicates
