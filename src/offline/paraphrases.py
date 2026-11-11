"""Generate reviewed paraphrases while preserving each condition's meaning."""

import json
import re
from pathlib import Path
from src.data.storage import atomic_json
from src.prompts import FILTER_PARAPHRASE_GENERATION, FILTER_PARAPHRASE_REVIEW


def normalized(text):
    """Normalized."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def shape(name, field, properties):
    """Build the structured response schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    field: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties),
                            "additionalProperties": False,
                        },
                    }
                },
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


def generate_table(llm, predicates, role, directory, count=3):
    """Generate table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    families = []
    for index, text in enumerate(predicates):
        path = directory / f"family-{index:02}.json"
        row = (
            json.loads(path.read_text())
            if path.exists()
            else dict(id=index, original=text, accepted=[], attempts=[])
        )
        if row["id"] != index or row["original"] != text:
            raise ValueError(
                "Paraphrase checkpoint differs from the original conditions"
            )
        families.append(row)
    used = {normalized(t) for t in predicates}
    for row in families:
        used.update(normalized(r["text"]) for r in row["accepted"])
    for round_id in range(8):
        pending = [r for r in families if len(r["accepted"]) < count]
        if not pending:
            return families
        prompts = [
            FILTER_PARAPHRASE_GENERATION
            + json.dumps(
                dict(
                    record_type=role,
                    original=r["original"],
                    count=count,
                    accepted=[a["text"] for a in r["accepted"]],
                    previous_candidates=list(
                        dict.fromkeys(
                            a["text"]
                            for attempt in r["attempts"]
                            for a in attempt["reviewed"]
                        )
                    ),
                )
            )
            for r in pending
        ]
        answers = llm.complete(
            prompts,
            900,
            response_format=shape(
                "filter_paraphrases", "paraphrases", {"text": {"type": "string"}}
            ),
        )
        candidates = []
        for row, answer in zip(pending, answers):
            parsed = json.loads(answer)["paraphrases"]
            if len(parsed) != count or not all(
                isinstance(p["text"], str) and p["text"].strip() for p in parsed
            ):
                raise ValueError(
                    "Each generation must return three nonempty paraphrases"
                )
            candidates.append(
                [dict(id=i, text=p["text"].strip()) for i, p in enumerate(parsed)]
            )
        requests = [
            FILTER_PARAPHRASE_REVIEW
            + json.dumps(
                dict(
                    record_type=role,
                    original=row["original"],
                    already_accepted=[r["text"] for r in row["accepted"]],
                    candidates=items,
                )
            )
            for row, items in zip(pending, candidates)
        ]
        reviews = llm.complete(
            requests,
            1200,
            response_format=shape(
                "paraphrase_review",
                "reviews",
                {
                    "id": {"type": "integer"},
                    "equivalent": {"type": "boolean"},
                    "distinct": {"type": "boolean"},
                    "reason": {"type": "string", "maxLength": 240},
                },
            ),
        )
        for row, items, answer in zip(pending, candidates, reviews):
            checked = json.loads(answer)["reviews"]
            if len(checked) != count or {r["id"] for r in checked} != set(range(count)):
                raise ValueError("Paraphrase review IDs do not match the candidates")
            checked = {r["id"]: r for r in checked}
            attempt = dict(round=round_id, reviewed=[])
            for item in items:
                review = checked[item["id"]]
                key = normalized(item["text"])
                accepted = (
                    review["equivalent"]
                    and review["distinct"]
                    and key not in used
                    and len(row["accepted"]) < count
                )
                record = dict(
                    text=item["text"],
                    equivalent=review["equivalent"],
                    distinct=review["distinct"],
                    reason=review["reason"],
                    accepted=bool(accepted),
                )
                attempt["reviewed"].append(record)
                if accepted:
                    row["accepted"].append(record)
                    used.add(key)
            row["attempts"].append(attempt)
            atomic_json(directory / f"family-{row['id']:02}.json", row)
        print(
            f"[paraphrases] {role}: {sum(len(r['accepted']) for r in families)}/{len(families) * count}",
            flush=True,
        )
    if any(len(r["accepted"]) != count for r in families):
        raise RuntimeError(
            "Paraphrase review remains incomplete; preserve accepted families"
        )
    return families
