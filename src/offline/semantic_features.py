"""Generate and review reusable semantic predicate features."""

import json
from src.prompts import FILTER_FEATURE_GENERATION, FILTER_FEATURE_REVIEW


def response_shape(name, field, properties):
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


class SemanticFeatures:
    def __init__(self, llm, role, examples=None):
        """Initialize model state."""
        self.llm, self.role = llm, role
        self.examples = list(examples or [])
        self.cache = {}

    def annotate(self, texts):
        """Describe and review predicate semantics."""
        rows = [dict(id=i, predicate=text) for i, text in enumerate(texts)]
        payload = dict(
            record_type=self.role, vocabulary_examples=self.examples, predicates=rows
        )
        reply = self.llm.complete(
            [FILTER_FEATURE_GENERATION + json.dumps(payload)],
            8192,
            response_format=response_shape(
                "semantic_features",
                "predicates",
                {
                    "id": {"type": "integer"},
                    "features": {"type": "array", "items": {"type": "string"}},
                },
            ),
        )[0]
        values = json.loads(reply)["predicates"]
        if len(values) != len(rows) or {r["id"] for r in values} != set(
            range(len(rows))
        ):
            raise ValueError("Semantic feature IDs do not match predicates")
        mapping = {r["id"]: r["features"] for r in values}
        generated = [
            dict(id=r["id"], text=r["predicate"], features=mapping[r["id"]])
            for r in rows
        ]
        reply = self.llm.complete(
            [FILTER_FEATURE_REVIEW + json.dumps(generated)],
            4096,
            response_format=response_shape(
                "feature_review",
                "reviews",
                {
                    "id": {"type": "integer"},
                    "accept": {"type": "boolean"},
                },
            ),
        )[0]
        checked = json.loads(reply)["reviews"]
        if len(checked) != len(rows) or {r["id"] for r in checked} != set(
            range(len(rows))
        ):
            raise ValueError("Semantic feature review IDs do not match predicates")
        accept = {r["id"]: r["accept"] for r in checked}
        return [r["features"] if accept[r["id"]] else None for r in generated]

    def __call__(self, text):
        """Evaluate the requested inputs."""
        if text not in self.cache:
            self.cache[text] = self.annotate([text])[0]
        return self.cache[text]
