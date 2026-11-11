"""Normalize equivalent predicate wording for filter prediction."""

import json
from src.prompts import FILTER_CANONICAL_GENERATION, FILTER_CANONICAL_REVIEW


def response_shape(name, field, kind):
    """Build the structured response schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field: {"type": kind}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


class CanonicalPredicate:
    def __init__(self, llm, role):
        """Initialize model state."""
        self.llm, self.role = llm, role
        self.cache = {}

    def __call__(self, text):
        """Evaluate the requested inputs."""
        if text not in self.cache:
            request = FILTER_CANONICAL_GENERATION + json.dumps(
                dict(record_type=self.role, condition=text)
            )
            answer = self.llm.complete(
                [request],
                256,
                response_format=response_shape("canonical", "predicate", "string"),
            )[0]
            proposed = json.loads(answer)["predicate"].strip()
            request = FILTER_CANONICAL_REVIEW + json.dumps(
                dict(record_type=self.role, original=text, proposed=proposed)
            )
            answer = self.llm.complete(
                [request],
                128,
                response_format=response_shape("equivalence", "accept", "boolean"),
            )[0]
            accepted = json.loads(answer)["accept"]
            self.cache[text] = proposed if accepted and proposed else text
        return self.cache[text]
