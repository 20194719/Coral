"""LLM relation, filter, and joint pair judgments."""

import itertools
import json

from src import prompts
from src.inference.llm import LLMClient


class Judge(LLMClient):
    def judge_filter(self, text, docs, entity_type):
        """Judge filter."""
        return [
            prompts.parse_yes_no(x)
            for x in self.complete(
                [
                    prompts.build_filter_judge_prompt(text, doc, entity_type)
                    for doc in docs
                ],
                8,
            )
        ]

    def relation_spec(self, labels, pairs):
        """Relation spec."""
        built = [
            prompts.build_relation_spec_prompt(list(labels), r, s, *self.entities)
            for r, s in pairs
        ]
        tokens = 160 + 80 * len(labels)
        replies = self.complete(built, tokens)
        parsed, pending = [None] * len(built), list(range(len(built)))
        for attempt in range(3):
            failed = []
            for index, reply in zip(pending, replies):
                try:
                    parsed[index] = prompts.parse_relation_spec(reply, list(labels))
                except ValueError:
                    failed.append(index)
            if not failed:
                return parsed
            if attempt == 2:
                raise ValueError(
                    f"Invalid relation judgments after retries: {failed[:10]}"
                )
            pending = failed
            replies = self.complete(
                [built[i] for i in pending],
                tokens * 2 ** (attempt + 1),
                response_format=prompts.relation_response_format(labels),
            )


class LLMProbe:
    def __init__(self, llm, tables, entities, condition, filters):
        """Initialize model state."""
        self.llm, self.tables, self.entities = llm, tables, entities
        self.condition, self.filters = condition, filters

    def __call__(self, pairs, requested):
        """Judge pairs, retrying malformed replies once with a JSON string enum."""
        r, s = self.tables
        built = [
            prompts.pair_prompt(
                self.condition if need[0] else None,
                self.filters.get(r.name) if need[1] else None,
                self.filters.get(s.name) if need[2] else None,
                r.texts[i],
                s.texts[j],
                *self.entities,
                strict=True,
            )
            for (i, j), need in zip(pairs, requested)
        ]
        replies = self.llm.complete([text for text, _ in built], 12)
        result = []
        for (text, tags), reply in zip(built, replies):
            words = reply.strip().lower().split()
            if len(words) != len(tags) or any(
                word not in ("yes", "no") for word in words
            ):
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "joint_yes_no",
                        "strict": True,
                        "schema": {
                            "type": "string",
                            "enum": [
                                " ".join(values)
                                for values in itertools.product(
                                    ["yes", "no"], repeat=len(tags)
                                )
                            ],
                        },
                    },
                }
                repaired = self.llm.complete(
                    [text], 12, response_format=response_format
                )[0]
                try:
                    answer = json.loads(repaired)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid schema joint judgment: {repaired!r}"
                    ) from exc
                if not isinstance(answer, str):
                    raise ValueError(f"Invalid schema joint judgment: {repaired!r}")
                words = answer.strip().lower().split()
                if len(words) != len(tags) or any(
                    word not in ("yes", "no") for word in words
                ):
                    raise ValueError(f"Invalid schema joint judgment: {repaired!r}")
            labels = dict(zip(tags, [float(word == "yes") for word in words]))
            result.append(tuple(labels.get(tag) for tag in ("theta", "R", "S")))
        return result
