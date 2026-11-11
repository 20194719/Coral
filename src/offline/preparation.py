"""Build offline data and profiles."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
from src import prompts
from src.offline.candidates import build_candidate_pool
from src.offline.seed_sampling import stratified_seed_sample
from src.offline.relations import build_cards
from src.offline.relation_types import discover_types
from src.offline.profiles import parse_batch
from src.offline.schema import induce, validate_schema


def parameters(cfg, view, encoder, output):
    """Parameters."""
    return SimpleNamespace(
        **cfg,
        encoder=encoder,
        sbert_dim=cfg["embedding_dimension"],
        entity_R=view["entities"][0],
        entity_S=view["entities"][1],
        results_dir=Path(output),
    )


def entities(llm, stages, texts, entity, side):
    """Entities."""

    def build():
        """Build reusable method artifacts."""
        batches = [texts[i : i + 5] for i in range(0, len(texts), 5)]
        replies = llm.complete(
            [prompts.build_profile_prompt(list(b), entity.lower()) for b in batches],
            2000,
        )
        return [
            p
            for reply, b in zip(replies, batches)
            for p in prompts.parse_profile_batch(reply, len(b))
        ]

    raw = stages.build(f"profiles_{side}", build)
    return [prompts.profile_entities(p) for p in raw]


def candidate_pool(params, view, stages, tables, emb, llm):
    """Candidate pool."""

    def build():
        """Build reusable method artifacts."""
        if getattr(llm, "oracle", False):
            from src.offline.dictionary import scan

            ent = [
                scan(table.texts, other.titles)
                for table, other in zip(tables, reversed(tables))
            ]
            return build_candidate_pool(
                params,
                *emb,
                *(t.titles for t in tables),
                *(t.texts for t in tables),
                *ent,
            )
        ent = [
            entities(llm, stages, table.texts, entity, side)
            for table, entity, side in zip(tables, view["entities"], ("R", "S"))
        ]
        return build_candidate_pool(
            params, *emb, *(t.titles for t in tables), *(t.texts for t in tables), *ent
        )

    return stages.build("candidate_pool", build)


def seeds(params, stages, pool):
    """Seeds."""
    return stages.build(
        "seed_sample", lambda: stratified_seed_sample(params, pool, params.seed_global)
    )


def relation_types(params, view, stages, tables, sample, llm):
    """Relation types."""
    stage = "relation_types"

    def build():
        """Build reusable method artifacts."""

        def mine():
            """Mine semantic relation descriptions."""
            built = [
                prompts.build_relation_card_prompt(
                    tables[0].texts[int(r)],
                    tables[1].texts[int(s)],
                    max_rel=12,
                    ent_r=view["entities"][0],
                    ent_s=view["entities"][1],
                )
                for r, s in zip(sample.r, sample.s)
            ]
            return [
                prompts.parse_relation_card(reply, 12)
                for reply in llm.complete(built, 320)
            ]

        raw = stages.build("relation_cards", mine)
        return discover_types(params, build_cards(params, raw), llm)

    return stages.build(stage, build)


def profile_schema(params, view, tables, llm, output, resume=False):
    """Profile schema."""
    target = Path(output) / "profile_schema.json"
    if resume and target.exists():
        return validate_schema(json.loads(target.read_text()))
    schema = {}
    for side, table, role, examples in zip(
        ("R", "S"),
        tables,
        view.get("schema_roles", view["profile_roles"]),
        view["schema_examples"],
    ):
        fields, _ = induce(params, llm, table.texts, role, *examples, 24, 0)
        schema[side] = fields
    validate_schema(schema)
    target.write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    return schema


def profiles(stages, texts, fields, role, side, llm):
    """Profiles."""

    def build():
        """Build reusable method artifacts."""
        batches = [texts[i : i + 4] for i in range(0, len(texts), 4)]
        reps = llm.complete(
            [prompts.profile_prompt(fields, list(b), role) for b in batches], 2000
        )
        return [
            p for rep, b in zip(reps, batches) for p in parse_batch(rep, fields, len(b))
        ]

    sig = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:8]
    return stages.build(f"profiles_{side}_{sig}", build)
