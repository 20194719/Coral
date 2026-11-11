"""Discover predicates from corpus profiles and document excerpts."""

from collections import Counter
import hashlib
import json
import numpy as np
from src.prompts import (
    FILTER_PREDICATES,
    FILTER_PREDICATE_REVIEW,
    FILTER_PREDICATE_FEEDBACK,
    FILTER_PREDICATE_GROUP,
    FILTER_PREDICATE_DEDUP,
    FILTER_PREDICATE_SELECTION,
    FILTER_PREDICATE_DEDUP_FEEDBACK,
    FILTER_PREDICATE_DIMENSIONS,
    FILTER_PREDICATE_COVERAGE_REQUIREMENTS,
    FILTER_CORPUS_DIMENSIONS,
    FILTER_CORPUS_DIMENSION_GUIDANCE,
    FILTER_CORPUS_DIMENSION_REVIEW,
)
from src.data.storage import atomic_json
from src.prompts import (
    FILTER_DIMENSION_RETRY_PROMPT,
    FILTER_PARTITION_RETRY_PROMPT,
    FILTER_SELECTION_RETRY_PROMPT,
    FILTER_DIMENSION_GUIDANCE_PROMPT,
    FILTER_COVERAGE_RETRY_PROMPT,
)


def profile_statistics(profiles, fields):
    """Profile statistics."""
    summaries = []
    for field in fields:
        values = [p.get(field["name"]) if p else None for p in profiles]
        summary = dict(name=field["name"], type=field["type"])
        if field.get("description"):
            summary["description"] = field["description"]
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
                        ("min", "p10", "p25", "median", "p75", "p90", "max"),
                        np.quantile(
                            numbers, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                        ).tolist(),
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
                    :64
                ]
            ]
            summary["singleton_values"] = sum(n == 1 for n in counts.values())
        summary.update(
            present_documents=present,
            missing_documents=len(profiles) - present,
        )
        summaries.append(summary)
    return dict(documents=len(profiles), fields=summaries)


def _terms(profile, field):
    """Extract normalized profile terms."""
    value = (profile or {}).get(field)
    items = value if isinstance(value, list) else [value]
    return {
        " ".join(v.split()).casefold()
        for v in items
        if isinstance(v, str) and v.strip()
    }


def _documents(table, profiles, fields, field, count, seed):
    """Sample documents for predicate mining."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table)).tolist()
    count = min(count, len(table))
    ids = order[: (count + 1) // 2]
    names = [
        f["name"]
        for f in fields
        if f["type"] != "number" and (not field or f["name"] == field)
    ]
    tokens = [set() for _ in profiles]
    for name in names:
        values = [_terms(p, name) for p in profiles]
        frequencies = Counter(v for terms in values for v in terms)
        allowed = {v for v, n in frequencies.items() if 2 <= n <= 0.8 * len(profiles)}
        for i, terms in enumerate(values):
            tokens[i].update((name, v) for v in terms & allowed)
    covered = set().union(*(tokens[i] for i in ids)) if ids else set()
    remaining = [i for i in order if i not in ids]
    while len(ids) < count:

        def score(i):
            """Score document coverage gains."""
            counts = Counter(name for name, value in tokens[i] - covered)
            return sum(min(n, 2) for n in counts.values())

        selected = max(remaining, key=score)
        ids.append(selected)
        covered.update(tokens[selected])
        remaining.remove(selected)
    docs = {}
    for i in ids:
        text = table.texts[i]
        if len(text) > 1200:
            middle = len(text) // 2
            text = (
                text[:350]
                + "\n[... omitted ...]\n"
                + text[middle - 225 : middle + 225]
                + "\n[... omitted ...]\n"
                + text[-400:]
            )
        docs[i] = text
    return docs


def _format_docs(documents):
    """Format sampled document evidence."""
    return "\n\n".join(f"[DOCUMENT {i}]\n{text}" for i, text in documents.items())


def _statistics_view(statistics, field=None):
    """Prepare field-level corpus statistics."""
    fields = []
    for item in statistics["fields"]:
        current = dict(item)
        if "top_values" in current:
            current["top_values"] = current["top_values"][
                : 64 if item["name"] == field else 8
            ]
        fields.append(current)
    return dict(documents=statistics["documents"], fields=fields)


def _duplicate_closure(reply, expected_ids):
    """Recover only complete dedup records with an unfinished outer wrapper."""
    def unique_object(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Repeated JSON object key")
        return dict(pairs)

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    position = 0

    def take(token):
        nonlocal position
        while position < len(reply) and reply[position] in " \t\r\n":
            position += 1
        if not reply.startswith(token, position):
            raise ValueError("Unexpected JSON wrapper or separator")
        position += len(token)

    try:
        take("{")
        take('"duplicates"')
        take(":")
        take("[")
        rows = []
        for index in range(len(expected_ids)):
            if index:
                take(",")
            take("")
            row, position = decoder.raw_decode(reply, position)
            if (
                not isinstance(row, dict)
                or set(row) != {"id", "duplicate_of"}
                or type(row["id"]) is not int
            ):
                return None
            duplicate = row["duplicate_of"]
            if duplicate is not None and (
                type(duplicate) is not int
                or duplicate not in expected_ids
                or duplicate >= row["id"]
            ):
                return None
            rows.append(row)
        if {row["id"] for row in rows} != expected_ids:
            return None
        suffix = reply[position:]
        for whitespace in " \t\r\n":
            suffix = suffix.replace(whitespace, "")
        if suffix not in ("", ",", "]", ",]", ",]}"):
            return None
        return rows
    except (ValueError, TypeError):
        return None


def _request(llm, prompt, key, item, count, tokens, *, expected_ids=None):
    """Request a structured model response."""
    schema = {
        "type": "object",
        "properties": {
            key: {
                "type": "array",
                "items": item,
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": [key],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": key, "strict": True, "schema": schema},
    }
    for _ in range(3):
        reply = llm.complete(
            [prompt], max_tokens=tokens, response_format=response_format
        )[0]
        try:
            values = json.loads(reply)[key]
            if isinstance(values, list) and len(values) == count:
                return values
        except json.JSONDecodeError:
            if key == "duplicates":
                if expected_ids is not None and len(expected_ids) == count:
                    values = _duplicate_closure(reply, expected_ids)
                    if values is not None:
                        print(
                            f"[offline] Recovered duplicates JSON wrapper: {count} "
                            "complete records; decisions unchanged; reply_sha256="
                            + hashlib.sha256(reply.encode()).hexdigest(),
                            flush=True,
                        )
                        return values
                tokens *= 2
        except (ValueError, KeyError, TypeError):
            pass
    raise ValueError(f"Invalid {key} after retries")


def _propose(llm, prompts, item, fields, count):
    """Propose corpus filter predicates."""
    properties = {k: v for k, v in item["properties"].items() if k != "field"}
    entry = dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )
    groups = [(f["name"], max(1, count // 4)) for f in fields if f["type"] != "number"]
    groups.append(("", max(1, count // 4)))
    results = {}
    for limit in sorted({limit for _, limit in groups}):
        pending = [(field, "") for field, size in groups if size == limit]
        schema = dict(
            type="object",
            properties={
                "candidates": dict(
                    type="array",
                    items=entry,
                    minItems=0,
                    maxItems=limit,
                )
            },
            required=["candidates"],
            additionalProperties=False,
        )
        response_format = dict(
            type="json_schema",
            json_schema=dict(name="candidate_group", strict=True, schema=schema),
        )
        for _ in range(3):
            replies = llm.complete(
                [
                    prompts[field]
                    + FILTER_PREDICATE_GROUP.format(
                        field=json.dumps(field), limit=limit
                    )
                    + feedback
                    for field, feedback in pending
                ],
                max_tokens=3500,
                response_format=response_format,
            )
            failed = []
            for (field, _), reply in zip(pending, replies):
                try:
                    values = json.loads(reply)["candidates"]
                    if (
                        not isinstance(values, list)
                        or len(values) > limit
                        or not all(isinstance(v, dict) for v in values)
                    ):
                        raise ValueError("Invalid candidate list")
                    results[field] = [dict(v, field=field) for v in values]
                except (ValueError, KeyError, TypeError):
                    failed.append(
                        (
                            field,
                            "\nPrevious output failed validation. Return a shorter, complete JSON candidate list.",
                        )
                    )
            pending = failed
            if not pending:
                break
        if pending:
            raise ValueError(
                f"Invalid candidate groups after retries: {[f for f, _ in pending]}"
            )
    return [c for field, _ in groups for c in results[field]]


def _support(candidate, profiles, fields, sampled_docs):
    """Check candidate predicate support."""
    names = {f["name"]: f for f in fields}
    text, field, kind = (candidate.get(k) for k in ("predicate", "field", "kind"))
    values, documents = candidate.get("values"), candidate.get("sample_documents")
    if (
        not isinstance(text, str)
        or not text.strip()
        or kind not in ("semantic", "existence")
    ):
        return None
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        return None
    if not isinstance(documents, list) or not all(
        type(i) is int and i in sampled_docs for i in documents
    ):
        return None
    quotes = candidate.get("evidence_quotes", [])
    if not isinstance(quotes, list):
        return None
    if field == "":
        if kind != "semantic" or len(set(documents)) < 2 or values:
            return None
        if len(quotes) != len(set(documents)):
            return None
        cited = set()
        for evidence in quotes:
            if not isinstance(evidence, dict):
                return None
            doc, quote = evidence.get("document"), evidence.get("quote")
            if (
                type(doc) is not int
                or doc not in documents
                or not isinstance(quote, str)
            ):
                return None
            normalized = " ".join(quote.split())
            if len(normalized) < 12 or normalized not in " ".join(
                sampled_docs[doc].split()
            ):
                return None
            cited.add(doc)
        if cited != set(documents):
            return None
        return dict(
            source="sample", documents=len(set(documents)), total=len(sampled_docs)
        )
    if quotes or documents:
        return None
    if field not in names:
        return None
    spec = names[field]
    matches, found = set(), set()
    if spec["type"] == "number":
        return None
    if kind == "semantic" and (not values or spec["type"] == "number"):
        return None
    if kind != "semantic" and values:
        return None
    wanted = {" ".join(v.split()).casefold() for v in values}
    for i, profile in enumerate(profiles):
        value = profile.get(field) if profile else None
        items = value if isinstance(value, list) else [value]
        terms = {
            " ".join(v.split()).casefold()
            for v in items
            if isinstance(v, str) and v.strip()
        }
        present = bool(terms)
        if kind == "semantic":
            matched = {
                v
                for v in wanted
                if any(t == v or len(v) == 160 and t.startswith(v) for t in terms)
            }
            found.update(matched)
            present = bool(matched)
        if present:
            matches.add(i)
    if len(matches) < 2 or kind == "semantic" and found != wanted:
        return None
    return dict(
        source="profile_evidence",
        documents=len(matches),
        total=len(profiles),
    )


def _eligible(candidates, reviews):
    """Select supported predicate candidates."""
    by_id = {c["id"]: c for c in candidates}
    if len(reviews) != len(candidates) or {r.get("id") for r in reviews} != set(by_id):
        raise ValueError("Predicate review IDs do not match candidates")
    eligible = []
    for review in reviews:
        duplicate = review.get("duplicate_of")
        if (
            type(review.get("eligible")) is not bool
            or duplicate is not None
            and (
                type(duplicate) is not int
                or duplicate not in by_id
                or duplicate >= review["id"]
            )
        ):
            raise ValueError("Invalid predicate review")
        if review["eligible"] and duplicate is None:
            eligible.append(dict(by_id[review["id"]], family=review["family"]))
    return eligible


def _requirements(dimensions, candidates, count, *, allow_empty=False):
    """Validate overlapping dimensions and allocate coverage minima."""
    if allow_empty and dimensions == []:
        return []
    if not isinstance(dimensions, list) or not dimensions or len(dimensions) > count:
        raise ValueError("Dimensions must be a nonempty list fitting the probe budget")
    expected = {c["id"] for c in candidates}
    assigned, names = [], set()
    for dimension in dimensions:
        for key in ("dimension", "definition"):
            if not isinstance(dimension.get(key), str) or not dimension[key].strip():
                raise ValueError("Each dimension needs a name and definition")
        name = " ".join(dimension["dimension"].split()).casefold()
        if name in names:
            raise ValueError("Dimension names must be distinct")
        names.add(name)
        ids = dimension.get("candidate_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError("Each dimension needs integer candidate IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("A dimension must not repeat a candidate ID")
        assigned.extend(ids)
    if set(assigned) != expected:
        raise ValueError("Assign every eligible ID at least once, without invented IDs")
    preserve_contrasts = (
        sum(min(2, len(d["candidate_ids"])) for d in dimensions) <= count
    )
    return [
        dict(
            d,
            minimum_selected=min(2, len(d["candidate_ids"]))
            if preserve_contrasts
            else 1,
        )
        for d in dimensions
    ]


def _corpus_dimensions(llm, prompt, count, statistics, docs):
    """Corpus dimensions."""
    item = dict(
        type="object",
        properties={
            k: {"type": "string"} for k in ("dimension", "definition", "evidence")
        },
        required=["dimension", "definition", "evidence"],
        additionalProperties=False,
    )
    schema = dict(
        type="object",
        properties={
            "dimensions": dict(
                type="array",
                items=item,
                minItems=1,
                maxItems=count,
            )
        },
        required=["dimensions"],
        additionalProperties=False,
    )
    response_format = dict(
        type="json_schema",
        json_schema=dict(
            name="corpus_dimensions",
            strict=True,
            schema=schema,
        ),
    )
    feedback = ""
    for _ in range(3):
        current = prompt + feedback
        reply = llm.complete(
            [current], max_tokens=3000, response_format=response_format
        )[0]
        try:
            dimensions = json.loads(reply)["dimensions"]
            if not isinstance(dimensions, list) or not 1 <= len(dimensions) <= count:
                raise ValueError("Invalid dimension count")
            if any(
                not isinstance(d[k], str) or not d[k].strip()
                for d in dimensions
                for k in ("dimension", "definition", "evidence")
            ):
                raise ValueError("Each dimension needs a name, definition and evidence")
            if len({d["dimension"].strip().casefold() for d in dimensions}) != len(
                dimensions
            ):
                raise ValueError("Dimension names must be distinct")
            review_prompt = FILTER_CORPUS_DIMENSION_REVIEW.format(
                statistics=statistics,
                docs=docs,
                dimensions=json.dumps(
                    [dict(d, id=i) for i, d in enumerate(dimensions)],
                    ensure_ascii=False,
                ),
            )
            reviews = _request(
                llm,
                review_prompt,
                "reviews",
                dict(
                    type="object",
                    properties={
                        "id": {"type": "integer"},
                        "eligible": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    required=["id", "eligible", "reason"],
                    additionalProperties=False,
                ),
                len(dimensions),
                1800,
            )
            if {r["id"] for r in reviews} != set(range(len(dimensions))) or any(
                type(r["id"]) is not int or type(r["eligible"]) is not bool
                for r in reviews
            ):
                raise ValueError("Dimension review IDs or decisions are invalid")
            accepted = {r["id"] for r in reviews if r["eligible"]}
            if not accepted:
                raise ValueError("No dimensions passed evidence review")
            return dict(
                dimensions=[d for i, d in enumerate(dimensions) if i in accepted],
                proposed_dimensions=dimensions,
                reviews=reviews,
                review_prompt_sha256=hashlib.sha256(review_prompt.encode()).hexdigest(),
                prompt_sha256=hashlib.sha256(current.encode()).hexdigest(),
            )
        except (ValueError, KeyError, TypeError) as exc:
            feedback = FILTER_DIMENSION_RETRY_PROMPT.format(error=str(exc))
    raise ValueError("Invalid corpus dimension description after retries")


def _plan_dimensions(llm, candidates, count, corpus_plan=None, *, allow_uncovered=False):
    """Plan dimensions."""
    prompt = FILTER_PREDICATE_DIMENSIONS.format(
        count=count,
        candidates=json.dumps(
            [{k: c[k] for k in ("id", "predicate", "family")} for c in candidates],
            ensure_ascii=False,
        ),
    )
    if corpus_plan is not None:
        prompt += FILTER_DIMENSION_GUIDANCE_PROMPT.format(
            dimensions=json.dumps(
                [
                    {k: d[k] for k in ("dimension", "definition")}
                    for d in corpus_plan["dimensions"]
                ],
                ensure_ascii=False,
            )
        )
    item = dict(
        type="object",
        properties={
            "dimension": {"type": "string"},
            "definition": {"type": "string"},
            "candidate_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 0,
            },
        },
        required=["dimension", "definition", "candidate_ids"],
        additionalProperties=False,
    )
    schema = dict(
        type="object",
        properties={
            "dimensions": {
                "type": "array",
                "items": item,
                "minItems": 1,
                "maxItems": count,
            },
        },
        required=["dimensions"],
        additionalProperties=False,
    )
    response_format = dict(
        type="json_schema",
        json_schema=dict(
            name="dimensions",
            strict=True,
            schema=schema,
        ),
    )
    feedback = ""
    for _ in range(3):
        current = prompt + feedback
        reply = llm.complete(
            [current], max_tokens=3500, response_format=response_format
        )[0]
        try:
            dimensions = json.loads(reply)["dimensions"]
            if not isinstance(dimensions, list) or not dimensions or len(dimensions) > count:
                raise ValueError("Invalid dimension count")
            names = set()
            for dimension in dimensions:
                if not isinstance(dimension, dict) or any(
                    not isinstance(dimension.get(key), str) or not dimension[key].strip()
                    for key in ("dimension", "definition")
                ):
                    raise ValueError("Each dimension needs a name and definition")
                name = " ".join(dimension["dimension"].split()).casefold()
                if name in names:
                    raise ValueError("Dimension names must be distinct")
                names.add(name)
                if not isinstance(dimension.get("candidate_ids"), list):
                    raise ValueError("Each dimension needs a candidate ID list")
            dropped = []
            if corpus_plan is not None:
                expected = {d["dimension"] for d in corpus_plan["dimensions"]}
                missing = expected - {d["dimension"] for d in dimensions}
                missing.update(
                    d["dimension"] for d in dimensions if not d.get("candidate_ids")
                )
                if missing and not allow_uncovered:
                    raise ValueError(
                        "Missing candidate coverage for inferred dimensions: "
                        + ", ".join(sorted(missing))
                    )
                if missing:
                    definitions = {
                        d["dimension"]: d["definition"]
                        for d in [*dimensions, *corpus_plan["dimensions"]]
                    }
                    dropped = [
                        dict(
                            dimension=name,
                            definition=definitions[name],
                            reason="No candidate coverage after three candidate rounds",
                        )
                        for name in sorted(missing)
                    ]
                    dimensions = [d for d in dimensions if d["dimension"] not in missing]
            requirements = _requirements(
                dimensions, candidates, count, allow_empty=bool(dropped)
            )
            plan = dict(
                dimensions=dimensions,
                requirements=requirements,
                prompt_sha256=hashlib.sha256(current.encode()).hexdigest(),
            )
            if dropped:
                plan["dropped_dimensions"] = dropped
                print(
                    "[offline] Skipped uncovered dimensions after three candidate rounds: "
                    + ", ".join(d["dimension"] for d in dropped),
                    flush=True,
                )
            return plan
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            feedback = FILTER_PARTITION_RETRY_PROMPT.format(error=str(exc))
    raise ValueError("Invalid LLM dimension plan after retries: " + feedback)


def _coverage_counts(requirements, chosen):
    """Coverage counts."""
    selected = {c["id"] for c in chosen}
    return [
        dict(
            d,
            selected_ids=sorted(selected.intersection(d["candidate_ids"])),
            selected_count=len(selected.intersection(d["candidate_ids"])),
        )
        for d in requirements
    ]


def _choose(llm, prompt, candidates, count, plan, audit=None):
    """Select reviewed filter predicates."""
    available = sum(c.get("kind") != "numeric" for c in candidates)
    if available < count:
        raise ValueError(
            f"Only {available} candidates fit the nonnumeric requirement; need {count}. Generate more distinct semantic conditions"
        )
    requirements = _requirements(
        plan["dimensions"], candidates, count,
        allow_empty=bool(plan.get("dropped_dimensions")),
    )
    prompt += FILTER_PREDICATE_COVERAGE_REQUIREMENTS.format(
        requirements=json.dumps(requirements, ensure_ascii=False),
    )
    item = dict(
        type="object",
        properties={
            "id": {"type": "integer"},
            "predicate": {"type": "string"},
        },
        required=["id", "predicate"],
        additionalProperties=False,
    )
    by_id = {c["id"]: c for c in candidates}
    feedback = ""
    for _ in range(3):
        current_prompt = prompt + feedback
        reply = _request(llm, current_prompt, "selected", item, count, 3000)
        try:
            ids = [r["id"] for r in reply]
            texts = [r["predicate"] for r in reply]
            if (
                len(set(ids)) != count
                or len({" ".join(t.split()).casefold() for t in texts}) != count
            ):
                raise ValueError("Selection repeats an ID or predicate")
            if any(
                type(r["id"]) is not int
                or r["id"] not in by_id
                or r["predicate"] != by_id[r["id"]]["predicate"]
                for r in reply
            ):
                raise ValueError(
                    "Selection must copy eligible IDs and predicate text verbatim"
                )
            chosen = [by_id[i] for i in ids]
            if any(c.get("kind") == "numeric" for c in chosen):
                raise ValueError("Select only nonnumeric predicates")
            counts = _coverage_counts(requirements, chosen)
            gaps = [d for d in counts if d["selected_count"] < d["minimum_selected"]]
            if audit is not None:
                audit.append(
                    dict(
                        selection_prompt_sha256=hashlib.sha256(
                            current_prompt.encode()
                        ).hexdigest(),
                        selected_ids=ids,
                        coverage_counts=counts,
                        coverage_gaps=gaps,
                    )
                )
            if gaps:
                feedback = FILTER_COVERAGE_RETRY_PROMPT.format(
                    gaps=json.dumps(gaps, ensure_ascii=False)
                )
                continue
            return chosen
        except (ValueError, KeyError, TypeError) as exc:
            feedback = FILTER_SELECTION_RETRY_PROMPT.format(error=str(exc))
    raise ValueError(
        "LLM final selection failed structural or coverage checks after retries"
    )


def _deduplicate(llm, stage, side, prompt, candidates):
    """Deduplicate."""
    ids = {c["id"] for c in candidates}
    feedback = ""
    for attempt in range(3):
        current = prompt + feedback
        signature = hashlib.sha256(current.encode()).hexdigest()
        rows = stage.build(
            f"filter_duplicates_{side}_{signature[:12]}",
            lambda: _request(
                llm,
                current,
                "duplicates",
                dict(
                    type="object",
                    properties={
                        "id": {"type": "integer"},
                        "duplicate_of": {"type": ["integer", "null"]},
                    },
                    required=["id", "duplicate_of"],
                    additionalProperties=False,
                ),
                len(candidates),
                2600,
                expected_ids=ids,
            ),
        )
        try:
            if len(rows) != len(ids) or {r["id"] for r in rows} != ids:
                raise ValueError(
                    "Duplicate review IDs do not match eligible candidates"
                )
            for row in rows:
                duplicate = row["duplicate_of"]
                if duplicate is not None and (
                    type(duplicate) is not int
                    or duplicate not in ids
                    or duplicate >= row["id"]
                ):
                    raise ValueError(
                        f"Invalid duplicate reference for ID {row['id']}: {duplicate}"
                    )
            return rows, signature
        except ValueError as exc:
            feedback = FILTER_PREDICATE_DEDUP_FEEDBACK.format(
                attempt=attempt + 1, error=str(exc)
            )
    raise ValueError("Invalid duplicate review after retries")


def generate(cfg, stage, table, role, side, llm, output, profiles, fields):
    """Generate training filter predicates."""
    if len(profiles) != len(table):
        raise ValueError("Profile count differs from table size")
    count = cfg["filter_predicate_count"]
    if isinstance(count, dict):
        count = count[table.name]
    statistics = profile_statistics(profiles, fields)
    groups = [f["name"] for f in fields if f["type"] != "number"] + [""]
    source_docs, context_history = {}, []
    dimension_docs = _documents(
        table,
        profiles,
        fields,
        None,
        cfg["filter_predicate_sample"],
        cfg["seed_global"],
    )
    dimension_limit = max(1, count // 4)
    dimension_statistics = json.dumps(_statistics_view(statistics), ensure_ascii=False)
    dimension_excerpts = _format_docs(dimension_docs)
    dimension_prompt = FILTER_CORPUS_DIMENSIONS.format(
        count=dimension_limit,
        statistics=dimension_statistics,
        docs=dimension_excerpts,
    )
    dimension_key = hashlib.sha256(
        (dimension_prompt + FILTER_CORPUS_DIMENSION_REVIEW).encode()
    ).hexdigest()
    corpus_plan = stage.build(
        f"filter_corpus_dimensions_{side}_{dimension_key[:12]}",
        lambda: _corpus_dimensions(
            llm,
            dimension_prompt,
            dimension_limit,
            dimension_statistics,
            dimension_excerpts,
        ),
    )
    atomic_json(
        output / f"filter-corpus-dimensions-{table.name}.json",
        dict(
            corpus_plan,
            document_ids=list(dimension_docs),
            statistics=statistics,
        ),
    )
    candidate_schema = {
        "type": "object",
        "properties": {
            "predicate": {"type": "string"},
            "field": {"type": "string", "enum": groups},
            "kind": {"type": "string", "enum": ["semantic", "existence"]},
            "values": {"type": "array", "items": {"type": "string"}},
            "sample_documents": {"type": "array", "items": {"type": "integer"}},
            "evidence_quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "integer"},
                        "quote": {"type": "string"},
                    },
                    "required": ["document", "quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "predicate",
            "field",
            "kind",
            "values",
            "sample_documents",
            "evidence_quotes",
        ],
        "additionalProperties": False,
    }
    review_schema = dict(
        type="object",
        properties={
            "id": {"type": "integer"},
            "eligible": {"type": "boolean"},
            "duplicate_of": {"type": ["integer", "null"]},
            "family": {"type": "string"},
            "reason": {"type": "string"},
        },
        required=["id", "eligible", "duplicate_of", "family", "reason"],
        additionalProperties=False,
    )
    attempts, feedback, carried, next_id = [], "", [], 0
    for round_index in range(3):
        contexts = {
            field: _documents(
                table,
                profiles,
                fields,
                field,
                cfg["filter_predicate_sample"],
                cfg["seed_global"] + round_index,
            )
            for field in groups
        }
        source_docs.update(
            {i: text for docs in contexts.values() for i, text in docs.items()}
        )
        context_history.append(
            dict(
                round=round_index,
                groups={field: list(docs) for field, docs in contexts.items()},
            )
        )
        prompts = {
            field: FILTER_PREDICATES.format(
                what=role,
                count=count,
                docs=_format_docs(contexts[field]),
                statistics=json.dumps(
                    _statistics_view(statistics, field), ensure_ascii=False
                ),
            )
            + FILTER_CORPUS_DIMENSION_GUIDANCE.format(
                dimensions=json.dumps(
                    [
                        {k: d[k] for k in ("dimension", "definition")}
                        for d in corpus_plan["dimensions"]
                    ],
                    ensure_ascii=False,
                ),
            )
            for field in groups
        }
        atomic_json(
            output / f"filter-source-context-{table.name}.json",
            dict(
                profile_statistics=statistics,
                documents=source_docs,
                groups=context_history[-1]["groups"],
                rounds=context_history,
            ),
        )
        current_prompts = {k: v + feedback for k, v in prompts.items()}
        signature = hashlib.sha256(
            json.dumps(
                dict(
                    prompts=current_prompts,
                    group_prompt=FILTER_PREDICATE_GROUP,
                    count=count,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        raw = stage.build(
            f"filter_candidates_{side}_{signature[:12]}",
            lambda: _propose(llm, current_prompts, candidate_schema, fields, 2 * count),
        )
        candidates = list(carried)
        reviews = [
            dict(
                id=c["id"],
                eligible=True,
                duplicate_of=None,
                family=c["family"],
                reason="",
            )
            for c in carried
        ]
        existing = {" ".join(c["predicate"].split()).casefold() for c in carried}
        new_candidates, unsupported = [], []
        for i, candidate in enumerate(raw):
            support = (
                _support(
                    candidate,
                    profiles,
                    fields,
                    contexts.get(candidate.get("field"), {}),
                )
                if isinstance(candidate, dict)
                else None
            )
            if support:
                text = " ".join(candidate["predicate"].split()).casefold()
                if text not in existing:
                    new_candidates.append(
                        dict(candidate, id=next_id + i, support=support)
                    )
                    existing.add(text)
            else:
                unsupported.append(
                    dict(
                        id=next_id + i,
                        predicate=candidate.get("predicate", "")
                        if isinstance(candidate, dict)
                        else "",
                        reason="Invalid evidence or fewer than two supporting documents",
                    )
                )
        next_id += len(raw)
        candidates.extend(new_candidates)
        audit = dict(
            candidates=raw,
            carried_ids=[c["id"] for c in carried],
            supported_candidates=candidates,
            unsupported=unsupported,
            reviews=[],
        )
        attempts.append(audit)
        try:
            if not candidates:
                raise ValueError("No candidates have recurring evidence")
            review_signatures = []
            for start in range(0, len(new_candidates), 12):
                batch = new_candidates[start : start + 12]
                cited = {
                    i: source_docs[i] for c in batch for i in c["sample_documents"]
                }
                review_prompt = FILTER_PREDICATE_REVIEW.format(
                    what=role,
                    statistics=json.dumps(
                        _statistics_view(statistics), ensure_ascii=False
                    ),
                    candidates=json.dumps(batch, ensure_ascii=False),
                    docs=_format_docs(cited),
                )
                review_signature = hashlib.sha256(review_prompt.encode()).hexdigest()
                review_signatures.append(review_signature)
                reviews.extend(
                    stage.build(
                        f"filter_review_{side}_{review_signature[:12]}",
                        lambda: _request(
                            llm,
                            review_prompt,
                            "reviews",
                            review_schema,
                            len(batch),
                            2400,
                        ),
                    )
                )
            audit["review_prompt_sha256"] = review_signatures
            audit["reviews"] = reviews
            eligible = _eligible(candidates, reviews)
            if eligible:
                meanings = [
                    dict(id=c["id"], predicate=c["predicate"]) for c in eligible
                ]
                duplicate_prompt = FILTER_PREDICATE_DEDUP.format(
                    candidates=json.dumps(meanings, ensure_ascii=False)
                )
                duplicates, duplicate_signature = _deduplicate(
                    llm, stage, side, duplicate_prompt, meanings
                )
                duplicate_map = {r["id"]: r["duplicate_of"] for r in duplicates}
                for review in reviews:
                    if review["id"] in duplicate_map:
                        review["duplicate_of"] = duplicate_map[review["id"]]
                audit["duplicate_prompt_sha256"] = duplicate_signature
                eligible = _eligible(candidates, reviews)
            carried = eligible
            if len(eligible) < count:
                raise ValueError(
                    f"Only {len(eligible)} eligible distinct candidates; need {count}"
                )
            selection_prompt = FILTER_PREDICATE_SELECTION.format(
                what=role,
                count=count,
                statistics=json.dumps(_statistics_view(statistics), ensure_ascii=False),
                candidates=json.dumps(
                    [
                        {
                            k: c[k]
                            for k in (
                                "id",
                                "predicate",
                                "field",
                                "kind",
                                "family",
                                "values",
                                "support",
                            )
                        }
                        for c in eligible
                    ],
                    ensure_ascii=False,
                ),
            )
            selection_signature = hashlib.sha256(selection_prompt.encode()).hexdigest()
            plan_inputs = [
                eligible, count, corpus_plan, FILTER_PREDICATE_DIMENSIONS,
                "coverage-min-two",
            ]
            allow_uncovered = round_index == 2
            if allow_uncovered:
                plan_inputs.append(dict(allow_uncovered=True))
            plan_key = hashlib.sha256(
                json.dumps(plan_inputs, sort_keys=True).encode()
            ).hexdigest()
            plan = stage.build(
                f"filter_dimensions_{side}_{plan_key[:12]}",
                lambda: _plan_dimensions(
                    llm, eligible, count, corpus_plan,
                    allow_uncovered=allow_uncovered,
                ),
            )
            plan["requirements"] = _requirements(
                plan["dimensions"], eligible, count,
                allow_empty=bool(plan.get("dropped_dimensions")),
            )
            selection_key = hashlib.sha256(
                json.dumps(
                    [
                        selection_prompt,
                        plan,
                        FILTER_PREDICATE_COVERAGE_REQUIREMENTS,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            def select_with_audit():
                """Select with audit."""
                records = []
                chosen = _choose(
                    llm, selection_prompt, eligible, count, plan, audit=records
                )
                return dict(selected=chosen, coverage_checks=records)

            selection = stage.build(
                f"filter_selection_{side}_{selection_key[:12]}",
                select_with_audit,
            )
            selected = selection["selected"]
            counts = _coverage_counts(plan["requirements"], selected)
            if any(d["selected_count"] < d["minimum_selected"] for d in counts):
                raise ValueError("Saved selection does not meet coverage requirements")
            audit.update(
                corpus_dimension_plan=corpus_plan,
                dimension_plan=plan,
                selected_ids=[c["id"] for c in selected],
                selection_initial_prompt_sha256=selection_signature,
                selection_cache_sha256=selection_key,
                selection_coverage_checks=selection["coverage_checks"],
                selection_prompt_sha256=selection["coverage_checks"][-1][
                    "selection_prompt_sha256"
                ],
            )
            selection_signature = audit["selection_prompt_sha256"]
            atomic_json(
                output / f"filter-candidates-{table.name}.json",
                dict(audit, attempts=attempts),
            )
            break
        except ValueError as exc:
            audit["error"] = str(exc)
            atomic_json(
                output / f"filter-candidates-{table.name}.json",
                dict(audit, attempts=attempts),
            )
            feedback = FILTER_PREDICATE_FEEDBACK.format(
                count=count,
                error=str(exc),
                retained=json.dumps(
                    [c["predicate"] for c in carried], ensure_ascii=False
                ),
                supported=json.dumps(dict(Counter(c["field"] for c in candidates))),
                rejected=json.dumps(
                    (
                        unsupported
                        + [
                            dict(
                                predicate=c["predicate"],
                                reason=r.get("reason") or "Semantic duplicate",
                            )
                            for c in candidates
                            for r in audit["reviews"]
                            if r.get("id") == c["id"]
                            and (
                                not r.get("eligible")
                                or r.get("duplicate_of") is not None
                            )
                        ]
                    )[:16],
                    ensure_ascii=False,
                ),
            )
    else:
        raise ValueError(
            f"Predicate screening failed after {len(attempts)} attempts: {audit['error']}"
        )
    predicates = [c["predicate"] for c in selected]
    atomic_json(
        output / f"filter-predicates-{table.name}.json",
        dict(
            table=table.name,
            document_ids=list(source_docs),
            document_excerpt_chars=1200,
            source="corpus-llm",
            selection_source="llm-verbatim",
            profile_statistics=statistics,
            prompt_sha256=signature,
            review_prompt_sha256=audit["review_prompt_sha256"],
            selection_prompt_sha256=selection_signature,
            selection_cache_sha256=audit["selection_cache_sha256"],
            corpus_dimension_plan=audit["corpus_dimension_plan"],
            dimension_plan=audit["dimension_plan"],
            selection_coverage_checks=audit["selection_coverage_checks"],
            selected_candidates=selected,
            predicates=predicates,
        ),
    )
    return predicates
