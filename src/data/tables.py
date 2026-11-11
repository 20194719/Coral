"""Load named CSV tables and keep evaluation labels outside estimator queries."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


from src.config import presets

TABLES = {name: tuple(cfg["tables"]) for name, cfg in presets().items()}


@dataclass(frozen=True)
class Table:
    name: str
    ids: tuple[str, ...]
    titles: tuple[str, ...]
    texts: tuple[str, ...]
    sha256: str

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.ids)


@dataclass(frozen=True)
class Query:
    id: int
    relation: str
    condition: str
    tables: tuple[str, ...]
    filters: dict[str, str]


@dataclass(frozen=True)
class Dataset:
    name: str
    tables: dict[str, Table]
    queries: tuple[Query, ...]
    ground_truth: dict[int, int]


def _table(name: str, path: Path) -> Table:
    """Table."""
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "wiki_text"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: required columns are id and wiki_text")
        rows = list(reader)
    rows.sort(key=lambda row: int(row["id"]))
    ids = tuple(row["id"] for row in rows)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: empty table or duplicate row IDs")
    if [int(value) for value in ids] != list(range(len(ids))):
        raise ValueError(f"{path}: row IDs must be contiguous from zero")
    titles = tuple(row.get("title") or row.get("name") or "" for row in rows)
    texts = tuple(
        ". ".join(value for value in (title.strip(), row["wiki_text"].strip()) if value)
        for title, row in zip(titles, rows)
    )
    return Table(name, ids, titles, texts, hashlib.sha256(raw).hexdigest())


def dataset_directory(root: Path, name: str) -> Path:
    """Resolve a dataset folder from a direct or parent path."""
    if name not in TABLES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {tuple(TABLES)}")
    root = Path(root).expanduser().resolve()
    return root if (root / "queries.json").is_file() else root / name


def load_dataset(root: Path, name: str) -> Dataset:
    """Load dataset."""
    directory = dataset_directory(root, name)
    document = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    expected = {table: f"{table}.csv" for table in TABLES[name]}
    if document.get("tables") != expected:
        raise ValueError(f"{directory}: expected table mapping {expected}")
    tables = {
        table: _table(table, directory / filename)
        for table, filename in expected.items()
    }
    for table, data in tables.items():
        if f"n_{table}" in document and document[f"n_{table}"] != len(data):
            raise ValueError(f"{directory}: wrong row count for {table}")
    queries, truth = [], {}
    for row in document["queries"]:
        qid = row["id"]
        if not isinstance(qid, int) or qid in truth:
            raise ValueError(f"{directory}: invalid or duplicate query ID {qid!r}")
        relation = row.get("theta", document.get("theta", {}).get("name"))
        sides = (
            tuple(document["theta"][relation]["sides"])
            if presets()[name]["relation_layout"] == "named"
            else TABLES[name]
        )
        if not relation or not set(sides).issubset(tables):
            raise ValueError(f"{directory}: invalid relation for query {qid}")
        filters = {
            table: row[f"phi_{table}_text"]
            for table in tables
            if row.get(f"phi_{table}_text")
        }
        if not filters or not set(filters).issubset(sides):
            raise ValueError(f"{directory}: query {qid} must filter a joined table")
        condition, gt = row["theta_text"], row["gt_cardinality"]
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"{directory}: missing join condition for query {qid}")
        if not isinstance(gt, int) or gt < 0:
            raise ValueError(f"{directory}: invalid ground truth for query {qid}")
        queries.append(Query(qid, relation, condition, sides, filters))
        truth[qid] = gt
    if not queries:
        raise ValueError(f"{directory}: empty query set")
    return Dataset(name, tables, tuple(queries), truth)
