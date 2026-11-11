"""Prepare BioDEX filter conditions, oracle labels, and full-text similarities."""

import csv
import json
from pathlib import Path
import numpy as np
from src.data.storage import read_pickle


BIODEX_PUBLICATION_PROBES = (
    ("Randomized Controlled Trial", "articles reporting a randomised controlled trial"),
    ("Observational Study", "articles reporting an observational study"),
    ("Multicenter Study", "articles reporting a multicentre study"),
    (
        "Research Support, Non-U.S",
        "articles supported by non-U.S. government research funding",
    ),
    ("Comparative Study", "comparative studies"),
)
BIODEX_ARTICLE_ATC_PROBES = (
    ("A", "articles whose patient was given a drug for the stomach, gut or metabolism"),
    ("H", "articles whose patient was given a corticosteroid or other systemic hormone"),
    ("B", "articles whose patient was given an anticoagulant or other blood-related drug"),
    ("R", "articles whose patient was given a drug for the airways"),
)
BIODEX_DRUG_PROBES = (
    ("G", "drugs for the genito-urinary system or sex hormones"),
    ("H", "systemic hormonal drugs, excluding sex hormones"),
    ("M", "drugs for the musculo-skeletal system"),
    ("S", "drugs for the sensory organs"),
    ("V", "drugs in the ATC 'various' class"),
)
BIODEX_REACTION_PROBES = (
    ("Q327657", "adverse reactions treated in rheumatology"),
    ("Q162555", "adverse reactions treated in oncology"),
    ("Q105650", "adverse reactions treated in urology"),
    ("Q2861470", "adverse reactions treated in emergency medicine"),
)


def _metadata(directory, name, fields):
    """Read annotation fields in document ID order."""
    path = Path(directory) / f"{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", *fields}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing BioDEX annotation columns {fields}")
        rows = sorted(reader, key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: row IDs must be contiguous from zero")
    return rows


def _atc_classes(rows):
    """Read drug classification labels."""
    return [
        {value[0] for value in json.loads(row["claims"] or "{}").get("P267", [])}
        for row in rows
    ]


def _article_probes(directory, rows):
    """Label fixed article calibration conditions."""
    probes = {
        text: [value in row["publication_types"] for row in rows]
        for value, text in BIODEX_PUBLICATION_PROBES
    }
    drugs = _metadata(directory, "drugs", ("claims",))
    atc = _atc_classes(drugs)
    library = read_pickle(
        Path(directory) / "oracle library" / "biodex_AD" / "theta_library.pkl"
    )
    if (library["n_R"], library["n_S"]) != (len(rows), len(drugs)):
        raise ValueError("Article-drug oracle dimensions do not match the tables")
    classes = [set() for _ in rows]
    for article, drug in library["thetas"]["theta1"]["positives"]:
        article, drug = int(article), int(drug)
        if not (0 <= article < len(rows) and 0 <= drug < len(drugs)):
            raise ValueError("Article-drug oracle contains an invalid row ID")
        classes[article].update(atc[drug])
    probes.update(
        (text, [code in values for values in classes])
        for code, text in BIODEX_ARTICLE_ATC_PROBES
    )
    return probes


def training_probes(table, directory):
    """Prepare fixed calibration conditions and their annotation-backed labels."""
    fields = ("publication_types",) if table.name == "articles" else ("claims",)
    rows = _metadata(directory, table.name, fields)
    if tuple(row["id"] for row in rows) != table.ids:
        raise ValueError("BioDEX filter annotations do not match the table IDs")
    if table.name == "articles":
        probes = _article_probes(directory, rows)
    elif table.name == "drugs":
        classes = _atc_classes(rows)
        probes = {
            text: [code in values for values in classes]
            for code, text in BIODEX_DRUG_PROBES
        }
    elif table.name == "reactions":
        specialties = [
            json.loads(row["claims"] or "{}").get("P1995", []) for row in rows
        ]
        probes = {
            text: [code in values for values in specialties]
            for code, text in BIODEX_REACTION_PROBES
        }
    else:
        raise ValueError(f"No BioDEX calibration conditions for {table.name!r}")
    texts, labels = list(probes), np.asarray(list(probes.values()))
    if not texts or labels.shape != (len(texts), len(table)):
        raise ValueError("Oracle filter labels do not match the table")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Oracle filter labels must be binary")
    return texts, [np.arange(len(table)) for _ in texts], list(labels.astype(int))


class FullTextStore:
    """Provide full-text filter features."""

    def __init__(self, documents, texts):
        """Initialize model state."""
        self.doc_emb = np.asarray(documents, np.float32)
        self.texts, self.n_docs = list(texts), len(documents)
        self.nums = [{} for _ in documents]

    def field_feats(self, predicate):
        """Build full-text similarity features."""
        documents = self.doc_emb / np.maximum(
            np.linalg.norm(self.doc_emb, axis=1, keepdims=True), 1e-9
        )
        query = np.asarray(predicate, np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        similarity = documents @ query
        return np.column_stack([similarity, similarity])

    def feats(self, predicate):
        """Compute document similarity features."""
        return self.field_feats(predicate)
