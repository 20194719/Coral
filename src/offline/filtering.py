"""Build filter predictors and prepare query probabilities."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from src.data.storage import Stages, atomic_json
from src.offline import preparation as P
from src.offline.profiles import ProfileStore
from src.offline.filters import NumericHead
from src.offline.canonical import CanonicalPredicate
from src.offline.semantic_features import SemanticFeatures
from src.offline.paraphrases import generate_table
from src.offline.training import fit_filter
from src.offline.validation import select


def collect_labels(cfg, table, store, texts, encoder, llm, entity, stages):
    """Label sampled documents for each training predicate."""
    vectors = encoder(texts)
    ids, labels = [], []
    for index, (text, vector) in enumerate(zip(texts, vectors)):

        def label():
            """Collect one predicate's document labels."""
            order = np.argsort(store.feats(vector)[:, 0], kind="mergesort")
            count = min(len(order), cfg["filter_calib_docs"])
            selected = order[
                np.unique(np.linspace(0, len(order) - 1, count).round().astype(int))
            ]
            answers = llm.judge_filter(
                text, [table.texts[int(i)] for i in selected], entity.lower()
            )
            if len(answers) != len(selected) or any(
                v not in (False, True) for v in answers
            ):
                raise ValueError("Every filter training sample requires a binary label")
            return selected, np.asarray(answers, int)

        selected, answers = stages.build(f"labels-{index:03}", label)
        ids.append(selected)
        labels.append(answers)
    return ids, labels


def training_bank(
    cfg, table, store, texts, ids, labels, role, encoder, llm, output, stages
):
    """Prepare labeled predicate families and semantic descriptions."""

    def describe():
        """Match the reference training annotation batches for each dataset."""
        descriptor = SemanticFeatures(llm, role)
        if cfg["dataset"] == "trex":
            tags = []
            for start in range(0, len(texts), 6):
                tags.extend(descriptor.annotate(texts[start : start + 6]))
            return tags
        return descriptor.annotate(texts)

    tags = stages.build("descriptions", describe)
    variants = (
        [[] for _ in texts]
        if not cfg["filter_rewrites"]
        else [
            [a["text"] for a in row["accepted"]]
            for row in generate_table(
                llm, texts, role, output / "paraphrases", cfg["filter_paraphrases"]
            )
        ]
    )
    bank = dict(texts=[], ids=[], labels=[], tags=[], families=[])
    for i, text in enumerate(texts):
        for form in [text, *variants[i]]:
            bank["texts"].append(form)
            bank["ids"].append(ids[i])
            bank["labels"].append(labels[i])
            bank["tags"].append(tags[i])
            bank["families"].append(i)
    bank["vectors"] = encoder(bank["texts"])
    return bank


class Filters:
    """Manage shared filter heads for a join view."""

    def __init__(
        self,
        cfg,
        view,
        paths,
        tables,
        emb,
        encoder,
        llm,
        output,
        shared,
        device="cpu",
        resume=False,
    ):
        """Initialize model state."""
        self.encoder, self.heads, self.normalizers, self.descriptors = (
            encoder,
            {},
            {},
            {},
        )
        self.cache = shared.setdefault("__predictions__", {})
        self.prediction_dir = Path(output).parent / "query-filters"
        self.prediction_dir.mkdir(exist_ok=True)
        self.resume = resume
        stages = Stages(Path(output) / "stages", resume=resume)
        oracle = cfg["dataset"] == "biodex"
        schema = (
            None
            if oracle
            else P.profile_schema(
                P.parameters(cfg, view, encoder, output),
                view,
                tables,
                llm,
                output,
                resume=resume,
            )
        )
        for side, table, documents, entity in zip(
            ("R", "S"), tables, emb, view["entities"]
        ):
            role = cfg["filter_roles"][table.name]
            if table.name not in shared:
                target = Path(output) / "filters" / table.name
                target.mkdir(parents=True, exist_ok=True)
                saved = Stages(target / "stages", resume=resume)
                runtime = SimpleNamespace(
                    encoder=encoder,
                    sbert_dim=cfg["embedding_dimension"],
                    numeric_example_subject=view.get(
                        "numeric_example_subject", "people"
                    ),
                )
                if oracle:
                    from src.offline.oracle_filters import (
                        training_probes,
                        FullTextStore,
                    )

                    fields = []
                    store = FullTextStore(documents, table.texts)
                    texts, ids, labels = training_probes(
                        table, paths["dataset_directory"]
                    )
                else:
                    fields = schema[side]
                    profiles = P.profiles(stages, table.texts, fields, role, side, llm)
                    store = saved.build(
                        "profile-features",
                        lambda: ProfileStore.build(
                            runtime, profiles, fields, documents, texts=table.texts
                        ),
                    )
                    if cfg["filter_verification"]:
                        from src.offline.predicate_review import generate
                    else:
                        from src.offline.predicates import generate
                    texts = generate(
                        cfg, saved, table, role, side, llm, target, profiles, fields
                    )
                    ids, labels = collect_labels(
                        cfg, table, store, texts, encoder, llm, entity, saved
                    )
                bank = saved.build(
                    "training-bank",
                    lambda: training_bank(
                        cfg,
                        table,
                        store,
                        texts,
                        ids,
                        labels,
                        role,
                        encoder,
                        llm,
                        target,
                        saved,
                    ),
                )
                options = dict(
                    cfg["tree_parameters"][table.name],
                    original_weight=cfg["filter_original_weight"],
                )
                if cfg["filter_model"] == "mlp":
                    options = dict(
                        attributes="values",
                        rank=options["rank"],
                        representation="identity",
                        original_weight=cfg["filter_original_weight"],
                        mlp=cfg["mlp_parameters"][table.name],
                        epochs=cfg["mlp_epochs"],
                    )
                if cfg["filter_cv"]:
                    options = saved.build(
                        "selected-parameters",
                        lambda: select(
                            bank,
                            store,
                            cfg["filter_model"],
                            device,
                            target / "validation.json",
                        ),
                    )
                model = saved.build(
                    "model",
                    lambda: fit_filter(
                        bank, store, cfg["filter_model"], options, device
                    ),
                )
                head = NumericHead(store, runtime, llm, fields, role)
                head._m = model
                shared[table.name] = head
                atomic_json(
                    target / "model.json",
                    dict(
                        model=cfg["filter_model"],
                        parameters=options,
                        cross_validation=cfg["filter_cv"],
                        original_predicates=len(texts),
                        labeled_pairs=sum(map(len, labels)),
                        training_rows=sum(map(len, bank["labels"])),
                        input_dimensions=int(
                            model.model.model.network.input.in_features
                        )
                        if cfg["filter_model"] == "mlp"
                        else int(model.model.model.n_features_in_),
                        pca_rank=int(model.model.pca.n_components_),
                    ),
                )
            head = shared[table.name]
            self.heads[table.name] = head
            if cfg["filter_normalization"]:
                self.normalizers[table.name] = CanonicalPredicate(llm, role)
            self.descriptors[table.name] = SemanticFeatures(
                llm, role, head._m.model.examples
            )

    def prediction(self, table, text):
        """Predict filter probabilities for a query condition."""
        if text is None:
            return None
        key = (table, text)
        filename = hashlib.sha256(repr(key).encode()).hexdigest() + ".npy"
        target = self.prediction_dir / filename
        if key not in self.cache and self.resume and target.exists():
            self.cache[key] = np.load(target)
        if key not in self.cache:
            head = self.heads[table]
            normalize = self.normalizers.get(table, lambda value: value)
            descriptor = self.descriptors[table]

            def encode(value):
                """Encode and register a semantic condition."""
                vector = self.encoder([normalize(value)])[0]
                head._m.model.register(vector, descriptor(value))
                return vector

            vector = (
                encode(text) if head.spec_for(text) is None else self.encoder([text])[0]
            )
            prediction = head.proba_pushdown(text, vector, encode=encode)[0]
            if (
                not np.isfinite(prediction).all()
                or ((prediction < 0) | (prediction > 1)).any()
            ):
                raise ValueError("Filter predictions must be finite probabilities")
            self.cache[key] = prediction
            with target.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, prediction)
            target.with_suffix(".tmp").replace(target)
        return self.cache[key]
