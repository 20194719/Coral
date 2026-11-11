"""Select filter parameters with grouped five-fold validation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from src.data.storage import atomic_json
from src.offline.filter_features import context_features
from src.offline.semantic_tree import SemanticTreeModel
from src.offline.training import row_weights
from src.offline.value_mlp import candidates as mlp_candidates, CHECKPOINTS, train_head


def tree_candidates():
    """Define the original tree parameter search."""
    base = dict(
        attributes="values",
        rank=8,
        leaves=7,
        iterations=160,
        regularization=10.0,
        minimum_leaf=25,
        learning_rate=0.06,
        representation="identity",
        original_weight=0.5,
    )
    variants = [
        ("control_original", dict(attributes="fields", original_weight=1.0)),
        ("control_augmented", dict(attributes="fields", original_weight=0.25)),
        ("values_equal", dict(original_weight=0.25)),
        ("values_half", {}),
        ("values_rank16", dict(rank=16)),
        ("values_leaves15", dict(leaves=15)),
        ("values_rank16_leaves15", dict(rank=16, leaves=15)),
        (
            "values_less_regularized",
            dict(rank=16, leaves=15, iterations=240, regularization=2.0),
        ),
        ("values_rank32", dict(rank=32, leaves=15)),
        ("values_leaves31", dict(rank=16, leaves=31, iterations=240)),
    ]
    return [dict(base, **changes, name=name) for name, changes in variants]


def assignments(documents, families):
    """Build document-disjoint and family-disjoint folds."""
    rng = np.random.default_rng(1729)
    if min(len(np.unique(documents)), len(np.unique(families))) < 5:
        raise ValueError(
            "Five-fold validation needs at least five documents and families"
        )
    doc_fold = {
        int(v): i % 5 for i, v in enumerate(rng.permutation(np.unique(documents)))
    }
    family_fold = {
        int(v): i % 5 for i, v in enumerate(rng.permutation(np.unique(families)))
    }
    return dict(
        document=np.array([doc_fold[int(v)] for v in documents]),
        family=np.array([family_fold[int(v)] for v in families]),
    )


def balanced_brier(y, prediction, original):
    """Score originals and rewrites with equal weight."""
    masks = [mask for mask in (original, ~original) if mask.any()]
    return float(
        np.mean([np.mean((prediction[mask] - y[mask]) ** 2) for mask in masks])
    )


def select(bank, store, kind, device, output):
    """Select parameters without workload cardinalities."""
    vectors = np.stack(bank["vectors"])
    raw = np.concatenate(
        [
            context_features(store.field_feats(v), "wide")[ids]
            for v, ids in zip(vectors, bank["ids"])
        ]
    )
    form = np.concatenate([np.full(len(ids), i) for i, ids in enumerate(bank["ids"])])
    docs, y = np.concatenate(bank["ids"]), np.concatenate(bank["labels"])
    families = np.asarray(bank["families"])[form]
    first = {f: bank["families"].index(f) for f in bank["families"]}
    original = np.array([i == first[bank["families"][i]] for i in form])
    folds = assignments(docs, families)
    settings = tree_candidates() if kind == "tree" else mlp_candidates()
    epochs = (None,) if kind == "tree" else CHECKPOINTS
    predictions = {
        (c["name"], e, p): np.full(len(y), np.nan)
        for c in settings
        for e in epochs
        for p in folds
    }
    state = dict(
        status="running",
        folds=5,
        kind=kind,
        candidates=settings,
        checkpoints=list(CHECKPOINTS) if kind == "mlp" else [],
        protocols=list(folds),
        progress=[],
        criterion="Equal document/family and original/rewrite Brier",
    )
    atomic_json(output, state)
    for protocol, fold_ids in folds.items():
        for fold in range(5):
            val, train = fold_ids == fold, fold_ids != fold
            feature_cache = {}
            for config in settings:
                print(
                    f"[filter-cv] table={output.parent.name} protocol={protocol} "
                    f"fold={fold + 1}/5 candidate={config['name']}",
                    flush=True,
                )
                weight = row_weights(bank, config.get("original_weight", 0.5))
                active = train & (weight > 0)
                selected_forms = np.unique(form[active])
                attributes = config.get("attributes", "values")
                key = (attributes, tuple(selected_forms))
                if key not in feature_cache:
                    pca = PCA(
                        n_components=min(32, len(selected_forms) - 1, vectors.shape[1]),
                        svd_solver="full",
                    ).fit(vectors[selected_forms])
                    projected = pca.transform(vectors)
                    extractor = SemanticTreeModel(attributes=attributes)
                    tags = [extractor.parts(t)[0] for t in bank["tags"]]
                    enc = DictVectorizer(sparse=False).fit(
                        [tags[i] for i in selected_forms]
                    )
                    category = enc.transform(tags)
                    known = [
                        sum(k in enc.vocabulary_ for k in t) / max(len(t), 1)
                        for t in tags
                    ]
                    count = [len(extractor.parts(t)[1]) for t in bank["tags"]]
                    feature_cache[key] = category, projected, count, known
                category, projected, count, known = feature_cache[key]
                rank = min(config.get("rank", 16), projected.shape[1])
                metadata = np.column_stack([category, projected[:, :rank], count, known])
                x = np.column_stack([raw, metadata[form]])
                if kind == "tree":
                    model = HistGradientBoostingClassifier(
                        max_iter=config["iterations"],
                        max_leaf_nodes=config["leaves"],
                        min_samples_leaf=config["minimum_leaf"],
                        l2_regularization=config["regularization"],
                        learning_rate=config["learning_rate"],
                        random_state=0,
                        early_stopping=False,
                    ).fit(x[active], y[active], sample_weight=weight[active])
                    predictions[config["name"], None, protocol][
                        val
                    ] = model.predict_proba(x[val])[:, 1]
                else:
                    x = x.astype(np.float32)
                    continuous = np.r_[
                        np.arange(raw.shape[1]),
                        np.arange(raw.shape[1] + category.shape[1], x.shape[1]),
                    ]

                    def record(epoch, prediction):
                        """Save held-out checkpoint predictions."""
                        predictions[config["name"], epoch, protocol][val] = prediction

                    train_head(
                        x[active],
                        y[active],
                        weight[active],
                        continuous,
                        config,
                        max(CHECKPOINTS),
                        device=device,
                        seed=fold,
                        validation_x=x[val],
                        checkpoint=record,
                    )
                state["progress"].append(
                    dict(protocol=protocol, fold=fold, candidate=config["name"])
                )
                atomic_json(output, state)
    scores = {}
    for config in settings:
        for epoch in epochs:
            key = config["name"] + (f"-epoch{epoch}" if epoch else "")
            p = [predictions[config["name"], epoch, protocol] for protocol in folds]
            if not all(np.isfinite(v).all() for v in p):
                raise ValueError("Validation predictions are incomplete")
            scores[key] = float(np.mean([balanced_brier(y, v, original) for v in p]))
    eligible = [c for c in settings if c.get("attributes", "values") == "values"]
    choices = []
    for config in eligible:
        for epoch in epochs:
            key = config["name"] + (f"-epoch{epoch}" if epoch else "")
            choice = (
                config
                if kind == "tree"
                else dict(
                    name=key,
                    attributes="values",
                    rank=16,
                    representation="identity",
                    original_weight=0.5,
                    mlp=config,
                    epochs=epoch,
                )
            )
            choices.append((scores[key], choice))
    chosen = min(choices, key=lambda x: x[0])[1]
    atomic_json(output, dict(state, status="complete", scores=scores, chosen=chosen))
    print(
        f"[filter-cv] table={output.parent.name} selected={chosen['name']}", flush=True
    )
    return chosen
