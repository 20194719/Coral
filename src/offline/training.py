"""Train shared filter models from labeled predicate families."""

import numpy as np
from src.offline.tree_filter import TreeAdapter
from src.offline.value_mlp import ValueMLPAdapter


def row_weights(bank, original_weight):
    """Weight original conditions and equivalent rewrites."""
    families = np.asarray(bank["families"])
    weights = []
    for index, labels in enumerate(bank["labels"]):
        family = np.flatnonzero(families == families[index])
        weight = (
            1.0
            if len(family) == 1
            else original_weight
            if index == family[0]
            else (1 - original_weight) / (len(family) - 1)
        )
        weights.append(np.full(len(labels), weight))
    return np.concatenate(weights)


def fit_filter(bank, store, kind, options, device):
    """Fit a filter probability model."""
    parameters = {
        k: v
        for k, v in options.items()
        if k not in ("name", "original_weight", "mlp", "epochs")
    }
    if kind == "mlp":
        model = ValueMLPAdapter(
            options["mlp"], options["epochs"], device=device, **parameters
        )
    else:
        model = TreeAdapter(**parameters)
    values = [store.field_feats(v) for v in bank["vectors"]]
    model.fit(
        values,
        bank["ids"],
        bank["labels"],
        bank["vectors"],
        store.texts,
        bank["tags"],
        bank["texts"],
        sample_weight=row_weights(bank, options["original_weight"]),
    )
    originals = [
        i
        for i, family in enumerate(bank["families"])
        if i == bank["families"].index(family)
    ]
    model.model.examples = [
        dict(predicate=bank["texts"][i], features=bank["tags"][i])
        for i in originals
        if bank["tags"][i]
    ]
    return model
