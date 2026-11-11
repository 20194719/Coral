"""Provide the shared tree filter interface."""

from src.offline.semantic_tree import SemanticTreeModel


class TreeAdapter:
    kind = "semantic_tree"

    def __init__(self, **parameters):
        """Initialize model state."""
        self.parameters = parameters

    def fit(
        self,
        values,
        ids,
        labels,
        vectors,
        texts,
        tags,
        predicate_texts,
        sample_weight=None,
    ):
        """Fit the model."""
        self.model = SemanticTreeModel(**self.parameters).fit(
            values,
            ids,
            labels,
            vectors,
            [""] * len(texts),
            tags,
            sample_weight=sample_weight,
        )
        self.model.examples = [
            dict(predicate=t, features=f) for t, f in zip(predicate_texts, tags) if f
        ]
        return self

    def predict(self, values, documents=None, predicate=None):
        """Predict probabilities."""
        return self.model.predict(values, predicate)
