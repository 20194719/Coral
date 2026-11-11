"""Boosted filters conditioned on reviewed predicate descriptions."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from src.offline.filter_features import context_features


class SemanticTreeModel:
    def __init__(
        self,
        leaves=7,
        rank=8,
        vocabulary=4000,
        representation="identity",
        attributes="values",
        iterations=160,
        regularization=10.0,
        minimum_leaf=25,
        learning_rate=0.06,
    ):
        """Initialize model state."""
        if representation != "identity":
            raise ValueError("Unknown semantic tree representation")
        self.leaves, self.rank, self.vocabulary = (
            int(leaves),
            int(rank),
            int(vocabulary),
        )
        if self.leaves < 2 or self.rank < 1 or self.vocabulary < 1:
            raise ValueError("Invalid semantic tree dimensions")
        self.representation = representation
        if attributes not in ("fields", "values"):
            raise ValueError("Unknown predicate attribute representation")
        self.attributes = attributes
        self.iterations = int(iterations)
        self.regularization = float(regularization)
        self.minimum_leaf = int(minimum_leaf)
        self.learning_rate = float(learning_rate)
        self.prepared = {}

    def parts(self, tags):
        """Parse semantic descriptions."""
        fields, values = {}, []
        for tag in tags or []:
            key, separator, value = str(tag).partition("=")
            if separator:
                key = " ".join(key.strip().casefold().replace("_", " ").split())
                fields[key] = 1.0
                if self.attributes == "values":
                    normalized = " ".join(
                        value.strip().casefold().replace("_", " ").split()
                    )
                    fields[f"{key}={normalized}"] = 1.0
            values.append((value if separator else key).strip().replace("_", " "))
        return fields, values

    def features(self, values, predicate, tags, ids=None):
        """Build filter features."""
        fields, descriptions = self.parts(tags)
        document_ids = np.arange(len(values)) if ids is None else np.asarray(ids, int)
        raw = context_features(values, "wide")[document_ids]
        query = self.pca.transform(np.asarray(predicate)[None])[0]
        category = self.fields.transform([fields])[0]
        known = sum(key in self.fields.vocabulary_ for key in fields)
        metadata = np.r_[
            category, query, len(descriptions), known / max(len(fields), 1)
        ]
        parts = [raw, np.broadcast_to(metadata, (len(raw), len(metadata)))]
        return np.column_stack(parts)

    def fit(
        self,
        values,
        ids,
        labels,
        predicates,
        texts,
        predicate_features,
        sample_weight=None,
    ):
        """Fit the model."""
        if not values or not (
            len(values) == len(ids) == len(labels) == len(predicates)
        ):
            raise ValueError("Misaligned semantic tree training data")
        if predicate_features is None or len(predicate_features) != len(values):
            raise ValueError("Reviewed descriptions must align with predicates")
        if len(predicates) < 2 or texts is None or len(texts) != len(values[0]):
            raise ValueError("Semantic trees require multiple probes and table texts")
        vectors = np.stack(predicates)
        self.pca = PCA(
            n_components=min(self.rank, len(vectors) - 1, vectors.shape[1]),
            svd_solver="full",
        ).fit(vectors)
        self.fields = DictVectorizer(sparse=False).fit(
            [self.parts(tags)[0] for tags in predicate_features]
        )
        x = np.concatenate(
            [
                self.features(v, p, tags, index)
                for v, p, tags, index in zip(
                    values, predicates, predicate_features, ids
                )
            ]
        )
        y = np.concatenate(labels)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Semantic tree training requires both labels")
        self.model = HistGradientBoostingClassifier(
            max_iter=self.iterations,
            max_leaf_nodes=self.leaves,
            min_samples_leaf=self.minimum_leaf,
            l2_regularization=self.regularization,
            learning_rate=self.learning_rate,
            random_state=0,
            early_stopping=False,
        ).fit(x, y, sample_weight=sample_weight)
        for predicate, tags in zip(predicates, predicate_features):
            self.register(predicate, tags)
        return self

    def register(self, predicate, features):
        """Register predicate descriptions."""
        self.prepared[np.asarray(predicate).tobytes()] = features

    def predict(self, values, predicate):
        """Predict probabilities."""
        tags = self.prepared.get(np.asarray(predicate).tobytes())
        return self.model.predict_proba(self.features(values, predicate, tags))[:, 1]
