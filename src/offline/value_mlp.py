"""Implement the MLP filter model."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from src.offline.semantic_tree import SemanticTreeModel


CHECKPOINTS = (40, 80, 160, 240)


def candidates():
    """Candidates."""
    base = dict(
        width=64,
        depth=2,
        residual=False,
        knots=0,
        dropout=0.0,
        weight_decay=0.01,
        learning_rate=0.001,
        batch_size=4096,
        brier_mix=0.0,
    )
    variants = [
        ("plain64", {}),
        ("plain128", dict(width=128)),
        ("residual64", dict(residual=True, dropout=0.1)),
        ("residual128", dict(residual=True, width=128, dropout=0.1)),
        ("piecewise64", dict(residual=True, knots=8, dropout=0.1)),
        ("piecewise128", dict(residual=True, knots=8, width=128, dropout=0.1)),
        (
            "piecewise64_strong",
            dict(residual=True, knots=8, dropout=0.2, weight_decay=0.1),
        ),
        ("piecewise64_brier", dict(residual=True, knots=8, dropout=0.1, brier_mix=0.5)),
    ]
    return [dict(base, **changes, name=name) for name, changes in variants]


class Preprocessor:
    def fit(self, x, continuous, knots):
        """Fit the model."""
        self.continuous = np.asarray(continuous, int)
        self.mean = np.zeros(x.shape[1], np.float32)
        self.scale = np.ones(x.shape[1], np.float32)
        self.mean[self.continuous] = x[:, self.continuous].mean(0)
        self.scale[self.continuous] = np.maximum(x[:, self.continuous].std(0), 0.025)

        self.knots = (
            np.quantile(
                (x[:, self.continuous] - self.mean[self.continuous])
                / self.scale[self.continuous],
                np.linspace(0.05, 0.95, knots),
                axis=0,
            ).T.astype(np.float32)
            if knots
            else np.empty((len(continuous), 0), np.float32)
        )
        return self

    def transform(self, x):
        """Transform model inputs."""
        z = ((np.asarray(x, np.float32) - self.mean) / self.scale).clip(-12, 12)
        if self.knots.shape[1]:
            hinges = np.maximum(z[:, self.continuous, None] - self.knots[None], 0)
            z = np.column_stack([z, hinges.reshape(len(z), -1)])
        return np.asarray(z, np.float32)


class Network(nn.Module):
    def __init__(self, dimension, options, initial_rate):
        """Initialize model state."""
        super().__init__()
        width = options["width"]
        self.residual = options["residual"]
        self.input = nn.Linear(dimension, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, width), nn.GELU(), nn.Dropout(options["dropout"])
                )
                for _ in range(options["depth"])
            ]
        )
        self.output = nn.Linear(width, 1)
        self.skip = nn.Linear(dimension, 1, bias=False)
        nn.init.zeros_(self.skip.weight)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(
            self.output.bias, float(np.log(initial_rate / (1 - initial_rate)))
        )

    def forward(self, x):
        """Compute model outputs."""
        h = nn.functional.gelu(self.input(x))
        for block in self.blocks:
            h = (h + block(h)) / np.sqrt(2) if self.residual else block(h)
        return (self.output(h) + self.skip(x)).squeeze(1)


def train_head(
    x,
    y,
    weights,
    continuous,
    options,
    epochs,
    device="cpu",
    seed=0,
    validation_x=None,
    checkpoint=None,
):
    """Train head."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    prep = Preprocessor().fit(x, continuous, options["knots"])
    tx = torch.as_tensor(prep.transform(x), device=device)
    ty = torch.as_tensor(np.asarray(y, np.float32), device=device)
    tw = torch.as_tensor(np.asarray(weights, np.float32), device=device)
    vx = (
        torch.as_tensor(prep.transform(validation_x), device=device)
        if validation_x is not None
        else None
    )
    rate = float(np.clip(np.average(y, weights=weights), 1e-5, 1 - 1e-5))
    net = Network(tx.shape[1], options, rate).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), options["batch_size"]):
            idx = torch.as_tensor(
                order[start : start + options["batch_size"]], device=device
            )
            logits = net(tx[idx])
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, ty[idx], reduction="none"
            )
            mix = options["brier_mix"]
            loss_rows = (1 - mix) * bce + mix * (
                torch.sigmoid(logits) - ty[idx]
            ).square()
            loss = (loss_rows * tw[idx]).sum() / tw[idx].sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
        if checkpoint is not None and epoch in CHECKPOINTS:
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(vx)).cpu().numpy()
            checkpoint(epoch, p)
    return ProbabilityHead(prep, net.cpu().eval(), options)


class ProbabilityHead:
    def __init__(self, preprocessing, network, options):
        """Initialize model state."""
        self.preprocessing, self.network, self.options = preprocessing, network, options

    def predict_proba(self, x):
        """Predict proba."""
        z = self.preprocessing.transform(x)
        rows = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(z), 4096):
                rows.append(
                    torch.sigmoid(
                        self.network(torch.from_numpy(z[start : start + 4096]))
                    ).numpy()
                )
        p = np.concatenate(rows).astype(np.float64)
        return np.column_stack([1 - p, p])


class ValueMLPAdapter:
    kind = "value_mlp"

    def __init__(self, mlp, epochs, device="cpu", **parameters):
        """Initialize model state."""
        self.device = device
        self.parameters, self.options, self.epochs = parameters, mlp, int(epochs)

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
        self.model = SemanticTreeModel(**self.parameters)
        vectors = np.stack(vectors)
        rank = min(self.parameters["rank"], len(vectors) - 1, vectors.shape[1])
        self.model.pca = PCA(n_components=rank, svd_solver="full").fit(vectors)
        self.model.fields = DictVectorizer(sparse=False).fit(
            [self.model.parts(t)[0] for t in tags]
        )
        x = np.concatenate(
            [
                self.model.features(v, p, t, i)
                for v, p, t, i in zip(values, vectors, tags, ids)
            ]
        )
        y = np.concatenate(labels)
        categorical = len(self.model.fields.feature_names_)
        raw = x.shape[1] - categorical - rank - 2
        continuous = np.r_[np.arange(raw), np.arange(raw + categorical, x.shape[1])]
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        self.model.model = train_head(
            x, y, weights, continuous, self.options, self.epochs, device=self.device
        )
        self.parameter_count = sum(
            p.numel() for p in self.model.model.network.parameters()
        )
        for p, t in zip(vectors, tags):
            self.model.register(p, t)
        self.model.examples = [
            dict(predicate=t, features=f) for t, f in zip(predicate_texts, tags) if f
        ]
        self.input_dimension = x.shape[1]
        return self

    def predict(self, values, documents=None, predicate=None):
        """Predict probabilities."""
        return self.model.predict(values, predicate)
