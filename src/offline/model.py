"""Implement join pair score models and training."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.online.sampling import STRATA

BG = STRATA.index("background")


@dataclass
class Rows:
    r: np.ndarray
    s: np.ndarray
    t: np.ndarray
    y: np.ndarray
    w: np.ndarray
    strat: np.ndarray

    def __len__(self) -> int:
        """Return the number of records."""
        return int(self.r.shape[0])


def build_rows(labeled, pool=None, strata_of_C=None) -> Rows:
    """Build rows."""
    if (pool is None) != (strata_of_C is None):
        raise ValueError("pool and strata_of_C must be provided together")
    geo: Dict[int, int] = {}
    if pool is not None and strata_of_C is not None:
        if int(pool.n_S) != int(labeled.n_S) or len(strata_of_C) != len(pool.pairs):
            raise ValueError("candidate geometry does not match the labelled corpus")
        n_S_pool = int(pool.n_S)
        pr = pool.pairs[:, 0].astype(np.int64)
        ps = pool.pairs[:, 1].astype(np.int64)
        ids = np.array([STRATA.index(str(x)) for x in strata_of_C], dtype=np.int8)
        geo = dict(zip((pr * n_S_pool + ps).tolist(), ids.tolist()))
    sel, strat_l = ([], [])
    for i in range(labeled.pairs.shape[0]):
        st = labeled.stratum[i]
        if st is not None:
            sel.append(i)
            strat_l.append(STRATA.index(str(st)))
            continue
        if pool is None:
            raise ValueError(
                "unstratified labels require pool and strata_of_C; refusing to drop harvest/expansion labels"
            )
        key = int(labeled.pairs[i, 0]) * int(labeled.n_S) + int(labeled.pairs[i, 1])
        sel.append(i)
        strat_l.append(int(geo.get(key, BG)))
    sel = np.asarray(sel, dtype=np.int64)
    if sel.size == 0:
        raise RuntimeError("no labelled rows available for training")
    r = labeled.pairs[sel, 0].astype(np.int64)
    s = labeled.pairs[sel, 1].astype(np.int64)
    strat = np.asarray(strat_l, dtype=np.int8)
    mass = dict(labeled.strata_mass)
    key = r * int(labeled.n_S) + s
    n_pairs_by_stratum: Dict[str, int] = {}
    for j, stn in enumerate(STRATA):
        m = strat == j
        n_pairs_by_stratum[stn] = int(np.unique(key[m]).size) if m.any() else 0
    uniq, counts = np.unique(key, return_counts=True)
    rows_per_pair = dict(zip(uniq.tolist(), counts.tolist()))
    w = np.empty(sel.size, np.float64)
    for i in range(sel.size):
        stn = STRATA[strat[i]]
        npair = n_pairs_by_stratum[stn]
        per = rows_per_pair[int(key[i])]
        w[i] = float(mass.get(stn, 0)) / max(npair, 1) / max(per, 1)
    if not np.all(np.isfinite(w) & (w > 0)):
        raise ValueError("every training label must have a positive finite weight")
    return Rows(
        r=r,
        s=s,
        t=labeled.rel_type[sel].astype(np.int64),
        y=labeled.label[sel].astype(bool),
        w=w,
        strat=strat,
    )


LIN_NAMES = ("cos_rs", "cos_rq", "cos_sq", "bm25", "mention")
N_LIN = len(LIN_NAMES)


@dataclass
class Grid:
    eR: np.ndarray
    eS: np.ndarray
    strat: np.ndarray
    bm25: np.ndarray
    mention: np.ndarray
    U: Optional[np.ndarray] = None
    zR: Optional[np.ndarray] = None
    zS: Optional[np.ndarray] = None

    @property
    def n_R(self) -> int:
        """Return the left table size."""
        return int(self.eR.shape[0])

    @property
    def n_S(self) -> int:
        """Return the right table size."""
        return int(self.eS.shape[0])

    @property
    def D(self) -> int:
        """Return the feature dimension."""
        return int((self.zR if self.zR is not None else self.eR).shape[1])

    def tower_R(self) -> np.ndarray:
        """Tower R."""
        return self.zR if self.zR is not None else self.eR

    def tower_S(self) -> np.ndarray:
        """Tower S."""
        return self.zS if self.zS is not None else self.eS

    def proj(self, q: np.ndarray) -> np.ndarray:
        """Project document embeddings."""
        q = np.asarray(q, np.float32)
        return q if self.U is None else (q @ self.U.T).astype(np.float32)


def build_grid(
    pool, e_R, e_S, strata_of_C, tower_dim: int = 256, seed: int = 0
) -> Grid:
    """Build the pair feature grid."""
    n_R, n_S = (int(pool.n_R), int(pool.n_S))
    strat = np.full((n_R, n_S), BG, dtype=np.int8)
    bm25 = np.zeros((n_R, n_S), dtype=np.float32)
    mention = np.zeros((n_R, n_S), dtype=bool)
    pr = pool.pairs[:, 0].astype(np.int64)
    ps = pool.pairs[:, 1].astype(np.int64)
    strat[pr, ps] = np.array([STRATA.index(str(x)) for x in strata_of_C], dtype=np.int8)
    bm25[pr, ps] = np.asarray(pool.bm25_score, dtype=np.float32)
    mention[pr, ps] = np.asarray(pool.mention_flag, dtype=bool)

    def nrm(e):
        """Normalize embedding vectors."""
        e = np.asarray(e, dtype=np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

    eR, eS = (nrm(e_R), nrm(e_S))
    pos = bm25[bm25 > 0]
    if pos.size:
        bm25 = (bm25 / float(np.quantile(pos, 0.99))).clip(0.0, 4.0)
    zR = zS = None
    D = eR.shape[1]
    U = None
    if 0 < tower_dim < D:
        from sklearn.decomposition import PCA

        pooled = np.concatenate([eR, eS], axis=0)
        k = int(min(tower_dim, pooled.shape[0] - 1, D))
        pca = PCA(n_components=k, random_state=seed)
        pca.fit(pooled)
        U = pca.components_.astype(np.float32)
        if k < tower_dim:
            U = np.vstack([U, np.zeros((tower_dim - k, D), np.float32)])
        zR, zS = (eR @ U.T, eS @ U.T)
        sd = float(np.concatenate([zR, zS]).std()) or 1.0
        U = U / sd
        zR, zS = ((zR / sd).astype(np.float32), (zS / sd).astype(np.float32))
        print(
            f"[grid] tower space: PCA {D} → {k}{('' if k == tower_dim else f' (zero-padded to {tower_dim})')}, explained var {float(pca.explained_variance_ratio_.sum()):.3f}"
        )
    return Grid(
        eR=eR, eS=eS, strat=strat, bm25=bm25, mention=mention, U=U, zR=zR, zS=zS
    )


def lin_feats_rows(g: Grid, r: np.ndarray, s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Lin feats rows."""
    r = np.asarray(r, np.int64)
    s = np.asarray(s, np.int64)
    X = np.empty((r.shape[0], N_LIN), dtype=np.float32)
    X[:, 0] = np.einsum("ij,ij->i", g.eR[r], g.eS[s])
    X[:, 1] = g.eR[r] @ q
    X[:, 2] = g.eS[s] @ q
    X[:, 3] = g.bm25[r, s]
    X[:, 4] = g.mention[r, s].astype(np.float32)
    return X


def _mlp(din: int, dh: int, dout: int, depth: int, dropout: float) -> nn.Module:
    """Construct a projection network."""
    if depth <= 0:
        return nn.Linear(din, dout)
    layers: List[nn.Module] = [nn.Linear(din, dh), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(dh, dh), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dh, dout))
    return nn.Sequential(*layers)


@dataclass
class Spec:
    d: int = 32
    hidden: int = 256
    depth: int = 1
    dropout: float = 0.0
    lam_init: float = 0.02
    l2_lam: float = 0.001
    wd: float = 0.0001
    lr: float = 0.003
    epochs: int = 400
    batch: int = 0
    w_alpha: float = 1.0
    seed: int = 0


class PairModel(nn.Module):
    def __init__(self, spec: Spec, D: int):
        """Initialize model state."""
        super().__init__()
        self.spec = spec
        self.lin = nn.Linear(N_LIN, 1, bias=True)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.weight[0, 0] = 1.0 / 0.05
            self.lin.bias.fill_(-9.0)
        self.d = int(spec.d)
        if self.d > 0:
            self.encR = _mlp(D, spec.hidden, self.d, spec.depth, spec.dropout)
            self.encS = _mlp(D, spec.hidden, self.d, spec.depth, spec.dropout)
            self.encT = _mlp(D, spec.hidden, self.d, spec.depth, spec.dropout)
            self.lam_raw = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(max(spec.lam_init, 1e-06)))))
            )

    @property
    def lam(self) -> torch.Tensor:
        """Return the residual scale."""
        if self.d <= 0:
            return torch.zeros((), device=self.lin.weight.device)
        return F.softplus(self.lam_raw)

    def towerR(self, eR: torch.Tensor) -> torch.Tensor:
        """Encode left document vectors."""
        return F.normalize(self.encR(eR), dim=-1)

    def towerS(self, eS: torch.Tensor) -> torch.Tensor:
        """Encode right document vectors."""
        return F.normalize(self.encS(eS), dim=-1)

    def towerT(self, q: torch.Tensor) -> torch.Tensor:
        """Encode relation vectors."""
        return torch.tanh(self.encT(q))

    def logit_rows(
        self,
        X: torch.Tensor,
        strat: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute join pair scores."""
        z = self.lin(X).squeeze(-1)
        if self.d > 0:
            z = z + self.lam * (a * A * B).sum(-1)
        return z


def _row_tensors(rows: Rows, g: Grid, q_types: np.ndarray, dev) -> dict:
    """Convert training rows to tensors."""
    X = np.empty((len(rows), N_LIN), dtype=np.float32)
    for t in np.unique(rows.t):
        m = rows.t == t
        X[m] = lin_feats_rows(g, rows.r[m], rows.s[m], q_types[int(t)])
    ur, inv_r = np.unique(rows.r, return_inverse=True)
    us, inv_s = np.unique(rows.s, return_inverse=True)
    ut, inv_t = np.unique(rows.t, return_inverse=True)
    return dict(
        X=torch.tensor(X, device=dev),
        strat=torch.tensor(rows.strat.astype(np.int64), device=dev),
        eR_u=torch.tensor(g.tower_R()[ur], device=dev),
        eS_u=torch.tensor(g.tower_S()[us], device=dev),
        qT_u=torch.tensor(
            np.stack([g.proj(q_types[int(t)]) for t in ut]),
            dtype=torch.float32,
            device=dev,
        ),
        inv_r=torch.tensor(inv_r.astype(np.int64), device=dev),
        inv_s=torch.tensor(inv_s.astype(np.int64), device=dev),
        inv_t=torch.tensor(inv_t.astype(np.int64), device=dev),
        y=torch.tensor(rows.y.astype(np.float32), device=dev),
    )


def _row_logits(
    model: "PairModel", T: dict, sel: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute training pair logits."""
    if model.d > 0:
        A_u = model.towerR(T["eR_u"])
        B_u = model.towerS(T["eS_u"])
        a_u = model.towerT(T["qT_u"])
    if sel is None:
        X, st = (T["X"], T["strat"])
        ir, is_, it = (T["inv_r"], T["inv_s"], T["inv_t"])
    else:
        X, st = (T["X"][sel], T["strat"][sel])
        ir, is_, it = (T["inv_r"][sel], T["inv_s"][sel], T["inv_t"][sel])
    if model.d <= 0:
        return model.logit_rows(X, st, None, None, None)
    return model.logit_rows(X, st, A_u[ir], B_u[is_], a_u[it])


def _wbce(z: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute weighted binary loss."""
    ll = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return (w * ll).sum() / w.sum().clamp_min(1e-08)


def _lin_params(model: PairModel):
    """Select linear model parameters."""
    out = [model.lin.weight, model.lin.bias]
    return out


def _fit_linear(
    model: PairModel,
    T: dict,
    wt: torch.Tensor,
    sel: torch.Tensor,
    offset: Optional[torch.Tensor],
    iters: int = 200,
) -> float:
    """Fit linear score parameters."""
    params = _lin_params(model)
    for p in params:
        p.requires_grad_(True)
    X, st, y, w = (T["X"][sel], T["strat"][sel], T["y"][sel], wt[sel])
    off = None if offset is None else offset[sel].detach()
    opt = torch.optim.LBFGS(
        params,
        max_iter=iters,
        history_size=20,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        """Evaluate the optimizer loss."""
        opt.zero_grad()
        z = model.lin(X).squeeze(-1)
        if off is not None:
            z = z + off
        loss = _wbce(z, y, w)
        loss.backward()
        return loss

    return float(opt.step(closure))


def _train_one(
    rows: Rows,
    g: Grid,
    q_types: np.ndarray,
    spec: Spec,
    device: str = "cpu",
    verbose: bool = False,
) -> PairModel:
    """Train one stratum model."""
    torch.manual_seed(spec.seed)
    dev = torch.device(device)
    model = PairModel(spec, g.D).to(dev)
    T = _row_tensors(rows, g, q_types, dev)
    w = np.asarray(rows.w, np.float64) ** spec.w_alpha
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    idx_all = torch.arange(len(rows), device=dev)
    hist: List[float] = []
    lA = _fit_linear(model, T, wt, idx_all, offset=None)
    hist.append(lA)
    if verbose:
        print(f"    [A] linear solved, loss {lA:.6f}", flush=True)
    if model.d > 0:
        for p in _lin_params(model):
            p.requires_grad_(False)
        tower_params = [p for n, p in model.named_parameters() if n.startswith("enc")]
        opt = torch.optim.Adam(
            [
                {"params": tower_params, "weight_decay": spec.wd},
                {"params": [model.lam_raw], "weight_decay": 0.0, "lr": spec.lr * 3},
            ],
            lr=spec.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(spec.epochs, 1)
        )
        for ep in range(spec.epochs):
            model.train()
            order = idx_all[torch.randperm(idx_all.numel(), device=dev)]
            chunks = (
                [order]
                if spec.batch <= 0
                else [
                    order[i : i + spec.batch]
                    for i in range(0, order.numel(), spec.batch)
                ]
            )
            tot = 0.0
            for sel in chunks:
                z = _row_logits(model, T, sel)
                loss = _wbce(z, T["y"][sel], wt[sel]) + spec.l2_lam * model.lam**2
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * int(sel.numel())
            sched.step()
            hist.append(tot / max(int(idx_all.numel()), 1))
            if verbose and (ep % 100 == 0 or ep == spec.epochs - 1):
                print(
                    f"    [B] ep {ep:>4} loss {hist[-1]:.6f} lam {float(model.lam):.4f}",
                    flush=True,
                )
        model.eval()
        with torch.no_grad():
            A_u = model.towerR(T["eR_u"])
            B_u = model.towerS(T["eS_u"])
            a_u = model.towerT(T["qT_u"])
            inter = float(model.lam) * (
                a_u[T["inv_t"]] * A_u[T["inv_r"]] * B_u[T["inv_s"]]
            ).sum(-1)
        for p in model.parameters():
            p.requires_grad_(False)
        lC = _fit_linear(model, T, wt, idx_all, offset=inter)
        hist.append(lC)
        if verbose:
            print(f"    [C] linear re-solved, loss {lC:.6f}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _score_one(
    model: PairModel, g: Grid, q: np.ndarray, device: str = "cpu", row_chunk: int = 128
) -> np.ndarray:
    """Score one stratum model."""
    dev = torch.device(device)
    m = model.to(dev).eval()
    eR = torch.tensor(g.eR, device=dev)
    eS = torch.tensor(g.eS, device=dev)
    qt = torch.tensor(np.asarray(q, np.float32), device=dev)
    zR = torch.tensor(g.tower_R(), device=dev)
    zS = torch.tensor(g.tower_S(), device=dev)
    zq = torch.tensor(g.proj(np.asarray(q, np.float32)), device=dev)
    A = m.towerR(zR) if m.d > 0 else None
    B = m.towerS(zS) if m.d > 0 else None
    a = m.towerT(zq.unsqueeze(0)).squeeze(0) if m.d > 0 else None
    lam = float(m.lam)
    wlin = m.lin.weight.detach().squeeze(0)
    blin = float(m.lin.bias.detach())
    cos_rq = eR @ qt
    cos_sq = eS @ qt
    out = np.empty((g.n_R, g.n_S), dtype=np.float32)
    for st in range(0, g.n_R, row_chunk):
        en = min(st + row_chunk, g.n_R)
        cos_rs = eR[st:en] @ eS.t()
        bm = torch.tensor(g.bm25[st:en], device=dev)
        me = torch.tensor(g.mention[st:en].astype(np.float32), device=dev)
        z = (
            blin
            + wlin[0] * cos_rs
            + wlin[1] * cos_rq[st:en, None]
            + wlin[2] * cos_sq[None, :]
            + wlin[3] * bm
            + wlin[4] * me
        )
        if A is not None:
            z = z + lam * (A[st:en] * a[None, :] @ B.t())
        out[st:en] = z.float().cpu().numpy()
    return out


class StratifiedModel(nn.ModuleDict):
    """Independent parameters for each populated offline stratum."""


def train(rows, g, q_types, spec, device="cpu", verbose=False):
    """Train join pair score models."""
    models = StratifiedModel()
    for group in np.unique(g.strat):
        mask = rows.strat == group
        if not mask.any():
            raise ValueError(
                f"No training observations in populated stratum {STRATA[group]}"
            )
        subset = Rows(
            **{name: getattr(rows, name)[mask] for name in Rows.__dataclass_fields__}
        )
        if verbose:
            print(f"[model] stratum={STRATA[group]} rows={len(subset)}", flush=True)
        models[str(int(group))] = _train_one(subset, g, q_types, spec, device, verbose)
    return models


@torch.no_grad()
def score_grid(models, g, q, device="cpu"):
    """Score the pair grid."""
    result = np.empty(g.strat.shape, dtype=np.float32)
    for group in np.unique(g.strat):
        scores = _score_one(models[str(int(group))], g, q, device)
        mask = g.strat == group
        result[mask] = scores[mask]
    return result
