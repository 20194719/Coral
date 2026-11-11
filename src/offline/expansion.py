"""Expand candidate pairs."""

from __future__ import annotations
import numpy as np
from src.offline.neighbors import NearestIndex
from src.offline.records import ExpansionCandidates

EXPANSION_POLICY = "same_side_positive_neighbors"


def _seed_positive_pairs_by_type(cards, types, seed_labels):
    """Seed positive pairs by type."""
    if (
        not isinstance(seed_labels, np.ndarray)
        or seed_labels.dtype != bool
        or seed_labels.shape != (len(cards), types.T)
    ):
        raise ValueError("expansion requires complete boolean LLM seed labels")
    out: dict[int, list] = {}
    for i, t in np.argwhere(seed_labels):
        t = int(t)
        out.setdefault(t, []).append((int(cards.pairs[i, 0]), int(cards.pairs[i, 1])))
    return out


def _same_side_neighbors(index, embeddings, anchors, k):
    """Same side neighbors."""
    anchors = sorted(anchors)
    if not anchors or k <= 0:
        return {}
    _, indices = index.search(embeddings[anchors], k=min(k + 1, len(embeddings)))
    return {
        anchor: [
            int(j) for j in neighbors if 0 <= j < len(embeddings) and int(j) != anchor
        ][:k]
        for anchor, neighbors in zip(anchors, indices)
    }


def expand_pool(
    cfg,
    pool,
    cards,
    types,
    e_R,
    e_S,
    seed_labels,
    per_anchor: int = 20,
) -> ExpansionCandidates:
    """Expand pool."""
    n_S = pool.n_S
    seed_set = (
        set((cards.pairs[:, 0].astype(np.int64) * n_S + cards.pairs[:, 1]).tolist())
        if len(cards)
        else set()
    )
    pos_by_type = _seed_positive_pairs_by_type(cards, types, seed_labels)
    annS = NearestIndex(e_S)
    annR = NearestIndex(e_R)
    neighbors_s = _same_side_neighbors(
        annS, e_S, {s for pos in pos_by_type.values() for _, s in pos}, per_anchor
    )
    neighbors_r = _same_side_neighbors(
        annR, e_R, {r for pos in pos_by_type.values() for r, _ in pos}, per_anchor
    )
    cand_pairs = []
    seen = set()
    for t, pos in pos_by_type.items():
        for r, s in sorted(pos):
            for other_s in neighbors_s.get(s, []):
                f = r * n_S + other_s
                if f in seed_set or f in seen:
                    continue
                seen.add(f)
                cand_pairs.append((r, other_s))
        for r, s in sorted(pos, key=lambda pair: (pair[1], pair[0])):
            for other_r in neighbors_r.get(r, []):
                f = other_r * n_S + s
                if f in seed_set or f in seen:
                    continue
                seen.add(f)
                cand_pairs.append((other_r, s))
    if not cand_pairs:
        return ExpansionCandidates(np.zeros((0, 2), np.int64))
    return ExpansionCandidates(
        pairs=np.asarray(cand_pairs, dtype=np.int64),
    )


def sample_expansion(expansion, count, seed):
    """Sample expansion."""
    if count < 0 or len(expansion) < count:
        raise ValueError(
            f"Expansion has {len(expansion)} distinct pairs, but {count} are required by the offline budget"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(expansion), size=count, replace=False)
    return expansion.pairs[indices]


def _allocate(count, capacities, weights):
    """Allocate."""
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if count > capacities.sum() or count < 0:
        raise ValueError("Insufficient unseen pairs for the offline budget")
    allocation = np.zeros(len(capacities), dtype=np.int64)
    remaining = int(count)
    while remaining:
        active = np.flatnonzero(allocation < capacities)
        w = weights[active]
        if not w.sum():
            w = np.ones(len(active))
        shares = remaining * w / w.sum()
        take = np.minimum(
            np.floor(shares).astype(np.int64), capacities[active] - allocation[active]
        )
        allocation[active] += take
        remaining -= int(take.sum())
        for index in np.argsort(-(shares - np.floor(shares)), kind="stable"):
            group = active[index]
            if remaining and allocation[group] < capacities[group]:
                allocation[group] += 1
                remaining -= 1
    return allocation


def _sample_complement(rng, size, blocked, count):
    """Sample complement."""
    blocked = np.unique(np.asarray(blocked, dtype=np.int64))
    rank = rng.choice(size - len(blocked), size=count, replace=False)
    low, high = rank.copy(), rank + len(blocked)
    while np.any(low < high):
        mid = (low + high) // 2
        enough = mid + 1 - np.searchsorted(blocked, mid, side="right") >= rank + 1
        high = np.where(enough, mid, high)
        low = np.where(enough, low, mid + 1)
    return low


def second_round(cfg, pool, seed_pairs, expansion):
    """Use expansion pairs first and stratify any remaining budget."""
    from src.online.sampling import STRATA
    from src.offline.records import _assign_C_strata

    count = cfg.offline_pairs - len(seed_pairs)
    selected = sample_expansion(expansion, min(count, len(expansion)), cfg.seed_global)
    deficit = count - len(selected)
    rng = np.random.default_rng(cfg.seed_global + 1)
    keys = pool.key()
    excluded = np.concatenate((seed_pairs, selected), axis=0)
    excluded_keys = excluded[:, 0] * pool.n_S + excluded[:, 1]
    strata = _assign_C_strata(cfg, pool)
    available = ~np.isin(keys, excluded_keys)
    candidates = [np.flatnonzero(available & (strata == name)) for name in STRATA[:-1]]
    blocked = np.union1d(keys, excluded_keys)
    total = pool.n_R * pool.n_S
    capacity = [len(ids) for ids in candidates] + [total - len(blocked)]
    allocation = _allocate(
        deficit, capacity, [cfg.stratum_caps[name] for name in STRATA]
    )
    supplement = []
    for ids, n in zip(candidates, allocation[:-1]):
        if n:
            supplement.extend(
                pool.pairs[rng.choice(ids, size=n, replace=False)].tolist()
            )
    if allocation[-1]:
        flat = _sample_complement(rng, total, blocked, int(allocation[-1]))
        supplement.extend(np.column_stack((flat // pool.n_S, flat % pool.n_S)).tolist())
    supplement = np.asarray(supplement, dtype=np.int64).reshape(-1, 2)
    pairs = np.concatenate((selected, supplement))
    source = np.array(
        ["expand"] * len(selected) + ["supplement"] * len(supplement), dtype=object
    )
    combined = np.concatenate((seed_pairs, pairs))
    if len(combined) != cfg.offline_pairs or len(np.unique(combined, axis=0)) != len(
        combined
    ):
        raise ValueError("Offline samples must meet the distinct-pair budget")
    audit = dict(
        budget=cfg.offline_pairs,
        seed_pairs=len(seed_pairs),
        neighbors=20,
        expansion_candidates=len(expansion),
        selected_expansion=len(selected),
        supplemental_pairs=len(supplement),
        supplemental_by_stratum=dict(zip(STRATA, map(int, allocation))),
    )
    return pairs, source, audit
