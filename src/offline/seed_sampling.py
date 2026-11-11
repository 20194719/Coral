"""Offline seed sampling."""

from __future__ import annotations
from src.offline.records import _assign_C_strata, CandidatePool, StratifiedSample
from src.online.sampling import STRATA
from typing import Dict
import numpy as np


def stratified_seed_sample(cfg, pool: CandidatePool, seed: int = 0) -> StratifiedSample:
    """Stratified seed sample."""
    n_R, n_S = (pool.n_R, pool.n_S)
    N = n_R * n_S
    rng = np.random.default_rng(seed)
    c_strat = _assign_C_strata(cfg, pool)
    c_key = pool.key()
    c_set = set(c_key.tolist())
    mass: Dict[str, int] = {}
    rs, ss, strata, incl = ([], [], [], [])
    for name in STRATA[:-1]:
        members = np.flatnonzero(c_strat == name)
        mass[name] = int(members.size)
        cap = int(cfg.stratum_caps.get(name, 0))
        if members.size == 0 or cap == 0:
            continue
        take = min(cap, members.size)
        chosen = rng.choice(members, size=take, replace=False)
        p = take / members.size
        rs.append(pool.pairs[chosen, 0])
        ss.append(pool.pairs[chosen, 1])
        strata.append(np.full(take, name, dtype=object))
        incl.append(np.full(take, p, dtype=np.float64))
    M_bg = N - len(c_set)
    mass["background"] = int(M_bg)
    cap_bg = int(cfg.stratum_caps.get("background", 0))
    if cap_bg > 0 and M_bg > 0:
        drawn, guard = ([], 0)
        target = min(cap_bg, M_bg)
        seen = set()
        while len(drawn) < target and guard < 50 * target + 1000:
            f = int(rng.integers(0, N))
            guard += 1
            if f in c_set or f in seen:
                continue
            seen.add(f)
            drawn.append(f)
        drawn = np.array(drawn, dtype=np.int64)
        rs.append(drawn // n_S)
        ss.append(drawn % n_S)
        strata.append(np.full(len(drawn), "background", dtype=object))
        incl.append(np.full(len(drawn), len(drawn) / M_bg, dtype=np.float64))
    return StratifiedSample(
        r=np.concatenate(rs) if rs else np.zeros(0, np.int64),
        s=np.concatenate(ss) if ss else np.zeros(0, np.int64),
        stratum=np.concatenate(strata) if strata else np.zeros(0, object),
        incl_prob=np.concatenate(incl) if incl else np.zeros(0, np.float64),
        mass=mass,
        N=N,
    )
