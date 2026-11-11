"""Relation clustering and representation."""

from __future__ import annotations
import src.inference.embeddings as sbert
from src.offline.records import DiscoveredTypes
from src.prompts import summary_prompt, parse_summary
import numpy as np


def _cluster_labels(cfg, X: np.ndarray) -> np.ndarray:
    """Cluster labels."""
    if cfg.rtype_kind == "hdbscan":
        import hdbscan

        cl = hdbscan.HDBSCAN(min_cluster_size=cfg.rtype_min)
        return cl.fit_predict(X)
    from sklearn.cluster import KMeans

    k = min(cfg.rtype_k, X.shape[0])
    return KMeans(n_clusters=k, n_init=5, random_state=cfg.seed_global).fit_predict(X)


def discover_types(cfg, cards, llm) -> DiscoveredTypes:
    """Discover types."""
    all_emb, all_txt = ([], [])
    for emb, ph in zip(cards.embeddings, cards.phrases):
        for j in range(len(ph)):
            all_emb.append(emb[j])
            all_txt.append(ph[j])
    if not all_emb:
        raise RuntimeError("no relation phrases mined; cannot discover θ-types")
    X = np.asarray(all_emb, dtype=np.float32)
    lab = _cluster_labels(cfg, X)
    clusters = [np.flatnonzero(lab == t) for t in sorted(set(lab.tolist())) if t >= 0]
    kept = [members for members in clusters if members.size >= cfg.rtype_min]
    if not kept:
        kept = clusters
    if not kept:
        raise RuntimeError("no relation clusters survived")
    top_n = int(cfg.rtype_summary_neighbors)
    if top_n <= 0:
        raise ValueError("rtype_summary_neighbors must be positive")
    representatives = []
    for members in kept:
        mu = X[members].mean(axis=0)
        mu = mu / max(np.linalg.norm(mu), 1e-08)
        nearest = members[np.argsort(-(X[members] @ mu), kind="stable")]
        phrases, seen = ([], set())
        for j in nearest:
            phrase = all_txt[j].strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
                if len(phrases) == top_n:
                    break
        if not phrases:
            raise ValueError("relation cluster has no nonempty representative phrases")
        representatives.append(phrases)
    replies = llm.complete(
        [summary_prompt(p, cfg.entity_R, cfg.entity_S) for p in representatives],
        max_tokens=256,
    )
    if len(replies) != len(representatives):
        raise ValueError("Missing relation-cluster summaries")
    labels = [parse_summary(reply) for reply in replies]
    q_types = sbert.encode(cfg, labels)
    return DiscoveredTypes(
        labels=labels,
        q_types=q_types,
        representatives=representatives,
    )
