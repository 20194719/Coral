"""Build candidate pairs."""

from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List
import numpy as np
from src.offline.neighbors import NearestIndex
from src.offline.records import CandidatePool

_TOK = re.compile("[a-z0-9]+")
_LEAD_ARTICLE = re.compile("^(the|a|an)\\s+")
_PAREN = re.compile("\\([^)]*\\)")


def _tokenize(text: str) -> List[str]:
    """Tokenize retrieval documents."""
    return _TOK.findall((text or "").lower())


def _clean_title(t: str) -> str:
    """Normalize retrieval titles."""
    t = (t or "").lower()
    t = _PAREN.sub(" ", t)
    t = re.split("[:\\-–]", t)[0]
    t = _LEAD_ARTICLE.sub("", t.strip())
    t = re.sub("[^a-z0-9 ]+", " ", t)
    return re.sub("\\s+", " ", t).strip()


class BM25Index:
    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        """Initialize model state."""
        from sklearn.feature_extraction.text import CountVectorizer

        self.vec = CountVectorizer(
            tokenizer=_tokenize, token_pattern=None, lowercase=False
        )
        counts = self.vec.fit_transform(corpus_texts).astype(np.float32).tocsr()
        n_docs, _ = counts.shape
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(dl.mean()) if n_docs else 1.0
        denom_doc = k1 * (1 - b + b * dl / max(avgdl, 1e-06))
        coo = counts.tocoo()
        tf = coo.data
        w = idf[coo.col] * (tf * (k1 + 1.0)) / (tf + denom_doc[coo.row])
        import scipy.sparse as sp

        self.W = sp.csr_matrix(
            (w, (coo.row, coo.col)), shape=counts.shape, dtype=np.float32
        )
        self.n_docs = n_docs

    def topk(
        self, query_texts: List[str], k: int, q_chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Topk."""
        Q = self.vec.transform(query_texts).astype(np.float32)
        Q.data[:] = 1.0
        k = min(k, self.n_docs)
        nq = Q.shape[0]
        out_i = np.zeros((nq, k), dtype=np.int64)
        out_s = np.zeros((nq, k), dtype=np.float32)
        for a in range(0, nq, q_chunk):
            b_ = min(a + q_chunk, nq)
            scores = (Q[a:b_] @ self.W.T).toarray()
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(b_ - a)[:, None]
            order = np.argsort(-scores[rows, part], axis=1)
            idx = part[rows, order]
            out_i[a:b_] = idx
            out_s[a:b_] = scores[rows, idx]
        return (out_i, out_s)


def _retrieve_pairs(idx: np.ndarray, score: np.ndarray, query_is_R: bool):
    """Retrieve candidate document pairs."""
    nq, k = idx.shape
    qids = np.repeat(np.arange(nq), k)
    cids = idx.ravel()
    sc = score.ravel()
    if query_is_R:
        r, s = (qids, cids)
    else:
        r, s = (cids, qids)
    return (r.astype(np.int64), s.astype(np.int64), sc.astype(np.float32))


def _mention_pairs(titles_R, titles_S, entities_R, entities_S):
    """Match mentioned entity names."""
    norm_titles = [_clean_title(t) for t in titles_R]
    norm_names = [_clean_title(t) for t in titles_S]
    title_to_r: Dict[str, List[int]] = defaultdict(list)
    name_to_s: Dict[str, List[int]] = defaultdict(list)
    for r, t in enumerate(norm_titles):
        if t:
            title_to_r[t].append(r)
    for s, t in enumerate(norm_names):
        if t:
            name_to_s[t].append(s)
    out = set()

    def _aliases(name: str):
        """Generate entity name aliases."""
        c = _clean_title(name)
        toks = c.split()
        al = {c} if c else set()
        if len(toks) >= 2:
            al.add(f"{toks[0]} {toks[-1]}")
        return al

    for s, ents in enumerate(entities_S or []):
        for e in ents:
            al = _aliases(e)
            for a in al:
                for r in title_to_r.get(a, []):
                    out.add((r, s))
    for r, ents in enumerate(entities_R or []):
        for e in ents:
            al = _aliases(e)
            for a in al:
                for s in name_to_s.get(a, []):
                    out.add((r, s))
    return out


def build_candidate_pool(
    cfg,
    e_R,
    e_S,
    titles_R,
    titles_S,
    texts_R,
    texts_S,
    entities_R=None,
    entities_S=None,
) -> CandidatePool:
    """Build candidate pool."""
    n_R, n_S = (len(texts_R), len(texts_S))
    bm25: Dict[int, float] = {}
    dense: Dict[int, float] = {}
    mention: Dict[int, bool] = {}

    def _flat(r, s):
        """Flatten extracted entity values."""
        return int(r) * n_S + int(s)

    idx_RtoS = BM25Index(texts_S)
    i, sc = idx_RtoS.topk(texts_R, cfg.bm25_topk)
    r, s, v = _retrieve_pairs(i, sc, query_is_R=True)
    for rr, ss, vv in zip(r, s, v):
        f = _flat(rr, ss)
        if vv > bm25.get(f, -1):
            bm25[f] = float(vv)
    idx_StoR = BM25Index(texts_R)
    i, sc = idx_StoR.topk(texts_S, cfg.bm25_topk)
    r, s, v = _retrieve_pairs(i, sc, query_is_R=False)
    for rr, ss, vv in zip(r, s, v):
        f = _flat(rr, ss)
        if vv > bm25.get(f, -1):
            bm25[f] = float(vv)
    annS = NearestIndex(e_S)
    sc, i = annS.search(e_R, k=cfg.dense_topk)
    r, s, v = _retrieve_pairs(i, sc, query_is_R=True)
    for rr, ss, vv in zip(r, s, v):
        f = _flat(rr, ss)
        if vv > dense.get(f, -2):
            dense[f] = float(vv)
    annR = NearestIndex(e_R)
    sc, i = annR.search(e_S, k=cfg.dense_topk)
    r, s, v = _retrieve_pairs(i, sc, query_is_R=False)
    for rr, ss, vv in zip(r, s, v):
        f = _flat(rr, ss)
        if vv > dense.get(f, -2):
            dense[f] = float(vv)
    for rr, ss in _mention_pairs(titles_R, titles_S, entities_R, entities_S):
        f = _flat(rr, ss)
        mention[f] = True
    keys = sorted(set(bm25) | set(dense) | set(mention))
    M = len(keys)
    pairs = np.zeros((M, 2), dtype=np.int64)
    bm = np.zeros(M, dtype=np.float32)
    dn = np.full(M, -1.0, dtype=np.float32)
    mn = np.zeros(M, dtype=bool)
    src = np.empty(M, dtype=object)
    for j, f in enumerate(keys):
        pairs[j, 0] = f // n_S
        pairs[j, 1] = f % n_S
        has_b, has_d, has_m = (f in bm25, f in dense, f in mention)
        bm[j] = bm25.get(f, 0.0)
        dn[j] = dense.get(f, -1.0)
        mn[j] = has_m
        chans = [
            c for c, h in (("bm25", has_b), ("dense", has_d), ("mention", has_m)) if h
        ]
        src[j] = "multiple" if len(chans) > 1 else chans[0]
    return CandidatePool(
        n_R=n_R,
        n_S=n_S,
        pairs=pairs,
        bm25_score=bm,
        dense_score=dn,
        mention_flag=mn,
        source=src,
    )
