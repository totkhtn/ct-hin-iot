"""Core of CT-HIN: batch heterogeneous network, k-nearest-neighbour file graph and label spreading.

The functions follow Section IV of the paper and the frozen pre-registered implementation
(``prereg/frozen/ct_v33.py``). Two implementation details differ only in determinism:
entity vocabularies are built from sorted token lists, so column order no longer depends on
Python's hash seed. Floating-point ties in the neighbour search can therefore resolve
differently from the frozen run in a handful of edges (see ``docs/REPRODUCIBILITY.md``).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

ENTITY_TYPES = ("s", "ip", "dom", "url")  # configuration frozen for the holdout


def incidence(records: Sequence[dict], kinds: Iterable[str]) -> sp.csr_matrix:
    """Binary file--entity incidence matrix X (files x entities) over the given entity types.

    ``records[i][k]`` is the list of entities of type ``k`` that occur in file ``i``.
    Entities of different types are kept apart by prefixing them with their type.
    """
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, rec in enumerate(records):
        for k in kinds:
            for tok in sorted(set(rec.get(k, ()))):
                j = vocab.setdefault(f"{k}:{tok}", len(vocab))
                rows.append(i)
                cols.append(j)
    n = len(records)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(vocab)))


def entity_embedding(X: sp.csr_matrix, max_df: float = 0.02, min_df: int = 2) -> sp.csr_matrix:
    """In-batch IDF weighting and row L2 normalisation (Eq. 1).

    Entities that occur in fewer than ``min_df`` files or in more than ``max_df * n`` files are
    dropped; the remaining entity e receives the weight log(n / d_e).
    """
    n = X.shape[0]
    df = np.asarray(X.sum(0)).ravel()
    keep = (df >= min_df) & (df <= max_df * n)
    w = np.where(keep, np.log(n / np.maximum(df, 1)), 0.0).astype(np.float32)
    Z = X @ sp.diags(w)
    norm = np.sqrt(np.asarray(Z.multiply(Z).sum(1)).ravel())
    return sp.diags(1.0 / np.maximum(norm, 1e-12)) @ Z


def knn_graph(Z, k: int = 20, chunk: int = 2000):
    """Symmetrised, row-normalised k-nearest-neighbour graph on cosine similarity (Section IV-C).

    Returns ``(P, has_edge)`` where P = D^{-1} A with A_ij = max(S_ij [j in N_k(i)], S_ji [i in N_k(j)]),
    and ``has_edge[i]`` is False for files without any edge.
    """
    n = Z.shape[0]
    ZT = Z.T.tocsc() if sp.issparse(Z) else Z.T
    rows, cols, vals = [], [], []
    for start in range(0, n, chunk):
        S = Z[start:start + chunk] @ ZT
        S = S.toarray() if sp.issparse(S) else np.asarray(S)
        for r in range(S.shape[0]):
            S[r, start + r] = -1.0  # exclude self-similarity
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        v = np.clip(np.take_along_axis(S, idx, 1), 0, None)
        rows.append(np.repeat(np.arange(start, start + S.shape[0]), k))
        cols.append(idx.ravel())
        vals.append(v.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1e-12)) @ A, deg > 0


def label_spreading(P, has_edge: np.ndarray, prior: np.ndarray, alpha: float = 0.9, iterations: int = 20) -> np.ndarray:
    """f <- (1 - alpha) p + alpha P f for files with an edge; isolated files keep their prior (Eq. 3)."""
    f = prior.astype(np.float64).copy()
    for _ in range(iterations):
        f = np.where(has_edge, (1 - alpha) * prior + alpha * (P @ f), prior)
    return f


def ct_hin(records: Sequence[dict], prior: np.ndarray, kinds: Iterable[str] = ENTITY_TYPES, max_df: float = 0.02,
           k: int = 20, alpha: float = 0.9, iterations: int = 20) -> np.ndarray:
    """Run CT-HIN on one batch and return the propagated scores f (Algorithm 1).

    ``prior`` is the score of the unchanged local detector for each file (the LightGBM margin of the
    leave-one-architecture-out model in the paper). Label a file malicious when f >= tau (tau = 0.5
    in the pre-registered configuration).
    """
    P, has_edge = graph(records, kinds, max_df, k)
    return label_spreading(P, has_edge, np.asarray(prior, np.float64), alpha, iterations)


def graph(records: Sequence[dict], kinds: Iterable[str] = ENTITY_TYPES, max_df: float = 0.02, k: int = 20):
    """Build the CT-HIN file graph of a batch; the graph does not depend on the prior."""
    return knn_graph(entity_embedding(incidence(records, list(kinds)), max_df), k)
