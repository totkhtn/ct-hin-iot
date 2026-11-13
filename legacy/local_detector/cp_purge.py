#!/usr/bin/env python3
"""CP-Purge: composition-preserving group purge for LOAO evaluation.

Implements, for one held-out architecture h:
  * protocol "none"  : T = files of h; training side = all other files      (Alg. 1, purge=False)
  * protocol "purge" : standard purge, drop training files whose similarity
                       group or program key occurs in h                      (Alg. 1, purge=True)
  * protocol "cp"    : composition-preserving purge (Alg. CP-Purge)
  * the purge impact audit (Alg. 2) for the three protocols side by side.

CP-Purge formulation (exact pairwise semantics of Alg. 1)
---------------------------------------------------------
A *node* is a similarity group ("g:<id>") or a program key ("k:<id>").
A node is *shared* if files of h and files of other architectures both carry it.
Each shared node n gets a binary decision x_n:
    x_n = 0 -> node goes to the test side   (training files carrying n are dropped;
                                              this is the standard purge)
    x_n = 1 -> node goes to the training side (test files carrying n are dropped)
A training file is kept iff all its shared nodes have x=1; a test file is kept iff
all its shared nodes have x=0. Hence no kept training file and no kept test file
share a group or a program key (leakage-free), and x = 0 reproduces the standard
purge exactly, so CP-Purge never has a smaller minimum stratum retention.

Stage 1:  max t  s.t. retained_Q[s] >= t * n_Q[s],  retained_T[s] >= t * n_T[s]
Stage 2:  with t >= t*, maximise sum over sides and strata of retained/n
          (spreads retention over the non-bottleneck strata).
Files are aggregated into cells (side, shared-node set, stratum), so the MILP has
one binary per shared node plus one continuous variable per multi-node cell.
Solver: scipy.optimize.milp (HiGHS, SciPy >= 1.9); no extra dependency.

Usage
-----
    python cp_purge.py --manifest dev.csv --out runs/cp --seeds 0-9 \
        --id sha256 --label label --arch arch --link link --group group --key program

The manifest needs one row per development file. label: 1 = malicious, 0 = benign.
link: linking type; values equal to --static-value (default "static") are static.
--key is optional (Corpus B: Buildroot program identity; empty = no key).

Outputs in --out:
    audit.csv                     one row per (held-out arch, protocol, side, stratum)
    summary.csv                   one row per (held-out arch, protocol)
    cp_info.json                  solver status, t*, runtime per held-out arch
    splits/<protocol>_<arch>_seed<s>.csv   columns: id, pool (C/R/T)

The C/R split uses StratifiedGroupKFold(5, shuffle, seed+1701) on strata (label, arch)
with similarity groups as groups, R = first fold. If your pipeline already has its own
StratGroupSplit, call `select_pools(...)` and feed the kept Q/T masks to it instead.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROTOCOLS = ("none", "purge", "cp")


# --------------------------------------------------------------------------- data
@dataclass
class Cols:
    id: str = "id"
    label: str = "label"
    arch: str = "arch"
    link: str = "link"
    group: str = "group"
    key: str | None = None
    static_value: str = "static"


def _clean_key(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None", "NaN") else s


def prepare(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    """Normalise the manifest into internal columns: _y, _a, _static, _g, _k, _stratum."""
    out = pd.DataFrame(index=df.index)
    out["_id"] = df[cols.id].astype(str)
    out["_y"] = df[cols.label].astype(int)
    if not set(out["_y"].unique()) <= {0, 1}:
        raise ValueError("label must be 0 (benign) / 1 (malicious)")
    out["_a"] = df[cols.arch].astype(str)
    out["_static"] = df[cols.link].astype(str).str.lower().eq(cols.static_value.lower())
    out["_g"] = df[cols.group].astype(str)
    if cols.key and cols.key in df.columns:
        out["_k"] = pd.Series([_clean_key(v) or "" for v in df[cols.key]], index=df.index, dtype=object)
    else:
        out["_k"] = ""
    out["_stratum"] = np.where(out["_y"] == 1, "mal", "ben") + "|" + np.where(out["_static"], "static", "dynamic")
    if out["_id"].duplicated().any():
        raise ValueError("duplicate file ids in manifest")
    return out.reset_index(drop=True)


def _nodes(row_g, row_k):
    n = ["g:" + row_g]
    if row_k:
        n.append("k:" + row_k)
    return n


# --------------------------------------------------------------------------- protocols
def standard_purge(P: pd.DataFrame, h: str):
    """Alg. 1 with purge: returns (keepQ, keepT) boolean arrays over P."""
    T = (P["_a"] == h).to_numpy()
    Q = ~T
    Gh = set(P.loc[T, "_g"])
    Kh = {k for k in P.loc[T, "_k"] if k}
    keepQ = Q & ~P["_g"].isin(Gh).to_numpy() & ~np.array([bool(k) and k in Kh for k in P["_k"]], dtype=bool)
    return keepQ, T.copy()


def no_purge(P: pd.DataFrame, h: str):
    T = (P["_a"] == h).to_numpy()
    return ~T, T.copy()


@dataclass
class _Problem:
    strata: list
    nQ: np.ndarray
    nT: np.ndarray
    freeQ: np.ndarray
    freeT: np.ndarray
    shared: list                      # shared node names
    cellsQ: list = field(default_factory=list)   # (node_idx tuple, counts vector)
    cellsT: list = field(default_factory=list)
    file_nodes: list = field(default_factory=list)  # per file: tuple of shared node idx


def _build(P: pd.DataFrame, h: str) -> _Problem:
    T = (P["_a"] == h).to_numpy()
    strata = sorted(P["_stratum"].unique())
    sidx = {s: i for i, s in enumerate(strata)}
    S = len(strata)
    file_all_nodes = [_nodes(g, k) for g, k in zip(P["_g"], P["_k"])]
    nodesT, nodesQ = set(), set()
    for is_t, ns in zip(T, file_all_nodes):
        (nodesT if is_t else nodesQ).update(ns)
    shared = sorted(nodesT & nodesQ)
    nidx = {n: i for i, n in enumerate(shared)}

    nQ, nT = np.zeros(S), np.zeros(S)
    freeQ, freeT = np.zeros(S), np.zeros(S)
    cQ, cT = defaultdict(lambda: np.zeros(S)), defaultdict(lambda: np.zeros(S))
    file_nodes = []
    for is_t, ns, st in zip(T, file_all_nodes, P["_stratum"]):
        sn = tuple(sorted(nidx[n] for n in ns if n in nidx))
        file_nodes.append(sn)
        j = sidx[st]
        if is_t:
            nT[j] += 1
            if sn:
                cT[sn][j] += 1
            else:
                freeT[j] += 1
        else:
            nQ[j] += 1
            if sn:
                cQ[sn][j] += 1
            else:
                freeQ[j] += 1
    return _Problem(strata, nQ, nT, freeQ, freeT, shared,
                    list(cQ.items()), list(cT.items()), file_nodes)


def _retained(prob: _Problem, x: np.ndarray):
    rQ, rT = prob.freeQ.copy(), prob.freeT.copy()
    for nodes, cnt in prob.cellsQ:
        if all(x[i] == 1 for i in nodes):
            rQ += cnt
    for nodes, cnt in prob.cellsT:
        if all(x[i] == 0 for i in nodes):
            rT += cnt
    return rQ, rT


def _ratios(prob, rQ, rT):
    r = [rQ[j] / prob.nQ[j] for j in range(len(prob.strata)) if prob.nQ[j] > 0]
    r += [rT[j] / prob.nT[j] for j in range(len(prob.strata)) if prob.nT[j] > 0]
    return np.array(r)


def _score(prob, x):
    """(min retention, sum of retentions) -- the two-stage objective."""
    r = _ratios(prob, *_retained(prob, x))
    return (float(r.min()) if len(r) else 1.0, float(r.sum()))


def _solve_milp(prob: _Problem, time_limit: float, verbose: bool):
    """Two-stage MILP with scipy.optimize.milp (HiGHS). Variables: [x (N bin), zQ, zT, t]."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    N, S = len(prob.shared), len(prob.strata)
    # single-node cells use x directly; multi-node cells get a continuous z <= x (or <= 1-x)
    mQ = [k for k, (n, _) in enumerate(prob.cellsQ) if len(n) > 1]
    mT = [k for k, (n, _) in enumerate(prob.cellsT) if len(n) > 1]
    zQ = {k: N + i for i, k in enumerate(mQ)}
    zT = {k: N + len(mQ) + i for i, k in enumerate(mT)}
    it = N + len(mQ) + len(mT)
    V = it + 1

    lb, ub = [], []
    n_link = sum(len(prob.cellsQ[k][0]) for k in mQ) + sum(len(prob.cellsT[k][0]) for k in mT)
    A = lil_matrix((max(n_link, 1), V))
    r = 0
    for k in mQ:                                   # zQ_k - x_i <= 0
        for i in prob.cellsQ[k][0]:
            A[r, zQ[k]] = 1; A[r, i] = -1; lb.append(-np.inf); ub.append(0); r += 1
    for k in mT:                                   # zT_k + x_i <= 1
        for i in prob.cellsT[k][0]:
            A[r, zT[k]] = 1; A[r, i] = 1; lb.append(-np.inf); ub.append(1); r += 1

    # retained_Q[j] = freeQ + sum c*x (single) + sum c*z ;  retained_T[j] = freeT + sum c*(1-x) + sum c*z
    retQ = np.zeros((S, V)); retQc = prob.freeQ.copy()
    retT = np.zeros((S, V)); retTc = prob.freeT.copy()
    for k, (nodes, c) in enumerate(prob.cellsQ):
        col = zQ[k] if len(nodes) > 1 else nodes[0]
        retQ[:, col] += c
    for k, (nodes, c) in enumerate(prob.cellsT):
        if len(nodes) > 1:
            retT[:, zT[k]] += c
        else:
            retT[:, nodes[0]] -= c; retTc += c
    side_rows = []                                 # (coef row, const, n) for each constrained stratum
    for j in range(S):
        if prob.nQ[j] > 0:
            side_rows.append((retQ[j], retQc[j], prob.nQ[j]))
        if prob.nT[j] > 0:
            side_rows.append((retT[j], retTc[j], prob.nT[j]))
    A = A.tocsr()[:r]
    A_ret = np.zeros((len(side_rows), V))
    for q, (coef, const, n) in enumerate(side_rows):   # coef.v - n*t >= -const
        A_ret[q] = coef; A_ret[q, it] = -n
    cons = [LinearConstraint(A_ret, [-c for _, c, _ in side_rows], np.inf)]
    if r:
        cons.append(LinearConstraint(A, lb, ub))
    integrality = np.zeros(V); integrality[:N] = 1
    bounds = Bounds(np.zeros(V), np.ones(V))
    opts = {"time_limit": time_limit, "disp": verbose}

    c1 = np.zeros(V); c1[it] = -1                     # maximise t
    res1 = milp(c1, constraints=cons, integrality=integrality, bounds=bounds, options=opts)
    if res1.x is None:
        return None, f"stage1:{res1.message}"
    t_star = float(res1.x[it])
    opt1 = res1.status == 0

    # stage 2: t >= t*, maximise sum_q retained_q / n_q
    lb2 = bounds.lb.copy(); lb2[it] = max(0.0, t_star - 1e-7)
    c2 = -sum(coef / n for coef, _, n in side_rows)
    res2 = milp(c2, constraints=cons, integrality=integrality, bounds=Bounds(lb2, bounds.ub), options=opts)
    xbest = res2.x if res2.x is not None else res1.x
    xs = np.round(xbest[:N]).astype(int)
    status = "optimal" if (opt1 and res2.status == 0) else "feasible(time limit)"
    return xs, status


def _solve_greedy(prob: _Problem, max_iter: int = 100000):
    """Fallback: flip one node at a time (0 -> 1) while the two-stage score improves."""
    N = len(prob.shared)
    x = np.zeros(N, dtype=int)
    best = _score(prob, x)
    # incidence for fast evaluation
    inc = defaultdict(list)
    for side, cells in (("Q", prob.cellsQ), ("T", prob.cellsT)):
        for k, (nodes, _) in enumerate(cells):
            for i in nodes:
                inc[i].append((side, k))
    rQ, rT = _retained(prob, x)
    for _ in range(max_iter):
        cand = None
        for i in range(N):
            if x[i] == 1:
                continue
            dQ, dT = np.zeros_like(rQ), np.zeros_like(rT)
            for side, k in inc[i]:
                nodes, cnt = (prob.cellsQ if side == "Q" else prob.cellsT)[k]
                if side == "Q":
                    if all(x[j] == 1 for j in nodes if j != i):
                        dQ += cnt                         # becomes kept
                else:
                    if all(x[j] == 0 for j in nodes):
                        dT -= cnt                         # was kept, now dropped
            r = _ratios(prob, rQ + dQ, rT + dT)
            sc = (float(r.min()), float(r.sum()))
            if sc > best and (cand is None or sc > cand[0]):
                cand = (sc, i, dQ, dT)
        if cand is None:
            break
        best, i, dQ, dT = cand
        x[i] = 1
        rQ, rT = rQ + dQ, rT + dT
    return x


def cp_purge(P: pd.DataFrame, h: str, time_limit: float = 120.0, solver: str = "milp",
             verbose: bool = False):
    """Alg. CP-Purge: returns (keepQ, keepT, info)."""
    t0 = time.time()
    prob = _build(P, h)
    x0 = np.zeros(len(prob.shared), dtype=int)
    status = "trivial"
    x = x0
    if prob.shared:
        xs = None
        if solver == "milp":
            xs, status = _solve_milp(prob, time_limit, verbose)
        if xs is None or status != "optimal":
            xg = _solve_greedy(prob)
            if xs is None or _score(prob, xg) > _score(prob, xs):
                xs, status = xg, f"greedy ({status})"
        # guarantee of Proposition 1: never worse than the standard purge
        x = xs if _score(prob, xs) >= _score(prob, x0) else x0
    keepT_all = (P["_a"] == h).to_numpy()
    keepQ = np.zeros(len(P), bool)
    keepT = np.zeros(len(P), bool)
    for f, (is_t, nodes) in enumerate(zip(keepT_all, prob.file_nodes)):
        if is_t:
            keepT[f] = all(x[i] == 0 for i in nodes)
        else:
            keepQ[f] = all(x[i] == 1 for i in nodes)
    info = {
        "held_out": h,
        "status": status,
        "n_shared_nodes": len(prob.shared),
        "n_nodes_to_train": int(x.sum()),
        "t_star": _score(prob, x)[0],
        "t_standard_purge": _score(prob, x0)[0],
        "runtime_s": round(time.time() - t0, 2),
    }
    return keepQ, keepT, info


def protocol_masks(P, h, protocol, **kw):
    if protocol == "none":
        return (*no_purge(P, h), {})
    if protocol == "purge":
        return (*standard_purge(P, h), {})
    if protocol == "cp":
        return cp_purge(P, h, **kw)
    raise ValueError(protocol)


# --------------------------------------------------------------------------- checks
def assert_leakage_free(P, keepQ, keepT):
    gQ, gT = set(P.loc[keepQ, "_g"]), set(P.loc[keepT, "_g"])
    kQ = {k for k in P.loc[keepQ, "_k"] if k}
    kT = {k for k in P.loc[keepT, "_k"] if k}
    if gQ & gT or kQ & kT:
        raise AssertionError(f"leakage: {len(gQ & gT)} groups, {len(kQ & kT)} program keys cross the split")


# --------------------------------------------------------------------------- pools
def select_pools(P, keepQ, keepT, seed, n_splits=5):
    """StratGroupSplit of the kept training side into C and R (R = one fold)."""
    from sklearn.model_selection import StratifiedGroupKFold

    qi = np.flatnonzero(keepQ)
    strat = P["_y"].astype(str).to_numpy()[qi] + "|" + P["_a"].to_numpy()[qi]
    groups = P["_g"].to_numpy()[qi]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1701)
    _, r_local = next(skf.split(qi, strat, groups))
    pool = np.array([""] * len(P), dtype=object)
    pool[qi] = "C"
    pool[qi[r_local]] = "R"
    pool[keepT] = "T"
    return pool


# --------------------------------------------------------------------------- audit (Alg. 2)
def _jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def audit(P, h, masks: dict, tau=0.2):
    """masks: protocol -> (keepQ, keepT). Returns (rows per stratum, summary rows)."""
    strata = sorted(P["_stratum"].unique())
    base_Q, base_T = masks["none"]
    rows, summ = [], []
    n0Q = P.loc[base_Q, "_stratum"].value_counts().reindex(strata, fill_value=0)
    n0T = P.loc[base_T, "_stratum"].value_counts().reindex(strata, fill_value=0)
    for proto, (kQ, kT) in masks.items():
        n1Q = P.loc[kQ, "_stratum"].value_counts().reindex(strata, fill_value=0)
        n1T = P.loc[kT, "_stratum"].value_counts().reindex(strata, fill_value=0)
        ratios = []
        for s in strata:
            for side, a, b in (("train", n0Q[s], n1Q[s]), ("test", n0T[s], n1T[s])):
                rho = b / a if a else float("nan")
                if a:
                    ratios.append(rho)
                rows.append(dict(held_out=h, protocol=proto, side=side, stratum=s,
                                 n_before=int(a), n_after=int(b), retention=rho))
        stat = P.loc[kQ, "_static"].to_numpy()
        pi = float(P.loc[kQ, "_y"].to_numpy()[stat].mean()) if stat.any() else float("nan")
        sb = int(((P["_y"] == 0) & P["_static"] & kQ).sum())
        summ.append(dict(
            held_out=h, protocol=proto,
            n_train_side=int(kQ.sum()), n_test=int(kT.sum()), test_frac_kept=kT.sum() / max(base_T.sum(), 1),
            static_benign_train=sb, min_retention=min(ratios) if ratios else float("nan"),
            jsd_train=_jsd(n0Q.values, n1Q.values) if n1Q.sum() else float("nan"),
            jsd_test=_jsd(n0T.values, n1T.values) if n1T.sum() else float("nan"),
            pi_static_malicious=pi, warn=bool(ratios and min(ratios) < tau)))
    return rows, summ


# --------------------------------------------------------------------------- CLI
def _parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run(df, cols: Cols, out_dir, seeds, archs=None, time_limit=120.0, solver="milp",
        tau=0.2, write_splits=True, verbose=False):
    import os
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)
    P = prepare(df, cols)
    archs = archs or sorted(P["_a"].unique())
    all_rows, all_summ, infos = [], [], []
    for h in archs:
        masks = {}
        for proto in PROTOCOLS:
            kQ, kT, info = protocol_masks(P, h, proto, **({"time_limit": time_limit, "solver": solver,
                                                            "verbose": verbose} if proto == "cp" else {}))
            if proto != "none":
                assert_leakage_free(P, kQ, kT)
            masks[proto] = (kQ, kT)
            if info:
                infos.append(info)
                print(f"[{h}] CP-Purge {info['status']}: t*={info['t_star']:.3f} "
                      f"(standard purge {info['t_standard_purge']:.3f}), "
                      f"{info['n_nodes_to_train']}/{info['n_shared_nodes']} shared nodes to train, "
                      f"{info['runtime_s']}s")
        rows, summ = audit(P, h, masks, tau)
        all_rows += rows
        all_summ += summ
        if write_splits:
            for proto, (kQ, kT) in masks.items():
                for s in seeds:
                    pool = select_pools(P, kQ, kT, s)
                    sel = pool != ""
                    pd.DataFrame({"id": P["_id"][sel], "pool": pool[sel]}).to_csv(
                        os.path.join(out_dir, "splits", f"{proto}_{h}_seed{s}.csv"), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "audit.csv"), index=False)
    summ = pd.DataFrame(all_summ)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "cp_info.json"), "w") as f:
        json.dump(infos, f, indent=2)
    return summ, infos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV or parquet, one row per development file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--archs", default=None, help="comma list of held-out architectures (default: all)")
    ap.add_argument("--id", default="id")
    ap.add_argument("--label", default="label")
    ap.add_argument("--arch", default="arch")
    ap.add_argument("--link", default="link")
    ap.add_argument("--static-value", default="static")
    ap.add_argument("--group", default="group")
    ap.add_argument("--key", default=None, help="program key column (optional)")
    ap.add_argument("--solver", choices=["milp", "greedy"], default="milp")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--no-splits", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.manifest) if a.manifest.endswith(".parquet") else pd.read_csv(a.manifest)
    cols = Cols(a.id, a.label, a.arch, a.link, a.group, a.key, a.static_value)
    summ, _ = run(df, cols, a.out, _parse_seeds(a.seeds), a.archs.split(",") if a.archs else None,
                  a.time_limit, a.solver, a.tau, not a.no_splits, a.verbose)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
