"""Reproduce the development-window results (Tables II and III of the paper).

    python scripts/reproduce_development.py --data data/dev_D                 # all configurations of Table III
    python scripts/reproduce_development.py --data data/dev_D --configs s-ip-dom-url,0.02,20,0.9

A configuration is ``entity-types,max_df,k,alpha``; entity types are joined by '-' and drawn from
s (normalised strings), r (raw strings), f (function signatures), ip, dom, url.
Each window (W2025, W26Q1, W26Q2) is processed as one separate batch. AUT is the unweighted mean of M.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_hin import batch_metrics, knn_graph, label_spreading  # noqa: E402
from ct_hin.core import graph  # noqa: E402
from ct_hin.data import load_entities, load_manifest, load_matrix  # noqa: E402

WINDOWS = ("W2025", "W26Q1", "W26Q2")
TABLE_III = [
    "s,0.02,10,0.8", "r,0.02,10,0.8", "ip-dom-url,0.02,10,0.8", "s-ip-dom-url,0.02,10,0.8", "r-ip-dom-url,0.02,10,0.8",
    "s-f,0.02,10,0.8", "s-ip-dom-url-f,0.02,10,0.8", "s,0.02,10,0.5", "s,0.02,10,0.9", "s,0.005,10,0.8",
    "s,0.05,10,0.8", "s,0.02,5,0.8", "s,0.02,20,0.8", "s-ip-dom-url,0.02,10,0.9", "s,0.02,20,0.9",
    "s-ip-dom-url,0.02,20,0.9",  # frozen for the holdout
]
TAU = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev_D")
    ap.add_argument("--configs", nargs="*", default=TABLE_III)
    ap.add_argument("--out", default="results/dev_D_reproduced.json")
    a = ap.parse_args()
    d = Path(a.data)
    man = load_manifest(d)
    order = man.sha256.tolist()
    prior, seeds = load_matrix(d / "priors_local.npz", "margin", order)
    x, _ = load_matrix(d / "local_features.npz", "x", order)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)            # standardised over the whole development set
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    win = man.window.values
    idx = {w: np.flatnonzero(win == w) for w in WINDOWS}
    recs = {w: load_entities(d / f"entities_{w}.jsonl.gz", [order[i] for i in idx[w]]) for w in WINDOWS}

    def evaluate(label, make_graph, alpha):
        res = {}
        for w in WINDOWS:
            i = idx[w]
            G = make_graph(w, i)
            runs = []
            for j in range(len(seeds)):
                p = prior[i, j].astype(np.float64)
                f = label_spreading(*G, p, alpha) if G is not None else p
                runs.append(batch_metrics(man.label.values[i], man.arch.values[i], man.static.values[i],
                                          (f >= TAU).astype(int), f))
            res[w] = {q: float(np.mean([r[q] for r in runs])) for q in ("M", "W", "RD", "FS", "FPR", "REC")}
        res["AUT"] = float(np.mean([res[w]["M"] for w in WINDOWS]))
        print(f"{label:30s} " + " | ".join(
            f"{w}: M {res[w]['M']:.1f} W {res[w]['W']:.1f} RD {res[w]['RD']:.1f} FS {res[w]['FS']:.1f} FPR {res[w]['FPR']:.2f}"
            for w in WINDOWS) + f" | AUT {res['AUT']:.2f}", flush=True)
        return res

    out = {"Local": evaluate("Local", lambda w, i: None, 0.0),
           "kNN-feature": evaluate("kNN-feature (k=10, alpha=0.8)", lambda w, i: knn_graph(x[i], 10), 0.8)}
    for cfg in a.configs:
        kinds, max_df, k, alpha = cfg.split(",")
        out[cfg] = evaluate(f"CT-HIN {cfg}", lambda w, i, kinds=kinds.split("-"), m=float(max_df), k=int(k):
                            graph(recs[w], kinds, m, k), float(alpha))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seeds=[int(s) for s in seeds], results=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
