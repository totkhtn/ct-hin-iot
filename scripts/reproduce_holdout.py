"""Reproduce the pre-registered evaluation on the sealed holdout H (Tables IV-VII of the paper).

    python scripts/reproduce_holdout.py --data data/holdout_H --out results/holdout_H_reproduced.json

Inputs are the released manifest, entity records, Local priors (LightGBM margins of the
leave-one-architecture-out models, ten seeds), LR-HIN scores and local features. Nothing is retrained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_hin import ARCHS, batch_metrics, holm, knn_graph, label_spreading, wilcoxon_one_sided  # noqa: E402
from ct_hin.core import graph  # noqa: E402
from ct_hin.data import load_entities, load_manifest, load_matrix  # noqa: E402

TAU = 0.5  # frozen threshold, applied on the margin scale


def standardised(x):
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/holdout_H")
    ap.add_argument("--out", default="results/holdout_H_reproduced.json")
    a = ap.parse_args()
    d = Path(a.data)
    man = load_manifest(d)
    order = man.sha256.tolist()
    y, arch, static = man.label.values, man.arch.values, man.static.values
    small = (man.source == "benign_static_small").values
    prior, seeds = load_matrix(d / "priors_local.npz", "margin", order)
    lrhin, _ = load_matrix(d / "scores_lrhin.npz", "score", order)
    feats, _ = load_matrix(d / "local_features.npz", "x", order)

    P_ct, h_ct = graph(load_entities(d / "entities.jsonl.gz", order))          # frozen: s, ip, dom, url; 0.02; k=20
    P_kn, h_kn = knn_graph(standardised(feats.astype(np.float64)), 10)        # kNN-feature baseline, k=10

    arms = {k: [] for k in ("Local", "LR-HIN", "kNN-feature", "CT-HIN")}
    for j, s in enumerate(seeds):
        p = prior[:, j].astype(np.float64)
        for name, f in (("Local", p), ("LR-HIN", lrhin[:, j]), ("kNN-feature", label_spreading(P_kn, h_kn, p, 0.8)),
                        ("CT-HIN", label_spreading(P_ct, h_ct, p, 0.9))):
            arms[name].append(batch_metrics(y, arch, static, (f >= TAU).astype(int), f, small))

    print(f"{'Model':12s}   M      W      RD    REC   FPR    FS    FP-SS  AUC   | " + " ".join(a_[3:] for a_ in ARCHS))
    for name, runs in arms.items():
        m = lambda q: np.mean([r[q] for r in runs])
        per = " ".join(f"{np.mean([r['per_arch'][x] for r in runs]):.1f}" for x in ARCHS)
        print(f"{name:12s} {m('M'):6.2f} {m('W'):6.2f} {m('RD'):5.1f} {m('REC'):5.1f} {m('FPR'):5.2f} {m('FS'):5.2f} "
              f"{m('FP_SS'):6.2f} {m('AUC'):.3f} | {per}")

    tests = {}
    for q, alt in (("M", "greater"), ("W", "greater"), ("RD", "greater"), ("FPR", "less")):
        diff = [c[q] - b[q] for c, b in zip(arms["CT-HIN"], arms["Local"])]
        wins = sum((x > 0) if alt == "greater" else (x < 0) for x in diff)
        tests[f"H1:{q}"] = dict(diff=float(np.mean(diff)), wins=int(wins), p=wilcoxon_one_sided(diff, alt))
    for key, padj in zip([k for k in tests], holm([t["p"] for t in tests.values()])):
        tests[key]["p_holm"] = padj
    diff = [c["M"] - b["M"] for c, b in zip(arms["CT-HIN"], arms["kNN-feature"])]
    tests["H2:M_vs_kNN-feature"] = dict(diff=float(np.mean(diff)), wins=int(sum(x > 0 for x in diff)),
                                        p=wilcoxon_one_sided(diff, "greater"))
    print("\nPre-registered tests (CT-HIN minus baseline, one-sided exact Wilcoxon over seeds)")
    for k, t in tests.items():
        print(f"  {k:22s} {t['diff']:+7.2f}  {t['wins']}/{len(seeds)}  p={t['p']:.4f}"
              + (f"  p_Holm={t['p_holm']:.4f}" if "p_holm" in t else ""))
    h1 = all(tests[f"H1:{q}"]["p_holm"] < 0.05 for q in ("M", "W", "RD", "FPR")) \
        and tests["H1:M"]["wins"] >= 8 and tests["H1:FPR"]["wins"] >= 8
    print(f"H1 {'confirmed' if h1 else 'not confirmed'}; H2 {'met' if tests['H2:M_vs_kNN-feature']['p'] < 0.05 else 'not met'}")

    # supplementary threshold check (Table VII): native LightGBM threshold 0 for both models
    thr = {}
    for name, fn in (("Local", lambda p: p), ("CT-HIN", lambda p: label_spreading(P_ct, h_ct, p, 0.9))):
        for tau in (0.0, 0.5):
            runs = [batch_metrics(y, arch, static, (fn(prior[:, j].astype(np.float64)) >= tau).astype(int),
                                  fn(prior[:, j].astype(np.float64))) for j in range(len(seeds))]
            thr[f"{name}@{tau}"] = {q: float(np.mean([r[q] for r in runs])) for q in ("M", "W", "RD", "REC", "FPR")}
    print("\nThreshold check (margin scale):")
    for k, v in thr.items():
        print(f"  {k:12s} " + "  ".join(f"{q} {x:.2f}" for q, x in v.items()))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(seeds=[int(s) for s in seeds], arms=arms, tests=tests, threshold=thr,
                   graph=dict(nodes=len(order), undirected_edges=int(P_ct.nnz // 2), isolated=int((~h_ct).sum()))),
              open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
