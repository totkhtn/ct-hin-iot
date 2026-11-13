"""List the files whose label CT-HIN changes relative to the prior, and the entities they share (Table VIII).

    python scripts/analyse_changes.py --data data/holdout_H --seed 2

Only indicator entities (ip, dom, url) are shown in clear text; strings are released as hashes.
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_hin import incidence, label_spreading  # noqa: E402
from ct_hin.core import graph  # noqa: E402
from ct_hin.data import load_entities, load_manifest, load_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/holdout_H")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    d = Path(a.data)
    man = load_manifest(d)
    order = man.sha256.tolist()
    recs = load_entities(d / "entities.jsonl.gz", order)
    prior, seeds = load_matrix(d / "priors_local.npz", "margin", order)
    p = prior[:, list(seeds).index(a.seed)].astype(np.float64)
    f = label_spreading(*graph(recs), p, 0.9)
    y = man.label.values
    up, down = (p < a.tau) & (f >= a.tau), (p >= a.tau) & (f < a.tau)
    print(f"raised to malicious: {up.sum()} files, {(up & (y == 1)).sum()} malicious;",
          collections.Counter(man.family.values[up & (y == 1)]).most_common(5))
    print(f"lowered to benign:   {down.sum()} files, {(down & (y == 0)).sum()} benign;",
          collections.Counter(man.source.values[down & (y == 0)]).most_common(3))

    X = incidence(recs, ["ip", "dom", "url"]).tocsc()
    # rebuild entity names in the same (sorted) order as ct_hin.core.incidence
    names, seen = [], set()
    for r in recs:
        for k in ["ip", "dom", "url"]:
            for t in sorted(set(r.get(k, ()))):
                if (k, t) not in seen:
                    seen.add((k, t)); names.append(f"{k}:{t}")
    df = np.asarray(X.sum(0)).ravel()
    keep = (df >= 2) & (df <= 0.02 * len(order))
    for label, mask in (("raised (malicious)", up & (y == 1)), ("lowered (benign)", down & (y == 0))):
        cnt = np.asarray(X[np.flatnonzero(mask)].sum(0)).ravel() * keep
        print(f"\nIndicator entities most shared among files {label}:")
        for j in np.argsort(-cnt)[:a.top]:
            if cnt[j] == 0:
                break
            rows = X[:, j].nonzero()[0]
            print(f"  {names[j]:40s} changed {int(cnt[j]):4d} / in batch {int(df[j]):4d}; malicious share {y[rows].mean():.3f}")


if __name__ == "__main__":
    main()
