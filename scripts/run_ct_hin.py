"""Apply CT-HIN to a new batch of files.

    python scripts/run_ct_hin.py --entities batch_entities.jsonl.gz --priors priors.csv --out scores.csv

``--entities``  output of ``extract_entities.py`` for every file of the batch
``--priors``    CSV with columns ``sha256,prior``: the score of an existing detector for each file. In the paper
                the prior is the LightGBM margin of the model that did not see the file's architecture; any
                real-valued score works, but ``--tau`` must be on the same scale.
Output: CSV with ``sha256,prior,score,label`` where label = [score >= tau].
Defaults are the pre-registered configuration (s, ip, dom, url; max_df 0.02; k 20; alpha 0.9; tau 0.5).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct_hin import ENTITY_TYPES, label_spreading  # noqa: E402
from ct_hin.core import graph  # noqa: E402
from ct_hin.data import load_entities  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="-".join(ENTITY_TYPES))
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()

    pri = pd.read_csv(a.priors, dtype={"sha256": str})
    order = pri.sha256.str.lower().tolist()
    t0 = time.time()
    P, has_edge = graph(load_entities(a.entities, order), a.types.split("-"), a.max_df, a.k)
    t1 = time.time()
    f = label_spreading(P, has_edge, pri.prior.to_numpy(np.float64), a.alpha, a.iterations)
    out = pd.DataFrame(dict(sha256=order, prior=pri.prior, score=f, label=(f >= a.tau).astype(int)))
    out.to_csv(a.out, index=False)
    changed = int((out.label != (out.prior >= a.tau)).sum())
    print(f"{len(order)} files; graph {t1 - t0:.1f}s, {P.nnz // 2} undirected edges, {int((~has_edge).sum())} isolated; "
          f"spreading {time.time() - t1:.2f}s; {changed} labels changed", file=sys.stderr)


if __name__ == "__main__":
    main()
