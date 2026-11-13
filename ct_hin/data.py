"""Loading the released manifests, entity records, priors and local features."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(directory: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / "manifest.csv", dtype={"sha256": str, "family": str, "first_seen_utc": str},
                       keep_default_na=False)


def load_entities(paths, order) -> list[dict]:
    """Read entity records (one JSON object per file) and return them in the order of ``order`` (sha256 list)."""
    want = set(order)
    found: dict[str, dict] = {}
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        with gzip.open(p, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["sha256"] in want:
                    found[rec["sha256"]] = rec
    missing = want - found.keys()
    if missing:
        raise KeyError(f"{len(missing)} files have no entity record")
    return [found[h] for h in order]


def load_matrix(path: str | Path, key: str, order) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an (n x seeds) or (n x d) array from an .npz file and align its rows with ``order``."""
    z = np.load(path, allow_pickle=False)
    pos = {h: i for i, h in enumerate(z["sha256"].tolist())}
    rows = np.array([pos[h] for h in order])
    seeds = z["seeds"] if "seeds" in z.files else None
    return z[key][rows], seeds
