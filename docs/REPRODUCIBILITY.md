# Reproducibility notes

## What can be reproduced from this repository

| Result in the paper | Command | Needs retraining? |
|---|---|---|
| Table II (development windows) and Table III (configurations) | `python scripts/reproduce_development.py` | No (uses released priors) |
| Tables IV–VI (holdout, per architecture, pre-registered tests) | `python scripts/reproduce_holdout.py` | No |
| Table VII (threshold check) | `python scripts/reproduce_holdout.py` | No |
| Table VIII (changed decisions; indicator entities) | `python scripts/analyse_changes.py --seed 2` | No |
| Self-training baseline (Section VI-C) | `legacy/supplementary/selftrain_H.py` | Yes, needs the 2020 feature tables (not released) |
| Local and LR-HIN priors | `legacy/local_detector/kaggle_v31.py --corpus H` | Yes, needs the 2020 feature tables (not released) |

## Agreement with the frozen run

The clean package `ct_hin/` re-implements the frozen code in `prereg/frozen/ct_v33.py`. We checked it against the
frozen results on 12 November 2026 (NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0, Python 3.13):

* Local, LR-HIN: identical (largest absolute difference over all metrics and seeds < 1e-13).
* CT-HIN on H: every reported value is identical at the printed precision. The largest absolute difference over
  M, W, RD, FPR and AUC across seeds is 3e-6.
* kNN-feature on H: largest absolute difference 0.002 points. It comes from computing the feature standardisation
  in float64 instead of float32.
* Development windows: Local, kNN-feature and every CT-HIN configuration agree with
  `results/dev_D_frozen_run.txt` at the printed precision.

The remaining differences come from floating-point ties. The frozen code built the entity vocabulary by iterating
over Python sets, so the column order, and with it the order of floating-point summation, depended on the hash
seed of the interpreter. Two runs of the frozen code differed by four stored entries of the neighbour graph
(736,730 and 736,734). The clean package sorts entities and is deterministic. On H it builds 368,368 undirected
edges; the frozen run reported in the paper had 368,367. 177 files have no edge.

## Verifying the pre-registration

`prereg/SHA256SUMS.txt` lists the SHA-256 of the frozen scripts and of the original pre-registration.
`prereg/SHA_v33_original.txt` is the hash file written at freeze time (2026-11-11 22:01 UTC+7). It also lists the
holdout entity file and the holdout feature table in their original internal format. The released
`data/holdout_H/*` files were re-exported from those originals, with field names translated and unused fields
dropped. Their content is the same, but their hashes differ.

## Runtime

On two CPU cores, one holdout batch (26,673 files) takes about 4 s to build the entity matrix, 8 s for the
neighbour search and under 0.1 s for 20 spreading iterations. `scripts/reproduce_holdout.py` runs in about 30 s
and `scripts/reproduce_development.py` (18 configurations × 3 windows) in about 40 minutes; it can be split with `--configs`.
