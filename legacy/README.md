# Legacy research scripts (verbatim)

The scripts in this directory produced the released priors and the supplementary analyses. They are published
unchanged so that the pipeline behind every number can be inspected. Comments and identifiers are partly in
Vietnamese. **The supported, documented implementation of CT-HIN is the `ct_hin/` package**; use these scripts
only to retrain the local detector or to audit the original runs.

## `local_detector/`: Local and LR-HIN models (LOAO)

* `run_uploaded_experiments.py`, `v16_core.py`, `v16_run.py`, `v24_core.py`: data loading, the Local feature
  block (630 numerical fields + up to 100 opcode TF-IDF features fitted on the training fold), LightGBM training
  (300 trees, learning rate 0.05, balanced class weights, 31 or 63 leaves chosen by three-fold stratified group
  cross-validation inside the training fold) and LOAO partitioning.
* `v28_xshin.py`, `v29_xshin.py`, `v30_ablation.py`, `v31_run.py`, `v31_eval.py`, `cp_purge.py`, `v26_core.py`:
  the earlier inductive HIN variants (LR-HIN, XS-HIN) and the protocol-level method that preceded CT-HIN. They are
  needed because the Kaggle driver imports them.
* `kaggle_v31.py`: the driver that trained Local and LR-HIN for the holdout (`--corpus H`, ten seeds × five held-out
  architectures = 50 jobs). The command used on Kaggle was:

  ```bash
  pip install numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 lightgbm==4.7.0
  python kaggle_v31.py --corpus H --code <this directory> --data <training tables> \
         --data2 <holdout tables> --phan 1 --so-phan 2 --jobs 4     # and --phan 2
  ```
  (`--phan k --so-phan m`: run part *k* of *m*.) Each job writes `results/<tag>/units/<seed>_<arch>/{local,hin2}__lgbm.npz`.
  The released `priors_local.npz` and `scores_lrhin.npz` collect the `external|score` arrays of these files.
* `MA_BAM_v31.txt`: SHA-256 of the code files used in that run.

Retraining requires the 2020 feature tables (`du_lieu_2020`, `du_lieu_lt2020`), which are not redistributed (see
`data/README.md`).

## `feature_extraction/`

`build_feature_tables_v52.py` turns static-analysis records of extractor v5.2 (`*.json.gz`, one per ELF file) into
the dense feature table and sparse blocks used by the local detector.

## `supplementary/`: post hoc analyses (Section VI-C/D of the paper)

* `selftrain_H.py`, `eval_selftrain.py`: self-training with pseudo-labels.
* `kiemtra_nguong.py`: threshold check. Re-implemented in `scripts/reproduce_holdout.py`.
* `interpret_H.py`: changed decisions and shared entities. Re-implemented in `scripts/analyse_changes.py`.

Vietnamese file and variable names used in these scripts:

| Name | Meaning |
|---|---|
| `nguon` | source |
| `nam` | year or window |
| `lanh tinh` (LT) | benign |
| `tinh nho` | static-small |
| `du2020` | 2020 leftover |
| `ket qua` | results |
| `phan` | part |
| `so phan` | number of parts |
| `tai lap` | reproduction |
| `nhan gia` | pseudo-label |
| `keo len` / `day xuong` | raised / lowered |
