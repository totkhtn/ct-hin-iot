# Pre-registration v33 — CT-HIN on a temporally sealed holdout (English translation)

This is a faithful English translation of `PREREG_v33_ct_hin.md`. The Vietnamese original is the frozen record;
its SHA-256 is `11ccbc5c15ae87620652c3a412d144ec6fb04c7d37f2744be6ba5abe0a8814ff` (see `SHA256SUMS.txt`).
Internal identifiers are kept as in the original and explained in brackets.

## 1. History (disclosure)
- The method was formed in a pilot on the third validation set C3 (`v33_ct_pilot`) and in a development phase on
  set D: W2025 (MalwareBazaar 2025) and W26Q1, W26Q2 (MalwareBazaar 2026 Q1–Q2), plus development benign files.
  Development results: `ket_qua_phat_trien_D.txt` [released as `results/dev_D_frozen_run.txt`].
- The aggregate false positive rate of the static-small benign group (`lanh_tinh_tinh_nho`) was inspected in the
  pilot. The group was split in two by a hash (parity of the SHA-256 of the file hash); the odd half (5,220 files)
  is used only in the holdout H.
- The holdout H had not been scored or evaluated in any way before this file was frozen.

## 2. Holdout H (fixed)
- Malicious: MalwareBazaar samples with first_seen in 2026 Q3–Q4 that were NOT used in C3 (13,466 in the list;
  after filtering by architecture and duplicates).
- Benign: the held-out half of `lanh_tinh_tinh_nho` (5,220 files, statically linked) plus 8,000 previously unused
  files from the 2020 leftover benign pool (sampled with seed 1).
- Total after construction: 26,681 files (`dung_v33.py H`). Entities: `H/tokens_ct.jsonl.gz`.
  [After the final de-duplication the evaluated set has 26,673 files; see `data/holdout_H/manifest.csv`.]

## 3. Training and prior (fixed)
Corpus B 2020, LOAO without purge, Local LightGBM (pipeline v16, grid + inner cross-validation); ten seeds:
2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007.
The prior of each file is the Local score of the model that held out the file's architecture.

## 4. Method (configuration CHOSEN ON D, fixed)
CT-HIN: one transductive batch = all of H; entities = normalised strings (s) + raw IP + domain + URL; IDF within
the batch; drop entities with df < 2 or df > 2% of the batch; kNN cosine with k = 20; label spreading with
alpha = 0.9, 20 iterations; threshold 0.5.
(`ct_v33.py s-ip-dom-url,0.02,20,0.9`)

## 5. Baselines
(i) Local; (ii) kNN over Local features with the same spreading (alpha 0.8; not heterogeneous); (iii) LR-HIN (hin2).

## 6. Hypotheses (one-sided Wilcoxon, paired by seed, ten seeds)
- **H1 (primary family, Holm over 4 tests):** CT-HIN versus Local: larger M; larger W; larger RD (recall on
  dynamically linked malware); smaller FPR (benign). Confirmed if p_Holm < 0.05 and CT-HIN wins in at least 8/10
  seeds on M and on FPR.
- **H2:** M(CT-HIN) > M(kNN over Local features) — the specific value of the heterogeneous entities.
- **H3 (descriptive):** FPR on the held-out half of the static-small benign group; results per architecture; per
  malware family.

## 7. Freeze
- Freeze time: 2026-11-11 22:01 (UTC+7), confirmed by the author before H was scored.
- SHA-256 of `ct_v33.py`, `v33_eval_H.py` (test script), `dung_v33.py`, `H/tokens_ct.jsonl.gz` and
  `H/work/input/du_lieu_2024/bang_v5_dense.csv.gz` (recorded in `SHA_v33.txt`
  [released as `SHA_v33_original.txt`]).
