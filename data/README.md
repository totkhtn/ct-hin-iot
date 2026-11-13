# Data

This directory holds everything needed to reproduce the CT-HIN results of the paper **without retraining the
local detector**: file manifests, entity records, the priors (scores of the local detector) and the local
features used by the kNN-feature baseline. No binaries are distributed.

| Directory | Files | Content |
|---|---|---|
| `holdout_H/` | 26,673 | Sealed holdout: MalwareBazaar samples first seen 1 July – 12 October 2026, held-out static-small benign files, unused 2020 benign files |
| `dev_D/` | 59,941 | Development windows W2025 (first seen 15 April – 31 December 2025), W26Q1 and W26Q2 |
| `train_2020/` | 20,761 of 24,400 | Training corpus of the local detector (hashes only; see below) |

File integrity: `sha256sum -c SHA256SUMS.txt`.

## Files

**`manifest.csv`**: one row per file.

| Column | Meaning |
|---|---|
| `sha256` | SHA-256 of the ELF file |
| `label` | 1 malicious, 0 benign |
| `arch` | ELF machine (`EM_386`, `EM_ARM`, `EM_MIPS`, `EM_PPC`, `EM_X86_64`) |
| `static` | 1 if statically linked |
| `source` | see "Sources" below |
| `family` | MalwareBazaar signature (`n/a` if none); `benign` for benign files |
| `window` | `W2025`, `W26Q1`, `W26Q2` or `H` |
| `first_seen_utc` | MalwareBazaar `first_seen_utc` (malicious files only) |

**`entities*.jsonl.gz`**: one JSON object per file, `{"sha256", "s", "r", "f", "ip", "dom", "url"}`, produced by
`scripts/extract_entities.py`. Strings (`s`, `r`) and function signatures (`f`) are 12-hex-digit BLAKE2b hashes.
IP addresses, domain-like tokens and URLs are in clear text. The development entities are split by window
to keep each file under 100 MB.

**`priors_local.npz`**: arrays `sha256`, `seeds` and `margin` (files × seeds, float32). `margin[i, j]` is the raw
LightGBM margin (log-odds) for file *i* from the seed-*j* model trained leave-one-architecture-out without the
architecture of file *i*. Development: seeds 17, 43, 101, 211, 401. Holdout: seeds 2, 17, 43, 101, 211, 401, 809,
1601, 3203, 6007.

**`scores_lrhin.npz`** (holdout only): `sha256`, `seeds` and `score`, the outputs of the LR-HIN baseline.

**`local_features.npz`**: `sha256` and `x` (files × 630, float32), the numerical static features of extractor v5.2,
used only by the kNN-feature baseline.

## Sources

| `source` | Description |
|---|---|
| `malwarebazaar` | ELF samples from MalwareBazaar (abuse.ch). Binaries can be downloaded from MalwareBazaar by hash under its terms of use. |
| `benign_static_small` | Small (< 200 KB) statically linked benign programs (sbase, ubase, toybox, dash, dropbear) built by the authors with five toolchains for the five architectures. The group was split by the parity of a hash of the file hash: one half went to development and the other half to the holdout. |
| `benign_2020_leftover` | Benign files from the 2020 collection that were not used in training |
| `benign_2024_unused` | Unused part of the 2024 benign collection of the authors' earlier study (development only) |

## Training corpus

`train_2020/manifest.csv` lists the 2020 training files with `sha256`, `label`, `arch`, `static` and `source`.
3,639 dynamically linked malicious files derived from IoTPOT Dataset C (Yokohama National University) are
omitted, because that dataset is distributed only with the provider's approval. The feature tables of the
training corpus are not redistributed. The released priors therefore replace retraining. The training pipeline
is in `legacy/local_detector/`.

## Ethics and terms

The release contains hashes, static features, hashed strings and network indicators extracted from malware.
It contains no executable code. MalwareBazaar content is provided by abuse.ch; please follow its terms when you
download samples. Indicators such as IP addresses and URLs may point to infrastructure that has since been
reassigned. Do not use them for blocking without independent verification.
