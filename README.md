# CT-HIN: Campaign-Transductive Heterogeneous Information Networks for Cross-Architecture IoT Malware Detection under Temporal Drift

This repository contains the code, data manifests, entity records, detector scores and pre-registration of the paper

> D.-T. Tran, A.-T. Tran, K. Nguyen-An and T.-D. Luong, "Campaign-Transductive Heterogeneous Information Networks for
> Cross-Architecture IoT Malware Detection under Temporal Drift," *Journal of Science and Technology on Information
> Security*, 2026 (under review).

With the released files, all CT-HIN results of the paper can be reproduced in under an hour on a laptop, without
access to malware binaries and without retraining the detector.

---

## 1. Problem

A static detector for IoT malware meets two shifts once it is deployed:

1. **Architecture shift.** IoT malware is compiled for many instruction-set architectures (ISAs). The detector
   will see binaries for ISAs that were rare or absent in its training data. We measure this with
   leave-one-architecture-out (LOAO) evaluation: the model that scores a file never saw that file's ISA during
   training.
2. **Temporal drift.** New campaigns differ from the training period.

The two shifts add up. A LightGBM detector on static features, trained on 2020 samples and evaluated LOAO on
MalwareBazaar samples, reaches a mean macro F1 of 88.2% on samples first seen in 2025 and 78.0% on those first
seen in 2026 Q2. Over the same period its recall on dynamically linked malware falls from 69.5% to 33.9%.
Retraining needs labels, and labels for a new campaign arrive late, especially for uncommon ISAs.

## 2. Idea

New campaigns arrive as **batches of related binaries**. An IoT botnet compiles one code base for many ISAs, so
files of one campaign share hard-coded command-and-control (C2) addresses, download URL templates, lock-file names
and process lists, whatever their instruction set. A per-file detector ignores these relations. They are
available without labels, inside the incoming batch.

An inductive alternative fails under drift. Our earlier HIN placed each test file in a network of *training*
files. On the 2025–2026 samples it lowered mean macro F1 from 88.0% to 80.8% and raised the false positive rate on
statically linked benign files from 8.8% to 42.1%. The 2020 references do not contain the entities of new
campaigns, and the entities they do share with new files are mostly library strings found in both classes.

**CT-HIN** therefore builds the network from the batch itself and uses the existing detector only as a prior.

<p align="center"><code>batch of files → file–entity HIN (4 entity types) → meta-path similarity → kNN file graph → label spreading of detector scores</code></p>

### 2.1 Batch heterogeneous network

For a batch *B* of *n* unlabelled files, the network has a *file* node type and four entity node types. A file is
linked to the entities that occur in it:

| Type | Entity | Example (from the holdout) |
|---|---|---|
| `s` | normalised string (IPv4 → `<IP>`, long hex → `<HEX>`, digits → `#`) | `/proc/%d/environ` |
| `ip` | IPv4 address, with port when present | `198.144.179.82`, `239.255.255.250:1900` |
| `dom` | domain-like token (also matches dotted file names) | `s.upd`, `ba4dff.lck` |
| `url` | URL without query string | `http://%s:%s/%s` |

Entities that occur in fewer than 2 files cannot link two files and are dropped. Entities that occur in more
than 2% of the batch are mostly library or runtime strings and are also dropped. Each remaining entity *e* gets
the in-batch IDF weight *w*<sub>e</sub> = log(*n*/*d*<sub>e</sub>), and each file becomes an L2-normalised row
*z*<sub>i</sub>.

### 2.2 Meta-path similarity

The cosine similarity of two files,

*S*<sub>ij</sub> = *z*<sub>i</sub>ᵀ*z*<sub>j</sub> ∝ Σ<sub>t ∈ {s, ip, dom, url}</sub> Σ<sub>e ∈ E<sub>t</sub></sub> *w*<sub>e</sub>² *X*<sub>ie</sub>*X*<sub>je</sub>,

is a normalised, IDF-weighted count of path instances over the four meta-paths *file–t–file*. A string shared by
five files therefore counts more than one shared by five hundred. A C2 address links two files of different ISAs
exactly as strongly as two files of the same ISA. Once the meta-paths are combined, propagation runs on a
homogeneous file graph: heterogeneity enters through the entity types and their weights, not through
type-specific parameters.

### 2.3 Spreading the detector's scores

Each file keeps its *k* = 20 most similar files. The graph is symmetrised by the maximum and row-normalised
(*P* = *D*<sup>−1</sup>*A*). The prior *p*<sub>i</sub> is the raw LightGBM margin of the LOAO model that did not
see the ISA of file *i*. Label spreading (Zhou et al., 2004) iterates

*f* ← (1 − α) *p* + α *P f*, with α = 0.9, for 20 iterations.

Files without edges keep their prior. A file is labelled malicious when *f*<sub>i</sub> ≥ τ, with τ = 0.5 on the
margin scale. **No label from the batch is used and the detector is not retrained.** The method can be added to
or removed from an existing deployment without touching the detector.

## 3. Main contributions

1. **Method.** CT-HIN is a transductive heterogeneous information network for static IoT malware detection. It is
   built per batch from normalised strings and pattern-matched indicators (IP addresses, domain-like tokens and
   URLs), and it uses the scores of an existing detector as its prior. Unlike Polonium and AESOP, it needs no
   client telemetry: the relations are read from the binaries. It processes a batch of 26,673 files in about 12 s
   on two CPU cores.
2. **Evaluation protocol.** We propose a temporal LOAO protocol on public MalwareBazaar samples: three development
   windows (2025, 2026 Q1, 2026 Q2) and a sealed holdout (2026 Q3–Q4). The configuration, the hypotheses, the
   baselines and SHA-256 hashes of code and data were frozen in a pre-registration
   ([`prereg/`](prereg/)) before the holdout was scored.
3. **Pre-registered evidence.** On 26,673 holdout files and ten seeds, CT-HIN improved four metrics over the
   detector it starts from, in every seed:
   * mean macro F1 on unseen ISAs (M)
   * worst-ISA macro F1 (W)
   * recall on dynamically linked malware (RD)
   * false positive rate (FPR)

   It also outperformed the same spreading over a graph of local-feature similarity in every seed. The gain
   therefore depends on the entity relations, not on smoothing alone.
4. **Error analysis.** Shared C2 addresses, download templates and persistence-related strings raised malicious
   files. Shared error messages of command-line utilities lowered benign files. The main failure mode: 253
   malicious files whose strings resemble benign utilities were wrongly lowered.

## 4. Results

### Sealed holdout H (ten seeds; percent except AUC)

| Model | M | W | RD | REC | FPR | FS | FP-SS | AUC |
|---|---|---|---|---|---|---|---|---|
| Local (LightGBM, LOAO) | 75.75 | 49.63 | 26.0 | 61.2 | 3.86 | 8.12 | 9.66 | .830 |
| LR-HIN (inductive, label-aware) | 57.78 | 35.16 | 11.9 | 28.3 | 4.28 | 8.01 | 9.53 | .715 |
| kNN-feature (same spreading) | 76.31 | 50.33 | 29.9 | 61.9 | 3.46 | 7.21 | 8.58 | .836 |
| **CT-HIN** | **79.67** | **55.22** | **37.6** | **64.7** | **0.24** | **0.29** | **0.34** | **.844** |

Metric definitions:

* **M / W**: mean / worst macro F1 over the five held-out ISAs
* **RD**: recall on dynamically linked malware
* **REC**: recall on all malware
* **FPR**: false positive rate on all benign files
* **FS**: false positive rate on statically linked benign files
* **FP-SS**: false positive rate on static-small benign files

Pre-registered tests (one-sided exact Wilcoxon over seeds; Holm correction within H1):

| Test | Δ | Seeds won | *p* | *p*<sub>Holm</sub> |
|---|---|---|---|---|
| H1: CT-HIN − Local, M | +3.92 | 10/10 | 0.001 | 0.0039 |
| H1: CT-HIN − Local, W | +5.59 | 10/10 | 0.001 | 0.0039 |
| H1: CT-HIN − Local, RD | +11.63 | 10/10 | 0.001 | 0.0039 |
| H1: CT-HIN − Local, FPR | −3.62 | 10/10 | 0.001 | 0.0039 |
| H2: CT-HIN − kNN-feature, M | +3.36 | 10/10 | 0.001 | — |

### Development windows (five seeds; mean macro F1 M)

| Model | W2025 | W26Q1 | W26Q2 | AUT |
|---|---|---|---|---|
| Local | 88.2 | 83.3 | 78.0 | 83.18 |
| kNN-feature | 89.2 | 84.1 | 78.6 | 83.98 |
| CT-HIN | 91.4 | 85.9 | 81.3 | 86.21 |

All 16 CT-HIN configurations tried on the development windows exceeded Local (AUT 84.04–86.22).
Normalised strings carry almost all of the gain. Strings alone reach AUT 86.22; with indicator entities added,
86.21; indicator entities alone, 84.04.

Post hoc analyses are in [`results/supplementary_H.md`](results/supplementary_H.md): self-training with
pseudo-labels, the decision threshold, and the changed decisions.

## 5. Repository layout

```
ct_hin/                 clean implementation (core.py: Algorithm 1; evaluation.py: metrics and tests; data.py: loaders)
scripts/
  extract_entities.py   entities from v5.2 static-analysis records (*.json.gz)
  run_ct_hin.py         apply CT-HIN to a new batch given detector scores
  reproduce_holdout.py  Tables IV–VII
  reproduce_development.py  Tables II–III
  analyse_changes.py    Table VIII
data/                   manifests, entity records, priors, features (see data/README.md)
prereg/                 pre-registration (Vietnamese original + English translation), frozen scripts, SHA-256 sums
results/                outputs of the frozen runs and the supplementary analyses
legacy/                 verbatim research scripts: local-detector training (Kaggle), feature tables, supplementary analyses
docs/REPRODUCIBILITY.md agreement between this package and the frozen run
```

## 6. Quick start

```bash
git clone https://github.com/totkhtn/ct-hin-iot.git && cd ct-hin-iot
git lfs pull                       # large data files are stored with Git LFS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd data && sha256sum -c SHA256SUMS.txt)

python scripts/reproduce_holdout.py          # about 30 s: Tables IV–VII and the pre-registered tests
python scripts/reproduce_development.py      # about 40 min: Tables II–III
python scripts/analyse_changes.py --seed 2   # Table VIII
```

To apply CT-HIN to your own batch:

```bash
python scripts/extract_entities.py /path/to/v52_json_records batch_entities.jsonl.gz --jobs 4
# priors.csv: columns sha256,prior  (score of your detector; set --tau on the same scale)
python scripts/run_ct_hin.py --entities batch_entities.jsonl.gz --priors priors.csv --out scores.csv
```

```python
from ct_hin import ct_hin
scores = ct_hin(records, prior)   # records: list of {"s": [...], "ip": [...], "dom": [...], "url": [...]}
labels = scores >= 0.5
```

## 7. Scope and limitations

* **Data.** The holdout malware is 81% Mirai, and MalwareBazaar reflects what its contributors submit.
  Benign files have no timestamps; they come from 2020 leftovers and from programs we built.
* **Batch assumption.** CT-HIN needs a batch large enough to contain several files per campaign. For a single
  file it returns the prior.
* **Indicator extraction.** Strings come from the string sample stored in each extraction record, not from the
  whole binary.
* **Adversaries.** An attacker could embed benign utility strings or vary C2 addresses and file names per
  sample. Neither string stuffing nor batch poisoning was evaluated.
* **Development history.** Other designs were tried before CT-HIN, and the static-small benign files were
  inspected in aggregate in a pilot. Both are disclosed in the paper and in the pre-registration.

## 8. Citation

See [`CITATION.cff`](CITATION.cff). Until the paper is published, please cite this repository.

## 9. Licence

Code: MIT ([`LICENSE`](LICENSE)). Data files in `data/` and `results/`: CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
MalwareBazaar content is provided by abuse.ch under its own terms.

## Acknowledgements

Academy of Cryptography Techniques; Ho Chi Minh City University of Industry and Trade; Hanoi University of Science
and Technology; Ho Chi Minh City University of Technology (HCMUT), VNU-HCM; Institute of Cryptography Science and
Technology. We thank abuse.ch for MalwareBazaar.
