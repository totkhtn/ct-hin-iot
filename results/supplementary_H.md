# Supplementary analyses on the holdout H (post hoc, not pre-registered)

These analyses were designed after the pre-registered hypotheses H1 and H2 had been evaluated.
Ten seeds (2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007). Values are means over seeds, in percent except AUC.
Scripts: `legacy/supplementary/` (verbatim research scripts) and `scripts/reproduce_holdout.py` (threshold check).

## 1. Self-training with pseudo-labels

For each seed and target architecture, the H files of that architecture whose Local score was at least
tau_s (pseudo-label malicious) or at most 1 - tau_s (pseudo-label benign) were added to the training files.
LightGBM was then retrained with the parameters already selected and predicted the same files again.
Retraining without pseudo-labels reproduced 99.95% of the original Local predictions.

The thresholds were applied to the LightGBM margin. On that scale the rule admitted 98.8% of the batch,
and 21.2% of the pseudo-labels were wrong. This is a weak self-training baseline. Its training data
(the 2020 feature tables) are not redistributed, so this analysis cannot be rerun from the released files.

| Model | M | W | RD | REC | FPR | FS | FP-SS | AUC |
|---|---|---|---|---|---|---|---|---|
| Self-training, tau_s = 0.90 | 75.80 | 49.58 | 25.8 | 61.2 | 3.80 | 7.99 | 9.51 | 0.847 |
| Self-training, tau_s = 0.95 | 75.81 | 49.58 | 25.9 | 61.3 | 3.80 | 7.99 | 9.52 | 0.849 |

| Comparison | Mean difference in M | Seeds won | One-sided exact Wilcoxon p |
|---|---|---|---|
| Self-training (0.90) − Local | +0.05 | 7/10 | 0.097 |
| Self-training (0.95) − Local | +0.06 | 8/10 | 0.065 |
| CT-HIN − Self-training (either) | +3.86 | 10/10 | 0.001 |

## 2. Decision threshold

The prior is the LightGBM margin (log-odds), so the frozen threshold tau = 0.5 is stricter than the native
threshold 0.

| Model, tau | M | W | RD | REC | FPR |
|---|---|---|---|---|---|
| Local, 0 (native) | 75.93 | 50.05 | 27.8 | 62.0 | 4.48 |
| Local, 0.5 | 75.75 | 49.63 | 26.0 | 61.2 | 3.86 |
| CT-HIN, 0 | 79.95 | 55.70 | 38.1 | 65.4 | 0.45 |
| CT-HIN, 0.5 (frozen) | 79.67 | 55.22 | 37.6 | 64.7 | 0.24 |

CT-HIN at tau = 0.5 minus Local at its native threshold: +3.74 points of M, 10/10 seeds, p = 0.001.
CT-HIN at tau = 0 minus Local at its native threshold: +4.03, 10/10, p = 0.001.

## 3. Decisions changed by CT-HIN (seed 2)

Relative to Local at tau = 0.5, CT-HIN changed the label of 1,545 of 26,673 files.

* **Raised to malicious:** 781 files, of which 767 are malicious (Mirai 620, unlabelled 115, CoinMiner 18).
  The indicator entities these files most often share occur only in malicious files of the batch:
  * SSDP multicast address `239.255.255.250:1900` (497 files in the batch)
  * C2 address `198.144.179.82`, together with the URL template `http://%s:%s/%s` and the file names
    `ba4dff.lck`, `s.upd`, `x.upd` and `s.mk` (133–174 files)

  The C2 address occurs in files of all five architectures (ARM 70, MIPS 35, 386 33, PPC 20, x64 16).
  The shared strings include `auditd`, `xinetd`, `/usr/sbin/sshd`, `/proc/%d/environ` and `/etc/crontab`.
* **Lowered to benign:** 764 files, of which 511 are benign (505 static-small files). These files are linked
  by error messages of command-line utilities (`%s: not a base %d integer`, `%s: not an integer`,
  `%s: read error:`, `malloc:`), none of which occurs in a malicious file of the batch.
  The other 253 lowered files are malicious; these are the main error mode of the method.

`python scripts/analyse_changes.py --data data/holdout_H --seed 2` reproduces the counts and the indicator
entities. Strings are released only as hashes, so the clear-text strings above are not recoverable from the
release.
