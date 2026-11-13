"""Metrics and paired tests used in the paper (Section V-B and V-C)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

ARCHS = ("EM_386", "EM_ARM", "EM_MIPS", "EM_PPC", "EM_X86_64")


def batch_metrics(y, arch, static, pred, score, extra_benign_mask=None) -> dict:
    """Metrics of one batch, in percent except AUC.

    M / W   mean / minimum macro F1 over the held-out architectures present in the batch
    RD      recall on dynamically linked malicious files
    REC     recall on all malicious files
    FPR     false positive rate on all benign files
    FS      false positive rate on statically linked benign files
    FP_SS   false positive rate on ``extra_benign_mask`` (static-small files in the paper)
    AUC     ROC AUC of the score over the whole batch
    per_arch macro F1 per architecture
    """
    y, arch, static, pred = map(np.asarray, (y, arch, static, pred))
    static = static.astype(bool)
    per = {a: 100 * f1_score(y[arch == a], pred[arch == a], average="macro")
           for a in ARCHS if (arch == a).any() and len(set(y[arch == a])) == 2}
    rate = lambda m: 100 * float(pred[m].mean()) if m.any() else float("nan")
    out = dict(M=float(np.mean(list(per.values()))), W=float(np.min(list(per.values()))),
               RD=rate((y == 1) & ~static), REC=rate(y == 1), FPR=rate(y == 0), FS=rate((y == 0) & static),
               AUC=float(roc_auc_score(y, score)), per_arch=per)
    if extra_benign_mask is not None:
        out["FP_SS"] = rate(np.asarray(extra_benign_mask, bool) & (y == 0))
    return out


def wilcoxon_one_sided(diffs, alternative: str) -> float:
    """Exact one-sided Wilcoxon signed-rank test over paired seeds (p = 1 if all differences are zero)."""
    d = np.asarray(diffs, float)
    if not np.any(d != 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative=alternative, method="exact").pvalue)


def holm(pvalues) -> list[float]:
    """Holm step-down adjustment."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
