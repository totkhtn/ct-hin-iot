import os, json
os.environ.update(WINS='H26', TAG='v33_H_N', SEEDS='2 17 43 101 211 401 809 1601 3203 6007', TOKF='tokens_ct.jsonl.gz')
import numpy as np
from scipy import stats
import ct_v33 as CT
idx = np.flatnonzero(CT.win == 'H26')
Wct, hct = CT.knn(CT.ent(idx, ['s', 'ip', 'dom', 'url'], 0.02), 20)
R = {k: [] for k in ('Local_pred_goc', 'Local_diem>=0.5', 'Local_diem>=0', 'CT>=0.5', 'CT>=0')}
for s in CT.SEEDS:
    pg = np.zeros(CT.N, int)
    for a in CT.AR: pg[CT.ar == a] = np.load(f'results/{CT.TAG}/units/{s}_{a}/local__lgbm.npz')['local|lgbm|external|pred']
    pr = CT.PRIOR[s][idx]; f = CT.spread(Wct, hct, pr, 0.9)
    print(s, 'khoảng điểm', round(pr.min(), 2), round(pr.max(), 2), 'pred gốc khớp điểm>=0:', round(float((pg[idx] == (pr >= 0)).mean()), 4))
    for k, p, sc in (('Local_pred_goc', pg[idx], pr), ('Local_diem>=0.5', (pr >= 0.5).astype(int), pr), ('Local_diem>=0', (pr >= 0).astype(int), pr),
                     ('CT>=0.5', (f >= 0.5).astype(int), f), ('CT>=0', (f >= 0).astype(int), f)):
        R[k].append(CT.metr(idx, p, sc))
for k, v in R.items():
    mm = lambda q: np.mean([x[q] for x in v])
    print(f'{k:16s} M {mm("M"):.2f} W {mm("W"):.2f} RD {mm("RD"):.1f} REC {mm("REC"):.1f} FPR {mm("FPR"):.2f}')
for a_, b_ in (('CT>=0.5', 'Local_pred_goc'), ('CT>=0', 'Local_pred_goc')):
    d = [x['M'] - y['M'] for x, y in zip(R[a_], R[b_])]
    print(a_, '−', b_, f'ΔM {np.mean(d):+.2f} ({sum(x > 0 for x in d)}/10, p={stats.wilcoxon(d, alternative="greater", method="exact").pvalue:.4f})')
