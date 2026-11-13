import os, json
os.environ.update(WINS='H26', TAG='v33_H_N', SEEDS='2 17 43 101 211 401 809 1601 3203 6007', TOKF='tokens_ct.jsonl.gz')
import numpy as np
from scipy import stats
import ct_v33 as CT
ST = json.load(open('analysis/selftrain_H.json'))['out']
EV = json.load(open('analysis/v33_H_eval.json'))['arms']
idx = np.flatnonzero(CT.win == 'H26')
nho = CT.nguon[idx] == 'LT_tinh_nho'
for tau, d in ST.items():
    R = []
    for s in CT.SEEDS:
        pred = np.zeros(CT.N, int); sc = np.zeros(CT.N)
        for a in CT.AR:
            u = d[f'{s}_{a}']; pred[CT.ar == a] = u['pred']; sc[CT.ar == a] = u['score']
        m = CT.metr(idx, pred[idx], sc[idx]); m['FP_nho'] = 100 * float(pred[idx][nho & (CT.y[idx] == 0)].mean()); R.append(m)
    mm = lambda q: np.mean([x[q] for x in R])
    print(f'Tự huấn luyện tau={tau}: M {mm("M"):.2f} W {mm("W"):.2f} RD {mm("RD"):.1f} REC {mm("REC"):.1f} FPR {mm("FPR"):.2f} FS {mm("FS"):.2f} FP_nho {mm("FP_nho"):.2f} AUC {mm("AUC"):.4f}')
    for ref in ('Local', 'CT-HIN'):
        dd = [EV['CT-HIN'][i]['M'] - R[i]['M'] for i in range(10)] if ref == 'CT-HIN' else [R[i]['M'] - EV['Local'][i]['M'] for i in range(10)]
        lab = 'CT-HIN − tự huấn luyện' if ref == 'CT-HIN' else 'tự huấn luyện − Local'
        print(f'   {lab}: ΔM {np.mean(dd):+.2f} ({sum(x > 0 for x in dd)}/10, p một phía={stats.wilcoxon(dd, alternative="greater", method="exact").pvalue:.4f})')
