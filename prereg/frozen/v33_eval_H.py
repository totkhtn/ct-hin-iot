"""v33_eval_H.py — đánh giá đăng ký trước trên tập GIỮ KÍN H (PREREG_v33_ct_hin.md). Chạy trong thư mục dữ liệu H:
  TOKF=tokens_ct.jsonl.gz python3 v33_eval_H.py
Cần: results/v33_H_N/units/<seed>_<arch>/{local,hin2}__lgbm.npz (từ Kaggle).
"""
import os, sys, json
os.environ.setdefault('WINS', 'H26'); os.environ.setdefault('TAG', 'v33_H_N')
os.environ.setdefault('SEEDS', '2 17 43 101 211 401 809 1601 3203 6007')
import numpy as np
from scipy import stats
import ct_v33 as CT                      # mã đã đóng băng (cấu hình trong PREREG)

AR, S = CT.AR, CT.SEEDS
idx = np.flatnonzero(CT.win == 'H26')
# LR-HIN (hin2) dự đoán
HIN = {}
for s in S:
    p = np.zeros(CT.N)
    for a in AR: p[CT.ar == a] = np.load(f'results/{CT.TAG}/units/{s}_{a}/hin2__lgbm.npz')['hin2|lgbm|external|score']
    HIN[s] = p
Wct, hct = CT.knn(CT.ent(idx, ['s', 'ip', 'dom', 'url'], 0.02), 20)        # CẤU HÌNH ĐÓNG BĂNG
Wd, hd = CT.knn(CT.XD[idx], 10)
nho = CT.nguon[idx] == 'LT_tinh_nho'
R = {k: [] for k in ('Local', 'LR-HIN', 'kNN-dactrung', 'CT-HIN')}
for s in S:
    pr = CT.PRIOR[s][idx]
    for k, f in (('Local', pr), ('LR-HIN', HIN[s][idx]), ('kNN-dactrung', CT.spread(Wd, hd, pr, 0.8)),
                 ('CT-HIN', CT.spread(Wct, hct, pr, 0.9))):
        pred = (f >= 0.5).astype(int); m = CT.metr(idx, pred, f)
        m['FP_nho'] = 100 * float(pred[nho & (CT.y[idx] == 0)].mean())
        m['per'] = {a: float(100 * __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(
            CT.y[idx][CT.ar[idx] == a], pred[CT.ar[idx] == a], average='macro')) for a in AR}
        R[k].append(m)


def wil(d, alt):
    d = np.asarray(d, float)
    return 1.0 if not np.any(d != 0) else float(stats.wilcoxon(d, alternative=alt, method='exact').pvalue)


def holm(ps):
    k = len(ps); o = np.argsort(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    return adj.tolist()


L = ['v33 — tập GIỮ KÍN H (MalwareBazaar 2026 Q3–Q4 + lành tính giữ kín), 10 seed']
for k, v in R.items():
    mm = lambda q: np.mean([x[q] for x in v])
    L.append(f'{k:13s} M {mm("M"):5.2f}  W {mm("W"):5.2f}  RD {mm("RD"):5.1f}  REC {mm("REC"):5.1f}  FPR {mm("FPR"):5.2f}  '
             f'FS {mm("FS"):5.2f}  FP_tinh_nho {mm("FP_nho"):5.2f}  AUC {mm("AUC"):.4f} | '
             + ' '.join(f'{a[3:]}:{np.mean([x["per"][a] for x in v]):.1f}' for a in AR))
T = {}
for q, alt in (('M', 'greater'), ('W', 'greater'), ('RD', 'greater'), ('FPR', 'less')):
    d = [a[q] - b[q] for a, b in zip(R['CT-HIN'], R['Local'])]
    T[f'H1:{q}'] = dict(diff=float(np.mean(d)), wins=int(sum((x > 0) if alt == 'greater' else (x < 0) for x in d)), p=wil(d, alt))
adj = holm([T[k]['p'] for k in T])
for k, a in zip(list(T), adj): T[k]['p_holm'] = a
d = [a['M'] - b['M'] for a, b in zip(R['CT-HIN'], R['kNN-dactrung'])]
T['H2:M_vs_kNN'] = dict(diff=float(np.mean(d)), wins=int(sum(x > 0 for x in d)), p=wil(d, 'greater'))
L.append('=== Kiểm định (CT-HIN − đối chứng; thắng/10; p một phía; p Holm cho H1)')
for k, r in T.items():
    L.append(f'  {k:14s} {r["diff"]:+6.2f}  {r["wins"]}/10  p={r["p"]:.4f}' + (f'  p_holm={r["p_holm"]:.4f}' if 'p_holm' in r else ''))
ok = all(T[f'H1:{q}']['p_holm'] < 0.05 for q in ('M', 'W', 'RD', 'FPR')) and T['H1:M']['wins'] >= 8 and T['H1:FPR']['wins'] >= 8
L.append(f'KẾT LUẬN H1: {"XÁC NHẬN" if ok else "KHÔNG XÁC NHẬN"} | H2: {"đạt" if T["H2:M_vs_kNN"]["p"] < 0.05 else "không đạt"}')
os.makedirs('analysis', exist_ok=True)
json.dump(dict(arms=R, tests=T), open('analysis/v33_H_eval.json', 'w'), default=float, indent=1)
print('\n'.join(L))
