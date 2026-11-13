"""selftrain_H.py — PHÂN TÍCH BỔ SUNG (sau xác nhận, không thuộc PREREG): đối chứng tự huấn luyện bằng nhãn giả trên cùng lô H.
Với mỗi (seed, kiến trúc đích a): lấy đúng tập huấn luyện C và tham số đã chọn của Local (từ đơn vị Kaggle), thêm các tệp H của
kiến trúc a có điểm Local >= TAU (nhãn giả 1) hoặc <= 1-TAU (nhãn giả 0), huấn luyện lại LightGBM cùng tham số, dự đoán lại H_a.
Kiểm tra tái lập: không thêm nhãn giả phải cho lại đúng dự đoán Local.
  python3 selftrain_H.py 0.9 0.95
"""
import sys, ast, json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import run_uploaded_experiments as base
import v24_core, v16_core as C

TAUS = [float(t) for t in sys.argv[1:]] or [0.9]
P, T, A = C.load_all('work/input'); ti, _ = C.external_indices(P, T, A)
AR = ['EM_386', 'EM_ARM', 'EM_MIPS', 'EM_PPC', 'EM_X86_64']
SEEDS = [2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007]
out = {t: {} for t in TAUS}; chk = []
for s in SEEDS:
    for a in AR:
        z = np.load(f'results/v33_H_N/units/{s}_{a}/local__lgbm.npz')
        tr = z['train']; prm = ast.literal_eval(json.loads(str(z['rows']))[0]['params'])
        ix = ti[T.arch[ti] == a]; sc0 = z['local|lgbm|external|score']; pr0 = z['local|lgbm|external|pred']
        loc = base.Local().fit(P, tr)
        Xtr = loc.transform(P, tr); Xh = loc.transform(T, ix)
        XX, prep = C.scale_fit(Xtr, P.y[tr], prm['k'])
        clf = C.make('lgbm', prm, s).fit(XX, P.y[tr])
        chk.append(float((clf.predict(C.scale_apply(Xh, prep)) == pr0).mean()))
        for tau in TAUS:
            pos = sc0 >= tau; neg = sc0 <= 1 - tau; sel = pos | neg
            X2 = np.vstack([Xtr, Xh[sel]]); y2 = np.concatenate([P.y[tr], pos[sel].astype(int)])
            XX2, prep2 = C.scale_fit(X2, y2, prm['k'])
            c2 = C.make('lgbm', prm, s).fit(XX2, y2)
            xs = C.scale_apply(Xh, prep2)
            out[tau][f'{s}_{a}'] = dict(pred=c2.predict(xs).astype(int).tolist(), score=C.score_of(c2, xs).astype(float).tolist(),
                                        n_gia=int(sel.sum()), n_gia_sai=int((pos[sel].astype(int) != T.y[ix][sel]).sum()))
        print(s, a, 'tái lập', round(chk[-1], 4), {t: out[t][f'{s}_{a}']['n_gia'] for t in TAUS}, flush=True)
json.dump(dict(out=out, tai_lap=chk), open('analysis/selftrain_H.json', 'w'))
print('tái lập trung bình', np.mean(chk))
