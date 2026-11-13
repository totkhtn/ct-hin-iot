"""interpret_H.py — PHÂN TÍCH BỔ SUNG (diễn giải, sau xác nhận): CT-HIN kéo/đẩy những tệp nào và nhờ thực thể nào.
Seed 2, cấu hình đóng băng. Xuất analysis/interpret_H.json.
"""
import os, json, collections
os.environ.update(WINS='H26', TAG='v33_H_N', SEEDS='2', TOKF='tokens_ct.jsonl.gz')
import numpy as np, scipy.sparse as sp
import ct_v33 as CT

idx = np.flatnonzero(CT.win == 'H26'); n = len(idx)
KINDS = ['s', 'ip', 'dom', 'url']
# ma trận thực thể (giữ cả tên thực thể để diễn giải) — cùng quy tắc với CT.ent
voc = {}; R, Cc = [], []
for ii, i in enumerate(idx):
    r = CT.TOK[CT.H[i]]
    for k in KINDS:
        for t in set(r.get(k, [])):
            j = voc.setdefault(k + ':' + t, len(voc)); R.append(ii); Cc.append(j)
X = sp.csr_matrix((np.ones(len(R), np.float32), (R, Cc)), shape=(n, len(voc)))
df = np.asarray(X.sum(0)).ravel(); keep = (df >= 2) & (df <= 0.02 * n); names = np.array(sorted(voc, key=voc.get))
W, has = CT.knn(CT.ent(idx, KINDS, 0.02), 20)
pr = CT.PRIOR[2][idx]; f = CT.spread(W, has, pr, 0.9)
y = CT.y[idx]; fam = np.array([CT.meta[i]['family'] for i in idx]); src = CT.nguon[idx]
up = (pr < 0.5) & (f >= 0.5); down = (pr >= 0.5) & (f < 0.5)
res = dict(n=int(n), keo_len=dict(tong=int(up.sum()), dung=int((up & (y == 1)).sum())),
           day_xuong=dict(tong=int(down.sum()), dung=int((down & (y == 0)).sum())),
           keo_len_theo_ho=collections.Counter(fam[up & (y == 1)]).most_common(10),
           day_xuong_theo_nguon=collections.Counter(src[down & (y == 0)]).most_common(10))
Xk = X[:, keep].tocsc(); nk = names[keep]; dfk = df[keep]


def top_ent(mask, kinds, m=12):
    cnt = np.asarray(Xk[np.flatnonzero(mask)].sum(0)).ravel()
    sel = [j for j in np.argsort(-cnt) if nk[j].split(':', 1)[0] in kinds][:m]
    out = []
    for j in sel:
        rows = Xk[:, j].nonzero()[0]
        out.append(dict(the=nk[j], so_tep_bi_doi=int(cnt[j]), so_tep_trong_lo=int(dfk[j]), ty_le_ma_doc=float(y[rows].mean())))
    return out


res['ioc_chung_keo_len'] = top_ent(up & (y == 1), ('ip', 'dom', 'url'))
res['chuoi_chung_keo_len'] = top_ent(up & (y == 1), ('s',), 8)
res['chuoi_chung_day_xuong'] = top_ent(down & (y == 0), ('s',), 8)
res['ioc_chung_day_xuong'] = top_ent(down & (y == 0), ('ip', 'dom', 'url'), 8)
# ví dụ cụm: tệp mã độc được kéo lên có láng giềng nào
ex = []
for i in np.flatnonzero(up & (y == 1))[:5]:
    nb = W[i].indices[np.argsort(-W[i].data)][:5]
    ex.append(dict(tep=CT.H[idx[i]], ho=fam[i], prior=float(pr[i]), sau=float(f[i]),
                   lang_gieng=[dict(tep=CT.H[idx[j]], nhan=int(y[j]), prior=float(pr[j]), ho=fam[j]) for j in nb]))
res['vi_du'] = ex
os.makedirs('analysis', exist_ok=True)
json.dump(res, open('analysis/interpret_H.json', 'w'), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'vi_du'}, indent=1, ensure_ascii=False, default=str))
