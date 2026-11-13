"""ct_v33.py — CT-HIN theo cửa sổ thời gian (giai đoạn PHÁT TRIỂN, tập D).
Mỗi cửa sổ (W2025, W26Q1, W26Q2) là một lô chuyển nạp riêng. Tiên nghiệm = điểm Local LOAO (mô hình giữ lại kiến trúc của tệp).
Đồ thị tệp–thực thể trong lô: thực thể chọn bằng KINDS (s: chuỗi chuẩn hóa, r: chuỗi thô, ip, dom, url, f: chữ ký hàm),
IDF trong lô, bỏ thực thể df<2 hoặc df>MAXDF*N; kNN cosine (k); lan truyền f <- (1-a)*prior + a*W f.
  python3 ct_v33.py <cấu hình...>     ví dụ: s,0.02,10,0.8  sr-ip-dom,0.02,10,0.8
Đối chứng: Local; kNN đặc trưng Local (cùng lan truyền).
"""
import sys, json, gzip, os
import numpy as np, scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
import v24_core, v16_core as C

P, T, A = C.load_all('work/input'); ti, _ = C.external_indices(P, T, A)
AR = ['EM_386', 'EM_ARM', 'EM_MIPS', 'EM_PPC', 'EM_X86_64']
SEEDS = [int(x) for x in os.environ.get('SEEDS', '17 43 101 211 401').split()]
TAG = os.environ.get('TAG', 'v33_D_N'); TOKF = os.environ.get('TOKF', 'tokens_ct.jsonl.gz')
WINS = os.environ.get('WINS', 'W2025 W26Q1 W26Q2').split()
meta = [T.meta[i] for i in ti]
win = np.array([m['nam'] for m in meta]); nguon = np.array([m['nguon'] for m in meta])
y = T.y[ti]; st = T.is_static[ti]; ar = T.arch[ti]; H = [str(h) for h in T.hash[ti]]; N = len(ti)
TOK = {}
for l in gzip.open(TOKF, 'rt'):
    r = json.loads(l); TOK[r['h']] = r
XD = T.x[ti]; XD = (XD - XD.mean(0)) / (XD.std(0) + 1e-9); XD = (XD / np.linalg.norm(XD, axis=1, keepdims=True)).astype(np.float32)


def ent(idx, kinds, maxdf):
    voc = {}; R, Cc = [], []
    for ii, i in enumerate(idx):
        r = TOK[H[i]]
        for k in kinds:
            for t in set(r.get(k, [])):
                j = voc.setdefault(k + ':' + t, len(voc)); R.append(ii); Cc.append(j)
    n = len(idx); X = sp.csr_matrix((np.ones(len(R), np.float32), (R, Cc)), shape=(n, len(voc)))
    df = np.asarray(X.sum(0)).ravel(); keep = (df >= 2) & (df <= maxdf * n)
    X = X @ sp.diags(np.where(keep, np.log(n / np.maximum(df, 1)), 0).astype(np.float32))
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    return sp.diags(1 / np.maximum(nr, 1e-12)) @ X


def knn(X, k, chunk=2000):
    n = X.shape[0]; XT = X.T.tocsc() if sp.issparse(X) else X.T; rows, cols, vals = [], [], []
    for i in range(0, n, chunk):
        S = X[i:i + chunk] @ XT; S = S.toarray() if sp.issparse(S) else np.asarray(S)
        for r in range(S.shape[0]): S[r, i + r] = -1
        idx = np.argpartition(-S, k, axis=1)[:, :k]; v = np.clip(np.take_along_axis(S, idx, 1), 0, None)
        rows.append(np.repeat(np.arange(i, i + S.shape[0]), k)); cols.append(idx.ravel()); vals.append(v.ravel())
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)); W = W.maximum(W.T)
    d = np.asarray(W.sum(1)).ravel()
    return sp.diags(1 / np.maximum(d, 1e-12)) @ W, d > 0


def spread(W, has, prior, a, it=20):
    f = prior.copy()
    for _ in range(it): f = np.where(has, (1 - a) * prior + a * (W @ f), prior)
    return f


PRIOR = {}
for s in SEEDS:
    p = np.zeros(N)
    for a in AR: p[ar == a] = np.load(f'results/{TAG}/units/{s}_{a}/local__lgbm.npz')['local|lgbm|external|score']
    PRIOR[s] = p


def metr(idx, pred, score):
    yy, aa = y[idx], ar[idx]
    f1 = [f1_score(yy[aa == a], pred[aa == a], average='macro') for a in AR if (aa == a).any() and len(set(yy[aa == a])) == 2]
    m = lambda mask: float(pred[mask].mean()) if mask.any() else np.nan
    return dict(M=100 * np.mean(f1), W=100 * np.min(f1), RD=100 * m((yy == 1) & ~st[idx]), FS=100 * m((yy == 0) & st[idx]),
                FPR=100 * m(yy == 0), REC=100 * m(yy == 1), AUC=roc_auc_score(yy, score))


def run(label, graph_fn, a):
    res = {w: [] for w in WINS}
    for w in WINS:
        idx = np.flatnonzero(win == w); W_, has = graph_fn(idx)
        for s in SEEDS:
            pr = PRIOR[s][idx]; f = spread(W_, has, pr, a) if W_ is not None else pr
            res[w].append(metr(idx, (f >= 0.5).astype(int), f))
    line = f'{label:30s}'
    for w in WINS:
        r = {k: np.mean([x[k] for x in res[w]]) for k in res[w][0]}
        line += f' | {w}: M {r["M"]:.1f} W {r["W"]:.1f} RD {r["RD"]:.1f} FS {r["FS"]:.1f} FPR {r["FPR"]:.2f} REC {r["REC"]:.1f}'
    aut = np.mean([np.mean([x['M'] for x in res[w]]) for w in WINS])
    print(line + f' | AUT(M) {aut:.2f}', flush=True)
    return res


if __name__ == '__main__':
    out = {}
    out['Local'] = run('Local', lambda idx: (None, None), 0)
    out['kNN-dactrung'] = run('kNN đặc trưng Local a0.8', lambda idx: knn(XD[idx], 10), 0.8)
    for cfg in sys.argv[1:]:
        kinds, maxdf, k, a = cfg.split(','); kinds = kinds.split('-')
        out[cfg] = run(f'CT {cfg}', lambda idx, kinds=kinds, maxdf=float(maxdf), k=int(k): knn(ent(idx, kinds, maxdf), k), float(a))
    json.dump(out, open(f'ct_v33_{os.getpid()}.json', 'w'), default=float)
