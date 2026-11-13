"""v28_xshin.py — XS-HIN: HIN với lớp thực thể MỚI, bất biến kiến trúc, có ở cả tệp tĩnh lẫn động.
Thực thể (trích từ JSON v5.2 bằng trich_token_v28.py, không dùng syscall):
  S  chuỗi ký tự đã chuẩn hóa (IP/hex/số được thay thế)
  Fn chữ ký hàm ngữ nghĩa (tập token sem của hàm + bậc log2 kích thước)
  Ap API/import (chỉ có ở tệp động -> lộ kiểu liên kết; chỉ dùng trong biến thể 'xsp' để đối chứng)
Meta-path: F-S-F, F-Fn-F, trung bình (S+Fn), F-S-F chỉ trên tham chiếu khác kiến trúc (+ F-Ap-F ở 'xsp').
Tóm tắt nhãn láng giềng y hệt HIN2/LR-HIN (k=10: knn_mal, max_mal, max_ben, margin, mean_topk) -> so sánh công bằng:
chỉ lớp thực thể thay đổi. Gắn vào pipeline v16/v24 (cùng phân vùng, Local, lưới, inner CV, chỉ số).
  python3 v28_xshin.py --protocol loao --seeds 2 --modes xs local_xs xsp --models lgbm --tag v28_N --no-purge
"""
import pickle, sys
import numpy as np, scipy.sparse as sp
import v24_core
import v16_core as C
import v16_run as R

import os, gzip, json
_EMPTY = dict(s=[], f=[], p=[])


def _load_tokens():
    """Token theo băm tệp. XS_TOKENS = danh sách tệp .jsonl.gz (ngăn cách bởi ':'), mặc định tokens_v28.pkl (cũ)."""
    src = os.environ.get('XS_TOKENS')
    if not src: return pickle.load(open('tokens_v28.pkl', 'rb'))
    out = {}
    for f in src.split(':'):
        for l in gzip.open(f, 'rt'):
            r = json.loads(l); out[r['h']] = r
    return out


TOK = _load_tokens()
_orig_load = C.load_all


def _attach(datasets):
    vocab = {k: {} for k in 'sfp'}
    for d in datasets:
        for k in 'sfp':
            rows, cols = [], []
            for i, h in enumerate(d.hash):
                for t in TOK.get(str(h), _EMPTY)[k]:
                    j = vocab[k].setdefault(t, len(vocab[k])); rows.append(i); cols.append(j)
            setattr(d, 'tok_' + k, (rows, cols))
    for d in datasets:
        for k in 'sfp':
            rows, cols = getattr(d, 'tok_' + k)
            M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(d.hash), len(vocab[k])))
            M.data[:] = 1.0; M.sum_duplicates(); M.data[:] = 1.0
            setattr(d, 'tok_' + k, M)


def load_all(data_root, cache=None):
    P, T, A = _orig_load(data_root) if cache is None else _orig_load(data_root, cache)
    if not hasattr(P, 'tok_s'): _attach([P, T, A])
    return P, T, A


C.load_all = load_all


class XSHIN(C.HIN2):
    def __init__(self, P, ref, k=10):
        self.k = k; self.ref = ref
        self.y_ = P.y[ref].astype(float); self.arch_ = P.arch[ref]
        self.idf_, self.Rm_ = {}, {}
        for e in 'sfp':
            X = getattr(P, 'tok_' + e)[ref]; df = np.asarray(X.sum(0)).ravel()
            idf = np.where(df > 0, np.log((1 + len(ref)) / (1 + df)) + 1, 0.0).astype(np.float32)
            self.idf_[e] = idf; self.Rm_[e] = self._sn(X, idf).T.tocsr()

    @staticmethod
    def _sn(X, idf):
        X = X.multiply(idf[None, :]).tocsr(); n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        return sp.diags(1 / np.maximum(n, 1e-12)) @ X

    def transform_d(self, d, ix, chunk=2000):
        out = []
        Q = {e: self._sn(getattr(d, 'tok_' + e)[ix], self.idf_[e]) for e in 'sfp'}
        for i in range(0, len(ix), chunk):
            s = {e: (Q[e][i:i + chunk] @ self.Rm_[e]).toarray() for e in 'sfp'}
            a = d.arch[ix][i:i + chunk]; other = a[:, None] != self.arch_[None, :]
            out.append(np.column_stack([self._summ(s['s']), self._summ(s['f']), self._summ((s['s'] + s['f']) / 2),
                                        self._summ(s['s'], other), self._summ(s['p'])]))
        return np.vstack(out)


_ob, _oa = R.blocks_for, R.assemble
CACHE = {}


def blocks_for(P, d, ix, hin, local, extra):
    b = _ob(P, d, ix, hin, local, extra)
    ref = v24_core.LAST['ref']
    if CACHE.get('ref') is not ref:
        CACHE.clear(); CACHE['ref'] = ref; CACHE['m'] = XSHIN(v24_core.LAST['P'], ref)
    b['XS'] = CACHE['m'].transform_d(d, ix)
    return b


def assemble(mode, b):
    if mode == 'xs': return b['XS'][:, :20]
    if mode == 'xsp': return b['XS']
    if mode == 'local_xs': return np.column_stack([b['L'], b['XS'][:, :20]])
    return _oa(mode, b)


R.blocks_for = blocks_for; R.assemble = assemble

if __name__ == '__main__':
    R.main()
