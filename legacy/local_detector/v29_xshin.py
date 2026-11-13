"""v29_xshin.py — XS-HIN chống trôi dạt (pilot). Kế thừa v28_xshin (thực thể chuỗi S + chữ ký hàm Fn).
Hai cơ chế mới:
 (a) Láng giềng KHÁC KIẾN TRÚC cho MỌI truy vấn (xa): khi suy luận, tham chiếu không có kiến trúc đích; khi huấn luyện thì
     có. Nếu tóm tắt láng giềng của tệp huấn luyện dùng cả tham chiếu cùng kiến trúc, phân phối đặc trưng lúc huấn luyện
     khác lúc kiểm thử. Ở đây mọi truy vấn chỉ nhìn tham chiếu khác kiến trúc -> đặc trưng nhất quán giữa huấn luyện và kiểm thử.
 (b) Lọc thực thể ỔN ĐỊNH (stab): chỉ giữ token xuất hiện ở >= G nhóm tương đồng và >= 2 kiến trúc trong pool tham chiếu
     (bỏ chuỗi riêng của một chiến dịch/biến thể, dễ lỗi thời). Chỉ dùng dữ liệu tham chiếu, không nhìn tập 2024.
Mode: xa (S,Fn,S+Fn; 15 đặc trưng), xas (như xa + lọc ổn định), local_xa, local_xas.
"""
import numpy as np, scipy.sparse as sp
import v28_xshin as X
import v24_core, v16_core as C, v16_run as R

G_MIN = 3


class XSHIN2(X.XSHIN):
    def __init__(self, P, ref, k=10):
        super().__init__(P, ref, k)
        self.stab_ = {}
        gr = P.groups[ref]; ar = P.arch[ref]
        for e in 'sf':
            M = getattr(P, 'tok_' + e)[ref].tocsc()
            ng = np.zeros(M.shape[1], int); na = np.zeros(M.shape[1], int)
            for j in range(M.shape[1]):
                rows = M.indices[M.indptr[j]:M.indptr[j + 1]]
                if len(rows): ng[j] = len(set(gr[rows])); na[j] = len(set(ar[rows]))
            keep = ((ng >= G_MIN) & (na >= 2)).astype(np.float32)
            self.stab_[e] = (self.idf_[e] * keep, self._sn(getattr(P, 'tok_' + e)[ref], self.idf_[e] * keep).T.tocsr())

    def transform_xa(self, d, ix, chunk=2000):
        out = {'xa': [], 'xas': []}
        Q = {e: self._sn(getattr(d, 'tok_' + e)[ix], self.idf_[e]) for e in 'sf'}
        Qs = {e: self._sn(getattr(d, 'tok_' + e)[ix], self.stab_[e][0]) for e in 'sf'}
        for i in range(0, len(ix), chunk):
            other = d.arch[ix][i:i + chunk][:, None] != self.arch_[None, :]
            for key, QQ, RR in (('xa', Q, {e: self.Rm_[e] for e in 'sf'}), ('xas', Qs, {e: self.stab_[e][1] for e in 'sf'})):
                s = {e: (QQ[e][i:i + chunk] @ RR[e]).toarray() for e in 'sf'}
                out[key].append(np.column_stack([self._summ(s['s'], other), self._summ(s['f'], other),
                                                 self._summ((s['s'] + s['f']) / 2, other)]))
        return {k: np.vstack(v) for k, v in out.items()}


_ob, _oa = R.blocks_for, R.assemble   # (đã là bản v28)
CACHE = {}


def blocks_for(P, d, ix, hin, local, extra):
    b = X.blocks_for(P, d, ix, hin, local, extra)
    ref = v24_core.LAST['ref']
    if CACHE.get('ref') is not ref:
        CACHE.clear(); CACHE['ref'] = ref; CACHE['m'] = XSHIN2(v24_core.LAST['P'], ref)
    b.update(CACHE['m'].transform_xa(d, ix))
    return b


def assemble(mode, b):
    if mode in ('xa', 'xas'): return b[mode]
    if mode in ('local_xa', 'local_xas'): return np.column_stack([b['L'], b[mode[6:]]])
    return X.assemble(mode, b)


R.blocks_for = blocks_for; R.assemble = assemble
if __name__ == '__main__':
    R.main()
