"""v24_core.py — các điều chỉnh TỐI THIỂU để chạy code đóng băng v16/v22/v23 trên dữ liệu dataset-moi (bộ trích v5.2).

Không sửa run_uploaded_experiments.py, v16_core.py, v16_run.py, v23_run.py (giữ nguyên mã băm của v23). Module này chỉ:
  (1) đặt số chỉ báo syscall cho HinDroidSVM / JaccardKNN theo dữ liệu (v4.1 có 52, v5.2 có 49 tên đã chuẩn hóa),
      vì hai lớp này cắt X[:, :n_calls] với n_calls mặc định 52;
  (2) thay v16_core.partition bằng bản dùng thêm KHÓA PURGE (cột goi_deb): trong dataset-moi, cùng một chương trình
      Buildroot được build cho nhiều kiến trúc nhưng nằm ở các cụm TLSH khác nhau, nên khóa purge = nhóm chương trình
      (brprog:<id>) — tương đương "gói Debian" trong dữ liệu YNU. Với tệp không có goi_deb, khóa purge = nhóm chia
      (giống hệt hàm gốc). Holdout và LOAO no-purge KHÔNG đổi so với hàm gốc.
Mọi điểm khác (Local, HIN-SC, HIN-v2, lưới tham số, inner CV, chỉ số) giữ nguyên.
"""
import numpy as np
import run_uploaded_experiments as base
import v16_core as C

_orig_load_all = C.load_all
_orig_partition = C.partition


def _n_calls(P):
    return len(P.call_columns)


def load_all(data_root, cache=None):
    P, T, A = _orig_load_all(data_root) if cache is None else _orig_load_all(data_root, cache)
    n = _n_calls(P)
    assert n == len(T.call_columns) == len(A.call_columns)
    C.HinDroidSVM.__init__.__defaults__ = (('AA',), 1.0, n)
    C.JaccardKNN.__init__.__defaults__ = (5, n)
    for d in (P, T, A):
        d.purge_key = np.array([(m.get('goi_deb') or '') for m in d.meta], dtype=object)
    return P, T, A


def partition(P, seed, setting='holdout', purge=True):
    """Như v16_core.partition; LOAO + purge loại thêm các tệp nguồn có khóa purge (goi_deb) xuất hiện ở kiến trúc đích."""
    if setting == 'holdout' or not purge:
        return _orig_partition(P, seed, setting, purge)
    pool = np.flatnonzero(P.arch != setting); test = np.flatnonzero(P.arch == setting)
    held = set(P.groups[test]); held_key = {k for k in P.purge_key[test] if k}
    keep = np.array([(P.groups[i] not in held) and not (P.purge_key[i] and P.purge_key[i] in held_key) for i in pool])
    purged = int((~keep).sum()); pool = pool[keep]
    train, ref = base.grouped_split(P, pool, seed + 1701)
    assert not (set(P.groups[train]) & set(P.groups[ref]))
    assert not (set(P.groups[train]) & set(P.groups[test]) or set(P.groups[ref]) & set(P.groups[test]))
    assert not (set(P.arch[train]) | set(P.arch[ref])) & {setting}
    return train, ref, test, purged


C.load_all = load_all
C.partition = partition


# ============================================================================ GNN-HIN (v24, G1)
"""GNN-HIN: mạng nơ-ron đồ thị dị thể dạng tách rời (decoupled; kiểu SIGN/NARS: Frasca et al. 2020; Yu et al. 2020).
Đồ thị: nút tệp, nút chỉ báo syscall (S, trọng số IDF), nút n-gram opcode ngữ nghĩa (N, TF-IDF top 495), nút kiến trúc (A).
Truyền thông điệp không tham số trên reference pool R (giống HIN-SC/HIN-v2: truy vấn không bao giờ thuộc R):
  thực thể e:   h_e = trung bình (có trọng số cạnh) đặc trưng local của các tệp R nối với e
  tệp q, quan hệ t: g_t(q) = trung bình (có trọng số cạnh) của h_e trên các thực thể loại t của q   (meta-path F-t-F)
Biểu diễn tệp = [local(q) | g_S(q) | g_N(q) | g_A(q)] (mode 'gnn') hoặc chỉ [g_S | g_N | g_A] (mode 'gnn_only'),
rồi bộ phân lớp MLP hoặc LightGBM của v16 (cùng lưới, cùng inner CV). Không dùng nhãn của R.
"""
import v16_run as R

LAST = {}
_part_with_purge = partition


def partition_tracked(P, seed, setting='holdout', purge=True):
    out = _part_with_purge(P, seed, setting, purge)
    LAST['ref'] = out[1]; LAST['P'] = P; LAST.pop('gnn', None)
    return out


C.partition = partition_tracked


class GNNHIN:
    def __init__(self, P, ref):
        self.ref = ref
        S = P.calls[ref].astype(float); df = S.sum(0); self.idf_ = np.log((1 + len(ref)) / (1 + df)) + 1
        self.sem_ = C.NgramTfidf('sem', 'sem_vocab', 495).fit(P, ref)
        self.arches_ = sorted(set(P.arch))
        self.WR = [self._w(P, ref, t) for t in 'SNA']

    def _w(self, d, ix, t):
        if t == 'S': return d.calls[ix].astype(float) * self.idf_
        if t == 'N': return self.sem_.transform(d, ix)
        return np.array([[a == b for b in self.arches_] for a in d.arch[ix]], dtype=float)

    def _entity(self, P, local):
        # cache gắn vào chính đối tượng Local (không dùng id(): id có thể bị tái sử dụng sau khi đối tượng cũ bị hủy)
        if getattr(local, '_gnn_owner', None) is not self:
            XR = local.transform(P, self.ref); H = []
            for W in self.WR:
                den = W.sum(0); H.append((W.T @ XR) / np.maximum(den, 1e-12)[:, None])
            local._gnn_owner = self; local._gnn_H = H
        return local._gnn_H

    def transform(self, P, d, ix, local):
        H = self._entity(P, local); out = []
        for t, He in zip('SNA', H):
            W = self._w(d, ix, t); den = W.sum(1)
            out.append((W @ He) / np.maximum(den, 1e-12)[:, None])
        return np.column_stack(out)


_orig_blocks = R.blocks_for
_orig_assemble = R.assemble
GNN_ON = {'on': False}


def blocks_for(P, d, ix, hin, local, extra):
    b = _orig_blocks(P, d, ix, hin, local, extra)
    if GNN_ON['on'] and local is not None:
        if 'gnn' not in LAST: LAST['gnn'] = GNNHIN(LAST['P'], LAST['ref'])
        b['G'] = LAST['gnn'].transform(LAST['P'], d, ix, local)
    return b


def assemble(mode, b):
    if mode == 'gnn': return np.column_stack([b['L'], b['G']])
    if mode == 'gnn_only': return b['G']
    return _orig_assemble(mode, b)


R.blocks_for = blocks_for
R.assemble = assemble
