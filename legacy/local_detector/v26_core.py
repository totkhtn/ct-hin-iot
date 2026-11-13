"""v26_core.py — nhánh CP (Composition-Preserving Purge) cho pipeline v16/v24, cùng kiểu v25_core.

Không sửa code đóng băng (run_uploaded_experiments, v16_core, v16_run, v24_core). Module này chỉ thay
v16_core.partition cho LOAO khi V26_NHANH=CP:
  * khóa = nhóm tương đồng (P.groups) và, nếu có, khóa purge (P.purge_key: chương trình Buildroot, v24);
  * mỗi khóa dùng chung giữa kiến trúc đích và kiến trúc nguồn được gán về phía huấn luyện (x=1, bỏ tệp đích
    mang khóa đó) hoặc phía kiểm thử (x=0, bỏ tệp nguồn — chính là purge chuẩn);
  * chọn x bằng MILP hai giai đoạn của cp_purge.py: max-min tỷ lệ giữ lại theo tầng (nhãn x liên kết tĩnh/động)
    trên cả hai phía, rồi max tổng tỷ lệ giữ lại. x = 0 tái tạo đúng partition purge của v16/v24.
  * C/R chia bằng base.grouped_split(P, pool, seed+1701) như hàm gốc; test = tệp đích còn giữ.
Đánh giá chính vẫn trên tập ngoài (không chung nhóm với tập phát triển) nên không đổi.
Thống kê phân vùng ghi vào V26_LOG (jsonl).
"""
import json, os
import numpy as np
import pandas as pd
import run_uploaded_experiments as base
import v16_core as C
import cp_purge as cpp

_part = C.partition          # partition đang dùng (v16 cho YNU; bản v24 nếu v24_core đã import trước)
_CACHE = {}


def _frame(P):
    pk = getattr(P, 'purge_key', None)
    return pd.DataFrame({
        '_id': [str(h) for h in P.hash], '_y': np.asarray(P.y, int), '_a': np.asarray(P.arch).astype(str),
        '_static': np.asarray(P.is_static, bool), '_g': np.asarray(P.groups).astype(str),
        '_k': pd.Series([(k or '') for k in pk] if pk is not None else [''] * len(P.y), dtype=object),
        '_stratum': np.where(np.asarray(P.y) == 1, 'mal', 'ben').astype(object) + '|' +
                    np.where(np.asarray(P.is_static, bool), 'static', 'dynamic').astype(object)})


def cp_masks(P, setting, time_limit=300.0):
    key = (id(P), setting)
    if key not in _CACHE:
        F = _frame(P)
        kQ, kT, info = cpp.cp_purge(F, setting, time_limit=time_limit)
        cpp.assert_leakage_free(F, kQ, kT)
        _CACHE[key] = (kQ, kT, info)
    return _CACHE[key]


def _log(rec):
    p = os.environ.get('V26_LOG')
    if p:
        with open(p, 'a') as f: f.write(json.dumps(rec) + '\n')


def partition_v26(P, seed, setting='holdout', purge=True):
    if setting == 'holdout' or os.environ.get('V26_NHANH') != 'CP':
        return _part(P, seed, setting, purge)
    kQ, kT, info = cp_masks(P, setting)
    pool = np.flatnonzero(kQ); test = np.flatnonzero(kT)
    train, ref = base.grouped_split(P, pool, seed + 1701)
    assert not (set(P.groups[train]) & set(P.groups[ref]))
    assert not ((set(P.groups[train]) | set(P.groups[ref])) & set(P.groups[test]))
    assert not (set(P.arch[train]) | set(P.arch[ref])) & {setting}
    import sys
    if 'v24_core' in sys.modules:                # GNN-HIN của v24 lấy pool tham chiếu từ LAST
        L = sys.modules['v24_core'].LAST; L['ref'] = ref; L['P'] = P; L.pop('gnn', None)
    removed = int((P.arch != setting).sum() - len(pool))
    _log(dict(nhanh='CP', seed=seed, setting=setting, **{k: v for k, v in info.items() if k != 'held_out'},
              removed_source=removed, test_kept=int(len(test)), test_total=int((P.arch == setting).sum()),
              sb_CR=int(((P.y[pool] == 0) & P.is_static[pool]).sum())))
    return train, ref, test, removed


C.partition = partition_v26
