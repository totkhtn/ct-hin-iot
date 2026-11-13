"""v31_run.py — XS-HIN (thực thể chuỗi + chữ ký hàm, láng giềng khác kiến trúc) và ablation, trên 2 corpus x 3 giao thức.
  python3 v31_run.py <ds> <arm> --protocol loao --seeds S --architectures A --modes ... --models lgbm --tag TAG
  ds : ynu (Corpus A, IoTPOT) | v24 (Corpus B, dataset-moi)
  arm: N (LOAO không purge) | P (purge chuẩn) | CP (CP-Purge, v26_core)
Token: XS_TOKENS (mặc định tokens_A.jsonl.gz cho ynu, tokens_B.jsonl.gz cho v24, trong thư mục chạy).
Mode: xa, local_xa (XS-HIN-xa, đề xuất) | hin2xa, local_hin2xa | local_hin2 | local_xs | local, hin2 (baseline).
"""
import os, sys

if __name__ == '__main__':
    ds, arm = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]
    if ds not in ('ynu', 'v24') or arm not in ('N', 'P', 'CP'): sys.exit('ds: ynu|v24 ; arm: N|P|CP')
    os.environ.setdefault('XS_TOKENS', 'tokens_A.jsonl.gz' if ds == 'ynu' else 'tokens_B.jsonl.gz')
    import v30_ablation  # noqa: F401  (nạp v24_core -> v28 -> v29 -> v30)
    if ds == 'ynu':   # Corpus A: giữ đúng phân vùng v16 và CP-Purge v26 như các nhánh đã chạy (không dùng khóa purge goi_deb)
        import v24_core, v16_core as C
        v24_core._part_with_purge = v24_core._orig_partition
        _la = C.load_all

        def load_all_ynu(*a, **k):
            P, T, A = _la(*a, **k)
            for d in (P, T, A):
                if hasattr(d, 'purge_key'): del d.purge_key
            return P, T, A
        C.load_all = load_all_ynu
    if arm == 'CP':
        os.environ['V26_NHANH'] = 'CP'
        tag = rest[rest.index('--tag') + 1]
        os.makedirs(os.path.join('results', tag), exist_ok=True)
        os.environ.setdefault('V26_LOG', os.path.join('results', tag, 'phan_vung_CP.jsonl'))
        import v26_core  # noqa: F401
    elif arm == 'N':
        rest = rest + ['--no-purge']
    import v16_run as R
    sys.argv = [sys.argv[0]] + rest
    R.main()
