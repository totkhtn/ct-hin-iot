#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trich_bang_v52_jcvht.py — dựng dữ liệu đầu vào v24 (dataset-moi, bộ trích v5.2) theo đúng định dạng mà
run_uploaded_experiments.py / v16_core.py / v23_run.py đọc (giống cd2_chay_lai_v9_upload.zip của JCVHT):

  <ra>/du_lieu_2020/   bang_v5_dense.csv.gz, sp_opc1.npz, sp_opc2.npz, sp_sem1.npz, sp_sem2.npz, sp_sc_full.npz, sp_tu_vung.json
  <ra>/du_lieu_2024/   (cùng định dạng)      -> tập kiểm thử ngoài (2024A, tỉ lệ tự nhiên, 46.045 tệp)
  <ra>/du_lieu_lt2020/ (cùng định dạng)      -> tập lành tính phụ trợ = bộ tự build Buildroot 2025-26 (1.354 tệp)
  <ra>/bao_cao_v24_du_lieu.txt + .json       -> kiểm tra (số tệp, nhãn, NaN, chỉ báo syscall x liên kết, ...)

Ba bước (chạy lại được; bước 2 có cache theo tệp):
  1) lap   : lập danh sách tệp + siêu dữ liệu (nhãn, kiến trúc, nguồn, nhóm chia, khóa purge, họ)
  2) trich : đọc JSON v5.2 (.json.gz), trích đặc trưng từng tệp -> cache
  3) gop   : gộp theo bộ, căn chung schema dense cho 3 bộ, ghi bảng + ma trận thưa + báo cáo

  python3 trich_bang_v52_jcvht.py tat_ca --meta <thư mục CSV> --json2020 <...> --json2024 <...> --jsonlt <...> --ra work/input --jobs 14

Quy ước cột (18 cột siêu dữ liệu đầu tiên, đúng thứ tự META_COLS của trich_bang_v5.py):
  nhom_chia : nhóm dùng để chia (StratifiedGroupKFold): 2020 = cụm TLSH<=30 (c30:<id>); 2024 = rỗng (chỉ là tập ngoài);
              lt2020 = b2526:<package>:<path_goc>.
  goi_deb   : KHÓA PURGE (tương đương gói Debian trong dữ liệu YNU): Buildroot 2020 = brprog:<nhom_ct> (nhóm chương trình
              build cho nhiều kiến trúc); tệp khác = rỗng (khi đó purge dùng nhom_chia). Chỉ v24_core.partition dùng cột này.
Tiền tố đặc trưng giống bản v4.1: HDR LNK SEC ENT CFG STR SC BH BEH UPH UPS (bộ phân lớp chỉ dùng BH BEH ENT CFG STR SC).
LNK_is_static = liên kết THẬT (dùng up_is_static với tệp UPX đã giải nén); giá trị thô ở LNK_is_static_raw. Cột LNK chỉ dùng
để kiểm tra/phân tầng, không bao giờ là biến dự báo.
"""
import argparse, csv, gzip, json, os, pickle, sys, time, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np

csv.field_size_limit(10 ** 9)
REV = 'v24a'
META_COLS = ['file_hash', 'label', 'family', 'lop4', 'arch', 'nguon', 'goi_deb', 'nam', 'nhom_chia',
             'label_json', 'arch_json', 'status', 'extractor_rev', 'dong_goi', 'elf_type', 'cfg_timed_out',
             'n_thieu_ops_seq', 'chua_phan_loai']
SP_KINDS = ['opc1', 'opc2', 'sem1', 'sem2', 'sc_full']
PRED_PREFIX = ('BH_', 'BEH_', 'ENT_', 'CFG_', 'STR_', 'SC_')

SEC_KEYS = {'num_sections', 'nonstandard_section_names', 'no_section_headers', 'entry_in_text', 'text_size',
            'data_size', 'bss_size', 'rodata_size', 'total_section_size', 'total_section_size_file',
            'n_exec_sections', 'n_writable_sections', 'n_nobits_sections', 'text_entropy', 'max_section_entropy',
            'num_imports', 'num_dynsym', 'num_symtab', 'num_defined_funcs_symtab', 'is_stripped', 'n_header_anomalies'}
LNK_KEYS = {'is_static', 'n_needed_libs', 'has_interp', 'has_dynamic', 'elf_type_exec', 'elf_type_dyn'}
HDR_KEYS = {'e_machine', 'bits', 'endian_big', 'osabi', 'e_flags', 'arm_eabi_version', 'arm_float_hard', 'mips_pic',
            'file_size', 'entry_outside_exec_segment', 'num_segments', 'has_rwx_segment', 'exec_stack',
            'n_load_segments', 'n_exec_load_segments', 'max_memsz_filesz_ratio', 'exec_memsz_filesz_ratio',
            'total_load_memsz', 'has_tls', 'has_gnu_relro', 'is_packed', 'n_packer_hints'}
ENT_KEYS = {'file_entropy', 'window_entropy_mean', 'window_entropy_max', 'window_entropy_min', 'window_entropy_std'}
CFG_KEYS = {'cfg_num_nodes', 'cfg_num_edges', 'cfg_num_call_edges', 'cfg_total_insn', 'cfg_avg_block_insn',
            'fn_n_functions', 'fn_avg_func_size', 'fn_median_func_size', 'fn_max_func_size', 'fn_avg_cc',
            'fn_max_cc', 'fn_n_named_functions', 'n_called_named_functions'}


def nhom_cua(k):
    if k.startswith('str_'): return 'STR'
    if k.startswith('sc_') or k == 'sem_syscall_insn': return 'SC'
    if k.startswith('up_'):
        g = nhom_cua(k[3:])
        return None if g is None else ('UPS' if g == 'SEC' else 'UPH')
    for g, s in (('SEC', SEC_KEYS), ('LNK', LNK_KEYS), ('HDR', HDR_KEYS), ('ENT', ENT_KEYS), ('CFG', CFG_KEYS)):
        if k in s: return g
    return None


def ho_4_lop(fam, label):
    if str(label) == '0': return 'Benign'
    f = (fam or '').lower()
    if 'mirai' in f: return 'Mirai'
    if 'gafgyt' in f or 'bashlite' in f: return 'Gafgyt'
    return 'Rare/Other'


# ============================================================ 1) lập danh sách
def lap(a):
    M = a.meta
    rd = lambda f: list(csv.DictReader(open(os.path.join(M, f), encoding='utf-8')))
    fam20 = {}
    if os.path.exists(os.path.join(M, 'dataset_with_family_HoMD.csv')):
        for r in rd('dataset_with_family_HoMD.csv'):
            fam20[r['file_hash'].lower()] = r.get('malware_family') or ''
    br = {r['file_hash'].lower(): r for r in rd('cum_chuong_trinh_buildroot_v52.csv')}
    out = {}
    # ---- 2020: tập chính 1:1 (2020A u1)
    L = []
    for r in rd('tap_1_1_2020A_u1.csv'):
        if r['trong_tap'] != '1': continue
        h = r['file_hash'].lower(); b = br.get(h)
        L.append(dict(file_hash=h, label=r['label'], arch=r['arch'], nguon=r['nguon'],
                      nhom_chia='c30:%d' % int(float(r['cum30'])),
                      goi_deb=('brprog:%s' % b['nhom_ct']) if (b and r['nguon'] == 'LT_tinh_buildroot') else '',
                      nam=(b or {}).get('nam', ''), malware_family=fam20.get(h, '') if r['label'] == '1' else 'benign',
                      lk=r['lk']))
    out['2020'] = L
    # ---- 2024: tập kiểm thử tỉ lệ tự nhiên (2024A, bỏ dòng có ly_do)
    fam24 = {}
    if os.path.exists(os.path.join(M, 'Finale_DatasetF-1_report_15967.csv')):
        for r in rd('Finale_DatasetF-1_report_15967.csv'):
            fam24[r['file_hash'].lower()] = r.get('vt_malware_family') or ''
    L = []
    for r in rd('danh_sach_2024A.csv'):
        if r.get('ly_do'): continue
        h = r['file_hash'].lower()
        L.append(dict(file_hash=h, label=r['label'], arch=r['arch'], nguon=r['nguon'], nhom_chia='', goi_deb='', nam='',
                      malware_family=fam24.get(h, '') if r['label'] == '1' else 'benign', lk=''))
    out['2024'] = L
    # ---- lt2020: bộ lành tính tự build Buildroot 2025-26 (đã lọc theo manifest)
    L = []
    for r in rd('build2526_flat_loc.csv'):
        h = r['file_hash'].lower()
        L.append(dict(file_hash=h, label='0', arch=r['arch'], nguon='build2526_' + (r.get('libc') or ''),
                      nhom_chia='b2526:%s:%s' % (r.get('package', ''), r.get('path_goc', '')), goi_deb=r.get('package', ''),
                      nam=r.get('br_tag', ''), malware_family='benign', lk=r.get('link_manifest', '')))
    out['lt2020'] = L
    for k, L in out.items():
        assert len({x['file_hash'] for x in L}) == len(L), k
        print('danh sach', k, len(L), dict(Counter((x['nguon'], x['label']) for x in L).most_common(12)))
    os.makedirs(a.ra, exist_ok=True)
    json.dump(out, open(os.path.join(a.ra, 'danh_sach_v24.json'), 'w'))
    return out


# ============================================================ 2) trích từng tệp
def entropy_cols(d):
    seq = np.array([float(x) for x in (d.get('entropy_sequence') or []) if isinstance(x, (int, float))], dtype=float)
    mean = float(d.get('mean_entropy') or 0.0); mx = float(d.get('max_entropy') or 0.0); mn = float(d.get('min_entropy') or 0.0)
    if seq.size:
        v = [mean, mx, mn, float(seq.std()), float(np.quantile(seq, .1)), float(np.quantile(seq, .9)), float((seq > 7).mean()), float(seq.size)]
    else:
        v = [mean, mx, mn, 0.0, mean, mean, float(mean > 7.0), 0.0]
    return {'ENT_v14_' + t: x for t, x in zip(['mean', 'max', 'min', 'std', 'q10', 'q90', 'frac_gt7', 'seq_len'], v)}


def cfg8(cg):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
    ten = ['n', 'm', 'density', 'avg_path', 'avg_deg', 'cyclomatic', 'var_deg', 'cycle_rank']
    if not isinstance(cg, dict): return {'CFG_8_' + t: 0.0 for t in ten}
    idx = {}
    for nd in cg.get('nodes', []) or []:
        k = nd.get('id', nd.get('name')) if isinstance(nd, dict) else nd
        if k not in idx: idx[k] = len(idx)
    E = set()
    for e in cg.get('edges', []) or []:
        a_, b_ = (e.get('source'), e.get('target')) if isinstance(e, dict) else \
            (e[0], e[1]) if isinstance(e, (list, tuple)) and len(e) >= 2 else (None, None)
        if a_ is None and b_ is None: continue
        for k in (a_, b_):
            if k not in idx: idx[k] = len(idx)
        E.add((idx[a_], idx[b_]))
    n, m = len(idx), len(E)
    if n == 0: return {'CFG_8_' + t: 0.0 for t in ten}
    s = np.array([e[0] for e in E], dtype=np.int64); t_ = np.array([e[1] for e in E], dtype=np.int64)
    deg = np.bincount(s, minlength=n) + np.bincount(t_, minlength=n)
    A = csr_matrix((np.ones(m), (s, t_)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=True, connection='weak')
    nodes = np.where(lab == np.argmax(np.bincount(lab)))[0]
    avg = 0.0
    if len(nodes) > 1:
        U = (A + A.T).tocsr()[nodes][:, nodes]
        if len(nodes) <= 300:
            D = shortest_path(U, unweighted=True, directed=False); avg = float(D.sum() / (len(nodes) * (len(nodes) - 1)))
        else:
            src = np.array(random.Random(0).sample(range(len(nodes)), 50))
            D = shortest_path(U, unweighted=True, directed=False, indices=src)
            vals = [r[np.isfinite(r)].sum() / (np.isfinite(r).sum() - 1) for r in D if np.isfinite(r).sum() > 1]
            avg = float(np.mean(vals)) if vals else 0.0
    dens = m / (n * (n - 1)) if n > 1 else 0.0
    v = [n, m, dens, avg, deg.mean() / 2.0, max(0, m - n + 2 * ncomp), deg.var(), max(0, m - n + ncomp)]
    return {'CFG_8_' + t: float(x) for t, x in zip(ten, v)}


def chuoi_dem(cg):
    opc1, opc2, sem1, sem2 = Counter(), Counter(), Counter(), Counter(); thieu = 0
    for nd in (cg.get('nodes', []) if isinstance(cg, dict) else []) or []:
        if not isinstance(nd, dict): continue
        tf = {}
        for o, c in (nd.get('opcodes_tf') or {}).items():
            k = str(o).strip().lower(); tf[k] = tf.get(k, 0) + int(c)
        opc1.update(tf)
        raw = nd.get('ops_seq') or []
        if raw:
            seq = [str(o).strip().lower() for o in raw]; opc2.update(zip(seq, seq[1:]))
        elif sum(tf.values()) > 1:
            thieu += 1
        for k, c in (nd.get('sem_tf') or {}).items(): sem1[str(k)] += int(c)
        ss = [str(o) for o in (nd.get('sem_seq') or [])]
        if len(ss) > 1: sem2.update(zip(ss, ss[1:]))
    g = lambda C: {f'{x}_{y}': v for (x, y), v in C.items()}
    return dict(opc1=dict(opc1), opc2=g(opc2), sem1=dict(sem1), sem2=g(sem2)), thieu


def mot_tep(p):
    with (gzip.open(p, 'rb') if p.endswith('.gz') else open(p, 'rb')) as f:
        d = json.loads(f.read())
    h = str(d.get('file_hash') or os.path.basename(p)[:64]).lower()
    ff = d.get('features_flat') or {}
    row, cpl = {}, []
    for k, v in ff.items():
        g = nhom_cua(k)
        if g is None:
            cpl.append(k); g = 'XTR'   # khóa chưa phân nhóm: giữ lại với tiền tố XTR_ (không bao giờ là biến dự báo)
        row[f'{g}_{k}'] = float(v) if isinstance(v, (int, float)) else np.nan
    raw = ff.get('is_static'); up = ff.get('up_is_static')
    real = up if (str(d.get('dong_goi')) == 'upx_da_giai' and isinstance(up, (int, float))) else raw
    row['LNK_is_static_raw'] = float(raw) if isinstance(raw, (int, float)) else np.nan
    row['LNK_is_static'] = float(real) if isinstance(real, (int, float)) else np.nan
    row.update(entropy_cols(d))
    cg = d.get('cfg_graph') or {}
    row.update(cfg8(cg))
    for i, v in enumerate((d.get('byte_histogram') or [])[:256]): row[f'BH_{i:03d}'] = float(v)
    for i, v in enumerate((d.get('byte_entropy_histogram') or [])[:256]): row[f'BEH_{i:03d}'] = float(v)
    sp, thieu = chuoi_dem(cg)
    r2 = d.get('r2') or {}
    sc = r2.get('syscalls_chuan') or r2.get('syscalls') or {}
    sp['sc_full'] = {str(k): int(v) for k, v in sc.items() if isinstance(v, (int, float))}
    em = d.get('elf_meta') or {}
    meta = dict(label_json=int(d.get('label', -1)), arch_json=str(d.get('arch')), status=str(d.get('status')),
                extractor_rev=str(d.get('extractor_rev')), dong_goi=str(d.get('dong_goi')), elf_type=str(em.get('elf_type')),
                cfg_timed_out=int(bool(cg.get('timed_out'))) if isinstance(cg, dict) else -1, n_thieu_ops_seq=thieu,
                chua_phan_loai='|'.join(sorted(cpl)))
    return h, dict(meta=meta, row=row, sp=sp)


def _an_toan(p):
    try:
        return mot_tep(p)
    except Exception as e:
        return ('LOI', p, repr(e))


def tim_json(folders):
    """Tìm JSON v5.2 (đệ quy) trong một hoặc nhiều thư mục; khóa = sha256 ở đầu tên tệp."""
    if isinstance(folders, str): folders = [folders]
    m = {}
    for folder in folders:
        for root, _, fs in os.walk(folder):
            for fn in fs:
                if fn.endswith(('_static.json.gz', '_static.json')):
                    m.setdefault(fn.split('__')[0][:64].lower(), os.path.join(root, fn))
    return m


def trich(a, ds):
    cache = os.path.join(a.ra, 'cache_' + REV); os.makedirs(cache, exist_ok=True)
    dirs = {'2020': a.json2020, '2024': a.json2024, 'lt2020': a.jsonlt}
    viec, thieu = [], {}
    for k, L in ds.items():
        J = tim_json(dirs[k]); miss = [x['file_hash'] for x in L if x['file_hash'] not in J]
        thieu[k] = miss
        viec += [J[x['file_hash']] for x in L if x['file_hash'] in J and not os.path.exists(os.path.join(cache, x['file_hash'] + '.pkl.gz'))]
        print(f'{k}: {len(L)} trong danh sach, thieu JSON {len(miss)}', flush=True)
    print('can trich', len(viec), flush=True)
    t0, n, loi = time.time(), 0, []
    with Pool(a.jobs) as pool:
        for kq in pool.imap_unordered(_an_toan, viec, chunksize=4):
            if kq[0] == 'LOI':
                loi.append(kq[1:]); continue
            h, rec = kq; tmp = os.path.join(cache, h + '.tmp')
            with gzip.open(tmp, 'wb', compresslevel=1) as f: pickle.dump(rec, f, protocol=4)
            os.replace(tmp, os.path.join(cache, h + '.pkl.gz')); n += 1
            if n % 1000 == 0: print(f'  {n}/{len(viec)} tep, {time.time() - t0:.0f}s', flush=True)
    for p, e in loi: print('LOI', p, e, flush=True)
    json.dump(dict(thieu_json=thieu, loi=loi), open(os.path.join(a.ra, 'trich_v24_loi.json'), 'w'), indent=1)
    print(f'xong trich: {n} tep moi, {len(loi)} loi, {time.time() - t0:.0f}s', flush=True)
    return thieu, loi


# ============================================================ 3) gộp
def gop(a, ds):
    import pandas as pd
    from scipy import sparse
    cache = os.path.join(a.ra, 'cache_' + REV)
    tables, mats, rep = {}, {}, {}
    for k, L in ds.items():
        L = sorted([x for x in L if os.path.exists(os.path.join(cache, x['file_hash'] + '.pkl.gz'))], key=lambda x: x['file_hash'])
        metas, rows = [], []; voc = {s: {} for s in SP_KINDS}; coo = {s: ([], [], []) for s in SP_KINDS}
        for i, x in enumerate(L):
            with gzip.open(os.path.join(cache, x['file_hash'] + '.pkl.gz'), 'rb') as f: rec = pickle.load(f)
            m = rec['meta']
            metas.append(dict(file_hash=x['file_hash'], label=int(x['label']), family=x['malware_family'],
                              lop4=ho_4_lop(x['malware_family'], x['label']), arch=x['arch'], nguon=x['nguon'],
                              goi_deb=x['goi_deb'], nam=x['nam'], nhom_chia=x['nhom_chia'], **m))
            rows.append(rec['row'])
            for s in SP_KINDS:
                V = voc[s]; R, C, X = coo[s]
                for key, val in rec['sp'].get(s, {}).items():
                    j = V.get(key)
                    if j is None: j = V[key] = len(V)
                    R.append(i); C.append(j); X.append(val)
        df = pd.DataFrame(rows)
        tables[k] = (pd.DataFrame(metas)[META_COLS], df)
        mats[k] = (voc, coo, len(L))
        print('gop', k, len(L), 'tep', flush=True)
    # ---- schema dense chung cho 3 bộ (bắt buộc: run_uploaded_experiments kiểm tra P.columns == T.columns == A.columns)
    cols = sorted(set().union(*[set(t[1].columns) for t in tables.values()]))
    for k, (meta, df) in tables.items():
        miss_cols = [c for c in cols if c not in df.columns]
        df = df.reindex(columns=cols)
        pred = [c for c in cols if c.startswith(PRED_PREFIX)]
        nan = df[pred].isna().sum(); nan = {c: int(v) for c, v in nan.items() if v}
        df[pred] = df[pred].fillna(0.0)
        rep[k] = dict(n=len(df), cot_thieu_trong_bo=miss_cols[:50], so_cot_thieu=len(miss_cols), nan_bien_du_bao_da_dien_0=nan)
        out = os.path.join(a.ra, 'du_lieu_' + k); os.makedirs(out, exist_ok=True)
        full = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        full.to_csv(os.path.join(out, 'bang_v5_dense.csv.gz'), index=False, compression='gzip')
        voc, coo, n = mats[k]; vocab = {}
        for s in SP_KINDS:
            keys = sorted(voc[s]); gi = np.array([0] * len(voc[s]), dtype=np.int64)
            pos = {key: j for j, key in enumerate(keys)}
            for key, j in voc[s].items(): gi[j] = pos[key]
            R, C, X = coo[s]
            Mx = sparse.csr_matrix((np.array(X, np.int64).astype(np.float64), (np.array(R, np.int64), gi[np.array(C, np.int64)] if C else np.array([], np.int64))),
                                   shape=(n, len(keys)))
            Mx.sum_duplicates(); sparse.save_npz(os.path.join(out, f'sp_{s}.npz'), Mx.astype(np.int32), compressed=True)
            vocab[s] = keys
        json.dump(vocab, open(os.path.join(out, 'sp_tu_vung.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        # ---- kiểm tra
        r = rep[k]
        r['nhan_khop_json'] = int((full.label == full.label_json).sum()); r['kien_truc_khop_json'] = int((full.arch == full.arch_json).sum())
        r['status'] = full.status.value_counts().to_dict(); r['extractor_rev'] = full.extractor_rev.value_counts().to_dict()
        r['nguon_x_nhan'] = {f'{a_}|{b_}': int(v) for (a_, b_), v in full.groupby(['nguon', 'label']).size().items()}
        r['kien_truc_x_nhan'] = {f'{a_}|{b_}': int(v) for (a_, b_), v in full.groupby(['arch', 'label']).size().items()}
        sc = [c for c in cols if c.startswith('SC_sc_') and c not in ('SC_sc_sites', 'SC_sc_unique')]
        r['so_chi_bao_syscall'] = len(sc)
        has = (full[sc] > 0).any(axis=1); st = full.LNK_is_static > 0
        r['chi_bao_x_lien_ket'] = {f'nhan{y}|{"tinh" if s else "dong"}': f'{int((has & (st == s) & (full.label == y)).sum())}/{int(((st == s) & (full.label == y)).sum())}'
                                   for y in (0, 1) for s in (True, False)}
        r['upx_da_giai'] = int((full.dong_goi == 'upx_da_giai').sum())
        r['chua_phan_loai'] = sorted({c for s in full.chua_phan_loai.fillna('') if s for c in s.split('|')})
        r['tu_vung_thua'] = {s: len(v) for s, v in vocab.items()}
        r['nhom_chia'] = int(full.nhom_chia.replace('', np.nan).nunique()); r['khoa_purge_brprog'] = full.goi_deb[full.goi_deb.str.startswith('brprog:', na=False)].value_counts().to_dict()
        r['trung_hash'] = int(full.file_hash.duplicated().sum())
    pred_cols = [c for c in cols if c.startswith(PRED_PREFIX)]
    rep['schema'] = dict(so_cot_dense=len(cols), so_bien_du_bao=len(pred_cols),
                         theo_tien_to=dict(Counter(c.split('_')[0] for c in cols)))
    h20 = set(tables['2020'][0].file_hash); rep['giao_hash'] = {k: len(h20 & set(tables[k][0].file_hash)) for k in ('2024', 'lt2020')}
    json.dump(rep, open(os.path.join(a.ra, 'bao_cao_v24_du_lieu.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    lines = []
    for k, r in rep.items():
        lines.append(f'== {k}'); lines += [f'  {kk}: {vv}' for kk, vv in r.items()]
    open(os.path.join(a.ra, 'bao_cao_v24_du_lieu.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    ok = all(rep[k]['trung_hash'] == 0 and rep[k]['nhan_khop_json'] == rep[k]['n'] and rep[k]['kien_truc_khop_json'] == rep[k]['n'] for k in ds)
    print('KIEM TRA NHAN/KIEN TRUC/TRUNG HASH:', 'DAT' if ok else 'KHONG DAT')
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('buoc', choices=['lap', 'trich', 'gop', 'tat_ca'])
    ap.add_argument('--meta', required=True, help='thư mục chứa các CSV siêu dữ liệu (xem HUONG_DAN_v24.md)')
    ap.add_argument('--json2020', nargs='+'); ap.add_argument('--json2024', nargs='+'); ap.add_argument('--jsonlt', nargs='+')
    ap.add_argument('--ra', default='work/input'); ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--mau', type=int, default=0, help='chỉ lấy N tệp mỗi bộ (kiểm thử)')
    a = ap.parse_args()
    ds = lap(a) if a.buoc in ('lap', 'tat_ca') else json.load(open(os.path.join(a.ra, 'danh_sach_v24.json')))
    if a.mau:
        rng = random.Random(0)
        for k in ds:
            dirs = {'2020': a.json2020, '2024': a.json2024, 'lt2020': a.jsonlt}
            J = tim_json(dirs[k]) if dirs[k] else {}
            L = [x for x in ds[k] if x['file_hash'] in J] if J else ds[k]
            rng.shuffle(L); ds[k] = L[:a.mau]
        json.dump(ds, open(os.path.join(a.ra, 'danh_sach_v24.json'), 'w'))
    if a.buoc in ('trich', 'tat_ca'): trich(a, ds)
    if a.buoc in ('gop', 'tat_ca'): sys.exit(gop(a, ds))


if __name__ == '__main__':
    main()
