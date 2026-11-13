"""dung_v33.py D|H — dựng dữ liệu theo cửa sổ thời gian cho CT-HIN (cùng schema Corpus B).
  D (phát triển): mã độc MalwareBazaar W2025, W26Q1, W26Q2 + lành tính phát triển (lành tính C3 trừ nửa tĩnh-nhỏ giữ kín),
                  lành tính được gán vào cửa sổ theo băm, tỷ lệ theo số mã độc của cửa sổ. Cửa sổ ghi ở cột 'nam'.
  H (GIỮ KÍN):    mã độc 2026 Q3–Q4 CHƯA AI XEM + nửa tĩnh-nhỏ giữ kín + 8.000 lành tính động phần dư 2020 chưa dùng.
Ra: /home/claude/v33/<D|H>/work/input/du_lieu_{2020,lt2020 (liên kết B), 2024 (tập này)}, tokens_<D|H>.jsonl.gz
"""
import csv, glob, gzip, hashlib, json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0, '/home/claude/run/B')
import trich_bang_v52_jcvht as Bld

WHICH = sys.argv[1]
B_IN = '/home/claude/run/B/work/input'; OUT = f'/home/claude/v33/{WHICH}/work/input'
CH = json.load(open('/home/claude/v33/v33_chia_cua_so.json'))
WIN = CH['win']; NHO_H = set(CH['nho_holdout']); ARCH = {'EM_386', 'EM_ARM', 'EM_MIPS', 'EM_PPC', 'EM_X86_64'}
os.makedirs(OUT, exist_ok=True)
for k in ('du_lieu_2020', 'du_lieu_lt2020'):
    if not os.path.exists(os.path.join(OUT, k)): os.symlink(os.path.join(B_IN, k), os.path.join(OUT, k))

DEV_W = ['W2025', 'W26Q1', 'W26Q2']
nmal = Counter(v['w'] for v in WIN.values())
cum = np.cumsum([nmal[w] for w in DEV_W]) / sum(nmal[w] for w in DEV_W)


def ben_window(h):
    u = (int(hashlib.sha256(('cs' + h).encode()).hexdigest(), 16) % 10 ** 6) / 10 ** 6
    return DEV_W[int(np.searchsorted(cum, u, side='right'))]


def chon(r):
    """-> cửa sổ (chuỗi) nếu bản ghi thuộc tập WHICH, ngược lại None"""
    h, tag = r['h'], r['tag']
    if tag.startswith('MB'):
        v = WIN.get(h)
        if v is None: return None
        if WHICH == 'D': return v['w'] if v['w'] in DEV_W else None
        return 'H26' if (v['w'] == 'H26' and tag == 'MB_H26') else None   # chỉ bản trích mới = chưa xem
    if tag == 'LT_du2020_H': return 'H26' if WHICH == 'H' else None
    if tag == 'LT_tinh_nho':
        if h in NHO_H: return 'H26' if WHICH == 'H' else None
        return ben_window(h) if WHICH == 'D' else None
    if tag in ('LT_du2020', 'LT2024_begin'): return ben_window(h) if WHICH == 'D' else None
    return None


bh = set()
for k in ('du_lieu_2020', 'du_lieu_2024', 'du_lieu_lt2020'):
    with gzip.open(os.path.join(B_IN, k, 'bang_v5_dense.csv.gz'), 'rt') as f:
        bh |= {r['file_hash'] for r in csv.DictReader(f)}
with gzip.open(os.path.join(B_IN, 'du_lieu_2024', 'bang_v5_dense.csv.gz'), 'rt') as f:
    cols_all = next(csv.reader(f))
meta_cols = cols_all[:18]; feat_cols = cols_all[18:]; fidx = {c: j for j, c in enumerate(feat_cols)}
pred_mask = np.array([c.startswith(Bld.PRED_PREFIX) for c in feat_cols])
metas, dense, hashes, seen, bo = [], [], [], set(), Counter()
VOC = {s: {} for s in Bld.SP_KINDS}; COO = {s: ([], [], []) for s in Bld.SP_KINDS}
tokf = gzip.open(f'/home/claude/v33/{WHICH}/tokens_xs_moi.jsonl.gz', 'wt')
files = sorted(glob.glob('/home/claude/c3/lo/lo_*.pkl.gz')) + sorted(glob.glob('/home/claude/v33/lo_moi/lo_*.pkl.gz'))
for f in files:
    for r in pickle.load(gzip.open(f)):
        if 'loi' in r: bo['loi'] += 1; continue
        if r['tag'] in ('MB2025', 'MB2026'): r = dict(r, tag='MB_C3')
        w = chon(r)
        if w is None: continue
        m = r['rec']['meta']; y = 1 if r['tag'].startswith('MB') else 0
        if m['arch_json'] not in ARCH: bo['kien_truc'] += 1; continue
        if m['label_json'] != y: bo['nhan_lech'] += 1; continue
        if r['h'] in bh or r['h'] in seen or len(r['h']) != 64: bo['trung'] += 1; continue
        i = len(hashes); seen.add(r['h']); hashes.append(r['h'])
        src = r['tag'] if not r['tag'].startswith('MB') else 'MB'
        metas.append(dict(file_hash=r['h'], label=y, family=(WIN.get(r['h'], {}).get('fam', '') if y else 'benign'),
                          lop4=Bld.ho_4_lop('' if y else 'benign', str(y)), arch=m['arch_json'], nguon=src, goi_deb='',
                          nam=w, nhom_chia='', **m))
        v = np.full(len(feat_cols), np.nan)
        for k, x in r['rec']['row'].items():
            j = fidx.get(k)
            if j is not None: v[j] = x
        dense.append(v)
        for s in Bld.SP_KINDS:
            V = VOC[s]; R_, C_, X_ = COO[s]
            for key, val in r['rec']['sp'].get(s, {}).items():
                j = V.setdefault(key, len(V)); R_.append(i); C_.append(j); X_.append(val)
        tokf.write(json.dumps(r['tok']) + '\n')
    print(os.path.basename(f), len(hashes), flush=True)
tokf.close()
print('giữ', len(hashes), 'bỏ', dict(bo))
order = np.argsort(np.array(hashes)); inv = np.empty_like(order); inv[order] = np.arange(len(order))
D = np.vstack(dense)[order]; D[:, pred_mask] = np.nan_to_num(D[:, pred_mask], nan=0.0)
meta = pd.DataFrame(metas)[meta_cols].iloc[order].reset_index(drop=True)
d = os.path.join(OUT, 'du_lieu_2024'); os.makedirs(d, exist_ok=True)
pd.concat([meta, pd.DataFrame(D, columns=feat_cols)], axis=1).to_csv(os.path.join(d, 'bang_v5_dense.csv.gz'), index=False, compression='gzip')
vocab = {}
for s in Bld.SP_KINDS:
    V = VOC[s]; R_, C_, X_ = COO[s]; keys = sorted(V); pos = {k: j for j, k in enumerate(keys)}
    gi = np.zeros(len(V), np.int64)
    for k, j in V.items(): gi[j] = pos[k]
    M = sparse.csr_matrix((np.array(X_, np.float64), (inv[np.array(R_, np.int64)] if R_ else np.array([], np.int64),
                                                      gi[np.array(C_, np.int64)] if C_ else np.array([], np.int64))), shape=(len(hashes), len(keys)))
    M.sum_duplicates(); sparse.save_npz(os.path.join(d, f'sp_{s}.npz'), M.astype(np.int32), compressed=True); vocab[s] = keys; del COO[s]
json.dump(vocab, open(os.path.join(d, 'sp_tu_vung.json'), 'w', encoding='utf-8'), ensure_ascii=False)
rep = {f'{w}|{s}|{y}': int(n) for (w, s, y), n in meta.assign(st=(D[:, fidx['LNK_is_static']] > 0)).groupby(['nam', 'st', 'label']).size().items()}
json.dump(rep, open(f'/home/claude/v33/{WHICH}/bao_cao.json', 'w'), indent=1)
print(json.dumps(rep, indent=1) if WHICH == 'D' else f'H: {len(hashes)} tệp (không in phân bố chi tiết)')
