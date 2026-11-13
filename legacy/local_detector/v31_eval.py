"""v31_eval.py ynu|v24 — đánh giá đăng ký trước v31 (PREREG_v31_xshin.md).
Chỉ số theo seed (tập ngoài, kiến trúc đích, LightGBM, ngưỡng mặc định):
  M = macro F1 trung bình 5 kiến trúc đích; W = macro F1 kiến trúc tệ nhất;
  RD = recall mã độc liên kết động; FS = tỷ lệ báo nhầm lành tính liên kết tĩnh.
Kiểm định: Wilcoxon chính xác một phía (Local+XS-xa > đối chứng), Holm trong mỗi họ.
Xuất analysis/v31_<ds>_eval.json và in tóm tắt.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ds = sys.argv[1]
if ds == 'v24':
    import v24_core  # noqa: F401
import v16_core as C

AR = ['EM_386', 'EM_ARM', 'EM_MIPS', 'EM_PPC', 'EM_X86_64']
SEEDS = {'ynu': [3, 11, 29, 47, 151, 389, 757, 1201, 5101, 9001],
         'v24': [2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007]}[ds]
PILOT = {'ynu': [], 'v24': [2, 17, 43]}[ds]            # seed đã xem trong pilot (v29/v30)
BASE = {'ynu': {'N': 'v25_ynu_N', 'P': 'v25_ynu_P', 'CP': 'v26_ynu_CP'},
        'v24': {'N': 'v24_R4n', 'P': 'v24_R4p', 'CP': 'v31_v24_CP'}}[ds]
RES = Path('results'); MISSING = []
P, T, A = C.load_all('work/input'); ti, _ = C.external_indices(P, T, A)


def unit(tag, mode, s):
    v = {}
    for a in AR:
        f = RES / tag / 'units' / f'{s}_{a}' / f'{mode}__lgbm.npz'
        if not f.exists(): MISSING.append(str(f)); return None
        p = np.load(f)[f'{mode}|lgbm|external|pred'].astype(int); ix = ti[T.arch[ti] == a]
        y = T.y[ix]; st = T.is_static[ix]
        v[a] = dict(f1=f1_score(y, p, average='macro'),
                    rd=float(p[(y == 1) & ~st].mean()) if ((y == 1) & ~st).any() else np.nan,
                    fs=float(p[(y == 0) & st].mean()) if ((y == 0) & st).any() else np.nan)
    f1 = [v[a]['f1'] for a in AR]
    return dict(M=float(np.mean(f1)), W=float(np.min(f1)), RD=float(np.nanmean([v[a]['rd'] for a in AR])),
                FS=float(np.nanmean([v[a]['fs'] for a in AR])), per={a: v[a]['f1'] for a in AR})


def table(tag, mode):
    return {s: r for s in SEEDS if (r := unit(tag, mode, s)) is not None}


def wil1(d):
    d = np.asarray(d, float)
    if len(d) < 2 or not np.any(d != 0): return 1.0
    return float(stats.wilcoxon(d, alternative='greater', method='exact').pvalue)


def holm(ps):
    k = len(ps); order = np.argsort(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * ps[i])); adj[i] = run
    return adj.tolist()


def compare(a, b, q, seeds):
    ss = [s for s in seeds if s in a and s in b]
    d = [100 * (a[s][q] - b[s][q]) for s in ss]
    return dict(n=len(ss), diff=float(np.mean(d)) if d else None, wins=int(sum(x > 0 for x in d)), p=wil1(d))


out = dict(ds=ds, arms={}, tests={})
lines = [f'v31 {ds}: macro F1 ngoài (%), M/W trung bình qua seed; RD = recall mã độc động; FS = FPR lành tính tĩnh']
for arm in ('N', 'P', 'CP'):
    tag = f'v31_{ds}_{arm}'
    V = {'Local': table(BASE[arm], 'local'), 'LR-HIN': table(BASE[arm], 'hin2'),
         'XS-xa': table(tag, 'xa'), 'Local+XS-xa': table(tag, 'local_xa')}
    if arm == 'N':
        V.update({'Local+LR-HIN': table(tag, 'local_hin2'), 'Local+LR-HIN-xa': table(tag, 'local_hin2xa'),
                  'Local+XS': table(tag, 'local_xs'), 'LR-HIN-xa': table(tag, 'hin2xa')})
    lines.append(f'--- giao thức {arm}')
    out['arms'][arm] = {}
    for k, t in V.items():
        if not t: continue
        m = lambda q: 100 * float(np.mean([t[s][q] for s in t]))
        out['arms'][arm][k] = dict(n=len(t), M=m('M'), W=m('W'), RD=m('RD'), FS=m('FS'),
                                   per_seed={s: t[s] for s in t})
        lines.append(f'{k:17s} n={len(t):2d}  M {m("M"):5.1f}  W {m("W"):5.1f}  RD {m("RD"):5.1f}  FS {m("FS"):4.1f}')
    L, X = V['Local'], V['Local+XS-xa']
    fam = {}
    for q in ('M', 'W', 'RD'):
        fam[f'{arm}:{q}:all10'] = compare(X, L, q, SEEDS)
        if PILOT: fam[f'{arm}:{q}:unseen'] = compare(X, L, q, [s for s in SEEDS if s not in PILOT])
    fam[f'{arm}:FS_increase'] = compare(X, L, 'FS', SEEDS)      # an toàn: báo cáo mức tăng
    if arm == 'N':
        for k in ('Local+LR-HIN', 'Local+LR-HIN-xa', 'Local+XS'):
            fam[f'N:ablation_M_vs_{k}'] = compare(X, V[k], 'M', SEEDS)
    out['tests'].update(fam)

# Họ kiểm định chính (H1-H3: Corpus B, N, seed chưa xem) và họ ablation (H4) — Holm trong họ
fams = {'H1-H3': [f'N:{q}:unseen' if PILOT else f'N:{q}:all10' for q in ('M', 'W', 'RD')],
        'H4': [k for k in out['tests'] if k.startswith('N:ablation')],
        'phu': [f'{a}:{q}:all10' for a in ('P', 'CP') for q in ('M', 'W', 'RD')]}
for fname, keys in fams.items():
    keys = [k for k in keys if k in out['tests'] and out['tests'][k]['n'] >= 2]
    if not keys: continue
    adj = holm([out['tests'][k]['p'] for k in keys])
    lines.append(f'=== họ {fname} (Local+XS-xa − đối chứng, điểm %; thắng/seed; p một phía; p Holm)')
    for k, pa in zip(keys, adj):
        r = out['tests'][k]; r['p_holm'] = pa
        lines.append(f'  {k:32s} {r["diff"]:+6.2f}  {r["wins"]}/{r["n"]}  p={r["p"]:.4f}  p_holm={pa:.4f}')
r = out['tests'].get('N:FS_increase')
if r and r['n']: lines.append(f'An toàn: FPR lành tính tĩnh tăng {r["diff"]:+.2f} điểm (N, {r["n"]} seed)')
out['missing'] = len(MISSING); out['missing_examples'] = MISSING[:5]
lines.append(f'missing units: {len(MISSING)}')
Path('analysis').mkdir(exist_ok=True)
Path(f'analysis/v31_{ds}_eval.json').write_text(json.dumps(out, indent=1, default=float))
print('\n'.join(lines))
