#!/usr/bin/env python3
"""kaggle_v31.py — driver chạy v31 (XS-HIN) trên Kaggle hoặc máy Linux bất kỳ; chia phần, hạn giờ, chạy tiếp được.

  python kaggle_v31.py --corpus B --code <thư mục code v31> --data <thư mục có du_lieu_2020/2024/lt2020> \
         --phan 1 --so-phan 3 --jobs 4 --out /kaggle/working
  python kaggle_v31.py --corpus A ...                              (Corpus A = IoTPOT/YNU)
  python kaggle_v31.py --corpus B ... --liet-ke                     (chỉ in danh sách job của phần)
  python kaggle_v31.py --corpus B ... --danh-gia --tu <output 1> <output 2> ... <gói kết quả cũ>

Job = (seed x kiến trúc x nhóm mode) của v31_run.py. Corpus B: 350 job; Corpus A: 150 job.
Đánh giá cần thêm kết quả nhánh cũ trong --tu: Corpus B: ket_qua_v24_R4p-R4n-G1h_*.tar (Local/LR-HIN cho N, P);
Corpus A: ket_qua_v25ynu_*.tar (N, P) và ket_qua_v26ynu_claude.tar (CP).
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCHS = ['EM_386', 'EM_ARM', 'EM_MIPS', 'EM_PPC', 'EM_X86_64']
SEEDS = {'A': [3, 11, 29, 47, 151, 389, 757, 1201, 5101, 9001], 'B': [2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007]}
SEEDS['C'] = SEEDS['B']; SEEDS['H'] = SEEDS['B']
DS = {'A': 'ynu', 'B': 'v24', 'C': 'v24', 'H': 'v24'}
TAGP = {'A': 'v31_ynu', 'B': 'v31_v24', 'C': 'v32_c3', 'H': 'v33_H'}   # H = tập giữ kín v33 (PREREG_v33)   # C = tập xác nhận thứ ba (dev Corpus B, kiểm thử MalwareBazaar 2025-26)
G = {'G1': 'xa local_xa', 'G2': 'hin2xa local_hin2xa', 'G3': 'local_hin2', 'G4': 'local_xs', 'G5': 'local hin2'}
PLAN = {'B': [('N', ['G1', 'G2', 'G3', 'G4']), ('P', ['G1']), ('CP', ['G1', 'G5'])],
        'A': [('N', ['G1']), ('P', ['G1']), ('CP', ['G1'])],
        'C': [('N', ['G1', 'G5'])],
        'H': [('N', ['G5'])]}
CODE = ['v32_eval.py', 'run_uploaded_experiments.py', 'v16_core.py', 'v16_run.py', 'v24_core.py', 'cp_purge.py', 'v26_core.py',
        'v28_xshin.py', 'v29_xshin.py', 'v30_ablation.py', 'v31_run.py', 'v31_eval.py', 'MA_BAM_v31.txt']
WANT = dict(numpy='2.5.3', scipy='1.18.1', sklearn='1.9.1', lightgbm='4.7.0')


def jobs(corpus, seeds, nhom=None):
    ds = DS[corpus]; J = []
    for s in seeds:
        for arm, groups in PLAN[corpus]:
            tag = f'{TAGP[corpus]}_{arm}'
            for g in groups:
                if nhom and f'{arm}:{g}' not in nhom: continue
                for a in ARCHS:
                    J.append((f'{tag}_{s}_{a}_{g}', ['v31_run.py', ds, arm, '--protocol', 'loao', '--seeds', str(s),
                                                     '--architectures', a, '--modes'] + G[g].split() +
                              ['--models', 'lgbm', '--tag', tag]))
    return J


def kiem_ma_bam(run):
    bad = []
    for line in (run / 'MA_BAM_v31.txt').read_text().splitlines():
        h, f = line.split()
        if hashlib.sha256((run / f).read_bytes()).hexdigest() != h: bad.append(f)
    return bad


def chep_ket_qua(src_dirs, dst):
    n = 0
    for src in src_dirs:
        for f in glob.glob(os.path.join(src, '**', 'results', '*', 'units', '*', '*.npz'), recursive=True):
            if f.endswith('.tmp.npz'): continue
            t = dst / f[f.rindex('results' + os.sep):]
            if not t.exists(): t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, t); n += 1
        for tf in glob.glob(os.path.join(src, '**', 'ket_qua_*.tar'), recursive=True):
            with tarfile.open(tf) as tar:
                for m in tar.getmembers():
                    nm = m.name
                    if not (m.isfile() and nm.endswith('.npz') and not nm.endswith('.tmp.npz') and '/units/' in nm
                            and 'results/' in nm and '..' not in nm): continue
                    t = dst / nm[nm.index('results/'):]
                    if t.exists(): continue
                    t.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as fi, open(t, 'wb') as fo: shutil.copyfileobj(fi, fo)
                    n += 1
    return n


def guong_du_lieu(src, ten):
    """Gương thư mục dữ liệu trong /tmp (liên kết, không chép); nén lại bang_v5_dense.csv nếu Kaggle đã giải nén."""
    import gzip
    dst = Path(os.environ.get('TMPDIR', '/tmp')) / f'du_lieu_guong_{ten}'
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    for e in src.iterdir():
        if not e.is_dir(): (dst / e.name).symlink_to(e); continue
        (dst / e.name).mkdir()
        for f in e.iterdir(): (dst / e.name / f.name).symlink_to(f)
        gz, csv_ = dst / e.name / 'bang_v5_dense.csv.gz', e / 'bang_v5_dense.csv'
        if not gz.exists() and csv_.exists():
            with open(csv_, 'rb') as fi, gzip.open(gz, 'wb', compresslevel=1) as fo: shutil.copyfileobj(fi, fo, 1 << 22)
    return dst


def dong_goi(out, run, ten, muc):
    tf = out / f'ket_qua_{ten}.tar'
    with tarfile.open(tf, 'w') as tar:
        for m in muc:
            if (run / m).exists(): tar.add(run / m, arcname=f'{run.name}/{m}')
    shutil.rmtree(run, ignore_errors=True)
    print('đã gói:', tf.name, f'{tf.stat().st_size / 1e6:.1f} MB', flush=True)


def tim(root, pattern):
    c = sorted(glob.glob(os.path.join(root, '**', pattern), recursive=True))
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['A', 'B', 'C', 'H'], required=True)
    ap.add_argument('--data2', help='Corpus C: thư mục chứa du_lieu_2024 của tập C3 (thay du_lieu_2024 của --data)')
    ap.add_argument('--code', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='/kaggle/working'); ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--phan', type=int, default=1); ap.add_argument('--so-phan', type=int, default=1)
    ap.add_argument('--gio', type=float, default=10.0); ap.add_argument('--gio-cung', type=float, default=11.3)
    ap.add_argument('--seeds', nargs='+', type=int); ap.add_argument('--tu', nargs='*', default=[])
    ap.add_argument('--nhom', nargs='+', help='chỉ chạy các nhóm arm:G, ví dụ N:G1 N:G4 P:G1 CP:G1')
    ap.add_argument('--liet-ke', action='store_true'); ap.add_argument('--danh-gia', action='store_true')
    a = ap.parse_args(); t0 = time.time()
    if not (Path(a.data) / 'du_lieu_2020').is_dir():
        d = tim(a.data, 'du_lieu_2020')
        if not d: sys.exit(f'Không tìm thấy du_lieu_2020 trong {a.data}')
        a.data = str(Path(d).parent)
    tok_name = f'tokens_{a.corpus}.jsonl.gz'
    out = Path(a.out).resolve(); run = out / f'run_v31_{a.corpus}'; run.mkdir(parents=True, exist_ok=True)
    for f in CODE: shutil.copy2(Path(tim(a.code, f)), run / f)
    if not (run / tok_name).exists():
        tok = tim(a.code, tok_name)
        if tok and Path(tok).is_file():
            (run / tok_name).symlink_to(Path(tok).resolve())
        else:
            # Kaggle tự giải nén .gz khi tải lên: tokens_X.jsonl có thể là tệp, hoặc thư mục chứa một tệp -> nén lại
            raw = tim(a.code, f'tokens_{a.corpus}.jsonl')
            if raw and Path(raw).is_dir():
                fs = [p for p in Path(raw).rglob('*') if p.is_file()]
                raw = str(fs[0]) if len(fs) == 1 else None
            if not raw: sys.exit(f'Không tìm thấy {tok_name} hoặc tokens_{a.corpus}.jsonl trong {a.code}')
            import gzip
            h = hashlib.sha256()
            with open(raw, 'rb') as fi, gzip.open(run / tok_name, 'wb', compresslevel=1) as fo:
                while (b := fi.read(1 << 22)): h.update(b); fo.write(b)
            print('nén lại token từ', raw, '| sha256 nội dung (chưa nén):', h.hexdigest(), flush=True)
    print('dữ liệu:', a.data, '| token:', run / tok_name, flush=True)
    (run / 'work').mkdir(exist_ok=True); link = run / 'work' / 'input'
    if link.is_symlink(): link.unlink()
    g = guong_du_lieu(Path(a.data).resolve(), a.corpus)
    if a.corpus in ('C', 'H'):
        d2 = tim(a.data2 or '', 'du_lieu_2024') if a.data2 else None
        if not d2: sys.exit('Corpus C/H cần --data2 trỏ tới thư mục có du_lieu_2024 của tập kiểm thử')
        src2 = guong_du_lieu(Path(d2).parent.resolve(), 'C3')
        shutil.rmtree(g / 'du_lieu_2024'); shutil.copytree(src2 / 'du_lieu_2024', g / 'du_lieu_2024', symlinks=True)
    link.symlink_to(g, target_is_directory=True)
    J = jobs(a.corpus, a.seeds or SEEDS[a.corpus], a.nhom)
    mine = [j for i, j in enumerate(J) if i % a.so_phan == a.phan - 1]
    print(f'Corpus {a.corpus}: tổng {len(J)} job; phần {a.phan}/{a.so_phan}: {len(mine)} job', flush=True)
    if a.liet_ke:
        for k, cmd in mine: print(k, '|', ' '.join(cmd))
        return
    import numpy, scipy, sklearn, lightgbm
    ver = dict(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__,
               python=sys.version.split()[0], cpu=os.cpu_count())
    bad = kiem_ma_bam(run)
    print('môi trường', ver, '| mã băm:', 'OK' if not bad else f'SAI {bad}', flush=True)
    if bad: sys.exit('Mã băm code không khớp MA_BAM_v31.txt — dừng.')
    lech = {k: ver[k] for k in WANT if ver[k] != WANT[k]}
    if lech: sys.exit(f'Sai phiên bản thư viện: {lech} (cần {WANT}) — dừng.')
    n = chep_ket_qua(a.tu, run)
    if n: print('chép', n, 'đơn vị từ kết quả trước', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', XS_TOKENS=tok_name)
    py = sys.executable
    if a.danh_gia:
        os.chdir(run)
        r = subprocess.run([py, 'v32_eval.py' if a.corpus == 'C' else 'v31_eval.py', DS[a.corpus]], env=env, capture_output=True, text=True)
        txt = r.stdout + ('\n--- stderr ---\n' + r.stderr if r.stderr.strip() else '')
        (out / f'analysis_v31_{a.corpus}_tom_tat.txt').write_text(txt); print(txt, flush=True)
        if (run / 'analysis').is_dir(): shutil.copytree(run / 'analysis', out / f'analysis_v31_{a.corpus}', dirs_exist_ok=True)
        return
    pre = ('import os;os.environ.setdefault("XS_TOKENS","%s");import sys;sys.argv=["x","%s","N"];'
           'import v30_ablation,v28_xshin as X,v16_core as C;P,T,A=C.load_all("work/input");print(len(P.y),len(T.y),len(A.y));'
           'cov={n:sum(str(h) in X.TOK for h in d.hash)/len(d.hash) for n,d in (("P",P),("T",T),("A",A))};print("độ phủ token",cov);'
           'sys.exit("THIẾU TOKEN cho tập phát triển/tập ngoài -- dừng") if min(cov["P"],cov["T"])<0.99 else None') % (tok_name, DS[a.corpus])
    subprocess.run([py, '-c', pre], cwd=run, env=env, check=True)      # tạo cache dữ liệu một lần
    (run / 'logs').mkdir(exist_ok=True); trang = {}

    def chay(j):
        key, cmd = j
        if time.time() - t0 > a.gio * 3600: return key, 'chua_chay', 0.0
        t1 = time.time(); con = a.gio_cung * 3600 - (t1 - t0)
        if con < 300: return key, 'chua_chay', 0.0
        with open(run / 'logs' / f'{key}.log', 'a') as lg:
            try:
                r = subprocess.run([py] + cmd, cwd=run, env=env, stdout=lg, stderr=subprocess.STDOUT, timeout=con)
                st = 'xong' if r.returncode == 0 else 'loi'
            except subprocess.TimeoutExpired:
                st = 'het_gio'
        return key, st, (time.time() - t1) / 60

    tt = out / f'trang_thai_v31_{a.corpus}_phan{a.phan}of{a.so_phan}.json'
    with ThreadPoolExecutor(a.jobs) as ex:
        for fut in as_completed([ex.submit(chay, j) for j in mine]):
            k, st, m = fut.result(); trang[k] = dict(trang_thai=st, phut=round(m, 1))
            print(f'{st:9s} {k} ({m:.1f} phút; đã {(time.time() - t0) / 3600:.2f} giờ)', flush=True)
            tt.write_text(json.dumps(dict(moi_truong=ver, jobs=trang), indent=1))
    for f in glob.glob(str(run / 'results' / '**' / '*.tmp.npz'), recursive=True): os.remove(f)
    cnt = {s: sum(v['trang_thai'] == s for v in trang.values()) for s in ('xong', 'loi', 'het_gio', 'chua_chay')}
    if cnt['loi'] or cnt['het_gio'] or cnt['chua_chay']:
        print('CÒN VIỆC: chạy lại đúng lệnh này, thêm output của phiên này vào --tu để làm nốt.', flush=True)
    print('KẾT THÚC phần', a.phan, cnt, f'{(time.time() - t0) / 3600:.2f} giờ', flush=True)
    cache = run / 'work' / 'data_cache.pkl'
    if cache.exists(): cache.unlink()
    if link.is_symlink(): link.unlink()
    dong_goi(out, run, f'v31_{a.corpus}_phan{a.phan}of{a.so_phan}_{time.strftime("%m%d_%H%M", time.localtime(t0))}', ['results', 'logs'])


if __name__ == '__main__':
    main()
