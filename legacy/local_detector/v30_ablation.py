"""v30_ablation.py — ablation 2x2: thực thể (cũ: syscall+n-gram opcode của LR-HIN | mới: chuỗi+chữ ký hàm của XS-HIN)
x tóm tắt láng giềng (mọi tham chiếu | chỉ tham chiếu khác kiến trúc, 'xa'). Mọi biến thể đều ghép với Local.
Mode mới: hin2xa (LR-HIN, mọi đường meta-path chỉ trên tham chiếu khác kiến trúc; 15 đặc trưng), local_hin2xa.
Mode có sẵn: local_hin2 (v16), local_xs (v28), local_xa (v29).
"""
import numpy as np
import v29_xshin as X29
import v24_core, v16_core as C, v16_run as R


def hin2_xa(h, d, ix, chunk=3000):
    out = []
    QS_all = h._norm(d.calls[ix].astype(float) * h.idf_); QN_all = h._norm(h.sem_.transform(d, ix))
    for i in range(0, len(ix), chunk):
        QS, QN = QS_all[i:i + chunk], QN_all[i:i + chunk]
        other = d.arch[ix][i:i + chunk][:, None] != h.arch_[None, :]
        sS = QS @ h.RS_.T; sN = QN @ h.RN_.T
        out.append(np.column_stack([h._summ(sS, other), h._summ(sN, other), h._summ((sS + sN) / 2, other)]))
    return np.vstack(out)


def blocks_for(P, d, ix, hin, local, extra):
    b = X29.blocks_for(P, d, ix, hin, local, extra)
    if 'hin2' in R.HIN2_CACHE: b['Z2xa'] = hin2_xa(R.HIN2_CACHE['hin2'], d, ix)
    return b


def assemble(mode, b):
    if mode == 'hin2xa': return b['Z2xa']
    if mode == 'local_hin2xa': return np.column_stack([b['L'], b['Z2xa']])
    return X29.assemble(mode, b)


R.blocks_for = blocks_for; R.assemble = assemble
if __name__ == '__main__':
    R.main()
