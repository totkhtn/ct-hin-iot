# PREREG v33 — CT-HIN (HIN chuyển nạp theo chiến dịch) trên tập GIỮ KÍN theo thời gian

## 1. Lịch sử (minh bạch)
- Phương pháp hình thành qua pilot trên C3 (v33_ct_pilot) và giai đoạn phát triển trên tập D:
  W2025 (MalwareBazaar 2025), W26Q1, W26Q2 (MalwareBazaar 2026 Q1–Q2) + lành tính phát triển. Kết quả D: `ket_qua_phat_trien_D.txt`.
- Nhóm lành tính tĩnh nhỏ (`lanh_tinh_tinh_nho`) đã được xem tổng tỷ lệ báo nhầm trong pilot; được chia đôi theo băm
  (sha256 của file_hash, chẵn/lẻ): nửa lẻ (5.220 tệp) chỉ dùng cho tập giữ kín H.
- Tập giữ kín H chưa từng được chấm điểm hay đánh giá trước khi đóng băng tệp này.

## 2. Tập giữ kín H (cố định)
- Mã độc: MalwareBazaar first_seen 2026 Q3–Q4, CHƯA dùng trong C3 (13.466 trong danh sách; sau lọc kiến trúc/trùng).
- Lành tính: nửa giữ kín của `lanh_tinh_tinh_nho` (5.220, liên kết tĩnh) + 8.000 tệp phần dư 2020 chưa từng dùng (seed 1).
- Tổng sau dựng: 26.681 tệp (`dung_v33.py H`). Token: `H/tokens_ct.jsonl.gz`.

## 3. Huấn luyện và tiên nghiệm (cố định)
Corpus B 2020, LOAO không purge, Local LightGBM (pipeline v16, lưới + inner CV); 10 seed: 2, 17, 43, 101, 211, 401, 809, 1601, 3203, 6007.
Tiên nghiệm của mỗi tệp = điểm Local của mô hình giữ lại kiến trúc của tệp.

## 4. Phương pháp (cấu hình CHỌN TRÊN D, cố định)
CT-HIN: một lô chuyển nạp = toàn bộ H; thực thể = chuỗi chuẩn hóa (s) + IP + tên miền + URL thô; IDF trong lô;
bỏ thực thể df < 2 hoặc df > 2% lô; kNN cosine k = 20; lan truyền nhãn alpha = 0,9, 20 vòng; ngưỡng 0,5.
(`ct_v33.py s-ip-dom-url,0.02,20,0.9`)

## 5. Đối chứng
(i) Local; (ii) kNN đặc trưng Local + cùng lan truyền (alpha 0,8; không dị thể); (iii) LR-HIN (hin2).

## 6. Giả thuyết (Wilcoxon một phía, cặp theo seed, 10 seed)
- **H1 (họ chính, Holm 4 kiểm định):** CT-HIN so với Local: M lớn hơn; W lớn hơn; RD (recall mã độc động) lớn hơn;
  FPR (lành tính) nhỏ hơn. Xác nhận khi p_Holm < 0,05 và thắng ≥ 8/10 seed cho M và FPR.
- **H2:** M(CT-HIN) > M(kNN đặc trưng Local) — giá trị riêng của thực thể dị thể.
- **H3 (mô tả):** FPR trên nửa giữ kín của nhóm lành tính tĩnh nhỏ; kết quả theo kiến trúc; theo họ mã độc.

## 7. Đóng băng
- Giờ đóng băng: 2026-11-11 22:01 (UTC+7), xác nhận bởi tác giả trước khi chấm điểm H
- sha256: ct_v33.py, v33_eval_H.py (script kiểm định), dung_v33.py, H/tokens_ct.jsonl.gz, H/work/input/du_lieu_2024/bang_v5_dense.csv.gz (ghi trong SHA_v33.txt)
