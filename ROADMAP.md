# Lộ trình 5 giai đoạn — UAV bám người (PX4)

**Cập nhật:** 2026-10-26

## Tổng quan 5 giai đoạn

| # | Giai đoạn | Trạng thái | Người phụ trách |
|---|---|---|---|
| 1 | Môi trường và PX4 SITL | **MỘT PHẦN** | bạn |
| 2 | Thị giác và Re-ID | **ĐANG LÀM** | bạn (Claude thiết kế thuật toán, Gemini viết mã) |
| 3 | Bring-up STM32H7 | **CHƯA BẮT ĐẦU** | thầy phụ trách phần cứng |
| 4 | Offboard và tránh vật cản | **CHƯA BẮT ĐẦU** | chưa xác định |
| 5 | Bay thử và hiệu chỉnh | **CHƯA BẮT ĐẦU** | chưa xác định |

## BẠN ĐANG Ở ĐÂY

**BẠN ĐANG Ở ĐÂY: Giai đoạn 2 — Thị giác và Re-ID** (ĐANG LÀM)

## Chi tiết từng giai đoạn

### Giai đoạn 1 — Môi trường và PX4 SITL

- **Trạng thái:** MỘT PHẦN
- **Người phụ trách:** bạn
- **Checklist:**
  - [x] SITL + Gazebo + takeoff.py chạy được
  - [ ] ROS2 (chưa xác nhận)
  - [ ] Gazebo có vật thể động (chưa xác nhận)

### Giai đoạn 2 — Thị giác và Re-ID

- **Trạng thái:** ĐANG LÀM
- **Người phụ trách:** bạn (Claude thiết kế thuật toán, Gemini viết mã)
- **Ước lượng hoàn thành Giai đoạn 2: 68-70%** — đây là ước lượng theo tiêu chí hoàn thành đề xuất, không phải phép đo. _(estimate by proposed completion criteria, not a measurement)_
- **Checklist:**
  - [x] YOLOv8n + BoT-SORT + OSNet + máy trạng thái (15.44 FPS, GTX 1650)
  - [x] Ground truth làm lại bằng mắt (5_manual.csv, ±30 frame)
  - [x] Kiểm tra bằng mắt: 90 frame mẫu (44 đúng, 0 sai, 46 không có ô xanh) + ảnh các đoạn bám
  - [x] Ghi log lý do từ chối ứng viên (đã chạy, regression PASS)
  - [x] Thư mục gallery_debug riêng mỗi lần chạy (outputs/gallery_debug/<timestamp>_<out>)
  - [ ] Thay GIVE_UP bằng stale/invalid/reset có kiểm soát
  - [x] Đầu ra thống nhất valid/stale/box/confidence
  - [ ] Cải thiện độ phủ góc sau (hiện 25.6%)
  - [ ] Calibration độc lập với tracker
  - [ ] Kiểm tra nhiều video, có người ăn mặc giống nhau
  - [ ] Node ROS2 (chưa rõ thầy có bắt buộc)

### Giai đoạn 3 — Bring-up STM32H7

- **Trạng thái:** CHƯA BẮT ĐẦU
- **Người phụ trách:** thầy phụ trách phần cứng
- **Checklist:**
  - [ ] Biên dịch và nạp firmware PX4 cho bo STM32H7
  - [ ] Bench test: hiệu chuẩn IMU, la bàn, ánh xạ PWM
  - [ ] Cấu hình failsafe, RTL, pin yếu

### Giai đoạn 4 — Offboard và tránh vật cản

- **Trạng thái:** CHƯA BẮT ĐẦU
- **Người phụ trách:** chưa xác định
- **Checklist:**
  - [ ] Kết nối máy tính phụ với bo qua MAVLink
  - [ ] Gửi vận tốc qua Offboard để mô phỏng bám mục tiêu
  - [ ] Local grid map và quỹ đạo vòng tránh vật cản

### Giai đoạn 5 — Bay thử và hiệu chỉnh

- **Trạng thái:** CHƯA BẮT ĐẦU
- **Người phụ trách:** chưa xác định
- **Checklist:**
  - [ ] Lắp ráp khung, động cơ, cấp nguồn độc lập
  - [ ] Bay thủ công kiểm tra ổn định và chống nhiễu gió
  - [ ] Bay bám sát tự trị ngoài trời, có người sẵn sàng chiếm quyền điều khiển

## Chờ duyệt

- [ ] Công cụ gán nhãn mục tiêu bằng mắt, độc lập tracker (đã duyệt, chờ lệnh làm - STEP 3)
- [ ] Thay GIVE_UP bằng stale/invalid/reset có kiểm soát (CHƯA duyệt - chờ video mới có mất dấu dài)
- [ ] Thầy chốt ROS2 có bắt buộc không

## Sự thật đã kiểm chứng

- Cosine frame 904 = 0.5142 (replay khớp CSV) — nguồn: STATUS.md, Audit - TASK 1
- Sàn anchor 0.60 chặn người sai — nguồn: STATUS.md, frame 904 sim_anchor 0.5153 < 0.60 và không vẽ ô xanh
- 5_manual.csv thay 5.csv — nguồn: STATUS.md, mục 7
- PX4 VTE có timeout 3 mốc và reset thay vì bỏ cuộc (đọc từ mã nguồn)
- Frame 1062: mục tiêu KHÔNG có trong khung, không vẽ ô xanh; người điểm cao nhất 0.7982 (num_people 12) là người khác và bị bỏ qua đúng — nguồn: người dùng kiểm tra bằng mắt, 1 video (5.mp4), frame mẫu
- ID 435 tại frame 2195 và 2200 là ĐÚNG mục tiêu — nguồn: người dùng kiểm tra bằng mắt, 1 video, frame mẫu
- ID 468 tại frame 2458 là ĐÚNG mục tiêu — nguồn: người dùng kiểm tra bằng mắt, 1 video, frame mẫu
- Tại 2195 gate=nconfirm_fail (consec 4), 2200 gate=accept_fail (PARTIAL, 0.8173 < 0.83), 2458 gate=nconfirm_fail (consec 6): ứng viên ĐÚNG bị giữ lại bởi cổng accept/consec, không phải khóa nhầm — nguồn: log tracking_log_20261025_212543.csv + nhãn mắt; chỉ 3 frame

## Nhật ký thay đổi (mới nhất trước)

### 2026-10-26 (Step 2c unified output)

- **Đã làm:** STEP 2(c): thêm 3 cột valid,stale,age vào cuối CSV (22 -> 25 cột). valid = 1 đúng khi target_box is not None (cùng điều kiện vẽ ô xanh); stale = 1 khi valid=0 và đã từng có frame valid trước đó (0 trước frame valid đầu tiên); age = số frame liên tiếp kể từ frame valid gần nhất (0 khi valid, rỗng trước frame valid đầu tiên). Chỉ ghi log, không đưa vào quyết định. Backup trước khi sửa: backups/follow_target_2026-10-26_023351_pre_unified_output.py.
- **Bằng chứng:** Chạy lại baseline --out outputs/partE_baseline_step2c_unified.mp4: 2698 frame, 172.39 s, 15.65 FPS, log logs/tracking_log_20261026_023427.csv (25 cột). Regression PASS: 9 cột gốc giống hệt 2698/2698 dòng (0 dòng khác); 9 mốc chuyển trạng thái giống hệt. Cross-check toàn bộ 2698 dòng: valid khớp điều kiện hộp xanh 2698/2698 (0 lệch); valid=1 1361, valid=0 1337; stale/age 0 lỗi; tuổi lớn nhất 848; 7 khoảng stale. Kiểm tra pixel video: mọi frame valid=0 mẫu có 0 pixel xanh ngoài HUD, mọi frame valid=1 có hàng nghìn pixel xanh ngoài HUD. Không đổi ngưỡng/nhánh quyết định.
- **Chờ duyệt:** Chờ duyệt để làm STEP 3 (label_target.py + ground_truth/labels_5_eye.csv). STEP 4 chưa duyệt.

### 2026-10-25 (Step 1b gallery), tái kiểm tra 2026-10-26

- **Đã làm:** STEP 1(b): gallery_debug riêng cho mỗi lần chạy -> outputs/gallery_debug/<YYYYMMDD_HHMMSS>_<out-video-stem>/, không bao giờ ghi vào thư mục của lần chạy cũ (thêm hậu tố _2, _3 nếu trùng tên). Backup trước khi sửa: backups/follow_target_2026-10-25_220940_pre_gallery.py.
- **Bằng chứng:** Chạy lại baseline với --out outputs/partE_baseline_step1_gallery.mp4: 2698 frame, 171.22 s, 15.76 FPS; log logs/tracking_log_20261025_221109.csv. Regression PASS: 9 cột gốc giống hệt 2698/2698 dòng (0 dòng khác); chuyển trạng thái giống hệt (188 TRACKING, 905 LOST, 1751, 2110, 2460, 2513, 2581, 2625, 2684); danh sách nghi khóa nhầm giống hệt (904, 2510, 2605, 2608, 2624). Thư mục cũ: 162 file, md5 giống hệt trước/sau (162 OK, 0 lệch), không ghi thêm file ở cấp ngoài. Thư mục mới: 73 file (10 anchor + 63 dyn). Không đổi ngưỡng/nhánh quyết định. Tái kiểm tra độc lập 2026-10-26 (backup backups/follow_target_2026-10-26_022137_pre_step1_verify.py, không sửa mã): log logs/tracking_log_20261026_022150.csv, 2698 frame, 176.16 s, 15.32 FPS; 9 cột gốc giống hệt 2698/2698 dòng (0 dòng khác), 9 mốc chuyển trạng thái giống hệt; thư mục lần chạy cũ 73/73 file md5 OK, cấp ngoài gallery_debug vẫn 162 file md5 không đổi; thư mục mới outputs/gallery_debug/20261026_022150_partE_baseline_step1_gallery/.
- **Chờ duyệt:** Chờ duyệt để làm STEP 2 (đầu ra thống nhất valid/stale/box/confidence/age). STEP 3 (label_target.py) xếp sau. STEP 4 (GIVE_UP) chưa duyệt.

### 2026-10-25 (Stage 2 logging step)

- **Đã làm:** Thêm 13 cột chẩn đoán vào CSV của follow_target.py (chỉ logging; backup backups/follow_target_2026-10-25_pre_logging.py trước khi sửa), chạy lại baseline và đối chiếu với log cũ; ghi nhận xác nhận bằng mắt của bạn (frame 1062, ID 435 tại 2195/2200, ID 468 tại 2458).
- **Bằng chứng:** 9 cột gốc GIỐNG HỆT 2698/2698 dòng; chuyển trạng thái giống hệt (188 TRACKING, 905 LOST, 1751, 2110, 2460, 2513, 2581, 2625, 2684); log logs/tracking_log_20261025_212543.csv (22 cột), 170.36 s, 15.84 FPS; STATUS.md mục 'Stage 2 logging step'. Ước lượng Giai đoạn 2 giữ 68-70% (estimate theo tiêu chí, không phải phép đo). Không đổi ngưỡng/hằng số/quyết định.
- **Chờ duyệt:** Chờ duyệt 4 đề xuất: (a) công cụ gán nhãn mục tiêu bằng mắt, độc lập tracker (file mới); (b) gallery_debug riêng mỗi lần chạy; (c) đầu ra thống nhất valid/stale/box/confidence/age; (d) thay GIVE_UP bằng stale -> invalid -> reset có kiểm soát. Chưa triển khai mục nào.

### 2026-10-25

- **Đã làm:** Tạo hệ thống báo cáo lộ trình: roadmap_data.json (nguồn dữ liệu duy nhất) + update_roadmap.py sinh ROADMAP.md tiếng Việt và ảnh outputs/progress_roadmap.png (1920x1080).
- **Bằng chứng:** python3 update_roadmap.py chạy thành công; ROADMAP.md và outputs/progress_roadmap.png được tạo (xem ls -l). Kiểm tra tự động: 105 đối tượng chữ, 0 ngoài khung, 0 tràn ô, 0 chồng lấn. Checklist giai đoạn 3-5 do bạn cung cấp, đã đưa vào nguyên văn.
- **Chờ duyệt:** Chờ duyệt: 5 mục trong danh sách Chờ duyệt.

---

_Phần trăm là ước lượng. Chỉ kiểm tra trên 1 video (5.mp4); chưa được kết luận 'không có khóa nhầm'._
