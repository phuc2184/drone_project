# BÁO CÁO TIẾN ĐỘ VÀ ĐIỀU CHỈNH HƯỚNG NGHIÊN CỨU — 09/11/2026

**Đề tài:** XÂY DỰNG HỆ THỐNG NHẬN THỨC MÔI TRƯỜNG CHO DRONE TỰ HÀNH  
**Ngày lập:** 09/11/2026  
**Trạng thái:** Chuyển ưu tiên sang PX4 SITL + Gazebo và tránh vật cản; tạm dừng phát triển Person ReID.  
**Nguồn:** Tổng hợp nhật ký trao đổi, các báo cáo Ling/Codex do người nghiên cứu cung cấp và tình trạng tài liệu GitHub tại ngày lập. **Không phải kiểm toán trực tiếp thư mục Ubuntu hiện tại.**

> **Quy ước:** VERIFIED = có bằng chứng từ một lần kiểm toán được thuật lại hoặc thao tác trực tiếp xác nhận; REPORTED = thông tin do tác nhân/người nghiên cứu báo cáo; OBSERVED = người nghiên cứu tự quan sát; UNKNOWN = chưa có bằng chứng đủ để xác lập. Một test PASS không đồng nghĩa thuật toán nhận diện PASS.

## 1. Tóm tắt điều hành

Theo ý kiến hướng dẫn mới của giảng viên trong buổi trao đổi sáng 09/11/2026, công việc trước mắt là **làm chủ mô phỏng drone trong môi trường 3D, bố trí vật cản, thiết lập điều khiển bay và cảm biến**, sau đó mới tích hợp Deep Learning/nhận diện vào hệ thống. Chưa triển khai bay thật.

Hai nhánh kết quả đã hình thành:

1. **Nhánh AI perception:** YOLOv8n + BoT-SORT + OSNet Person ReID, giữ ID mục tiêu logic, tạo video/log và hạ tầng evaluator/human review. Người nghiên cứu nhiều lần xem video thử nghiệm và **chưa quan sát thấy ghim nhầm người**. Đây là OBSERVED, **không phải** false-lock rate đã được kiểm chứng bằng đánh giá danh tính độc lập.
2. **Nhánh PX4/SITL:** Đã có thao tác cất cánh/hạ cánh và thực hành điều khiển cơ bản trong mô phỏng. Các hoạt động QGroundControl/Go To/mission waypoint được nhắc lại trong tiến độ hội thoại nhưng cần xác nhận bằng nhật ký SITL hoặc bản ghi thực hành trước khi dùng làm bằng chứng nghiệm thu.

**Quyết định tại thời điểm lập:** Đóng băng thay đổi chức năng AI/ReID và evaluator, ưu tiên đào tạo và xây dựng bài thử PX4–Gazebo có vật cản. Các lỗi evaluator chưa được nghiệm thu phải được bảo lưu, không xóa hay ghi đè bằng kết luận PASS.

## 2. Trạng thái từng hạng mục

| Hạng mục | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| PX4 SITL cơ bản | Đã thực hành | Đã chạy `commander takeoff`, `commander land`, độ cao cất cánh mặc định 2,5 m theo log hội thoại |
| Gazebo / drone 3D | Đã tiếp xúc | Cần xác nhận phiên bản Gazebo, world, vehicle, model/cảm biến đang cài trên Ubuntu |
| QGroundControl / waypoint | REPORTED đã thực hành | Cần tái hiện mission và lưu log/screenshot để xác nhận |
| Thế giới có vật cản | Chưa nghiệm thu | Chưa có bằng chứng hoàn thành bài thử world có tường/vật cản |
| Depth camera / LiDAR / distance sensor | Chưa nghiệm thu | Chưa có bằng chứng sensor stream được kiểm tra đúng |
| Python/MAVSDK hoặc ROS 2 điều khiển | Chưa nghiệm thu | Chưa có kết quả tự động đọc telemetry, gửi lệnh và kiểm tra failsafe |
| Dừng an toàn / tránh vật cản | Chưa thực hiện đủ | Chưa có metric va chạm, khoảng cách tối thiểu hoặc tỷ lệ hoàn thành |
| Person detection/tracking/ReID | Đã triển khai, tạm dừng | Chỉ có kết quả video exploratory, không có held-out identity metric |
| Persistent logical Target ID | PASS trong phạm vi HUD theo kiểm toán trước | Không chứng minh độ chính xác Person ReID |
| Gate A Evaluator v3.1 | REPORTED triển khai COMPLETE, **chưa nghiệm thu độc lập phiên bản v3.1** | Ling báo 450/450 tests; Codex đã REVISE phiên bản v3 trước đó |
| Human Review v3.1 | REPORTED hoạt động kỹ thuật, chưa có nhãn thực | Session demo/kiểm thử không thay thế nhãn người độc lập |
| Gate A scientific validation | **NOT ESTABLISHED** | Chưa có nhãn độc lập/held-out đủ điều kiện |
| B0 Multi-Class | REVISE tại lần kiểm toán được cung cấp | Trust boundary bằng chứng cần xử lý khi quay lại |
| B1 Multi-Class / AI–PX4 integration | Chưa nghiệm thu | Không mặc định đã tích hợp |

## 3. Những vấn đề cần sửa hoặc tiếp tục xác minh — nhưng tạm hoãn

### 3.1. Person ReID / Gate A

- Không thay threshold, gallery, N_CONFIRM, model weights hoặc logic nhận diện khi chưa có một thí nghiệm được phê duyệt.
- Khi mở lại nhánh Gate A, trước hết phải **Codex kiểm toán độc lập Evaluator v3.1**. Ling đã báo sửa các lỗi P0/P1 của v3 (metric 0/0, episode qua UNKNOWN/partial, review video alignment, binding, box extraction, report generator...), nhưng đây chưa phải nghiệm thu độc lập.
- Xác nhận Human Review với video thật, đặc biệt những bbox mơ hồ và ràng buộc nguồn video/phiên gắn nhãn.
- Hoàn thiện protocol đánh giá mù, đủ nhãn danh tính người độc lập; không dùng `ground_truth/5.csv` (có tính circular) để công bố accuracy; `5_manual.csv` còn giới hạn provenance.
- Video 5 chỉ là exploratory/diagnostic; giữ nguyên các video held-out 1/2/3 cho protocol được khóa trước.
- Thống nhất định nghĩa *correct identity lock*, *false lock*, *reacquisition*, *tracking loss* và *latency*; không diễn giải `0 eligible judgments` thành `0% wrong lock`.
- Lưu trữ backup, hashes, change history, diff, artifacts và trạng thái Git trước khi chuyển nhánh công việc.

### 3.2. Multi-object (B0/B1)

- Khi quay lại: giải quyết lỗi provenance/trust-boundary của bằng chứng identity mà Codex từng nêu; re-audit B0, sau đó mới đưa B1 vào tích hợp.
- Không dùng chức năng multi-object chưa nghiệm thu để điều khiển drone.

### 3.3. Mô phỏng/PX4 — ưu tiên mới

- Xác minh chính xác phiên bản PX4, Gazebo, QGroundControl, ROS 2/MAVSDK, target X500 và thư mục `PX4-Autopilot`.
- Kiểm tra quy trình khởi động/tắt SITL, thao tác GUI, điều khiển cơ bản và log telemetry.
- Thiết lập một world chứa vật cản tĩnh với kích thước/tọa độ có thể tái lập.
- Tích hợp depth camera hoặc distance sensor tương thích; xác nhận dữ liệu đo bằng bài thử khoảng cách.
- Thực hiện **dừng trước vật cản** trong mô phỏng trước khi xây tránh vật cản chủ động.
- Thử nghiệm điều khiển ngoài PX4 với cơ chế mất kết nối, timeout, failsafe và giới hạn vận tốc.
- Chỉ triển khai path planning/avoidance khi dữ liệu cảm biến và vòng điều khiển ổn định.

## 4. Kiến trúc đích cho mô phỏng

```text
Gazebo 3D world (obstacles, dynamics, sensor simulation)
      | sensor/vehicle simulation
      v
PX4 SITL <---- MAVLink / ROS 2 / MAVSDK ----> Companion application
   |                                           | perception / obstacle mapping
   v                                           | planning and safety supervisor
QGroundControl                                 v
                                         Setpoints / stop / reroute
```

**Phân vai:**
- **Gazebo:** thế giới, vật lý, vật cản, camera/depth/LiDAR mô phỏng.
- **PX4:** bộ điều khiển bay, ước lượng trạng thái, thực thi setpoint và failsafe.
- **QGroundControl:** quan sát, cấu hình và lập mission.
- **Companion app (Python/ROS 2/MAVSDK):** tiếp nhận dữ liệu cảm biến, tính toán nhận thức và quyết định điều hướng.
- **Deep Learning:** có thể đưa vào perception (phát hiện/phân đoạn/ước lượng vật cản) sau khi pipeline mô phỏng cơ bản chạy ổn định. **Reinforcement Learning** là một hướng nghiên cứu điều khiển/học chính sách khác, không phải yêu cầu bắt buộc hiện tại.

Không mặc định đưa mô hình Deep Learning trực tiếp vào firmware PX4. Phần AI có thể chạy trên máy tính đồng hành, giao tiếp với PX4 qua giao diện được hỗ trợ. Cần xác nhận version-specific compatibility trước khi chốt giải pháp.

## 5. Kế hoạch học và thực hành theo thứ tự ưu tiên

| Giai đoạn | Công việc | Tiêu chí hoàn thành và bằng chứng |
|---|---|---|
| M1 — Khởi động và thao tác | PX4 SITL + Gazebo + QGC, X500, camera/Gazebo navigation, takeoff/land/RTL | Chạy lại độc lập, ảnh màn hình, terminal log, cấu hình phiên bản |
| M2 — World 3D có vật cản | World tường/hộp/cột/cây; đặt vật thể và xác minh tọa độ | File world/SDF, ảnh trước/sau, các kích thước và vị trí |
| M3 — Cảm biến | Depth/distance sensor; xác minh đọc dữ liệu trên các khoảng cách chuẩn | Sensor topics/telemetry, thử nghiệm 3–5 khoảng cách, sai số đo |
| M4 — Lập trình điều khiển | Python MAVSDK hoặc ROS 2, offboard setpoint, timeout/failsafe | Script tái lập, log lệnh, kiểm thử mất dữ liệu/kết nối |
| M5 — Dừng an toàn | Phát hiện vật cản trước mặt, giảm tốc/dừng ở ngưỡng an toàn | Số lần va chạm, khoảng cách dừng tối thiểu, log cảm biến/điều khiển |
| M6 — Tránh vật cản | Lập đường đi vòng vật cản tĩnh; nhiều kịch bản | Success rate, quãng đường, thời gian, khoảng cách tối thiểu |
| M7 — AI perception | Thử Deep Learning trên dữ liệu cảm biến mô phỏng, đối chiếu baseline không AI | Kết quả so sánh, latency, FPS, robustness |
| M8 — ReID integration | Khôi phục nhánh nhận diện, kiểm toán Gate A, sau đó tích hợp vào companion app | Review/metric hợp lệ, integration tests, an toàn fail-closed |

Thời hạn cho các giai đoạn chưa được ấn định; không tự khẳng định hoàn thành toàn bộ trong một số tuần nếu chưa có đo lường thực tế.

## 6. Bài thực hành ngay tiếp theo

**Mục tiêu:** mở X500 trong Gazebo với vật cản, chưa kích hoạt thuật toán tránh vật cản tự động.

Trên Ubuntu shell, trước hết kiểm tra đường dẫn PX4 và khả năng gọi Gazebo:

```bash
find ~ -maxdepth 4 -type d -name PX4-Autopilot 2>/dev/null
command -v gz
```

Nếu repository PX4 ở `~/PX4-Autopilot` và target tương thích tồn tại theo phiên bản đang cài:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_walls
```

Ở **một phiên riêng**, sau khi kết thúc SITL trước và kiểm tra target:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_depth
```

Các tên target trên là gợi ý để kiểm tra theo phiên bản PX4 được cài, **không phải kết quả đã chạy thành công trên máy người nghiên cứu**. Không chạy đồng thời hai phiên PX4 SITL mặc định; không nhập các lệnh shell vào dấu nhắc `pxh>`.

**Checklist nghiệm thu bài đầu:** Gazebo hiển thị X500, xuất hiện vật cản, camera di chuyển được; sau đó xác minh cảm biến (nếu dùng bản depth) và ghi lại ảnh/log/config. Bước kế là đọc dữ liệu khoảng cách và thử dừng có kiểm soát trước vật cản.

## 7. Bộ metric của nhánh mô phỏng

Khi tiến hành M5/M6, ghi rõ kịch bản, seed, model, world, vị trí vật cản, tốc độ, điều kiện ban đầu và cảm biến. Các metric tối thiểu:

- Collision count; minimum clearance (m).
- Safe-stop success rate; obstacle avoidance success rate.
- Mission completion rate; time to completion (s).
- Trajectory/path length (m), deviation from planned route.
- Sensor-to-command latency (ms), control-loop stability.
- Fail-safe activation and behavior during missing/stale sensor data.

Không gộp thử nghiệm mô phỏng thành bằng chứng an toàn bay thực tế.

## 8. Quản lý nghiên cứu và điểm dừng

1. Bảo lưu code, artifacts và kết quả Gate A hiện tại; không thay đổi threshold/GT/log để hợp thức hóa số liệu.
2. Không tuyên bố Evaluator v3.1 đã PASS độc lập khi chưa có kết quả audit Codex tương ứng.
3. Ghi một nhật ký thí nghiệm mới cho từng phiên PX4–Gazebo: lệnh, phiên bản, ngày, ảnh/log, PASS/FAIL/UNKNOWN.
4. Theo dõi tiến độ theo các tiêu chí M1–M8; chỉ chuyển sang bước điều khiển chủ động sau khi kiểm tra cảm biến/failsafe.
5. Quay lại Gate A và multi-object sau khi hoàn thành năng lực mô phỏng mà giảng viên yêu cầu.

## 9. Kết luận

**Tại ngày 09/11/2026, ưu tiên của đề tài là năng lực mô phỏng 3D, cảm biến và tránh vật cản trên PX4–Gazebo.** Nhánh Person ReID đã có kết quả phần mềm và quan sát thực nghiệm tích cực nhưng chưa xác lập độ chính xác khoa học; vì vậy được tạm dừng và bảo toàn. Công việc ngay tiếp theo là dựng lại một kịch bản X500 trong world có vật cản, xác minh cảm biến, sau đó lập trình dừng an toàn và tránh vật cản trong mô phỏng.

**Tài liệu này chỉ bổ sung báo cáo tiến độ và định hướng; không sửa mã nguồn, ngưỡng AI, ground truth hoặc dữ liệu gốc.**
