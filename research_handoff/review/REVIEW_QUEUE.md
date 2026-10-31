# Danh sách chẩn đoán — chỉ cho người điều phối

Đây là các frame chọn có chủ đích sau khi xem log/báo cáo, không phải mẫu ngẫu nhiên. Tránh cho reviewer biết nhóm nào liên quan tới track hoặc trạng thái; công cụ không hiện thông tin này. Đề nghị reviewer xem video gốc liên tục để tạo và kiểm tra tham chiếu target độc lập trước.

| Nhóm | Frame 0-based | Cơ sở chọn và giới hạn |
|---|---|---|
| Tham chiếu/context | 240, 900 | TRACKING; dùng để quan sát, không giả định output đúng. |
| 204 | 1061, 1062, 1066 | Log best candidate 204, LOST/valid=0; replay có track 204. Review tất cả vùng ẩn ID. |
| 246 | 1140, 1141, 1145 | Log best candidate 246, LOST/valid=0; replay có track 246. Review tất cả vùng ẩn ID. |
| Chuyển view và chồng lấn | 2085, 2105, 2445, 2465, 2485 | Nhãn view lịch sử và log; không xem chúng là ground truth identity. |
| LOST có người | 2125, 2165, 2195, 2445 | Replay có ít nhất một người, không có output; candidate review không phải false lock. |
| Mục tiêu và distractor cùng xuất hiện | 240, 900, 2085, 2105, 2465, 2485, 2585, 2695 | Replay có nhiều hộp người; việc target thực sự có mặt cần review. |
| TRACKING output chưa khớp | 2085, 2105, 2465, 2485, 2585 | Không có low-overlap candidate trong mẫu GT trước; các frame này là kiểm tra bổ sung, không được gọi là mismatch đã xác nhận. 2085 là IoU thấp nhất trong mẫu lịch sử có output (0.5302). |
| TRACKING nhưng không output | 2605 | State TRACKING, valid=0; không coi là khóa người. |
| Hai nhãn cũ bị loại | 2675, 2695 | Review nguyên nhân chất lượng nhãn độc lập; dữ liệu gốc giữ nguyên. |

**BLOCKED_BY_HUMAN_REVIEW:** chưa có quyết định của reviewer và bằng chứng xác lập target. Không tự điền nhãn. Nhãn box replay có giới hạn: state/target ID/valid khớp log ở 2696 frame, không chứng minh tất cả tọa độ khớp; đối chiếu 7 frame được gán nhãn có output cho IoU làm tròn trùng dữ liệu lịch sử.
