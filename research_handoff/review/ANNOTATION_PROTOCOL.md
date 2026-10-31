# Quy trình review identity — video 5 diagnostic

## Định nghĩa và tính độc lập

Target identity là **một người cụ thể được xác định trước** bởi bằng chứng thị giác/nguồn tham chiếu độc lập (chuỗi video liên tục, mốc xuất hiện và/hoặc mô tả do người gán cung cấp). Người review phải xác nhận tham chiếu trước, không mặc định track 1 đúng chỉ vì pipeline chọn. Distractor là bất kỳ người khác target; nhiều người khác nhau đều có thể là distractor. Không dùng tracker ID, state, cosine, hộp màu target hoặc output để xác lập identity. Nếu không thể nối identity qua che khuất, góc quay hay thời gian mất dấu, chọn UNKNOWN. Không dựa riêng vào trang phục khi có nguy cơ trùng/đổi trang phục.

- `TARGET`: ở mức scene, người mục tiêu được nhận ra và nhìn thấy; ở mức candidate, người trong vùng được nhận ra là mục tiêu.
- `NON_TARGET`: người trong candidate được nhận ra là người khác; không dùng nhãn này chỉ vì candidate có ID khác.
- `UNKNOWN`: không đủ bằng chứng, che khuất, quá nhỏ, đối chiếu tham chiếu không chắc, hoặc không thể xác định vùng là người.
- `NOT_VISIBLE`: mức scene, không thấy đủ để nhận ra target trong frame; **không tự suy ra target vắng khỏi cảnh**. Dùng UNKNOWN khi chưa biết target có đang nhìn thấy hay không. Không dùng NOT_VISIBLE để gán nhãn một candidate.

Công cụ trước hết hiển thị frame gốc và yêu cầu scene, rồi lần lượt hiện các hộp vàng vô danh cùng crop và yêu cầu candidate label. Không hiện tracker ID, score, state, tên track 204/246 hay hộp nào là output. Vị trí hộp đến từ tracker/output nên bước candidate có điều kiện theo đề xuất detector: đây là **blind với dự đoán identity**, không phải gán hộp hoàn toàn độc lập. Đánh giá đầy đủ còn cần một lượt gán hộp target độc lập trên video gốc trước khi xem bất kỳ hộp detector nào. Bản thân việc chọn frame có chủ đích cũng có thể tạo thiên lệch. Không cung cấp cho reviewer tên/đường dẫn file manifest đã chỉ rõ ID; không mở manifest trước khi hoàn thành session.

Reference frame mặc định 0 chỉ là ảnh gốc để hỗ trợ định hướng, không có identity đảm bảo. Reviewer nên xem video gốc theo trình tự trước, xác định mục tiêu bằng bằng chứng độc lập và truyền `--reference-frame`; cửa sổ reference không đánh dấu người. Nếu không thể xác định mục tiêu, gán UNKNOWN. Với đổi góc nhìn hoặc mất dấu cần xem chuỗi video liên tục ở ngoài công cụ; một ảnh đơn không đủ chứng minh identity.

## Sử dụng

Chuẩn bị giao diện đồ họa; cần OpenCV trong `venv` và file manifest đã được tạo. Ví dụ:

```bash
cd /home/foxsec/drone_project
venv/bin/python research_handoff/review/review_frames.py \
  --manifest research_handoff/review/candidates_video5_20261029.json \
  --video videos/5.mp4 --reference-frame 0 \
  --reviewer 'reviewer-id' --label-source 'raw video; independent reference description/version' \
  --out research_handoff/review/reviews_reviewer_session1.jsonl
```

Thay reviewer, source và reference frame bằng thông tin thật. Tạo đường dẫn output mới cho từng phiên, không ghi đè. Có thể chọn frame với `--frames 1062,1140`, nhưng việc biết lý do chọn frame có thể gây thiên lệch; kết quả là review diagnostic, không phải held-out. Để reviewer mù hoàn toàn với mục đích, người tổ chức chạy lệnh và không tiết lộ phân nhóm frame.

Bấm phím tại cửa sổ `Review`:

- Scene: `p` TARGET hiện diện, `n` NOT_VISIBLE, `u` UNKNOWN, `s` bỏ qua, `q` dừng.
- Candidate: `t` TARGET, `n` NON_TARGET, `u` UNKNOWN, `s` bỏ qua, `q` dừng.
- Sau mỗi nhãn: `1` HIGH, `2` MEDIUM, `3` LOW; `s` bỏ qua, `q` dừng.

Mỗi record được flush sau quyết định. Không bấm nhãn nếu chưa đủ bằng chứng; dùng UNKNOWN. Mỗi phiên riêng cho mỗi reviewer; không mở output của người khác trước khi hoàn thành. Nhãn được lưu JSONL với frame ID (0-based), timestamp xấp xỉ `frame / fps` (không phải PTS), UTC timestamp của quyết định, reviewer, nguồn nhãn, độ chắc chắn, scene, anonymous region index/box, SHA-256 video và manifest. Thiếu candidate do skip/quit là **không được review**, không được suy diễn là UNKNOWN hoặc NON_TARGET. Dữ liệu gốc không bị thay đổi. Không sửa JSONL đã lưu: nếu cần sửa, ghi phiên bổ sung và lưu provenance adjudication.

## Tiêu chí và xử lý sau review

Hai reviewer độc lập gán trên cùng tập con xác định trước, sau đó tính disagreement theo region và thảo luận/adjudicate trong file mới, giữ nguyên các quyết định ban đầu. Không tính false identity lock chỉ từ scene NOT_VISIBLE. Để xác nhận wrong output khi target present: scene TARGET dựa bằng chứng độc lập, và candidate tương ứng hộp **output** được review NON_TARGET. Với các candidate LOST, không suy ra đó là output. Chỉ sau khi nhãn đã khóa, một bước phân tích được phép đối chiếu anonymous region với manifest riêng để biết output/candidate; lưu phương pháp join, xử lý trường hợp output không trùng hộp tracker, scene/candidate thiếu, giá trị chưa chắc chắn và tham chiếu. Không suy đoán true identity từ track ID hay similarity. Độ chắc chắn LOW không mặc định là đúng; báo riêng hoặc định nghĩa policy trước khi đọc nhãn.

Video 5 là exploratory; các frame chọn sau khi đã xem log và báo cáo, không dùng để suy rộng hay ước tính accuracy/false-lock rate. Các tập calibration, validation, held-out phải được khóa trước khi xem output ở cấp nhóm người/buổi, không chia frame của cùng người/buổi giữa nhóm khi có thể; tách video 5 sang exploratory. Nếu quan hệ người/buổi tạo thành nhóm liên thông thì giữ cùng split; không đủ nhóm độc lập thì báo BLOCKED, không giả tạo held-out. Ghi manifest hash, group, version, seed, thời điểm khóa và kiểm tra leakage. Tránh tune dựa vào held-out.

## Xuất dữ liệu cho metric sau khi khóa review

Chỉ thực hiện sau khi hoàn thành và lưu session review. Quy tắc cố định của converter: `TARGET` scene độ chắc chắn HIGH/MEDIUM => present; `NOT_VISIBLE`, `UNKNOWN` và LOW => uncertain; không có nhãn `absent` được tạo tự động. Với output, candidate TARGET/NON_TARGET HIGH/MEDIUM => correct/wrong; thiếu candidate, UNKNOWN hoặc LOW => uncertain. Scene LOW vẫn làm output unresolved trong metric. Lọc `TRACKING` mặc định. Dữ liệu review cố ý nhắm frame rủi ro nên kết quả chỉ chẩn đoán.

```bash
venv/bin/python research_handoff/review/reviews_to_metrics.py \
  --manifest research_handoff/review/candidates_video5_20261029.json \
  --reviews research_handoff/review/reviews_reviewer_session1.jsonl \
  --video videos/5.mp4 --out research_handoff/review/metrics_input_reviewer_session1.csv
python3 research_handoff/identity_screen.py --reviews research_handoff/review/metrics_input_reviewer_session1.csv
```

Công cụ kiểm tra hash video/manifest, tính nhất quán scene, geometry và reviewer; không chứng minh reviewer đã mù hay nhãn đúng. Không chạy hoặc diễn giải metric cho tới khi có review thực, và không coi tỷ lệ trong mẫu chẩn đoán là hiệu năng quần thể. Không lấy frame chưa review làm mẫu no-output hoặc đúng; không dùng nhãn này để làm held-out.
