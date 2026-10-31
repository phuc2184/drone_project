# Phase A — audit và hướng đánh giá (2026-10-29)

## Kết quả kiểm chứng

**VERIFIED:** Git HEAD `eb3eb1f3623ed7cc2dc1c7f0821edaece667d3e9`; các thay đổi ban đầu trong ROADMAP.md, STATUS.md, roadmap_data.json, EYE_LABEL_REPORT.md, analyze_eye_labels.py được giữ nguyên. `5_manual.csv` có 13 khoảng; nguồn gốc người gán nhãn, phương pháp và mức độc lập đối với pipeline chưa được chứng minh bằng provenance trong file. `5_ai_draft_notes.md` tự mô tả một quy trình chỉ xem video, nhưng là tập draft riêng, không chứng minh provenance của `5_manual.csv`.

**MEASUREMENT:** `labels_5_eye.csv` có 46 dòng (44 box, 2 not_visible); `_fixed` có 44 dòng (42 box, 2 not_visible). Hai dòng bị loại là 2675 và 2695; 44 dòng còn lại giống từng trường với file gốc. File backup và file raw có SHA-256 giống nhau. Công cụ label không đọc tracker ID nhưng chọn frame và nhãn góc nhìn từ `5_manual.csv`; thiếu hồ sơ người gán, phiên, điều kiện làm mù và kiểm định độc lập. Do đó chưa chứng minh GT độc lập đầy đủ. Hai dòng bị loại vẫn nằm trong dữ liệu gốc; cần review mù thay vì coi loại bỏ là đúng.

**VERIFIED:** `evaluate.py` chỉ đếm TRACKING khi `target_visible=0` ngoài vùng đệm; nó bỏ sót người sai khi mục tiêu thật còn xuất hiện. Mẫu số false-lock rate là toàn bộ dòng log, không phải số lệnh/hộp đã xác nhận. Coverage dựa trên state, không kiểm tra hộp hay người. Reacquisition tìm TRACKING trong các khoảng `target_visible=1` sau khoảng đầu, kể cả chuyển view/partial; không chứng minh identity hoặc sự tái xuất hiện. ID switches là đổi tracker ID, không phải identity switch. FPS mặc định 30; thực tế metadata khác.

**MEASUREMENT:** Log tại frame 1061, 1062, 1066 có best_cand_id 204, sim_anchor lần lượt 0.7485, 0.7608, 0.7343, gate nconfirm_fail; 1140, 1141, 1145 có best_cand_id 246, sim_anchor 0.7207, 0.7183, 0.7144, gate accept_fail. Tất cả state LOST và valid=0. Điểm là similarity, không phải xác suất; chưa chứng minh ứng viên là người khác hay khóa nhầm, chưa loại trừ lỗi ngoài sáu frame này. Review hình ảnh 204/246 với hộp track và tham chiếu mục tiêu vẫn cần thiết.

**MEASUREMENT:** OpenCV metadata video: 1920×1080, 2699 frame, 29.843781789875006 FPS. Giải mã tuần tự bằng OpenCV cho 2698 frame, chỉ số cuối 2697, bằng 2698 dòng log. Metadata báo cao hơn một frame; nguyên nhân kỹ thuật sâu hơn chưa xác định. GPU torch nhận GTX 1650, CUDA available; OpenCV 5.0.0, torch 2.14.1+cu130 trong venv.

## Metric và giới hạn

`identity_screen.py` kiểm tra từng frame label với observation và log; phân loại không có hộp, overlap cao, overlap mơ hồ, low-overlap cần review, hoặc visibility chưa biết. Ngưỡng sàng lọc 0.1/0.5 chọn để minh họa hình học, không phải ngưỡng xác minh identity. IoU đầu vào đã làm tròn bốn chữ số; điểm sát ngưỡng cần xem lại. Kết quả tách state/view; không chuyển `not_visible` thành `absent`, không xem INITIALIZING là TRACKING. Hộp overlap cao có thể vẫn sai người trong cảnh chồng lấn; hộp overlap thấp có thể là sai lệch annotation/detection. Đây không phải tỉ lệ false lock.

Nhánh `--reviews` tính số lượng wrong khi presence=present hoặc absent, correct, unresolved và phân số wrong trên các output đã được review xác định; không tự tạo nhãn review, không tính no-output vào mẫu số. Presence uncertain hoặc identity uncertain luôn báo riêng, không làm mẫu số đáng tin hơn. Cần lọc và công bố state, view, cách chọn mẫu và provenance review trước khi tính. Không có nhãn review độc lập hiện tại nên **UNKNOWN:** tỉ lệ khóa nhầm và accuracy; chưa có số liệu.

## Thiết kế thu thập độc lập và phân chia

1. Thu thập video mới, nhiều cảnh và người/distractor, gồm front/back/side, che khuất, vắng và xuất hiện cùng distractor; giữ video gốc/hash. Ghi consent và điều kiện thu thập theo quy trình dự án.
2. Trước khi xem output, khóa manifest theo video/buổi/người và tách calibration, validation, held-out theo nhóm không trùng người và buổi khi khả thi; đặt khoảng cách/nhóm sự kiện tránh frame sát nhau lọt hai tập. Video 5 đã xem kết quả, chỉ dùng exploratory, không tuyên bố held-out. Chưa có đủ video/person để tạo split hợp lệ.
3. Gán nhãn từ video gốc tuần tự, không hiển thị tracker, điểm hoặc output: target present/absent/uncertain, hộp và view, occlusion, frame mốc. Lưu annotator, ngày, source hash, phiên bản công cụ, cách chọn frame, định nghĩa, đánh dấu uncertain. Không giả định `not_visible` là absent. Gán dày quanh chuyển trạng thái, để uncertainty interval khi không xác định được mốc.
4. Hai người gán độc lập cho subset định trước; báo bất đồng và adjudication có lưu lịch sử. Xem lại 2675/2695 và 204/246 theo mẫu chọn cố định, ghi rõ review sau khi biết output là phân tích diagnostic, không dùng làm blind held-out.
5. Review identity output với video gốc và hộp dự đoán được che track ID/similarity; nhãn correct/wrong/uncertain, đếm cả trường hợp target present with distractor. Mẫu ngẫu nhiên định trước theo strata view/state/presence/occlusion, lưu seed và inclusion probabilities; không suy rộng tỉ lệ mẫu có chủ đích. Với split và nhãn phù hợp, đo false lock ở target present/absent, correct output coverage, unresolved, theo view và episode; tính latency chỉ với mốc xuất hiện chính xác và xác nhận đúng identity. Báo đơn vị thống kê video/episode và độ không chắc chắn.

## Gate A

**NOT PASSED:** provenance/độc lập GT chưa chứng minh; nhãn identity review chưa có; split calibration/validation/held-out độc lập chưa tồn tại; leakage không thể loại trừ; metric hình học và nhánh reviewed có unit test nhưng chưa được đối chiếu bộ GT identity thực. Không tối ưu threshold hoặc kết luận Phase B. Tiếp theo: chuẩn bị công cụ/manifest annotation độc lập; review các frame rủi ro là diagnostic, sau đó thu thập/đánh giá tập mới.

## Tiếp tục P0 — review độc lập và metadata (2026-10-29)

**VERIFIED:** `review/ANNOTATION_PROTOCOL.md` định nghĩa TARGET, NON_TARGET, UNKNOWN, NOT_VISIBLE và phương pháp xác lập target; `review/review_frames.py` ghi JSONL có frame, timestamp danh định, reviewer, nguồn, độ chắc chắn và hash. Scene được quyết định trước khi hiện hộp; candidate hộp/crop không hiện ID, score, state hoặc vai trò output. Việc thấy hộp tracker vẫn có thể gây thiên lệch, nên đây là review identity blind một phần; chưa thay cho gán hộp GT độc lập. GUI workflow được kiểm thử bằng mock, **chưa được kiểm thử tương tác thực bởi reviewer**.

**MEASUREMENT:** Replay 0–2695 khớp state, target_track_id và valid với log 2696/2696 frame; thu 20 frame tại `review/candidates_video5_20261029.json`. 204 có mặt trong 1061/1062/1066; 246 có mặt trong 1140/1141/1145. Ở 7 frame có nhãn hộp và output, IoU replay làm tròn khớp observation lịch sử. Điều này không chứng minh tọa độ ở mọi frame hay danh tính. Danh sách review tại `review/REVIEW_QUEUE.md`; chưa có human review.

**VERIFIED:** Nhánh review đã có đường nối `reviews_to_metrics.py`: không đổi NOT_VISIBLE thành absent; thiếu candidate và confidence LOW được giữ uncertain; mặc định phân tích TRACKING. Kiểm thử synthetic có mục tiêu present + candidate output NON_TARGET và kiểm tra được wrong_present. `identity_screen.py` giờ ghi rõ các frame có thể tương quan; công cụ không tự xác minh provenance hoặc tính đại diện.

**MEASUREMENT:** Audit độc lập libavformat/libavcodec: metadata stream 2699; 2699 video packet; 2698 frame được decode kể cả flush; EOF bình thường; 0 lỗi giải mã báo cáo. Các packet H.264 đều chứa ít nhất một VCL NAL theo phép kiểm tra độ dài đơn giản, không có NAL-length lỗi. **UNKNOWN:** nguyên nhân chính xác chênh lệch packet/frame; không đồng nhất packet với frame. Xem `review/video_decode_audit_20261029.txt` và mã audit.

**GATE A: NOT PASSED.** Protocol, công cụ và unit test là cơ sở; chưa có ground truth identity được người xác minh, đánh giá tính độc lập/đồng thuận và tập held-out hợp lệ. Video 5 là exploratory. Các task review identity/204/246 **BLOCKED_BY_HUMAN_REVIEW**; không có cơ sở tính identity accuracy hoặc false-lock rate. Tách người/buổi cho tập mới như protocol, chưa tạo split giả từ một video.
