# Human identity review — video 5

Chạy tại gốc dự án:

```bash
venv/bin/python research_handoff/review/identity_review.py --manifest research_handoff/review/diagnostic_replay.json
```

Mở http://localhost:8501. Máy chủ chỉ nghe 127.0.0.1. Manifest có hộp replay cho frame 0, 60, 188, 904, 905, 1061, 1062, 1066, 1140, 1141, 1145; các frame khác chưa có hộp replay. Log gốc không chứa tọa độ output; nếu thiếu manifest hoặc frame không có trong manifest, hộp AI không khả dụng. Video gốc có thể phát trong trình duyệt nếu codec được hỗ trợ; frame hiển thị riêng được giải mã tuần tự từ đầu để bảo đảm ID. Bấm frame sau có thể chậm.

1. Nhập reviewer ID và mô tả nguồn bằng chứng độc lập xác định mục tiêu. Xem video gốc, chọn frame trong hàng đợi hoặc nhập frame ID. Frame đánh số từ 0; vùng hợp lệ 0–2697.
2. Kéo trên ảnh để chọn và phóng to vùng. Chỉ chọn TARGET khi xác định được người mục tiêu bằng bằng chứng độc lập và đã vẽ hộp. Nếu không thấy mục tiêu, chọn TARGET_NOT_VISIBLE; nếu không chắc, chọn UNKNOWN. Ghi chú và lưu từng nhãn. Không suy ra người thật từ track ID.
3. Lưu tất cả frame cần review, rồi bấm **Hoàn thành và khóa toàn bộ annotation**. Sau khi khóa không thể thêm nhãn trong phiên này. Việc review sau khi đã biết output hoặc tạo phiên mới sau khi nhìn output không đảm bảo mù; ghi rõ điều đó trong ghi chú và coi như diagnostic.
4. Chọn lại frame đã lưu, mở đối chiếu. Hộp vàng là nhãn người, hộp đỏ là output, xanh là track từ replay nếu có. Log hiển thị state, ID và điểm; ID chỉ là ID tracker. Xem output và bằng chứng độc lập trước khi tự chọn correct/wrong; nếu chưa đủ chứng cứ chọn uncertain. Với không output chọn none. `TARGET_NOT_VISIBLE` không xác nhận mục tiêu vắng mặt.
5. JSONL mới được ghi vào `research_handoff/review/sessions/` với mã phiên, SHA-256 video, reviewer, bằng chứng, timestamp và schema. Dữ liệu gốc không bị ghi đè. Không có sửa/xóa nhãn trong phiên; nếu sai, lưu phiên mới và yêu cầu người có thẩm quyền phân xử các phiên bất đồng. Không gộp các phiên mâu thuẫn một cách tự động.

Kiểm tra và tính thống kê trên một phiên đã đối chiếu:

```bash
venv/bin/python research_handoff/identity_screen.py --review-session research_handoff/review/sessions/TEN_PHien.jsonl --video videos/5.mp4 --log logs/tracking_log_20261026_023427.csv
```

Nếu phiên sử dụng manifest, thêm `--manifest` với đúng file đó. UNKNOWN và TARGET_NOT_VISIBLE map thành uncertain, không vào mẫu số kết luận output; annotation chưa đối chiếu được báo riêng. `low_overlap_review` (<0.1), `overlap_agreement` (>=0.5), khoảng giữa là ambiguous; chỉ là phân loại hình học, không kết luận đúng/sai danh tính. Tập frame chọn có chủ đích và không đại diện.

Tạo replay tùy chọn bằng `extract_candidates.py --video videos/5.mp4 --log logs/tracking_log_20261026_023427.csv --frames 0,60,188,904,905,1061,1062,1066,1140,1141,1145 --out research_handoff/review/diagnostic_replay.json`. Replay kiểm tra state/track ID/valid qua từng frame, nhưng không chứng minh tọa độ giống lần chạy cũ. Máy phải có dependency/model như lần chạy; replay thất bại nếu log không khớp. Không dùng dữ liệu này làm ground truth.
