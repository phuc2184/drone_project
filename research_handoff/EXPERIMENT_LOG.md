# Experiment log

## A-GEOM-20261029-01

- Research question: hệ thống sàng lọc có thể phát hiện hộp output khác vị trí mục tiêu khi mục tiêu có nhãn hộp? HYPOTHESIS: IoU thấp tạo candidate để review, không chứng minh sai identity.
- Baseline: `evaluate.py` chỉ đếm TRACKING trong khoảng vắng; số liệu báo cáo cũ chưa được chạy lại.
- Independent variable: output có hộp và IoU với nhãn; controlled: log, labels, derived observation cố định; thresholds review <0.1 và agreement >=0.5, còn lại mơ hồ. Đầu vào IoU đã làm tròn 4 chữ số.
- Dataset: video 5; 44 frame từ `ground_truth/labels_5_eye_fixed.csv` và `outputs/eye_label_analysis.csv`. Chọn mẫu theo khoảng/view, không đại diện; các frame và output đã được xem trước. Chưa phải held-out.
- Code: Git HEAD eb3eb1f3623ed7cc2dc1c7f0821edaece667d3e9; uncommitted `research_handoff/identity_screen.py` SHA-256 604f85c14da0077298c392ad4a0b216f7ffb9b9a973a7510b92d7fac1b42594f; script replay untracked analyze_eye_labels.py SHA-256 0f867a040d338b39c4df598db8a49e9f177ea03a2baa9db533f4edf2d03ca923. Existing observation is derived from historical replay, not regenerated here.
- Inputs and SHA-256 are in `identity_screen_2026-10-29.json`; video 561a81e9a2092fea40f9723ea1df838e6452a63445fc43e635400e00da12cf73, YOLO f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36, OSNet 48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727. Inherited replay flags view_chain=False, color_cue=False; thresholds and model pipeline in `follow_target.py` SHA-256 ef20c47e66403e763e96287b013d4148a52c1bf798be099f720cefa11f94f033; no fresh inference.
- Resource budget: lightweight CPU CSV read, no inference; acceptance: match input frame sets/state/valid and unit cases including wrong output while target present; not a scientific performance threshold.
- Reproduce: `python3 research_handoff/identity_screen.py --labels ground_truth/labels_5_eye_fixed.csv --observations outputs/eye_label_analysis.csv --log logs/tracking_log_20261026_023427.csv`; tests `python3 -m unittest discover -s tests -p 'test_identity_screen.py' -v` (8 passed).
- MEASUREMENT: TRACKING 17 overlap agreement (12 front, 3 back, 2 side), 1 no-output back; INITIALIZING 4 overlap agreement; LOST 20 no-output box and 2 not-visible/no-output. No low-overlap review candidate among these samples. This is not identity accuracy or a false-lock rate.
- Limits: sparse nonrandom sample; labels' independence not fully documented; existing observation replay-derived and exact output boxes absent from log; no absent identity confirmation; rounding, correlations and unknown missing frames. Earlier 204/246 claims are historical and require visual identity review.

## A-DECODE-20261029-02

- Research question: does video actually decode to the metadata or log frame count? HYPOTHESIS: metadata count may differ from decoded frames.
- Dataset: `videos/5.mp4`, SHA-256 561a81e9a2092fea40f9723ea1df838e6452a63445fc43e635400e00da12cf73; baseline metadata 2699, log 2698.
- Method/config: OpenCV 5.0.0 in local venv, `VideoCapture.read()` sequentially until first failure, no seek; independent variable reported metadata versus decoded count; acceptance successful open and sequential read through first end/failure; CPU, one pass.
- MEASUREMENT: 2698 successfully decoded frames, last index 2697; agrees with log row count. UNKNOWN: exact technical cause of metadata difference, and whether first read failure was strictly EOF. No model or weights used; video unchanged.
- Reproduce: `venv/bin/python -c 'import cv2; c=cv2.VideoCapture("videos/5.mp4"); n=0; exec("while True:\n ok,frame=c.read()\n if not ok: break\n n+=1"); print(n)'`.

## A-REVIEW-REPLAY-20261029-03

- Research question: có thể tạo candidate review trên cùng pipeline mà vẫn khớp trạng thái log không? HYPOTHESIS: replay là nguồn hộp chẩn đoán phù hợp khi state/ID/valid phù hợp, nhưng không phải identity ground truth.
- Dataset: `videos/5.mp4` hash 561a81e9a2092fea40f9723ea1df838e6452a63445fc43e635400e00da12cf73; log hash và weight/model/code hashes, config tại `review/candidates_video5_20261029.json`, manifest SHA-256 c3c3dd34e0eadb6626e65cc616a1e3c758b25213fabb495ef2b39af10a3d61fa. Git HEAD eb3eb1f3623ed7cc2dc1c7f0821edaece667d3e9; uncommitted extractor SHA-256 5b4efba4cae6a3b9b1aef93852a56fcc1b65363fb895e88d32d2cafc1ccf0c75. Python 3.12.3, OpenCV 5.0.0, torch 2.14.1+cu130, ultralytics 8.4.177, GTX 1650.
- Baseline historical log and `outputs/eye_label_analysis.csv`; independent variable chosen frames; controlled video/model/config with target initial track 1, view_chain/color_cue False. 20 frames chosen diagnostically after seeing history; no held-out.
- Metric/acceptance: exact equality state, target ID, valid over all processed frames; 7 overlapping labeled output frames compared to historical four-decimal IoU; no identity accuracy. Resource budget one GPU replay to frame 2695 (2696 frames).
- MEASUREMENT: 2696/2696 match; all 7 IoUs match after rounding; 20 frame box geometries and output-to-region mapping pass sanity check. Script refuses output if comparison fails; hashes preserve provenance. Limit: same state/ID/valid does not prove every box is equal, targeted sample nonrepresentative.
- Reproduce: `venv/bin/python -u research_handoff/review/extract_candidates.py --video videos/5.mp4 --log logs/tracking_log_20261026_023427.csv --frames 240,900,1061,1062,1066,1140,1141,1145,2085,2105,2125,2165,2195,2445,2465,2485,2585,2605,2675,2695 --out <new-path.json>`. No overwrite.

## A-METADATA-20261029-04

- Research question: are counts due to decode failure or container packet count? HYPOTHESIS: difference might reflect how coded packets relate to decoded frames; mechanism remains unknown.
- Input/video hash as above; libavformat 60.16.100, libavcodec 60.31.102, libavutil 58.29.100; independent variable packet/frame interpretation; baseline OpenCV 2698 decoded vs 2699 metadata; resource one CPU decode.
- Method: `cc -O2 -Wall -Wextra research_handoff/review/video_decode_audit.c -o /tmp/uav_video_decode_audit $(pkg-config --cflags --libs libavformat libavcodec libavutil)` then `/tmp/uav_video_decode_audit videos/5.mp4`; measure read-until-EOF, flush decoder, check packet NAL lengths. Script SHA-256 8454bdcd51de4693fbbd38e67547a1659aaf65a3b5144b795105c2dbdfc31e8d.
- MEASUREMENT: 2699 metadata frames, 2699 packets, 2698 decoded frames, normal EOF, zero decode errors; all packets contain VCL NAL and none have malformed length fields. Not enough to establish root cause. Result `review/video_decode_audit_20261029.txt`.

## A-REVIEW-TEST-20261029-05

- Objective: verify review form, anonymization of saved candidate records, mapping of a manually reviewed non-target output when target is present, and handling of uncertainty. Dataset synthetic only; no human review, no weights, CPU. Baseline eight pre-existing tests; criteria zero failures and rejection of inconsistent records. `python3 -m unittest discover -s tests -p 'test_*.py' -v`: 12 tests passed; `py_compile` passed. GUI interaction simulated via mock, not a real GUI session; no estimate of model performance.


## A-HUMAN-REVIEW-20261029-03

- Purpose: collect reviewer-authored target boxes / visibility and separately assess tracking. No inference or identity labels are generated by the tool.
- Input: original video and tracking log, SHA-256 checked when loading a replay or consuming a session. Frame indices use sequential decode; valid range 0–2697 established in A-DECODE.
- Protocol: reviewer provides ID and independent evidence, labels all selected frames before locking the annotation phase, then optionally reviews tracking output. Prior knowledge and diagnostic frame selection mean this is not held-out. Repeated annotation for the same frame in one session is rejected; cross-session disagreement requires human adjudication.
- Rules: TARGET requires valid bounds; UNKNOWN and TARGET_NOT_VISIBLE yield uncertain presence; overlap thresholds 0.1/0.5 are geometry screening only. Correct/wrong requires human judgment and an available output box.
- Validation: `venv/bin/python -m unittest discover -s tests -v` — 16 tests passed, including validation, UNKNOWN, provenance, conflicts and existing tests. HTTP smoke test and browser availability documented in checkpoint.
- No human identity labels have yet been collected, so identity accuracy remains UNKNOWN.

- Replay: `venv/bin/python research_handoff/review/extract_candidates.py --video videos/5.mp4 --log logs/tracking_log_20261026_023427.csv --frames 0,60,188,904,905,1061,1062,1066,1140,1141,1145 --out research_handoff/review/diagnostic_replay.json`. Completed, checking state, target ID and valid for every replay frame 0–1145. Historical box coordinates unavailable in log; replay boxes are not verified exact matches.
- HTTP smoke: root, info, frame 1061 and video byte-range returned 200/206; compare before locking rejected with 400. Browser host unavailable, so canvas and interactive review have not been visually verified.
