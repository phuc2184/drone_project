=====================================================================
GEMINI BRIEF - Autonomous person-following drone (PX4)   |   version 2026-10-15
Read this WHOLE brief at the start of every session. Then read STATUS.md (if it exists) and follow_target.py.
Start with PART A. Do NOT edit any file until I answer your PART A report.
=====================================================================

1. PROJECT
University research project: a drone based on PX4 must follow ONE chosen person.
Long-term goals: autonomous flight, obstacle avoidance, person following.
Current phase (required by my supervisor): single-target tracking WITH re-identification (ReID),
developed first on pre-recorded video (no real drone yet).
Requirements:
 REQ1 The user selects ONE target (a person for now).
 REQ2 The system follows it with a stable identity.
 REQ3 When the target leaves the frame or is fully occluded: report LOST and NEVER switch to another person.
 REQ4 When the target reappears: recognize it is the SAME person and resume. Never lock onto a wrong person.
Priority: a FALSE LOCK (following a wrong person) is much worse than a missed or slow re-acquisition.
Roles: Claude designs the algorithm; you (Gemini) write and run the code. If you disagree with the design, write it as a
clearly marked SUGGESTION and still implement what is specified.
I am a beginner: give complete runnable files, exact commands, short comments, no pseudo-code.

2. ENVIRONMENT (verified, do not reinstall or change)
- Ubuntu 24.04.5 LTS, Python 3.12, user foxsec, home /home/foxsec. Network available.
- GPU: NVIDIA GTX 1650, 4 GB VRAM (driver OK, CUDA works). Use lightweight models only.
- PX4 simulation works: cd ~/PX4-Autopilot && make px4_sitl gz_x500 (Gazebo Harmonic). Do not touch ~/PX4-Autopilot.
- Venv: /home/foxsec/drone_project/venv (source ~/drone_project/venv/bin/activate).
  Installed: ultralytics 8.4.177, torch 2.14.1, torchvision 0.29.1, opencv-python, torchreid, mavsdk 3.17.4.
  IMPORTANT: keep mavsdk < 4 (version 4 changed the API and breaks takeoff.py). Never upgrade it.
- Download model weights only from official sources (torchreid Model Zoo, BoxMOT releases, Ultralytics).
  For anything else, tell me the URL first.

3. FILE MAP (exact names; do not create other files, do not rename)
Existing, working:
  /home/foxsec/drone_project/takeoff.py            MAVSDK demo: connect udpin://0.0.0.0:14540, arm, takeoff, hover 10 s, land. Do not modify.
  /home/foxsec/drone_project/track_test.py         baseline YOLO+ByteTrack test. Do not modify.
  /home/foxsec/drone_project/follow_target.py      MAIN program (Step 2, current version).
  /home/foxsec/drone_project/backups/follow_target_2026-10-14.py   my backup of the file as I left it on 2026-10-14.
  /home/foxsec/drone_project/weights/osnet_x0_25_msmt17.pth, osnet_x1_0_msmt17.pth  (classifier outputs 4101 = MSMT17 person-ReID weights)
  /home/foxsec/drone_project/videos/               input videos (5.mp4 is the main test video)
  /home/foxsec/drone_project/outputs/              annotated videos, gallery_debug/, gallery_montage.jpg, check_frames/
  /home/foxsec/drone_project/logs/                 CSV logs (latest: tracking_log_20261014_184539.csv)
  /home/foxsec/drone_project/ground_truth/         ground-truth files
Files you may create (only these):
  /home/foxsec/drone_project/STATUS.md             session notes (see rules)
  /home/foxsec/drone_project/GEMINI_BRIEF.md       this brief (do not edit it)
  /home/foxsec/drone_project/calibrate_thresholds.py   (PART C)
  /home/foxsec/drone_project/evaluate.py               (PART C)
  A later drone_control.py is NOT allowed yet.

4. CURRENT ALGORITHM (implemented in follow_target.py; do not change decisions)
Pipeline: frame -> YOLOv8n (class person, conf >= 0.40) -> BoT-SORT (botsort.yaml, camera motion compensation)
-> TargetManager (OSNet x1_0 MSMT17 embeddings, L2-normalized, cosine similarity) -> HUD / CSV / output video.
Tracker IDs are only a hint: identity is verified by appearance because IDs change after occlusion and can swap.
Score = mean of the top-3 cosine similarities against the gallery. HUD shows "TARGET [ID] | all X | anchor Y"
(all = anchor + dynamic galleries; anchor = anchor gallery only).
Quality tiers: FULL (size >= 30x80 px, not touching the frame border, no overlap IoU >= 0.30);
PARTIAL (touches the border but size OK and no heavy overlap).
Galleries: anchor (10 embeddings, at least 5 frames apart, never removed), dynamic (max 30, FIFO).
States: IDLE -> INITIALIZING -> TRACKING <-> LOST -> GIVE_UP.
Constants (current values, do not change without my approval):
  T_VERIFY 0.55 (verify every 5 frames; fail limit 3; suspicious window 15 frames, fail limit 2)
  T_ANCHOR_FLOOR 0.60 (sim_anchor must reach it for verification and acceptance)
  T_UPDATE 0.75; dynamic additions require FULL quality, >= 30 frames since the last suspicious event, sim_anchor >= 0.80
  T_ACCEPT_LONG 0.78 (PARTIAL needs +0.05), N_CONFIRM 8 consecutive quality-ok frames, MARGIN_MIN 0.08
  CONSECUTIVE_EMB_SIM_THRESH 0.50, T_MAX_LOST 30 s, USE_SPATIAL_GATE False (camera moves), USE_COLOR_CUE False
Suspicious events (start the 15-frame window): center jump > 25% of frame width, area change > 2x, overlap with another
person ended, just re-acquired.
Fix list that was requested earlier (verify what is really implemented):
  R1 person-ReID weights from weights/ (verified DONE: osnet_x1_0_msmt17.pth, pretrained=False + load_pretrained_weights)
  R2 quality tiers FULL/PARTIAL (verification: both; acceptance: both, PARTIAL +0.05; gallery updates: FULL only)
  R3 anchor floor + dynamic-gallery rules + rollback (when LOST is caused by a verification/continuity failure,
     delete the dynamic embeddings added in the previous 60 frames)
  R4 continuity check on the target track in TRACKING (consecutive embedding cosine < 0.50 -> LOST at once, no gallery update)
  R5 better anchors (>= 5 frames apart, skip cosine > 0.95 duplicates, click frame not counted twice,
     continuity check also in INITIALIZING)
  R6 debug output (gallery crops in outputs/gallery_debug/, CSV columns sim_anchor, sim_dynamic, quality_tier, HUD numbers,
     post-run list of frame ranges where state == TRACKING but sim_anchor < T_ANCHOR_FLOOR)

5. TEST DATA AND KNOWN RESULTS
videos/5.mp4: handheld selfie-style street vlog, 2698 frames, 1080p, panning camera; the person is often cut at the frame
border and often seen from the front, sometimes from behind.
The true target in videos/5.mp4 is the man in the: <FILL IN: denim jacket + black turtleneck / grey hoodie with glasses>.
History: with ImageNet weights the system showed "TARGET [ID: 1]" on two different men (false lock). After switching to
MSMT17 weights those errors are reported fixed.
Latest run (outputs/output_step2b.mp4, logs/tracking_log_20261014_184539.csv), state changes:
  0 INITIALIZING(ID1), 188 TRACKING(ID1), 905 LOST, 1751 TRACKING(ID350), 2110 LOST, 2460 TRACKING(ID468),
  2513 LOST, 2581 TRACKING(ID534), 2625 LOST, 2684 TRACKING(ID544). gallery_debug has 73 crops.
Open problem: recognition from BEHIND is poor and late (example: TARGET [ID: 534] all 0.80 / anchor 0.71 appears late).
Suspected causes: anchors are front-view; the sim_anchor >= 0.80 rule blocks adding back-view embeddings; PARTIAL needs +0.05;
N_CONFIRM = 8 and margin 0.08. These are hypotheses, measure before changing anything.

6. RULES (all sessions)
- Do NOT tune any constant or threshold just to make videos/5.mp4 pass. Thresholds are chosen from data in PART C.
  If you think a constant must change, show the evidence and ask me first. Never raise CONSECUTIVE_EMB_SIM_THRESH blindly.
- When you edit code: return the COMPLETE file, start with the exact file path, fix only what I asked, no other new files.
- Never overwrite older outputs: use a new --out name for each run (output_step2c.mp4, ...).
- Keep STATUS.md updated at the end of every session: done / pending / every constant changed (old -> new, reason) /
  latest acceptance results (state changes with frame numbers, false-lock ranges, FPS) / exact commands to reproduce.
- Do one PART at a time. After each PART report results and WAIT for my approval.
- Do not start drone control or obstacle avoidance yet (later phases: offboard velocity setpoints via MAVSDK-Python mavsdk<4,
  Gazebo x500 with camera, then obstacle avoidance).

7. WORK PLAN

PART A - STATUS REPORT ONLY (no code changes; run only what is needed to answer)
 A1 For each of R1..R6: DONE / PARTIAL / NOT DONE with line numbers as evidence.
 A2 Weight file loaded now (exact name, model name, classifier size) and the pipeline FPS.
 A3 EVERY edit made after the last full acceptance run, including each constant changed: "name: old -> new, reason".
    Compare with backups/follow_target_2026-10-14.py if needed.
 A4 What happened around frame 904 (tracker ID switch): who had the target ID before/after, the consecutive-embedding cosine,
    sim_anchor of the old and the new person, and why the continuity check did or did not fire.
 A5 Answer with evidence (sim_anchor, quality_tier, best candidate score at those frames):
    (1) why anchor enrollment took 188 frames; (2) was LOST at 905 really the ID switch;
    (3) were the re-acquisitions at 1751, 2460, 2581, 2684 on the TRUE target, and why did the two short TRACKING stretches
    (2460-2513, 2581-2625) end in LOST; (4) the LOST period 905-1751 (~28 s) is close to T_MAX_LOST = 30 s.
 A6 Anything in the code that contradicts section 4.
 Create STATUS.md with this report. Wait for my reply.

PART B - finish whatever is NOT DONE among R1..R6, then run the acceptance test:
  python3 follow_target.py --video videos/5.mp4 --out outputs/output_step2c.mp4
 Pass condition: the person who is NOT the true target never gets the green TARGET box. Show state changes, FPS, and the
 suspected false-lock ranges.

PART C - STEP 3: calibrate_thresholds.py and evaluate.py (no threshold changes in follow_target.py)
 Ground truth, lightweight format: ground_truth/<video>.csv with intervals: start_frame,end_frame,target_visible(0/1),view
 (view optional: front / back / side). Plus a keypress spot-check mode in evaluate.py: show one frame every 30 frames of the
 output video and let me press y = green box is on the correct person, n = on a wrong person, s = no green box shown.
 Explain step by step how I create the ground-truth file for 5.mp4.
 calibrate_thresholds.py: collect embeddings from tracks I label as the target vs other people; plot histograms and print
 percentiles of SAME-person vs DIFFERENT-person similarity (all-gallery score and anchor-only score), separately per view and
 for front-vs-back of the same person. Rule: T_ACCEPT must be above the 99th percentile of DIFFERENT-person similarities;
 report what fraction of SAME-person similarities it keeps.
 evaluate.py: reacquisition success rate; FALSE LOCK rate (frames with the TARGET label on a wrong person / total frames,
 must be ~0); average time to reacquire after the target returns; number of identity switches; average FPS; per view.

PART D - VIEW-ROBUSTNESS EXPERIMENTS (after C; optional switches, all default False; do not change existing thresholds)
 E1 USE_VIEW_CHAIN_UPDATE: allow adding a dynamic embedding with T_ANCHOR_FLOOR <= sim_anchor < 0.80 only if ALL hold:
    same track id TRACKING continuously for >= 60 frames; no suspicious event and no other person with IoU > 0.10 in the last
    30 frames; cosine to the mean of the last 5 target embeddings >= 0.80; FULL or PARTIAL quality; >= 15 frames since the last
    chain addition; stored in a separate view_gallery (max 30, with timestamps) covered by the same rollback (R3).
 E2 USE_COLOR_CUE = True (already implemented): report its effect.
 E3 If torchreid provides osnet_ain_x1_0 with person-ReID weights, add it as an alternative REID_MODEL_NAME; report FPS and metrics.
 Report one table: baseline vs E1 vs E2 vs E3 with reacquisition time per view, FALSE LOCK frames (must stay ~0),
 number of LOST/TRACKING cycles, FPS. Recommend nothing until I have seen the numbers.

PART E - USABILITY OPTIONS (no algorithm or threshold changes)
 O1 --start-frame N: seek to frame N before target selection; frame numbers in CSV and logs stay absolute.
 O2 Selection screen: key 'f' skips 30 frames; 'n' stays next frame.
 O3 Key 'r' during the run: pause, let me click a different person, reset the target model (anchors, dynamic and view
    galleries, counters, candidate history) and continue from the current frame without rewinding. Print "TARGET RESELECTED at frame X".
 O4 --classes "0" (comma-separated COCO class ids, default "0"). If a class other than 0 is used with PersonReIDExtractor,
    print a clear warning that identity matching is unreliable and a generic extractor (e.g. DINOv2-small) is needed; print
    which quality config is used for each class and do not silently fall back to the person config.
 O5 Do NOT implement GenericExtractor yet.

PART F - SPEED (last; only make it faster, same decisions)
 S1 Profiling first: print every 30 frames the average ms per frame of: detect+track, ReID embedding, TargetManager logic,
    CSV logging, draw_hud, video write, imshow, and the total FPS. Show me the table before optimizing.
 S2 ReID only when needed: while TRACKING compute embeddings only for the target track and only on scheduled verification
    frames (or in the suspicious window); while LOST compute embeddings for candidate tracks every REID_CANDIDATE_STRIDE = 2
    frames for at most REID_MAX_CANDIDATES = 6 largest boxes. SHOW_DEBUG_SCORES = False: no scores for non-target people.
 S3 CSV: open the log once before the loop, flush every 60 frames, close in finally.
 S4 Flags: --no-video, --out-scale 0.5, --no-show (click selection at the start still works).
 S5 Constants YOLO_IMGSZ = 640 (try 480) and USE_HALF = False (try True; if detections vanish or look wrong keep False; GTX 16xx
    can have FP16 problems). Print the FPS for each setting.
 S6 FRAME_STRIDE = 1 (process every N-th frame; default unchanged).
 Report FPS before/after each change and confirm that the state changes and the lost/re-acquired frames on videos/5.mp4
 are unchanged.

Start with PART A now.

ROADMAP RULE (all sessions): at the END of every task, before stopping, update roadmap_data.json
(item statuses, estimated percent, one new entry in "log" with date, what was done, evidence, what is pending approval),
then run: python3 update_roadmap.py and paste the raw output. Only mark an item done if there is evidence
(command output, file, or my visual check). Never write "false lock = 0%". Do not change statuses of items you did not work on.
