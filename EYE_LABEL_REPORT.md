# EYE_LABEL_REPORT - what the tracker/ReID really does vs hand-drawn boxes

Date: 2026-10-26. Scope: `videos/5.mp4` (2698 processed frames), baseline settings
(`--target-id 1`, headless, `osnet_x1_0` MSMT17 weights). Read-only analysis:
no decision branch, constant or threshold was changed; no video and no log were written
by the analysis; nothing under `ground_truth/` was written.

Epistemic labels used below: **VERIFIED FACT** (reproduced from code/log/output),
**MEASUREMENT** (printed by the run, descriptive), **HYPOTHESIS** (not established).

Inputs:
- Labels: `ground_truth/labels_5_eye_fixed.csv` (used, because it exists).
- Reference log: `logs/tracking_log_20261026_023427.csv` (accepted STEP 2 run, 2698 rows).
- Script: `analyze_eye_labels.py` (imports `DetectorTracker`, `TargetManager`,
  `PersonReIDExtractor`, `QualityFilter`, `compute_iou` from `follow_target.py`).
- Output data: `outputs/eye_label_analysis.csv` (44 rows, 12 columns).

---

## 1. Label set and exclusions

**MEASUREMENT** (from the run header and the CSVs):

```
  labels file        : /home/foxsec/drone_project/ground_truth/labels_5_eye_fixed.csv
  label rows used    : 44  (excluded: 2)
  EXCLUDED frame  2675 box          not in labels_5_eye_fixed.csv (known suspicious)
  EXCLUDED frame  2695 box          not in labels_5_eye_fixed.csv (known suspicious)
```

| Item | Value |
|---|---|
| Rows in `labels_5_eye_fixed.csv` | 44 (42 `box`, 2 `not_visible`) |
| Views | front 16, back 24, side 4 |
| Excluded 2675 | box 382,344,394,367 = 12 x 23 px (in `labels_5_eye.csv`, absent from `_fixed`) |
| Excluded 2695 | box 404,330,408,335 = 4 x 5 px (in `labels_5_eye.csv`, absent from `_fixed`) |
| Additional rows removed by the 40x80 rule | 0 |
| `not_visible` rows | 2175 (back), 2325 (back) |

`ground_truth/labels_5_eye.csv` was only read (to identify the two suspicious rows);
no GT file was modified. **VERIFIED FACT** (script opens GT read-only; `git status` in §8).

---

## 2. Replay faithfulness proof

**MEASUREMENT** - raw output of the full run:

```
  frames replayed            : 2698
  frames in log              : 2698
  state/target_id mismatches : 0
  RESULT: identical on every frame (state AND target_track_id)
```

Reproduced state changes (raw):

```
[Frame 00000] State Change: IDLE -> INITIALIZING   (Target ID: 1,   Anchors: 1)
[Frame 00188] State Change: INITIALIZING -> TRACKING (Target ID: 1, Anchors: 10)
[Frame 00905] State Change: TRACKING -> LOST       (Target ID: 1,   Dynamic: 21)
[Frame 01751] State Change: LOST -> TRACKING       (Target ID: 350, t_lost: 28.35s)
[Frame 02110] State Change: TRACKING -> LOST       (Target ID: 350, Dynamic: 30)
[Frame 02460] State Change: LOST -> TRACKING       (Target ID: 468, t_lost: 11.83s)
[Frame 02513] State Change: TRACKING -> LOST       (Target ID: 468)
[Frame 02581] State Change: LOST -> TRACKING       (Target ID: 534, t_lost: 2.38s)
[Frame 02625] State Change: TRACKING -> LOST       (Target ID: 534)
[Frame 02684] State Change: LOST -> TRACKING       (Target ID: 544, t_lost: 1.98s)
```

These match the state changes recorded in `STATUS.md` for the accepted run. The replay is
therefore faithful on state and target id for all 2698 frames, so the per-frame geometry
and scores below come from the same trajectory the accepted log describes.

Anchor gallery determinism check (**VERIFIED FACT**): two independent replays to frame 188
produced `anchors (10, 512)`, identical element sums `82.79872131347656`, max abs diff
`0.0`, per-anchor cosine `[1.0] x 10`. The anchor gallery is bit-identical across runs.

Per-frame anchor availability: labels at frames 0, 60, 120, 180 were taken while the anchor
gallery still held 1-9 anchors (TRACKING starts at 188). The 10-anchor scores in §4 are
computed against the frame-188 gallery; §4 also gives the front statistics restricted to
frames >= 188.

---

## 3. Task 2 - hand box vs tracker output (per labeled frame)

Raw per-frame table from the run (`state` = tracker state at that frame; `drawn` = a green
target box exists; `iou_drawn` = IoU(hand box, drawn target box); `best_iou/id/tier/...` =
best-overlapping tracker box of ANY track, with its pipeline scores):

```
           frame           view         status          state          drawn      iou_drawn       best_iou        best_id      best_tier   best_sim_all best_sim_anchor any_iou_ge_0.3
               0          front            box   INITIALIZING              1         0.8966         0.8966              1           FULL         1.0000         1.0000              1
              60          front            box   INITIALIZING              1         0.8086         0.8086              1           FULL         0.9379         0.9379              1
             120          front            box   INITIALIZING              1         0.9782         0.9782              1           FULL         0.9633         0.9633              1
             180          front            box   INITIALIZING              1         0.8644         0.8644              1           FULL         0.9198         0.9198              1
             240          front            box       TRACKING              1         0.8116         0.8116              1           FULL         0.9454         0.9379              1
             300          front            box       TRACKING              1         0.6971         0.6971              1           FULL         0.9461         0.8975              1
             360          front            box       TRACKING              1         0.7905         0.7905              1           FULL         0.9500         0.8946              1
             420          front            box       TRACKING              1         0.7248         0.7248              1           FULL         0.9340         0.9128              1
             480          front            box       TRACKING              1         0.6857         0.6857              1           FULL         0.9365         0.9118              1
             540          front            box       TRACKING              1         0.8281         0.8281              1           FULL         0.9486         0.8995              1
             600          front            box       TRACKING              1         0.7955         0.7955              1           FULL         0.9517         0.9207              1
             660          front            box       TRACKING              1         0.7424         0.7424              1           FULL         0.9424         0.9157              1
             720          front            box       TRACKING              1         0.7972         0.7972              1           FULL         0.9649         0.9323              1
             780          front            box       TRACKING              1         0.7987         0.7987              1           FULL         0.9439         0.9148              1
             840          front            box       TRACKING              1         0.9112         0.9112              1           FULL         0.9372         0.9018              1
             900          front            box       TRACKING              1         0.7396         0.7396              1           FULL         0.9101         0.8475              1
            2085           back            box       TRACKING              1         0.5302         0.5302            350           FULL         0.8574         0.7485              1
            2105           back            box       TRACKING              1         0.7513         0.7513            350           FULL         0.7777         0.6571              1
            2125           back            box           LOST              0                        0.7321            412           NONE         0.6669         0.6096              1
            2145           back            box           LOST              0                        0.5530            412           FULL         0.7581         0.6748              1
            2165           back            box           LOST              0                        0.0000            427           FULL         0.4770         0.4238              0
            2175           back    not_visible           LOST              0
            2195           back            box           LOST              0                        0.7290            435           FULL         0.8261         0.7260              1
            2205           back            box           LOST              0                        0.9252            435        PARTIAL         0.7627         0.6736              1
            2225           back            box           LOST              0                        0.7753            435           FULL         0.7517         0.6811              1
            2245           back            box           LOST              0                        0.7497            435           FULL         0.7627         0.7155              1
            2265           back            box           LOST              0                        0.8502            435           FULL         0.6768         0.6198              1
            2285           back            box           LOST              0                        0.1015            447           FULL         0.5205         0.4567              0
            2305           back            box           LOST              0                        0.6024            458           FULL         0.7164         0.6494              1
            2325           back    not_visible           LOST              0
            2345           back            box           LOST              0                        0.2946            468           NONE         0.7395         0.6187              0
            2365           back            box           LOST              0                        0.2779            468           FULL         0.6968         0.6314              0
            2385           back            box           LOST              0                        0.6252            468           FULL         0.7240         0.6481              1
            2405           back            box           LOST              0                        0.7192            468           FULL         0.7674         0.6653              1
            2425           back            box           LOST              0                        0.7112            468        PARTIAL         0.7584         0.6574              1
            2445           side            box           LOST              0                        0.8047            468           FULL         0.7956         0.7237              1
            2465           side            box       TRACKING              1         0.8245         0.8245            468           FULL         0.8348         0.7863              1
            2485           side            box       TRACKING              1         0.8642         0.8642            468           FULL         0.8194         0.7671              1
            2565           back            box           LOST              0                        0.8523            534           FULL         0.7594         0.6720              1
            2585           back            box       TRACKING              1         0.8078         0.8078            534           FULL         0.7690         0.6805              1
            2605           back            box       TRACKING              0                        0.7339            534           FULL         0.6851         0.5938              1
            2625           back            box           LOST              0                        0.6815            534        PARTIAL         0.7118         0.5972              1
            2645           back            box           LOST              0                        0.6779            544           FULL         0.7277         0.6357              1
            2655           side            box           LOST              0                        0.7732            544           FULL         0.7221         0.6233              1
```

### 3.1 Distributions (MEASUREMENT)

Best IoU between the hand box and ANY tracker box, per view:

```
front n= 16 min=0.686 p50/p90/p95/p99/max=[0.798, 0.904, 0.928, 0.968, 0.978] | >=0.3: 16/16
back  n= 22 min=0.000 p50/p90/p95/p99/max=[0.715, 0.846, 0.852, 0.910, 0.925] | >=0.3: 18/22
side  n=  4 min=0.773 p50/p90/p95/p99/max=[0.815, 0.852, 0.858, 0.863, 0.864] | >=0.3: 4/4
```

Green box drawn: 21 of 42 hand-box frames - front 16/16, back 3/22 (2085, 2105, 2585),
side 2/4 (2465, 2485).

IoU(hand box, drawn target box), 21 frames:

```
IoU count=21 min=0.530 p50/p90/p95/p99/max=[0.799, 0.897, 0.911, 0.965, 0.978]
IoU>=0.5: 21 | IoU>=0.3: 21 | IoU<0.1: 0 | IoU==0: 0
```

### 3.2 (a) Target visible (status=box) but state != valid (no green box)

**MEASUREMENT**: 21 frames, of which **17 were detected by YOLO/tracker with IoU >= 0.3**
to the hand box; 4 were not (2165 IoU 0.000, 2285 0.102, 2345 0.295, 2365 0.278).

| Quantity over the 21 frames | min | p50 | p90 | max |
|---|---|---|---|---|
| best IoU (any track) | 0.000 | 0.719 | 0.850 | 0.925 |
| best candidate `sim_anchor` | 0.424 | 0.648 | 0.716 | 0.726 |
| best candidate `sim_all` | 0.477 | 0.728 | 0.767 | 0.826 |

Counts: `sim_anchor >= 0.60` for 17/21; `sim_all >= 0.78` for 2/21; `sim_all >= 0.83` for
0/21. Best-box tiers: FULL 16, PARTIAL 3, NONE 2.

Counts are on different subsets: 17/21 had a tracker box at IoU >= 0.3 (geometry match),
17/21 had a best-candidate `sim_anchor >= 0.60`, and **15/21 satisfy both**. In those 15
frames (all `state=LOST`) the true-target person was present as a tracker box with a
plausible anchor score and still no green box was drawn, i.e. the acceptance gates, not the
detector, kept it out. Frames 2605 and 2625 were detected but scored `sim_anchor` 0.5938 /
0.5972 (just under the floor). The 4 frames with best IoU < 0.3 (2165, 2285, 2345, 2365) are
geometry misses; two of them (2345, 2365) still score 0.6187 / 0.6314 against a box that
only partially overlaps the hand box, so those scores do not describe the target's appearance.

Examples with score context (raw lines from the run):
```
frame  2125 back  state=LOST best_iou=0.732 id=412 tier=NONE    sim_all=0.6669 sim_anchor=0.6096
frame  2195 back  state=LOST best_iou=0.729 id=435 tier=FULL    sim_all=0.8261 sim_anchor=0.7260
frame  2205 back  state=LOST best_iou=0.925 id=435 tier=PARTIAL sim_all=0.7627 sim_anchor=0.6736
frame  2445 side  state=LOST best_iou=0.805 id=468 tier=FULL    sim_all=0.7956 sim_anchor=0.7237
frame  2605 back  state=TRACKING but no box: best_iou=0.734 id=534 tier=FULL sim_all=0.6851 sim_anchor=0.5938
```
The last row is the only TRACKING row without a drawn box: `sim_anchor 0.5938` is below
`T_ANCHOR_FLOOR = 0.60`, which is the documented draw condition for TRACKING.

### 3.3 (b) Green box drawn: is it on the hand-labeled person?

**MEASUREMENT**: 21 drawn frames; every drawn box had IoU >= 0.530 to the hand box
(p50 0.799, p90 0.897, max 0.978). **No drawn box with IoU < 0.1** in this sample, so there
is no POSSIBLE false-lock candidate to list here. Lowest values: frame 2085 IoU 0.5302,
frame 2165 is not drawn. This is a statement about 21 frames of one video, not a rate.

### 3.4 (c) Hand label not_visible but a green box is drawn

**MEASUREMENT**: 2 `not_visible` labeled frames (2175, 2325), **0** with a drawn box
(both `state=LOST`, `drawn=0`). Raw:

```
  not_visible frames: 2  with drawn box: 0
```

---

## 4. Task 3 - embeddings from the hand boxes (tracker-independent labels)

42 hand boxes were cropped from the decoded video and embedded with the same
`PersonReIDExtractor` (osnet_x1_0 MSMT17), then compared against the 10 frame-188 anchors
(top-3 mean cosine, the pipeline formula). Cross-check: these per-view distributions match
the in-run hand-box values (front p50 0.897 vs 0.8969, back 0.653 vs 0.6533, side 0.743 vs
0.7434), which supports that the frame-188 gallery is the gallery the pipeline actually used
after 188.

### 4.1 Same-person / cross-view cosine between hand-box embeddings (MEASUREMENT)

```
front-front  gap>=0  n=240 min=0.689 p50/p90/p95/max=[0.861, 0.915, 0.927, 0.944]
front-front  gap>=20 n=240 min=0.689 p50/p90/p95/max=[0.861, 0.915, 0.927, 0.944]
back-back    gap>=0  n=462 min=0.291 p50/p90/p95/max=[0.726, 0.838, 0.870, 0.933]
back-back    gap>=20 n=460 min=0.291 p50/p90/p95/max=[0.725, 0.837, 0.870, 0.933]
side-side    gap>=0  n= 12 min=0.601 p50/p90/p95/max=[0.747, 0.868, 0.870, 0.870]
side-side    gap>=20 n= 12 min=0.601 p50/p90/p95/max=[0.747, 0.868, 0.870, 0.870]
front-back   gap>=0  n=352 min=0.355 p50/p90/p95/max=[0.617, 0.693, 0.710, 0.769]
front-back   gap>=20 n=352 min=0.355 p50/p90/p95/max=[0.617, 0.693, 0.710, 0.769]
front-side   gap>=0  n= 64 min=0.504 p50/p90/p95/max=[0.665, 0.778, 0.795, 0.814]
front-side   gap>=20 n= 64 min=0.504 p50/p90/p95/max=[0.665, 0.778, 0.795, 0.814]
back-side    gap>=0  n= 88 min=0.411 p50/p90/p95/max=[0.698, 0.785, 0.801, 0.845]
back-side    gap>=20 n= 87 min=0.411 p50/p90/p95/max=[0.697, 0.784, 0.796, 0.845]
```

Dropping pairs closer than 20 frames changes almost nothing (the labels are already 20-60
frames apart). Front-front (median 0.861) is clearly separated from front-back (0.617);
back-back (0.726) is in between.

### 4.2 `sim_anchor` of the TRUE target's hand box, by view (MEASUREMENT)

Scores against the 10 frame-188 anchors:

```
front n=16 min=0.8483 p50=0.8969 p90=0.9437 p95=0.9550 max=0.9583
back  n=22 min=0.4342 p50=0.6533 p90=0.6941 p95=0.6949 max=0.7554
side  n= 4 min=0.5754 p50=0.7434 p90=0.7857 p95=0.7876 max=0.7896
back+side n=26 p50/p90/p95/max = [0.6609, 0.7328, 0.7714] 0.7896
front (frames >=188 only) n=12 min=0.8483 p50=0.8888 p90=0.9176 max=0.9335
```

Threshold crossings for the TRUE target (descriptive only):

| Group | > `T_ANCHOR_FLOOR` 0.60 | > `T_ACCEPT_LONG` 0.78 |
|---|---|---|
| back + side (n=26) | 21/26 (80.8%) | 1/26 (3.8%) |
| front (n=16) | 16/16 (100.0%) | 16/16 (100.0%) |

Per-box detail (sequential decode; 42 rows) is printed by the auxiliary run; the
low back-side values are 2165 0.4342, 2285 0.5546, 2625 0.5367, 2645 0.5341, 2655 0.5754.

### 4.3 Different-person reference (MEASUREMENT)

A. Other tracker boxes in the SAME labeled frames with IoU < 0.1 to the hand box:

```
count=170 (frames=42) min=0.205 p50/p90/p95/p99/max=[0.374, 0.492, 0.519, 0.551, 0.686]
```

B. All scored people in frames 936-1719 (ground truth: target absent):

```
boxes scored=6640 distinct tracks=134 frames=784
sim_anchor min=0.171 p50/p90/p95/p99/max=[0.369, 0.505, 0.539, 0.605, 0.761]
sim_all    min=0.176 p50/p90/p95/p99/max=[0.393, 0.537, 0.582, 0.670, 0.798]
tiers: {'FULL': 5745, 'NONE': 278, 'PARTIAL': 617}
top-10 sim_anchor boxes in that range:
  frame 1062 id=204 sim_anchor=0.761 sim_all=0.798 tier=FULL area=143136
  frame 1061 id=204 sim_anchor=0.748 sim_all=0.789 tier=FULL area=146700
  frame 1066 id=204 sim_anchor=0.734 sim_all=0.782 tier=FULL area=171216
  frame 1140 id=246 sim_anchor=0.721 sim_all=0.768 tier=FULL area=56109
  frame 1141 id=246 sim_anchor=0.718 sim_all=0.767 tier=FULL area=54036
  frame 1145 id=246 sim_anchor=0.714 sim_all=0.766 tier=FULL area=53492
  frame 1065 id=204 sim_anchor=0.708 sim_all=0.763 tier=FULL area=190684
  frame 1142 id=246 sim_anchor=0.707 sim_all=0.757 tier=FULL area=54008
  frame 1146 id=246 sim_anchor=0.705 sim_all=0.762 tier=FULL area=58100
  frame 1064 id=204 sim_anchor=0.704 sim_all=0.754 tier=FULL area=189202
```

Descriptive comparison (no conclusion drawn): the TRUE target's back/side `sim_anchor`
scores (p50 0.661, max 0.7896) overlap the upper tail of the absent-period different-person
scores (p99 0.605, max 0.761). Two different-person tracks (204, 246) reach 0.70-0.76 while
the target is, per ground truth, not visible.

### 4.4 What the numbers say about the gates (descriptive, not a proposal)

- In 15/21 visible-but-undrawn frames the target WAS a tracker box (IoU >= 0.3) with
  `sim_anchor` >= 0.60, while `sim_all` stayed at p50 0.728 / max 0.826 (0/21 reached 0.83).
- One further frame (2605) is the opposite case: state TRACKING but no box because
  `sim_anchor` 0.5938 < 0.60; frame 2625 was detected with 0.5972.
- 4/21 are geometry misses (best IoU < 0.3), not ReID rejections.
- Every drawn box (21) overlapped the hand box at IoU >= 0.53.

No constant was changed, no E1/E2/E3 option was recommended, and no claim about a false-lock
rate is made (`false lock` appears here only as the IoU-based screen requested in Task 2b,
which found no IoU<0.1 case in 21 frames of one video).

---

## 5. Limits

- One video (`videos/5.mp4`), 44 labels total (42 boxes: 16 front, 22 back, 4 side).
  This is a spot sample; no percentage here is a rate for the system.
- Hand boxes are drawn by eye and differ from tracker boxes (IoU 0.53-0.98 on drawn frames),
  so IoU is not an identity ground truth; it only measures geometry agreement.
- The 4 front labels at frames 0-180 were taken while the gallery had 1-9 anchors; their
  "10-anchor" scores are computed post hoc. Front statistics restricted to frames >= 188 are
  given (n=12).
- The different-person reference B assumes "target absent" across 936-1719 exactly as the
  manual GT says; tracks 204/246 (area up to ~190k px) cannot be identity-verified here.
- Frame seeking (`CAP_PROP_POS_FRAMES`) and sequential decoding do not return identical
  pixels in this file: an auxiliary pass using seeks produced slightly different crop
  embeddings (e.g. frame 2195 anchor score 0.6704 sequential vs 0.7451 via seek). All
  reported numbers use sequential decoding, and they match the in-run hand-box
  distributions. MEASUREMENT, one file.
- Percentiles on n=4 (side) are not meaningful beyond showing spread.

## 6. FINAL CRITIQUE

**Three weaknesses**
1. **Small, self-selected label sample.** 42 boxes from one video, hand-drawn on frames
   chosen for analysis (not randomly). The front/tracking phase is over-represented (16/42),
   the side phase has only 4 boxes. Any conclusion about "the system" would over-reach.
2. **IoU is not identity.** A box can overlap the hand box and still be another person
   (e.g. 2085 IoU 0.53 with a 574x906 box that is heavily border-clipped), and the hand box
   itself is imprecise. The identity question in 3b is answered only geometrically.
3. **The different-person reference is not a clean negative set.** It is built from tracker
   boxes whose identity is unknown; frames 936-1719 are taken as "target absent" from manual
   GT, and two tracks reach 0.70-0.76 there, close to true-target back scores
   (p50 0.661). Without an eye check of those specific boxes the separation between
   "same person, back view" and "similar-looking different person" is not established.

**What would change the conclusion**
- Eye-verifying tracks 204 and 246 in frames 1062/1140 (are they really not the target?)
  would either confirm or strongly weaken the overlap finding in §4.3.
- A second video with a similarly dressed person, plus more side-view labels, would show
  whether the back/side score overlap is a property of the model or of this clip.
- Labels drawn on a random frame subset (not analysis-chosen frames) would turn the current
  counts into something closer to a rate.
- A log that records candidate identity per frame would let the undrawn frames be attributed
  to a specific track without manual inspection.

## 7. Reproduction

```bash
cd /home/foxsec/drone_project
. venv/bin/activate
python3 analyze_eye_labels.py                # full replay + analysis, writes outputs/eye_label_analysis.csv
python3 analyze_eye_labels.py --verify-only 200   # faithfulness smoke test, writes nothing
```

Temp gallery dirs under `/tmp/eye_labels_gallery_*` and `/tmp/eye_labels_anchor*` were left
in place (no files deleted).

## 8. Files touched

| File | Status |
|---|---|
| `analyze_eye_labels.py` | created (new) |
| `outputs/eye_label_analysis.csv` | created (new, 44 data rows) |
| `EYE_LABEL_REPORT.md` | created (this file) |
| `STATUS.md` | appended section "Eye-label analysis" (timestamped backup in `backups/`) |
| `roadmap_data.json` | new log entry + pending item (timestamped backup in `backups/`) |
| `ROADMAP.md`, `outputs/progress_roadmap.png` | regenerated by `update_roadmap.py` |
| `ground_truth/*` | **unchanged** (read-only) |
| `follow_target.py`, `evaluate.py`, `calibrate_thresholds.py`, `GEMINI_BRIEF.md`, `.gitignore`, `logs/*` | **unchanged** |
