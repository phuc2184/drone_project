# STATUS REPORT - Autonomous Person-Following Drone Tracking (PX4)
**Date:** 2026-10-20  
**Phase:** Step 3 / PART C (Completed)  
**Main File:** `/home/foxsec/drone_project/follow_target.py`  
**Test Video:** `/home/foxsec/drone_project/videos/5.mp4` (2698 frames, 1080p, 30 FPS)

---

## 1. Project Overview & Target Identity
- **True Target in `videos/5.mp4`:** **Man in denim jacket + black turtleneck**.
- **Secondary / Distractor Person:** Man in **grey hoodie with glasses** (who takes over the center of the frame at frame 904).
- **Core Priority:** FALSE LOCK (locking onto the wrong person) is unacceptable. Missed or delayed re-acquisition is preferable to a false lock.

---

## 2. PART B - Implementation & Acceptance Test

### B1. Completion of R1 - R6
All items in R1 through R6 are now **DONE**:
| Rule | Status | Implementation Details |
|---|---|---|
| **R1** | **DONE** | OSNet MSMT17 weights (`osnet_x1_0_msmt17.pth`) loaded from `weights/`, built with `pretrained=False`, weights loaded via `torchreid.utils.load_pretrained_weights`. |
| **R2** | **DONE** | `QualityTier` (FULL, PARTIAL, NONE). Verification allows FULL and PARTIAL. Acceptance allows both with +0.05 penalty for PARTIAL. Dynamic/anchor additions strictly require FULL. |
| **R3** | **DONE** | `T_ANCHOR_FLOOR = 0.60`. Verification requires `sim_anchor >= 0.60`. Dynamic gallery addition requires `sim_anchor >= 0.80`, `sim_all >= 0.75`, >= 30 frames since suspicious event. 60-frame rollback (`rollback_dynamic_gallery`) triggers on continuity failure and verification failure. |
| **R4** | **DONE** | `CONSECUTIVE_EMB_SIM_THRESH = 0.50`. In TRACKING, consecutive embedding cosine < 0.50 triggers immediate rollback, resets state to LOST, and exits update immediately without gallery update. Continuity check also enforced in INITIALIZING. |
| **R5** | **DONE** | Anchors require >= 5 frames spacing (`ANCHOR_FRAME_SPACING = 5`) and cosine sim <= 0.95 (`ANCHOR_MAX_SIM = 0.95`). Initial click frame enrolled at frame 0 is not added twice. |
| **R6** | **DONE** | Gallery crops saved to `outputs/gallery_debug/`. CSV columns `sim_anchor`, `sim_dynamic`, `quality_tier` present. HUD displays `all X \| anchor Y`. Restored post-run summary reporting formatted ranges of `suspected_false_lock_frames` (`_group_ranges`). |

### B2. Constants Changed
- **No threshold constants were changed** (in strict adherence to Rule 6; threshold calibration is reserved for PART C).

### B3. Acceptance Test Results (`output_step2c.mp4`)
- **Command Executed:**
  ```bash
  python3 follow_target.py --video videos/5.mp4 --out outputs/output_step2c.mp4 --target-id 1 --headless
  ```
- **Execution Summary:**
  - **ReID Weights File:** `osnet_x1_0_msmt17.pth`
  - **Frames Processed:** 2698 / 2698 (100%)
  - **Total Processing Time:** 174.72 s
  - **Average Pipeline FPS:** **15.44 FPS**
  - **Annotated Video:** `/home/foxsec/drone_project/outputs/output_step2c.mp4`
  - **CSV Log:** `/home/foxsec/drone_project/logs/tracking_log_20261020_111522.csv`
  - **Gallery Crops:** `/home/foxsec/drone_project/outputs/gallery_debug/` (73 crops)

- **State Transitions:**
  | Frame | Transition | Target ID | Details |
  |---|---|---|---|
  | **0** | `IDLE -> INITIALIZING` | 1 | Seeded at frame 0 with initial click crop (Anchor 1). |
  | **188** | `INITIALIZING -> TRACKING` | 1 | Collected 10 diverse anchors satisfying `<= 0.95` max cosine. |
  | **905** | `TRACKING -> LOST` | 1 | ID switch at 904 triggered anchor floor violation (`sim_anchor=0.5153 < 0.60`). 2 consecutive verification fails triggered LOST and rolled back 9 dynamic embeddings. |
  | **1751** | `LOST -> TRACKING` | 350 | True target returned into frame; re-acquired after 8 confirm frames (`all=0.9012`, `anchor=0.8555`). |
  | **2110** | `TRACKING -> LOST` | 350 | Target left frame border. |
  | **2460** | `LOST -> TRACKING` | 468 | True target re-acquired (back view; `all=0.8433`, `anchor=0.8098`). |
  | **2513** | `TRACKING -> LOST` | 468 | Target turned/occluded (`tier=NONE` for 3 frames). |
  | **2581** | `LOST -> TRACKING` | 534 | True target re-acquired (back view; `all=0.7868`, `anchor=0.6969`). |
  | **2625** | `TRACKING -> LOST` | 534 | Target leaving frame; `sim_anchor` dropped below 0.60 for 2 frames. |
  | **2684** | `LOST -> TRACKING` | 544 | True target re-acquired (side view; `all=0.8441`, `anchor=0.7366`). |

- **Suspected False-Locks Analysis:**
  - Reported frames: `['904', '2510', '2605', '2608', '2624']`
  - **Frame 904:** Distractor person took ID 1. `sim_anchor = 0.5153 < 0.60`. Target bounding box was suppressed (`output_box = None`), so **no green TARGET box was drawn on the distractor**. Next frame (905) immediately transitioned to `LOST`.
  - **Frames 2510, 2605, 2608, 2624:** All occur on the true target when turning away or leaving frame. Target box was safely suppressed on any momentary dips below 0.60.
  - **Pass Condition:** The distractor person (or any non-target person) **never** received the green TARGET box. **PASS**.

---

## 3. PART C - STEP 3: Calibration & Evaluation
> **INVALID - computed with the old circular ground truth / ID-based false-lock rule, see section 7**

### C1. Ground-Truth File (`ground_truth/5.csv`)

Already created in the previous session. Format: `start_frame,end_frame,target_visible,view`

```
start_frame,end_frame,target_visible,view
0,904,1,front
905,1750,0,none
1751,2109,1,front
2110,2459,0,none
2460,2513,1,back
2514,2580,0,none
2581,2625,1,back
2626,2683,0,none
2684,2697,1,side
```

**How to create/extend this file for a new video:**
1. Play the video in VLC with frame counter visible (View → Status Bar).
2. For each contiguous segment, note: first frame where target appears, last frame before target disappears.
3. Record `target_visible=1` with the dominant view (`front`/`back`/`side`); use `target_visible=0` for absence.
4. Write rows in order, non-overlapping, covering the full video length.
5. Optionally add a `target_id` column (tracker ID for that segment) for finer false-lock checking.

---

### C2. `calibrate_thresholds.py` Results

**Command:**
```bash
python3 calibrate_thresholds.py --video videos/5.mp4 --gt ground_truth/5.csv --stride 5
```

**Embedding Extraction Summary:**
| Category | Count |
|---|---|
| Target embeddings (total) | 274 |
| — Front view | 252 |
| — Back view | 19 |
| — Side view | 3 |
| Non-target embeddings | 3,074 |

**Different-Person Similarity Percentiles (842,276 cross-person pairs):**
| Percentile | Score |
|---|---|
| P50 (Median) | 0.3583 |
| P90 | 0.5149 |
| P95 | 0.5692 |
| P98 | 0.6421 |
| **P99** | **0.6866** ← minimum safe T_ACCEPT |
| P99.9 | 0.7888 |
| Maximum | 0.9856 |

**Same-Person Similarity & Retention at P99 = 0.6866:**
| Category | N | Mean | P05 | P50 | Retained ≥ P99 |
|---|---|---|---|---|---|
| All views | 37,401 | 0.841 | 0.651 | 0.868 | **91.8%** |
| Front-Front | 31,626 | 0.868 | 0.757 | 0.877 | **98.0%** |
| Back-Back | 171 | 0.739 | 0.500 | 0.753 | **64.3%** |
| Side-Side | 3 | 0.927 | 0.904 | 0.933 | **100.0%** |
| Front-vs-Back | 4,788 | 0.684 | 0.509 | 0.693 | **53.8%** |

**Gallery Score Distributions (Top-3 Mean, as in pipeline):**
| View | N | Mean Anchor | ≥ 0.60 Floor | ≥ 0.78 Accept | ≥ P99 |
|---|---|---|---|---|---|
| Front | 252 | 0.8982 | 100.0% | 98.0% | 98.8% |
| Back | 19 | 0.6968 | 84.2% | **26.3%** | 57.9% |
| Side | 3 | 0.7098 | 100.0% | **0.0%** | 100.0% |
| All | 274 | 0.8822 | 98.9% | 92.0% | 96.0% |

**Rule Verification:**
- `T_ACCEPT_LONG = 0.78 >= P99 = 0.6866` → **RULE COMPLIANCE: PASS** ✅
- Back-view retention at `T_ACCEPT = 0.78`: **26.3%** → explains why back-view reacquisition is slow.
- Different-person anchor-only P99 = 0.6746; `T_ANCHOR_FLOOR = 0.60` is below P99 (intentionally permissive floor).
- **Histogram plots saved:** `outputs/calibration_histograms.png`

---

### C3. `evaluate.py` Results

**Command:**
```bash
python3 evaluate.py --log logs/tracking_log_20261020_111522.csv --gt ground_truth/5.csv --fps 30.0
```

**Overall Metrics:**
| Metric | Value | Requirement |
|---|---|---|
| Total Video Frames | 2,698 (89.93 s) | — |
| Tracked Frames | 1,187 (44.0%) | — |
| **FALSE LOCK Frames** | **0** | **~0 ✅** |
| **FALSE LOCK Rate** | **0.00%** | **~0% ✅** |
| Reacquisition Success | 4/4 (100.0%) | — |
| Avg Reacquisition Time | 0.00 s | — |
| Target ID Switches | 4 | Expected (tracker ID changes after LOST) |

**Per-View Breakdown:**
| View | Visible Frames | Tracked Frames | Coverage | Reacq Success | Avg Delay | False Locks |
|---|---|---|---|---|---|---|
| front | 1,264 | 1,076 | 85.1% | 1/1 | 0.00 s | 0 |
| back | 99 | 97 | 98.0% | 2/2 | 0.00 s | 0 |
| side | 14 | 14 | 100.0% | 1/1 | 0.00 s | 0 |

**Spot-Check Mode:** Available via `python3 evaluate.py --spot-check outputs/output_step2c.mp4 [--stride 30]`

---

## 4. Key Observations & Notes for Supervisor
> **INVALID - computed with the old circular ground truth / ID-based false-lock rule, see section 7**

### 4a. Neutralized statements (valid for the supervisor)
1. **False locks:** NOT VERIFIED. The earlier report of 0 false locks used `ground_truth/5.csv`, which was derived from the baseline tracker log (circular). No false-lock statement can be based on it. Re-evaluation against `ground_truth/5_manual.csv` is done in section 8 (STEP B), numbers only.
2. **T_ACCEPT_LONG = 0.78 vs P99 = 0.6866:** NOT VERIFIED. That P99 came from labels in the circular `5.csv`; the comparison says nothing about the true target. A recalibration on `5_manual.csv` is scheduled (STEP D), not done.
3. **Back-view difficulty (front-vs-back P50 = 0.693, 53.8% above P99, back coverage 98%):** NOT VERIFIED, all figures derive from the circular ground truth. The statement that the target is mostly seen from behind in 2110-2459 and 2565-2697 does come from the manual labels (section 7), but the similarity/coverage numbers do not.
4. **T_ANCHOR_FLOOR = 0.60 below different-person P99 = 0.6746 / "no false locks occurred":** NOT VERIFIED, same circular source. The design description (floor is a rejection guard, T_ACCEPT is the accept gate) is a code property (VERIFIED FACT in `follow_target.py`), but the empirical claim is not.
5. **Reacquisition delay 0.00 s:** NOT VERIFIED. It was produced by the old metric against the circular GT; the revised `evaluate.py` now reports delays in frames and flags anything below 30 frames as boundary uncertainty. Current numbers are in section 8 (STEP B).
6. **FPS:** the 15.44 FPS of the PART B run (174.72 s / 2698 frames) is a MEASUREMENT independent of ground truth (VERIFIED by timing, not by GT). `evaluate.py` prints no FPS because it reads a log. FPS of the PART D runs is unreliable because they ran partly concurrently / shared the GPU.

### 4b. Original text (history - invalid, kept only as record)

1. **No false locks** in the entire 2698-frame video (0.00%). The primary safety requirement is met.
2. **T_ACCEPT_LONG = 0.78 is well above P99 (0.6866)** — safe margin confirmed by calibration data.
3. **Back-view reacquisition is the hardest case**: front-vs-back cosine similarity drops to P50=0.693, and only 53.8% of cross-view pairs exceed the P99 threshold. Back coverage is still 98% because the pipeline re-accepted the target at slightly lower scores (PARTIAL quality tolerance).
4. **T_ANCHOR_FLOOR = 0.60 is below the P99 of different-person scores (0.6746)** — this is a deliberate design choice: the floor is used as a *rejection* guard for obvious impostors, not as a final accept gate (T_ACCEPT does that). No false locks occurred despite this.
5. **Reacquisition time reported as 0.00 s**: the evaluate.py measures delay from when target_visible=1 starts; since the tracker re-acquired on the same or next frame where a FULL-quality crop was available, the log rows show TRACKING already in effect at the first visible frame of each reappearance interval.
6. **Avg FPS not printed by evaluate.py** (it reads a pre-recorded log, not a live run). FPS was 15.44 from the PART B run.

---

## 5. PART D - View-Robustness Experiments
> **INVALID - computed with the old circular ground truth / ID-based false-lock rule, see section 7**

All four variants run on `videos/5.mp4` with `--target-id 1 --headless`.
No threshold constants changed. All switches default `False`; each run uses only one optional flag.

**Weights downloaded:** `osnet_ain_x1_0_msmt17.pth` (17 MB) from `huggingface.co/kaiyangzhou/osnet`.

---

### D1. Experiment Summary Table

| Variant | Model | FPS | FL Frames | FL % | Reacq | Front cover | Back cover | Side cover | LOST/TRACK cycles | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | osnet_x1_0 | **15.53** | **0** | **0.00% ✅** | **4/4 ✅** | 85.1% | 98.0% | 100.0% | 5 | Reference |
| **E1** (view chain) | osnet_x1_0 | 15.76 | **31** | **1.15% ❌** | 4/4 | 85.1% | 98.0% | 100.0% | 6 | False lock during invisible period (frames 2142–2160, 2671–2683) |
| **E2** (color cue) | osnet_x1_0 | 14.46 | 0 | 0.00% ✅ | **3/4 ❌** | 85.1% | **51.5%** | 85.7% | 4 | Missed back-view window 2581–2625 entirely |
| **E3** (osnet_ain_x1_0) | osnet_ain_x1_0 | **15.84** | 3 | 0.11% ⚠️ | **3/4 ❌** | **92.6%** | **49.5%** | 100.0% | 4 | 3 FL frames at 2681–2683 (re-acq before target visible); missed back window 2581–2625 |

### D2. Per-Experiment State Transitions

**Baseline (osnet_x1_0):**
| Frame | Event | t_lost |
|---|---|---|
| 0→188 | INITIALIZING → TRACKING | — |
| 905 | TRACKING → LOST | 0.00 s |
| 1751 | LOST → TRACKING (ID 350, front) | 28.35 s |
| 2110 | TRACKING → LOST | 0.10 s |
| 2460 | LOST → TRACKING (ID 468, back) | 11.83 s |
| 2513 | TRACKING → LOST | 0.10 s |
| 2581 | LOST → TRACKING (ID 534, back) | 2.38 s |
| 2625 | TRACKING → LOST | 0.00 s |
| 2684 | LOST → TRACKING (ID 544, side) | 1.98 s |

**E1 (USE_VIEW_CHAIN_UPDATE=True):**
| Frame | Event | t_lost | Notes |
|---|---|---|---|
| 0→188 | INITIALIZING → TRACKING | — | |
| 869 | View chain addition | — | Size 1/30 |
| 905 | TRACKING → LOST | 0.00 s | Rolled back 9 dyn + 1 view-chain |
| 1751 | LOST → TRACKING (ID 350) | 28.35 s | |
| 2088, 2103 | View chain additions | — | Size 1→2/30 |
| 2110 | TRACKING → LOST | 0.10 s | ViewChain: 2 |
| **2142** | **LOST → TRACKING (ID 412)** | **1.17 s** | **⚠️ FALSE LOCK: target invisible per GT** |
| **2160** | **TRACKING → LOST** | 0.00 s | Rolled back 0 dyn + 1 view-chain |
| 2460 | LOST → TRACKING (ID 468, back) | 10.05 s | |
| 2513 | TRACKING → LOST | 0.10 s | |
| 2581 | LOST → TRACKING (ID 534, back) | 2.38 s | |
| 2625 | TRACKING → LOST | 0.00 s | |
| **2671** | **LOST → TRACKING (ID 544)** | **1.54 s** | **⚠️ 13 FL frames (GT invisible until 2684)** |

**E2 (USE_COLOR_CUE=True):**
| Frame | Event | t_lost | Notes |
|---|---|---|---|
| 0→188 | INITIALIZING → TRACKING | — | |
| 905 | TRACKING → LOST | 0.00 s | |
| 1751 | LOST → TRACKING (ID 350, front) | 28.35 s | |
| 2110 | TRACKING → LOST | 0.10 s | |
| 2462 | LOST → TRACKING (ID 468, back) | 11.90 s | 2-frame delay vs baseline |
| 2513 | TRACKING → LOST | 0.10 s | |
| *(2581–2625 back window missed)* | | | |
| 2686 | LOST → TRACKING (ID 544, side) | 5.90 s | Combined gap 3+4 = 5.73s lost |

**E3 (osnet_ain_x1_0):**
| Frame | Event | t_lost | Notes |
|---|---|---|---|
| 0→**94** | INITIALIZING → TRACKING | — | Faster convergence than baseline (188) |
| 905 | TRACKING → LOST | 0.00 s | Rolled back 10 dyn |
| 1751 | LOST → TRACKING (ID 350, front) | 28.35 s | |
| 2110 | TRACKING → LOST | 0.10 s | |
| 2464 | LOST → TRACKING (ID 468, back) | 11.96 s | 4-frame delay |
| 2513 | TRACKING → LOST | 0.10 s | |
| *(2581–2625 back window missed)* | | | |
| **2681** | LOST → TRACKING (ID 544) | 5.73 s | **3 FL frames: re-acq at 2681, GT visible from 2684** |

---

### D3. Observations (raw numbers, no recommendations)

1. **E1 (view chain) introduced false locks (31 frames, 1.15%).**
   The view-chain addition at frame 869 (when a borderline embedding was accepted) was rolled back when LOST occurred at frame 905. However, during the second gap (frames 2110–2459, target invisible), the view-chain embeddings added at frames 2088/2103 relaxed the similarity threshold enough to accept track ID 412 at frame 2142 — locking onto a wrong person for 18 frames. A further 13 false-lock frames occurred at the end (frames 2671–2683) where E1 re-acquired before the target was physically visible per ground truth.

2. **E2 (color cue) hurt back-view coverage (98.0% → 51.5%) with 0 false locks.**
   The HSV histogram from the back view (dark jacket, ambiguous hue) is not discriminative enough vs. other people in similar clothing, causing the combined score to drop below T_ACCEPT. The 2581–2625 back window (44 frames) was completely missed. Front coverage was unaffected. FPS dropped from 15.53 → 14.46 (−7%) due to histogram computation.

3. **E3 (osnet_ain_x1_0) has the highest FPS (15.84) and better front coverage (85.1% → 92.6%), due to faster initialization (94 frames vs. 188).**
   However, it also missed the back window 2581–2625 and introduced 3 false-lock frames (re-acquired 3 frames before the target was visually present). Back coverage dropped 98.0% → 49.5%. ID switches: 3 (vs. 4 baseline).

4. **The 2581–2625 back view window** (44 frames, ~1.5s) is the hardest segment: `back-vs-back` cosine P50 = 0.739, and with a gallery built mostly on front-view embeddings, the score often falls below T_ACCEPT=0.78. Baseline managed this because its accumulated dynamic gallery was slightly better calibrated than AIN's different embedding space.

5. **LOST/TRACKING cycle counts** (number of LOST→TRACKING transitions + 1):
   - Baseline: 5 states (1 INIT→TRACK + 4 re-acqs)
   - E1: 6 states (added 1 spurious cycle at 2142)
   - E2: 4 states (missed 1 re-acq cycle)
   - E3: 4 states (missed 1 re-acq cycle)

---

## 6. Current Status & Next Actions
> **INVALID - computed with the old circular ground truth / ID-based false-lock rule, see section 7**
- **Current State:** **GT Correction & Evaluation Metric Overhaul Completed.**
- **Pending Actions:** Awaiting user approval of Task 1-4 reports before proceeding to any subsequent tasks (PART E / F not started).

---

## 7. GT Correction & Metric Revision
- **Ground Truth `5.csv` is INVALID**: The old `ground_truth/5.csv` was derived directly from the baseline log, not from an independent annotation. It is invalid.
- **`5_manual.csv` Replaces `5.csv`**: The user labeled the video by eye and produced `ground_truth/5_manual.csv` (`start_frame,end_frame,target_visible,view; 1 = clear, 2 = partially occluded, 0 = absent`). Its boundaries have an uncertainty of approx. +-30 frames. True absence occurs only around frames 905–1750 and 2505–2564; the target is visible (primarily from behind) during 2110–2459 and 2565–2697.
- **PART C and PART D Numbers Computed on `5.csv` are INVALID**: All coverage, false lock, and reacquisition figures previously calculated against `5.csv` are invalid.
- **ID-Based False-Lock Rule Was Wrong**: Deciding correctness based on whether a tracker ID is in a fixed set was fundamentally flawed because tracker IDs change after every `LOST` event. Visual inspection of E1 video confirms frame 2143 (green box [ID: 412]) is on the correct target (denim jacket, black turtleneck, backpack from behind). Calling ID 412 an "impostor" was an evaluation error.
- **Redefined Metric in `evaluate.py`**:
  - `FALSE LOCK = TARGET label while target_visible = 0`, strictly excluding (a) frames within 30 frames of any GT boundary and (b) frames with `target_visible = 2`.
  - Excluded frames are reported separately as "uncertain frames".
  - Coverage and reacquisition are evaluated on `target_visible = 1` frames outside the 30-frame boundary margin.
  - Reacquisition delays are reported in frames; delays below 30 frames are marked as within boundary margin uncertainty.
  - `--segments` option exports full-resolution first, mid, and last frames of every tracking segment to `outputs/check_frames/` for required visual verification.

---

## 8. Audit session 2026-10-24 (STEP A-C)

### STEP A - invalid banners (DONE)
- Banner `INVALID - computed with the old circular ground truth / ID-based false-lock rule, see section 7` added under sections 3, 4, 5 and 6.
- Section 4 split into 4a (neutralized statements, each marked VERIFIED FACT / MEASUREMENT / NOT VERIFIED) and 4b (original text kept only as history). No conclusion in 4a rests on `ground_truth/5.csv` or on the ID-based false-lock rule.

### STEP B - re-evaluation of the four existing Part D logs (DONE)

**Log to variant mapping - VERIFIED** (from the distinctive state transitions in section 5 D2 and the logs themselves; rows = 2698 each):
| Variant | Log | Distinguishing evidence |
|---|---|---|
| baseline | `logs/tracking_log_20261020_132332.csv` | TRACKING 188, 1751(350), 2460(468), 2581(534), 2684(544) |
| E1 | `logs/tracking_log_20261020_132641.csv` | extra TRACKING 2142(ID 412)-2160 and 2671(ID 544) |
| E2 | `logs/tracking_log_20261020_133004.csv` | re-acq 2462 and 2686 only, no 2581 |
| E3 | `logs/tracking_log_20261020_133334.csv` | INITIALIZING->TRACKING at 94, re-acq 2464 and 2681 |

**Commands (one per row, identical except the log):**
```
venv/bin/python evaluate.py --log logs/tracking_log_20261020_132332.csv --gt ground_truth/5_manual.csv --fps 30.0   # baseline
venv/bin/python evaluate.py --log logs/tracking_log_20261020_132641.csv --gt ground_truth/5_manual.csv --fps 30.0   # E1
venv/bin/python evaluate.py --log logs/tracking_log_20261020_133004.csv --gt ground_truth/5_manual.csv --fps 30.0   # E2
venv/bin/python evaluate.py --log logs/tracking_log_20261020_133334.csv --gt ground_truth/5_manual.csv --fps 30.0   # E3
```
The per-segment rows below come from the same logs and GT with an inline check: first TRACKING frame inside each target_visible=1 interval, and margin-clean visible/tracked counts per interval (same +-30 boundary rule as evaluate.py).

**All values are MEASUREMENTs.** FPS of the PART D runs is NOT VERIFIED / unreliable (partly concurrent, shared GPU); the logs contain no timing.

**Baseline**
| Metric | Value |
|---|---|
| Coverage, margin-clean (target_visible=1) | front 958/1115 = 85.9% ; back 0/28 = 0.0% ; side 0/0 = n/a |
| Coverage, raw (incl. margin) | front 1051/1239 = 84.8% ; back 69/270 = 25.6% ; side 59/103 = 57.3% |
| target_visible=2 tracked | 0/180 = 0.0% |
| FALSE LOCK (vis=0, outside margin) | 0 frames |
| "Uncertain" TRACKING frames | 8, all boundary-margin (frames 2505-2512) |
| LOST->TRACKING / TRACKING->LOST | 4 / 4 |
| Reacq delay per visible segment | 0-904: 188 (init); 1751-2084: 0 (already TRACKING at start); 2085-2174: 0 (already TRACKING); 2205-2264: not re-acquired; 2385-2444: not re-acquired; 2445-2504: 15; 2565-2624: 16; 2655-2697: 29 |

**E1**
| Metric | Value |
|---|---|
| Coverage, margin-clean | front 958/1115 = 85.9% ; back 2/28 = 7.1% ; side 0/0 = n/a |
| Coverage, raw | front 1051/1239 = 84.8% ; back 87/270 = 32.2% ; side 72/103 = 69.9% |
| target_visible=2 tracked | 0/180 = 0.0% |
| FALSE LOCK (vis=0, outside margin) | 0 frames |
| "Uncertain" TRACKING frames | 8, all boundary-margin (2505-2512) |
| LOST->TRACKING / TRACKING->LOST | 5 / 5 |
| Reacq delay per visible segment | 0-904: 188 (init); 1751-2084: 0; 2085-2174: 0; 2205-2264: not re-acquired; 2385-2444: not re-acquired; 2445-2504: 15; 2565-2624: 16; 2655-2697: 16 |

**E2**
| Metric | Value |
|---|---|
| Coverage, margin-clean | front 958/1115 = 85.9% ; back 0/28 = 0.0% ; side 0/0 = n/a |
| Coverage, raw | front 1051/1239 = 84.8% ; back 25/270 = 9.3% ; side 55/103 = 53.4% |
| target_visible=2 tracked | 0/180 = 0.0% |
| FALSE LOCK (vis=0, outside margin) | 0 frames |
| "Uncertain" TRACKING frames | 8, all boundary-margin (2505-2512) |
| LOST->TRACKING / TRACKING->LOST | 3 / 3 |
| Reacq delay per visible segment | 0-904: 188 (init); 1751-2084: 0; 2085-2174: 0; 2205-2264: not re-acquired; 2385-2444: not re-acquired; 2445-2504: 17; 2565-2624: not re-acquired; 2655-2697: 31 |

**E3**
| Metric | Value |
|---|---|
| Coverage, margin-clean | front 1052/1115 = 94.3% ; back 0/28 = 0.0% ; side 0/0 = n/a |
| Coverage, raw | front 1145/1239 = 92.4% ; back 25/270 = 9.3% ; side 58/103 = 56.3% |
| target_visible=2 tracked | 0/180 = 0.0% |
| FALSE LOCK (vis=0, outside margin) | 0 frames |
| "Uncertain" TRACKING frames | 8, all boundary-margin (2505-2512) |
| LOST->TRACKING / TRACKING->LOST | 3 / 3 |
| Reacq delay per visible segment | 0-904: 94 (init); 1751-2084: 0; 2085-2174: 0; 2205-2264: not re-acquired; 2385-2444: not re-acquired; 2445-2504: 19; 2565-2624: not re-acquired; 2655-2697: 26 |

Notes (MEASUREMENT, no recommendation):
- The +-30 margin leaves only 28 back frames and 0 side frames outside it, so the back/side coverage columns rest on tiny or empty denominators; raw coverage is shown for reference.
- Reacquisition-success counts from evaluate.py are 5/7 (baseline, E1) and 4/7 (E2, E3). It treats every target_visible=1 interval as a return, so the segments starting while the state is already TRACKING (2085) count as zero-delay returns. All reported delays except E2's 31 frames at 2655-2697 are below 30 frames, i.e. inside boundary uncertainty.
- evaluate.py itself reports FALSE LOCK = 0 for all four variants; it cannot see a wrong-person lock while the target is visible (STEP C).

### STEP C - visual verification pack (DONE, awaiting approval)

**Command pattern (one per variant, run from `/home/foxsec/drone_project`):**
```
venv/bin/python evaluate.py --segments --log logs/tracking_log_20261020_132332.csv --video outputs/partD_baseline.mp4 --variant baseline --out-dir outputs/check_frames/baseline
venv/bin/python evaluate.py --segments --log logs/tracking_log_20261020_132641.csv --video outputs/partD_E1.mp4       --variant E1       --out-dir outputs/check_frames/E1
venv/bin/python evaluate.py --segments --log logs/tracking_log_20261020_133004.csv --video outputs/partD_E2.mp4       --variant E2       --out-dir outputs/check_frames/E2
venv/bin/python evaluate.py --segments --log logs/tracking_log_20261020_133334.csv --video outputs/partD_E3.mp4       --variant E3       --out-dir outputs/check_frames/E3
```

**Exported tiles** (first / middle / last frame of every TRACKING segment; one `seg_<variant>_<start>_<first|mid|last>.jpg` per tile):
| Variant | Segments (start-end, ID) | Tiles | Contact sheet |
|---|---|---|---|
| baseline | 188-904 ID1; 1751-2109 ID350; 2460-2512 ID468; 2581-2624 ID534; 2684-2697 ID544 | 15 | `outputs/check_frames/contact_baseline.jpg` |
| E1 | 188-904 ID1; 1751-2109 ID350; 2142-2159 ID412; 2460-2512 ID468; 2581-2624 ID534; 2671-2697 ID544 | 18 | `outputs/check_frames/contact_E1.jpg` |
| E2 | 188-904 ID1; 1751-2109 ID350; 2462-2512 ID468; 2686-2697 ID544 | 12 | `outputs/check_frames/contact_E2.jpg` |
| E3 | 94-904 ID1; 1751-2109 ID350; 2464-2512 ID468; 2681-2697 ID544 | 12 | `outputs/check_frames/contact_E3.jpg` |

Each contact sheet uses 4 columns; every tile is labelled `f<frame>  <first|mid|last>  ID <track_id>` above the full-resolution annotated frame (HUD visible). Four sheets total, as required by the image budget.

**Log-to-video association - VERIFIED (MEASUREMENT):** green-box pixel counts at the same frames match the state column: baseline 2142 none / E1 2142 green; baseline 2460 green / E2 and E3 2460 none; E1 2671, E3 2681, baseline 2684 all green.

**Limitation - VERIFIED FACT:** `evaluate.py` only tests state=TRACKING while target_visible=0. A green box on the WRONG person while the target IS visible produces no FALSE LOCK by this metric. Such an error can only be judged from these images (or by the user's own inspection), not measured by the script. I could not visually inspect the JPEGs in this session (image contents were not rendered to me), so identity correctness of every segment is NOT VERIFIED here and is left to the user.

**Gate:** STEP D and STEP E not started. Waiting for "APPROVED, continue with STEP D and E" or corrections.

---

## Visual verification

**User-reported, videos/5.mp4 only (MEASUREMENT, sample):**
- Keypress spot-check: **90 frames checked = 44 correct / 0 wrong / 46 no box**.
- All exported first/mid/last frames of every TRACKING segment in baseline, E1, E2 and E3 were checked by the user; every one shows the green box on the correct person.

This is a **sample from one video**, not a false-lock rate. It does not support the statement "false lock = 0%". The automatic `evaluate.py` FALSE LOCK count (state=TRACKING while target_visible=0, outside the boundary margin) is a different, narrower measure and does not cover a wrong person while the target is visible.

---

## Audit - TASK 1

### Where the R4 continuity check runs (CURRENT `follow_target.py`)
| Context | Lines | Behaviour |
|---|---|---|
| TRACKING (main R4) | 689-707 | cosine(new emb, `target_last_quality_emb`) < `CONSECUTIVE_EMB_SIM_THRESH` (0.50) -> rollback, LOST, early return (700-706); otherwise the reference is replaced (707) |
| INITIALIZING | 612-628 | same comparison; on failure clears anchors and goes to IDLE (617-620) |
| LOST (per candidate) | 866-877 | same threshold but only resets that candidate's `consecutive_ok_count`; never causes LOST |

### Conditions under which R4 is skipped (VERIFIED FACT, line numbers)
1. State is not TRACKING (block starts at 665; IDLE returns early at 555-561).
2. `target_track is None` (666): the stored target ID is absent from the tracker output; only `missing_frames` counts (842-851).
3. `target_tier` is NONE (685): no embedding is evaluated, and the old `target_last_quality_emb` stays stale.
4. `target_track.track_id not in embeddings` (685): no valid crop (>=10x10) / embedding for that track.
5. `target_last_quality_emb is None` (689): first quality frame after selection, LOST, re-acquisition (925), continuity failure (697), verification failure (774) or missing target (849) is stored without comparison.
6. Frames with tier NONE in between are not compared: the reference is "last quality frame", not literally the previous frame.
7. Once state leaves TRACKING earlier in the same frame (verification path 768-776), the later gates do not run; the continuity block itself has already executed.
The check is unconditional with respect to E1/E2 switches.

### Frames 903-904 recomputation (exact replay of `TargetManager`)
Method (MEASUREMENT): replayed frames 0-905 with the current `DetectorTracker` + `TargetManager`, `target_track_id=1`, same weights, no video writing; before each `update()` the target crop was embedded and compared with the live `manager.target_last_quality_emb`. The replayed POST values reproduce the baseline CSV exactly (904: 0.5332/0.5153; 905: 0.4378/0.4140).

| Frame | Tier | Consecutive cosine vs `target_last_quality_emb` | Tracker box |
|---|---|---|---|
| 903 | PARTIAL | 0.9966 (vs 902) | (0, 5, 1166, 1068) |
| **904** | **FULL** | **0.5142** | **(185, 12, 1472, 1015)** |
| 905 | PARTIAL | 0.8928 (vs 904) | (387, 0, 1585, 1055) |

**0.4534 provenance:** VERIFIED FACT that it was **not logged by the program**: the CSV header (line 1177) has no consecutive-cosine column, and `consec_sim` exists only as a local variable (615, 690, 873) with no print. It was therefore computed after the run. The exact value of the program's comparison at 904 is **0.5142**, so 0.4534 is **not reproducible** with `target_last_quality_emb`; a post-hoc crop with YOLO's box instead of the tracker box gives 0.4707, showing the sensitivity to box choice.

### Baseline rows 898-908 (CSV `tracking_log_20261020_132332.csv`)
| Frame | state | target_track_id | target_score | sim_anchor | quality_tier |
|---|---|---|---|---|---|
| 898 | TRACKING | 1 | 0.9255 | 0.8475 | FULL |
| 899 | TRACKING | 1 | 0.9150 | 0.8304 | FULL |
| 900 | TRACKING | 1 | 0.9101 | 0.8475 | FULL |
| 901 | TRACKING | 1 | 0.9268 | 0.8473 | PARTIAL |
| 902 | TRACKING | 1 | 0.8951 | 0.8281 | PARTIAL |
| 903 | TRACKING | 1 | 0.8967 | 0.8310 | PARTIAL |
| 904 | TRACKING | 1 | 0.5332 | 0.5153 | FULL |
| 905 | LOST | 1 | 0.4378 | 0.4140 | PARTIAL |
| 906 | LOST | 1 | - | - | NONE |
| 907 | LOST | 1 | - | - | NONE |
| 908 | LOST | 1 | - | - | NONE |

### Why there was no immediate LOST at 904 (VERIFIED FACT from code + replay)
1. R4 cosine 0.5142 >= 0.50 at 904 -> continuity passes; reference updated.
2. `sim_anchor = 0.5153 < T_ANCHOR_FLOOR (0.60)` triggers suspicious mode (736-738), so the fail limit becomes `SUSPICIOUS_FAIL_LIMIT = 2` (767).
3. Verification runs at 904 (5-frame interval) and fails: score 0.5332 < `T_VERIFY` 0.55 and anchor 0.5153 < 0.60 -> `fail_counter = 1`.
4. At 905 suspicious mode verifies again, fails again (0.4378 / 0.4140) -> `fail_counter = 2 >= 2` -> LOST (768-776). `MISSING_FRAMES_FOR_LOST` does not apply because the target track still exists.

### Tracker and "abrupt camera cut"
- **BoT-SORT: VERIFIED FACT.** `TRACKER_CONFIG = "botsort.yaml"` (line 49), passed as `tracker=` (348). ByteTrack is not used.
- **Camera cut: HYPOTHESIS, not verified.** MEASUREMENT: mean absolute gray-frame difference at 320x180 is 0.0 for 902->903, **62.2 for 903->904**, 4.8 for 904->905; tracker output drops from 6 tracks to 1 and the box jumps. This is consistent with an abrupt scene change, but no metadata or visual confirmation exists and a close-up person/fast pan is not excluded.

---

## Audit - TASK 2

**Question: can the current log or code tell which gate rejected the best candidate at 2160-2260 / 2380-2460?**
**No (VERIFIED FACT).** The CSV header (line 1177) has only: frame, state, target_track_id, target_score, sim_anchor, sim_dynamic, best_candidate_score, quality_tier, num_people. In LOST rows (e.g. 2155-2165) `target_score`, `sim_anchor`, `sim_dynamic` are blank, `quality_tier` is the **target's** tier (NONE), and `best_candidate_score` (937-939) is only the maximum `sim_all` of non-target tracks. There is no candidate tier, anchor score, consecutive count, margin, or rejection reason in the file. The code holds these values internally in `candidate_history` (862-912) but emits none of them.

MEASUREMENT of the logged `best_candidate_score` in the two requested windows (baseline log): 2160-2260 (101 rows, all LOST) min 0.4490 / mean 0.6846 / max 0.8261; 2380-2460 (81 rows, all LOST) min 0.4531 / mean 0.7515 / max 0.8373. These are `sim_all` values of the best non-target track only; the rejection gate is unknown.

### Proposed LOGGING-ONLY change list (NOT implemented; no decision/threshold change)
1. Add CSV columns: `best_cand_id`, `best_cand_tier`, `best_cand_sim_all`, `best_cand_sim_anchor`, `best_cand_consec`, `best_cand_req_accept`, `best_cand_margin`, `n_tier_ok`, `n_consec_ok`, `n_pass_accept`, `n_pass_margin`, `gate`.
2. Return the already-computed LOST-loop values in the output dict (946-958) and write them at 1223; no branch changed.
3. Add a second per-candidate CSV (`lost_candidates_<timestamp>.csv`), one row per candidate per frame: frame, tid, tier, sim_all, sim_anchor, consec_count, other_best, margin, pass_req.
4. Add verification diagnostics: `fail_counter`, `suspicious_countdown`, `should_verify`, `passes_verify`, `consec_sim`, `last_emb_present`.
5. `gate` is a string derived only from existing booleans (tier_none / accept_fail / floor_fail / nconfirm_fail / margin_fail / no_candidate), without altering any decision.

---

## Audit - TASK 3

### How embeddings are labelled (CURRENT `calibrate_thresholds.py`) - PRELIMINARY
- Target (lines 154-161): only when `target_visible == 1` AND `expected_target_id is not None` AND `tid == expected_target_id` (fallback: `tid in --target-ids`, 157-158).
- `expected_target_id` comes from the GT `target_id` column if present, otherwise from the hard-coded `default_interval_targets` (101-107) that encodes **baseline tracker IDs** (1, 350, 468, 534, 544). `5_manual.csv` has no `target_id` column, so every target label is tracker-ID derived.
- Non-target (184-187): every other valid track in a sampled frame, including **all** tracks in target_visible=0 frames and all tracks in visible frames where no expected ID is available.
- Consequence: identity is **not independent of the tracker**; all results are **PRELIMINARY**. Anchor/dynamic galleries (168-176) are built from the same labels, so gallery scores are preliminary too.

Exclusions implemented (84-87, 136): target_visible=2 and every frame within +-30 of any GT boundary, for both classes.

**Command:** `venv/bin/python calibrate_thresholds.py --video videos/5.mp4 --gt ground_truth/5_manual.csv --stride 5 --plot-name calibration_histograms_manual.png`

| Item | Count |
|---|---|
| Target embeddings | 222 (front 222, back 0, side 0) |
| Non-target embeddings | 2,232 |
| Different-person pairs | 495,504 |
| Sampled visible frames with no expected target ID | 5 |
| SAME front-front pairs | 24,531 (mean 0.8792, min 0.6831, P50 0.8816) |
| SAME back-back pairs | 0 (not computable) |
| SAME front-back pairs | 0 (not computable) |
| DIFFERENT-person P99 | 0.6326 |
| DIFFERENT-person P99.9 | 0.7213 |
| DIFFERENT-person max | 0.7928 |

**Back/side limitation (MEASUREMENT):** under the required +-30 exclusion, every back/side frame outside the margin falls outside the hard-coded ID intervals, so no target back/side embedding survives. Back-back and front-back statistics cannot be computed from this run.

**Top-20 highest different-person pairs** (all non-target frames are vis=0, i.e. GT absence; tid 246 at 1140/1145 and tid 204 at 1065):
| # | score | target frame/view | non-target frame/view | vis | tid |
|---|---|---|---|---|---|
| 1 | 0.7928 | 330 front | 1145 none | 0 | 246 |
| 2 | 0.7927 | 355 front | 1140 none | 0 | 246 |
| 3 | 0.7920 | 830 front | 1065 none | 0 | 204 |
| 4 | 0.7900 | 685 front | 1140 none | 0 | 246 |
| 5 | 0.7893 | 350 front | 1140 none | 0 | 246 |
| 6 | 0.7875 | 675 front | 1140 none | 0 | 246 |
| 7 | 0.7851 | 350 front | 1145 none | 0 | 246 |
| 8 | 0.7847 | 355 front | 1145 none | 0 | 246 |
| 9 | 0.7836 | 670 front | 1140 none | 0 | 246 |
| 10 | 0.7814 | 305 front | 1140 none | 0 | 246 |
| 11 | 0.7811 | 340 front | 1145 none | 0 | 246 |
| 12 | 0.7809 | 360 front | 1145 none | 0 | 246 |
| 13 | 0.7806 | 375 front | 1140 none | 0 | 246 |
| 14 | 0.7805 | 330 front | 1140 none | 0 | 246 |
| 15 | 0.7803 | 345 front | 1140 none | 0 | 246 |
| 16 | 0.7802 | 820 front | 1065 none | 0 | 204 |
| 17 | 0.7799 | 370 front | 1140 none | 0 | 246 |
| 18 | 0.7799 | 825 front | 1065 none | 0 | 204 |
| 19 | 0.7799 | 805 front | 1065 none | 0 | 204 |
| 20 | 0.7796 | 345 front | 1145 none | 0 | 246 |

Plot (new file): `outputs/calibration_histograms_manual.png`. No threshold was changed.

---

## Audit - TASK 4

| Option | Status | Evidence in current `follow_target.py` |
|---|---|---|
| O1 `--start-frame` | **NOT DONE** | No such argument in argparse (1296-1302); `start_frame_idx = 0` hard-set (1168) and used at 1183; only printed at 1275 |
| O2 'f' skips 30 frames on selection screen | **NOT DONE** | Selection loop handles SPACE/ENTER (1097), 'n' (1101), 'q'/ESC (1104); no 'f' (1096-1106) |
| O3 'r' reselects target during run | **NOT DONE** | Run loop handles only 'q'/ESC (1238-1239); no 'r' |
| O4 `--classes` | **NOT DONE** | No `--classes` in argparse (1296-1302); `DetectorTracker` accepts `classes` (333) and passes it to the model (346) with default `TARGET_CLASSES = [0]` (48), but it is not exposed on the CLI |

---

## Audit - TASK 5

**VERIFIED FACT (fetched `docs/MODEL_ZOO.md` from the official torchreid repository):**
- `osnet_ain_x1_0` appears as an **ImageNet-pretrained** model, in cross-domain Market1501->DukeMTMC and DukeMTMC->Market1501, in "MSMT17 (combineall=True) -> Market1501 & DukeMTMC", and in the multi-source table.
- There is **no same-domain MSMT17-trained `osnet_ain_x1_0` row** in MODEL_ZOO.md. The official MSMT17 download links shown are Google Drive links; none is a same-domain osnet_ain_x1_0 file.

**VERIFIED FACT (HuggingFace `kaiyangzhou/osnet`, author Kaiyang Zhou):**
- The repository contains `osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth`.
- MD5 of that file = **d73369b062de601b85d4fbd3045ba7a2**, size **17,293,009 bytes**; the local `weights/osnet_ain_x1_0_msmt17.pth` has the **same MD5 and size**. So the weights actually used are exactly this HF checkpoint.
- The local file has 552 keys including `module.classifier.weight` (4101, 512), consistent with an MSMT17-trained (4101-ID) checkpoint.

**NOT VERIFIED:**
- That this HF file is the same artifact as any official MODEL_ZOO download (MODEL_ZOO does not publish a same-domain osnet_ain MSMT17 link, so there is nothing to compare against).
- That the HF repository is an "official" torchreid source: MODEL_ZOO does not link it. The author name matches, which makes it plausible (HYPOTHESIS), but the page itself is the only evidence.
- Its training provenance beyond the filename (ep50, cosine LR, label smoothing, flip+jitter) - the file name states it; the training run was not reproduced.

---

## Stage 2 logging step (2026-10-25)

Scope: Python only (no ROS2, no PX4, no `drone_control.py`); no E1/E2/E3 run and no recommendation.
`follow_target.py` was edited for **LOGGING ONLY**; pre-edit backup: `backups/follow_target_2026-10-25_pre_logging.py`.

### Changes (logging only)
- The first 9 CSV columns are unchanged; 13 columns were **appended**: `consec_sim, fail_counter, suspicious_countdown, should_verify, passes_verify, best_cand_id, best_cand_tier, best_cand_sim_all, best_cand_sim_anchor, best_cand_consec, best_cand_req_accept, best_cand_margin, gate`.
- Verified by diff against the backup: no decision expression, constant, threshold or default was changed (only dict writes, read-only diagnostics, CSV formatting).
- `gate` is derived only from existing booleans: `tier_none / accept_fail / floor_fail / nconfirm_fail / margin_fail / no_candidate / not_lost`. `best_cand_*` are filled only while `state == LOST`; otherwise the fields stay empty.
- `fail_counter` / `suspicious_countdown` are logged at **decision time** (before the reset that follows a LOST transition). On the four TRACKING->LOST transition frames (905, 2110, 2513, 2625) `gate` is empty because those LOST events come from verification/continuity failure, not from candidate rejection.
- No separate per-candidate CSV; no `--start-frame`, no `--classes`, no keys `f`/`r`.

### Commands
```bash
cp follow_target.py backups/follow_target_2026-10-25_pre_logging.py
python3 follow_target.py --video videos/5.mp4 --out outputs/partE_baseline_logging.mp4 --target-id 1 --headless
```
Output: `outputs/partE_baseline_logging.mp4`, log `logs/tracking_log_20261025_212543.csv` (2698 rows x 22 cols). Run: 2698 frames, 170.36 s, 15.84 FPS.

### Regression result: PASS (MEASUREMENT)
- Original 9 columns: **IDENTICAL in all 2698 rows** (0 differing rows, so no first differing frame).
- State changes identical: 188 TRACKING(ID 1), 905 LOST, 1751 TRACKING(350), 2110 LOST, 2460 TRACKING(468), 2513 LOST, 2581 TRACKING(534), 2625 LOST, 2684 TRACKING(544).
- Suspected false-lock frames identical: 904, 2510, 2605, 2608, 2624.
- Cross-check of the new columns against the earlier audit replay: `consec_sim` 0.5142 at frame 904 and 0.8928 at 905; `fail_counter` 2 at 905.
- Note: `outputs/gallery_debug/` crops are re-written with frame-based names on every run (pre-existing behaviour).

### User eye-verification, 2026-10-25 (VERIFIED, one video, sampled frames)
The user inspected specific frames of `videos/5.mp4` by eye and confirmed:
1. **Frame 1062:** the target is **NOT in the frame** and **no green box** was drawn; the highest-scoring person (`best_candidate_score` 0.7982, `num_people` 12) is **another person** and was correctly ignored.
2. **ID 435** at frames **2195 and 2200** is the **true target**.
3. **ID 468** at frame **2458** is the **true target**.

Combining these eye labels with the logged diagnostics (MEASUREMENT, three frames only): at 2195 `gate=nconfirm_fail` (`sim_all` 0.8261, FULL, `consec=4`); at 2200 `gate=accept_fail` (PARTIAL, `sim_all` 0.8173 < required 0.83, `consec=0`); at 2458 `gate=nconfirm_fail` (`sim_all` 0.8373, FULL, `consec=6`). In these sampled frames the **correct** candidate was held back by the acceptance/consecutive gates - not by a wrong-person lock and not by the anchor floor.

Scope: one video, a handful of frames. This is a sample and does **not** support any rate statement (e.g. "false lock = 0%"). No threshold or decision was changed.

### Stage 2 completion (ESTIMATE, not a measurement)
**~68-70%** by the proposed completion criteria. This stays an estimate; the logging step does not convert it into a measurement.

### Calibration label caveat (PRELIMINARY)
All calibration labels are **tracker-ID derived**: `ground_truth/5_manual.csv` has no `target_id` column, so `calibrate_thresholds.py` falls back to the hard-coded baseline tracker-ID intervals. **Back-back and front-back same-person statistics are not computable**, and every similarity result is **PRELIMINARY** - it is not independent of the tracker. No threshold was calibrated or changed in this step.

---

## Step 1 (b) - per-run gallery_debug (2026-10-25)

Backup before editing: `backups/follow_target_2026-10-25_220940_pre_gallery.py`.

### Change (VERIFIED FACT by diff)
- `TargetManager.__init__` takes `gallery_debug_dir` (default = the old base dir, so other callers are unaffected).
- The four crop writes (anchor x2, dyn, view_chain) use `self.gallery_debug_dir`.
- `run_tracking` creates `outputs/gallery_debug/<YYYYMMDD_HHMMSS>_<out-video-stem>/` and **refuses to reuse an existing directory** (a `_2`, `_3`, ... suffix is added if the name is taken). Nothing writes into an older run's directory.
- `GALLERY_DEBUG_DIR` stays the base directory; no threshold, constant or decision branch changed.

### Command
```bash
python3 follow_target.py --video videos/5.mp4 --out outputs/partE_baseline_step1_gallery.mp4 --target-id 1 --headless
```
Run: 2698 frames, 171.22 s, 15.76 FPS. Log `logs/tracking_log_20261025_221109.csv`. New gallery dir `outputs/gallery_debug/20261025_221109_partE_baseline_step1_gallery/` with **73 files (10 anchor + 63 dyn)** - same count as the historical run.

### Regression result: PASS (MEASUREMENT)
- Original 9 columns **IDENTICAL in all 2698 rows** (0 differing rows).
- State changes identical: 188 TRACKING(ID 1), 905 LOST, 1751 TRACKING(350), 2110 LOST, 2460 TRACKING(468), 2513 LOST, 2581 TRACKING(534), 2625 LOST, 2684 TRACKING(544).
- Suspected false-lock frames identical: 904, 2510, 2605, 2608, 2624.
- Old outputs untouched: the 162 pre-existing files directly under `outputs/gallery_debug/` have **identical MD5 before and after (162 OK, 0 mismatched)** and the top-level file count is still 162.

### Independent re-verification (2026-10-26, MEASUREMENT)
Backup before re-verification (no code edited): `backups/follow_target_2026-10-26_022137_pre_step1_verify.py`.

```bash
venv/bin/python follow_target.py --video videos/5.mp4 --out outputs/partE_baseline_step1_gallery.mp4 --target-id 1 --headless
venv/bin/python <scratchpad>/regress.py logs/tracking_log_20261026_022150.csv
```
Raw run summary:
```text
  Frames processed        : 2698
  Total processing time   : 176.16 s
  Average pipeline FPS    : 15.32 FPS
  CSV log saved           : logs/tracking_log_20261026_022150.csv
  Gallery debug images    : outputs/gallery_debug/20261026_022150_partE_baseline_step1_gallery
  Suspected False-Locks   : ['904', '2510', '2605', '2608', '2624']
```
Raw regression output (reference `logs/tracking_log_20261020_132332.csv`, 9 original columns):
```text
REF logs/tracking_log_20261020_132332.csv: 2698 rows, 9 cols
NEW logs/tracking_log_20261026_022150.csv: 2698 rows, 22 cols
NEW first 9 header identical: True
ROW COUNT identical: True
ORIGINAL 9 COLUMNS: differing rows = 0 / 2698
STATE CHANGES: ref=10 new=10 identical=True
  frame    0 -> INITIALIZING target_track_id 1
  frame  188 -> TRACKING     target_track_id 1
  frame  905 -> LOST         target_track_id 1
  frame 1751 -> TRACKING     target_track_id 350
  frame 2110 -> LOST         target_track_id 350
  frame 2460 -> TRACKING     target_track_id 468
  frame 2513 -> LOST         target_track_id 468
  frame 2581 -> TRACKING     target_track_id 534
  frame 2625 -> LOST         target_track_id 534
  frame 2684 -> TRACKING     target_track_id 544
SUSPECTED FALSE-LOCK FRAMES (TRACKING & sim_anchor<0.60): [904, 2510, 2605, 2608, 2624]
```
(10 change rows = the initial `0 -> INITIALIZING` plus the 9 state-change frames listed by the user.)

Raw gallery-isolation output:
```text
NEW RUN DIR: outputs/gallery_debug/20261026_022150_partE_baseline_step1_gallery
files in new dir: 73 (anchor=10, dyn=63, view_chain=0)
OLD RUN DIR (20261025_221109) after: 73 OK (md5 unchanged), files now: 73
TOP-LEVEL gallery_debug: files now 162 (before 162), MD5 identical (0 changes)
```
No threshold, constant, or decision branch was changed in this step.

---

## Step 2 (c) - unified output: valid / stale / age (2026-10-26)

Backup before editing: `backups/follow_target_2026-10-26_023351_pre_unified_output.py` (md5 `53af141873f2f88a281c681dbc366a63`, identical to the pre-edit `follow_target.py`).

### Definitions (implemented exactly)
- **valid** = `output["target_box"] is not None`. This is the same condition `draw_hud` uses to draw the green box; `output_box` is only non-None when `state in (TRACKING, INITIALIZING)` and (INITIALIZING or `sim_anchor >= T_ANCHOR_FLOOR = 0.60`). No new state test.
- **stale** = 1 only when `valid == 0` **and** at least one valid frame already occurred earlier in the run (the last drawn box is no longer current). It is 0 on every valid frame and 0 before the first valid frame. No threshold.
- **age** = number of consecutive frames since the last `valid == 1` frame: `0` on valid frames, `1..n` on the nth consecutive invalid frame, empty before the first valid frame. No threshold.
- New columns are appended **after** the existing 22 columns: `...,gate,valid,stale,age` (CSV 22 -> 25 columns). The first 9 columns are untouched.
- All three are computed locally in `run_tracking` immediately before the CSV write. They are never written back into `TargetManager` and nothing reads them for a decision.

### Code scope (VERIFIED FACT by diff vs backup)
The complete diff is: the CSV header string, `frames_since_valid: Optional[int] = None`, the valid/stale/age derivation block, and the appended write fields. No threshold, constant, condition or default changed.

### Command
```bash
venv/bin/python follow_target.py --video videos/5.mp4 --out outputs/partE_baseline_step2c_unified.mp4 --target-id 1 --headless
```
Raw run summary:
```text
  Frames processed        : 2698
  Total processing time   : 172.39 s
  Average pipeline FPS    : 15.65 FPS
  CSV log saved           : logs/tracking_log_20261026_023427.csv
  Gallery debug images    : outputs/gallery_debug/20261026_023427_partE_baseline_step2c_unified
  Suspected False-Locks   : ['904', '2510', '2605', '2608', '2624']
```

### Regression result: PASS (MEASUREMENT)
Raw output vs `logs/tracking_log_20261020_132332.csv`:
```text
REF logs/tracking_log_20261020_132332.csv: 2698 rows, 9 cols
NEW logs/tracking_log_20261026_023427.csv: 2698 rows, 25 cols
NEW first 9 header identical: True
ROW COUNT identical: True
ORIGINAL 9 COLUMNS: differing rows = 0 / 2698
STATE CHANGES: ref=10 new=10 identical=True
  frame    0 -> INITIALIZING target_track_id 1
  frame  188 -> TRACKING     target_track_id 1
  frame  905 -> LOST         target_track_id 1
  frame 1751 -> TRACKING     target_track_id 350
  frame 2110 -> LOST         target_track_id 350
  frame 2460 -> TRACKING     target_track_id 468
  frame 2513 -> LOST         target_track_id 468
  frame 2581 -> TRACKING     target_track_id 534
  frame 2625 -> LOST         target_track_id 534
  frame 2684 -> TRACKING     target_track_id 544
SUSPECTED FALSE-LOCK FRAMES (TRACKING & sim_anchor<0.60): [904, 2510, 2605, 2608, 2624]
```
(10 change rows = the initial `0 -> INITIALIZING` plus the 9 state-change frames.)

### valid cross-check on all 2698 rows (MEASUREMENT)
`valid` was compared with the green-box condition derived from the existing columns (`state == INITIALIZING`, or `state == TRACKING` and `sim_anchor >= 0.60`):
```text
VALID CROSS-CHECK (vs state+anchor-floor condition): 2698/2698 match, 0 mismatch
VALID COUNTS: valid=1 -> 1361, valid=0 -> 1337
STALE/AGE CONSISTENCY: 0 errors over 2698 rows
STALE COUNTS: stale=1 -> 1337, stale=0 -> 1361
AGE: max consecutive frames without a drawn box = 848
STALE GAPS (first last length): 7
   904 1751  848
  2108 2460  353
  2510 2581   72
  2605 2605    1
  2608 2608    1
  2610 2610    1
  2624 2684   61
```
Direct pixel check of the rendered video (green = BGR `(0,255,0)`; outside-HUD counts exclude the HUD rectangle) on sampled frames:
```text
 frame valid   age  green_total green_outside_HUD
     0     1     0        18247             15308
   100     1     0        18661             15494
   500     1     0        20314             19569
   903     1     0        18264              7036
   904     0     1          764                 0
  1000     0    97            0                 0
  1751     0   848          698                 0
  1752     1     0        20617             19898
  2000     1     0        19761             11035
  2195     0    88            0                 0
  2510     0     1          740                 0
  2605     0     1          750                 0
  2684     0    61          701                 0
  2697     1     0        13513             12794
```
Every `valid=0` sampled frame has zero green pixels outside the HUD; every `valid=1` frame has thousands. The residual 698-764 green pixels on `state=TRACKING` but `valid=0` frames are the HUD text, not a box.

Scope: one video (5.mp4), 2698 frames. The counts above describe how often the box was drawn; they are **not** a correctness or false-lock rate. No threshold or decision branch was changed, and nothing in this step feeds back into tracking decisions.
