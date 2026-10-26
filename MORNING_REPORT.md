# MORNING REPORT - unattended overnight run (2026-10-26)

Start time: 02:54 +07 (checked with `date`). Budget: ~3 h.
Working dir: `/home/foxsec/drone_project`.

## STEP 0 - STATE CHECK (no edits)

Raw evidence:
```text
Mon Oct 26 02:54:06 +07 2026
--- git log ---
4f26e13 Full source backup 2026-10-26 (includes in-progress step 2 work)
b63b371 refactor: restructure repository layout and clean code
80cd612 feat: initial clean commit of drone tracking system
--- git status ---
(empty)
--- backups ---
follow_target_2026-10-26_023351_pre_unified_output.py
follow_target_2026-10-26_022137_pre_step1_verify.py
follow_target_2026-10-25_220940_pre_gallery.py
follow_target_2026-10-25_pre_logging.py
follow_target_2026-10-24.py
follow_target_2026-10-14.py
--- disk ---
/dev/nvme0n1p5   89G   45G   40G  54% /
```
- `label_target.py`: **does not exist** (`ls` -> "No such file or directory"). Needs to be created (STEP 3).
- `MORNING_REPORT.md`: did not exist before this file.
- `HANDOFF_2026-10-25.md`: **does not exist** at the stated path and `find . -maxdepth 3 -iname "*HANDOFF*"` returned nothing. I could not read it. Decision (ambiguity, conservative): proceed with GEMINI_BRIEF.md + STATUS.md only, make no assumption that depends on the missing handoff.
- Already done before tonight: STEP 1(b) per-run gallery dir, STEP 2(c) valid/stale/age. Both are recorded in STATUS.md and roadmap_data.json (commit 4f26e13). No working-tree changes at start.
- No video runs tonight. No edits to `follow_target.py`, `evaluate.py` logic, `calibrate_thresholds.py`, `GEMINI_BRIEF.md`, `.gitignore`, `ground_truth/*`.

## STEP 2 VERIFY - read-only

### PLAN
- Files touched: **none**. Only read `logs/tracking_log_20261026_023427.csv`, `logs/tracking_log_20261020_132332.csv`, `logs/tracking_log_20261026_022150.csv`.
- Will NOT touch: `follow_target.py` (not edited), all logs, all ground truth.
- Checks: (1) 25 columns / 2698 rows; (2) first 9 columns identical to the 9-col reference, position by position, all rows; (3) first 22 columns identical to the 22-col run; (4) `valid` against the green-box condition (`INITIALIZING`, or `TRACKING` with `sim_anchor >= 0.60`), all rows; (5) explicitly list INITIALIZING frames where `valid=1`.

### PRE-MORTEM (3 ways this could silently mislead) and detection
1. **Column shift/rename makes a positional comparison lie.** Detection: print both headers, compare header names, and compare columns by name, not only by position (per-column differing-row counts).
2. **Comparing only the first 9 columns could hide a change in the appended columns.** Detection: compare all 22 shared columns row-by-row against `022150`, and print counts of differing rows per column.
3. **`valid` may not equal "a green box is drawn"** (e.g. wrong state test, or INITIALIZING edge). Detection: recompute expected valid from the documented draw condition and count mismatches on all 2698 rows; separately print the INITIALIZING rows with `valid=1`.

Detection commands are executed below; raw output is pasted.

### RESULT: PASS (MEASUREMENT)

Raw output of the verification command:
```text
REF9  2698 rows, 9 cols
RUN22 2698 rows, 22 cols
RUN25 2698 rows, 25 cols
RUN25 header: frame,state,target_track_id,target_score,sim_anchor,sim_dynamic,best_candidate_score,quality_tier,num_people,consec_sim,fail_counter,suspicious_countdown,should_verify,passes_verify,best_cand_id,best_cand_tier,best_cand_sim_all,best_cand_sim_anchor,best_cand_consec,best_cand_req_accept,best_cand_margin,gate,valid,stale,age

CHECK 1: rows and columns
  RUN25 rows == 2698: True
  RUN25 cols == 25: True
  RUN25 last 3 == ['valid','stale','age']: True

CHECK 2: original 9 columns vs REF9 (per column, all 2698 rows)
  frame                differing rows = 0
  state                differing rows = 0
  target_track_id      differing rows = 0
  target_score         differing rows = 0
  sim_anchor           differing rows = 0
  sim_dynamic          differing rows = 0
  best_candidate_score differing rows = 0
  quality_tier         differing rows = 0
  num_people           differing rows = 0
  TOTAL differing cells in first 9 columns = 0

CHECK 3: first 22 columns vs RUN22 (per column, all 2698 rows)
  header 25[:22] == RUN22 header: True
  TOTAL differing cells in first 22 columns = 0

CHECK 4: valid vs documented green-box condition
  rows checked = 2698
  mismatches = 0
  valid=1 count = 1361
  valid=0 count = 1337

CHECK 5: INITIALIZING rows with valid=1
  INITIALIZING rows = 188 (frames 0..187)
  INITIALIZING rows with valid=1 = 188
  frame 0 row: 0,INITIALIZING,1,1.0000,1.0000,0.0000,0.3455,FULL,7,1.0000,0,0,,,,,,,,,,not_lost,1,0,0

CHECK 6: stale/age derived-rule consistency (report definitions)
  consistency errors = 0
  max age = 848
```

Interpretation (MEASUREMENT, one video):
- The log is exactly the accepted STEP 2(c) artifact: 25 columns, 2698 rows, original 9 columns identical to the 9-column reference in every cell, and the first 22 columns identical to the 22-column run in every cell.
- `valid` matches the documented draw condition on 2698/2698 rows.
- **IMPORTANT for downstream control:** `valid=1` is true for **all 188 INITIALIZING rows (frames 0-187)**, not only frame 0. `valid` means "a green box is drawn", and a box is drawn during INITIALIZING. Any control code that uses `valid` as "safe to follow" **must also require `state == TRACKING`**; otherwise it would act on the initialization box.
- `stale`/`age` follow the recorded definitions with 0 inconsistencies.

### SELF-REVIEW (hostile reviewer)
- No file was edited in this step, so the mandatory `diff <backup> follow_target.py` checks do not apply; `git status` still shows a clean tree after the checks.
- The comparison is by header names *and* positions; both headers and per-column counts are printed, so a silent column shift would have shown non-zero differences.
- The cross-check condition is derived from the existing columns and the constant `T_ANCHOR_FLOOR = 0.60` read from the code; it is not a guess. Its limitation: it assumes INITIALIZING always has a target track (true in this run; 188/188 rows valid).

### CLAIM AUDIT
| Claim | Producing output |
|---|---|
| 25 columns / 2698 rows | CHECK 1 lines |
| 0 differing cells in original 9 columns | CHECK 2 total |
| 0 differing cells in first 22 columns | CHECK 3 total |
| valid matches 2698/2698 | CHECK 4 mismatches = 0 |
| valid=1 1361, valid=0 1337 | CHECK 4 counts |
| all 188 INITIALIZING rows valid=1 | CHECK 5 |
| stale/age 0 errors, max age 848 | CHECK 6 |

## STEP 3 (a) - labeling tool (create only)

### PLAN
- Create exactly one new file: `/home/foxsec/drone_project/label_target.py`.
- Do NOT create `ground_truth/labels_5_eye.csv`; do NOT run any labeling; do NOT touch any existing GT file.
- Read-only inputs: `ground_truth/5_manual.csv` for intervals/views, `videos/5.mp4` for frame validation in `--dry-run`.
- Test only headless: `python3 -m py_compile label_target.py` and `--dry-run`.

### PRE-MORTEM and detection
1. **Tool could silently overwrite a GT file.** Mitigation: refuse if `--out` exists without `--append`; additionally refuse if `--out` resolves to an existing file in `ground_truth/` whose name is not `labels_5_eye.csv`, even with `--append`. Detection: inspect the guard in the file and run `--dry-run` (which writes nothing) + check `ls ground_truth` before/after.
2. **Sampling plan could silently include the wrong intervals (e.g. front 1751-2084 or invisible intervals).** Mitigation: explicit rule - only the `start_frame == 0` front interval uses stride 60, only `back`/`side` intervals use stride 20, everything else is printed as skipped. Detection: `--dry-run` output must list every GT interval with its rule and the total frame count; verify against the GT by hand.
3. **Box coordinates could be stored in display coordinates instead of video coordinates** (silent wrong GT). Mitigation: fixed display scale, divide mouse coordinates by the scale, clamp to frame bounds, and print the scale. Detection: a dry-run cannot test this; it is untested tonight (no display) and is stated as NOT VERIFIED in the report. No labeling is run.

### Ambiguities decided alone (conservative)
- The GT has a second front interval (1751-2084). The instruction names only "front 0-904 at stride 60" and "back/side intervals"; I **skip** the 1751-2084 front interval rather than invent a stride for it.
- The instruction names keys `s/n/q`; a box needs a confirm action. I add Enter/Space = accept box and advance, `r` = clear box. These are additions, not replacements.
- Box rows store `frame,status,x1,y1,x2,y2,view` with `status=box` or `not_visible`; no tracker ID column exists.

### RESULT: compile + dry-run PASS; interactive path NOT TESTED

Created `/home/foxsec/drone_project/label_target.py` (only new code file tonight).
`ground_truth/labels_5_eye.csv` was **not created** (confirmed by `ls`, exit 2).

Raw test output:
```text
PY_COMPILE OK
=== DRY RUN ===
GT: ground_truth/5_manual.csv
[   0- 904] visible=1 view=front front 0-904, stride 60
  frames (16): 0,60,120,180,240,300,360,420,480,540,600,660,720,780,840,900
[ 905-1750] visible=0 view=none  skipped (not the front 0-904 interval and not back/side)
  frames (0): -
[1751-2084] visible=1 view=front skipped (not the front 0-904 interval and not back/side)
  frames (0): -
[2085-2174] visible=1 view=back  back, stride 20
  frames (5): 2085,2105,2125,2145,2165
[2175-2204] visible=2 view=back  back, stride 20
  frames (2): 2175,2195
[2205-2264] visible=1 view=back  back, stride 20
  frames (3): 2205,2225,2245
[2265-2384] visible=2 view=back  back, stride 20
  frames (6): 2265,2285,2305,2325,2345,2365
[2385-2444] visible=1 view=back  back, stride 20
  frames (3): 2385,2405,2425
[2445-2504] visible=1 view=side  side, stride 20
  frames (3): 2445,2465,2485
[2505-2564] visible=0 view=none  skipped (not the front 0-904 interval and not back/side)
  frames (0): -
[2565-2624] visible=1 view=back  back, stride 20
  frames (3): 2565,2585,2605
[2625-2654] visible=2 view=back  back, stride 20
  frames (2): 2625,2645
[2655-2697] visible=1 view=side  side, stride 20
  frames (3): 2655,2675,2695
TOTAL SAMPLED FRAMES: 46
Video: videos/5.mp4 (2699 frames); max sampled frame 2695; within range: True
DRY RUN: no file was written, no display was opened.
=== GT UNCHANGED CHECK ===
ground_truth/5.csv: OK
ground_truth/5_ai_draft.csv: OK
ground_truth/5_ai_draft_samples.csv: OK
ground_truth/5_manual.csv: OK
=== labels file exists? ===
ls: cannot access 'ground_truth/labels_5_eye.csv': No such file or directory
```

Implemented against the spec:
- Shows the frame; left-drag draws a box; Enter/Space accepts and advances; `s` records `not_visible`; `n` skips; `q`/Esc saves and quits; `r` clears. Output has no tracker-ID column and the tool never reads a log.
- Stride 20 for every `back`/`side` interval; stride 60 for the front interval that starts at frame 0; other intervals are skipped and printed as skipped.
- Refuses to start if the output exists without `--append`; even with `--append` it refuses to append to any existing file under `ground_truth/` whose name is not `labels_5_eye.csv`.
- Checks for a display (`DISPLAY` or `WAYLAND_DISPLAY`) before opening a window; `--dry-run` writes nothing and opens no window.
- `--append` reads the existing header, skips frames already present, and appends only missing frames.

Gaps / NOT VERIFIED:
- The interactive path (mouse callback, scale conversion, key handling, actual writing) **could not be tested headless and was not run**, by instruction. It needs one manual smoke test by you.
- The display scale is fixed at min(1.0, 1600/w, 900/h); mouse coordinates are divided by that scale. This conversion is unverified tonight.
- A frame-count metadata discrepancy: OpenCV reports 2699 frames while the accepted runs process 2698. The largest sampled frame (2695) is inside both counts, so it does not affect sampling; noted as an observation.
- The second front interval 1751-2084 is skipped by the conservative ambiguity decision above.

### SELF-REVIEW (hostile reviewer)
- `follow_target.py` was **not** edited, so the mandatory diff-based branch audit does not apply. No existing file was modified by this step; GT md5s are identical before/after.
- The only new file is `label_target.py`; the label CSV does not exist.
- A hostile reviewer should test: (a) run once with `--out ground_truth/5_manual.csv --append` and confirm refusal; (b) draw a box at a known pixel location and compare the saved coordinates; (c) check that `q` mid-session leaves a valid CSV with a header. These are left for your manual test.

### CLAIM AUDIT
| Claim | Producing output |
|---|---|
| file compiles | `PY_COMPILE OK` |
| 46 sampled frames; per-interval lists | `--dry-run` output |
| sampled frames within video length | `within range: True` |
| no GT file changed | `md5sum -c` -> 4x OK |
| labels_5_eye.csv not created | `ls` -> No such file |

## SAFE ANALYSIS - read-only

### PLAN
- Files touched: **none**. Read `logs/tracking_log_20261026_023427.csv` and `ground_truth/5_manual.csv`; run `evaluate.py` read-only against them.
- Will NOT touch: `evaluate.py`, `follow_target.py`, GT files, thresholds, decision branches.
- A: For three windows around the named re-acquisition regions, count rows per `gate` and report `best_cand_consec` statistics: maximum value, longest consecutive non-empty run, longest strictly increasing run.
- B: Run `python3 evaluate.py --log logs/tracking_log_20261026_023427.csv --gt ground_truth/5_manual.csv --fps 30.0` and paste coverage per view and false-lock frames.

### PRE-MORTEM and detection
1. **Window definitions could silently mix GT and tracker states.** Detection: print the exact frame ranges, the GT rows they intersect, and the state counts (TRACKING/LOST) inside each window, so every number is attributable.
2. **`gate` counts could be misread as "the target candidate".** Detection: label the table only as gate reasons; explicitly state that the log does not identify which candidate is the target, and that this analysis cannot say whether any candidate was the true target.
3. **evaluate.py might modify files or rely on a default output.** Detection: run it with only `--log/--gt/--fps`, check `git status --short` and GT md5 before/after; it is read-only by design.

Two window sets are reported because the instruction is ambiguous: the exact named ranges, and the LOST-state subsets where candidate gates are actually populated (`gate` is `not_lost` outside LOST). Both are labeled MEASUREMENT. No candidate is called the target.

### A. Gate counts and `best_cand_consec` (MEASUREMENT)

Raw output:
```text
rows=2698 cols=25

==============================================================================
W1 1751-2084 (GT visible after absence): frames 1751-2084, rows=334
states: {'TRACKING': 334}
gate counts: {'not_lost': 334}
best_cand_consec populated rows: 0

==============================================================================
W1-pre 905-1750 (GT absence; candidate search): frames 905-1750, rows=846
states: {'LOST': 846}
gate counts: {'accept_fail': 804, 'tier_none': 30, 'nconfirm_fail': 10, '(empty)': 1, 'no_candidate': 1}
best_cand_consec populated rows: 844
  max best_cand_consec = 7 at frames [1750] (count 1)
  longest consecutive populated run = 837 frames 906-1742
  longest strictly increasing run = 1744-1750 (7 frames, 1 -> 7)

==============================================================================
W2 2110-2459 (named 2110-2460 region; tracker LOST): frames 2110-2459, rows=350
states: {'LOST': 350}
gate counts: {'accept_fail': 298, 'nconfirm_fail': 32, 'tier_none': 19, '(empty)': 1}
best_cand_consec populated rows: 349
  max best_cand_consec = 7 at frames [2459] (count 1)
  longest consecutive populated run = 349 frames 2111-2459
  longest strictly increasing run = 2452-2459 (8 frames, 0 -> 7)

==============================================================================
W3 2565-2697 (GT visible after 2505-2564 absence): frames 2565-2697, rows=133
states: {'LOST': 75, 'TRACKING': 58}
gate counts: {'not_lost': 58, 'accept_fail': 47, 'nconfirm_fail': 27, '(empty)': 1}
best_cand_consec populated rows: 74
  max best_cand_consec = 7 at frames [2580, 2671, 2683] (count 3)
  longest consecutive populated run = 58 frames 2626-2683
  longest strictly increasing run = 2573-2580 (8 frames, 0 -> 7)
```

What these numbers do and do not say:
- Window definitions are printed above. W1 is the GT-visible interval right after the 905-1750 absence; every row there is TRACKING and `gate=not_lost`, so the candidate gates only exist in the preceding absence (W1-pre). W2 is not a GT absence: in `5_manual.csv` the target is visible (1 or 2) throughout 2085-2504; 2110-2459 is a **tracker-LOST** window. W3 contains both LOST and TRACKING rows.
- In all three LOST/candidate windows, `accept_fail` is the most frequent gate. `nconfirm_fail` is present in all three and is the second most frequent in W2 and W3.
- The maximum logged `best_cand_consec` is 7 in every window. The log has no candidate fields on TRACKING rows, and each of these windows ends with a TRACKING transition on the next frame (1751, 2460, 2581/2684). This is consistent with `N_CONFIRM = 8` being completed on the first TRACKING frame; it is **not** evidence of a bug and no threshold is proposed.
- **The log does not identify which candidate is the true target.** These gate counts describe the best-scoring candidate only; they cannot state whether that candidate was the target. Confirming that requires your eyes.

### B. evaluate.py (MEASUREMENT)

Raw output:
```text
STEP 3: QUANTITATIVE TRACKING EVALUATION REPORT (evaluate.py)
================================================================================
  CSV Log File                  : logs/tracking_log_20261026_023427.csv
  Ground Truth File             : ground_truth/5_manual.csv
  Total Video Frames            : 2698 (89.93 s)
  Total Tracked Frames          : 1187 (44.0%)
  Target ID Switches            : 4
  Boundary Margin (uncertainty) : +-30 frames
  Total Uncertain Frames in GT  : 302 (partial or boundary margin)
  Uncertain Frames in TRACKING  : 8 (boundary margin: 8, partial vis=2: 0)
  FALSE LOCK Frames             : 0 [state=TRACKING while target_visible=0 outside margin]
  FALSE LOCK RATE               : 0.00% (Requirement: ~0%)
  Reacquisition Success         : 5/7 (71.4%)
  Avg Reacquisition Delay       : 12.0 frames (0.40 s)
================================================================================

PER-VIEW PERFORMANCE BREAKDOWN:
View     | Visible  | Tracked  | Coverage  | Raw Cov (all)  | Reacq Success  | Avg Reacq Delay    | False Lock
front    | 1115     | 958      | 85.9    % | 1051/1239 (84.8%) | 1/1            | 0.0f (0.00s)       | 0
back     | 28       | 0        | 0.0     % | 69/270 (25.6%) | 2/4            | 8.0f (0.27s)       | 0
side     | 0        | 0        | 0.0     % | 59/103 (57.3%) | 2/2            | 22.0f (0.73s)      | 0

PARTIAL VISIBILITY (target_visible = 2):
front 0/0 (0.0%), back 0/180 (0.0%), side 0/0 (0.0%), TOTAL 0/180 (0.0%)
```

Interpretation, with the tool's own caveat:
- The automatic FALSE LOCK count is **narrow**: it only counts `state=TRACKING` while `target_visible=0` outside the +-30 boundary margin. It **cannot** detect a green box on the wrong person while the target is visible. The `0.00%` printed by the script is not a general false-lock rate and must not be quoted as one.
- Correct formulation for this run: **no false lock found in sampled frames checked by eye, one video only** (the earlier eye checks of 90 sampled frames and of the exported segment tiles, one video). The automatic metric reports 0 absence-period frames, which is a different, narrower measure.
- Coverage per view is as printed; the back/side margin-clean denominators are tiny (28 back frames, 0 side frames after the +-30 margin), so those percentages rest on very small samples.
- evaluate.py left every GT md5 unchanged (`md5sum -c` -> 4x OK) and `git status` shows only the two new files from tonight (`MORNING_REPORT.md`, `label_target.py`).

### SELF-REVIEW (hostile reviewer)
- No file was edited; no decision branch could have changed. The only writes were this report and `label_target.py`.
- The gate table is descriptive. A hostile reviewer could object that W2 is mislabeled "after an absence"; the GT inspection above shows it is a tracker-LOST window, and I state that explicitly rather than hiding it.
- `best_cand_consec` is only the best-scoring candidate's counter; it is not an identity claim. No candidate is called the target.
- No threshold proposal and no E1/E2/E3 recommendation is made anywhere in this report.

### CLAIM AUDIT
| Claim | Producing output |
|---|---|
| W1 334 rows all TRACKING/not_lost | A output block W1 |
| W1-pre gate counts (804 accept_fail, 30 tier_none, 10 nconfirm_fail, 1 empty, 1 no_candidate) | A output block W1-pre |
| max consec 7 at 1750; increasing run 1744-1750 | A output block W1-pre |
| W2 gate counts (298 accept_fail, 32 nconfirm_fail, 19 tier_none, 1 empty) | A output block W2 |
| W2 max consec 7 at 2459; increasing run 2452-2459 | A output block W2 |
| W3 counts (58 not_lost, 47 accept_fail, 27 nconfirm_fail, 1 empty) | A output block W3 |
| W3 max consec 7 at 2580/2671/2683 | A output block W3 |
| coverage front 85.9%, back 0.0%, side n/a; raw 84.8/25.6/57.3 | B raw output |
| reacq 5/7, avg delay 12.0 frames | B raw output |
| automatic false-lock count 0 (narrow metric) | B raw output + tool warning |
| GT unchanged | `md5sum -c` -> 4x OK |

---

## DECISIONS MADE ALONE (ambiguity; conservative option chosen)

1. **`HANDOFF_2026-10-25.md` is missing.** It does not exist at the stated path and a depth-3 search found no HANDOFF file. I read `GEMINI_BRIEF.md` and `STATUS.md` completely and proceeded without the handoff. No conclusion depends on it.
2. **Second front GT interval 1751-2084 is skipped** by `label_target.py`. The instruction named only "front 0-904 at stride 60" and back/side intervals; I did not invent a stride for another front interval.
3. **Extra confirm keys.** `s/n/q` alone cannot record a box, so Enter/Space = accept box and advance and `r` = clear were added. No required key was removed.
4. **Label schema** `frame,status,x1,y1,x2,y2,view` with `status=box|not_visible`. No tracker ID column and no tracker log is read.
5. **SAFE ANALYSIS windows.** The instruction mixed GT and tracker language. I report the exact named ranges plus the LOST subset where gates exist (W1-pre). W2 (2110-2459) is stated as a tracker-LOST window, not a GT absence, because `5_manual.csv` shows visible=1/2 across it.
6. **No label CSV, no labeling session, no video run.** All three were forbidden or need your eyes/display.

## WHAT IS NOT DONE AND WHY

- **STEP 3 labeling itself was not run** and `ground_truth/labels_5_eye.csv` was not created: it requires your eyes and a display, and the instruction forbids running it.
- **Box IoU distribution was not computed**: it needs the hand boxes that do not exist yet.
- **Interactive path of `label_target.py` is untested**: no display use tonight; only compile and dry-run were allowed.
- **STEP 4 (GIVE_UP replacement) was not started**: explicitly not approved.
- **No video runs at all**, per instruction. A regression is not needed because `follow_target.py` was not edited.
- **No E1/E2/E3 recommendation and no threshold proposal** was made.

## FILES CHANGED / CREATED (as of 02:57)

| File | Status | Size |
|---|---|---|
| `label_target.py` | created (new) | 12,650 bytes |
| `MORNING_REPORT.md` | created (new; grows with these final sections) | 21,665 bytes at 02:57 |
| `STATUS.md` | edited: added "Overnight run 2026-10-26" | (see `ls -l`) |
| `roadmap_data.json` | edited: log entry + pending item | (see `ls -l`) |
| `ROADMAP.md`, `outputs/progress_roadmap.png` | regenerated by `update_roadmap.py` | (see `ls -l`) |
| `ground_truth/*` | **unchanged** | md5 verified before/after |
| `follow_target.py`, `evaluate.py`, `calibrate_thresholds.py`, `GEMINI_BRIEF.md`, `.gitignore` | **unchanged** | — |

`git status --short` at 02:57 showed only `?? MORNING_REPORT.md` and `?? label_target.py`; no tracked file was modified before the STATUS/roadmap bookkeeping.
Disk space left: **40G available** (`df -h .`: 89G size, 45G used, 40G avail, 54%).

## EXACT COMMANDS FOR YOU TO VERIFY

```bash
cd /home/foxsec/drone_project
. venv/bin/activate

# STEP 2 evidence (25 cols, 2698 rows, 0 differing cells in first 9 / first 22)
python3 - <<'PY'
import csv
def load(p):
    r = csv.reader(open(p, newline="")); h = next(r); return h, list(r)
h9,r9   = load("logs/tracking_log_20261020_132332.csv")
h22,r22 = load("logs/tracking_log_20261026_022150.csv")
h25,r25 = load("logs/tracking_log_20261026_023427.csv")
print(len(r25), len(h25), h25[-3:])
print("first9 diff:", sum(1 for a,b in zip(r9,r25) for j in range(9) if a[j]!=b[j]))
print("first22 diff:", sum(1 for a,b in zip(r22,r25) for j in range(22) if a[j]!=b[j]))
iv,ist,isa = h25.index("valid"),h25.index("state"),h25.index("sim_anchor")
m = sum(1 for x in r25 if int(x[iv]) != (1 if x[ist]=="INITIALIZING" or (x[ist]=="TRACKING" and x[isa]!="" and float(x[isa])>=0.60) else 0))
print("valid mismatches:", m)
PY

# STEP 3 tool (headless)
python3 -m py_compile label_target.py
python3 label_target.py --video videos/5.mp4 --gt ground_truth/5_manual.csv --dry-run

# SAFE ANALYSIS B
python3 evaluate.py --log logs/tracking_log_20261026_023427.csv --gt ground_truth/5_manual.csv --fps 30.0

# GT safety
md5sum ground_truth/*.csv
ls ground_truth/labels_5_eye.csv   # expected: No such file

# Interactive smoke test for the tool (needs your display; NOT run tonight)
python3 label_target.py --video videos/5.mp4
```

## FINAL CRITIQUE

**Three biggest weaknesses of tonight's work**
1. **The labeling tool's interactive path is unverified.** The mouse-to-video coordinate conversion, the key loop and the append/skip logic were never executed. A silent scale error would corrupt every hand box. Evidence that would change this: one manual session saving a box at a known pixel position, then comparing the CSV values with the frame.
2. **The SAFE ANALYSIS cannot answer "why was the correct candidate delayed".** The log has candidate scores but no identity, so gate counts describe the best-scoring candidate, not the target. W1's gate table is empty by construction (all TRACKING), and the 1751-2084 front interval is skipped in the labeling plan. Evidence that would change this: hand-labeled boxes per frame plus IoU, or a log that carries candidate IDs together with your eye labels.
3. **Identity correctness still rests on old, small samples.** The automatic false-lock metric is narrow and only sees target-absence periods; tonight adds no visual check. The correct statement remains: no false lock found in sampled frames checked by eye, one video only. Evidence that would change this: a systematic eye check on a second video, especially with similarly dressed people.

**What I am least sure about**
- That `best_cand_consec = 7` followed by a TRACKING transition fully accounts for `N_CONFIRM = 8`: the transition frame is logged as TRACKING with candidate fields empty, so the 8th step is inferred, not logged.
- That skipping the 1751-2084 front interval matches your intent for the label set.
- That `valid=1` during all 188 INITIALIZING frames is the intended semantics for downstream control; it is faithful to "a green box is drawn", but it is a hazard if control treats `valid` as "TRACKING".

**Evidence that would change my conclusions**
- A manual test of `label_target.py` (box coordinates and a clean `q` save).
- The 46-frame hand-label set and the box-IoU distribution against tracker boxes.
- A second video with a long loss and a similarly dressed person (STEP 4 material; not approved).
- Any log where the candidate identity is recorded together with your eye labels.

## SUGGESTED NEXT STEPS (for your approval; none executed)

1. Smoke-test `label_target.py` on 2-3 frames (draw, accept, `q`), verify the saved coordinates.
2. Run the 46-frame labeling session, then compute hand-box vs tracker-box IoU as a distribution (no IoU threshold fixed in advance).
3. If `valid` will feed a controller, require `state == TRACKING` in addition to `valid == 1` (no control code exists yet; nothing changed tonight).
4. For STEP 4, provide the new video with a long loss and a similarly dressed person; STEP 4 remains unapproved.

## TIME BUDGET

Start: 02:54. Last step executed: 02:58. Well inside the ~3 h budget; no step was skipped for time. No video run was attempted.

## FINAL FILE SIZES AND SAFETY RE-CHECK (02:58)

Raw output:
```text
 M ROADMAP.md
 M STATUS.md
 M roadmap_data.json
?? MORNING_REPORT.md
?? label_target.py
ground_truth/5.csv: OK
ground_truth/5_ai_draft.csv: OK
ground_truth/5_ai_draft_samples.csv: OK
ground_truth/5_manual.csv: OK
ls: cannot access 'ground_truth/labels_5_eye.csv': No such file or directory
2699 logs/tracking_log_20261026_023427.csv
cols=25
/dev/nvme0n1p5   89G   45G   40G  54% /
```

- `2699` lines in the log = **1 header + 2698 data rows**; 25 columns confirmed.
- Sizes (`ls -l` at 02:58): `label_target.py` 12,650 B; `STATUS.md` 57,701 B; `roadmap_data.json` 11,941 B; `ROADMAP.md` 11,491 B; `outputs/progress_roadmap.png` 337,628 B; this report was 28,747 B before this addendum.
- `follow_target.py`, `evaluate.py`, `calibrate_thresholds.py`, `GEMINI_BRIEF.md`, `.gitignore` do **not** appear in `git status` -> unchanged.
- `py_compile` created one incidental build artifact, `__pycache__/label_target.cpython-312.pyc` (19,183 B, gitignored). It is the only file besides `label_target.py` and `MORNING_REPORT.md` created by the compile test; it is not a data or report file and was left in place (no files were deleted).
- Roadmap regeneration output: `KẾT QUẢ: PASS - không có chữ bị cắt, không có chữ chồng lấn` (107 text objects, 0 outside frame, 0 overflow, 0 overlap). The PNG was opened and inspected; it renders correctly with the date 2026-10-26.
- GT md5 re-check after all work: 4x OK; `labels_5_eye.csv` still absent.
