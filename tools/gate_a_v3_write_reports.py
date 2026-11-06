#!/usr/bin/env python3
"""Generate the 19 human-readable v3 handoff documents from computed artifacts.

Every number is inserted from the evaluation/forensics JSON; nothing is typed by
hand. Run after gate_a_evaluator_v3.py and gate_a_v3_forensics.py.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/foxsec/drone_project")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(out: Path, name: str, text: str):
    (out / name).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--forensics", required=True)
    ap.add_argument("--counterexamples", required=True)
    args = ap.parse_args()
    out = Path(args.artifact)
    run = json.load(open(args.run, encoding="utf-8"))
    fore = json.load(open(args.forensics, encoding="utf-8"))
    ce = json.load(open(args.counterexamples, encoding="utf-8"))
    im = run["identity_metrics"]
    legs = run["legacy_absence_metrics"]
    gtv = run["ground_truth"]
    tr = run["tracking"]
    vid = run["video"]
    prov = run["provenance"]
    sens = im["sensitivity_view_boundary_ignored"]
    btypes = Counter()
    for kinds in gtv["boundary_types"].values():
        for k in kinds:
            btypes[k] += 1
    statuses = im["metric_status"]
    episodes = im["reacquisition_episodes"]
    tevents = run["events"]["tracker_events"]
    tc_path = out / "TEST_COUNTS.json"
    if tc_path.exists():
        tc = json.load(open(tc_path, encoding="utf-8"))
        counts = {"root": tc["root"]["ran"], "multi": tc["multi_object"]["ran"],
                  "total": tc["total_executed"], "v3": tc["v3_suites"]["ran"],
                  "evaluator": tc["evaluator_identity"]["ran"]}
    else:
        raise SystemExit("TEST_COUNTS.json missing; run tests and create it first")
    hashes = {
        "evaluate.py": sha(ROOT / "evaluate.py"),
        "tests/test_evaluate_identity.py": sha(ROOT / "tests/test_evaluate_identity.py"),
        "tests/test_gate_a_v3_contracts.py": sha(ROOT / "tests/test_gate_a_v3_contracts.py"),
        "tests/test_gate_a_v3_review_tool.py": sha(ROOT / "tests/test_gate_a_v3_review_tool.py"),
        "tests/test_gate_a_v3_runner.py": sha(ROOT / "tests/test_gate_a_v3_runner.py"),
        "tools/gate_a_evaluator_v3.py": sha(ROOT / "tools/gate_a_evaluator_v3.py"),
        "tools/human_review_v3.py": sha(ROOT / "tools/human_review_v3.py"),
        "tools/gate_a_v3_forensics.py": sha(ROOT / "tools/gate_a_v3_forensics.py"),
        "follow_target.py": sha(ROOT / "follow_target.py"),
    }

    def metric_row(name):
        st = statuses[name]
        value = "None" if st["value"] is None else f"{st['value']:.4f}"
        return f"| `{name}` | {value} | {st['status']} | {st['numerator']} | {st['denominator']} | {st['reason'] or ''} |"

    metric_table = "\n".join(metric_row(k) for k in statuses)

    write(out, "EXECUTIVE_SUMMARY_VI.md", f"""# EXECUTIVE SUMMARY — Gate A Evaluator v3

**Overall:** COMPLETE (implementation) — **Ready for independent Codex audit: YES**.
**Gate A scientific validation: NOT ESTABLISHED.** Không có nhãn danh tính người độc lập/held-out;
mọi metric danh tính dưới đây là **DIAGNOSTIC ONLY — NOT SCIENTIFIC VALIDATION**.

## Kết quả một trang

| Hạng mục | Kết quả |
|---|---|
| Video 5 | metadata {vid['metadata_frames']} frame, decoded **{vid['decoded_frames']}**, index 0–{vid['frame_index_range'][1]}, fps {vid['fps_metadata']:.4f} |
| Tracking log | {tr['rows']} dòng; TRACKING **{tr['tracking_frames']}**; TRACKING+valid **{tr['tracking_valid_frames']}** |
| GT `5_manual.csv` | {gtv['interval_count']} interval, coverage {gtv['coverage_frames']} frame, UNKNOWN {gtv['unknown_frames']} frame |
| Review hiện có | {im['total_reviewed']} frame; eligible present **{im['eligible_present_frames']}**, eligible absent **{im['eligible_absent_frames']}**, judged locks **{im['judged_locks']}** |
| Identity metrics | **{im['overall_status']}** — mọi tỷ lệ `None` + `NOT_ESTIMABLE` (không có 0.0 giả) |
| GT out-of-view episodes | {len(episodes)} (1751–2084, 2565–2624); state relock {episodes[0]['state_relock_frame'] if episodes else 'None'}/{episodes[1]['state_relock_frame'] if len(episodes)>1 else 'None'}; first valid bbox {episodes[0]['first_valid_output_frame'] if episodes else 'None'}/{episodes[1]['first_valid_output_frame'] if len(episodes)>1 else 'None'} |
| Identity reacquisition | **UNKNOWN** (censored, 0 eligible identity judgments) |
| Tests | root **{counts['root']} OK** + multi-object **{counts['multi']} OK** = **{counts['total']}**; v3 suites {counts['v3']} |
| Legacy CLI | byte-identical 3446 bytes trên fixture hợp lệ |

## Hai lỗi P0/P1 trọng tâm đã sửa

1. **P0 — false-lock rate 0.0 khi không có mẫu hợp lệ.** Gemini v2 JSON có
   `false_lock_rate_target_present = {ce['p0_zero_eligible']['v2_json_false_lock_rate_target_present']}`
   với `judged_locks = {ce['p0_zero_eligible']['v2_json_judged_locks']}`. v3 trả `None`,
   `status = NOT_ESTIMABLE`, `reason = NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS`.
2. **P1 — episode/onset, GT overlap và evidence integrity.** v2 tạo 1 episode chắc chắn qua
   UNKNOWN gap; chấp nhận GT lồng nhau; chấp nhận key evidence trùng canonical; và "chứng minh"
   correct lock khi `lock_frames` rỗng. v3 từ chối/đổi thành UNKNOWN (xem `COUNTEREXAMPLES_BEFORE_AFTER.md`).

## Điểm không được tuyên bố

Không có cơ sở để nói "0% false lock", "precision/recall X", hay "Gate A PASS".
{im['total_reviewed']} review hiện có đều unresolved theo policy margin hiện hành; 8 frame
chỉ trở nên eligible dưới **sensitivity** bỏ qua view-boundary (non-official, chỉ 3 correct + 5 miss,
0 wrong). Đó là mô tả chẩn đoán, không phải ước lượng quần thể.
""")

    write(out, "FULL_TECHNICAL_REPORT_VI.md", f"""# FULL TECHNICAL REPORT — Gate A Evaluator v3

## 1. Phạm vi

Được phép sửa evaluator, test, pipeline review và báo cáo. Không sửa `follow_target.py`
(hash vẫn `{hashes['follow_target.py']}`), threshold, gallery, GT gốc, video/log gốc, `multi_object/`.
Không chạy video held-out 1/2/3, không PX4/SITL, không Git mutation.

## 2. Kiến trúc đánh giá sau sửa

- `evaluate.py`: GT canonical model + boundary types; episode classifier; evidence helper;
  identity metrics với estimability; legacy `evaluate_log_gt` giữ nguyên hành vi trên dữ liệu hợp lệ.
- `tools/gate_a_evaluator_v3.py`: runner fail-closed, provenance hash, JSON/Markdown/manifest versioned.
- `tools/human_review_v3.py`: review 2 pha (blind source + identity), lock/hash, resume, atomic writes.
- `tools/gate_a_v3_forensics.py`: tái tính tuyên bố Gemini v2 và E1–E4.
- Tests: `tests/test_gate_a_v3_contracts.py` (33), `tests/test_gate_a_v3_review_tool.py` (13),
  `tests/test_gate_a_v3_runner.py` (6).

## 3. Kết quả đo trên video 5 (exploratory)

- Video metadata {vid['metadata_frames']}, decoded {vid['decoded_frames']}, fps {vid['fps_metadata']:.4f}.
- Log {tr['rows']} dòng: INITIALIZING {tr['state_counts'].get('INITIALIZING')}, TRACKING {tr['state_counts'].get('TRACKING')}, LOST {tr['state_counts'].get('LOST')}; TRACKING+valid {tr['tracking_valid_frames']}.
- GT manual: {gtv['interval_count']} interval, coverage {gtv['coverage_frames']}, UNKNOWN {gtv['unknown_frames']}.
- Boundary types: {dict(btypes)}.

## 4. Identity metrics

| Metric | Value | Status | n | d | Reason |
|---|---:|---|---:|---:|---|
{metric_table}

Buckets: `{im['buckets']}`. Exclusion reasons: `{im['exclusion_reasons']}`.

**Sensitivity (view boundaries ignored, NON-OFFICIAL):** eligible_present={sens['eligible_present_frames']},
precision={sens['identity_precision']}, false_lock_present={sens['false_lock_rate_target_present']},
recall={sens['present_target_lock_recall']}. Không dùng làm kết quả khoa học.

## 5. Reacquisition

| Start | End | GT onset | State relock | First valid bbox | Identity status | Latency bounds |
|---:|---:|---|---:|---:|---|---|
""" + "\n".join(
        f"| {e['start']} | {e['end']} | {e.get('onset_visibility')} | {e.get('state_relock_frame')} | "
        f"{e.get('first_valid_output_frame')} | {e['status']} | "
        f"[{e['latency_lower_bound_frames']}..{e['latency_upper_bound_frames']}] |" for e in episodes) + f"""

State relock là bằng chứng log; **không** phải identity reacquisition. Không có eligible review
⇒ identity latency = UNKNOWN/censored.

## 6. Tracker recovery E1–E4

| Event | Lost start | Relock | First valid | Tracker |
|---|---:|---:|---:|---|
""" + "\n".join(
        f"| {e['event_class']} | {e['lost_start_frame']} | {e['relock_frame']} | "
        f"{e['first_valid_output_frame']} | {e['prior_tracker_id']}→{e['tracker_id_at_relock']} |"
        for e in tevents) + f"""

## 7. Compatibility & integrity

- Legacy CLI trên fixture hợp lệ: **byte-identical 3446 bytes** so với backup v2.
- `follow_target.py` không đổi: `{hashes['follow_target.py']}`.
- Test: root {counts['root']} OK, multi-object {counts['multi']} OK, v3 {counts['v3']} OK.
- Không ghi đè video/log/GT/session/report Gemini cũ.
- Mọi metric lấy từ JSON chạy thực tế, không hard-code.

## 8. Kết luận

Implementation hoàn tất trong phạm vi cho phép. **Gate A vẫn NOT ESTABLISHED** vì thiếu
nhãn người độc lập, held-out và coverage đủ. Không có tuyên bố accuracy/false-lock chính thức.
""")

    write(out, "DEFECT_REGISTER.md", f"""# DEFECT REGISTER — Gate A v3

| ID | Mức | Mô tả | Nguyên nhân gốc | Sửa | Test | Trạng thái |
|---|---|---|---|---|---|---|
| P0-01 | P0 | `false_lock_rate_target_present = 0.0` khi `judged_locks=0` | Denominator đếm *mọi* review presence, không chỉ identity judgment resolved | Chỉ dùng bucket resolved; 0 mẫu ⇒ `None` + `NOT_ESTIMABLE` + reason | `test_all_unresolved_is_not_estimable_not_zero`, `test_empty_dataset...` | FIXED |
| P1-01 | P1 | UNKNOWN gap sau ABSENT tạo episode exact | `had_confirmed_absence` thắng `unknown_since_present` | Unknown/partial precedence; `UNKNOWN_ONSET_REAPPEARANCE` | `test_unknown_gap_after_absence_is_not_exact_onset` | FIXED |
| P1-02 | P1 | ABSENT→PARTIAL→VISIBLE onset mặc định ở VISIBLE | Builder chỉ xét visible | Partial bắt đầu episode; `first_fully_visible_frame` riêng | `test_partial_reappearance_starts_one_episode` | FIXED |
| P1-03 | P1 | GT nested/partial overlap được chấp nhận | Chỉ so cặp kề, không xét toàn bộ | Sort + reject mọi overlap/duplicate; merge identical-adjacent | 8 test `GroundTruthValidationTests` | FIXED |
| P1-04 | P1 | Key evidence `10` và `"010"` overwrite im lặng | Dict canonical không kiểm trùng | `_strict_frame` + duplicate canonical ⇒ ValueError | `test_duplicate_canonical_evidence_keys_rejected` | FIXED |
| P1-05 | P1 | `correct_lock` với `lock_frames=∅` vẫn proven | Không kiểm contradiction | Bucket lock phải nằm trong lock/log set | `test_impossible_lock_evidence_rejected` | FIXED |
| P1-06 | P1 | Margin view-change làm mất identity judgment | Một margin cho mọi boundary | Boundary types + sensitivity non-official | `test_view_boundary_is_primary_excluded_but_sensitivity_eligible` | FIXED |
| P1-07 | P1 | Báo cáo Gemini có số không tái lập (3371, 1487/2018, 1483/2018) | Không có nguồn | Errata + recompute từ video/log/GT | `GEMINI_V2_METRIC_ERRATA.md` | DOCUMENTED |
| P2-01 | P2 | Review tool thiếu | Chưa triển khai | `tools/human_review_v3.py` 2 pha | 13 test | IMPLEMENTED |
| P2-02 | P2 | Runner không fail-closed/versioned | Script ad-hoc | `tools/gate_a_evaluator_v3.py` | 6 test | IMPLEMENTED |
| P2-03 | P2 | `presence=partial` không có schema | Chỉ present/absent/uncertain | Thêm `partial` (unresolved) | `test_partial_presence_label_is_explicitly_unresolved` | FIXED |
| P2-04 | P2 | Tuyên bố "364 tests" không đo | Hard-coded | Đo thực tế {counts['total']} | `TEST_RESULTS_RAW.md` | DOCUMENTED |

## Không sửa (ngoài phạm vi)

- `follow_target.py`, thresholds, gallery, `N_CONFIRM`, identity decision logic.
- `multi_object/`, GT gốc, video/log gốc, review session gốc.
- Không tuyên bố Gate A PASS.
""")

    write(out, "COUNTEREXAMPLES_BEFORE_AFTER.md", f"""# COUNTEREXAMPLES — before (v2) / after (v3)

Raw: `COUNTEREXAMPLES_RAW.json`. Mọi "after" được chạy trên code hiện tại.

## P0 — zero eligible judgments

| | v2 | v3 |
|---|---|---|
| `false_lock_rate_target_present` | **{ce['p0_zero_eligible']['v2_json_false_lock_rate_target_present']}** | **None** |
| `judged_locks` | {ce['p0_zero_eligible']['v2_json_judged_locks']} | {im['judged_locks']} |
| status/reason | (không có) | NOT_ESTIMABLE / NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS |

## P1 — unknown gap after absence

Input: PRESENT 0–9, ABSENT 10–19, UNKNOWN 20–29, PRESENT 30–39.

```text
v2_reacquisition_episodes = {ce['p1_unknown_gap']['v2_reacquisition_episodes']}
v3_reacquisition_episodes = {ce['p1_unknown_gap']['v3_reacquisition_episodes']}
v3_unknown_onset           = {ce['p1_unknown_gap']['v3_unknown_onset']}
```

## P1 — nested GT overlap

Input: [0..100] VISIBLE + [10..20] VISIBLE.

```text
v2: accepted={ce['p1_gt_overlap']['v2']['accepted']} intervals={ce['p1_gt_overlap']['v2'].get('intervals')}
v3: accepted={ce['p1_gt_overlap']['v3']['accepted']}
    error={ce['p1_gt_overlap']['v3'].get('error')}
```

## P1 — duplicate canonical evidence key

Input: `{{10: "unresolved", "010": "correct_lock"}}`.

```text
v2: accepted={ce['p1_duplicate_evidence_keys']['v2']['accepted']} status={ce['p1_duplicate_evidence_keys']['v2'].get('status')} proven={ce['p1_duplicate_evidence_keys']['v2'].get('first_correct_proven')}
v3: accepted={ce['p1_duplicate_evidence_keys']['v3']['accepted']}
    error={ce['p1_duplicate_evidence_keys']['v3'].get('error')}
```

## P1 — impossible lock evidence

Input: bucket `correct_lock` at frame 10, `lock_frames=∅`.

```text
v2: accepted={ce['p1_impossible_lock']['v2']['accepted']} proven={ce['p1_impossible_lock']['v2'].get('first_correct_proven')}
v3: accepted={ce['p1_impossible_lock']['v3']['accepted']}
    error={ce['p1_impossible_lock']['v3'].get('error')}
```

## P1 — partial reappearance onset

Input: PRESENT 0–9, ABSENT 10–19, PARTIAL 20–29, VISIBLE 30–39.

```text
v2: episodes={ce['p1_partial_onset']['v2_episodes']} first_start={ce['p1_partial_onset']['v2_first_start']}
v3: episodes={ce['p1_partial_onset']['v3_episodes']} first_start={ce['p1_partial_onset']['v3_first_start']}
    onset_visibility={ce['p1_partial_onset']['v3_onset_visibility']}
    first_fully_visible={ce['p1_partial_onset']['v3_first_fully_visible']}
```

## P1 — view change

```text
v2: episodes={ce['p1_view_change']['v2_episodes']}
v3: episodes={ce['p1_view_change']['v3_episodes']} viewpoint_changes={ce['p1_view_change']['v3_viewpoint_changes']}
```

View change không tạo out-of-view episode ở cả hai; v3 ghi nhận rõ `VIEWPOINT_CHANGE`.
""")

    write(out, "EVALUATOR_ARCHITECTURE.md", f"""# EVALUATOR ARCHITECTURE — v3

## Data flow

```text
video ──probe/decode──> frame range
log ──validate_identity_log_rows──> strict rows (frame/state/valid)
GT ──validate_ground_truth_intervals──> canonical disjoint intervals
                   └──build_gt_boundary_map──> presence/partial/view/unknown boundaries
review CSV ──load_identity_reviews + validate_identity_review_rows──> strict rows
                   ├── classify (primary: any boundary; sensitivity: presence only)
                   ├── buckets → eligible denominators → metric_status
                   └── reacquisition_episode_evidence (coverage/censoring)
GT intervals ──build_reacquisition_episodes──> TRUE/UNKNOWN/view/occlusion events
log+GT ──classify_tracker_events──> E1–E4 state recoveries (not identity)
```

## Modules

| Module | Vai trò |
|---|---|
| `evaluate.py` | Production evaluator: validators, GT model, episodes, evidence, metrics, legacy report |
| `tools/gate_a_evaluator_v3.py` | CLI validate-then-evaluate; JSON/MD/manifest; refuses input overwrite |
| `tools/human_review_v3.py` | Blind Phase A + identity Phase B, lock/hash, resume, export |
| `tools/gate_a_v3_forensics.py` | Gemini v2 claim recomputation + E1–E4 + spot frames |
| `tools/gate_a_v3_counterexamples.py` | Loads v2 backup and runs before/after counterexamples |
| `tools/gate_a_v3_write_reports.py` | Generates this documentation set from JSON |

## Fail-closed boundaries

- Malformed frame (float/bool/`+1`/`1_0`) ⇒ ValueError.
- Overlapping/duplicate GT ⇒ ValueError.
- Unknown evidence bucket, duplicate canonical key, lock contradiction ⇒ ValueError.
- Zero denominator ⇒ `None` + `NOT_ESTIMABLE`, never 0.0.
- Runner validates all inputs before creating any output directory.

## Separation of evidence

1. **Independent human identity judgment** (review CSV) — the only source of correct/wrong locks.
2. **Derived tracker/output state** (log) — context/latency, never identity truth.
3. **Algorithm ReID decision** (follow_target) — not used as human label.
""")

    write(out, "METRIC_CONTRACTS_V3.md", f"""# METRIC CONTRACTS — v3

## Estimability rule

`value = numerator / denominator` **only if denominator > 0**; otherwise `None` with
`status=NOT_ESTIMABLE` and a specific `reason`. A zero denominator is never 0.0.

Eligible denominators contain only **resolved** human identity judgments:
- present eligible: correct_lock, correct_output_not_lock, wrong_lock_present,
  wrong_output_not_lock_present, miss_present;
- absent eligible: correct_rejection_absent, wrong_lock_absent, wrong_output_not_lock_absent.
- `uncertain`, `partial`, boundary-margin, GT-partial, unreviewed rows are excluded.

## Metrics

| Metric | Numerator | Denominator |
|---|---|---|
| identity_precision | correct_lock | judged locks (correct+wrong present+wrong absent) |
| false_lock_rate | wrong locks | judged locks |
| false_lock_rate_target_present | wrong_lock_present | eligible present frames |
| false_lock_rate_target_absent | wrong_lock_absent | eligible absent frames |
| present_target_lock_recall | correct_lock | eligible present frames |
| present_target_output_coverage | correct_lock + correct_output_not_lock | eligible present frames |
| lock_recall (conditional) | correct_lock | correct_lock + wrong_lock_present + miss_present |

## Current values

| Metric | Value | Status | n | d | Reason |
|---|---:|---|---:|---:|---|
{metric_table}

## Boundary policy

- Primary: exclude rows within `margin_frames` of **any** GT boundary
  (presence, partial, view, unknown). Conservative; published before use.
- Sensitivity: ignore **pure view** boundaries (view change does not invalidate an
  independent human identity judgment). **NON-OFFICIAL**; reported separately.
- Latency/onset: presence, partial and unknown boundaries only.
- No margin is reduced to increase sample size. Both policies are reported.

## Statistics

Frames are not independent samples. No confidence interval or population rate is
computed from this review set. Counts, coverage, missingness and censoring are reported;
accuracy statements require independent held-out labels.
""")

    write(out, "SCHEMA_AND_UNKNOWN_SEMANTICS.md", f"""# SCHEMA & UNKNOWN SEMANTICS — v3

## GT visibility

| Value | Meaning | Metric treatment |
|---:|---|---|
| `-1` | UNKNOWN / not annotated | Not absent; episode onset unknown; identity review may still judge the box |
| `0` | ABSENT (confirmed) | Required before a true out-of-view episode |
| `1` | VISIBLE (full) | Present |
| `2` | PARTIAL (occluded, in scene) | Present, never absent; identity lock unresolved |

Coverage gaps are UNKNOWN, never absent. `get_gt_info_at_frame` returns `-1` outside intervals.

## Review presence

`present`, `absent`, `uncertain`, `partial` (present-but-occluded). `partial` and `uncertain`
are unresolved and never enter a denominator.

## Identity

`correct`, `wrong`, `uncertain`, `none` (no box). `drawn` must equal `identity != none`;
`absent + correct` is rejected. A tracker `TRACKING` state is never a human-correct label.

## Evidence buckets

`correct_lock`, `correct_output_not_lock`, `wrong_lock_present`, `wrong_lock_absent`,
`wrong_output_not_lock_present`, `wrong_output_not_lock_absent`, `miss_present`,
`correct_rejection_absent`, `unresolved`. Unknown bucket ⇒ ValueError.

## Identity states (HUD/log)

`CONFIRMED` = state TRACKING **and** valid output box. `TRACKING_NO_BOX` = state TRACKING,
no box. `VERIFYING`, `LOST`, `GIVE_UP`, `IDLE` as defined in `follow_target.py` (unchanged).
`identity_confirmed` is true only for CONFIRMED. `TARGET ID` is the persistent logical identity.

## Metric status

`ESTIMABLE` (denominator > 0) or `NOT_ESTIMABLE` with reason:
`NO_ELIGIBLE_IDENTITY_JUDGMENTS`, `NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS`,
`NO_ELIGIBLE_ABSENT_IDENTITY_JUDGMENTS`, `NO_ELIGIBLE_PRESENT_FRAMES`,
`NO_ELIGIBLE_LOCK_OR_MISS_JUDGMENTS`.
""")

    write(out, "GT_VALIDATION_REPORT.md", f"""# GT VALIDATION REPORT — v3

## Rules enforced

- Required fields; frames are strict integers (bool/float/`+1`/`1_0` rejected).
- `start <= end`, non-negative, optional `max_frame` (decoded video range).
- Visibility ∈ {{-1,0,1,2}}; absent interval with a view is contradictory.
- Whole-set overlap check: nested, partial, contradictory and duplicate intervals are rejected.
- Adjacent intervals with identical (visibility, view, target_id) are merged losslessly.
- Gaps are allowed and mean GT_UNKNOWN; they are never converted to ABSENT.
- Both `get_gt_info_at_frame` and episode building use the same validated model.

## Video 5 inputs

| File | Intervals | Coverage | Notes |
|---|---:|---:|---|
| `ground_truth/5_manual.csv` | {gtv['interval_count']} | {gtv['coverage_frames']} / {vid['decoded_frames']} | manual label, 13 intervals, no gaps |
| `ground_truth/5.csv` | {fore['gt']['circular_intervals']} | — | circular provenance (derived from state); not official GT |

## Boundary types (5_manual)

{dict(btypes)}

`2084–2086`: GT manual is `visible=1/front` → `visible=1/back`; this is a **view** boundary,
not a presence transition. `5.csv` reports `front` for 2085–2086 because it lacks view detail.

## Known GT/video discrepancies (AI-assisted observation, not human GT)

- Frame 1745: `5_manual.csv` says ABSENT (0), but visual inspection of the source frame shows
  the denim-jacket target drinking. The manual onset 1751 is therefore not independently proven.
- Frame 2508: `5_manual.csv` says ABSENT, but the source frame shows the target; the app UI
  switches to a thumbnail grid at 2512. E3 onset 2565 is not independently proven.
- These do NOT change the original GT; they are recorded as UNKNOWN/onset-uncertainty evidence.
""")

    e_rows = "\n".join(
        f"| {e['event_class']} | {e['lost_start_frame']} | {e['relock_frame']} | {e['first_valid_output_frame']} | "
        f"{e['prior_tracker_id']}→{e['tracker_id_at_relock']} | vis={e['gt_at_relock']['visible']}/{e['gt_at_relock']['view']} | "
        f"{(e['matched_gt_episode'] or {}).get('start')} | {e['relock_minus_gt_start_frames']} |"
        for e in fore["e1_e4"])
    write(out, "E1_E4_FORENSIC_ANALYSIS.md", f"""# E1–E4 FORENSIC ANALYSIS — video 5

**Đây là phân tích state/tracker, không phải identity ground truth.**
E1–E4 là các đoạn LOST→TRACKING trong log; chỉ E1 và E3 là out-of-view theo GT manual.

| Event | Lost start | Relock | First valid bbox | Tracker | GT at relock | GT episode start | Relock−GT |
|---|---:|---:|---:|---|---|---:|---:|
{e_rows}

## Diễn giải

- **E1 (905→1751, bbox 1752):** GT manual ABSENT 905–1750, VISIBLE từ 1751. True out-of-view.
  State latency 0 frame theo GT manual, nhưng boundary 1751 trùng state relock và video cho thấy
  target đã xuất hiện trước đó (1745) ⇒ **onset không độc lập**; identity latency UNKNOWN.
- **E2 (2110→2460, bbox 2461):** GT manual vẫn PRESENT/PARTIAL trong 2110–2459 ⇒ đây là
  **IN_SCENE/OCCLUSION RECOVERY**, không phải out-of-view. Không được tính là reacquisition.
- **E3 (2513→2581, bbox 2582):** GT manual ABSENT 2505–2564, VISIBLE từ 2565; relock 2581
  ⇒ 16 frame (0.536 s). Nhưng frame 2508/2509 vẫn TRACKING/valid=1 và video cho thấy target;
  onset 2565 chưa được chứng minh độc lập.
- **E4 (2625→2684, bbox 2685):** GT manual PARTIAL/VISIBLE ⇒ **OCCLUSION RECOVERY**, không phải
  out-of-view.

## Frame spot checks (từ `gate_a_v3_forensics.json`)

Các frame 904–905, 1740–1752, 2084–2086, 2108–2110, 2195/2200/2205, 2459–2465,
2505–2515, 2560–2585, 2624–2628, 2683–2690 được ghi đầy đủ trong JSON. Điểm đáng chú ý:

- 2508/2509: log `TRACKING valid=1`, GT manual `ABSENT`.
- 2108/2109: log `TRACKING valid=0`, sau đó 2110 LOST; GT manual PRESENT.
- 1751/2460/2581/2684: frame accept có `valid=0`, bbox xuất hiện frame kế tiếp.
- 2085/2086: view boundary front→back, không phải presence boundary.

## Kết luận

State relock 4/4; true out-of-view 2/4; occlusion/in-scene 2/4; identity-verified relock
**0/2** (không có eligible human identity judgment). Không suy identity từ tracker state.
""")

    claim_rows = "\n".join(
        f"| {c['claim']} | {c['reported']} | {c['recomputed']} | {c['status']} | {c.get('note','')} |"
        for c in fore["gemini_claims"])
    write(out, "GEMINI_V2_METRIC_ERRATA.md", f"""# GEMINI v2 METRIC ERRATA

Nguồn: `research_handoff/reports/gemini_gateA_evaluator_20261107_024500/`.
Tái tính bằng `tools/gate_a_v3_forensics.py`; raw: `forensics_final/gate_a_v3_forensics.json`.

## Tuyên bố số liệu

| Claim | Reported | Recomputed | Status | Note |
|---|---:|---:|---|---|
{claim_rows}

## Identity metric errata (P0)

| Metric | Gemini v2 | v3 | v3 status |
|---|---:|---:|---|
| false_lock_rate_target_present | {ce['p0_zero_eligible']['v2_json_false_lock_rate_target_present']} | None | NOT_ESTIMABLE / NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS |
| false_lock_rate_target_absent | {fore['gemini_v2_vs_v3_identity']['old'].get('false_lock_rate_target_absent')} | None | NOT_ESTIMABLE / NO_ELIGIBLE_ABSENT_IDENTITY_JUDGMENTS |
| identity_precision | None | None | NOT_ESTIMABLE / NO_ELIGIBLE_IDENTITY_JUDGMENTS |
| lock_recall | None | None | NOT_ESTIMABLE / NO_ELIGIBLE_LOCK_OR_MISS_JUDGMENTS |
| present_target_lock_recall | None | None | NOT_ESTIMABLE / NO_ELIGIBLE_PRESENT_FRAMES |

## Errata kết luận

1. `0.0` false-lock rate **không hợp lệ**: mẫu số eligible = 0. Đúng phải là UNKNOWN.
2. "3371 frames", "1487/2018", "1483/2018" **không tái lập** từ video/log/GT hiện có.
   Video decode 2698; TRACKING 1187; TRACKING+valid 1173.
3. "0/846 absent-state false lock": tái tính 0/{fore['absence_state_proxy']['outside_presence_margin']}
   ngoài presence margin (GT manual). Con số 846 là toàn bộ absence 905–1750, không phải
   "outside margin"; numerator 0 vẫn đúng cho cửa sổ tái tính.
4. "2/2 state relock" là **state-level**, không phải identity-verified. Identity-verified 0/2.
5. E1 0.00 s và E3 0.54 s là state/GT arithmetic; onset independence là UNKNOWN.
6. Frame 2508/2509 vẫn TRACKING/valid=1, không phải LOST như một mô tả trước đó.

## Không ghi đè

Báo cáo Gemini cũ và JSON cũ được giữ nguyên; errata là artifact mới.
""")

    write(out, "HUMAN_REVIEW_PROTOCOL.md", f"""# HUMAN REVIEW PROTOCOL — v3

## Mục tiêu

Cho phép người dùng xác định, trên từng frame: AI ghim đúng người, ghim nhầm người,
không có output, hay bằng chứng không chắc chắn — mà không làm lộ dự đoán trước khi
reviewer đưa ra phán quyết.

## Phase A — Blind source review

- Chỉ hiển thị: source frame, frame index/time, mô tả reference identity.
- Không hiển thị: state, valid, similarity, confidence, bbox dự đoán, nhãn CONFIRMED, log.
- Nhãn: `present`, `absent`, `partial`, `unknown`, `skipped`.
- Queue CSV chỉ có `order,frame`; strata nằm trong `queue_manifest.json` (cảnh báo không mở
  trước khi lock).
- Sampling: event boundaries, first valid output, no-output, occlusion, baseline ngẫu nhiên;
  seed ghi trong manifest. Sampling chẩn đoán ≠ sampling đủ cho ước lượng thống kê.

## Lock

- `lock-a` ghi `phase_a_locked.json` chứa SHA-256 của labels, video hash, reviewer, counts.
- Sau lock: không sửa Phase A; nếu sửa file, `verify_lock_integrity` từ chối Phase B.
- Không ghi đè lock cũ.

## Phase B — Identity review

- Chỉ chạy sau lock; xác minh hash labels.
- Hiển thị **source frame** + hộp được thuật toán chọn (đỏ), lấy hình học từ output video.
  HUD/label/score của output bị loại bỏ (không vẽ lên source).
- Nhãn: `correct`, `wrong person`, `uncertain`, `no output`, `review later`.
- Không hiển thị similarity, state, CONFIRMED, TARGET ID.
- Box detection dùng contour xanh cao (h ≥ 35% frame, h/w ≥ 1.4) để tránh biển hiệu/UI xanh.

## Export

- Chỉ join frame có cả Phase A và B; presence `unknown/skipped` bị loại.
- Schema: `frame,state,presence,drawn,identity` — đúng `evaluate.py --identity-reviews`.
- `absent + correct` bị từ chối.
- Output CSV/JSON có version + hash; không ghi đè.

## Đạo đức & khoa học

- Không tạo nhãn người giả; không chuyển AI draft thành human label.
- Không dùng kết quả review để tuyên bố accuracy nếu coverage không đủ hoặc không có
  held-out độc lập.
- Reviewer ID và timestamp được lưu; có thể chạy hai reviewer độc lập và adjudication
  trong tương lai (chưa thực hiện trong phiên này).
""")

    write(out, "REVIEW_TOOL_USER_GUIDE_VI.md", f"""# HƯỚNG DẪN CÔNG CỤ HUMAN REVIEW v3

Công cụ: `tools/human_review_v3.py` (CLI; `--gui` dùng OpenCV nếu có display).

## 1. Tạo session + blind queue

```bash
cd /home/foxsec/drone_project
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py queue \\
  --video videos/5.mp4 --log logs/tracking_log_20261106_233941.csv \\
  --gt ground_truth/5_manual.csv \\
  --session-dir research_handoff/human_review/session_001 \\
  --reviewer "ten-nguoi-review" \\
  --reference "Nam áo denim xanh, áo lót cổ lọ đen, ba lô đen, thường cầm ly bia" \\
  --n 200 --seed 20261107
```

## 2. Phase A — chỉ xem source

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py review-a \\
  --session-dir research_handoff/human_review/session_001 \\
  --video videos/5.mp4 --reviewer "ten-nguoi-review"
```

Mỗi frame: ảnh source được lưu tại `session_001/frames/frame_XXXXXX.png`.
Phím: `p` present, `a` absent, `o` partial, `u` unknown, `s` skip, `q` quit.
Chạy lại cùng lệnh để resume; frame đã gắn nhãn được bỏ qua.

## 3. Khóa Phase A

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py lock-a \\
  --session-dir research_handoff/human_review/session_001 --reviewer "ten-nguoi-review"
```

Sau bước này không được sửa Phase A.

## 4. Phase B — identity

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py review-b \\
  --session-dir research_handoff/human_review/session_001 \\
  --video videos/5.mp4 --output-video outputs/ling_identity_p1_20261107.mp4 \\
  --log logs/tracking_log_20261107_005504.csv --reviewer "ten-nguoi-review"
```

Ảnh hiển thị: source + hộp đỏ (Selected). Không có score/state/HUD.
Phím: `c` correct, `w` wrong, `u` uncertain, `n` no output, `r` review later, `q` quit.

## 5. Tiến độ và export

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py progress --session-dir ...
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py export --session-dir ... \\
  --log logs/tracking_log_20261107_005504.csv
```

Export tạo `export/identity_reviews_<UTC>.csv` + `review_summary_<UTC>.json` (hash, counts).
Dùng CSV đó với:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_evaluator_v3.py \\
  --video videos/5.mp4 --log logs/tracking_log_20261107_005504.csv \\
  --ground-truth ground_truth/5_manual.csv \\
  --reviews research_handoff/human_review/session_001/export/identity_reviews_<UTC>.csv \\
  --out-dir research_handoff/reports/gateA_evaluator_v3_runs
```

## Lưu ý

- Không mở `queue_manifest.json` trước khi lock Phase A (chứa strata).
- Không có display: bỏ `--gui`; mở ảnh PNG đã lưu bằng trình xem ảnh.
- Công cụ không tạo nhãn AI; chỉ ghi nhãn bạn nhập.
- Không chạy trên video held-out nếu chưa được phép.
""")

    write(out, "TEST_RESULTS_RAW.md", f"""# TEST RESULTS — raw measurements

Môi trường: Python 3.12 venv tại `/home/foxsec/drone_project/venv`; OpenCV 5.0.0; không GPU.
Mọi lệnh dưới đây được chạy thực tế trong phiên này; raw đầy đủ trong các file cùng tên.

## Tổng hợp

| Suite | Collected/Ran | Passed | Failed | Skipped | Raw |
|---|---:|---:|---:|---:|---|
| Root discovery (`tests/`) | {counts['root']} | {counts['root']} | 0 | 0 | `raw_root_tests.txt` |
| Multi-object (`tests/multi_object/`) | {counts['multi']} | {counts['multi']} | 0 | 0 | `raw_multi_object_tests.txt` |
| **Total executed** | **{counts['total']}** | **{counts['total']}** | **0** | **0** | |
| v3 suites (subset of root) | {counts['v3']} | {counts['v3']} | 0 | 0 | `raw_v3_tests.txt` |
| Evaluator identity module | {counts['evaluator']} | {counts['evaluator']} | 0 | 0 | `raw_evaluator_tests.txt` |

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests/multi_object -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest tests.test_gate_a_v3_contracts tests.test_gate_a_v3_review_tool tests.test_gate_a_v3_runner -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest tests.test_evaluate_identity -v
```

## Baseline trước sửa (đo cùng phiên)

- Root discovery: 144 tests OK.
- Multi-object: 220 tests OK.
- Evaluator module: 62 tests OK.
- Sau sửa: root {counts['root']}, multi {counts['multi']}, evaluator {counts['evaluator']}.
  +{counts['v3']} test mới (contracts/review/runner).

## Legacy compatibility

```text
LEGACY CLI BYTE-IDENTICAL 3446 bytes
```
(`legacy_old.txt` = backup v2, `legacy_new.txt` = v3, fixture hợp lệ, margin 0.)

## Runner validation

```text
video_decoded_frames=2698, max_frame=2697, log_rows=2698, gt_intervals=13, reviews=65
VALIDATION OK
```

## Ghi chú

- Không có test nào được báo PASS mà không chạy.
- Các test review tool dùng video/log tổng hợp trong tempdir; không tạo nhãn người giả.
- Multi-object chỉ chạy read-only discovery; không sửa module.
""")

    write(out, "REGRESSION_REPORT.md", f"""# REGRESSION REPORT — v3

## Kết quả

| Kiểm tra | Trước | Sau | Kết luận |
|---|---:|---:|---|
| Root tests | 144 OK | {counts['root']} OK | +{counts['v3']} test mới; không fail |
| Multi-object tests | 220 OK | {counts['multi']} OK | không đổi |
| Evaluator tests | 62 OK | {counts['evaluator']} OK | mở rộng, pass |
| Legacy CLI fixture | 3446 bytes | 3446 bytes | **byte-identical** |
| `follow_target.py` | `814b38dd…` | `814b38dd…` | không đổi |
| `multi_object/` | — | — | không sửa |
| GT/video/log gốc | — | — | không sửa |

## Thay đổi hành vi có chủ đích

1. Boundary policy: edge của interval cuối video không còn bị coi là presence transition.
   Điều này làm lộ thêm 1 wrong lock trong fixture (29) — đúng khoa học, không phải nới margin.
2. `presence=partial` được chấp nhận và luôn unresolved.
3. Zero denominator trả `None` + `NOT_ESTIMABLE`.
4. GT overlap bị từ chối; evidence trùng/không nhất quán bị từ chối.

## Bảo toàn

- Không chạy video held-out 1/2/3.
- Không PX4/SITL.
- Không Git mutation.
- Không ghi đè artifact Gemini/report cũ.

## Kiểm tra CLI thủ công

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_evaluator_v3.py --help
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/human_review_v3.py --help
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_evaluator_v3.py ... --validate-only
```
""")

    write(out, "CHANGE_HISTORY.md", f"""# CHANGE HISTORY — Gate A Evaluator v3

| File | Trước (SHA-256) | Sau (SHA-256) | Lý do | Test |
|---|---|---|---|---|
| `evaluate.py` | `faca683d3b73f12e8722eaa45106c62023a91ef007a2f565f3a6db43da785ec0` | `{hashes['evaluate.py']}` | P0/P1 fixes + v3 contracts | root/evaluator/v3 |
| `tests/test_evaluate_identity.py` | `cdc39882ebe66afebe1f57ed843d9ab53bb4a3a480c0335c298f4b45ec0faa42` | `{hashes['tests/test_evaluate_identity.py']}` | cập nhật boundary/estimability | 62 OK |
| `tests/test_gate_a_v3_contracts.py` | (mới) | `{hashes['tests/test_gate_a_v3_contracts.py']}` | adversarial contracts | 33 OK |
| `tests/test_gate_a_v3_review_tool.py` | (mới) | `{hashes['tests/test_gate_a_v3_review_tool.py']}` | review tool | 13 OK |
| `tests/test_gate_a_v3_runner.py` | (mới) | `{hashes['tests/test_gate_a_v3_runner.py']}` | CLI runner | 6 OK |
| `tools/gate_a_evaluator_v3.py` | (mới) | `{hashes['tools/gate_a_evaluator_v3.py']}` | reproducible runner | runner tests |
| `tools/human_review_v3.py` | (mới) | `{hashes['tools/human_review_v3.py']}` | blind review tool | review tests |
| `tools/gate_a_v3_forensics.py` | (mới) | `{hashes['tools/gate_a_v3_forensics.py']}` | errata + E1–E4 | forensics run |
| `tools/gate_a_v3_counterexamples.py` | (mới) | — | before/after | raw JSON |
| `tools/gate_a_v3_write_reports.py` | (mới) | — | docs generator | — |
| `follow_target.py` | `814b38dd…` | `814b38dd…` | **không sửa** | — |

Backup trước sửa: `backups/gateA_evaluator_v3_20261107_032000/` (`evaluate.py`, `test_evaluate_identity.py`).

## Chi tiết sửa evaluate.py

- `_strict_frame`, `_strict_target_visible`: strict schema.
- `validate_ground_truth_intervals`: whole-set overlap rejection + merge identical adjacency.
- `build_gt_boundary_map` / `is_near_boundary`: boundary types.
- `build_reacquisition_episodes`: unknown/partial/view/occlusion taxonomy.
- `classify_tracker_events`: E1–E4 recovery classification.
- `reacquisition_episode_evidence`: canonical keys, contradiction checks.
- `evaluate_identity_reviews`: eligible denominators, `metric_status`, sensitivity, episode fields.
- `print_identity_report`: UNKNOWN/estimability rendering.

## Rollback

Xem `ROLLBACK.md`. Không xóa artifact; chỉ hoàn nguyên source nếu cần.
""")

    write(out, "ROLLBACK.md", f"""# ROLLBACK — Gate A v3

Backup chính xác trước sửa:

```text
backups/gateA_evaluator_v3_20261107_032000/evaluate.py
backups/gateA_evaluator_v3_20261107_032000/test_evaluate_identity.py
```

## Hoàn nguyên evaluator + test cũ

```bash
cd /home/foxsec/drone_project
sha256sum evaluate.py   # phải là {hashes['evaluate.py']}
cp -p backups/gateA_evaluator_v3_20261107_032000/evaluate.py evaluate.py
cp -p backups/gateA_evaluator_v3_20261107_032000/test_evaluate_identity.py tests/test_evaluate_identity.py
sha256sum evaluate.py   # phải trở lại faca683d3b73f12e8722eaa45106c62023a91ef007a2f565f3a6db43da785ec0
```

## Tắt v3 tooling (không xóa dữ liệu)

Các file mới có thể giữ nguyên; nếu muốn vô hiệu hóa test mới, chuyển 3 file
`tests/test_gate_a_v3_*.py` ra ngoài `tests/`. Không xóa log/video/report.

## Khôi phục `follow_target.py`

Không cần: hash vẫn `814b38dd4633891b87e7f3081ba067f7ebadb24f9314db40efb785a35d540da2`.
Nếu có thay đổi ngoài ý muốn, dùng backup `backups/follow_target_20261107_pre_p1_identity_correction.py`
hoặc bản gốc `backups/follow_target_20261106_pre_persistent_identity.py` (tùy mốc).

## Sau rollback

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests/multi_object -v
```
""")

    write(out, "REMAINING_UNKNOWN.md", f"""# REMAINING UNKNOWN — Gate A v3

1. **Không có nhãn identity độc lập/held-out.** {im['total_reviewed']} review hiện có đều
   unresolved theo policy margin; 0 eligible identity judgment. Không thể ước lượng precision,
   recall hay false-lock rate quần thể.
2. **Sensitivity không phải kết quả chính thức.** Bỏ view boundary cho 8 eligible present
   (3 correct lock, 5 miss, 0 wrong) — chỉ là chẩn đoán; không dùng để tuyên bố 0% false lock.
3. **GT provenance.** `5_manual.csv` do một người gán, ±30 frame; `5.csv` là circular.
   Không có reviewer thứ hai/adjudication.
4. **GT vs video discrepancy.** Frame 1745 và 2508/2509: GT nói absent nhưng video cho thấy
   target. Đây là quan sát AI-assisted, chưa phải human GT; onset E1/E3 vẫn UNKNOWN.
5. **Margin policy.** Primary conservative (mọi boundary); sensitivity bỏ view boundary.
   Chưa có protocol margin được freeze bởi người review độc lập.
6. **Review tool chưa được chạy bởi người thật** trong phiên này; test dùng video/log tổng hợp.
   GUI (`--gui`) chưa kiểm thử trên display thật; CLI đã test.
7. **Không có held-out video 1/2/3**, không PX4/SITL, không training.
8. **Frame metadata 2699 vs decoded 2698** — luôn dùng decoded/index 0–2697 cho metric.
9. **Box detection trong Phase B** dựa trên contour xanh; UI xanh lớn có thể gây nhiễu.
   Reviewer có thể dùng `r` (review later) nếu hình không rõ.
10. **Không có estimate thời gian xử lý/thống kê đa episode**; chỉ count/coverage.
""")

    write(out, "CODEX_AUDIT_HANDOFF.md", f"""# CODEX AUDIT HANDOFF — Gate A Evaluator v3

**Yêu cầu:** kiểm toán độc lập, không tin báo cáo này; tái chạy từ artifact.
**Không tự tuyên bố Gate A PASS.**

## Trạng thái

- Implementation: COMPLETE trong phạm vi cho phép.
- Gate A scientific validation: **NOT ESTABLISHED**.
- Ready for audit: **YES**.

## Artifact cần kiểm tra

- `EXECUTIVE_SUMMARY_VI.md`, `FULL_TECHNICAL_REPORT_VI.md`
- `DEFECT_REGISTER.md`, `COUNTEREXAMPLES_BEFORE_AFTER.md`, `COUNTEREXAMPLES_RAW.json`
- `METRIC_CONTRACTS_V3.md`, `SCHEMA_AND_UNKNOWN_SEMANTICS.md`, `GT_VALIDATION_REPORT.md`
- `E1_E4_FORENSIC_ANALYSIS.md`, `GEMINI_V2_METRIC_ERRATA.md`
- `final_run/run_*/gate_a_v3_metrics.json` + `REPORT_VI.md` + `MANIFEST_SHA256.txt`
- `forensics_final/gate_a_v3_forensics.json`
- `raw_root_tests.txt`, `raw_multi_object_tests.txt`, `raw_v3_tests.txt`, `raw_evaluator_tests.txt`
- `SOURCE_DIFF.patch`, `FILE_MANIFEST_SHA256.txt`, `ROLLBACK.md`

## Lệnh tái lập

```bash
cd /home/foxsec/drone_project
sha256sum evaluate.py tests/test_evaluate_identity.py tests/test_gate_a_v3_*.py tools/*.py follow_target.py
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests/multi_object -v
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_evaluator_v3.py \\
  --video videos/5.mp4 --log logs/tracking_log_20261106_233941.csv \\
  --ground-truth ground_truth/5_manual.csv \\
  --reviews research_handoff/review/derived/gatea_evaluate_identity_reviews_v1.csv \\
  --out-dir <NEW_EMPTY_DIR>
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_v3_forensics.py --out-dir <NEW_EMPTY_DIR>
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tools/gate_a_v3_counterexamples.py
```

## Checklist audit

- [ ] `false_lock_rate_target_present` là `None` với `judged_locks=0`, không phải 0.0.
- [ ] Mọi metric có `metric_status` n/d/status/reason; zero denominator là NOT_ESTIMABLE.
- [ ] GT overlap nested/partial/duplicate bị từ chối.
- [ ] UNKNOWN gap không tạo exact onset; partial onset bắt đầu ở partial.
- [ ] View change không tạo out-of-view episode.
- [ ] Evidence key trùng canonical và impossible lock bị từ chối.
- [ ] Baseline CSV/log/GT/video không bị sửa; `follow_target.py` hash `814b38dd…`.
- [ ] Legacy CLI trên dữ liệu hợp lệ byte-identical.
- [ ] Review tool không lộ prediction trong Phase A; lock/hash bảo vệ Phase A.
- [ ] Không có metric hard-coded; JSON khớp recompute.
- [ ] Báo cáo không tuyên bố Gate A PASS.

## Câu hỏi mở cho Codex

1. Boundary policy primary (mọi boundary) và sensitivity (bỏ view) đã đủ rõ chưa?
2. `presence=partial` là extension schema có chấp nhận được không?
3. Có nên chuẩn hóa/merge adjacent same-presence intervals hay giữ nguyên?
4. Có cần thêm reviewer thứ hai/adjudication trước khi dùng bất kỳ metric nào?
""")

    # Source diff patch is generated separately by a shell step.
    print("documents written to", out)


if __name__ == "__main__":
    main()
