#!/usr/bin/env python3
"""
/home/foxsec/drone_project/evaluate.py

Tracking evaluation & Spot-Check / Segment verification tool.

Key Metrics & Rules:
1. FALSE LOCK = TARGET label (state == 'TRACKING') while target_visible = 0 (ABSENT),
   EXCLUDING:
   (a) frames within 30 frames of any GT boundary (boundary uncertainty),
   (b) frames with target_visible = 2 (partial visibility), and
   (c) frames with target_visible = -1 (unannotated GT; missing annotation is UNKNOWN, never absent).
   These excluded frames are reported separately as 'uncertain frames' / 'unannotated frames'.
2. Coverage is computed over target_visible = 1 frames outside the 30-frame boundary margin.
   (Raw total visible coverage is also reported for reference).
3. Reacquisition delay is computed in frames (with seconds equivalent).
   Delays below 30 frames are noted as within GT boundary uncertainty (+-30 frames).
4. Out-of-view reacquisition requires a ground-truth-supported target absence (target_visible = 0)
   followed by reappearance (target_visible = 1). Viewpoint changes (front -> back) and occlusion
   recoveries (partial -> visible) are tracked separately and never create false reacquisition episodes.
5. WARNING: Quantitative absence metrics cannot detect a green box on the WRONG person while
   the target is visible; visual checks (--segments) or identity reviews (--identity-reviews) are required.
6. Segment export (--segments): exports first, middle, last frame for all tracking segments.
7. Identity-aware metrics (--identity-reviews): additive Section 5 metrics computed ONLY from
   converted human review rows (frame,state,presence,drawn,identity). Tracker output is context,
   never ground truth. Uncertain rows are excluded from every denominator. Without this flag
   behaviour is unchanged.
"""

import os
import csv
import argparse
import numbers
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

GT_UNKNOWN = -1
GT_ABSENT = 0
GT_VISIBLE = 1
GT_PARTIAL = 2

VALID_GT_VISIBILITIES = frozenset({GT_UNKNOWN, GT_ABSENT, GT_VISIBLE, GT_PARTIAL})

IDENTITY_REVIEW_COLUMNS = ("frame", "state", "presence", "drawn", "identity")
_IDENTITY_BUCKETS = (
    "correct_lock", "correct_output_not_lock",
    "wrong_lock_present", "wrong_lock_absent",
    "wrong_output_not_lock_present", "wrong_output_not_lock_absent",
    "miss_present", "correct_rejection_absent", "unresolved",
)
_VALID_EVIDENCE_BUCKETS = frozenset(_IDENTITY_BUCKETS)
_WRONG_LOCK_BUCKETS = frozenset({"wrong_lock_present", "wrong_lock_absent"})

_ALLOWED_LOG_STATES = frozenset({"IDLE", "INITIALIZING", "TRACKING", "LOST", "GIVE_UP"})
_ALLOWED_VALID_VALUES = frozenset({"0", "1"})
_ALLOWED_PRESENCE = frozenset({"present", "absent", "uncertain", "partial"})
_ALLOWED_IDENTITY = frozenset({"correct", "wrong", "uncertain", "none"})


# ---------------------------------------------------------------------------
# Strict Validation Helpers (Shared across legacy & identity paths)
# ---------------------------------------------------------------------------

def _strict_frame(value: Any, context: str) -> int:
    """Single frame contract shared across log, GT, review and CSV rows.

    Accepts only non-negative integers (bool excluded) and plain decimal digit strings.
    Floats -- including integral-looking 3.0 -- booleans, None, '+1', '1_0', '1e2',
    and '1.9' are rejected. A malformed frame is an unusable row, never silently coerced.
    """
    if isinstance(value, bool):
        raise ValueError(f"{context}: frame must be an integer, got boolean {value!r}")
    if isinstance(value, numbers.Integral):
        frame = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        frame = int(value.strip())
    else:
        raise ValueError(f"{context}: frame must be a non-negative integer, got {value!r}")
    if frame < 0:
        raise ValueError(f"{context}: frame must be non-negative, got {frame}")
    return frame


def _strict_target_visible(value: Any, context: str) -> int:
    """Validate and parse ground truth visibility value (-1, 0, 1, 2).

    Rejects booleans, fractional floats, NaN, Inf, and unrecognized strings.
    """
    if isinstance(value, bool):
        raise ValueError(f"{context}: target_visible must be an integer in {{-1, 0, 1, 2}}, got boolean {value!r}")
    if isinstance(value, numbers.Integral):
        tv = int(value)
    elif isinstance(value, str):
        s = value.strip().lower()
        if s in ("unknown", "not_annotated", "none_annotated", "-1"):
            return GT_UNKNOWN
        if re.fullmatch(r"-?[0-9]+", s):
            tv = int(s)
        else:
            raise ValueError(f"{context}: target_visible must be in {{-1, 0, 1, 2}}, got {value!r}")
    else:
        raise ValueError(f"{context}: target_visible must be in {{-1, 0, 1, 2}}, got {value!r}")
    if tv not in VALID_GT_VISIBILITIES:
        raise ValueError(f"{context}: target_visible must be in {{-1, 0, 1, 2}}, got {tv}")
    return tv


def _normalise_valid(value: Any, frame: int) -> str:
    """Strict valid parsing: only 0/1 are meaningful.

    A non-binary value means the log cannot tell us whether a box was drawn, so
    the frame's lock capability would be UNKNOWN. Treating it as 'no box' would
    silently turn an unknown into evidence of absence, so it is rejected instead.
    """
    if isinstance(value, bool):
        raise ValueError(f"log frame {frame}: valid must be '0' or '1', got boolean {value!r}")
    if isinstance(value, int):
        if value in (0, 1):
            return str(value)
        raise ValueError(f"log frame {frame}: valid must be '0' or '1', got {value!r}")
    if isinstance(value, str) and value.strip() in _ALLOWED_VALID_VALUES:
        return value.strip()
    raise ValueError(f"log frame {frame}: valid must be '0' or '1', got {value!r}")


GT_BOUNDARY_PRESENCE = "presence"
GT_BOUNDARY_PARTIAL = "partial"
GT_BOUNDARY_VIEW = "view"
GT_BOUNDARY_UNKNOWN = "unknown"
GT_BOUNDARY_CONTINUATION = "continuation"

EVENT_TRUE_REAPPEARANCE = "TRUE_OUT_OF_VIEW_REAPPEARANCE"
EVENT_OCCLUSION_RECOVERY = "OCCLUSION_RECOVERY"
EVENT_IN_SCENE_RECOVERY = "IN_SCENE_TRACKING_RECOVERY"
EVENT_VIEWPOINT_CHANGE = "VIEWPOINT_CHANGE"
EVENT_TRACKER_ID_CHANGE = "TRACKER_ID_CHANGE"
EVENT_UNKNOWN_ONSET = "UNKNOWN_ONSET_REAPPEARANCE"
EVENT_UNCLASSIFIABLE = "UNCLASSIFIABLE_EVENT"


def _gt_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"GT interval {idx}: expected a mapping, got {type(item).__name__}")
    for key in ("start_frame", "end_frame", "target_visible"):
        if key not in item:
            raise ValueError(f"GT interval {idx}: missing required field {key!r}")
    sf = _strict_frame(item["start_frame"], f"GT interval {idx} start_frame")
    ef = _strict_frame(item["end_frame"], f"GT interval {idx} end_frame")
    if sf > ef:
        raise ValueError(f"GT interval {idx}: start_frame ({sf}) > end_frame ({ef})")
    tv = _strict_target_visible(item["target_visible"], f"GT interval {idx} target_visible")
    view = str(item.get("view", "none")).strip().lower() or "none"
    if tv == GT_ABSENT and view not in ("none", "unknown", ""):
        raise ValueError(f"GT interval {idx}: absent interval carries view {view!r} (contradictory semantics)")
    tid = item.get("target_id")
    if tid is not None and str(tid).strip():
        tid = _strict_frame(tid, f"GT interval {idx} target_id")
    else:
        tid = None
    return {"start_frame": sf, "end_frame": ef, "target_visible": tv,
            "view": view, "target_id": tid, "original_index": idx}


def validate_ground_truth_intervals(gt_intervals: List[Dict[str, Any]],
                                    max_frame: Optional[int] = None) -> List[Dict[str, Any]]:
    """Strict, whole-set GT validation.

    - Every field is validated (missing, malformed, negative, reversed).
    - Intervals are sorted and checked as a whole: ANY overlap (nested, partial,
      duplicate) is rejected. Coverage gaps are allowed and mean GT_UNKNOWN.
    - Adjacent intervals with identical (visibility, view, target_id) are merged;
      this is lossless normalisation, not data deletion.
    - `max_frame` (optional) rejects intervals beyond the decoded video range.
    """
    if gt_intervals is None:
        raise ValueError("gt_intervals must be a list, got None")
    validated = [_gt_item(item, idx) for idx, item in enumerate(gt_intervals)]
    validated.sort(key=lambda x: (x["start_frame"], x["end_frame"], x["original_index"]))

    for i in range(1, len(validated)):
        prev, curr = validated[i - 1], validated[i]
        if curr["start_frame"] <= prev["end_frame"]:
            kind = "duplicate" if (prev["start_frame"] == curr["start_frame"]
                                   and prev["end_frame"] == curr["end_frame"]) else "overlapping"
            raise ValueError(
                f"{kind} GT intervals: [{prev['start_frame']}..{prev['end_frame']}] "
                f"(vis={prev['target_visible']}, view={prev['view']}) and "
                f"[{curr['start_frame']}..{curr['end_frame']}] "
                f"(vis={curr['target_visible']}, view={curr['view']})"
            )

    merged: List[Dict[str, Any]] = []
    for item in validated:
        if (merged and item["start_frame"] == merged[-1]["end_frame"] + 1
                and item["target_visible"] == merged[-1]["target_visible"]
                and item["view"] == merged[-1]["view"]
                and item["target_id"] == merged[-1]["target_id"]):
            merged[-1]["end_frame"] = item["end_frame"]
        else:
            merged.append(dict(item))

    if max_frame is not None:
        max_frame = _strict_frame(max_frame, "max_frame")
        for item in merged:
            if item["end_frame"] > max_frame:
                raise ValueError(
                    f"GT interval [{item['start_frame']}..{item['end_frame']}] exceeds "
                    f"decoded video range 0..{max_frame}"
                )
    return merged


def build_gt_boundary_map(gt_intervals: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, set]]:
    """Return validated intervals plus a boundary frame -> type-set map.

    Boundary types let metrics apply a margin only where it has meaning:
    presence transitions, partial/occlusion transitions, view changes and
    unknown gaps are distinct. A view change is NOT a presence onset.
    """
    intervals = validate_ground_truth_intervals(gt_intervals)
    boundaries: Dict[int, set] = {}

    def add(frame: int, kind: str) -> None:
        boundaries.setdefault(frame, set()).add(kind)

    for i in range(1, len(intervals)):
        prev, curr = intervals[i - 1], intervals[i]
        if curr["start_frame"] > prev["end_frame"] + 1:
            add(prev["end_frame"], GT_BOUNDARY_UNKNOWN)
            add(curr["start_frame"], GT_BOUNDARY_UNKNOWN)
            continue
        if prev["target_visible"] == GT_ABSENT or curr["target_visible"] == GT_ABSENT:
            kind = GT_BOUNDARY_PRESENCE
        elif prev["target_visible"] == GT_PARTIAL or curr["target_visible"] == GT_PARTIAL:
            kind = GT_BOUNDARY_PARTIAL
        elif prev["view"] != curr["view"]:
            kind = GT_BOUNDARY_VIEW
        else:
            kind = GT_BOUNDARY_CONTINUATION
        if kind != GT_BOUNDARY_CONTINUATION:
            add(prev["end_frame"], kind)
            add(curr["start_frame"], kind)
    return intervals, boundaries


def is_near_boundary(frame: int, boundaries: Dict[int, set], margin: int,
                     types: Optional[Set[str]] = None) -> bool:
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    for b, kinds in boundaries.items():
        if types is not None and not (kinds & types):
            continue
        if abs(frame - b) <= margin:
            return True
    return False


def build_reacquisition_episodes(gt_intervals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify GT events without conflating absence, partiality and view changes.

    Contract (see METRIC_CONTRACTS_V3):
      * out_of_view episode requires a CONFIRMED ABSENT interval and a later
        present interval with NO unknown gap between them; if an unannotated
        gap or GT_UNKNOWN interval sits between absence and reappearance, the
        event is UNKNOWN_ONSET_REAPPEARANCE, never a confirmed onset.
      * a partial interval after absence starts the episode (partial onset);
        the later fully-visible interval is not a second episode.
      * view changes while continuously visible are VIEWPOINT_CHANGE.
      * partial->full while present is OCCLUSION_RECOVERY.
    """
    intervals, boundaries = build_gt_boundary_map(gt_intervals)
    confirmed, unknown_onset, occlusions, views = [], [], [], []
    initial_appearance = None
    pending_absence: Optional[Dict[str, Any]] = None
    partial_episode: Optional[Dict[str, Any]] = None
    last_visibility: Optional[int] = None
    last_view: Optional[str] = None
    target_previously_present = False
    unknown_since_present = False
    prev_end: Optional[int] = None

    def _abs_dict(ab: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if ab is None:
            return None
        return {"start_frame": ab["start_frame"], "end_frame": ab["end_frame"],
                "duration_frames": ab["end_frame"] - ab["start_frame"] + 1}

    for item in intervals:
        sf, ef = item["start_frame"], item["end_frame"]
        tv, view = item["target_visible"], item["view"]

        if prev_end is not None and sf > prev_end + 1:
            unknown_since_present = True

        if tv == GT_UNKNOWN:
            unknown_since_present = True
            last_visibility = GT_UNKNOWN
        elif tv == GT_ABSENT:
            if pending_absence is None:
                pending_absence = {"start_frame": sf, "end_frame": ef}
            else:
                pending_absence["end_frame"] = ef
            unknown_since_present = False
            last_visibility = GT_ABSENT
        elif tv == GT_PARTIAL:
            base = {"start_frame": sf, "end_frame": ef, "view": view,
                    "first_confirmed_present_frame": sf,
                    "earliest_possible_present_frame": (pending_absence["end_frame"] + 1
                                                       if pending_absence else sf),
                    "latest_confirmed_absent_frame": (pending_absence["end_frame"]
                                                      if pending_absence else None),
                    "prior_absence": _abs_dict(pending_absence)}
            if pending_absence:
                base["prior_absence_start"] = pending_absence["start_frame"]
                base["prior_absence_end"] = pending_absence["end_frame"]
                base["prior_absence_duration"] = pending_absence["end_frame"] - pending_absence["start_frame"] + 1
            if pending_absence is not None and unknown_since_present:
                ev = dict(base)
                ev.update({"event_type": "unclassifiable_unknown_gap",
                           "event_class": EVENT_UNKNOWN_ONSET,
                           "onset_visibility": "partial", "onset_status": "unknown"})
                unknown_onset.append(ev)
                pending_absence = None
                unknown_since_present = False
            elif pending_absence is not None:
                ev = dict(base)
                ev.update({"event_type": "out_of_view_reacquisition",
                           "event_class": EVENT_TRUE_REAPPEARANCE,
                           "onset_visibility": "partial", "onset_status": "exact",
                           "first_fully_visible_frame": None})
                confirmed.append(ev)
                partial_episode = ev
                pending_absence = None
            elif target_previously_present and last_visibility in (GT_VISIBLE, GT_PARTIAL):
                rec = dict(base)
                rec.update({"event_type": "occlusion_recovery",
                            "event_class": EVENT_OCCLUSION_RECOVERY,
                            "from_visibility": last_visibility})
                occlusions.append(rec)
            elif not target_previously_present:
                initial_appearance = {"start_frame": sf, "end_frame": ef, "view": view,
                                      "event_type": "initial_appearance",
                                      "event_class": EVENT_UNCLASSIFIABLE,
                                      "initial_visibility": "partial"}
            target_previously_present = True
            last_visibility = GT_PARTIAL
            last_view = view
        elif tv == GT_VISIBLE:
            base = {"start_frame": sf, "end_frame": ef, "view": view,
                    "first_confirmed_present_frame": sf,
                    "earliest_possible_present_frame": (pending_absence["end_frame"] + 1
                                                       if pending_absence else sf),
                    "latest_confirmed_absent_frame": (pending_absence["end_frame"]
                                                      if pending_absence else None),
                    "prior_absence": _abs_dict(pending_absence)}
            if pending_absence:
                base["prior_absence_start"] = pending_absence["start_frame"]
                base["prior_absence_end"] = pending_absence["end_frame"]
                base["prior_absence_duration"] = pending_absence["end_frame"] - pending_absence["start_frame"] + 1
            if pending_absence is not None and unknown_since_present:
                ev = dict(base)
                ev.update({"event_type": "unclassifiable_unknown_gap",
                           "event_class": EVENT_UNKNOWN_ONSET,
                           "onset_visibility": "full", "onset_status": "unknown"})
                unknown_onset.append(ev)
                pending_absence = None
                unknown_since_present = False
            elif pending_absence is not None:
                ev = dict(base)
                ev.update({"event_type": "out_of_view_reacquisition",
                           "event_class": EVENT_TRUE_REAPPEARANCE,
                           "onset_visibility": "full", "onset_status": "exact",
                           "first_fully_visible_frame": sf})
                confirmed.append(ev)
                pending_absence = None
            elif partial_episode is not None:
                partial_episode["first_fully_visible_frame"] = sf
                partial_episode = None
                rec = dict(base)
                rec.update({"event_type": "partial_to_full_visibility",
                            "event_class": EVENT_OCCLUSION_RECOVERY})
                occlusions.append(rec)
            elif not target_previously_present:
                initial_appearance = {"start_frame": sf, "end_frame": ef, "view": view,
                                      "event_type": "initial_appearance",
                                      "event_class": EVENT_UNCLASSIFIABLE,
                                      "initial_visibility": "full"}
            elif unknown_since_present:
                ev = dict(base)
                ev.update({"event_type": "unclassifiable_unknown_gap",
                           "event_class": EVENT_UNKNOWN_ONSET,
                           "onset_visibility": "full", "onset_status": "unknown",
                           "reason": "unknown_gap_within_visibility"})
                unknown_onset.append(ev)
                unknown_since_present = False
            elif last_visibility == GT_VISIBLE:
                if view != last_view and last_view not in (None, "none"):
                    vc = {"start_frame": sf, "end_frame": ef,
                          "event_type": "viewpoint_change",
                          "event_class": EVENT_VIEWPOINT_CHANGE,
                          "prev_view": last_view, "next_view": view}
                    views.append(vc)
            target_previously_present = True
            last_visibility = GT_VISIBLE
            last_view = view
        prev_end = ef

    return {
        "reacquisition_episodes": confirmed,
        "unknown_onset_reappearances": unknown_onset,
        "viewpoint_changes": views,
        "occlusion_recoveries": occlusions,
        "unclassifiable_reappearances": unknown_onset,
        "initial_appearance": initial_appearance,
        "gt_boundaries": {int(f): sorted(t) for f, t in sorted(boundaries.items())},
        "gt_intervals": intervals,
    }


def classify_tracker_events(log_rows: List[Dict[str, Any]],
                            gt_intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify state LOST->TRACKING recoveries and tracker-ID changes.

    This is log-level evidence, never identity ground truth. It distinguishes
    in-scene tracking recovery from a true out-of-view reappearance using the
    validated GT model, and reports unknown when GT cannot tell.
    """
    rows = validate_identity_log_rows(log_rows)
    intervals, _ = build_gt_boundary_map(gt_intervals)
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    events: List[Dict[str, Any]] = []
    lost_start: Optional[int] = None
    lost_tid: Optional[int] = None
    prior_state = None
    prior_tid = None
    for idx, row in enumerate(rows):
        frame = int(row["frame"])
        state = row["state"]
        tid = int(row["target_track_id"]) if str(row.get("target_track_id", "")).strip().isdigit() else None
        if prior_state == "TRACKING" and state == "LOST":
            lost_start, lost_tid = frame, prior_tid
        elif lost_start is not None and state == "TRACKING":
            # first valid output box at/after relock
            first_valid = None
            for r2 in rows[idx:]:
                if r2["state"] == "TRACKING" and r2["valid"] == "1":
                    first_valid = int(r2["frame"])
                    break
            window = range(lost_start, frame + 1)
            gts = [get_gt_info_at_frame(intervals, f)[0] for f in window]
            if all(g == GT_UNKNOWN for g in gts):
                event_class = EVENT_UNKNOWN_ONSET
                presence = "unknown"
            elif any(g == GT_ABSENT for g in gts):
                event_class = EVENT_TRUE_REAPPEARANCE
                presence = "absent_confirmed"
            elif any(g == GT_PARTIAL for g in gts):
                event_class = EVENT_OCCLUSION_RECOVERY
                presence = "partial"
            else:
                event_class = EVENT_IN_SCENE_RECOVERY
                presence = "present"
            events.append({
                "event_class": event_class,
                "lost_start_frame": lost_start,
                "relock_frame": frame,
                "first_valid_output_frame": first_valid,
                "lost_duration_frames": frame - lost_start,
                "prior_tracker_id": lost_tid,
                "tracker_id_at_relock": tid,
                "tracker_id_changed": (lost_tid is not None and tid is not None and lost_tid != tid),
                "gt_presence_in_lost_window": presence,
                "gt_absent_frames": sum(1 for g in gts if g == GT_ABSENT),
                "gt_present_frames": sum(1 for g in gts if g in (GT_VISIBLE, GT_PARTIAL)),
                "gt_unknown_frames": sum(1 for g in gts if g == GT_UNKNOWN),
            })
            lost_start = None
            lost_tid = None
        elif prior_state == "TRACKING" and state == "TRACKING" and prior_tid is not None and tid is not None and prior_tid != tid:
            events.append({"event_class": EVENT_TRACKER_ID_CHANGE, "frame": frame,
                           "prior_tracker_id": prior_tid, "tracker_id": tid})
        prior_state, prior_tid = state, tid
    return events


def validate_identity_log_rows(log_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate the tracker-log rows used by evaluation.

    Rejects, with an explicit error: missing columns, non-integer/negative/duplicate
    frames, states outside the tracker's own enum, and non-binary valid values.
    Returns normalised copies (the caller's data is never mutated).
    """
    seen = set()
    normalised = []
    for index, row in enumerate(log_rows):
        for column in ("frame", "state", "valid"):
            if column not in row:
                raise ValueError(f"log row {index} missing required column {column!r}")
        frame = _strict_frame(row["frame"], f"log row {index}")
        if frame in seen:
            raise ValueError(f"duplicate log frame {frame}")
        state = row["state"]
        if state not in _ALLOWED_LOG_STATES:
            raise ValueError(f"log frame {frame}: unknown state {state!r}")
        valid = _normalise_valid(row["valid"], frame)
        seen.add(frame)
        copy = dict(row)
        copy["frame"] = str(frame)
        copy["valid"] = valid
        normalised.append(copy)
    return normalised


def validate_identity_review_rows(review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate review rows and reject duplicates that disagree with each other."""
    seen = {}
    normalised = []
    for index, row in enumerate(review_rows):
        for column in ("frame", "state", "presence", "drawn", "identity"):
            if column not in row:
                raise ValueError(f"review row {index} missing required column {column!r}")
        frame = _strict_frame(row["frame"], f"review row {index}")
        presence = row["presence"]
        identity = row["identity"]
        if presence not in _ALLOWED_PRESENCE:
            raise ValueError(f"review frame {frame}: invalid presence {presence!r}")
        if identity not in _ALLOWED_IDENTITY:
            raise ValueError(f"review frame {frame}: invalid identity {identity!r}")
        if not isinstance(row["drawn"], bool):
            raise TypeError(f"review frame {frame}: drawn must be a bool, got {type(row['drawn']).__name__}")
        if row["drawn"] != (identity != "none"):
            raise ValueError(f"review frame {frame}: drawn/identity contradiction")
        if presence == "absent" and identity == "correct":
            raise ValueError(f"review frame {frame}: correct identity contradicts confirmed absence")
        record = (row["state"], presence, row["drawn"], identity)
        if frame in seen:
            if seen[frame] == record:
                raise ValueError(f"duplicate review frame {frame}")
            raise ValueError(f"conflicting review records for frame {frame}")
        seen[frame] = record
        copy = dict(row)
        copy["frame"] = frame
        normalised.append(copy)
    return normalised


# ---------------------------------------------------------------------------
# Ground Truth Loading & Querying
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """Parse and strictly validate ground truth CSV intervals."""
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    gt_intervals = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ("start_frame", "end_frame", "target_visible")
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"ground truth CSV missing required columns: {missing}")
        for line_no, row in enumerate(reader, 2):
            sf = _strict_frame(row["start_frame"], f"GT line {line_no} start_frame")
            ef = _strict_frame(row["end_frame"], f"GT line {line_no} end_frame")
            if sf > ef:
                raise ValueError(f"GT line {line_no}: start_frame ({sf}) > end_frame ({ef})")
            tv = _strict_target_visible(row["target_visible"], f"GT line {line_no} target_visible")
            tid = None
            if row.get("target_id") and row["target_id"].strip():
                tid = _strict_frame(row["target_id"], f"GT line {line_no} target_id")
            gt_intervals.append({
                "start_frame": sf,
                "end_frame": ef,
                "target_visible": tv,
                "view": str(row.get("view", "none")).strip().lower(),
                "target_id": tid
            })
    return validate_ground_truth_intervals(gt_intervals)


def get_gt_info_at_frame(gt_intervals: List[Dict[str, Any]], frame_idx: int) -> Tuple[int, str, Optional[int]]:
    """Query ground truth info at a given frame.

    Returns (target_visible, view, target_id).
    If frame_idx is not covered by any interval, returns (GT_UNKNOWN, "unknown", None).
    Missing coverage is NEVER silently converted to target absent (0).
    """
    f = _strict_frame(frame_idx, "get_gt_info_at_frame frame_idx")
    intervals = validate_ground_truth_intervals(gt_intervals)
    for item in intervals:
        if item["start_frame"] <= f <= item["end_frame"]:
            return item["target_visible"], item["view"], item.get("target_id")
    return GT_UNKNOWN, "unknown", None


# ---------------------------------------------------------------------------
# Tracking Segment Extraction
# ---------------------------------------------------------------------------

def get_tracking_segments(log_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract continuous tracking segments from log rows with strict validation."""
    if not log_rows:
        return []
    segments = []
    in_seg = False
    start_f = None
    curr_tid = None
    seen = set()

    for idx, r in enumerate(log_rows):
        if "frame" not in r or "state" not in r:
            raise ValueError(f"log row {idx} missing required column 'frame' or 'state'")
        fn = _strict_frame(r["frame"], f"log row {idx}")
        if fn in seen:
            raise ValueError(f"duplicate log frame {fn} at index {idx}")
        seen.add(fn)
        st = r["state"]
        if st not in _ALLOWED_LOG_STATES:
            raise ValueError(f"log frame {fn}: unknown state {st!r}")
        tid = r.get("target_track_id", "").strip()

        if st == "TRACKING":
            if not in_seg:
                in_seg = True
                start_f = fn
                curr_tid = tid
            else:
                if tid != curr_tid:
                    segments.append({
                        "start": start_f,
                        "end": fn - 1,
                        "length": fn - start_f,
                        "track_id": curr_tid
                    })
                    start_f = fn
                    curr_tid = tid
        else:
            if in_seg:
                segments.append({
                    "start": start_f,
                    "end": fn - 1,
                    "length": fn - start_f,
                    "track_id": curr_tid
                })
                in_seg = False
                start_f = None
                curr_tid = None

    if in_seg and start_f is not None:
        last_fn = _strict_frame(log_rows[-1]["frame"], "log last row")
        segments.append({
            "start": start_f,
            "end": last_fn,
            "length": last_fn - start_f + 1,
            "track_id": curr_tid
        })
    return segments


def export_segments_for_visual_check(csv_log_path: str,
                                    video_path: str,
                                    variant_name: str,
                                    out_dir: str = "/home/foxsec/drone_project/outputs/check_frames") -> List[Dict[str, Any]]:
    if not os.path.exists(csv_log_path):
        raise FileNotFoundError(f"CSV log file not found: {csv_log_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)

    with open(csv_log_path, 'r', encoding='utf-8') as f:
        log_rows = list(csv.DictReader(f))

    segments = get_tracking_segments(log_rows)

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 90)
    print(f"TASK 2: TRACKING SEGMENTS TABLE FOR VISUAL CHECKING (Variant: {variant_name})")
    print(f"  Log File   : {csv_log_path}")
    print(f"  Video File : {video_path} ({total_video_frames} frames)")
    print(f"  Output Dir : {out_dir}")
    print("=" * 90)
    print(f"{'Variant':<10} | {'Start':<7} | {'End':<7} | {'Length':<8} | {'Track ID':<8} | {'Sampled Frames (first, mid, last)':<35}")
    print("-" * 90)

    for seg in segments:
        s = seg["start"]
        e = seg["end"]
        length = seg["length"]
        tid = seg["track_id"]
        mid = s + length // 2

        frame_samples = [
            (s, "first"),
            (mid, "mid"),
            (e, "last")
        ]

        saved_files = []
        for fn, tag in frame_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if ret:
                fname = f"seg_{variant_name}_{s}_{tag}.jpg"
                fpath = os.path.join(out_dir, fname)
                cv2.imwrite(fpath, frame)
                saved_files.append(fname)
            else:
                saved_files.append(f"failed_f{fn}")

        samples_str = f"first:{s}, mid:{mid}, last:{e}"
        print(f"{variant_name:<10} | {s:<7d} | {e:<7d} | {length:<8d} | {tid:<8s} | {samples_str:<35}")

    print("-" * 90)
    print("Visual check instruction: Inspect the exported full-resolution images in outputs/check_frames/.")
    print("Automated evaluation does NOT judge correctness of identity; visual inspection is required.")
    print("=" * 90 + "\n")

    cap.release()
    return segments


# ---------------------------------------------------------------------------
# Step 3 Quantitative Evaluation (evaluate_log_gt)
# ---------------------------------------------------------------------------

def evaluate_log_gt(csv_log_path: str,
                    gt_path: str,
                    fps: float = 30.0,
                    margin_frames: int = 30,
                    known_target_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Evaluate log against ground truth with strict validation and semantic safety."""
    if not os.path.exists(csv_log_path):
        raise FileNotFoundError(f"CSV log not found: {csv_log_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if margin_frames < 0:
        raise ValueError(f"margin_frames must be non-negative, got {margin_frames}")

    log_rows = []
    with open(csv_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            log_rows.append(row)

    total_frames = len(log_rows)
    if total_frames == 0:
        print("[Error] CSV log is empty.")
        return {}

    # Strict upfront validation of log rows
    seen_frames = set()
    validated_log_rows = []
    for idx, r in enumerate(log_rows):
        if "frame" not in r or "state" not in r:
            raise ValueError(f"log row {idx} missing required column 'frame' or 'state'")
        fn = _strict_frame(r["frame"], f"log row {idx}")
        if fn in seen_frames:
            raise ValueError(f"duplicate log frame {fn} at index {idx}")
        seen_frames.add(fn)
        st = r["state"]
        if st not in _ALLOWED_LOG_STATES:
            raise ValueError(f"log frame {fn}: unknown state {st!r}")
        validated_log_rows.append(r)

    gt_intervals = parse_ground_truth(gt_path)

    # Collect all GT boundaries (start_frame and end_frame of every interval)
    boundaries: Set[int] = set()
    for item in gt_intervals:
        boundaries.add(item["start_frame"])
        boundaries.add(item["end_frame"])

    def is_near_boundary(frame_num: int, margin: int = margin_frames) -> bool:
        return any(abs(frame_num - b) <= margin for b in boundaries)

    views = ["front", "back", "side"]
    view_stats = {v: {
        "raw_visible_frames": 0,
        "raw_tracked_frames": 0,
        "margin_visible_frames": 0,
        "margin_tracked_frames": 0,
        "partial_frames": 0,
        "partial_tracked_frames": 0,
        "false_lock_frames": 0,
        "uncertain_tracking_frames": 0,
        "reacquisitions_attempted": 0,
        "reacquisitions_succeeded": 0,
        "reacquisition_delays_frames": [],
        "reacquisition_delays_sec": []
    } for v in views}

    tracking_rows = [r for r in validated_log_rows if r["state"] == "TRACKING"]
    total_tracked_frames = len(tracking_rows)

    false_lock_frames = 0
    false_lock_frame_indices = []

    uncertain_frames_total = 0
    uncertain_tracking_frames = 0
    uncertain_tracking_boundary_frames = 0
    uncertain_tracking_partial_frames = 0
    unannotated_frames = 0
    unannotated_tracking_frames = 0

    # Frame-by-frame evaluation
    for r in validated_log_rows:
        f = _strict_frame(r["frame"], "log row frame")
        state = r["state"]

        is_visible, view, _ = get_gt_info_at_frame(gt_intervals, f)
        near_bnd = is_near_boundary(f, margin_frames)

        # Track uncertain frame counts
        if is_visible == GT_PARTIAL or (is_visible == GT_ABSENT and near_bnd):
            uncertain_frames_total += 1
        elif is_visible == GT_UNKNOWN:
            unannotated_frames += 1

        # Visible frame counting (raw and margin-filtered)
        if is_visible == GT_VISIBLE:
            if view in view_stats:
                view_stats[view]["raw_visible_frames"] += 1
                if not near_bnd:
                    view_stats[view]["margin_visible_frames"] += 1
        elif is_visible == GT_PARTIAL:
            if view in view_stats:
                view_stats[view]["partial_frames"] += 1

        if state == "TRACKING":
            # Tracked frame counting
            if is_visible == GT_VISIBLE:
                if view in view_stats:
                    view_stats[view]["raw_tracked_frames"] += 1
                    if not near_bnd:
                        view_stats[view]["margin_tracked_frames"] += 1
            elif is_visible == GT_PARTIAL:
                if view in view_stats:
                    view_stats[view]["partial_tracked_frames"] += 1
            elif is_visible == GT_UNKNOWN:
                unannotated_tracking_frames += 1

            # FALSE LOCK logic:
            # FALSE LOCK = TARGET label while target_visible = 0 (ABSENT),
            # EXCLUDING:
            #  (a) frames within margin of any GT boundary (reported as uncertain)
            #  (b) frames with target_visible = 2 (partial, reported as uncertain)
            #  (c) frames with target_visible = -1 (unannotated GT, never false lock)
            is_false_lock = False
            if is_visible == GT_ABSENT:
                if near_bnd:
                    uncertain_tracking_frames += 1
                    uncertain_tracking_boundary_frames += 1
                    if view in view_stats:
                        view_stats[view]["uncertain_tracking_frames"] += 1
                else:
                    is_false_lock = True
            elif is_visible == GT_PARTIAL:
                uncertain_tracking_frames += 1
                uncertain_tracking_partial_frames += 1
                if view in view_stats:
                    view_stats[view]["uncertain_tracking_frames"] += 1
            elif is_visible in (GT_VISIBLE, GT_UNKNOWN):
                is_false_lock = False

            if is_false_lock:
                false_lock_frames += 1
                false_lock_frame_indices.append(f)
                if view in view_stats:
                    view_stats[view]["false_lock_frames"] += 1

    false_lock_rate = (false_lock_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

    # Target ID switches
    id_switches = 0
    prev_id = None
    for r in tracking_rows:
        tid_str = r.get("target_track_id", "").strip()
        if tid_str and tid_str.isdigit():
            curr_id = int(tid_str)
            if prev_id is not None and curr_id != prev_id:
                id_switches += 1
            prev_id = curr_id

    # Reacquisition episodes analysis using deterministic contract
    reacq_classification = build_reacquisition_episodes(gt_intervals)
    reacquisition_episodes = reacq_classification["reacquisition_episodes"]
    viewpoint_changes = reacq_classification["viewpoint_changes"]
    occlusion_recoveries = reacq_classification["occlusion_recoveries"]
    unclassifiable_reappearances = reacq_classification["unclassifiable_reappearances"]

    total_return_episodes = len(reacquisition_episodes)
    successful_reacquisitions = 0
    all_reacq_delays_frames = []
    all_reacq_delays_sec = []

    log_frame_map = {_strict_frame(r["frame"], "log_frame_map"): r for r in validated_log_rows}

    for ep in reacquisition_episodes:
        start_f = ep["start_frame"]
        end_f = ep["end_frame"]
        view = ep["view"]

        if view in view_stats:
            view_stats[view]["reacquisitions_attempted"] += 1

        reacquired = False
        reacq_frame = None

        for f in range(start_f, end_f + 1):
            if f in log_frame_map:
                r = log_frame_map[f]
                if r["state"] == "TRACKING":
                    reacquired = True
                    reacq_frame = f
                    break

        if reacquired and reacq_frame is not None:
            successful_reacquisitions += 1
            delay_frames = reacq_frame - start_f
            delay_sec = delay_frames / fps
            all_reacq_delays_frames.append(delay_frames)
            all_reacq_delays_sec.append(delay_sec)

            if view in view_stats:
                view_stats[view]["reacquisitions_succeeded"] += 1
                view_stats[view]["reacquisition_delays_frames"].append(delay_frames)
                view_stats[view]["reacquisition_delays_sec"].append(delay_sec)

    reacq_success_rate = (successful_reacquisitions / total_return_episodes * 100.0) if total_return_episodes > 0 else 0.0
    avg_reacq_frames = float(np.mean(all_reacq_delays_frames)) if all_reacq_delays_frames else None
    avg_reacq_time = float(np.mean(all_reacq_delays_sec)) if all_reacq_delays_sec else None

    # Print Report
    print("\n" + "=" * 80)
    print("STEP 3: QUANTITATIVE TRACKING EVALUATION REPORT (evaluate.py)")
    print("=" * 80)
    print(f"  CSV Log File                  : {csv_log_path}")
    print(f"  Ground Truth File             : {gt_path}")
    print(f"  Total Video Frames            : {total_frames} ({total_frames / fps:.2f} s)")
    print(f"  Total Tracked Frames          : {total_tracked_frames} ({total_tracked_frames / total_frames * 100:.1f}%)")
    print(f"  Target ID Switches            : {id_switches}")
    print(f"  Boundary Margin (uncertainty) : +-{margin_frames} frames")
    print(f"  Total Uncertain Frames in GT  : {uncertain_frames_total} (partial or boundary margin)")
    print(f"  Uncertain Frames in TRACKING  : {uncertain_tracking_frames} "
          f"(boundary margin: {uncertain_tracking_boundary_frames}, partial vis=2: {uncertain_tracking_partial_frames})")
    if unannotated_frames > 0:
        print(f"  Unannotated GT Frames         : {unannotated_frames} (tracking: {unannotated_tracking_frames})")
    print(f"  FALSE LOCK Frames             : {false_lock_frames} [state=TRACKING while target_visible=0 outside margin]")
    print(f"  FALSE LOCK RATE               : {false_lock_rate:.2f}% (Requirement: ~0%)")
    print(f"  Reacquisition Success         : {successful_reacquisitions}/{total_return_episodes} ({reacq_success_rate:.1f}%)")
    if avg_reacq_frames is not None:
        print(f"  Avg Reacquisition Delay       : {avg_reacq_frames:.1f} frames ({avg_reacq_time:.2f} s)")
    else:
        print(f"  Avg Reacquisition Delay       : N/A (no reacquisitions)")
    if viewpoint_changes or occlusion_recoveries or unclassifiable_reappearances:
        print(f"  Non-absence Transitions       : viewpoint changes: {len(viewpoint_changes)}, "
              f"occlusion recoveries: {len(occlusion_recoveries)}, unclassifiable: {len(unclassifiable_reappearances)}")
    print("=" * 80)

    # Prominent warning required by Task 1
    print("\n" + "#" * 80)
    print("WARNING: This quantitative metric only evaluates target absence periods")
    print("(state == TRACKING while target_visible == 0 outside the +-30 frame boundary margin).")
    print("It CANNOT detect if a green TARGET box is on the WRONG person while the target is visible.")
    print("Visual checks (--segments or video inspection) are required to verify identity correctness.")
    print("#" * 80 + "\n")

    # Per-view breakdown table
    print("PER-VIEW PERFORMANCE BREAKDOWN:")
    print("-" * 110)
    print(f"{'View':<8} | {'Visible':<8} | {'Tracked':<8} | {'Coverage':<9} | {'Raw Cov (all)':<14} | {'Reacq Success':<14} | {'Avg Reacq Delay':<18} | {'False Lock':<10}")
    print("-" * 110)

    for v in views:
        st = view_stats[v]
        m_vis = st["margin_visible_frames"]
        m_trk = st["margin_tracked_frames"]
        cov = (m_trk / m_vis * 100.0) if m_vis > 0 else 0.0

        r_vis = st["raw_visible_frames"]
        r_trk = st["raw_tracked_frames"]
        raw_cov_str = f"{r_trk}/{r_vis} ({(r_trk/r_vis*100):.1f}%)" if r_vis > 0 else "0/0"

        reacq_str = f"{st['reacquisitions_succeeded']}/{st['reacquisitions_attempted']}"
        if st["reacquisition_delays_frames"]:
            d_fr = np.mean(st["reacquisition_delays_frames"])
            d_sc = np.mean(st["reacquisition_delays_sec"])
            avg_del = f"{d_fr:.1f}f ({d_sc:.2f}s)"
        else:
            avg_del = "N/A"
        fl = st["false_lock_frames"]
        print(f"{v:<8} | {m_vis:<8d} | {m_trk:<8d} | {cov:<8.1f}% | {raw_cov_str:<14} | {reacq_str:<14} | {avg_del:<18} | {fl:<10d}")

    print("-" * 110)
    print("Note on Coverage & Reacquisition:")
    print("  1. 'Coverage' is evaluated on target_visible = 1 frames OUTSIDE the 30-frame boundary margin.")
    print("     'Raw Cov (all)' shows tracking over all visible frames including boundary margin.")
    print("  2. Ground truth boundaries have an uncertainty of approx. +-30 frames.")
    print("     Reacquisition delays below 30 frames are within this margin of error and not statistically meaningful.")
    print("-" * 110 + "\n")

    # Partial Visibility Breakdown
    print("PARTIAL VISIBILITY BREAKDOWN (target_visible = 2, reported separately):")
    print("-" * 65)
    print(f"{'View':<8} | {'Partial Frames':<16} | {'Tracked Partial':<16} | {'Partial Coverage':<16}")
    print("-" * 65)
    total_partial_frames = 0
    total_partial_tracked_frames = 0
    for v in views:
        st = view_stats[v]
        part = st["partial_frames"]
        part_trk = st["partial_tracked_frames"]
        part_cov = (part_trk / part * 100.0) if part > 0 else 0.0
        total_partial_frames += part
        total_partial_tracked_frames += part_trk
        print(f"{v:<8} | {part:<16d} | {part_trk:<16d} | {part_cov:<15.1f}%")
    tot_part_cov = (total_partial_tracked_frames / total_partial_frames * 100.0) if total_partial_frames > 0 else 0.0
    print(f"{'TOTAL':<8} | {total_partial_frames:<16d} | {total_partial_tracked_frames:<16d} | {tot_part_cov:<15.1f}%")
    print("-" * 65 + "\n")

    return {
        "total_frames": total_frames,
        "tracked_frames": total_tracked_frames,
        "uncertain_frames_total": uncertain_frames_total,
        "uncertain_tracking_frames": uncertain_tracking_frames,
        "unannotated_frames": unannotated_frames,
        "unannotated_tracking_frames": unannotated_tracking_frames,
        "false_lock_frames": false_lock_frames,
        "false_lock_rate": false_lock_rate,
        "id_switches": id_switches,
        "reacquisition_episodes": reacquisition_episodes,
        "viewpoint_changes": viewpoint_changes,
        "occlusion_recoveries": occlusion_recoveries,
        "unclassifiable_reappearances": unclassifiable_reappearances,
        "reacq_success_rate": reacq_success_rate,
        "avg_reacq_frames": avg_reacq_frames,
        "avg_reacq_time": avg_reacq_time,
        "view_stats": view_stats
    }


def run_spot_check(video_out_path: str, stride: int = 30) -> None:
    if not os.path.exists(video_out_path):
        raise FileNotFoundError(f"Annotated output video not found: {video_out_path}")

    cap = cv2.VideoCapture(video_out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 70)
    print("KEYPRESS SPOT-CHECK MODE:")
    print("  'y' = Green TARGET box is on the CORRECT target")
    print("  'n' = Green TARGET box is on a WRONG person (FALSE LOCK)")
    print("  's' = No green box shown (LOST / SEARCHING)")
    print("  'q' = Quit spot check")
    print(f"  Sampling: 1 frame every {stride} frames")
    print("=" * 70 + "\n")

    counts = {"y": 0, "n": 0, "s": 0}
    frame_idx = 0
    window_name = "Spot-Check Evaluation (Step 3)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            banner = np.zeros((60, frame.shape[1], 3), dtype=np.uint8)
            msg = f"Frame {frame_idx:05d}/{total_frames:05d} | Press: [y] Correct Target | [n] False Lock | [s] No Box | [q] Quit"
            cv2.putText(banner, msg, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            combined = np.vstack([banner, frame])

            cv2.imshow(window_name, combined)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    counts["y"] += 1
                    break
                elif key == ord('n'):
                    counts["n"] += 1
                    break
                elif key == ord('s'):
                    counts["s"] += 1
                    break
                elif key in (ord('q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n[Spot Check Terminated by User]")
                    _print_spot_check_summary(counts)
                    return

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    _print_spot_check_summary(counts)


def _print_spot_check_summary(counts: Dict[str, int]) -> None:
    total_samples = sum(counts.values())
    print("\n" + "=" * 70)
    print("SPOT-CHECK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total Checked Frames       : {total_samples}")
    print(f"  Correct Locks ('y')        : {counts['y']} ({counts['y']/max(1,total_samples)*100:.1f}%)")
    print(f"  False Locks ('n')          : {counts['n']} ({counts['n']/max(1,total_samples)*100:.1f}%)")
    print(f"  No Box / Searching ('s')   : {counts['s']} ({counts['s']/max(1,total_samples)*100:.1f}%)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Priority B: Identity-Aware Evaluation over Human Reviews
# ---------------------------------------------------------------------------

def _identity_screen_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "identity_screen_module",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_handoff", "identity_screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_identity_reviews(path: str) -> List[Dict[str, Any]]:
    """Load and validate converted human-review rows; contradictions are rejected, not guessed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Identity reviews file not found: {path}")
    rows, seen = [], set()
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing = [c for c in IDENTITY_REVIEW_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"identity reviews missing columns: {missing}")
        for line_no, row in enumerate(reader, 2):
            frame = _strict_frame(row["frame"], f"review CSV line {line_no}")
            if frame in seen:
                raise ValueError(f"duplicate frame {frame} at line {line_no}")
            seen.add(frame)
            state, presence, drawn, identity = row["state"], row["presence"], row["drawn"], row["identity"]
            if state not in ("IDLE", "INITIALIZING", "TRACKING", "LOST"):
                raise ValueError(f"invalid state {state!r} at frame {frame}")
            if presence not in ("present", "absent", "uncertain", "partial"):
                raise ValueError(f"invalid presence {presence!r} at frame {frame}")
            if drawn not in ("0", "1"):
                raise ValueError(f"invalid drawn {drawn!r} at frame {frame}")
            if identity not in ("correct", "wrong", "uncertain", "none"):
                raise ValueError(f"invalid identity {identity!r} at frame {frame}")
            if (drawn == "1") != (identity != "none"):
                raise ValueError(f"drawn/identity contradiction at frame {frame}")
            if presence == "absent" and identity == "correct":
                raise ValueError(f"correct identity contradicts confirmed absence at frame {frame}")
            rows.append({"frame": frame, "state": state, "presence": presence,
                         "drawn": drawn == "1", "identity": identity})
    return rows


def _classify_identity_row(row: Dict[str, Any], is_near_boundary: bool, gt_visible: int) -> str:
    """Bucket one reviewed row.

    Review presence/identity decide the bucket; GT visibility only adds uncertainty
    (partial visibility) or disagreement bookkeeping. Correct identity on the
    output box requires scene-confirmed presence; otherwise the row stays unresolved.
    """
    # "partial" means present-but-occluded: it cannot prove a clean identity lock.
    if (is_near_boundary or gt_visible == GT_PARTIAL or row["presence"] in ("uncertain", "partial")
            or row["identity"] == "uncertain"):
        return "unresolved"
    lock = row["state"] == "TRACKING" and row["drawn"]
    if row["identity"] == "correct":
        if row["presence"] != "present":
            return "unresolved"  # candidate confirmed TARGET but scene presence not confirmed
        return "correct_lock" if lock else "correct_output_not_lock"
    if row["identity"] == "wrong":
        if lock:
            return "wrong_lock_present" if row["presence"] == "present" else "wrong_lock_absent"
        return "wrong_output_not_lock_present" if row["presence"] == "present" else "wrong_output_not_lock_absent"
    # identity == "none" implies drawn == False (validated at load)
    return "miss_present" if row["presence"] == "present" else "correct_rejection_absent"


def _frame_ranges(frames: List[int]) -> List[List[int]]:
    """Compress sorted frame indices into inclusive [first, last] ranges."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def reacquisition_episode_evidence(start_f: int,
                                   end_f: int,
                                   bucket_by_frame: Dict[int, str],
                                   lock_frames: Optional[Set[int]] = None,
                                   log_frames: Optional[Set[int]] = None,
                                   onset_uncertainty_frames: int = 0) -> Dict[str, Any]:
    """Evidence-limited reacquisition evidence for one visible GT episode.

    Defensive and strictly typed:
      1. Validates all inputs: start_f, end_f, bucket_by_frame, lock_frames, log_frames.
      2. Rejects invalid evidence buckets with ValueError: malformed evidence cannot
         silently become negative evidence or resolved claims.
      3. An episode with no reviewed frame is UNKNOWN, not negative evidence:
         status='not_evaluated', could_be_no_correct_lock=True, no_wrong_lock_proven=False.
      4. A reviewed frame is evidence only when it carries a RESOLVED bucket from _VALID_EVIDENCE_BUCKETS.
    """
    start_f = _strict_frame(start_f, "reacquisition_episode start_f")
    end_f = _strict_frame(end_f, "reacquisition_episode end_f")
    if start_f > end_f:
        raise ValueError(f"start_f ({start_f}) must be <= end_f ({end_f})")
    onset_uncertainty_frames = _strict_frame(onset_uncertainty_frames, "onset_uncertainty_frames")

    if not isinstance(bucket_by_frame, dict):
        raise TypeError(f"bucket_by_frame must be a dict, got {type(bucket_by_frame).__name__}")

    # Validate all buckets against explicit allowlist and canonical frame identity.
    validated_buckets = {}
    raw_keys: Dict[int, Any] = {}
    for f_raw, bucket in bucket_by_frame.items():
        f_clean = _strict_frame(f_raw, f"bucket_by_frame frame key {f_raw}")
        if bucket not in _VALID_EVIDENCE_BUCKETS:
            raise ValueError(
                f"invalid evidence bucket {bucket!r} for frame {f_clean}; must be one of {sorted(_VALID_EVIDENCE_BUCKETS)}"
            )
        if f_clean in validated_buckets:
            raise ValueError(
                f"duplicate canonical evidence frame {f_clean}: keys {raw_keys[f_clean]!r} and {f_raw!r} "
                f"({validated_buckets[f_clean]!r} vs {bucket!r})"
            )
        validated_buckets[f_clean] = bucket
        raw_keys[f_clean] = f_raw
    bucket_by_frame = validated_buckets

    def _canonical_frame_set(values, name):
        canonical = set()
        for raw in values:
            f_clean = _strict_frame(raw, f"{name} element {raw}")
            if f_clean in canonical:
                raise ValueError(f"{name}: duplicate canonical frame {f_clean} (value {raw!r})")
            canonical.add(f_clean)
        return canonical

    if lock_frames is not None:
        lock_frames = _canonical_frame_set(lock_frames, "lock_frames")
    if log_frames is not None:
        log_frames = _canonical_frame_set(log_frames, "log_frames")

    # A lock bucket at a frame the supplied capability/log evidence says cannot
    # hold a lock is contradictory input, not something to silently "prove".
    for f_clean, bucket in bucket_by_frame.items():
        if bucket in _WRONG_LOCK_BUCKETS or bucket == "correct_lock":
            if lock_frames is not None and f_clean not in lock_frames:
                raise ValueError(
                    f"contradictory evidence at frame {f_clean}: bucket {bucket!r} "
                    f"but the frame is not lock-capable (lock_frames)"
                )
            if log_frames is not None and f_clean not in log_frames:
                raise ValueError(
                    f"contradictory evidence at frame {f_clean}: bucket {bucket!r} "
                    f"but the frame has no log row (log_frames)"
                )

    search_start = max(0, start_f - onset_uncertainty_frames)
    frames = list(range(search_start, end_f + 1))
    capable = set(frames) if lock_frames is None else {f for f in frames if f in lock_frames}
    known_log = set(frames) if log_frames is None else {f for f in frames if f in log_frames}
    reviewed = [f for f in frames if f in bucket_by_frame]
    resolved = [f for f in frames if bucket_by_frame.get(f) not in (None, "unresolved")]
    unresolved = [f for f in frames if bucket_by_frame.get(f) == "unresolved"]
    unreviewed = [f for f in frames if f not in bucket_by_frame]
    capable_unresolved = [f for f in frames if f in capable and f not in resolved]
    capable_unreviewed = [f for f in frames if f in capable and f not in bucket_by_frame]
    non_lock_unresolved = [f for f in frames if f in unresolved and f not in capable]
    log_unknown = [f for f in frames if f not in known_log]
    # A capability that cannot be checked is treated like an unresolved one.
    blocking_pool = sorted(set(capable_unresolved) | set(log_unknown))

    wrong_frames = [f for f in resolved if bucket_by_frame[f] in _WRONG_LOCK_BUCKETS]
    correct_frames = [f for f in resolved if bucket_by_frame[f] == "correct_lock"]
    first_correct = correct_frames[0] if correct_frames else None

    blocking = [f for f in blocking_pool if first_correct is None or f < first_correct]
    wrong_before = [f for f in wrong_frames if first_correct is None or f < first_correct]
    wrong_after = [f for f in wrong_frames if first_correct is not None and f > first_correct]

    wrong_first = (wrong_before[0] if wrong_before
                   else (wrong_frames[0] if first_correct is None and wrong_frames else None))
    wrong_first_proven = (wrong_first is not None
                          and not any(f < wrong_first for f in blocking_pool))
    no_wrong_proven = (not wrong_frames) and (not blocking_pool)

    latency = max(0, first_correct - start_f) if first_correct is not None else None
    lower = upper = latency
    exact = False
    could_be_no_correct = False

    if not reviewed:
        status = "not_evaluated"
        lower = upper = None
        # UNKNOWN: absence of review cannot exclude a correct lock.
        could_be_no_correct = True
    elif first_correct is not None:
        upper = latency
        if blocking:
            status = "first_correct_unproven"
            lower = max(0, blocking[0] - start_f)
            exact = False
        else:
            exact = True
            lower = latency
            status = "ok_with_prior_wrong_lock" if wrong_before else "ok"
    elif not blocking_pool:
        status = "no_correct_lock_proven"
        lower = upper = None
        could_be_no_correct = True
    else:
        status = "correct_not_observed_censored"
        lower = max(0, blocking_pool[0] - start_f)
        upper = end_f - start_f
        could_be_no_correct = True

    band = None if lower is None else [
        max(0, lower - onset_uncertainty_frames), upper + onset_uncertainty_frames]

    return {
        "start": start_f,
        "end": end_f,
        "state_latency_frames": None,
        "identity_latency_frames": latency,
        "first_correct_frame": first_correct,
        "first_correct_proven": status in ("ok", "ok_with_prior_wrong_lock"),
        "latency_lower_bound_frames": lower,
        "latency_upper_bound_frames": upper,
        "latency_exact": exact,
        "latency_exact_with_onset_uncertainty": exact and onset_uncertainty_frames == 0,
        "onset_uncertainty_frames": onset_uncertainty_frames,
        "latency_band_frames": band,
        "could_be_no_correct_lock": could_be_no_correct,
        "wrong_lock_observed": bool(wrong_frames),
        "wrong_lock_first_frame": wrong_first,
        "wrong_lock_first_frame_proven": wrong_first_proven,
        "no_wrong_lock_proven": no_wrong_proven,
        "wrong_locks_before_correct": len(wrong_before),
        "wrong_locks_in_episode": len(wrong_frames),
        "wrong_lock_after_correct_first_frame": wrong_after[0] if wrong_after else None,
        "evidence": {
            "frames_in_episode": end_f - start_f + 1,
            "search_window_start": search_start,
            "search_window_end": end_f,
            "search_window_frames": len(frames),
            "pre_onset_frames_considered": start_f - search_start,
            "lock_capable_frames": len(capable),
            "log_known_frames": len(known_log),
            "reviewed": len(reviewed),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "unreviewed": len(unreviewed),
            "blocking_ranges": _frame_ranges(blocking),
            "capable_unreviewed_ranges": _frame_ranges(capable_unreviewed),
            "non_lock_unresolved_ranges": _frame_ranges(non_lock_unresolved),
            "log_unknown_ranges": _frame_ranges(log_unknown),
            "unresolved_ranges": _frame_ranges(unresolved),
            "unreviewed_ranges": _frame_ranges(unreviewed),
        },
        "status": status,
    }


_PRESENT_ELIGIBLE_BUCKETS = frozenset({
    "correct_lock", "correct_output_not_lock", "wrong_lock_present",
    "wrong_output_not_lock_present", "miss_present",
})
_ABSENT_ELIGIBLE_BUCKETS = frozenset({
    "correct_rejection_absent", "wrong_lock_absent", "wrong_output_not_lock_absent",
})
_PRESENCE_BOUNDARY_TYPES = frozenset({
    GT_BOUNDARY_PRESENCE, GT_BOUNDARY_PARTIAL, GT_BOUNDARY_UNKNOWN,
})


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def _ratio_status(num: int, den: int, zero_reason: str) -> Dict[str, Any]:
    return {
        "numerator": num,
        "denominator": den,
        "value": _ratio(num, den),
        "status": "ESTIMABLE" if den else "NOT_ESTIMABLE",
        "reason": None if den else zero_reason,
    }


def evaluate_identity_reviews(review_rows: List[Dict[str, Any]],
                              log_rows: List[Dict[str, Any]],
                              gt_intervals: List[Dict[str, Any]],
                              fps: float = 30.0,
                              margin_frames: int = 30) -> Dict[str, Any]:
    """Identity metrics over human-reviewed records, with explicit estimability.

    DenomINATORS contain only RESOLVED human identity judgments. A reviewed row
    with presence=present but identity=uncertain (or a boundary/GT-partial row)
    is UNKNOWN and never enters a denominator. When a denominator is zero the
    value is None with status NOT_ESTIMABLE; it is never reported as 0.0.

    Boundary policy (published, not tuned to video 5):
      * primary metrics exclude rows within `margin_frames` of ANY GT boundary
        (presence, partial, view, unknown) — conservative;
      * a sensitivity block additionally ignores PURE VIEW boundaries because a
        view change does not invalidate an independent human identity judgment.
        It is labelled non-official and never used as the scientific result.
      * onset/latency uses presence/partial/unknown boundaries only.

    All metrics remain DIAGNOSTIC ONLY: there are no independent held-out labels.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if margin_frames < 0:
        raise ValueError(f"margin_frames must be non-negative, got {margin_frames}")

    log_rows = validate_identity_log_rows(log_rows)
    review_rows = validate_identity_review_rows(review_rows)
    intervals, boundaries = build_gt_boundary_map(gt_intervals)
    screen = _identity_screen_module()

    log_map = {_strict_frame(r["frame"], "log_row"): r for r in log_rows}

    def near_all(f: int) -> bool:
        return is_near_boundary(f, boundaries, margin_frames)

    def near_presence(f: int) -> bool:
        return is_near_boundary(f, boundaries, margin_frames, _PRESENCE_BOUNDARY_TYPES)

    def boundary_types_at(f: int) -> List[str]:
        found = set()
        for b, kinds in boundaries.items():
            if abs(f - b) <= margin_frames:
                found.update(kinds)
        return sorted(found)

    def classify(near_fn) -> Tuple[Dict[str, int], Dict[int, str], List[Dict[str, Any]]]:
        counts = {k: 0 for k in _IDENTITY_BUCKETS}
        by_frame: Dict[int, str] = {}
        rows: List[Dict[str, Any]] = []
        for row in review_rows:
            f = row["frame"]
            if f not in log_map:
                raise ValueError(f"review frame {f} missing from log")
            lr = log_map[f]
            if lr["state"] != row["state"]:
                raise ValueError(f"review/log state mismatch at frame {f}: {row['state']} vs {lr['state']}")
            if (lr["valid"] == "1") != row["drawn"]:
                raise ValueError(f"review/log drawn mismatch at frame {f}")
            gt_visible, gt_view, _ = get_gt_info_at_frame(intervals, f)
            bucket = _classify_identity_row(row, near_fn(f), gt_visible)
            counts[bucket] += 1
            by_frame[f] = bucket
            rows.append({"frame": f, "bucket": bucket, "state": row["state"],
                         "presence": row["presence"], "drawn": row["drawn"],
                         "identity": row["identity"], "gt_visible": gt_visible,
                         "gt_view": gt_view, "near_boundary_any": near_all(f),
                         "near_presence_boundary": near_presence(f),
                         "boundary_types": boundary_types_at(f)})
        return counts, by_frame, rows

    buckets, bucket_by_frame, details = classify(near_all)
    sensitivity_buckets, sensitivity_by_frame, _ = classify(near_presence)

    reviewed_present_frames = sum(1 for r in review_rows if r["presence"] == "present")
    reviewed_absent_frames = sum(1 for r in review_rows if r["presence"] == "absent")
    reviewed_partial_frames = sum(1 for r in review_rows if r["presence"] == "partial")

    eligible_present_frames = sum(buckets[b] for b in _PRESENT_ELIGIBLE_BUCKETS)
    eligible_absent_frames = sum(buckets[b] for b in _ABSENT_ELIGIBLE_BUCKETS)
    sensitivity_eligible_present = sum(sensitivity_buckets[b] for b in _PRESENT_ELIGIBLE_BUCKETS)
    sensitivity_eligible_absent = sum(sensitivity_buckets[b] for b in _ABSENT_ELIGIBLE_BUCKETS)

    exclusion_counts: Counter = Counter()
    for row in review_rows:
        f = row["frame"]
        if bucket_by_frame[f] != "unresolved":
            continue
        reasons = []
        for kind in boundary_types_at(f):
            reasons.append(f"boundary_{kind}")
        if row["presence"] == "uncertain":
            reasons.append("presence_uncertain")
        if row["identity"] == "uncertain":
            reasons.append("identity_uncertain")
        gt_visible = get_gt_info_at_frame(intervals, f)[0]
        if gt_visible == GT_PARTIAL:
            reasons.append("gt_partial")
        if not reasons:
            reasons.append("unresolved_other")
        exclusion_counts.update(reasons)

    # Metrics use only resolved buckets. Zero denominators are UNKNOWN, never 0.0.
    def _metrics(counts, eligible_present, eligible_absent):
        correct_lock = counts["correct_lock"]
        wrong_present = counts["wrong_lock_present"]
        wrong_absent = counts["wrong_lock_absent"]
        miss_present = counts["miss_present"]
        judged = correct_lock + wrong_present + wrong_absent
        wrong_total = wrong_present + wrong_absent
        correct_outputs = correct_lock + counts["correct_output_not_lock"]
        return {
            "judged_locks": judged,
            "identity_precision": _ratio(correct_lock, judged),
            "lock_recall": _ratio(correct_lock, correct_lock + wrong_present + miss_present),
            "present_target_lock_recall": _ratio(correct_lock, eligible_present),
            "present_target_output_coverage": _ratio(correct_outputs, eligible_present),
            "false_lock_rate": _ratio(wrong_total, judged),
            "false_lock_rate_target_present": _ratio(wrong_present, eligible_present),
            "false_lock_rate_target_absent": _ratio(wrong_absent, eligible_absent),
            "wrong_lock_total": wrong_total,
            "identity_coverage": _ratio(correct_outputs, eligible_present),
            "correct_output_total": correct_outputs,
            "target_miss_total": miss_present + wrong_present + counts["wrong_output_not_lock_present"],
            "eligible_present_frames": eligible_present,
            "eligible_absent_frames": eligible_absent,
            "counts": {
                "correct_lock": correct_lock, "wrong_lock_present": wrong_present,
                "wrong_lock_absent": wrong_absent, "miss_present": miss_present,
                "correct_output_not_lock": counts["correct_output_not_lock"],
                "wrong_output_not_lock_present": counts["wrong_output_not_lock_present"],
                "wrong_output_not_lock_absent": counts["wrong_output_not_lock_absent"],
                "correct_rejection_absent": counts["correct_rejection_absent"],
            },
        }

    primary = _metrics(buckets, eligible_present_frames, eligible_absent_frames)
    sensitivity = _metrics(sensitivity_buckets, sensitivity_eligible_present, sensitivity_eligible_absent)

    metric_status = {
        "identity_precision": _ratio_status(primary["counts"]["correct_lock"], primary["judged_locks"],
                                            "NO_ELIGIBLE_IDENTITY_JUDGMENTS"),
        "false_lock_rate": _ratio_status(primary["wrong_lock_total"], primary["judged_locks"],
                                        "NO_ELIGIBLE_IDENTITY_JUDGMENTS"),
        "false_lock_rate_target_present": _ratio_status(primary["counts"]["wrong_lock_present"],
                                                       eligible_present_frames,
                                                       "NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS"),
        "false_lock_rate_target_absent": _ratio_status(primary["counts"]["wrong_lock_absent"],
                                                      eligible_absent_frames,
                                                      "NO_ELIGIBLE_ABSENT_IDENTITY_JUDGMENTS"),
        "present_target_lock_recall": _ratio_status(primary["counts"]["correct_lock"],
                                                   eligible_present_frames,
                                                   "NO_ELIGIBLE_PRESENT_FRAMES"),
        "present_target_output_coverage": _ratio_status(primary["counts"]["correct_output_not_lock"]
                                                       + primary["counts"]["correct_lock"],
                                                       eligible_present_frames,
                                                       "NO_ELIGIBLE_PRESENT_FRAMES"),
        "lock_recall": _ratio_status(primary["counts"]["correct_lock"],
                                     primary["counts"]["correct_lock"]
                                     + primary["counts"]["wrong_lock_present"]
                                     + primary["counts"]["miss_present"],
                                     "NO_ELIGIBLE_LOCK_OR_MISS_JUDGMENTS"),
    }

    lock_frames = {_strict_frame(r["frame"], "lock_frame") for r in log_rows
                   if r["state"] == "TRACKING" and r.get("valid") == "1"}
    log_frames = {_strict_frame(r["frame"], "log_frame") for r in log_rows}

    reacq_classification = build_reacquisition_episodes(intervals)
    episodes_gt = reacq_classification["reacquisition_episodes"]
    episodes = []
    for interval in episodes_gt:
        start_f, end_f = interval["start_frame"], interval["end_frame"]
        episode = reacquisition_episode_evidence(start_f, end_f, bucket_by_frame, lock_frames,
                                                 log_frames, margin_frames)
        state_relock = None
        first_valid = None
        for f in range(start_f, end_f + 1):
            if f in log_map and log_map[f]["state"] == "TRACKING":
                if state_relock is None:
                    state_relock = f
                if first_valid is None and log_map[f]["valid"] == "1":
                    first_valid = f
        episode["state_relock_frame"] = state_relock
        episode["state_latency_frames"] = (state_relock - start_f) if state_relock is not None else None
        episode["first_valid_output_frame"] = first_valid
        episode["output_latency_frames"] = (first_valid - start_f) if first_valid is not None else None
        episode["first_verified_correct_frame"] = episode["first_correct_frame"]
        for key in ("event_class", "onset_visibility", "first_confirmed_present_frame",
                    "first_fully_visible_frame", "earliest_possible_present_frame",
                    "latest_confirmed_absent_frame", "prior_absence",
                    "prior_absence_start", "prior_absence_end", "prior_absence_duration"):
            if key in interval:
                episode[key] = interval[key]
        episodes.append(episode)

    # State-proxy absence lock count; view boundaries are irrelevant here.
    state_proxy_false_locks = sum(
        1 for r in log_rows
        if r["state"] == "TRACKING"
        and get_gt_info_at_frame(intervals, _strict_frame(r["frame"], "log_row"))[0] == GT_ABSENT
        and not near_presence(_strict_frame(r["frame"], "log_row")))

    proven_latency_episodes = sum(1 for e in episodes if e["first_correct_proven"])
    unproven_latency_episodes = sum(1 for e in episodes if e["status"] == "first_correct_unproven")
    censored_episodes = sum(1 for e in episodes if e["status"] == "correct_not_observed_censored")
    no_correct_lock_episodes = sum(1 for e in episodes if e["status"] == "no_correct_lock_proven")
    not_evaluated_episodes = sum(1 for e in episodes if e["status"] == "not_evaluated")
    episodes_with_prior_wrong_lock = sum(1 for e in episodes if e["wrong_locks_before_correct"] > 0)

    disagreements = sum(
        1 for row in review_rows
        if not near_presence(row["frame"])
        and get_gt_info_at_frame(intervals, row["frame"])[0] in (GT_ABSENT, GT_VISIBLE)
        and ((get_gt_info_at_frame(intervals, row["frame"])[0] == GT_VISIBLE and row["presence"] == "absent")
             or (get_gt_info_at_frame(intervals, row["frame"])[0] == GT_ABSENT and row["presence"] == "present")))

    # identity_screen predates the explicit "partial" presence value; for the
    # cross-check only, partial is treated as uncertain (it is unresolved in the
    # primary metrics too). Raw review rows are never mutated.
    str_rows = [{"frame": r["frame"],
                 "presence": "uncertain" if r["presence"] == "partial" else r["presence"],
                 "drawn": "1" if r["drawn"] else "0", "identity": r["identity"]} for r in review_rows]
    base = screen.reviewed_identity_metrics(str_rows)

    view_only_excluded = sorted(
        row["frame"] for row in review_rows
        if near_all(row["frame"]) and not near_presence(row["frame"]))

    return {
        "total_reviewed": len(review_rows),
        "buckets": dict(buckets),
        "judged_locks": primary["judged_locks"],
        "identity_precision": primary["identity_precision"],
        "lock_recall": primary["lock_recall"],
        "present_target_lock_recall": primary["present_target_lock_recall"],
        "false_lock_rate": primary["false_lock_rate"],
        "false_lock_rate_target_absent": primary["false_lock_rate_target_absent"],
        "false_lock_rate_target_present": primary["false_lock_rate_target_present"],
        "wrong_lock_total": primary["wrong_lock_total"],
        "identity_coverage": primary["identity_coverage"],
        "present_target_output_coverage": primary["present_target_output_coverage"],
        "correct_output_total": primary["correct_output_total"],
        "target_miss_total": primary["target_miss_total"],
        "gt_review_disagreements": disagreements,
        "reviewed_present_frames": reviewed_present_frames,
        "reviewed_absent_frames": reviewed_absent_frames,
        "reviewed_partial_frames": reviewed_partial_frames,
        "eligible_present_frames": eligible_present_frames,
        "eligible_absent_frames": eligible_absent_frames,
        "eligible_counts": {
            "present_identity_judgments": eligible_present_frames,
            "absent_identity_judgments": eligible_absent_frames,
            "judged_locks": primary["judged_locks"],
        },
        "metric_status": metric_status,
        "overall_status": "ESTIMABLE_DIAGNOSTIC_ONLY" if primary["judged_locks"] else "NOT_ESTIMABLE",
        "overall_reason": None if primary["judged_locks"] else "NO_ELIGIBLE_IDENTITY_JUDGMENTS",
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        "boundary_policy": {
            "primary": "exclude rows within margin of ANY GT boundary (presence/partial/view/unknown)",
            "sensitivity": "ignore PURE VIEW boundaries (presence/partial/unknown only); non-official",
            "latency": "presence/partial/unknown boundaries",
            "margin_frames": margin_frames,
            "boundary_types": {str(f): sorted(t) for f, t in sorted(boundaries.items())},
        },
        "sensitivity_view_boundary_ignored": {
            "official": False,
            "policy": "view boundaries ignored for identity eligibility",
            "buckets": dict(sensitivity_buckets),
            "eligible_present_frames": sensitivity_eligible_present,
            "eligible_absent_frames": sensitivity_eligible_absent,
            "identity_precision": sensitivity["identity_precision"],
            "false_lock_rate": sensitivity["false_lock_rate"],
            "false_lock_rate_target_present": sensitivity["false_lock_rate_target_present"],
            "false_lock_rate_target_absent": sensitivity["false_lock_rate_target_absent"],
            "present_target_lock_recall": sensitivity["present_target_lock_recall"],
            "present_target_output_coverage": sensitivity["present_target_output_coverage"],
        },
        "view_boundary_only_excluded_frames": view_only_excluded,
        "reacquisition_episodes": episodes,
        "unknown_onset_episodes": reacq_classification["unknown_onset_reappearances"],
        "viewpoint_changes": reacq_classification["viewpoint_changes"],
        "occlusion_recoveries": reacq_classification["occlusion_recoveries"],
        "unclassifiable_episodes": reacq_classification["unclassifiable_reappearances"],
        "proven_latency_episodes": proven_latency_episodes,
        "unproven_latency_episodes": unproven_latency_episodes,
        "censored_latency_episodes": censored_episodes,
        "no_correct_lock_episodes": no_correct_lock_episodes,
        "episodes_with_prior_wrong_lock": episodes_with_prior_wrong_lock,
        "unresolved_latency_episodes": unproven_latency_episodes + censored_episodes,
        "not_evaluated_episodes": not_evaluated_episodes,
        "state_proxy_false_lock_frames": state_proxy_false_locks,
        "identity_screen_counts": base["counts"],
        "wrong_fraction_among_reviewed_outputs": base["wrong_fraction_among_reviewed_outputs"],
        "per_frame": details,
        "warning": ("DIAGNOSTIC ONLY — NOT SCIENTIFIC VALIDATION. Metrics use resolved human "
                    "identity judgments only; unresolved/unreviewed rows never enter a denominator. "
                    "A zero denominator yields None + NOT_ESTIMABLE, never 0.0. No independent "
                    "held-out labels exist."),
    }


def print_identity_report(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("IDENTITY-AWARE EVALUATION (human-review driven; evaluate.py --identity-reviews)")
    print("=" * 80)
    b = res["buckets"]

    def _fmt(name: str) -> str:
        st = res.get("metric_status", {}).get(name)
        if not st:
            return str(res.get(name))
        if st["status"] == "ESTIMABLE":
            return f"{st['value']:.4f}  (n={st['numerator']}, d={st['denominator']})"
        return f"NOT_ESTIMABLE  (reason={st['reason']}, n={st['numerator']}, d={st['denominator']})"

    print(f"  Reviewed frames (raw)          : {res['total_reviewed']}")
    print(f"  Eligible present judgments     : {res['eligible_present_frames']}")
    print(f"  Eligible absent judgments      : {res['eligible_absent_frames']}")
    print(f"  Judged locks (correct+wrong)   : {res['judged_locks']}")
    print("  Buckets:")
    for k in _IDENTITY_BUCKETS:
        print(f"    {k:<28}: {b[k]}")
    print(f"  Identity precision             : {_fmt('identity_precision')}")
    print(f"  Lock recall (conditional)      : {_fmt('lock_recall')}")
    print(f"  Lock recall (present-certain)  : {_fmt('present_target_lock_recall')}")
    print(f"  Output coverage (present)      : {_fmt('present_target_output_coverage')}")
    print(f"  False-lock rate (all locks)    : {_fmt('false_lock_rate')}")
    print(f"  False-lock rate target present : {_fmt('false_lock_rate_target_present')}")
    print(f"  False-lock rate target absent  : {_fmt('false_lock_rate_target_absent')}")
    print(f"  Overall estimability           : {res['overall_status']}"
          + (f" ({res['overall_reason']})" if res.get("overall_reason") else ""))
    print(f"  Exclusion reasons              : {res['exclusion_reasons']}")
    sens = res.get("sensitivity_view_boundary_ignored", {})
    if sens:
        print(f"  SENSITIVITY (view boundaries ignored; NON-OFFICIAL): "
              f"eligible_present={sens['eligible_present_frames']}, "
              f"false_lock_present={sens['false_lock_rate_target_present']}, "
              f"precision={sens['identity_precision']}, recall={sens['present_target_lock_recall']}")
    print(f"  Target miss total              : {res['target_miss_total']} (no-output {b['miss_present']} + wrong-present locks {b['wrong_lock_present']} + wrong non-lock outputs {b['wrong_output_not_lock_present']})")
    print(f"  GT vs review disagreements     : {res['gt_review_disagreements']} (margin-clean; GT has +-30 frame uncertainty)")
    print(f"  State-proxy false locks (old)  : {res['state_proxy_false_lock_frames']} (TRACKING while GT absent, outside presence margin — comparison only)")
    print("  Reacquisition episodes (state latency vs identity latency, evidence-limited):")
    for e in res["reacquisition_episodes"]:
        if e["latency_exact"]:
            lat = str(e["identity_latency_frames"])
        elif e["latency_lower_bound_frames"] is None:
            lat = "None"
        else:
            lat = f"[{e['latency_lower_bound_frames']}..{e['latency_upper_bound_frames']}]"
        print(f"    {e['start']}-{e['end']}: state={e['state_latency_frames']} identity={lat} "
              f"wrong_before={e['wrong_locks_before_correct']} wrong_first={e['wrong_lock_first_frame']} "
              f"status={e['status']} could_be_no_correct={e['could_be_no_correct_lock']} "
              f"resolved={e['evidence']['resolved']}/{e['evidence']['frames_in_episode']}")
    print(f"  Proven first-correct episodes: {res['proven_latency_episodes']}; "
          f"unproven (coverage gap): {res['unproven_latency_episodes']}; "
          f"censored (no correct observed): {res['censored_latency_episodes']}; "
          f"no correct lock proven: {res['no_correct_lock_episodes']}; "
          f"not evaluated: {res['not_evaluated_episodes']}")
    print(f"  Episodes with a reviewed wrong lock before a correct lock: {res['episodes_with_prior_wrong_lock']}")
    if res.get("viewpoint_changes"):
        print(f"  Viewpoint changes (not episodes): {len(res['viewpoint_changes'])}")
    if res.get("occlusion_recoveries"):
        print(f"  Occlusion recoveries (not out-of-view): {len(res['occlusion_recoveries'])}")
    if res.get("unknown_onset_episodes"):
        print(f"  UNKNOWN-onset reappearances (excluded from exact latency): {len(res['unknown_onset_episodes'])}")
    print(f"  Margin policy                  : {res['boundary_policy']['primary']}")
    print(f"  identity_screen base counts   : {res['identity_screen_counts']}")
    print("=" * 80)
    print("WARNING: " + res["warning"])
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point (Validate-Then-Evaluate Architecture)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking performance (PART C/Task 1).")
    parser.add_argument("--log", type=str, help="Path to CSV tracking log")
    parser.add_argument("--gt", type=str, default="/home/foxsec/drone_project/ground_truth/5_manual.csv", help="Path to ground truth CSV")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    parser.add_argument("--margin", type=int, default=30, help="GT boundary margin uncertainty in frames (default: 30)")
    parser.add_argument("--target-ids", type=int, nargs="*", default=None, help="List of target track IDs across video")
    parser.add_argument("--spot-check", type=str, help="Path to annotated video for keypress spot-checking")
    parser.add_argument("--stride", type=int, default=30, help="Spot-check frame stride (default: 30)")
    parser.add_argument("--segments", action="store_true", help="Export first, mid, last frames for all tracking segments")
    parser.add_argument("--video", type=str, default=None, help="Path to matching video for segment export")
    parser.add_argument("--variant", type=str, default=None, help="Variant name (e.g. baseline, E1, E2, E3)")
    parser.add_argument("--out-dir", type=str, default="/home/foxsec/drone_project/outputs/check_frames", help="Output directory for segments")
    parser.add_argument("--identity-reviews", type=str, default=None, help="CSV of converted human identity reviews (frame,state,presence,drawn,identity); adds Section 5 identity metrics over the reviewed subset")
    args = parser.parse_args()

    if args.spot_check:
        run_spot_check(args.spot_check, stride=args.stride)
    elif args.segments:
        if not args.log:
            print("[Error] --segments requires --log <path_to_log.csv>")
        else:
            video_map = {
                "tracking_log_20261020_132332.csv": ("baseline", "/home/foxsec/drone_project/outputs/partD_baseline.mp4"),
                "tracking_log_20261020_111522.csv": ("baseline", "/home/foxsec/drone_project/outputs/partD_baseline.mp4"),
                "tracking_log_20261020_132641.csv": ("E1", "/home/foxsec/drone_project/outputs/partD_E1.mp4"),
                "tracking_log_20261020_133004.csv": ("E2", "/home/foxsec/drone_project/outputs/partD_E2.mp4"),
                "tracking_log_20261020_133334.csv": ("E3", "/home/foxsec/drone_project/outputs/partD_E3.mp4"),
            }
            log_basename = os.path.basename(args.log)
            variant = args.variant
            vpath = args.video
            if (not vpath or not variant) and log_basename in video_map:
                inferred_var, inferred_vid = video_map[log_basename]
                variant = variant or inferred_var
                vpath = vpath or inferred_vid

            if not vpath or not variant:
                print("[Error] Please specify --video and --variant for --segments")
            else:
                export_segments_for_visual_check(args.log, vpath, variant, out_dir=args.out_dir)
    elif args.log and args.gt:
        # Validate-then-evaluate pattern:
        # If --identity-reviews is provided, validate ALL inputs before printing any report!
        if args.identity_reviews:
            with open(args.log, 'r', encoding='utf-8') as f:
                log_rows_raw = list(csv.DictReader(f))
            gt_intervals_raw = parse_ground_truth(args.gt)
            reviews_raw = load_identity_reviews(args.identity_reviews)

            # Strict pre-validation of all inputs
            validated_log = validate_identity_log_rows(log_rows_raw)
            validated_reviews = validate_identity_review_rows(reviews_raw)
            validated_gt = validate_ground_truth_intervals(gt_intervals_raw)

            # Execute reports only after full validation passes
            evaluate_log_gt(args.log, args.gt, fps=args.fps, margin_frames=args.margin, known_target_ids=args.target_ids)
            print_identity_report(evaluate_identity_reviews(validated_reviews, validated_log, validated_gt,
                                                           fps=args.fps, margin_frames=args.margin))
        else:
            evaluate_log_gt(args.log, args.gt, fps=args.fps, margin_frames=args.margin, known_target_ids=args.target_ids)
    else:
        print("Usage:")
        print("  Quantitative   : python3 evaluate.py --log logs/<log>.csv [--gt ground_truth/<gt>.csv]")
        print("  Segment Export : python3 evaluate.py --segments --log logs/<log>.csv [--video outputs/<video>.mp4] [--variant <name>]")
        print("  Spot check     : python3 evaluate.py --spot-check outputs/<output>.mp4 [--stride 30]")
