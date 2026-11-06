#!/usr/bin/env python3
"""Gate A Evaluator v3 — reproducible, fail-closed evaluation runner.

Validates inputs BEFORE computing anything, records provenance hashes, and
writes a versioned JSON + Markdown + manifest. It never overwrites inputs and
never invents metrics: every number comes from evaluate.py at runtime.

Usage:
  python3 tools/gate_a_evaluator_v3.py \
    --video videos/5.mp4 \
    --log logs/tracking_log_20261106_233941.csv \
    --ground-truth ground_truth/5_manual.csv \
    --reviews research_handoff/review/derived/gatea_evaluate_identity_reviews_v1.csv \
    --out-dir research_handoff/reports/gateA_evaluator_v3_runs

  python3 tools/gate_a_evaluator_v3.py ... --validate-only
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evaluate as ev

PROTOCOL_VERSION = "gate_a_v3_20261107"
PROTECTED_ROOTS = ("ground_truth", "videos", "weights", "logs", "outputs")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def probe_video(path: str, decode: bool = True) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = None
    if decode:
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
    cap.release()
    return {"metadata_frames": meta_frames, "decoded_frames": decoded,
            "fps": fps, "width": width, "height": height}


def load_csv(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assert_output_safe(out_dir: str, input_paths) -> None:
    out_abs = os.path.abspath(out_dir)
    for path in input_paths:
        if not path:
            continue
        p = os.path.abspath(path)
        if out_abs == p:
            raise ValueError(f"output path equals input evidence path: {out_dir}")
        if p.startswith(out_abs + os.sep):
            raise ValueError(f"output directory would contain input evidence: {path}")
    rel = os.path.relpath(out_abs, PROJECT_ROOT)
    if rel.split(os.sep)[0] in PROTECTED_ROOTS and rel.split(os.sep)[0] != "logs":
        # logs/ is allowed only for new versioned runs; protected sources are not.
        raise ValueError(f"refusing to write under protected root: {rel}")


def validate_inputs(args) -> dict:
    for label, path in (("video", args.video), ("log", args.log),
                        ("ground_truth", args.ground_truth)):
        if not path:
            raise ValueError(f"--{label.replace('_', '-')} is required")
        if not os.path.exists(path):
            raise ValueError(f"{label} not found: {path}")
    if args.reviews and not os.path.exists(args.reviews):
        raise ValueError(f"reviews not found: {args.reviews}")

    video = probe_video(args.video, decode=not args.no_decode_count)
    frame_count = video["decoded_frames"] if video["decoded_frames"] is not None else video["metadata_frames"]
    if not frame_count or frame_count <= 0:
        raise ValueError("video has no decodable frames")
    max_frame = frame_count - 1

    gt_raw = ev.parse_ground_truth(args.ground_truth)
    gt = ev.validate_ground_truth_intervals(gt_raw, max_frame=max_frame)
    if gt:
        if gt[0]["start_frame"] > 0:
            pass  # leading GT_UNKNOWN is allowed and reported
        if gt[-1]["end_frame"] < max_frame:
            pass  # trailing GT_UNKNOWN is allowed and reported
    boundaries = ev.build_gt_boundary_map(gt)[1]

    log_raw = load_csv(args.log)
    log = ev.validate_identity_log_rows(log_raw)
    log_frames = []
    for row in log:
        f = int(row["frame"])
        if f > max_frame:
            raise ValueError(f"log frame {f} exceeds decoded video range 0..{max_frame}")
        log_frames.append(f)
    if log_frames != sorted(log_frames):
        raise ValueError("log rows are not in ascending frame order")
    if len(set(log_frames)) != len(log_frames):
        raise ValueError("duplicate log frames")

    reviews = []
    if args.reviews:
        reviews = ev.load_identity_reviews(args.reviews)
        reviews = ev.validate_identity_review_rows(reviews)
        log_set = set(log_frames)
        for row in reviews:
            if row["frame"] not in log_set:
                raise ValueError(f"review frame {row['frame']} has no log row")

    episodes = ev.build_reacquisition_episodes(gt)
    tracker_events = ev.classify_tracker_events(log, gt)

    return {
        "video": video, "max_frame": max_frame, "gt": gt, "boundaries": boundaries,
        "log": log, "reviews": reviews, "episodes": episodes,
        "tracker_events": tracker_events,
    }


def compute_all(args, validated) -> dict:
    log = validated["log"]
    gt = validated["gt"]
    reviews = validated["reviews"]
    video = validated["video"]

    with contextlib.redirect_stdout(io.StringIO()) as buf:
        legacy = ev.evaluate_log_gt(args.log, args.ground_truth, fps=args.fps,
                                    margin_frames=args.margin)
    identity = ev.evaluate_identity_reviews(reviews, log, gt, fps=args.fps,
                                            margin_frames=args.margin)
    state_counts = {}
    for row in log:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    tracking_frames = state_counts.get("TRACKING", 0)
    valid_tracking = sum(1 for r in log if r["state"] == "TRACKING" and r["valid"] == "1")

    return {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fps_argument": args.fps,
            "margin_frames": args.margin,
            "note": "Video 5 is exploratory; no held-out labels; NOT a scientific validation.",
        },
        "provenance": {
            "video": {"path": args.video, "sha256": sha256_file(args.video),
                      "metadata_frames": video["metadata_frames"],
                      "decoded_frames": video["decoded_frames"],
                      "fps_metadata": video["fps"]},
            "log": {"path": args.log, "sha256": sha256_file(args.log), "rows": len(log)},
            "ground_truth": {"path": args.ground_truth, "sha256": sha256_file(args.ground_truth),
                             "intervals": len(gt)},
            "reviews": ({"path": args.reviews, "sha256": sha256_file(args.reviews),
                         "rows": len(reviews)} if args.reviews else None),
            "evaluator": {"path": os.path.join(PROJECT_ROOT, "evaluate.py"),
                          "sha256": sha256_file(os.path.join(PROJECT_ROOT, "evaluate.py"))},
            "runner": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
        },
        "video": {"metadata_frames": video["metadata_frames"],
                  "decoded_frames": video["decoded_frames"],
                  "frame_index_range": [0, validated["max_frame"]],
                  "fps_metadata": video["fps"],
                  "fitz_note": "metadata count may be one larger than decoded count"},
        "ground_truth": {
            "validated_intervals": gt,
            "interval_count": len(gt),
            "boundary_types": {str(f): sorted(t) for f, t in sorted(validated["boundaries"].items())},
            "coverage_frames": sum(i["end_frame"] - i["start_frame"] + 1 for i in gt),
            "unknown_frames": validated["max_frame"] + 1 - sum(
                i["end_frame"] - i["start_frame"] + 1 for i in gt),
        },
        "tracking": {"rows": len(log), "state_counts": state_counts,
                     "tracking_frames": tracking_frames,
                     "tracking_valid_frames": valid_tracking},
        "legacy_absence_metrics": legacy,
        "identity_metrics": identity,
        "events": {
            "tracker_events": validated["tracker_events"],
            "gt_episodes": validated["episodes"]["reacquisition_episodes"],
            "gt_unknown_onset": validated["episodes"]["unknown_onset_reappearances"],
            "gt_viewpoint_changes": validated["episodes"]["viewpoint_changes"],
            "gt_occlusion_recoveries": validated["episodes"]["occlusion_recoveries"],
        },
        "scientific_validity": {
            "status": "NOT_ESTABLISHED",
            "reason": ("No independent held-out human identity labels. Identity metrics are "
                       "diagnostic only; zero-denominator metrics are NOT_ESTIMABLE."),
            "identity_overall_status": identity["overall_status"],
            "eligible_identity_judgments": identity["eligible_counts"],
        },
    }


def render_markdown(result: dict) -> str:
    im = result["identity_metrics"]
    prov = result["provenance"]
    video = result["video"]
    gt = result["ground_truth"]
    tr = result["tracking"]
    lines = [
        "# Gate A Evaluator v3 — kết quả chạy",
        "",
        f"- Protocol: `{result['protocol']['version']}`",
        f"- Video: decoded **{video['decoded_frames']}** frame, metadata {video['metadata_frames']} frame, "
        f"fps {video['fps_metadata']:.4f}, index 0–{video['frame_index_range'][1]}",
        f"- Log: {tr['rows']} dòng, TRACKING {tr['tracking_frames']}, TRACKING+valid {tr['tracking_valid_frames']}",
        f"- GT: {gt['interval_count']} interval, coverage {gt['coverage_frames']} frame, "
        f"UNKNOWN {gt['unknown_frames']} frame",
        f"- SHA256 video: `{prov['video']['sha256']}`",
        f"- SHA256 log: `{prov['log']['sha256']}`",
        f"- SHA256 GT: `{prov['ground_truth']['sha256']}`",
        "",
        "## Identity metrics (human-review driven)",
        "",
        f"- Reviewed: {im['total_reviewed']}; eligible present: {im['eligible_present_frames']}; "
        f"eligible absent: {im['eligible_absent_frames']}; judged locks: {im['judged_locks']}",
        f"- Overall: **{im['overall_status']}**"
        + (f" ({im['overall_reason']})" if im.get("overall_reason") else ""),
        "",
        "| Metric | Value | Status | n | d | Reason |",
        "|---|---:|---|---:|---:|---|",
    ]
    for name, st in im["metric_status"].items():
        value = "None" if st["value"] is None else f"{st['value']:.4f}"
        lines.append(f"| {name} | {value} | {st['status']} | {st['numerator']} | "
                     f"{st['denominator']} | {st['reason'] or ''} |")
    lines += [
        "",
        f"- Exclusion reasons: `{im['exclusion_reasons']}`",
        f"- Sensitivity (view boundaries ignored, NON-OFFICIAL): "
        f"eligible_present={im['sensitivity_view_boundary_ignored']['eligible_present_frames']}, "
        f"false_lock_present={im['sensitivity_view_boundary_ignored']['false_lock_rate_target_present']}",
        "",
        "## Reacquisition episodes",
        "",
    ]
    if im["reacquisition_episodes"]:
        lines.append("| Start | End | GT onset | State relock | First valid bbox | Identity status | Latency |")
        lines.append("|---:|---:|---|---:|---:|---|---|")
        for e in im["reacquisition_episodes"]:
            lines.append(
                f"| {e['start']} | {e['end']} | {e.get('onset_visibility')} | "
                f"{e.get('state_relock_frame')} | {e.get('first_valid_output_frame')} | "
                f"{e['status']} | {e['identity_latency_frames']} |")
    else:
        lines.append("_No confirmed out-of-view GT episode._")
    lines += [
        "",
        f"- Unknown-onset events: {len(result['events']['gt_unknown_onset'])}",
        f"- Viewpoint changes: {len(result['events']['gt_viewpoint_changes'])}",
        f"- Occlusion recoveries: {len(result['events']['gt_occlusion_recoveries'])}",
        f"- Tracker events: {len(result['events']['tracker_events'])}",
        "",
        "## Scientific validity",
        "",
        f"**{result['scientific_validity']['status']}** — {result['scientific_validity']['reason']}",
        "",
        f"_{im['warning']}_",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: str, result: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = os.path.join(out_dir, f"run_{stamp}")
    tmp_dir = run_dir + ".tmp"
    if os.path.exists(run_dir):
        raise ValueError(f"run directory already exists: {run_dir}")
    os.makedirs(tmp_dir)
    json_path = os.path.join(tmp_dir, "gate_a_v3_metrics.json")
    md_path = os.path.join(tmp_dir, "REPORT_VI.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    manifest = []
    for name in sorted(os.listdir(tmp_dir)):
        p = os.path.join(tmp_dir, name)
        manifest.append(f"{sha256_file(p)}  {name}")
    with open(os.path.join(tmp_dir, "MANIFEST_SHA256.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    os.replace(tmp_dir, run_dir)
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate A Evaluator v3: fail-closed, reproducible identity evaluation.")
    parser.add_argument("--video", required=True, help="source video (read-only)")
    parser.add_argument("--log", required=True, help="tracking CSV log (read-only)")
    parser.add_argument("--ground-truth", required=True, help="GT interval CSV (read-only)")
    parser.add_argument("--reviews", default=None, help="converted human identity review CSV (read-only)")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "research_handoff",
                                                          "reports", "gateA_evaluator_v3_runs"),
                        help="parent directory for a NEW versioned run directory")
    parser.add_argument("--validate-only", action="store_true", help="validate inputs and exit")
    parser.add_argument("--fps", type=float, default=29.844, help="fps for latency conversion")
    parser.add_argument("--margin", type=int, default=30, help="GT boundary margin (frames)")
    parser.add_argument("--no-decode-count", action="store_true",
                        help="skip full video decode (metadata frame count only)")
    args = parser.parse_args(argv)

    try:
        _assert_output_safe(args.out_dir, [args.video, args.log, args.ground_truth, args.reviews])
        validated = validate_inputs(args)
    except Exception as exc:
        print(f"[FAIL-CLOSED] validation failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "video_decoded_frames": validated["video"]["decoded_frames"],
        "max_frame": validated["max_frame"],
        "log_rows": len(validated["log"]),
        "gt_intervals": len(validated["gt"]),
        "reviews": len(validated["reviews"]),
        "gt_episodes": len(validated["episodes"]["reacquisition_episodes"]),
        "unknown_onset": len(validated["episodes"]["unknown_onset_reappearances"]),
        "tracker_events": len(validated["tracker_events"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.validate_only:
        print("VALIDATION OK")
        return 0
    try:
        result = compute_all(args, validated)
        run_dir = write_outputs(args.out_dir, result)
    except Exception as exc:
        print(f"[FAIL-CLOSED] evaluation failed: {exc}", file=sys.stderr)
        return 3
    print(f"RUN DIR: {run_dir}")
    print(f"IDENTITY STATUS: {result['identity_metrics']['overall_status']}")
    print(f"SCIENTIFIC VALIDITY: {result['scientific_validity']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
