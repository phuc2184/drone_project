#!/usr/bin/env python3
"""Gate A v3 forensics: recompute Gemini v2 claims from source evidence.

Read-only with respect to videos/logs/GT. Writes one JSON + one Markdown table
to the supplied output directory.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import evaluate as ev  # noqa: E402


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_log(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gt_at(gt, f):
    v, view, tid = ev.get_gt_info_at_frame(gt, f)
    return {"visible": v, "view": view, "target_id": tid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="videos/5.mp4")
    ap.add_argument("--log", default="logs/tracking_log_20261106_233941.csv")
    ap.add_argument("--gt", default="ground_truth/5_manual.csv")
    ap.add_argument("--gt-circular", default="ground_truth/5.csv")
    ap.add_argument("--reviews", default="research_handoff/review/derived/gatea_evaluate_identity_reviews_v1.csv")
    ap.add_argument("--gemini-json",
                    default="research_handoff/reports/gemini_gateA_evaluator_20261107_024500/gatea_v2_evaluation_5_manual.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    gt = ev.parse_ground_truth(args.gt)
    gt_circ = ev.parse_ground_truth(args.gt_circular)
    reviews = ev.load_identity_reviews(args.reviews)
    intervals, boundaries = ev.build_gt_boundary_map(gt)
    episodes = ev.build_reacquisition_episodes(gt)
    tracker_events = ev.classify_tracker_events(log, gt)
    identity = ev.evaluate_identity_reviews(reviews, log, gt, fps=29.8438, margin_frames=30)

    cap = cv2.VideoCapture(args.video)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

    state_counts = {}
    for r in log:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    tracking = state_counts.get("TRACKING", 0)
    tracking_valid = sum(1 for r in log if r["state"] == "TRACKING" and r["valid"] == "1")
    log_map = {int(r["frame"]): r for r in log}
    ordered = sorted(log, key=lambda r: int(r["frame"]))
    id_switch_frames = [int(r["frame"]) for i, r in enumerate(ordered)
                        if i > 0 and r.get("target_track_id") != ordered[i - 1].get("target_track_id")]

    # Absence-state proxy with presence/partial/unknown boundary margin only.
    presence_types = {ev.GT_BOUNDARY_PRESENCE, ev.GT_BOUNDARY_PARTIAL, ev.GT_BOUNDARY_UNKNOWN}
    absent_frames = list(range(0, decoded))
    absent_gt = [f for f in absent_frames if ev.get_gt_info_at_frame(intervals, f)[0] == ev.GT_ABSENT]
    absent_outside_margin = [f for f in absent_gt
                             if not ev.is_near_boundary(f, boundaries, 30, presence_types)]
    tracking_on_absent = [f for f in absent_outside_margin
                          if log_map.get(f, {}).get("state") == "TRACKING"]

    # E1-E4 forensic table, keyed by tracker recovery events.
    def review_rows_in(a, b):
        return [r for r in reviews if a <= r["frame"] <= b]

    e_rows = []
    for e in tracker_events:
        a, b = e["lost_start_frame"], e["relock_frame"]
        rows = review_rows_in(a, b + 5)
        buckets = {}
        for r in rows:
            # classification independent of margin: just report raw labels
            key = f'{r["presence"]}/{r["identity"]}/drawn={int(r["drawn"])}'
            buckets[key] = buckets.get(key, 0) + 1
        matched_gt = None
        for ep in episodes["reacquisition_episodes"]:
            if ep["start_frame"] <= b <= ep["end_frame"] or abs(ep["start_frame"] - b) <= 40:
                matched_gt = ep
                break
        e_rows.append({
            **e,
            "gt_at_lost_start": gt_at(intervals, a),
            "gt_at_relock": gt_at(intervals, b),
            "gt_at_first_valid": gt_at(intervals, e["first_valid_output_frame"]) if e["first_valid_output_frame"] else None,
            "output_latency_frames": (e["first_valid_output_frame"] - a) if e["first_valid_output_frame"] else None,
            "matched_gt_episode": None if matched_gt is None else {
                "start": matched_gt["start_frame"], "end": matched_gt["end_frame"],
                "onset_visibility": matched_gt.get("onset_visibility"),
                "first_fully_visible_frame": matched_gt.get("first_fully_visible_frame"),
            },
            "relock_minus_gt_start_frames": (b - matched_gt["start_frame"]) if matched_gt else None,
            "identity_reviews_in_window": len(rows),
            "identity_review_labels": buckets,
        })

    spot_frames = [904, 905, 1740, 1745, 1750, 1751, 1752, 2084, 2085, 2086,
                   2108, 2109, 2110, 2195, 2200, 2205, 2459, 2460, 2461, 2465,
                   2505, 2508, 2509, 2510, 2515, 2560, 2565, 2580, 2581, 2585,
                   2624, 2625, 2628, 2683, 2684, 2690]
    spot = []
    for f in spot_frames:
        r = log_map.get(f)
        spot.append({
            "frame": f, "gt": gt_at(intervals, f), "gt_circular": gt_at(gt_circ, f),
            "log_state": None if r is None else r["state"],
            "log_valid": None if r is None else r["valid"],
            "log_tracker_id": None if r is None else r.get("target_track_id"),
        })

    old = json.load(open(args.gemini_json)) if os.path.exists(args.gemini_json) else None
    old_im = old["identity_metrics"] if old else {}

    claims = [
        {"claim": "video metadata 2699 frames", "reported": 2699,
         "recomputed": meta_frames, "status": "VERIFIED" if meta_frames == 2699 else "CONTRADICTED"},
        {"claim": "video decodes to 2698 frames", "reported": 2698,
         "recomputed": decoded, "status": "VERIFIED" if decoded == 2698 else "CONTRADICTED"},
        {"claim": "3371 total frames", "reported": 3371, "recomputed": decoded,
         "status": "UNREPRODUCIBLE", "note": "no source in video/log/GT supports 3371"},
        {"claim": "1487/2018 tracking coverage", "reported": "1487/2018",
         "recomputed": f"{tracking}/{decoded}", "status": "UNREPRODUCIBLE",
         "note": "denominator 2018 is not derivable; actual TRACKING is 1187"},
        {"claim": "1483/2018 output coverage", "reported": "1483/2018",
         "recomputed": f"{tracking_valid}/{decoded}", "status": "UNREPRODUCIBLE",
         "note": "actual TRACKING+valid is 1173"},
        {"claim": "0/846 absent-state false lock outside margin", "reported": "0/846",
         "recomputed": f"{len(tracking_on_absent)}/{len(absent_outside_margin)}",
         "status": "VERIFIED_FOR_RECOMPUTED_WINDOW" if not tracking_on_absent else "CONTRADICTED",
         "note": "GT=5_manual; absence windows 905-1750 and 2505-2564; presence-boundary margin 30"},
        {"claim": "2/2 state relock", "reported": "2/2",
         "recomputed": f"{sum(1 for e in tracker_events if e['event_class']=='TRUE_OUT_OF_VIEW_REAPPEARANCE')}/"
                       f"{len(episodes['reacquisition_episodes'])}",
         "status": "STATE_LEVEL_ONLY",
         "note": "both GT out-of-view episodes relocked at state level; 0/2 identity-verified (no eligible review)"},
        {"claim": "E1 latency 0.00 s", "reported": 0.0,
         "recomputed": next((e["relock_minus_gt_start_frames"] for e in e_rows
                             if e["lost_start_frame"] == 905), None),
         "status": "STATE_LEVEL_ONLY",
         "note": "GT boundary 1751 equals state relock; onset independence is UNKNOWN; identity latency is None"},
        {"claim": "E3 latency 0.54 s", "reported": 0.54,
         "recomputed": round((2581 - 2565) / 29.8438, 4),
         "status": "ARITHMETIC_VERIFIED",
         "note": "16 frames / 29.8438 fps = 0.536 s; first reappearance at 2565 is GT-derived, not independently proven"},
    ]

    result = {
        "provenance": {
            "video": {"path": args.video, "sha256": sha256(args.video)},
            "log": {"path": args.log, "sha256": sha256(args.log)},
            "gt_manual": {"path": args.gt, "sha256": sha256(args.gt)},
            "gt_circular": {"path": args.gt_circular, "sha256": sha256(args.gt_circular)},
            "reviews": {"path": args.reviews, "sha256": sha256(args.reviews)},
            "gemini_json": ({"path": args.gemini_json, "sha256": sha256(args.gemini_json)}
                            if old else None),
        },
        "video": {"metadata_frames": meta_frames, "decoded_frames": decoded, "fps": fps},
        "tracking": {"rows": len(log), "state_counts": state_counts,
                     "tracking_frames": tracking, "tracking_valid_frames": tracking_valid,
                     "id_switch_frames": id_switch_frames},
        "gt": {
            "manual_intervals": len(intervals),
            "manual_coverage_frames": sum(i["end_frame"] - i["start_frame"] + 1 for i in intervals),
            "manual_viewpoint_changes": len(episodes["viewpoint_changes"]),
            "manual_occlusion_recoveries": len(episodes["occlusion_recoveries"]),
            "manual_out_of_view_episodes": len(episodes["reacquisition_episodes"]),
            "manual_unknown_onset": len(episodes["unknown_onset_reappearances"]),
            "circular_intervals": len(gt_circ),
            "boundary_2084_2086": {str(f): gt_at(intervals, f) for f in (2084, 2085, 2086)},
        },
        "absence_state_proxy": {
            "gt_absent_frames": len(absent_gt),
            "outside_presence_margin": len(absent_outside_margin),
            "tracking_on_absent_frames": tracking_on_absent,
            "windows": {"905-1750": 846, "2505-2564": 60},
        },
        "gemini_claims": claims,
        "gemini_v2_vs_v3_identity": {
            "old": {k: old_im.get(k) for k in (
                "total_reviewed", "judged_locks", "identity_precision", "lock_recall",
                "false_lock_rate_target_present", "false_lock_rate_target_absent",
                "present_target_lock_recall", "reviewed_present_frames", "reviewed_absent_frames")},
            "v3": {k: identity.get(k) for k in (
                "total_reviewed", "judged_locks", "identity_precision", "lock_recall",
                "false_lock_rate_target_present", "false_lock_rate_target_absent",
                "present_target_lock_recall", "reviewed_present_frames", "reviewed_absent_frames")},
            "v3_metric_status": identity["metric_status"],
            "v3_eligible": identity["eligible_counts"],
            "v3_sensitivity_non_official": identity["sensitivity_view_boundary_ignored"],
        },
        "tracker_events": tracker_events,
        "e1_e4": e_rows,
        "spot_frames": spot,
        "identity_metrics_v3": identity,
        "gt_episodes": episodes,
        "scientific_status": "NOT_ESTABLISHED",
        "warning": "Diagnostic recomputation only. No independent held-out human labels.",
    }
    json_path = out_dir / "gate_a_v3_forensics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    lines = ["# Gate A v3 — Forensic recomputation & Gemini v2 errata", "",
             f"- Video: metadata {meta_frames}, decoded {decoded}, fps {fps:.4f}",
             f"- Log: {len(log)} rows, TRACKING {tracking}, TRACKING+valid {tracking_valid}",
             f"- GT manual: {len(intervals)} intervals, coverage "
             f"{result['gt']['manual_coverage_frames']} frames",
             "", "## Gemini v2 claims", "",
             "| Claim | Reported | Recomputed | Status | Note |", "|---|---:|---:|---|---|"]
    for c in claims:
        lines.append(f"| {c['claim']} | {c['reported']} | {c['recomputed']} | {c['status']} | {c.get('note','')} |")
    lines += ["", "## Identity metric errata (P0)", "",
              "| Metric | Gemini v2 | v3 | v3 status |", "|---|---:|---:|---|"]
    for k in ("false_lock_rate_target_present", "false_lock_rate_target_absent",
              "identity_precision", "lock_recall", "present_target_lock_recall"):
        oldv = old_im.get(k)
        st = identity["metric_status"].get(k, {})
        lines.append(f"| {k} | {oldv} | {identity.get(k)} | {st.get('status')} / {st.get('reason')} |")
    lines += ["", "## E1–E4 tracker recovery", "",
              "| Event | Lost start | Relock | First valid | Prior→new tracker | GT at relock | GT episode start | Relock−GT |",
              "|---|---:|---:|---:|---|---|---:|---:|"]
    for e in e_rows:
        lines.append(f"| {e['event_class']} | {e['lost_start_frame']} | {e['relock_frame']} | "
                     f"{e['first_valid_output_frame']} | {e['prior_tracker_id']}→{e['tracker_id_at_relock']} | "
                     f"vis={e['gt_at_relock']['visible']}/{e['gt_at_relock']['view']} | "
                     f"{(e['matched_gt_episode'] or {}).get('start')} | {e['relock_minus_gt_start_frames']} |")
    lines += ["", "## Spot frames", "",
              "| Frame | GT vis/view | 5.csv vis/view | log state | valid | tracker |", "|---:|---|---|---|---:|---|"]
    for s in spot:
        lines.append(f"| {s['frame']} | {s['gt']['visible']}/{s['gt']['view']} | "
                     f"{s['gt_circular']['visible']}/{s['gt_circular']['view']} | "
                     f"{s['log_state']} | {s['log_valid']} | {s['log_tracker_id']} |")
    lines += ["", f"**Scientific status: {result['scientific_status']}**",
              "", "_Diagnostic only; no independent held-out human labels._"]
    md_path = out_dir / "GEMINI_V2_METRIC_ERRATA_DATA.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
