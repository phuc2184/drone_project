#!/usr/bin/env python3
"""
/home/foxsec/drone_project/evaluate.py

Tracking evaluation & Spot-Check / Segment verification tool.

Key Metrics & Rules:
1. FALSE LOCK = TARGET label (state == 'TRACKING') while target_visible = 0,
   EXCLUDING:
   (a) frames within 30 frames of any GT boundary (boundary uncertainty), and
   (b) frames with target_visible = 2 (partial visibility).
   These excluded frames are reported separately as 'uncertain frames'.
2. Coverage is computed over target_visible = 1 frames outside the 30-frame boundary margin.
   (Raw total visible coverage is also reported for reference).
3. Reacquisition delay is computed in frames (with seconds equivalent).
   Delays below 30 frames are noted as within GT boundary uncertainty (+-30 frames).
4. WARNING: This metric cannot detect a green box on the WRONG person while
   the target is visible; visual checks (--segments) are required for that.
5. Segment export (--segments): exports first, middle, last frame for all tracking segments.
"""

import os
import csv
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set

import cv2
import numpy as np


def parse_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    gt_intervals = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_intervals.append({
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "target_visible": int(row["target_visible"]),
                "view": str(row.get("view", "none")).strip().lower(),
                "target_id": int(row["target_id"]) if row.get("target_id") and row["target_id"].strip().isdigit() else None
            })
    return gt_intervals


def get_gt_info_at_frame(gt_intervals: List[Dict[str, Any]], frame_idx: int) -> Tuple[int, str, Optional[int]]:
    for item in gt_intervals:
        if item["start_frame"] <= frame_idx <= item["end_frame"]:
            return item["target_visible"], item["view"], item["target_id"]
    return 0, "none", None


def get_tracking_segments(log_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = []
    in_seg = False
    start_f = None
    curr_tid = None
    for r in log_rows:
        fn = int(r["frame"])
        st = r["state"]
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
        last_fn = int(log_rows[-1]["frame"])
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


def evaluate_log_gt(csv_log_path: str,
                    gt_path: str,
                    fps: float = 30.0,
                    margin_frames: int = 30,
                    known_target_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    if not os.path.exists(csv_log_path):
        raise FileNotFoundError(f"CSV log not found: {csv_log_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    log_rows = []
    with open(csv_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            log_rows.append(row)

    gt_intervals = parse_ground_truth(gt_path)
    total_frames = len(log_rows)
    if total_frames == 0:
        print("[Error] CSV log is empty.")
        return {}

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

    tracking_rows = [r for r in log_rows if r["state"] == "TRACKING"]
    total_tracked_frames = len(tracking_rows)

    false_lock_frames = 0
    false_lock_frame_indices = []

    uncertain_frames_total = 0
    uncertain_tracking_frames = 0
    uncertain_tracking_boundary_frames = 0
    uncertain_tracking_partial_frames = 0

    # Frame-by-frame evaluation
    for r in log_rows:
        f = int(r["frame"])
        state = r["state"]
        tid_str = r.get("target_track_id", "").strip()

        is_visible, view, _ = get_gt_info_at_frame(gt_intervals, f)
        near_bnd = is_near_boundary(f, margin_frames)

        # Track uncertain frame counts
        if is_visible == 2 or (is_visible == 0 and near_bnd):
            uncertain_frames_total += 1

        # Visible frame counting (raw and margin-filtered)
        if is_visible == 1:
            if view in view_stats:
                view_stats[view]["raw_visible_frames"] += 1
                if not near_bnd:
                    view_stats[view]["margin_visible_frames"] += 1
        elif is_visible == 2:
            if view in view_stats:
                view_stats[view]["partial_frames"] += 1

        if state == "TRACKING":
            # Tracked frame counting
            if is_visible == 1:
                if view in view_stats:
                    view_stats[view]["raw_tracked_frames"] += 1
                    if not near_bnd:
                        view_stats[view]["margin_tracked_frames"] += 1
            elif is_visible == 2:
                if view in view_stats:
                    view_stats[view]["partial_tracked_frames"] += 1

            # Redefined FALSE LOCK logic:
            # FALSE LOCK = TARGET label while target_visible = 0,
            # EXCLUDING:
            #  (a) frames within 30 frames of any GT boundary (reported as uncertain)
            #  (b) frames with target_visible = 2 (reported as uncertain)
            is_false_lock = False
            if is_visible == 0:
                if near_bnd:
                    uncertain_tracking_frames += 1
                    uncertain_tracking_boundary_frames += 1
                    if view in view_stats:
                        view_stats[view]["uncertain_tracking_frames"] += 1
                else:
                    is_false_lock = True
            elif is_visible == 2:
                uncertain_tracking_frames += 1
                uncertain_tracking_partial_frames += 1
                if view in view_stats:
                    view_stats[view]["uncertain_tracking_frames"] += 1
            elif is_visible == 1:
                # Target visible: this metric evaluates absence only; cannot detect identity error without visual check
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

    # Reacquisition episodes analysis
    # Target reappearances where target_visible transitions from non-visible to 1
    reacquisition_episodes = []
    for i, interval in enumerate(gt_intervals):
        if interval["target_visible"] == 1 and i > 0:
            reacquisition_episodes.append(interval)

    total_return_episodes = len(reacquisition_episodes)
    successful_reacquisitions = 0
    all_reacq_delays_frames = []
    all_reacq_delays_sec = []

    log_frame_map = {int(r["frame"]): r for r in log_rows}

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
    print(f"  FALSE LOCK Frames             : {false_lock_frames} [state=TRACKING while target_visible=0 outside margin]")
    print(f"  FALSE LOCK RATE               : {false_lock_rate:.2f}% (Requirement: ~0%)")
    print(f"  Reacquisition Success         : {successful_reacquisitions}/{total_return_episodes} ({reacq_success_rate:.1f}%)")
    if avg_reacq_frames is not None:
        print(f"  Avg Reacquisition Delay       : {avg_reacq_frames:.1f} frames ({avg_reacq_time:.2f} s)")
    else:
        print(f"  Avg Reacquisition Delay       : N/A (no reacquisitions)")
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
        "false_lock_frames": false_lock_frames,
        "false_lock_rate": false_lock_rate,
        "id_switches": id_switches,
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
    args = parser.parse_args()

    if args.spot_check:
        run_spot_check(args.spot_check, stride=args.stride)
    elif args.segments:
        if not args.log:
            print("[Error] --segments requires --log <path_to_log.csv>")
        else:
            # Infer video if not provided
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
        evaluate_log_gt(args.log, args.gt, fps=args.fps, margin_frames=args.margin, known_target_ids=args.target_ids)
    else:
        print("Usage:")
        print("  Quantitative   : python3 evaluate.py --log logs/<log>.csv [--gt ground_truth/<gt>.csv]")
        print("  Segment Export : python3 evaluate.py --segments --log logs/<log>.csv [--video outputs/<video>.mp4] [--variant <name>]")
        print("  Spot check     : python3 evaluate.py --spot-check outputs/<output>.mp4 [--stride 30]")
