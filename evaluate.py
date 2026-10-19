#!/usr/bin/env python3
import os
import csv
import argparse
import cv2
import numpy as np


def evaluate_log_gt(csv_log_path: str, gt_path: str):
    if not os.path.exists(csv_log_path) or not os.path.exists(gt_path):
        raise FileNotFoundError("CSV log or Ground Truth file missing.")

    log_rows = []
    with open(csv_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            log_rows.append(row)

    gt_intervals = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_intervals.append({
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "target_visible": int(row["target_visible"]),
                "view": str(row["view"]).strip()
            })

    total_frames = len(log_rows)
    tracking_rows = [r for r in log_rows if r["state"] == "TRACKING"]

    false_lock_frames = 0
    for row in tracking_rows:
        f = int(row["frame"])
        is_visible = 0
        for item in gt_intervals:
            if item["start_frame"] <= f <= item["end_frame"]:
                is_visible = item["target_visible"]
                break
        if is_visible == 0:
            false_lock_frames += 1

    false_lock_rate = (false_lock_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

    id_switches = 0
    prev_id = None
    for row in tracking_rows:
        tid_str = row.get("target_track_id", "").strip()
        if tid_str:
            tid = int(tid_str)
            if prev_id is not None and tid != prev_id:
                id_switches += 1
            prev_id = tid

    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total Video Frames       : {total_frames}")
    print(f"  Tracked Frames           : {len(tracking_rows)}")
    print(f"  False Lock Frames        : {false_lock_frames}")
    print(f"  FALSE LOCK RATE          : {false_lock_rate:.2f}% (Requirement: ~0%)")
    print(f"  Target ID Switches       : {id_switches}")
    print("=" * 60 + "\n")


def run_spot_check(video_out_path: str, stride: int = 30):
    if not os.path.exists(video_out_path):
        raise FileNotFoundError(f"Annotated output video not found: {video_out_path}")

    cap = cv2.VideoCapture(video_out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 60)
    print("KEYPRESS SPOT-CHECK MODE:")
    print("  'y' = Green box is on CORRECT target")
    print("  'n' = Green box is on WRONG person (FALSE LOCK)")
    print("  's' = No green box shown (LOST / SEARCHING)")
    print("  'q' = Quit spot check")
    print("=" * 60 + "\n")

    counts = {"y": 0, "n": 0, "s": 0}
    frame_idx = 0

    cv2.namedWindow("Spot-Check Evaluation", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            banner = np.zeros((50, frame.shape[1], 3), dtype=np.uint8)
            msg = f"Frame {frame_idx}/{total_frames} | Press: 'y' Correct | 'n' False Lock | 's' Skipped/No Box"
            cv2.putText(banner, msg, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            combined = np.vstack([banner, frame])

            cv2.imshow("Spot-Check Evaluation", combined)
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
                    return

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    total_samples = sum(counts.values())
    print("\n" + "=" * 60)
    print("SPOT-CHECK RESULTS SUMMARY")
    print(f"  Total Checked Frames : {total_samples}")
    print(f"  Correct Locks (y)    : {counts['y']} ({counts['y']/max(1,total_samples)*100:.1f}%)")
    print(f"  False Locks (n)      : {counts['n']} ({counts['n']/max(1,total_samples)*100:.1f}%)")
    print(f"  No Box / Search (s)  : {counts['s']} ({counts['s']/max(1,total_samples)*100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking performance.")
    parser.add_argument("--log", type=str, help="Path to CSV tracking log")
    parser.add_argument("--gt", type=str, help="Path to ground truth CSV")
    parser.add_argument("--spot-check", type=str, help="Path to annotated video for spot-checking")
    args = parser.parse_args()

    if args.spot_check:
        run_spot_check(args.spot_check)
    elif args.log and args.gt:
        evaluate_log_gt(args.log, args.gt)
    else:
        print("Please provide --spot-check <video.mp4> OR --log <log.csv> --gt <gt.csv>")
