#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import cv2
import torch

from follow_target import DetectorTracker, PersonReIDExtractor, compute_iou


def parse_ground_truth(gt_path: str):
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
                "view": str(row["view"]).strip()
            })
    return gt_intervals


def get_gt_info(gt_intervals, frame_idx):
    for item in gt_intervals:
        if item["start_frame"] <= frame_idx <= item["end_frame"]:
            return item["target_visible"], item["view"]
    return 0, "none"


def calibrate(video_path: str, gt_path: str, sample_stride: int = 5):
    print(f"\n[Calibrate] Loading video: {video_path}")
    print(f"[Calibrate] Loading ground truth: {gt_path}")

    gt_intervals = parse_ground_truth(gt_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    detector = DetectorTracker()
    reid = PersonReIDExtractor()

    frame_idx = 0
    target_embeddings = {"front": [], "back": [], "side": [], "all": []}
    non_target_embeddings = []

    print("[Calibrate] Extracting feature embeddings across frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride == 0:
            tracks = detector.track(frame)
            if tracks:
                target_vis, target_view = get_gt_info(gt_intervals, frame_idx)

                frame_h, frame_w = frame.shape[:2]
                crops, tids = [], []

                for t in tracks:
                    x1, y1 = max(0, t.x1), max(0, t.y1)
                    x2, y2 = min(frame_w, t.x2), min(frame_h, t.y2)
                    if (x2 - x1) >= 30 and (y2 - y1) >= 80:
                        crops.append(frame[y1:y2, x1:x2])
                        tids.append(t.track_id)

                if crops:
                    feats = reid.embed_batch(crops)
                    for tid, feat in zip(tids, feats):
                        if target_vis == 1:
                            target_embeddings["all"].append(feat)
                            if target_view in target_embeddings:
                                target_embeddings[target_view].append(feat)
                        else:
                            non_target_embeddings.append(feat)

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 60)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("=" * 60)

    if target_embeddings["all"] and non_target_embeddings:
        tgt_mat = np.array(target_embeddings["all"])
        non_tgt_mat = np.array(non_target_embeddings)
        diff_sims = np.dot(tgt_mat, non_tgt_mat.T).flatten()

        p95_diff = np.percentile(diff_sims, 95)
        p99_diff = np.percentile(diff_sims, 99)
        p999_diff = np.percentile(diff_sims, 99.9)

        print(f"\n[DIFFERENT-person Similarities] (Count: {len(diff_sims)})")
        print(f"  50th percentile : {np.percentile(diff_sims, 50):.4f}")
        print(f"  95th percentile : {p95_diff:.4f}")
        print(f"  99th percentile : {p99_diff:.4f}  <-- Recommended T_ACCEPT Min")
        print(f"  99.9th percentile: {p999_diff:.4f}")

        print(f"\n[SAME-person Similarities & Retention Rate at T_ACCEPT >= {p99_diff:.2f}]")
        for v in ["all", "front", "back", "side"]:
            embs = target_embeddings[v]
            if len(embs) > 1:
                mat = np.array(embs)
                sim_matrix = np.dot(mat, mat.T)
                triu_indices = np.triu_indices_from(sim_matrix, k=1)
                same_sims = sim_matrix[triu_indices]

                kept_ratio = np.mean(same_sims >= p99_diff) * 100.0
                print(f"  View [{v:5s}] | Count: {len(same_sims):5d} | "
                      f"Mean Sim: {np.mean(same_sims):.4f} | "
                      f"Retained at T_ACCEPT: {kept_ratio:5.1f}%")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate ReID similarity thresholds.")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth CSV path")
    args = parser.parse_args()

    calibrate(args.video, args.gt)
