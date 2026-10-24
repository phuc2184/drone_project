#!/usr/bin/env python3
"""
/home/foxsec/drone_project/calibrate_thresholds.py

PART C / STEP 3: ReID similarity threshold calibration tool.
Collects embeddings from ground-truth labeled tracks (target vs other people),
computes similarity distributions (pairwise, anchor-only gallery, all-gallery),
reports percentiles per view and for front-vs-back, and validates the T_ACCEPT
threshold against the 99th percentile of different-person similarities.
"""

import os
import csv
import argparse
from typing import List, Dict, Any, Tuple
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from follow_target import DetectorTracker, PersonReIDExtractor


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


def get_gt_info(gt_intervals: List[Dict[str, Any]], frame_idx: int) -> Tuple[int, str, Any]:
    for item in gt_intervals:
        if item["start_frame"] <= frame_idx <= item["end_frame"]:
            return item["target_visible"], item["view"], item["target_id"]
    return 0, "none", None


def compute_top_k_similarity(gallery: List[np.ndarray], query: np.ndarray, k: int = 3) -> float:
    if not gallery:
        return 0.0
    mat = np.array(gallery)
    sims = np.dot(mat, query)
    actual_k = min(k, len(sims))
    return float(np.mean(np.sort(sims)[-actual_k:]))


def calibrate(video_path: str,
              gt_path: str,
              sample_stride: int = 5,
              target_track_ids: List[int] = None,
              output_dir: str = "/home/foxsec/drone_project/outputs") -> None:
    print("\n" + "=" * 70)
    print("STEP 3: REID THRESHOLD CALIBRATION (calibrate_thresholds.py)")
    print("=" * 70)
    print(f"  Video file       : {video_path}")
    print(f"  Ground truth     : {gt_path}")
    print(f"  Sampling stride  : every {sample_stride} frames")
    os.makedirs(output_dir, exist_ok=True)

    gt_intervals = parse_ground_truth(gt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = DetectorTracker()
    reid = PersonReIDExtractor()

    # Track ID mapping for target in 5.mp4 if not explicitly in CSV
    # 0-904: 1 (front), 1751-2109: 350 (front), 2460-2513: 468 (back), 2581-2625: 534 (back), 2684-2697: 544 (side)
    default_interval_targets = [
        (0, 904, 1),
        (1751, 2109, 350),
        (2460, 2513, 468),
        (2581, 2625, 534),
        (2684, 2697, 544),
    ]

    target_embeddings: Dict[str, List[np.ndarray]] = {"all": [], "front": [], "back": [], "side": []}
    non_target_embeddings: List[np.ndarray] = []

    anchor_gallery: List[np.ndarray] = []
    dynamic_gallery: List[np.ndarray] = []
    last_anchor_frame = -999

    print(f"\n[1/3] Extracting embeddings across {total_frames} frames (sampling every {sample_stride} frames)...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # IMPORTANT: Run detector & tracker on EVERY frame to preserve Kalman filter & motion continuity
        tracks = detector.track(frame)

        if tracks and (frame_idx % sample_stride == 0):
            target_vis, target_view, row_tgt_id = get_gt_info(gt_intervals, frame_idx)

            # Determine expected target ID for current frame
            expected_target_id = row_tgt_id
            if expected_target_id is None and target_vis == 1:
                for s_f, e_f, t_id in default_interval_targets:
                    if s_f <= frame_idx <= e_f:
                        expected_target_id = t_id
                        break

            frame_h, frame_w = frame.shape[:2]
            crops, valid_tids = [], []
            for t in tracks:
                x1, y1 = max(0, t.x1), max(0, t.y1)
                x2, y2 = min(frame_w, t.x2), min(frame_h, t.y2)
                if (x2 - x1) >= 30 and (y2 - y1) >= 80:
                    crops.append(frame[y1:y2, x1:x2])
                    valid_tids.append(t.track_id)

            if crops:
                feats = reid.embed_batch(crops)
                for tid, feat in zip(valid_tids, feats):
                    is_target = False
                    if target_vis == 1 and expected_target_id is not None:
                        is_target = (tid == expected_target_id)
                    elif target_vis == 1 and target_track_ids:
                        is_target = (tid in target_track_ids)

                    if is_target:
                        target_embeddings["all"].append(feat)
                        if target_view in target_embeddings:
                            target_embeddings[target_view].append(feat)

                        # Build anchor gallery (frames 0 to 188)
                        if frame_idx <= 188 and len(anchor_gallery) < 10:
                            if frame_idx - last_anchor_frame >= 5:
                                max_sim = max([float(np.dot(a, feat)) for a in anchor_gallery]) if anchor_gallery else 0.0
                                if max_sim <= 0.95:
                                    anchor_gallery.append(feat)
                                    last_anchor_frame = frame_idx
                        elif len(anchor_gallery) >= 10 and target_view == "front":
                            # Dynamic collection candidates
                            a_sim = compute_top_k_similarity(anchor_gallery, feat, k=3)
                            if a_sim >= 0.80 and len(dynamic_gallery) < 30:
                                dynamic_gallery.append(feat)
                    else:
                        non_target_embeddings.append(feat)

        frame_idx += 1

    cap.release()

    all_gallery = list(anchor_gallery) + list(dynamic_gallery)

    print(f"\n[Extraction Summary]")
    print(f"  Target Embeddings Total : {len(target_embeddings['all'])}")
    print(f"    - Front view          : {len(target_embeddings['front'])}")
    print(f"    - Back view           : {len(target_embeddings['back'])}")
    print(f"    - Side view           : {len(target_embeddings['side'])}")
    print(f"  Non-Target Embeddings   : {len(non_target_embeddings)}")
    print(f"  Simulated Anchor Gallery: {len(anchor_gallery)} embeddings")
    print(f"  Simulated All Gallery   : {len(all_gallery)} embeddings")

    if not target_embeddings["all"] or not non_target_embeddings:
        print("\n[Error] Insufficient target or non-target embeddings collected for calibration.")
        return

    # =========================================================================
    # 1. PAIRWISE SIMILARITIES
    # =========================================================================
    tgt_all_mat = np.array(target_embeddings["all"])
    non_tgt_mat = np.array(non_target_embeddings)
    diff_sims = np.dot(tgt_all_mat, non_tgt_mat.T).flatten()

    def _pairwise_triu(embs: List[np.ndarray]) -> np.ndarray:
        if len(embs) < 2:
            return np.array([])
        mat = np.array(embs)
        sim_mat = np.dot(mat, mat.T)
        triu_idx = np.triu_indices_from(sim_mat, k=1)
        return sim_mat[triu_idx]

    same_all_sims = _pairwise_triu(target_embeddings["all"])
    same_front_sims = _pairwise_triu(target_embeddings["front"])
    same_back_sims = _pairwise_triu(target_embeddings["back"])
    same_side_sims = _pairwise_triu(target_embeddings["side"])

    # Front-vs-back pairwise similarities
    front_mat = np.array(target_embeddings["front"]) if target_embeddings["front"] else None
    back_mat = np.array(target_embeddings["back"]) if target_embeddings["back"] else None
    front_back_sims = np.dot(front_mat, back_mat.T).flatten() if front_mat is not None and back_mat is not None else np.array([])

    # Percentiles of DIFFERENT-person similarities
    p50_diff = float(np.percentile(diff_sims, 50))
    p90_diff = float(np.percentile(diff_sims, 90))
    p95_diff = float(np.percentile(diff_sims, 95))
    p98_diff = float(np.percentile(diff_sims, 98))
    p99_diff = float(np.percentile(diff_sims, 99))
    p999_diff = float(np.percentile(diff_sims, 99.9))
    max_diff = float(np.max(diff_sims))

    print("\n" + "-" * 70)
    print("1. DIFFERENT-PERSON SIMILARITY DISTRIBUTION (Pairs: {})".format(len(diff_sims)))
    print("-" * 70)
    print(f"  50th Percentile (Median)  : {p50_diff:.4f}")
    print(f"  90th Percentile           : {p90_diff:.4f}")
    print(f"  95th Percentile           : {p95_diff:.4f}")
    print(f"  98th Percentile           : {p98_diff:.4f}")
    print(f"  99th Percentile (P99)     : {p99_diff:.4f}  <-- MINIMUM SAFE T_ACCEPT THRESHOLD")
    print(f"  99.9th Percentile         : {p999_diff:.4f}")
    print(f"  Maximum Observed Diff     : {max_diff:.4f}")

    print("\n" + "-" * 70)
    print(f"2. SAME-PERSON SIMILARITY DISTRIBUTIONS & RETENTION AT T_ACCEPT >= {p99_diff:.4f}")
    print("-" * 70)
    print(f"{'Category':<22} | {'Count':<7} | {'Mean':<7} | {'Min':<7} | {'P05':<7} | {'P50':<7} | {'Retained (>= P99)':<18}")
    print("-" * 95)

    categories = [
        ("SAME (All views)", same_all_sims),
        ("SAME (Front-Front)", same_front_sims),
        ("SAME (Back-Back)", same_back_sims),
        ("SAME (Side-Side)", same_side_sims),
        ("SAME (Front-vs-Back)", front_back_sims)
    ]

    for name, sims in categories:
        if len(sims) > 0:
            mean_s = np.mean(sims)
            min_s = np.min(sims)
            p05_s = np.percentile(sims, 5)
            p50_s = np.percentile(sims, 50)
            ret_ratio = np.mean(sims >= p99_diff) * 100.0
            print(f"{name:<22} | {len(sims):<7d} | {mean_s:<7.4f} | {min_s:<7.4f} | {p05_s:<7.4f} | {p50_s:<7.4f} | {ret_ratio:<17.1f}%")
        else:
            print(f"{name:<22} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<18}")

    # =========================================================================
    # 2. GALLERY SCORES (Anchor-only and All-gallery top-3)
    # =========================================================================
    diff_anchor_scores = [compute_top_k_similarity(anchor_gallery, q) for q in non_target_embeddings]
    diff_all_scores = [compute_top_k_similarity(all_gallery, q) for q in non_target_embeddings]

    target_anchor_scores = {v: [compute_top_k_similarity(anchor_gallery, q) for q in target_embeddings[v]]
                            for v in ["all", "front", "back", "side"]}
    target_all_scores = {v: [compute_top_k_similarity(all_gallery, q) for q in target_embeddings[v]]
                         for v in ["all", "front", "back", "side"]}

    p99_diff_anchor = float(np.percentile(diff_anchor_scores, 99)) if diff_anchor_scores else 0.0
    p99_diff_all = float(np.percentile(diff_all_scores, 99)) if diff_all_scores else 0.0

    print("\n" + "-" * 70)
    print("3. GALLERY SCORE DISTRIBUTIONS (TOP-3 MEAN AS IN PIPELINE)")
    print("-" * 70)
    print(f"  DIFFERENT-person Anchor-Only Score P99 : {p99_diff_anchor:.4f} (Max: {np.max(diff_anchor_scores):.4f})")
    print(f"  DIFFERENT-person All-Gallery Score P99 : {p99_diff_all:.4f} (Max: {np.max(diff_all_scores):.4f})")
    print("\n  [Target Gallery Scores & Retention]")
    for v in ["front", "back", "side", "all"]:
        sc_anc = target_anchor_scores[v]
        sc_all = target_all_scores[v]
        if sc_anc and sc_all:
            ret_anc_floor = np.mean(np.array(sc_anc) >= 0.60) * 100.0  # T_ANCHOR_FLOOR = 0.60
            ret_accept_long = np.mean(np.array(sc_all) >= 0.78) * 100.0  # T_ACCEPT_LONG = 0.78
            ret_p99 = np.mean(np.array(sc_all) >= p99_diff_all) * 100.0
            print(f"    View [{v:5s}] (N={len(sc_anc):3d}) | "
                  f"Mean Anc: {np.mean(sc_anc):.4f} (>= 0.60 Floor: {ret_anc_floor:5.1f}%) | "
                  f"Mean All: {np.mean(sc_all):.4f} (>= 0.78 Accept: {ret_accept_long:5.1f}%, >= P99: {ret_p99:5.1f}%)")

    # =========================================================================
    # 3. PLOT HISTOGRAMS
    # =========================================================================
    print("\n[3/3] Generating calibration distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(0.0, 1.0, 50)

    # Plot 1: Pairwise Different vs Same All
    ax = axes[0, 0]
    ax.hist(diff_sims, bins=bins, alpha=0.6, color="crimson", density=True, label="Different Person")
    ax.hist(same_all_sims, bins=bins, alpha=0.6, color="forestgreen", density=True, label="Same Person (All Views)")
    ax.axvline(p99_diff, color="red", linestyle="--", linewidth=2, label=f"Diff P99 ({p99_diff:.2f})")
    ax.axvline(0.78, color="blue", linestyle=":", linewidth=2, label="T_ACCEPT_LONG (0.78)")
    ax.axvline(0.60, color="orange", linestyle="-.", linewidth=2, label="T_ANCHOR_FLOOR (0.60)")
    ax.set_title("Pairwise Similarity: Different vs Same (All Views)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Front-vs-Front vs Front-vs-Back
    ax = axes[0, 1]
    if len(same_front_sims) > 0:
        ax.hist(same_front_sims, bins=bins, alpha=0.6, color="forestgreen", density=True, label="Front vs Front")
    if len(front_back_sims) > 0:
        ax.hist(front_back_sims, bins=bins, alpha=0.6, color="purple", density=True, label="Front vs Back")
    ax.hist(diff_sims, bins=bins, alpha=0.3, color="crimson", density=True, label="Different Person (ref)")
    ax.axvline(p99_diff, color="red", linestyle="--", linewidth=2, label=f"Diff P99 ({p99_diff:.2f})")
    ax.axvline(0.78, color="blue", linestyle=":", linewidth=2, label="T_ACCEPT_LONG (0.78)")
    ax.axvline(0.60, color="orange", linestyle="-.", linewidth=2, label="T_ANCHOR_FLOOR (0.60)")
    ax.set_title("Target View Invariance: Front-to-Front vs Front-to-Back", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Gallery Anchor-Only Scores
    ax = axes[1, 0]
    ax.hist(diff_anchor_scores, bins=bins, alpha=0.6, color="crimson", density=True, label="Different Person")
    if target_anchor_scores["front"]:
        ax.hist(target_anchor_scores["front"], bins=bins, alpha=0.6, color="forestgreen", density=True, label="Target Front")
    if target_anchor_scores["back"]:
        ax.hist(target_anchor_scores["back"], bins=bins, alpha=0.6, color="purple", density=True, label="Target Back")
    ax.axvline(0.60, color="orange", linestyle="-.", linewidth=2, label="T_ANCHOR_FLOOR (0.60)")
    ax.axvline(p99_diff_anchor, color="red", linestyle="--", linewidth=2, label=f"Diff Anchor P99 ({p99_diff_anchor:.2f})")
    ax.set_title("Anchor-Only Gallery Score (Top-3 Mean against 10 Anchors)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Anchor Similarity Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: All-Gallery Scores
    ax = axes[1, 1]
    ax.hist(diff_all_scores, bins=bins, alpha=0.6, color="crimson", density=True, label="Different Person")
    if target_all_scores["front"]:
        ax.hist(target_all_scores["front"], bins=bins, alpha=0.6, color="forestgreen", density=True, label="Target Front")
    if target_all_scores["back"]:
        ax.hist(target_all_scores["back"], bins=bins, alpha=0.6, color="purple", density=True, label="Target Back")
    ax.axvline(0.78, color="blue", linestyle=":", linewidth=2, label="T_ACCEPT_LONG (0.78)")
    ax.axvline(p99_diff_all, color="red", linestyle="--", linewidth=2, label=f"Diff All P99 ({p99_diff_all:.2f})")
    ax.set_title("All-Gallery Score (Top-3 Mean against Anchor+Dynamic)", fontsize=11, fontweight="bold")
    ax.set_xlabel("All Gallery Similarity Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_out_path = os.path.join(output_dir, "calibration_histograms.png")
    plt.savefig(hist_out_path, dpi=200)
    plt.close()
    print(f"  Plots saved to: {hist_out_path}")

    # =========================================================================
    # 4. RECOMMENDATION SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION FINDINGS & THRESHOLD RULE VERIFICATION")
    print("=" * 70)
    print(f"  1. 99th Percentile of Different-Person Similarity: P99 = {p99_diff:.4f}")
    print(f"  2. Current T_ACCEPT_LONG: 0.78")
    if 0.78 >= p99_diff:
        print(f"     -> RULE COMPLIANCE: PASS (0.78 >= {p99_diff:.4f}). Safe against false locks.")
    else:
        print(f"     -> RULE COMPLIANCE: WARNING (0.78 < {p99_diff:.4f}). Risk of false lock.")

    back_ret = np.mean(np.array(target_all_scores["back"]) >= 0.78) * 100.0 if target_all_scores["back"] else 0.0
    print(f"  3. Back-view Retention under T_ACCEPT=0.78: {back_ret:.1f}%")
    print(f"     -> Explains why back-view reacquisition occurs late without view-chain updates.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate ReID similarity thresholds.")
    parser.add_argument("--video", type=str, default="/home/foxsec/drone_project/videos/5.mp4", help="Input video path")
    parser.add_argument("--gt", type=str, default="/home/foxsec/drone_project/ground_truth/5.csv", help="Ground truth CSV path")
    parser.add_argument("--stride", type=int, default=5, help="Sampling frame stride")
    parser.add_argument("--target-ids", type=int, nargs="*", default=None, help="List of target track IDs across video")
    parser.add_argument("--out-dir", type=str, default="/home/foxsec/drone_project/outputs", help="Output directory for plots")
    args = parser.parse_args()

    calibrate(
        video_path=args.video,
        gt_path=args.gt,
        sample_stride=args.stride,
        target_track_ids=args.target_ids,
        output_dir=args.out_dir
    )
