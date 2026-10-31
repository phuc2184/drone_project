#!/usr/bin/env python3
"""Read-only eye-label analysis for videos/5.mp4.

Replays the baseline tracking run in memory (no video written) with the exact
DetectorTracker / TargetManager / PersonReIDExtractor / QualityFilter classes
from follow_target.py, proves the replay is faithful against the accepted log,
then measures box IoU and embedding similarities against hand-drawn labels.

No decision logic is copied or changed. Nothing under ground_truth/ is written.
"""

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from follow_target import (  # noqa: E402
    DetectorTracker,
    PersonReIDExtractor,
    QualityFilter,
    TargetManager,
    TargetState,
    compute_iou,
    REID_MODEL_NAME,
    REID_WEIGHTS_PATH,
)

PROJECT = "/home/foxsec/drone_project"
VIDEO = os.path.join(PROJECT, "videos/5.mp4")
LOG = os.path.join(PROJECT, "logs/tracking_log_20261026_023427.csv")
LABELS_FIXED = os.path.join(PROJECT, "ground_truth/labels_5_eye_fixed.csv")
LABELS_RAW = os.path.join(PROJECT, "ground_truth/labels_5_eye.csv")
OUT_CSV = os.path.join(PROJECT, "outputs/eye_label_analysis.csv")

MIN_W = 40
MIN_H = 80
ANCHOR_SNAPSHOT_FRAME = 188
ABSENT_RANGE = (936, 1719)


def load_labels():
    """Read the eye labels; auto-exclude boxes smaller than 40x80 px."""
    path = LABELS_FIXED if os.path.exists(LABELS_FIXED) else LABELS_RAW
    raw_dropped = []
    if path == LABELS_FIXED and os.path.exists(LABELS_RAW):
        fixed_frames = set()
        with open(LABELS_FIXED, newline="") as f:
            for r in csv.DictReader(f):
                fixed_frames.add(int(r["frame"]))
        with open(LABELS_RAW, newline="") as f:
            for r in csv.DictReader(f):
                if int(r["frame"]) not in fixed_frames:
                    raw_dropped.append(r)

    rows, excluded = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rec = {
                "frame": int(r["frame"]),
                "status": r["status"].strip(),
                "view": (r["view"] or "").strip(),
            }
            if rec["status"] == "box":
                box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
                rec["box"] = box
                w, h = box[2] - box[0], box[3] - box[1]
                if w < MIN_W or h < MIN_H:
                    excluded.append({**rec, "w": w, "h": h, "why": f"{w}x{h} < {MIN_W}x{MIN_H}"})
                    continue
            rows.append(rec)

    for r in raw_dropped:
        rec = {"frame": int(r["frame"]), "status": r["status"], "view": r["view"]}
        if r["status"] == "box":
            box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            rec["box"] = box
            rec["w"], rec["h"] = box[2] - box[0], box[3] - box[1]
        excluded.append({**rec, "why": "not in labels_5_eye_fixed.csv (known suspicious)"})
    return path, rows, excluded, raw_dropped


def load_log():
    """frame -> (state, target_track_id_string) from the accepted run."""
    out = {}
    with open(LOG, newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["frame"])] = (r["state"], r["target_track_id"])
    return out


def pct(values, qs=(50, 90, 95, 99)):
    if not values:
        return []
    return [float(np.percentile(values, q)) for q in qs]


def fmt(vals, nd=3):
    return "[" + ", ".join(f"{v:.{nd}f}" for v in vals) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", type=int, default=0,
                    help="Replay only the first N frames and check faithfulness (writes nothing)")
    args = ap.parse_args()

    label_path, labels, excluded, raw_dropped = load_labels()
    label_by_frame = {r["frame"]: r for r in labels}
    log_map = load_log()

    print("=" * 78)
    print("EYE-LABEL ANALYSIS (read-only)")
    print(f"  labels file        : {label_path}")
    print(f"  label rows used    : {len(labels)}  (excluded: {len(excluded)})")
    print(f"  log reference      : {LOG} ({len(log_map)} rows)")
    print("=" * 78)
    for e in excluded:
        print(f"  EXCLUDED frame {e['frame']:5d} {e['status']:<12s} {e.get('why','')}")

    gallery_dir = tempfile.mkdtemp(prefix="eye_labels_gallery_", dir="/tmp")
    print(f"  temp gallery_debug : {gallery_dir}")

    detector = DetectorTracker()
    extractor = PersonReIDExtractor(model_name=REID_MODEL_NAME, weights_path=REID_WEIGHTS_PATH)
    quality_filter = QualityFilter()
    manager = TargetManager(
        feature_extractor=extractor,
        quality_filter=quality_filter,
        use_view_chain=False,
        use_color_cue=False,
        gallery_debug_dir=gallery_dir,
    )
    manager.target_track_id = 1

    import cv2

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  video              : {total} frames @ {fps:.3f} FPS")

    replay = []          # (frame, state, target_track_id)
    recs = {}            # labeled frame -> measurements
    anchor_188 = None
    absent_scores = []   # (frame, track_id, sim_anchor, sim_all, tier, box_area)
    hand_embs = {}       # frame -> embedding of the hand box
    label_anchors = {}   # frame -> list of anchor embeddings at that frame
    frame_count = 0
    limit = args.verify_only

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tracks = detector.track(frame)
            if manager.state == TargetState.IDLE and manager.target_track_id is not None:
                for t in tracks:
                    if t.track_id == manager.target_track_id:
                        manager.select_target(t.track_id, t.box, frame_count, frame)
                        break
            out = manager.update(tracks, frame, frame_count, fps)
            replay.append((frame_count, out["state"], manager.target_track_id))

            if frame_count == ANCHOR_SNAPSHOT_FRAME:
                anchor_188 = [a.copy() for a in manager.anchor_gallery]

            if 0 <= frame_count < (limit or 10 ** 9):
                if 936 <= frame_count <= 1719 and out["person_score_details"]:
                    frame_h, frame_w = frame.shape[:2]
                    for t in tracks:
                        d = out["person_score_details"].get(t.track_id)
                        if d is None:
                            continue
                        tier, _ = quality_filter.check_quality(t, tracks, (frame_h, frame_w))
                        absent_scores.append((frame_count, t.track_id, d["sim_anchor"],
                                              d["sim_all"], tier.value, t.area))

                lab = label_by_frame.get(frame_count)
                if lab is not None:
                    frame_h, frame_w = frame.shape[:2]
                    boxes, tiers, details = [], {}, {}
                    for t in tracks:
                        tier, _ = quality_filter.check_quality(t, tracks, (frame_h, frame_w))
                        tiers[t.track_id] = tier.value
                        d = out["person_score_details"].get(t.track_id)
                        details[t.track_id] = d
                        boxes.append((t.track_id, (t.x1, t.y1, t.x2, t.y2), tier.value, d))

                    drawn = out["target_box"]
                    rec = {
                        "frame": frame_count,
                        "view": lab["view"],
                        "status": lab["status"],
                        "state": out["state"],
                        "drawn": drawn is not None,
                        "drawn_box": drawn,
                        "tracks": boxes,
                        "target_track_id": manager.target_track_id,
                        "n_anchors": len(manager.anchor_gallery),
                        "anchor_copy": [a.copy() for a in manager.anchor_gallery],
                    }
                    if lab["status"] == "box":
                        hb = lab["box"]
                        rec["hand_box"] = hb
                        rec["drawn_iou"] = compute_iou(hb, drawn) if drawn is not None else None
                        best = None
                        for tid, box, tier, d in boxes:
                            iou = compute_iou(hb, box)
                            if best is None or iou > best[1]:
                                best = (tid, iou, tier, d)
                        rec["best"] = best
                        x1, y1 = max(0, hb[0]), max(0, hb[1])
                        x2, y2 = min(frame_w, hb[2]), min(frame_h, hb[3])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            emb = extractor.embed(crop)
                            hand_embs[frame_count] = emb
                            rec["hand_sim"] = manager.compute_similarity_details(emb)
                    recs[frame_count] = rec

            frame_count += 1
            if limit and frame_count >= limit:
                break
    finally:
        cap.release()

    print(f"\nReplayed frames: {frame_count}")

    # ---------------------------------------------------------------- fidelity
    mism = []
    for fr, state, tid in replay:
        if fr not in log_map:
            mism.append((fr, "missing_in_log", state, str(tid) if tid is not None else ""))
            continue
        l_state, l_tid = log_map[fr]
        if state != l_state or str(tid) != l_tid:
            mism.append((fr, "diff", state, str(tid) if tid is not None else "", l_state, l_tid))

    print("\n=== REPLAY FAITHFULNESS vs logs/tracking_log_20261026_023427.csv ===")
    print(f"  frames replayed            : {len(replay)}")
    print(f"  frames in log              : {len(log_map)}")
    print(f"  state/target_id mismatches : {len(mism)}")
    if mism:
        for m in mism[:20]:
            print("   MISMATCH", m)
        print("  STOP: replay is not faithful; no analysis output written.")
        sys.exit(2)
    print("  RESULT: identical on every frame (state AND target_track_id)")

    if args.verify_only:
        print("\n--verify-only: stopping before analysis (nothing written).")
        print(f"anchor gallery size at frame {ANCHOR_SNAPSHOT_FRAME}: "
              f"{0 if anchor_188 is None else len(anchor_188)}")
        return

    # ------------------------------------------------------------ task 2 table
    print("\n=== TASK 2: PER-LABELED-FRAME BOX GEOMETRY ===")
    hdr = ("frame", "view", "status", "state", "drawn", "iou_drawn", "best_iou",
           "best_id", "best_tier", "best_sim_all", "best_sim_anchor", "any_iou_ge_0.3")
    print("  " + " ".join(f"{h:>14s}" for h in hdr))
    rows_out = []
    for fr in sorted(recs):
        r = recs[fr]
        if r["status"] == "box":
            best = r["best"]
            b_iou = best[1] if best else 0.0
            d = best[3] if best else None
            row = [fr, r["view"], r["status"], r["state"], int(r["drawn"]),
                   "" if r["drawn_iou"] is None else f"{r['drawn_iou']:.4f}",
                   f"{b_iou:.4f}",
                   "" if best is None else best[0],
                   "" if best is None else best[2],
                   "" if not d else f"{d['sim_all']:.4f}",
                   "" if not d else f"{d['sim_anchor']:.4f}",
                   int(b_iou >= 0.30)]
        else:
            row = [fr, r["view"], r["status"], r["state"], int(r["drawn"]),
                   "", "", "", "", "", "", ""]
        rows_out.append(row)
        print("  " + " ".join(f"{str(v):>14s}" for v in row))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows_out)
    print(f"\n  written: {OUT_CSV} ({len(rows_out)} rows)")

    # ------------------------------------------------------------ task 2 stats
    box_rows = [recs[f] for f in sorted(recs) if recs[f]["status"] == "box"]
    drawn_rows = [r for r in box_rows if r["drawn"]]
    print("\n--- (b) drawn-box IoU against the hand box (status=box frames) ---")
    print(f"  hand-box frames: {len(box_rows)}   frames with a drawn green box: {len(drawn_rows)}")
    if drawn_rows:
        ious = [r["drawn_iou"] for r in drawn_rows]
        print(f"  IoU count={len(ious)} min={min(ious):.3f} "
              f"p50/p90/p95/p99/max={fmt(pct(ious) + [max(ious)], 3)}")
        print(f"  IoU>=0.5: {sum(1 for v in ious if v >= 0.5)} | "
              f"IoU>=0.3: {sum(1 for v in ious if v >= 0.3)} | "
              f"IoU<0.1: {sum(1 for v in ious if v < 0.1)} | IoU==0: {sum(1 for v in ious if v == 0)}")
        print("  POSSIBLE false-lock candidates (drawn box IoU<0.1 to hand box):")
        none = True
        for r in drawn_rows:
            if r["drawn_iou"] < 0.1:
                none = False
                bb = r["drawn_box"]
                print(f"    frame {r['frame']:5d} {r['view']:<5s} state={r['state']:<12s} "
                      f"IoU={r['drawn_iou']:.4f} drawn={tuple(round(v,1) for v in bb)} "
                      f"hand={r['hand_box']}")
        if none:
            print("    (none)")
        low = [r for r in drawn_rows if r["drawn_iou"] < 0.3]
        print(f"  drawn-box IoU<0.3 count: {len(low)}")
        for r in low:
            print(f"    frame {r['frame']:5d} IoU={r['drawn_iou']:.3f}")

    print("\n--- (a) target visible (status=box) but no green box drawn ---")
    vis_nodraw = [r for r in box_rows if not r["drawn"]]
    print(f"  such frames: {len(vis_nodraw)}")
    for r in vis_nodraw:
        best = r["best"]
        print(f"    frame {r['frame']:5d} {r['view']:<5s} state={r['state']:<12s} "
              f"best_iou={best[1]:.3f} id={best[0]} tier={best[2]} "
              f"sim_all={'' if not best[3] else round(best[3]['sim_all'],4)} "
              f"sim_anchor={'' if not best[3] else round(best[3]['sim_anchor'],4)} "
              f"detected(>=0.3)={best[1] >= 0.3}")
    det = [r for r in vis_nodraw if r["best"] and r["best"][1] >= 0.3]
    print(f"  of which DETECTED by YOLO/tracker at IoU>=0.3: {len(det)}/{len(vis_nodraw)}")
    if det:
        print("  tier / sim_anchor of those detections:")
        for r in det:
            d = r["best"][3]
            print(f"    frame {r['frame']:5d} id={r['best'][0]} tier={r['best'][2]} "
                  f"sim_all={'' if not d else round(d['sim_all'],4)} "
                  f"sim_anchor={'' if not d else round(d['sim_anchor'],4)}")

    print("\n--- (c) hand label not_visible but a green box is drawn ---")
    nv = [recs[f] for f in sorted(recs) if recs[f]["status"] != "box"]
    nv_drawn = [r for r in nv if r["drawn"]]
    print(f"  not_visible frames: {len(nv)}  with drawn box: {len(nv_drawn)}")
    for r in nv_drawn:
        print(f"    frame {r['frame']:5d} {r['view']:<5s} state={r['state']:<12s} "
              f"drawn={tuple(round(v,1) for v in r['drawn_box'])} "
              f"target_id={r['target_track_id']}")

    # ---------------------------------------------- distribution per view (task2)
    print("\n--- best tracker-box IoU to the hand box, by view ---")
    for view in ("front", "back", "side"):
        vals = [r["best"][1] for r in box_rows if r["view"] == view and r["best"]]
        if vals:
            print(f"  {view:<5s} n={len(vals):3d} min={min(vals):.3f} "
                  f"p50/p90/p95/p99/max={fmt(pct(vals) + [max(vals)],3)} "
                  f"| >=0.3: {sum(1 for v in vals if v >= 0.3)}/{len(vals)}")

    # ------------------------------------------------------------ task 3
    print("\n=== TASK 3: EMBEDDINGS FROM HAND BOXES (tracker-independent) ===")
    frs = sorted(hand_embs)
    views = {f: recs[f]["view"] for f in frs}
    print(f"  hand-box embeddings: {len(frs)}")
    print(f"  anchors at frame {ANCHOR_SNAPSHOT_FRAME}: "
          f"{0 if anchor_188 is None else len(anchor_188)}")

    same_snap = all(len(recs[f]["anchor_copy"]) == len(anchor_188 or []) and
                    all(np.allclose(a, b) for a, b in zip(recs[f]["anchor_copy"], anchor_188 or []))
                    for f in frs)
    print(f"  anchor gallery at every labeled frame identical to the frame-188 snapshot: {same_snap}")

    def pair_stats(idx_a, idx_b, min_gap=0):
        out = []
        for i, fa in enumerate(idx_a):
            for fb in idx_b:
                if fa == fb:
                    continue
                if min_gap and abs(fa - fb) < min_gap:
                    continue
                out.append(float(np.dot(hand_embs[fa], hand_embs[fb])))
        return out

    by_view = {v: [f for f in frs if views[f] == v] for v in ("front", "back", "side")}
    print(f"  per view: " + ", ".join(f"{v}={len(by_view[v])}" for v in by_view))

    groups = [
        ("front-front", by_view["front"], by_view["front"]),
        ("back-back", by_view["back"], by_view["back"]),
        ("side-side", by_view["side"], by_view["side"]),
    ]
    for name, a, b in groups:
        for gap in (0, 20):
            vals = pair_stats(a, b, gap)
            if vals:
                print(f"  {name:<12s} gap>={gap:<2d} n={len(vals):3d} "
                      f"min={min(vals):.3f} p50/p90/p95/max={fmt(pct(vals, (50,90,95)) + [max(vals)],3)}")
            else:
                print(f"  {name:<12s} gap>={gap:<2d} n=0")

    # cross-view: every direction, both filters
    cross = [
        ("front-back", by_view["front"], by_view["back"]),
        ("front-side", by_view["front"], by_view["side"]),
        ("back-side", by_view["back"], by_view["side"]),
    ]
    for name, a, b in cross:
        for gap in (0, 20):
            vals = pair_stats(a, b, gap)
            if vals:
                print(f"  {name:<12s} gap>={gap:<2d} n={len(vals):3d} "
                      f"min={min(vals):.3f} p50/p90/p95/max={fmt(pct(vals, (50,90,95)) + [max(vals)],3)}")
            else:
                print(f"  {name:<12s} gap>={gap:<2d} n=0")

    # anchor scores of true-target hand boxes, per view
    print("\n--- sim_anchor (top-3 mean vs 10 anchors) of TRUE-target hand boxes ---")
    target_scores = {}
    for v in ("front", "back", "side"):
        vals = [recs[f]["hand_sim"]["sim_anchor"] for f in by_view[v] if "hand_sim" in recs[f]]
        target_scores[v] = vals
        if vals:
            print(f"  {v:<5s} n={len(vals):3d} min={min(vals):.3f} "
                  f"p50/p90/p95/max={fmt(pct(vals, (50,90,95)) + [max(vals)],3)}")
    all_vals = [x for v in target_scores.values() for x in v]
    if all_vals:
        print(f"  ALL   n={len(all_vals):3d} min={min(all_vals):.3f} "
              f"p50/p90/p95/max={fmt(pct(all_vals, (50,90,95)) + [max(all_vals)],3)}")
    for floor_name, thr in (("T_ANCHOR_FLOOR", 0.60), ("T_ACCEPT_LONG", 0.78)):
        bs = [x for v in ("back", "side") for x in target_scores[v]]
        if bs:
            n_above = sum(1 for x in bs if x > thr)
            print(f"  {floor_name}={thr:.2f}: TRUE-target back+side above = {n_above}/{len(bs)} "
                  f"({100.0*n_above/len(bs):.1f}%)  [n={len(bs)}]")
        fs = target_scores["front"]
        if fs:
            n_above = sum(1 for x in fs if x > thr)
            print(f"  {floor_name}={thr:.2f}: TRUE-target front above = {n_above}/{len(fs)} "
                  f"({100.0*n_above/len(fs):.1f}%)  [n={len(fs)}]")

    # different-person reference: other boxes in the SAME labeled frames
    print("\n--- different-person reference A: other tracker boxes in the same labeled frames "
          "(IoU<0.1 to the hand box) ---")
    diff_same = []
    for f in frs:
        r = recs[f]
        hb = r["hand_box"]
        for tid, box, tier, d in r["tracks"]:
            if d is None:
                continue
            if compute_iou(hb, box) < 0.1:
                diff_same.append((f, tid, d["sim_anchor"], tier, compute_iou(hb, box)))
    if diff_same:
        vals = [x[2] for x in diff_same]
        print(f"  count={len(vals)} (frames={len(set(x[0] for x in diff_same))}) "
              f"min={min(vals):.3f} p50/p90/p95/p99/max={fmt(pct(vals)+[max(vals)],3)}")
        print(f"  per-box values: " + ", ".join(f"{x[0]}:{x[1]}={x[2]:.3f}" for x in diff_same))
    else:
        print("  count=0")

    print(f"\n--- different-person reference B: all scored people in frames "
          f"{ABSENT_RANGE[0]}-{ABSENT_RANGE[1]} (GT: target absent) ---")
    if absent_scores:
        vals = [x[2] for x in absent_scores]
        print(f"  boxes scored={len(vals)} distinct tracks={len(set(x[1] for x in absent_scores))} "
              f"frames={len(set(x[0] for x in absent_scores))}")
        print(f"  sim_anchor min={min(vals):.3f} p50/p90/p95/p99/max={fmt(pct(vals)+[max(vals)],3)}")
        vals_all = [x[3] for x in absent_scores]
        print(f"  sim_all    min={min(vals_all):.3f} p50/p90/p95/p99/max={fmt(pct(vals_all)+[max(vals_all)],3)}")
        top = sorted(absent_scores, key=lambda x: -x[2])[:10]
        print("  top-10 sim_anchor boxes in that range:")
        for fr, tid, sa, sall, tier, area in top:
            print(f"    frame {fr} id={tid} sim_anchor={sa:.3f} sim_all={sall:.3f} tier={tier} area={int(area)}")
        tiers = {}
        for x in absent_scores:
            tiers[x[4]] = tiers.get(x[4], 0) + 1
        print(f"  tiers: {tiers}")
    else:
        print("  no scored boxes collected")

    print("\n" + "=" * 78)
    print("DONE. Raw CSV:", OUT_CSV)
    print("=" * 78)


if __name__ == "__main__":
    main()
