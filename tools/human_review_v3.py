#!/usr/bin/env python3
"""Gate A human review tool v3 — blind source review + identity review.

Phase A (blind source review) shows ONLY the source frame, frame index/time and
the reference identity description. It never reads or displays tracking state,
similarity, confidence, predicted boxes or CONFIRMED labels.

Phase B (identity review) runs only after Phase A is locked. It shows the output
box on the SOURCE frame but masks the output HUD and label text, so only the
selected person is visible — not the algorithm's scores or state.

All writes are versioned/atomic and never overwrite original evidence.

Commands:
  queue     build a deterministic blind queue (frame/order only)
  review-a  label target presence on source frames
  lock-a    seal Phase A labels (hash) and enable Phase B
  review-b  label identity of the selected box
  progress  per-phase progress
  export    emit evaluate.py-compatible identity review CSV + summary
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROTOCOL_VERSION = "gate_a_human_review_v3_20261107"
PHASE_A_LABELS = ("present", "absent", "partial", "unknown", "skipped")
PHASE_B_LABELS = {
    "c": ("correct", True),
    "w": ("wrong", True),
    "u": ("uncertain", True),
    "n": ("none", False),
    "r": ("none", None),  # not reviewed / excluded
}
FORBIDDEN_IN_BLIND_QUEUE = {"state", "valid", "tracker_id", "target_track_id", "sim", "similarity",
                            "confidence", "target_score", "identity_state", "identity_confirmed"}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Deterministic blind queue
# ---------------------------------------------------------------------------

def _log_frames(log_rows):
    return {int(r["frame"]) for r in log_rows}


def build_queue(log_rows=None, gt_intervals=None, n=200, seed=20261107, max_frame=None):
    """Build a deterministic, blind queue. Predictions are NOT emitted.

    Returns (frames_in_order, manifest). The returned frames are shuffled with
    `seed`; strata are recorded only in the manifest (not in the reviewer CSV).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    candidates = set()
    strata = {}
    if log_rows:
        rows = sorted(log_rows, key=lambda r: int(r["frame"]))
        frames = [int(r["frame"]) for r in rows]
        candidates.update(frames)
        prev_state = None
        for i, r in enumerate(rows):
            f = int(r["frame"])
            state = r["state"]
            valid = str(r["valid"])
            if state == "LOST" and prev_state != "LOST":
                strata.setdefault("event_boundary", []).append(f)
            if prev_state == "LOST" and state == "TRACKING":
                strata.setdefault("event_boundary", []).append(f)
            if state == "TRACKING" and valid == "0":
                strata.setdefault("no_output", []).append(f)
            prev_state = state
        for i, r in enumerate(rows):
            if r["state"] == "TRACKING" and str(r["valid"]) == "1":
                strata.setdefault("first_valid_output", []).append(int(r["frame"]))
                break
    if gt_intervals:
        for item in gt_intervals:
            strata.setdefault("boundaries", []).extend(
                [item["start_frame"], item["end_frame"]])
            if item["target_visible"] == 2:
                strata.setdefault("occlusion", []).extend(
                    range(item["start_frame"], item["end_frame"] + 1))
    if max_frame is None:
        max_frame = max(candidates) if candidates else 0
    all_frames = list(range(0, max_frame + 1))

    chosen = set()
    per_stratum = max(1, n // max(1, len(strata) + 1))
    for name in sorted(strata):
        pool = sorted(set(f for f in strata[name] if 0 <= f <= max_frame))
        chosen.update(pool[:per_stratum])
    if len(chosen) < n:
        baseline_pool = [f for f in all_frames if f not in chosen]
        rng = random.Random(seed)
        rng.shuffle(baseline_pool)
        chosen.update(baseline_pool[:n - len(chosen)])
    chosen = sorted(chosen)[:n]
    random.Random(seed).shuffle(chosen)
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "seed": seed,
        "requested": n,
        "selected": len(chosen),
        "strata": {k: sorted(set(v)) for k, v in sorted(strata.items())},
        "blind_note": ("blind queue contains frame/order only; strata are recorded here and "
                       "must not be opened before Phase A is locked"),
        "created_at_utc": now_utc(),
    }
    return chosen, manifest


def write_queue(session_dir, frames, manifest):
    os.makedirs(session_dir, exist_ok=True)
    queue_path = os.path.join(session_dir, "blind_queue.csv")
    if os.path.exists(queue_path):
        raise ValueError(f"queue already exists (not overwriting): {queue_path}")
    tmp = queue_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "frame"])
        for i, frame in enumerate(frames):
            w.writerow([i, frame])
    os.replace(tmp, queue_path)
    atomic_write_json(os.path.join(session_dir, "queue_manifest.json"), manifest)
    return queue_path


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class ReviewSession:
    def __init__(self, session_dir: str):
        self.dir = os.path.abspath(session_dir)
        self.meta_path = os.path.join(self.dir, "session_meta.json")
        self.queue_path = os.path.join(self.dir, "blind_queue.csv")
        self.phase_a_path = os.path.join(self.dir, "phase_a_labels.jsonl")
        self.lock_path = os.path.join(self.dir, "phase_a_locked.json")
        self.phase_b_path = os.path.join(self.dir, "phase_b_labels.jsonl")

    def init(self, video_path, reviewer, reference_identity, queue_frames, manifest):
        if os.path.exists(self.meta_path):
            raise ValueError(f"session already initialized: {self.dir}")
        os.makedirs(self.dir, exist_ok=True)
        write_queue(self.dir, queue_frames, manifest)
        atomic_write_json(self.meta_path, {
            "protocol": PROTOCOL_VERSION,
            "session_id": os.path.basename(self.dir),
            "reviewer": reviewer,
            "reference_identity": reference_identity,
            "video_path": os.path.abspath(video_path),
            "video_sha256": sha256_file(video_path),
            "queue_sha256": sha256_file(self.queue_path),
            "created_at_utc": now_utc(),
            "phase": "A",
        })

    def meta(self):
        if not os.path.exists(self.meta_path):
            raise ValueError(f"session not initialized: {self.dir}")
        with open(self.meta_path, encoding="utf-8") as f:
            return json.load(f)

    def queue(self):
        rows = read_csv(self.queue_path)
        return [int(r["frame"]) for r in rows]

    def labeled_a(self):
        return {int(r["frame"]): r for r in read_jsonl(self.phase_a_path)}

    def labeled_b(self):
        return {int(r["frame"]): r for r in read_jsonl(self.phase_b_path)}

    def is_locked(self) -> bool:
        return os.path.exists(self.lock_path)

    def pending_a(self):
        """Frames still needing Phase A (resume-friendly, queue order)."""
        done = self.labeled_a()
        return [f for f in self.queue() if f not in done]

    def pending_b(self):
        """Phase B queue: labelled-present/absent/partial and not yet judged."""
        self.require_locked()
        a = self.labeled_a()
        done = self.labeled_b()
        return [f for f in sorted(a)
                if a[f]["label"] in ("present", "absent", "partial") and f not in done]

    def require_locked(self):
        if not self.is_locked():
            raise ValueError("Phase A is not locked; run `lock-a` first")

    def append_a(self, frame, label, reviewer, note=""):
        self.meta()
        if self.is_locked():
            raise ValueError("Phase A is locked; labels cannot be changed")
        if label not in PHASE_A_LABELS:
            raise ValueError(f"invalid Phase A label: {label!r}")
        if frame not in self.queue():
            raise ValueError(f"frame {frame} is not in the blind queue")
        if frame in self.labeled_a():
            raise ValueError(f"frame {frame} already has a Phase A label")
        append_jsonl(self.phase_a_path, {
            "frame": int(frame), "label": label, "reviewer": reviewer,
            "timestamp_nominal_utc": now_utc(), "note": note,
        })

    def append_b(self, frame, label_key, reviewer, has_box: bool):
        self.require_locked()
        if label_key not in PHASE_B_LABELS:
            raise ValueError(f"invalid Phase B label key: {label_key!r}")
        identity, drawn_expected = PHASE_B_LABELS[label_key]
        if drawn_expected is None:
            raise ValueError("'not reviewed' is not recorded as a label")
        if has_box != drawn_expected:
            raise ValueError(
                f"label {label_key!r} contradicts box presence ({has_box}); "
                "use 'n' only when no box exists and c/w/u only when a box exists")
        if frame in self.labeled_b():
            raise ValueError(f"frame {frame} already has a Phase B label")
        append_jsonl(self.phase_b_path, {
            "frame": int(frame), "identity": identity, "drawn": bool(has_box),
            "reviewer": reviewer, "timestamp_nominal_utc": now_utc(),
        })

    def lock_a(self, reviewer):
        self.meta()
        if self.is_locked():
            raise ValueError("Phase A is already locked (not overwriting)")
        queue = set(self.queue())
        labels = self.labeled_a()
        unknown = sorted(queue - set(labels))
        counts = {k: 0 for k in PHASE_A_LABELS}
        for r in labels.values():
            counts[r["label"]] += 1
        atomic_write_json(self.lock_path, {
            "protocol": PROTOCOL_VERSION,
            "locked_at_utc": now_utc(),
            "reviewer": reviewer,
            "queue_frames": len(queue),
            "labeled_frames": len(labels),
            "unlabeled_frames": unknown,
            "label_counts": counts,
            "labels_sha256": sha256_file(self.phase_a_path) if os.path.exists(self.phase_a_path) else None,
            "video_sha256": self.meta()["video_sha256"],
        })

    def progress(self):
        queue = self.queue()
        a = self.labeled_a()
        b = self.labeled_b()
        return {"queue": len(queue), "phase_a_labeled": len(a),
                "phase_a_remaining": len(set(queue) - set(a)),
                "phase_a_locked": self.is_locked(),
                "phase_b_labeled": len(b)}

    def verify_lock_integrity(self):
        self.require_locked()
        with open(self.lock_path, encoding="utf-8") as f:
            lock = json.load(f)
        current = sha256_file(self.phase_a_path) if os.path.exists(self.phase_a_path) else None
        if lock["labels_sha256"] != current:
            raise ValueError("Phase A labels changed after lock; refusing Phase B")
        return lock


# ---------------------------------------------------------------------------
# Frame extraction / display
# ---------------------------------------------------------------------------

def _frame_to_png(video_path, frame, dest):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame} from {video_path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp.png"
    cv2.imwrite(tmp, img)
    os.replace(tmp, dest)
    return img


def extract_output_box(frame_bgr, min_w=30, min_h=60):
    """Detect the drawn tall green target rectangle in an output frame.

    Person boxes are tall (height >= 35% of the frame, h/w >= 1.4); this rejects
    wide shop-signs/UI bars. Only geometry is returned, never labels/scores.
    """
    H, W = frame_bgr.shape[:2]
    b, g, r = cv2.split(frame_bgr)
    mask = ((g > 150) & (r < 100) & (b < 100)).astype("uint8") * 255
    mask = cv2.dilate(mask, None, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= max(min_h, int(0.35 * H)) and h >= 1.4 * w:
            if best is None or h > best[3]:
                best = (x, y, w, h)
    return best


def last_decodable_frame(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for candidate in range(meta - 1, max(meta - 4, -1), -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, _ = cap.read()
        if ok:
            cap.release()
            return candidate
    cap.release()
    raise ValueError("video has no decodable frame")


def phase_b_display(source_frame, output_frame):
    """Render source frame + selected box only; mask HUD and label bands."""
    display = source_frame.copy()
    box = extract_output_box(output_frame) if output_frame is not None else None
    if box:
        x, y, w, h = box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 3)
        display = cv2.putText(display, "SELECTED", (x + 4, max(20, y - 8)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return display, box


def _prompt_phase_a(frame, total, time_s, ref, image_path):
    print(f"\n[{frame}/{total}] frame={frame} time={time_s:.2f}s")
    print(f"Reference target: {ref}")
    print(f"Source frame: {image_path}")
    return input("Target presence? p=present a=absent o=partial u=unknown s=skip q=quit: ").strip().lower()


def _prompt_phase_b(frame, has_box):
    print(f"\nframe={frame}  " + ("SELECTED box shown (red)" if has_box else "NO OUTPUT BOX"))
    return input("[c]orrect [w]rong person [u]ncertain [n]o output [r]eview later [q]uit: ").strip().lower()


# ---------------------------------------------------------------------------
# Export to evaluate.py
# ---------------------------------------------------------------------------

def export_review_csv(session: ReviewSession, log_path: str, out_path: str):
    """Join locked Phase A + Phase B into evaluate.py identity review schema."""
    session.verify_lock_integrity()
    a = session.labeled_a()
    b = session.labeled_b()
    log_rows = {int(r["frame"]): r for r in read_csv(log_path)}
    out_rows = []
    skipping = {"skipped", "unknown"}
    bad = [f for f in sorted(set(a) & set(b))
           if a[f]["label"] == "absent" and b[f]["identity"] == "correct"]
    if bad:
        raise ValueError(f"absent frames cannot be 'correct': {bad[:20]}")
    for frame in sorted(set(a) & set(b)):
        ra, rb = a[frame], b[frame]
        if ra["label"] in skipping:
            continue
        if frame not in log_rows:
            raise ValueError(f"review frame {frame} missing from log")
        lr = log_rows[frame]
        out_rows.append({
            "frame": frame,
            "state": lr["state"],
            "presence": ra["label"],
            "drawn": int(bool(rb["drawn"])),
            "identity": rb["identity"],
        })
    if os.path.exists(out_path):
        raise ValueError(f"export already exists (not overwriting): {out_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "state", "presence", "drawn", "identity"])
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp, out_path)
    return out_rows


def summary(session: ReviewSession):
    session.verify_lock_integrity()
    a = session.labeled_a()
    b = session.labeled_b()
    a_counts = {k: 0 for k in PHASE_A_LABELS}
    for r in a.values():
        a_counts[r["label"]] += 1
    b_counts = {}
    for r in b.values():
        key = r["identity"] if r["drawn"] else "no_output"
        b_counts[key] = b_counts.get(key, 0) + 1
    return {
        "protocol": PROTOCOL_VERSION,
        "session": session.dir,
        "queue": len(session.queue()),
        "phase_a": {"labeled": len(a), "counts": a_counts},
        "phase_b": {"labeled": len(b), "counts": b_counts},
        "joined_frames": len(set(a) & set(b)),
        "provenance": {
            "phase_a_sha256": sha256_file(session.phase_a_path),
            "phase_b_sha256": sha256_file(session.phase_b_path),
            "locked_sha256": json.load(open(session.lock_path))["labels_sha256"],
            "video_sha256": session.meta()["video_sha256"],
        },
        "note": "Descriptive review counts only. NOT an accuracy estimate.",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_queue(args):
    log_rows = read_csv(args.log) if args.log else None
    gt = None
    if args.gt:
        import evaluate as ev
        gt = ev.parse_ground_truth(args.gt)
    last = last_decodable_frame(args.video)
    frames, manifest = build_queue(log_rows, gt, args.n, args.seed, last)
    manifest["video_sha256"] = sha256_file(args.video)
    session = ReviewSession(args.session_dir)
    session.init(args.video, args.reviewer, args.reference, frames, manifest)
    print(f"queue: {session.queue_path}")
    print(f"session: {session.dir}  frames={len(frames)}")


def _cmd_review_a(args):
    session = ReviewSession(args.session_dir)
    meta = session.meta()
    if session.is_locked():
        raise ValueError("Phase A is locked")
    frames = session.pending_a()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    for i, frame in enumerate(frames):
        img_path = os.path.join(session.dir, "frames", f"frame_{frame:06d}.png")
        if not os.path.exists(img_path):
            _frame_to_png(args.video, frame, img_path)
        try:
            if args.gui:
                img = cv2.imread(img_path)
                cv2.imshow("Phase A - source only", img)
                key = cv2.waitKey(0) & 0xFF
                label = chr(key) if key < 128 else ""
                cv2.destroyAllWindows()
            else:
                label = _prompt_phase_a(frame, len(frames), frame / fps,
                                        meta["reference_identity"], img_path)
        except (EOFError, KeyboardInterrupt):
            print("\nstopped; resume with the same session-dir")
            break
        if label == "q":
            break
        mapping = {"p": "present", "a": "absent", "o": "partial", "u": "unknown", "s": "skipped"}
        if label not in mapping:
            print("invalid label; skipped")
            continue
        session.append_a(frame, mapping[label], args.reviewer)
    print(json.dumps(session.progress(), ensure_ascii=False, indent=2))


def _cmd_lock_a(args):
    session = ReviewSession(args.session_dir)
    session.lock_a(args.reviewer)
    print(f"locked: {session.lock_path}")


def _cmd_review_b(args):
    session = ReviewSession(args.session_dir)
    session.verify_lock_integrity()
    meta = session.meta()
    a = session.labeled_a()
    frames = session.pending_b()
    log_rows = {int(r["frame"]): r for r in read_csv(args.log)}
    src = cv2.VideoCapture(args.video)
    out = cv2.VideoCapture(args.output_video)
    if not src.isOpened() or not out.isOpened():
        raise ValueError("cannot open source/output video")
    src_frames = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    out_frames = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_frames != out_frames:
        raise ValueError(f"source/output frame count mismatch: {src_frames} vs {out_frames}")
    for frame in frames:
        src.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok1, sf = src.read()
        out.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok2, of = out.read()
        if not ok1 or not ok2:
            raise ValueError(f"cannot read frame {frame}")
        display, box = phase_b_display(sf, of)
        dest = os.path.join(session.dir, "phase_b_frames", f"frame_{frame:06d}.png")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cv2.imwrite(dest + ".tmp.png", display)
        os.replace(dest + ".tmp.png", dest)
        print(f"\nframe={frame} display={dest}")
        try:
            if args.gui:
                cv2.imshow("Phase B - box only (HUD masked)", display)
                key = cv2.waitKey(0) & 0xFF
                label = chr(key) if key < 128 else ""
                cv2.destroyAllWindows()
            else:
                label = _prompt_phase_b(frame, box is not None)
        except (EOFError, KeyboardInterrupt):
            break
        if label == "q":
            break
        if label == "r":
            continue
        if label not in PHASE_B_LABELS:
            print("invalid label; skipped")
            continue
        try:
            session.append_b(frame, label, args.reviewer, has_box=box is not None)
        except ValueError as exc:
            print(f"rejected: {exc}")
    print(json.dumps(session.progress(), ensure_ascii=False, indent=2))


def _cmd_progress(args):
    print(json.dumps(ReviewSession(args.session_dir).progress(), ensure_ascii=False, indent=2))


def _cmd_export(args):
    session = ReviewSession(args.session_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = os.path.join(session.dir, "export", f"identity_reviews_{stamp}.csv")
    rows = export_review_csv(session, args.log, out_csv)
    sum_path = os.path.join(session.dir, "export", f"review_summary_{stamp}.json")
    atomic_write_json(sum_path, summary(session))
    print(json.dumps({"csv": out_csv, "rows": len(rows), "summary": sum_path,
                      "csv_sha256": sha256_file(out_csv)}, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gate A blind human review tool v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("queue", help="create a deterministic blind queue/session")
    q.add_argument("--video", required=True)
    q.add_argument("--log", default=None)
    q.add_argument("--gt", default=None)
    q.add_argument("--session-dir", required=True)
    q.add_argument("--reviewer", required=True)
    q.add_argument("--reference", required=True, help="reference identity description")
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--seed", type=int, default=20261107)
    q.set_defaults(func=_cmd_queue)

    ra = sub.add_parser("review-a", help="blind source review")
    ra.add_argument("--session-dir", required=True)
    ra.add_argument("--video", required=True)
    ra.add_argument("--reviewer", required=True)
    ra.add_argument("--gui", action="store_true")
    ra.set_defaults(func=_cmd_review_a)

    la = sub.add_parser("lock-a", help="seal Phase A labels")
    la.add_argument("--session-dir", required=True)
    la.add_argument("--reviewer", required=True)
    la.set_defaults(func=_cmd_lock_a)

    rb = sub.add_parser("review-b", help="identity review of the selected box")
    rb.add_argument("--session-dir", required=True)
    rb.add_argument("--video", required=True, help="source video")
    rb.add_argument("--output-video", required=True, help="annotated output video")
    rb.add_argument("--log", required=True)
    rb.add_argument("--reviewer", required=True)
    rb.add_argument("--gui", action="store_true")
    rb.set_defaults(func=_cmd_review_b)

    pr = sub.add_parser("progress", help="show review progress")
    pr.add_argument("--session-dir", required=True)
    pr.set_defaults(func=_cmd_progress)

    ex = sub.add_parser("export", help="export evaluate.py-compatible review CSV")
    ex.add_argument("--session-dir", required=True)
    ex.add_argument("--log", required=True)
    ex.set_defaults(func=_cmd_export)

    args = p.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"[REVIEW ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
