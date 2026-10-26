#!/usr/bin/env python3
"""Eye-labeling tool for one target person in a video.

Labels are boxes drawn by hand. No tracker ID is stored or read.

Controls while a frame is shown:
  left mouse drag   draw the target box
  Enter or Space    accept the box and go to the next frame
  r                 clear the current box
  s                 target is not visible in this frame
  n                 skip this frame (no row is written)
  q or Esc          save and quit

Output CSV columns:
  frame,status,x1,y1,x2,y2,view
  status = "box" or "not_visible"; box coordinates are in original video pixels.
"""

import argparse
import csv
import os
import sys

import cv2

DEFAULT_GT = "ground_truth/5_manual.csv"
DEFAULT_OUT = "ground_truth/labels_5_eye.csv"
LABEL_NAME = "labels_5_eye.csv"
LABEL_HEADER = ["frame", "status", "x1", "y1", "x2", "y2", "view"]
MAX_DISPLAY_W = 1600
MAX_DISPLAY_H = 900


def load_intervals(path):
    intervals = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"start_frame", "end_frame", "target_visible", "view"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f"GT file {path} must have columns: {sorted(required)}")
        for row in reader:
            intervals.append({
                "start": int(row["start_frame"]),
                "end": int(row["end_frame"]),
                "visible": int(row["target_visible"]),
                "view": (row["view"] or "").strip().lower(),
            })
    intervals.sort(key=lambda item: item["start"])
    return intervals


def build_plan(intervals, front_stride, back_side_stride):
    plan = []
    for iv in intervals:
        if iv["view"] == "front" and iv["start"] == 0:
            stride = front_stride
            rule = f"front 0-{iv['end']}, stride {stride}"
        elif iv["view"] in ("back", "side"):
            stride = back_side_stride
            rule = f"{iv['view']}, stride {stride}"
        else:
            stride = None
            rule = "skipped (not the front 0-904 interval and not back/side)"
        frames = list(range(iv["start"], iv["end"] + 1, stride)) if stride else []
        plan.append({"interval": iv, "stride": stride, "rule": rule, "frames": frames})
    return plan


def print_plan(plan):
    total = 0
    for entry in plan:
        iv = entry["interval"]
        print(f"[{iv['start']:4d}-{iv['end']:4d}] visible={iv['visible']} "
              f"view={iv['view']:<5s} {entry['rule']}")
        if entry["frames"]:
            print(f"  frames ({len(entry['frames'])}): "
                  f"{','.join(str(f) for f in entry['frames'])}")
        else:
            print("  frames (0): -")
        total += len(entry["frames"])
    print(f"TOTAL SAMPLED FRAMES: {total}")


def sample_view_map(plan):
    view_by_frame = {}
    for entry in plan:
        for frame in entry["frames"]:
            view_by_frame.setdefault(frame, entry["interval"]["view"])
    return view_by_frame


def check_output_path(out_path, append):
    out_abs = os.path.abspath(out_path)
    gt_dir = os.path.abspath("ground_truth")
    if not os.path.exists(out_abs):
        return
    if not append:
        raise SystemExit(
            f"Refusing to overwrite existing file: {out_path}. "
            "Use --append to add missing rows, or choose a new --out."
        )
    if os.path.dirname(out_abs) == gt_dir and os.path.basename(out_abs) != LABEL_NAME:
        raise SystemExit(
            f"Refusing to append to an existing ground-truth file: {out_path}. "
            f"Only {LABEL_NAME} may be appended."
        )


def load_existing_frames(out_path):
    frames = set()
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return frames
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != LABEL_HEADER:
            raise SystemExit(
                f"Existing file {out_path} has header {header}, expected {LABEL_HEADER}."
            )
        for row in reader:
            if row:
                frames.add(int(row[0]))
    return frames


def make_mouse_callback(sel):
    def on_mouse(event, x, y, flags, param):
        x = max(0, min(x, sel["disp_w"] - 1))
        y = max(0, min(y, sel["disp_h"] - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            sel["start"] = (x, y)
            sel["end"] = (x, y)
            sel["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and sel["drawing"]:
            sel["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            sel["end"] = (x, y)
            sel["drawing"] = False
    return on_mouse


def box_in_video_coords(sel, scale, frame_w, frame_h):
    if sel["start"] is None or sel["end"] is None:
        return None
    (sx, sy), (ex, ey) = sel["start"], sel["end"]
    x1, x2 = sorted((sx / scale, ex / scale))
    y1, y2 = sorted((sy / scale, ey / scale))
    x1 = max(0, min(frame_w - 1, int(round(x1))))
    y1 = max(0, min(frame_h - 1, int(round(y1))))
    x2 = max(0, min(frame_w, int(round(x2))))
    y2 = max(0, min(frame_h, int(round(y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def write_row(writer, frame, status, box, view):
    if box is None:
        writer.writerow([frame, status, "", "", "", "", view])
    else:
        writer.writerow([frame, status, box[0], box[1], box[2], box[3], view])


def run_interactive(args, plan):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "No display detected. This tool needs a display for drawing boxes. "
            "Use --dry-run to inspect the sampling plan headlessly."
        )

    view_by_frame = sample_view_map(plan)
    ordered = sorted(view_by_frame)
    done = load_existing_frames(args.out) if args.append else set()
    if done:
        ordered = [frame for frame in ordered if frame not in done]
    if not ordered:
        print("Nothing to label: all sampled frames are already in the output file.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if ordered[-1] >= frame_count:
        raise SystemExit(
            f"Sampling frame {ordered[-1]} but the video has only {frame_count} frames."
        )

    scale = min(1.0, MAX_DISPLAY_W / frame_w, MAX_DISPLAY_H / frame_h)
    disp_w = max(1, int(round(frame_w * scale)))
    disp_h = max(1, int(round(frame_h * scale)))

    append_mode = args.append and os.path.exists(args.out) and os.path.getsize(args.out) > 0
    out_file = open(args.out, "a" if append_mode else "w", newline="", encoding="utf-8")
    writer = csv.writer(out_file)
    if not append_mode:
        writer.writerow(LABEL_HEADER)

    win = "label_target"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    sel = {"start": None, "end": None, "drawing": False,
           "disp_w": disp_w, "disp_h": disp_h}
    cv2.setMouseCallback(win, make_mouse_callback(sel))

    print(f"Video: {args.video} ({frame_w}x{frame_h}, {frame_count} frames)")
    print(f"Display scale: {scale:.4f} ({disp_w}x{disp_h})")
    print("Drag with the left mouse button. Enter/Space=accept, r=clear, "
          "s=not visible, n=skip, q=save and quit.")
    print(f"Frames to label this session: {len(ordered)}")

    written = 0
    not_visible = 0
    skipped = 0
    try:
        for position, frame in enumerate(ordered, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"Could not read frame {frame} from {args.video}.")
            base = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            sel["start"] = None
            sel["end"] = None
            sel["drawing"] = False

            while True:
                shown = base.copy()
                if sel["start"] is not None and sel["end"] is not None:
                    cv2.rectangle(shown, sel["start"], sel["end"], (0, 255, 0), 2)
                bar = f"frame {frame}  view {view_by_frame[frame]}  " \
                      f"{position}/{len(ordered)}  |  Enter=accept  r=clear  " \
                      f"s=not visible  n=skip  q=save+quit"
                cv2.rectangle(shown, (0, 0), (disp_w, 26), (0, 0, 0), -1)
                cv2.putText(shown, bar, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                cv2.imshow(win, shown)
                key = cv2.waitKey(20) & 0xFF

                if key == 255:
                    continue
                if key in (ord("q"), 27):
                    print("Save and quit.")
                    return
                if key == ord("s"):
                    write_row(writer, frame, "not_visible", None, view_by_frame[frame])
                    out_file.flush()
                    written += 1
                    not_visible += 1
                    break
                if key == ord("n"):
                    skipped += 1
                    break
                if key == ord("r"):
                    sel["start"] = None
                    sel["end"] = None
                    sel["drawing"] = False
                    continue
                if key in (13, 32):
                    box = box_in_video_coords(sel, scale, frame_w, frame_h)
                    if box is None:
                        print(f"Frame {frame}: draw a box first (or press s / n).")
                        continue
                    write_row(writer, frame, "box", box, view_by_frame[frame])
                    out_file.flush()
                    written += 1
                    break
    finally:
        cap.release()
        out_file.close()
        cv2.destroyAllWindows()
        print(f"Rows written: {written} (not_visible {not_visible}), skipped {skipped}")
        print(f"Output: {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description="Label the target person by hand. Stores boxes, never tracker IDs."
    )
    parser.add_argument("--video", default=None,
                        help="input video (required unless --dry-run)")
    parser.add_argument("--gt", default=DEFAULT_GT,
                        help=f"ground-truth intervals (default {DEFAULT_GT})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"label CSV (default {DEFAULT_OUT})")
    parser.add_argument("--front-stride", type=int, default=60,
                        help="stride for the front 0-904 interval (default 60)")
    parser.add_argument("--back-side-stride", type=int, default=20,
                        help="stride for back/side intervals (default 20)")
    parser.add_argument("--append", action="store_true",
                        help="append rows, skipping frames already labeled")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sampling plan and exit; writes nothing")
    args = parser.parse_args()

    if args.front_stride < 1 or args.back_side_stride < 1:
        raise SystemExit("Strides must be >= 1.")
    if not os.path.exists(args.gt):
        raise SystemExit(f"GT file not found: {args.gt}")
    if not args.dry_run and not args.video:
        raise SystemExit("--video is required unless --dry-run is used.")
    if not args.dry_run and not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    intervals = load_intervals(args.gt)
    plan = build_plan(intervals, args.front_stride, args.back_side_stride)

    if args.dry_run:
        print(f"GT: {args.gt}")
        print_plan(plan)
        if args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Failed to open video: {args.video}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            max_frame = max((f for p in plan for f in p["frames"]), default=-1)
            print(f"Video: {args.video} ({frame_count} frames); "
                  f"max sampled frame {max_frame}; within range: {max_frame < frame_count}")
        print("DRY RUN: no file was written, no display was opened.")
        return

    check_output_path(args.out, args.append)
    run_interactive(args, plan)


if __name__ == "__main__":
    sys.exit(main())
