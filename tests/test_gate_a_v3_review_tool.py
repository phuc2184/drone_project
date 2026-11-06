"""Tests for the blind human-review tool v3 (no human data is fabricated)."""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import human_review_v3 as hr  # noqa: E402
import evaluate as ev  # noqa: E402


def make_video(path, frames=6, size=(64, 96), color=(10, 10, 10)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 10.0, (size[1], size[0]))
    for i in range(frames):
        img = np.full((size[0], size[1], 3), color, dtype=np.uint8)
        img[5, 5] = (i, i, i)
        w.write(img)
    w.release()


def make_log(path, frames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "state", "target_track_id", "valid"])
        for i, (state, valid) in enumerate(frames):
            w.writerow([i, state, "1", valid])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "v.mp4"
        make_video(self.video, 6)
        self.log = self.dir / "l.csv"
        make_log(self.log, [("TRACKING", "1"), ("TRACKING", "1"), ("LOST", "0"),
                            ("LOST", "0"), ("TRACKING", "1"), ("TRACKING", "1")])

    def test_queue_is_deterministic_and_blind(self):
        rows = hr.read_csv(str(self.log))
        f1, m1 = hr.build_queue(rows, None, n=6, seed=42, max_frame=5)
        f2, m2 = hr.build_queue(rows, None, n=6, seed=42, max_frame=5)
        self.assertEqual(f1, f2)
        self.assertEqual(m1["strata"], m2["strata"])
        self.assertEqual(sorted(f1), sorted(set(f1)))
        session = hr.ReviewSession(str(self.dir / "s"))
        session.init(str(self.video), "rev", "reference", f1, m1)
        with open(session.queue_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, ["order", "frame"])
        for forbidden in hr.FORBIDDEN_IN_BLIND_QUEUE:
            self.assertNotIn(forbidden, header)

    def test_queue_manifest_holds_strata_not_the_queue_csv(self):
        rows = hr.read_csv(str(self.log))
        frames, manifest = hr.build_queue(rows, None, n=6, seed=1, max_frame=5)
        session = hr.ReviewSession(str(self.dir / "s2"))
        session.init(str(self.video), "rev", "reference", frames, manifest)
        with open(session.queue_path, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("LOST", text)
        self.assertNotIn("TRACKING", text)
        with open(self.dir / "s2" / "queue_manifest.json", encoding="utf-8") as f:
            self.assertIn("strata", json.load(f))

    def test_queue_cli_with_gt_runs(self):
        gt = self.dir / "gt.csv"
        with open(gt, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_frame", "end_frame", "target_visible", "view"])
            w.writerow([0, 5, 1, "front"])
        rc = hr.main(["queue", "--video", str(self.video), "--log", str(self.log),
                      "--gt", str(gt), "--session-dir", str(self.dir / "cli_s"),
                      "--reviewer", "rev", "--reference", "ref", "--n", "6", "--seed", "3"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.dir / "cli_s" / "blind_queue.csv").exists())

    def test_last_decodable_frame(self):
        self.assertEqual(hr.last_decodable_frame(str(self.video)), 5)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "v.mp4"
        make_video(self.video, 6)
        self.log = self.dir / "l.csv"
        make_log(self.log, [("TRACKING", "1")] * 6)
        frames, manifest = hr.build_queue(hr.read_csv(str(self.log)), None, 6, 7, 5)
        self.session = hr.ReviewSession(str(self.dir / "sess"))
        self.session.init(str(self.video), "rev", "denim jacket", frames, manifest)

    def test_phase_a_validation_and_duplicate_rejection(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        with self.assertRaises(ValueError):
            self.session.append_a(frame, "present", "rev")
        with self.assertRaises(ValueError):
            self.session.append_a(frame, "maybe", "rev")
        with self.assertRaises(ValueError):
            self.session.append_a(9999, "present", "rev")

    def test_resume_pending_helpers(self):
        frames = self.session.queue()
        self.session.append_a(frames[0], "present", "rev")
        self.assertEqual(self.session.pending_a(), frames[1:])
        self.session.lock_a("rev")
        pending_b = self.session.pending_b()
        self.assertIn(frames[0], pending_b)
        self.session.append_b(frames[0], "c", "rev", has_box=True)
        self.assertNotIn(frames[0], self.session.pending_b())

    def test_lock_prevents_changes_and_relock(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        self.session.lock_a("rev")
        with self.assertRaises(ValueError):
            self.session.append_a(frame, "absent", "rev")
        with self.assertRaises(ValueError):
            self.session.lock_a("rev")
        self.session.verify_lock_integrity()

    def test_tamper_after_lock_is_detected(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        self.session.lock_a("rev")
        with open(self.session.phase_a_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"frame": frame, "label": "absent"}) + "\n")
        with self.assertRaises(ValueError):
            self.session.verify_lock_integrity()

    def test_phase_b_requires_lock_and_box_consistency(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        with self.assertRaises(ValueError):
            self.session.append_b(frame, "c", "rev", has_box=True)
        self.session.lock_a("rev")
        with self.assertRaises(ValueError):
            self.session.append_b(frame, "c", "rev", has_box=False)
        self.session.append_b(frame, "c", "rev", has_box=True)
        with self.assertRaises(ValueError):
            self.session.append_b(frame, "w", "rev", has_box=True)

    def test_export_schema_is_accepted_by_evaluator(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        self.session.lock_a("rev")
        self.session.append_b(frame, "c", "rev", has_box=True)
        out = self.dir / "export.csv"
        rows = hr.export_review_csv(self.session, str(self.log), str(out))
        self.assertEqual(len(rows), 1)
        accepted = ev.load_identity_reviews(str(out))
        self.assertEqual(accepted[0]["identity"], "correct")
        self.assertTrue(accepted[0]["drawn"])
        with self.assertRaises(ValueError):
            hr.export_review_csv(self.session, str(self.log), str(out))

    def test_absent_plus_correct_is_rejected_on_export(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "absent", "rev")
        self.session.lock_a("rev")
        self.session.append_b(frame, "c", "rev", has_box=True)
        with self.assertRaises(ValueError):
            hr.export_review_csv(self.session, str(self.log), str(self.dir / "bad.csv"))

    def test_summary_is_descriptive_and_hashed(self):
        frame = self.session.queue()[0]
        self.session.append_a(frame, "present", "rev")
        self.session.lock_a("rev")
        self.session.append_b(frame, "w", "rev", has_box=True)
        s = hr.summary(self.session)
        self.assertEqual(s["phase_a"]["counts"]["present"], 1)
        self.assertEqual(s["phase_b"]["counts"]["wrong"], 1)
        self.assertIn("phase_a_sha256", s["provenance"])
        self.assertIn("NOT an accuracy estimate", s["note"])


class DisplayTests(unittest.TestCase):
    def test_box_detection_ignores_wide_green_ui(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (200, 40), (0, 255, 0), -1)   # wide green bar
        cv2.rectangle(frame, (100, 50), (160, 190), (0, 255, 0), 2)  # tall target box
        box = hr.extract_output_box(frame)
        self.assertIsNotNone(box)
        self.assertLessEqual(abs(box[0] - 100), 4)
        self.assertLessEqual(abs(box[1] - 50), 4)
        self.assertLessEqual(abs(box[2] - 60), 6)
        self.assertLessEqual(abs(box[3] - 140), 6)

    def test_phase_b_draws_box_on_source_not_output_hud(self):
        source = np.full((200, 300, 3), 7, dtype=np.uint8)
        output = np.full((200, 300, 3), 99, dtype=np.uint8)
        cv2.rectangle(output, (100, 50), (160, 190), (0, 255, 0), 3)
        cv2.putText(output, "CONFIRMED 0.99", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display, box = hr.phase_b_display(source, output)
        self.assertIsNotNone(box)
        self.assertLessEqual(abs(box[0] - 100), 4)
        self.assertLessEqual(abs(box[1] - 50), 4)
        self.assertEqual(int(display[0, 0, 0]), 7)  # source pixel, not output HUD
        self.assertEqual(int(display[199, 299, 0]), 7)

    def test_phase_b_prompt_does_not_leak_scores_or_state(self):
        with mock.patch("builtins.input", return_value="r"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                hr._prompt_phase_b(123, True)
        text = buf.getvalue()
        for forbidden in ("similarity", "CONFIRMED", "state=", "TARGET ID", "anchor"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
