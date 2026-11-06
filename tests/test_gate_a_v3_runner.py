"""End-to-end tests for the Gate A Evaluator v3 CLI runner."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gate_a_evaluator_v3 as runner  # noqa: E402


def make_video(path, frames=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    for i in range(frames):
        img = np.full((64, 96, 3), i, dtype=np.uint8)
        w.write(img)
    w.release()


def write_log(path, frames=10, state="TRACKING", valid="1"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "state", "target_track_id", "valid"])
        for i in range(frames):
            w.writerow([i, state, "1", valid])


def write_gt(path, frames=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_frame", "end_frame", "target_visible", "view"])
        w.writerow([0, frames - 1, 1, "front"])


def write_reviews(path, frames=10, presence="present", identity="uncertain", drawn="1"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "state", "presence", "drawn", "identity"])
        for i in range(frames):
            w.writerow([i, "TRACKING", presence, drawn, identity])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "v.mp4"
        self.log = self.dir / "log.csv"
        self.gt = self.dir / "gt.csv"
        self.reviews = self.dir / "reviews.csv"
        self.out = self.dir / "out"
        make_video(self.video)
        write_log(self.log)
        write_gt(self.gt)
        write_reviews(self.reviews)

    def run_main(self, *extra):
        return runner.main([
            "--video", str(self.video), "--log", str(self.log),
            "--ground-truth", str(self.gt), "--reviews", str(self.reviews),
            "--out-dir", str(self.out), "--no-decode-count", *extra,
        ])

    def test_validate_only_succeeds(self):
        self.assertEqual(self.run_main("--validate-only"), 0)
        self.assertFalse(self.out.exists())

    def test_full_run_writes_versioned_json_and_not_estimable(self):
        self.assertEqual(self.run_main(), 0)
        runs = sorted(self.out.glob("run_*"))
        self.assertEqual(len(runs), 1)
        run = runs[0]
        for name in ("gate_a_v3_metrics.json", "REPORT_VI.md", "MANIFEST_SHA256.txt"):
            self.assertTrue((run / name).exists(), name)
        with open(run / "gate_a_v3_metrics.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsNone(data["identity_metrics"]["false_lock_rate_target_present"])
        self.assertEqual(data["identity_metrics"]["overall_status"], "NOT_ESTIMABLE")
        self.assertEqual(data["scientific_validity"]["status"], "NOT_ESTABLISHED")
        self.assertEqual(data["provenance"]["log"]["sha256"], runner.sha256_file(str(self.log)))
        # second run must create a new directory, never overwrite
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(sorted(self.out.glob("run_*"))), 2)

    def test_invalid_log_frame_fails_closed_without_output(self):
        write_log(self.log)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write("999,TRACKING,1,1\n")
        self.assertEqual(self.run_main(), 2)
        self.assertFalse(self.out.exists())

    def test_output_path_equal_to_input_is_rejected(self):
        rc = runner.main([
            "--video", str(self.video), "--log", str(self.log),
            "--ground-truth", str(self.gt), "--out-dir", str(self.log),
        ])
        self.assertEqual(rc, 2)

    def test_missing_reviews_is_allowed_but_identity_not_estimable(self):
        rc = runner.main([
            "--video", str(self.video), "--log", str(self.log),
            "--ground-truth", str(self.gt), "--out-dir", str(self.out),
            "--no-decode-count",
        ])
        self.assertEqual(rc, 0)
        run = sorted(self.out.glob("run_*"))[0]
        with open(run / "gate_a_v3_metrics.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["identity_metrics"]["total_reviewed"], 0)
        self.assertEqual(data["identity_metrics"]["overall_status"], "NOT_ESTIMABLE")

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            runner.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
