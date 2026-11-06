"""Adversarial contracts for the Gate A evaluator v3.

Covers the Codex P0/P1 defects:
  * zero eligible judgments must be NOT_ESTIMABLE, never 0.0;
  * GT overlaps are rejected as a whole, view changes are not episodes;
  * unknown/partial onset cannot become an exact out-of-view episode;
  * evidence helpers reject canonical duplicates and impossible locks.
"""
import unittest

import evaluate as ev


def make_log(frames, state="TRACKING", valid="1", tid="1"):
    return [{"frame": str(f), "state": state, "target_track_id": tid, "valid": valid}
            for f in frames]


def make_gt(intervals):
    return [{"start_frame": s, "end_frame": e, "target_visible": v, "view": view, "target_id": None}
            for s, e, v, view in intervals]


def make_reviews(rows):
    return [{"frame": f, "state": state, "presence": p, "drawn": d, "identity": i}
            for f, state, p, d, i in rows]


class ZeroEligibleEstimabilityTests(unittest.TestCase):
    def test_all_unresolved_is_not_estimable_not_zero(self):
        log = make_log(range(65))
        gt = make_gt([(0, 64, 1, "front")])
        reviews = make_reviews([(f, "TRACKING", "present", True, "uncertain") for f in range(65)])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res["buckets"]["unresolved"], 65)
        self.assertEqual(res["reviewed_present_frames"], 65)
        self.assertEqual(res["eligible_present_frames"], 0)
        self.assertIsNone(res["false_lock_rate_target_present"])
        self.assertIsNot(res["false_lock_rate_target_present"], 0.0)
        st = res["metric_status"]["false_lock_rate_target_present"]
        self.assertEqual(st["status"], "NOT_ESTIMABLE")
        self.assertEqual(st["reason"], "NO_ELIGIBLE_PRESENT_IDENTITY_JUDGMENTS")
        self.assertEqual(st["numerator"], 0)
        self.assertEqual(st["denominator"], 0)
        self.assertEqual(res["overall_status"], "NOT_ESTIMABLE")
        self.assertIsNone(res["identity_precision"])

    def test_empty_dataset_all_metrics_not_estimable(self):
        res = ev.evaluate_identity_reviews([], make_log(range(10)), make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["total_reviewed"], 0)
        for name in ("identity_precision", "false_lock_rate", "false_lock_rate_target_present",
                     "false_lock_rate_target_absent", "present_target_lock_recall",
                     "present_target_output_coverage", "lock_recall"):
            self.assertIsNone(res[name], name)
            self.assertEqual(res["metric_status"][name]["status"], "NOT_ESTIMABLE", name)

    def test_all_presence_uncertain_is_unknown(self):
        log = make_log(range(10))
        reviews = make_reviews([(f, "TRACKING", "uncertain", True, "correct") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["buckets"]["unresolved"], 10)
        self.assertIsNone(res["false_lock_rate_target_present"])

    def test_tracking_without_box_is_miss_not_correct(self):
        log = make_log(range(10), state="TRACKING", valid="0")
        reviews = make_reviews([(f, "TRACKING", "present", False, "none") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["buckets"]["miss_present"], 10)
        self.assertEqual(res["buckets"]["correct_lock"], 0)
        self.assertEqual(res["eligible_present_frames"], 10)
        # 0 wrong over 10 eligible present frames is a legitimate 0.0.
        self.assertEqual(res["false_lock_rate_target_present"], 0.0)
        self.assertEqual(res["metric_status"]["false_lock_rate_target_present"]["status"], "ESTIMABLE")
        self.assertEqual(res["present_target_lock_recall"], 0.0)

    def test_output_box_with_uncertain_identity_is_not_correct(self):
        log = make_log(range(10))
        reviews = make_reviews([(f, "TRACKING", "present", True, "uncertain") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["buckets"]["correct_lock"], 0)
        self.assertEqual(res["eligible_present_frames"], 0)
        self.assertIsNone(res["present_target_lock_recall"])

    def test_zero_wrong_over_ten_eligible_is_zero(self):
        log = make_log(range(10))
        reviews = make_reviews([(f, "TRACKING", "present", True, "correct") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["judged_locks"], 10)
        self.assertEqual(res["identity_precision"], 1.0)
        self.assertEqual(res["false_lock_rate_target_present"], 0.0)
        self.assertEqual(res["metric_status"]["false_lock_rate_target_present"]["denominator"], 10)

    def test_one_wrong_nine_correct(self):
        log = make_log(range(10))
        rows = [(f, "TRACKING", "present", True, "correct") for f in range(10)]
        rows[4] = (4, "TRACKING", "present", True, "wrong")
        res = ev.evaluate_identity_reviews(make_reviews(rows), log,
                                           make_gt([(0, 9, 1, "front")]), margin_frames=0)
        self.assertAlmostEqual(res["identity_precision"], 0.9)
        self.assertAlmostEqual(res["false_lock_rate_target_present"], 0.1)

    def test_gt_unknown_with_independent_human_review_is_usable(self):
        # Coverage gap at frame 5 -> GT_UNKNOWN. A human identity judgment is still
        # independent evidence; GT unknown is never converted to absence.
        log = make_log(range(10))
        gt = make_gt([(0, 4, 1, "front"), (6, 9, 1, "front")])
        reviews = make_reviews([(5, "TRACKING", "present", True, "correct")])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(ev.get_gt_info_at_frame(gt, 5)[0], ev.GT_UNKNOWN)
        self.assertEqual(res["buckets"]["correct_lock"], 1)
        self.assertEqual(res["gt_review_disagreements"], 0)

    def test_view_boundary_is_primary_excluded_but_sensitivity_eligible(self):
        log = make_log(range(10))
        gt = make_gt([(0, 4, 1, "front"), (5, 9, 1, "back")])
        reviews = make_reviews([(4, "TRACKING", "present", True, "correct"),
                                (5, "TRACKING", "present", True, "correct")])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res["buckets"]["unresolved"], 2)
        self.assertEqual(res["sensitivity_view_boundary_ignored"]["buckets"]["correct_lock"], 2)
        self.assertFalse(res["sensitivity_view_boundary_ignored"]["official"])

    def test_presence_boundary_still_excluded_in_sensitivity(self):
        log = make_log(range(10))
        gt = make_gt([(0, 4, 1, "front"), (5, 9, 0, "none")])
        reviews = make_reviews([(4, "TRACKING", "present", True, "correct")])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res["buckets"]["unresolved"], 1)
        self.assertEqual(res["sensitivity_view_boundary_ignored"]["buckets"]["unresolved"], 1)

    def test_partial_presence_label_is_explicitly_unresolved(self):
        log = make_log(range(10))
        reviews = make_reviews([(f, "TRACKING", "partial", True, "correct") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, make_gt([(0, 9, 1, "front")]),
                                           margin_frames=0)
        self.assertEqual(res["reviewed_partial_frames"], 10)
        self.assertEqual(res["buckets"]["unresolved"], 10)
        self.assertEqual(res["eligible_present_frames"], 0)

    def test_partial_gt_is_unresolved_not_absent(self):
        log = make_log(range(10))
        gt = make_gt([(0, 9, 2, "back")])
        reviews = make_reviews([(f, "TRACKING", "present", True, "correct") for f in range(10)])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res["buckets"]["unresolved"], 10)
        self.assertEqual(ev.get_gt_info_at_frame(gt, 3)[0], ev.GT_PARTIAL)


class GroundTruthValidationTests(unittest.TestCase):
    def test_nested_overlap_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 0, "end_frame": 100, "target_visible": 1, "view": "front"},
                {"start_frame": 10, "end_frame": 20, "target_visible": 1, "view": "front"},
            ])

    def test_partial_overlap_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 0, "end_frame": 100, "target_visible": 1, "view": "front"},
                {"start_frame": 50, "end_frame": 150, "target_visible": 1, "view": "front"},
            ])

    def test_contradictory_overlap_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 0, "end_frame": 100, "target_visible": 1, "view": "front"},
                {"start_frame": 30, "end_frame": 40, "target_visible": 0, "view": "none"},
            ])

    def test_inclusive_endpoint_overlap_rejected(self):
        # start/end are inclusive: sharing one frame is an overlap, not adjacency.
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front"},
                {"start_frame": 9, "end_frame": 19, "target_visible": 1, "view": "front"},
            ])

    def test_duplicate_interval_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 10, "end_frame": 20, "target_visible": 1, "view": "front"},
                {"start_frame": 10, "end_frame": 20, "target_visible": 1, "view": "front"},
            ])

    def test_reversed_negative_malformed_missing_rejected(self):
        cases = [
            [{"start_frame": 10, "end_frame": 5, "target_visible": 1, "view": "front"}],
            [{"start_frame": -1, "end_frame": 5, "target_visible": 1, "view": "front"}],
            [{"start_frame": 0, "end_frame": 5, "target_visible": "maybe", "view": "front"}],
            [{"start_frame": 0, "end_frame": 5, "view": "front"}],
            [{"start_frame": 0, "end_frame": 5, "target_visible": 1, "view": "front"}],
        ]
        for case in cases[:4]:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    ev.validate_ground_truth_intervals(case)
        # absent interval carrying a view is contradictory
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals([
                {"start_frame": 0, "end_frame": 5, "target_visible": 0, "view": "back"}])

    def test_out_of_video_range_rejected(self):
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals(
                [{"start_frame": 0, "end_frame": 10, "target_visible": 1, "view": "front"}],
                max_frame=5)

    def test_adjacent_identical_intervals_are_merged_losslessly(self):
        merged = ev.validate_ground_truth_intervals([
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front"},
            {"start_frame": 10, "end_frame": 19, "target_visible": 1, "view": "front"},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start_frame"], merged[0]["end_frame"]), (0, 19))


class EpisodeContractTests(unittest.TestCase):
    def test_unknown_gap_after_absence_is_not_exact_onset(self):
        res = ev.build_reacquisition_episodes(make_gt([
            (0, 9, 1, "front"), (10, 19, 0, "none"), (20, 29, -1, "unknown"), (30, 39, 1, "front")]))
        self.assertEqual(len(res["reacquisition_episodes"]), 0)
        self.assertEqual(len(res["unknown_onset_reappearances"]), 1)
        u = res["unknown_onset_reappearances"][0]
        self.assertEqual(u["event_class"], ev.EVENT_UNKNOWN_ONSET)
        self.assertEqual(u["onset_status"], "unknown")

    def test_unannotated_gap_after_absence_is_unknown(self):
        res = ev.build_reacquisition_episodes(make_gt([
            (0, 9, 1, "front"), (10, 19, 0, "none"), (30, 39, 1, "front")]))
        self.assertEqual(len(res["reacquisition_episodes"]), 0)
        self.assertEqual(len(res["unknown_onset_reappearances"]), 1)

    def test_partial_reappearance_starts_one_episode(self):
        res = ev.build_reacquisition_episodes(make_gt([
            (0, 9, 1, "front"), (10, 19, 0, "none"),
            (20, 29, 2, "back"), (30, 39, 1, "back")]))
        self.assertEqual(len(res["reacquisition_episodes"]), 1)
        ep = res["reacquisition_episodes"][0]
        self.assertEqual(ep["start_frame"], 20)
        self.assertEqual(ep["onset_visibility"], "partial")
        self.assertEqual(ep["first_fully_visible_frame"], 30)
        self.assertEqual(ep["earliest_possible_present_frame"], 20)

    def test_view_change_is_not_an_episode(self):
        res = ev.build_reacquisition_episodes(make_gt([(0, 9, 1, "front"), (10, 19, 1, "back")]))
        self.assertEqual(len(res["reacquisition_episodes"]), 0)
        self.assertEqual(len(res["viewpoint_changes"]), 1)

    def test_occlusion_recovery_is_not_out_of_view(self):
        res = ev.build_reacquisition_episodes(make_gt([
            (0, 9, 1, "back"), (10, 19, 2, "back"), (20, 29, 1, "back")]))
        self.assertEqual(len(res["reacquisition_episodes"]), 0)
        self.assertGreaterEqual(len(res["occlusion_recoveries"]), 1)

    def test_in_scene_tracking_loss_is_not_out_of_view(self):
        gt = make_gt([(0, 29, 1, "front")])
        log = (make_log(range(0, 10)) + make_log(range(10, 20), state="LOST", valid="0")
               + make_log(range(20, 30)))
        events = ev.classify_tracker_events(log, gt)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_class"], ev.EVENT_IN_SCENE_RECOVERY)
        self.assertEqual(len(ev.build_reacquisition_episodes(gt)["reacquisition_episodes"]), 0)

    def test_true_out_of_view_event(self):
        gt = make_gt([(0, 9, 1, "front"), (10, 19, 0, "none"), (20, 29, 1, "back")])
        log = (make_log(range(0, 10), tid="1") + make_log(range(10, 20), state="LOST", valid="0", tid="1")
               + make_log(range(20, 30), tid="7"))
        events = ev.classify_tracker_events(log, gt)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e["event_class"], ev.EVENT_TRUE_REAPPEARANCE)
        self.assertEqual((e["lost_start_frame"], e["relock_frame"]), (10, 20))
        self.assertEqual((e["prior_tracker_id"], e["tracker_id_at_relock"]), (1, 7))
        self.assertTrue(e["tracker_id_changed"])

    def test_tracker_id_change_without_lost(self):
        gt = make_gt([(0, 9, 1, "front")])
        log = make_log(range(0, 5), tid="1") + make_log(range(5, 10), tid="2")
        events = ev.classify_tracker_events(log, gt)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_class"], ev.EVENT_TRACKER_ID_CHANGE)
        self.assertEqual(events[0]["frame"], 5)

    def test_removing_review_cannot_prove_more(self):
        frames = list(range(20, 28))
        buckets = {f: "correct_lock" for f in frames}
        capable = set(frames)
        log_frames = set(range(20, 30))
        proven = ev.reacquisition_episode_evidence(20, 29, buckets, capable, log_frames, 0)
        self.assertTrue(proven["first_correct_proven"])
        reduced = dict(buckets)
        del reduced[20]  # gap BEFORE the observed first correct lock
        unproven = ev.reacquisition_episode_evidence(20, 29, reduced, capable, log_frames, 0)
        self.assertFalse(unproven["first_correct_proven"])
        self.assertEqual(unproven["status"], "first_correct_unproven")


class EvidenceIntegrityTests(unittest.TestCase):
    def test_duplicate_canonical_evidence_keys_rejected(self):
        with self.assertRaises(ValueError):
            ev.reacquisition_episode_evidence(
                10, 15, {10: "unresolved", "010": "correct_lock"},
                lock_frames={10}, log_frames=set(range(10, 16)), onset_uncertainty_frames=0)

    def test_impossible_lock_evidence_rejected(self):
        with self.assertRaises(ValueError):
            ev.reacquisition_episode_evidence(
                10, 15, {10: "correct_lock"}, lock_frames=set(),
                log_frames=set(range(10, 16)), onset_uncertainty_frames=0)

    def test_lock_without_log_row_rejected(self):
        with self.assertRaises(ValueError):
            ev.reacquisition_episode_evidence(
                10, 15, {10: "correct_lock"}, lock_frames={10},
                log_frames=set(range(11, 16)), onset_uncertainty_frames=0)

    def test_duplicate_lock_frame_forms_rejected(self):
        with self.assertRaises(ValueError):
            ev.reacquisition_episode_evidence(
                10, 15, {10: "unresolved"}, lock_frames={10, "010"},
                log_frames=set(range(10, 16)), onset_uncertainty_frames=0)

    def test_invalid_bucket_rejected(self):
        with self.assertRaises(ValueError):
            ev.reacquisition_episode_evidence(
                10, 15, {10: "uncertan"}, lock_frames={10},
                log_frames=set(range(10, 16)), onset_uncertainty_frames=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
