import contextlib
import csv
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/fixtures/identity'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = load('evaluate', ROOT / 'evaluate.py')


def make_log(rows):
    return [{'frame': str(f), 'state': s, 'target_track_id': t, 'valid': v} for f, s, t, v in rows]


def make_gt(intervals):
    return [{'start_frame': s, 'end_frame': e, 'target_visible': v, 'view': w, 'target_id': None}
            for s, e, v, w in intervals]


def make_reviews(rows):
    return [{'frame': f, 'state': s, 'presence': p, 'drawn': d, 'identity': i}
            for f, s, p, d, i in rows]


class LoadIdentityReviewsTests(unittest.TestCase):
    def _write(self, header, lines):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        p = Path(td.name) / 'reviews.csv'
        p.write_text(','.join(header) + '\n' + '\n'.join(lines) + '\n', encoding='utf-8')
        return p

    def test_valid_load(self):
        p = self._write(ev.IDENTITY_REVIEW_COLUMNS, ['1,TRACKING,present,1,correct', '2,LOST,absent,0,none'])
        rows = ev.load_identity_reviews(str(p))
        self.assertEqual([r['frame'] for r in rows], [1, 2])
        self.assertTrue(rows[0]['drawn']); self.assertFalse(rows[1]['drawn'])

    def test_rejections(self):
        cases = {
            'duplicate_frame': ['1,TRACKING,present,1,correct', '1,TRACKING,present,1,correct'],
            'bad_state': ['1,SLEEPING,present,1,correct'],
            'bad_presence': ['1,TRACKING,maybe,1,correct'],
            'drawn1_identity_none': ['1,TRACKING,present,1,none'],
            'drawn0_identity_correct': ['1,LOST,present,0,correct'],
            'absent_and_correct': ['1,TRACKING,absent,1,correct'],
            'bad_identity': ['1,TRACKING,present,1,right'],
        }
        for name, lines in cases.items():
            with self.subTest(case=name):
                p = self._write(list(ev.IDENTITY_REVIEW_COLUMNS), lines)
                with self.assertRaises(ValueError):
                    ev.load_identity_reviews(str(p))
        p = self._write(['frame', 'state', 'presence', 'drawn'], ['1,TRACKING,present,1'])
        with self.assertRaises(ValueError):
            ev.load_identity_reviews(str(p))


class ClassifyIdentityTests(unittest.TestCase):
    def test_state_vs_identity_confusion(self):
        # Same TRACKING+valid state, opposite identity truth: old metric blind to frames 2,3.
        log = make_log([(f, 'TRACKING', 1, '1') for f in range(6)])
        gt = make_gt([(0, 9, 1, 'front')])
        reviews = make_reviews([
            (0, 'TRACKING', 'present', True, 'correct'),
            (1, 'TRACKING', 'present', True, 'correct'),
            (2, 'TRACKING', 'present', True, 'wrong'),
            (3, 'TRACKING', 'present', True, 'wrong'),
            (4, 'TRACKING', 'present', True, 'correct'),
            (5, 'TRACKING', 'present', True, 'uncertain'),
        ])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        b = res['buckets']
        # frame 0 is a GT boundary (margin 0) and frame 5 has uncertain identity -> unresolved
        # v3: frame 0 is not a boundary (no preceding transition); only frame 5
        # (uncertain identity) stays unresolved.
        self.assertEqual(b['correct_lock'], 3)   # frames 0, 1, 4
        self.assertEqual(b['wrong_lock_present'], 2)  # frames 2, 3
        self.assertEqual(b['unresolved'], 1)
        self.assertAlmostEqual(res['identity_precision'], 3 / 5)
        self.assertAlmostEqual(res['false_lock_rate'], 2 / 5)
        # state proxy counts zero false locks here (GT present throughout) — the blind spot.
        self.assertEqual(res['state_proxy_false_lock_frames'], 0)

    def test_initializing_correct_is_not_a_lock(self):
        log = make_log([(f, 'INITIALIZING', 1, '1') for f in range(6)])
        gt = make_gt([(0, 5, 1, 'front')])
        reviews = make_reviews([(2, 'INITIALIZING', 'present', True, 'correct')])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res['buckets']['correct_output_not_lock'], 1)
        self.assertEqual(res['buckets']['correct_lock'], 0)
        self.assertEqual(res['correct_output_total'], 1)

    def test_partial_and_boundary_unresolved(self):
        log = make_log([(f, 'TRACKING', 1, '1') for f in range(6)])
        gt = make_gt([(0, 1, 2, 'side'), (2, 5, 1, 'front')])
        reviews = make_reviews([(f, 'TRACKING', 'present', True, 'correct') for f in range(6)])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        # vis=2 frames 0,1 and GT-boundary frames 2,5 are unresolved
        # v3: the last interval's closing edge is not a transition, so frame 5 is
        # eligible. margin=1 covers only the partial/presence transition at 1/2.
        self.assertEqual(res['buckets']['unresolved'], 3)   # frames 0,1,2
        self.assertEqual(res['buckets']['correct_lock'], 3)  # frames 3,4,5
        res2 = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=1)
        self.assertEqual(res2['buckets']['unresolved'], 4)  # frames 0,1,2,3
        self.assertEqual(res2['buckets']['correct_lock'], 2)

    def test_correct_requires_scene_confirmed_presence(self):
        log = make_log([(0, 'TRACKING', 1, '1')])
        gt = make_gt([(0, 0, 1, 'front')])
        reviews = make_reviews([(0, 'TRACKING', 'uncertain', True, 'correct')])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res['buckets']['unresolved'], 1)
        self.assertEqual(res['buckets']['correct_lock'], 0)

    def test_provenance_mismatch_rejected(self):
        log = make_log([(0, 'LOST', 1, '0')])
        gt = make_gt([(0, 0, 1, 'front')])
        reviews = make_reviews([(0, 'TRACKING', 'present', True, 'correct')])
        with self.assertRaises(ValueError):
            ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        reviews2 = make_reviews([(0, 'LOST', 'present', True, 'correct')])  # drawn but log valid=0
        with self.assertRaises(ValueError):
            ev.evaluate_identity_reviews(reviews2, log, gt, margin_frames=0)
        reviews3 = make_reviews([(5, 'LOST', 'absent', False, 'none')])  # frame not in log
        with self.assertRaises(ValueError):
            ev.evaluate_identity_reviews(reviews3, log, gt, margin_frames=0)

    def test_reuses_production_identity_screen_counts(self):
        log = make_log([(0, 'TRACKING', 1, '1'), (1, 'TRACKING', 1, '1'), (2, 'TRACKING', 1, '0')])
        gt = make_gt([(0, 2, 1, 'front')])
        reviews = make_reviews([
            (0, 'TRACKING', 'present', True, 'correct'),
            (1, 'TRACKING', 'present', True, 'wrong'),
            (2, 'TRACKING', 'present', False, 'none'),
        ])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        self.assertEqual(res['identity_screen_counts']['samples'], 3)
        self.assertEqual(res['identity_screen_counts']['correct_output'], 1)
        self.assertEqual(res['identity_screen_counts']['wrong_output'], 1)
        self.assertEqual(res['identity_screen_counts']['no_output'], 1)
        self.assertAlmostEqual(res['wrong_fraction_among_reviewed_outputs'], 0.5)


class FixtureMetricsTests(unittest.TestCase):
    def _load_csv(self, path):
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def test_fixture_exact_expected_metrics(self):
        log = self._load_csv(FIX / 'synthetic_log.csv')
        gt = ev.parse_ground_truth(str(FIX / 'synthetic_gt.csv'))
        reviews = ev.load_identity_reviews(str(FIX / 'synthetic_reviews.csv'))
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        b = res['buckets']
        # v3 boundary contract: an interval edge that is not a presence transition
        # (frame 29 is the last frame of the video) is NOT a boundary. It therefore
        # remains eligible and its wrong lock is counted instead of being hidden.
        self.assertEqual(b, {
            'correct_lock': 5, 'correct_output_not_lock': 2,
            'wrong_lock_present': 4, 'wrong_lock_absent': 5,
            'wrong_output_not_lock_present': 1, 'wrong_output_not_lock_absent': 0,
            'miss_present': 2, 'correct_rejection_absent': 3, 'unresolved': 8,
        })
        self.assertEqual(res['judged_locks'], 14)
        self.assertAlmostEqual(res['identity_precision'], 5 / 14)
        self.assertAlmostEqual(res['lock_recall'], 5 / 11)
        self.assertAlmostEqual(res['false_lock_rate'], 9 / 14)
        self.assertAlmostEqual(res['identity_coverage'], 0.5)
        self.assertEqual(res['correct_output_total'], 7)
        self.assertEqual(res['target_miss_total'], 7)
        self.assertEqual(res['gt_review_disagreements'], 2)
        self.assertEqual(res['state_proxy_false_lock_frames'], 5)
        self.assertEqual(len(res['reacquisition_episodes']), 1)
        ep = res['reacquisition_episodes'][0]
        self.assertEqual(ep['start'], 20); self.assertEqual(ep['end'], 29)
        self.assertEqual(ep['state_latency_frames'], 4)
        # Evidence-limited semantics (2026-11-05): frame 25 is lock-capable and was
        # reviewed as uncertain, so the observed correct lock at 27 cannot be proven
        # to be the FIRST correct event. 7 is the observed value, [5..7] the bound.
        self.assertEqual(ep['identity_latency_frames'], 7)
        self.assertFalse(ep['first_correct_proven'])
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['latency_lower_bound_frames'], 5)
        self.assertEqual(ep['latency_upper_bound_frames'], 7)
        self.assertFalse(ep['latency_exact'])
        self.assertEqual(ep['wrong_lock_first_frame'], 24)
        self.assertEqual(ep['wrong_locks_before_correct'], 1)
        self.assertEqual(ep['evidence']['blocking_ranges'], [[25, 25]])
        # identity_screen cross-check through production code
        self.assertEqual(res['identity_screen_counts']['samples'], 30)
        self.assertEqual(res['identity_screen_counts']['reviewed_output'], 17)
        self.assertEqual(res['identity_screen_counts']['correct_output'], 7)
        self.assertEqual(res['identity_screen_counts']['wrong_output'], 10)
        self.assertEqual(res['identity_screen_counts']['wrong_present'], 5)
        self.assertEqual(res['identity_screen_counts']['wrong_absent'], 5)

    def test_old_state_metric_blind_spot_on_same_fixture(self):
        # The pre-existing state-proxy metric on the same synthetic data:
        # counts 5 false locks (all TRACKING-while-GT-absent), sees NONE of the
        # 4 wrong locks while the target is present, and cannot judge identity at all.
        log = self._load_csv(FIX / 'synthetic_log.csv')
        gt = ev.parse_ground_truth(str(FIX / 'synthetic_gt.csv'))
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            old = ev.evaluate_log_gt(str(FIX / 'synthetic_log.csv'), str(FIX / 'synthetic_gt.csv'),
                                     fps=30.0, margin_frames=0)
        self.assertEqual(old['false_lock_frames'], 5)
        self.assertEqual(old['false_lock_rate'], 5 / 30 * 100.0)
        self.assertIn('total_frames', old)  # return structure unchanged (regression)


class _EpisodeCaseMixin:
    """Shared episode builders.

    A plain mixin (not a TestCase) so that no test is collected twice when two
    test classes share the same fixtures.
    """

    CORRECT = ('present', 'correct')
    WRONG = ('present', 'wrong')
    UNCERTAIN = ('uncertain', 'uncertain')
    MISS = ('present', 'none')
    ABSENT = ('absent', 'none')
    NON_LOCK_UNCERTAIN = ('uncertain', 'none')

    def build(self, spec, margin=0, skip_log=()):
        """spec: {frame: (state, valid, review or None)}; unspecified frames are LOST/valid=0, unreviewed.
        Frames in skip_log get no log row at all (log gaps)."""
        log, reviews = [], []
        for f in range(30):
            state, valid, review = spec.get(f, ('LOST', '0', None))
            if f not in skip_log:
                log.append({'frame': str(f), 'state': state, 'target_track_id': '1', 'valid': valid})
            if review is not None:
                presence, identity = review
                reviews.append({'frame': f, 'state': state, 'presence': presence,
                                'drawn': valid == '1', 'identity': identity})
        gt = make_gt([(0, 9, 1, 'front'), (10, 19, 0, 'none'), (20, 29, 1, 'back')])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=margin)
        return res, res['reacquisition_episodes'][0]

    def tracking(self, frames, review, state='TRACKING'):
        return {f: (state, '1', review) for f in frames}

    def capable_unreviewed(self, frames):
        return {f: ('TRACKING', '1', None) for f in frames}


class ReacquisitionEpisodeEvidenceTests(_EpisodeCaseMixin, unittest.TestCase):
    """Evidence-limited reacquisition semantics (2026-11-05).

    Episode under test is GT interval 20-29 (target visible); 0-9 is a visible
    first interval (excluded) and 10-19 is absent. A lock can only exist where
    the log is TRACKING with valid=1.

    Two rules of the existing classifier matter for these tests:
      * frames exactly on a GT boundary (20 and 29 here) are always `unresolved`,
        even with margin_frames=0, so they can never prove a first correct lock;
      * a review with drawn=0 must carry identity=none (scene presence may still
        be uncertain), so non-lock frames are expressed as ('uncertain', 'none').
    """

    # --- 1. partial review cannot prove the first event (Codex's case) -------
    def test_single_correct_frame_is_not_a_proven_first_event(self):
        # Only frame 27 reviewed; the log shows lock-capable (unreviewed) frames 24-25,
        # so a wrong lock at 24 cannot be excluded. This is the reported defect.
        spec = self.capable_unreviewed([24, 25, 28, 29])
        spec[27] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec)
        self.assertEqual(ep['identity_latency_frames'], 7)      # observed only
        self.assertFalse(ep['first_correct_proven'])
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['latency_lower_bound_frames'], 4)   # could be as early as 24
        self.assertEqual(ep['latency_upper_bound_frames'], 7)
        self.assertEqual(ep['evidence']['blocking_ranges'], [[24, 25]])
        self.assertFalse(ep['could_be_no_correct_lock'])

    def test_correct_observed_after_unreviewed_capable_frames_is_bounded(self):
        spec = self.capable_unreviewed([21, 22, 23])
        spec[24] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual((ep['latency_lower_bound_frames'], ep['latency_upper_bound_frames']), (1, 4))
        self.assertFalse(ep['latency_exact'])

    # --- 2. complete review proves the first event ---------------------------
    def test_complete_resolved_review_proves_first_correct(self):
        spec = self.tracking(range(24, 29), self.CORRECT)
        res, ep = self.build(spec)
        self.assertTrue(ep['first_correct_proven'])
        self.assertTrue(ep['latency_exact'])
        self.assertEqual(ep['status'], 'ok')
        self.assertEqual(ep['identity_latency_frames'], 4)
        self.assertEqual((ep['latency_lower_bound_frames'], ep['latency_upper_bound_frames']), (4, 4))
        self.assertEqual(ep['wrong_locks_in_episode'], 0)

    def test_non_lock_unresolved_frames_do_not_block_the_proof(self):
        # Frames 20-23 are LOST/valid=0: no lock was possible there, so uncertain
        # reviews cannot hide a first correct lock.
        spec = {f: ('LOST', '0', self.NON_LOCK_UNCERTAIN) for f in range(20, 24)}
        spec.update(self.tracking(range(24, 29), self.CORRECT))
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'ok')
        self.assertTrue(ep['latency_exact'])
        self.assertEqual(ep['evidence']['non_lock_unresolved_ranges'], [[20, 23]])

    # --- 3. wrong lock before a later correct reacquisition -----------------
    def test_wrong_before_correct_is_surfaced_in_a_proven_episode(self):
        spec = self.tracking(range(22, 29), self.CORRECT)
        spec[22] = ('TRACKING', '1', self.WRONG)
        res, ep = self.build(spec)
        self.assertTrue(ep['first_correct_proven'])
        self.assertEqual(ep['status'], 'ok_with_prior_wrong_lock')
        self.assertEqual(ep['wrong_lock_first_frame'], 22)
        self.assertEqual(ep['wrong_locks_before_correct'], 1)
        self.assertEqual(ep['identity_latency_frames'], 3)      # 23 - 20
        self.assertEqual(res['episodes_with_prior_wrong_lock'], 1)
        self.assertEqual(res['proven_latency_episodes'], 1)

    def test_wrong_before_correct_with_coverage_gap_is_unproven_but_recorded(self):
        spec = self.capable_unreviewed([23])
        spec[22] = ('TRACKING', '1', self.WRONG)
        spec[24] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec)
        self.assertFalse(ep['first_correct_proven'])
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['wrong_lock_first_frame'], 22)
        self.assertEqual(ep['wrong_locks_before_correct'], 1)
        self.assertEqual(ep['evidence']['blocking_ranges'], [[23, 23]])
        self.assertEqual((ep['latency_lower_bound_frames'], ep['latency_upper_bound_frames']), (3, 4))

    def test_wrong_after_correct_is_reported_but_does_not_change_first_event(self):
        spec = self.tracking(range(24, 28), self.CORRECT)
        spec[26] = ('TRACKING', '1', self.WRONG)
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'ok')
        self.assertIsNone(ep['wrong_lock_first_frame'])
        self.assertEqual(ep['wrong_locks_in_episode'], 1)
        self.assertEqual(ep['wrong_lock_after_correct_first_frame'], 26)

    # --- 4. missing frames: censoring and proven absence --------------------
    def test_no_correct_observed_with_missing_capable_frames_is_censored(self):
        spec = {22: ('LOST', '0', self.MISS)}
        spec.update(self.capable_unreviewed([24, 25]))
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'correct_not_observed_censored')
        self.assertIsNone(ep['identity_latency_frames'])
        self.assertIsNone(ep['first_correct_frame'])
        self.assertTrue(ep['could_be_no_correct_lock'])
        self.assertFalse(ep['latency_exact'])
        self.assertEqual(ep['latency_lower_bound_frames'], 4)
        self.assertEqual(ep['latency_upper_bound_frames'], 9)

    def test_all_lock_frames_resolved_without_correct_proves_absence(self):
        spec = self.tracking(range(21, 29), self.WRONG)
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'no_correct_lock_proven')
        self.assertIsNone(ep['identity_latency_frames'])
        self.assertTrue(ep['could_be_no_correct_lock'])
        self.assertEqual(ep['wrong_locks_in_episode'], 8)

    def test_unreviewed_episode_is_not_evaluated(self):
        spec = {f: ('LOST', '0', self.ABSENT) for f in range(10, 20)}
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'not_evaluated')
        self.assertIsNone(ep['latency_lower_bound_frames'])
        self.assertIsNone(ep['latency_upper_bound_frames'])
        self.assertEqual(res['not_evaluated_episodes'], 1)
        # UNKNOWN is not negative evidence: no review cannot exclude a correct lock.
        self.assertTrue(ep['could_be_no_correct_lock'])
        self.assertFalse(ep['first_correct_proven'])
        self.assertEqual(res['no_correct_lock_episodes'], 0)
        self.assertEqual(res['censored_latency_episodes'], 0)

    def test_unreviewed_capable_episode_is_unknown_not_negative(self):
        spec = self.capable_unreviewed([21, 22, 23])
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'not_evaluated')
        self.assertTrue(ep['could_be_no_correct_lock'])
        self.assertFalse(ep['first_correct_proven'])
        self.assertIsNone(ep['identity_latency_frames'])
        self.assertFalse(ep['latency_exact'])
        self.assertFalse(ep['no_wrong_lock_proven'])
        self.assertEqual(res['not_evaluated_episodes'], 1)
        self.assertEqual(res['no_correct_lock_episodes'], 0)
        self.assertEqual(res['censored_latency_episodes'], 0)

    def test_report_surfaces_the_unknown_flag(self):
        spec = self.capable_unreviewed([21, 22])
        res, ep = self.build(spec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev.print_identity_report(res)
        output = buf.getvalue()
        self.assertIn('status=not_evaluated', output)
        self.assertIn('could_be_no_correct=True', output)

    # --- 5. ambiguous annotations ------------------------------------------
    def test_uncertain_lock_frame_blocks_the_proof(self):
        spec = self.capable_unreviewed([28])
        spec[24] = ('TRACKING', '1', self.UNCERTAIN)
        spec[27] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['evidence']['blocking_ranges'], [[24, 24]])
        self.assertEqual(ep['latency_lower_bound_frames'], 4)

    def test_boundary_margin_turns_every_judgment_into_unresolved(self):
        spec = self.tracking(range(24, 30), self.CORRECT)
        spec.update(self.tracking(range(20, 24), self.WRONG))
        res, ep = self.build(spec, margin=30)   # 20-29 all within +-30 of a boundary
        self.assertEqual(ep['status'], 'correct_not_observed_censored')
        self.assertFalse(ep['first_correct_proven'])
        self.assertEqual(res['buckets']['correct_lock'], 0)
        self.assertGreater(res['buckets']['unresolved'], 0)

    def test_gt_boundary_frame_alone_blocks_a_first_frame_proof(self):
        # Frame 20 sits exactly on the GT boundary, so it is unresolved even at
        # margin_frames=0 and a correct lock there can never be "proven first".
        spec = self.tracking(range(20, 29), self.CORRECT)
        res, ep = self.build(spec)
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['evidence']['blocking_ranges'], [[20, 20]])
        self.assertEqual(ep['latency_lower_bound_frames'], 0)

    def test_log_gap_is_not_evidence_of_absence(self):
        # A frame with no log row could have been TRACKING+valid, so it blocks.
        spec = self.tracking(range(24, 29), self.CORRECT)
        res, ep = self.build(spec, skip_log=(22,))
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['evidence']['log_unknown_ranges'], [[22, 22]])
        self.assertEqual(ep['evidence']['blocking_ranges'], [[22, 22]])
        self.assertEqual(ep['latency_lower_bound_frames'], 2)

    def test_multiple_episodes_are_evaluated_independently(self):
        log, reviews = [], []
        spec = {22: ('TRACKING', '1', self.CORRECT),
                32: ('TRACKING', '1', self.WRONG),
                34: ('TRACKING', '1', self.CORRECT)}
        for f in range(40):
            state, valid, review = spec.get(f, ('LOST', '0', None))
            log.append({'frame': str(f), 'state': state, 'target_track_id': '1', 'valid': valid})
            if review is not None:
                reviews.append({'frame': f, 'state': state, 'presence': review[0],
                                'drawn': True, 'identity': review[1]})
        gt = make_gt([(0, 9, 1, 'front'), (10, 19, 0, 'none'),
                      (20, 24, 1, 'back'), (25, 29, 0, 'none'),
                      (30, 39, 1, 'back')])
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)
        first, second = res['reacquisition_episodes']
        self.assertEqual((first['start'], first['end']), (20, 24))
        self.assertEqual((first['status'], first['identity_latency_frames']), ('ok', 2))
        self.assertEqual((second['start'], second['end']), (30, 39))
        self.assertEqual(second['status'], 'ok_with_prior_wrong_lock')
        self.assertEqual((second['identity_latency_frames'], second['wrong_lock_first_frame']), (4, 32))
        self.assertEqual(res['proven_latency_episodes'], 2)
        self.assertEqual(res['episodes_with_prior_wrong_lock'], 1)

    # --- 6. compatible surface ---------------------------------------------
    def test_episode_dict_keeps_backwards_compatible_keys(self):
        spec = self.tracking(range(24, 29), self.CORRECT)
        res, ep = self.build(spec)
        for key in ('start', 'end', 'state_latency_frames', 'identity_latency_frames',
                    'wrong_lock_first_frame', 'status'):
            self.assertIn(key, ep)
        for key in ('reacquisition_episodes', 'unresolved_latency_episodes',
                    'not_evaluated_episodes'):
            self.assertIn(key, res)

    def test_counters_split_proven_unproven_and_censored(self):
        spec = self.tracking(range(22, 29), self.CORRECT)
        spec[22] = ('TRACKING', '1', self.WRONG)
        res, _ = self.build(spec)
        self.assertEqual(res['proven_latency_episodes'], 1)
        self.assertEqual(res['unproven_latency_episodes'], 0)
        self.assertEqual(res['censored_latency_episodes'], 0)
        self.assertEqual(res['unresolved_latency_episodes'], 0)
        self.assertEqual(res['no_correct_lock_episodes'], 0)


class EpisodeEvidenceFunctionTests(unittest.TestCase):
    """Direct tests of the evidence helper: onset uncertainty and observed-vs-proven."""

    def frames(self, mapping, start=20, end=25):
        return set(range(start, end + 1))

    def test_onset_uncertainty_is_carried_in_the_latency_band(self):
        buckets = {20: 'miss_present', 21: 'miss_present', 22: 'miss_present',
                   23: 'miss_present', 24: 'correct_lock'}
        # With onset +-3 the log must also cover the pre-onset window 17-19,
        # otherwise those frames are UNKNOWN and (correctly) block the proof.
        ep = ev.reacquisition_episode_evidence(
            20, 24, buckets, lock_frames={24}, log_frames=set(range(17, 26)),
            onset_uncertainty_frames=3)
        self.assertTrue(ep['latency_exact'])                     # review coverage exact
        self.assertFalse(ep['latency_exact_with_onset_uncertainty'])
        self.assertEqual(ep['identity_latency_frames'], 4)
        self.assertEqual(ep['latency_band_frames'], [1, 7])
        self.assertEqual(ep['onset_uncertainty_frames'], 3)
        self.assertEqual(ep['evidence']['pre_onset_frames_considered'], 3)

    def test_missing_pre_onset_log_rows_block_the_proof(self):
        # Same case, but the log only covers the episode: pre-onset frames are
        # UNKNOWN, so a zero-latency lock cannot be excluded.
        buckets = {20: 'miss_present', 21: 'miss_present', 22: 'miss_present',
                   23: 'miss_present', 24: 'correct_lock'}
        ep = ev.reacquisition_episode_evidence(
            20, 24, buckets, lock_frames={24}, log_frames=self.frames(buckets),
            onset_uncertainty_frames=3)
        self.assertFalse(ep['latency_exact'])
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertEqual(ep['latency_lower_bound_frames'], 0)
        self.assertEqual(ep['latency_band_frames'][0], 0)

    def test_zero_onset_uncertainty_keeps_exactness(self):
        buckets = {22: 'miss_present', 23: 'miss_present', 24: 'correct_lock'}
        ep = ev.reacquisition_episode_evidence(20, 24, buckets, lock_frames={24},
                                               log_frames=self.frames(buckets))
        self.assertTrue(ep['latency_exact_with_onset_uncertainty'])
        self.assertEqual(ep['latency_band_frames'], [4, 4])

    def test_first_wrong_is_proven_only_with_full_preceding_evidence(self):
        buckets = {20: 'miss_present', 21: 'miss_present', 22: 'wrong_lock_present',
                   23: 'miss_present', 24: 'wrong_lock_present', 25: 'correct_lock'}
        capable = {21, 22, 24, 25}
        proven = ev.reacquisition_episode_evidence(20, 25, buckets, capable, self.frames(buckets))
        self.assertEqual(proven['wrong_lock_first_frame'], 22)
        self.assertTrue(proven['wrong_lock_first_frame_proven'])
        self.assertFalse(proven['no_wrong_lock_proven'])

        gapped = dict(buckets)
        del gapped[21]                     # 21 is lock-capable but unreviewed
        unproven = ev.reacquisition_episode_evidence(20, 25, gapped, capable, self.frames(gapped))
        self.assertEqual(unproven['wrong_lock_first_frame'], 22)
        self.assertFalse(unproven['wrong_lock_first_frame_proven'])
        self.assertEqual(unproven['evidence']['blocking_ranges'], [[21, 21]])

    def test_absence_of_wrong_locks_is_proven_only_when_everything_is_resolved(self):
        buckets = {20: 'miss_present', 21: 'miss_present', 22: 'correct_lock'}
        proven = ev.reacquisition_episode_evidence(20, 22, buckets, {22}, self.frames(buckets))
        self.assertTrue(proven['no_wrong_lock_proven'])
        self.assertFalse(proven['wrong_lock_observed'])

        gapped = dict(buckets)
        del gapped[21]
        unproven = ev.reacquisition_episode_evidence(20, 22, gapped, {21, 22}, self.frames(gapped))
        self.assertFalse(unproven['no_wrong_lock_proven'])
        self.assertFalse(unproven['wrong_lock_observed'])   # unobserved is not the same as proven


class OnsetBoundCounterexampleTests(_EpisodeCaseMixin, unittest.TestCase):
    """Codex P0 counterexample: onset +-1 with a pre-onset lock-capable frame.

    Nominal onset is frame 20 and the observed correct lock is frame 22, so the
    naive bound is [1, 3]. A lock-capable frame at 19 (true onset could be 19)
    means latency 0 is possible, so the band must start at 0.
    """

    def test_pre_onset_lock_capable_frame_forces_zero_lower_bound(self):
        spec = self.capable_unreviewed([19])
        spec[22] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec, margin=1)
        self.assertEqual(ep['status'], 'first_correct_unproven')
        self.assertFalse(ep['latency_exact'])
        self.assertEqual(ep['identity_latency_frames'], 2)          # observed only
        self.assertEqual(ep['latency_lower_bound_frames'], 0)
        self.assertEqual(ep['latency_upper_bound_frames'], 2)
        self.assertEqual(ep['latency_band_frames'], [0, 3])
        self.assertLessEqual(ep['latency_band_frames'][0], 0)       # covers zero latency
        self.assertEqual(ep['evidence']['blocking_ranges'], [[19, 19]])
        self.assertEqual(ep['evidence']['search_window_start'], 19)
        self.assertEqual(ep['evidence']['pre_onset_frames_considered'], 1)

    def test_pre_onset_frame_without_a_lock_keeps_exactness(self):
        # Frame 19 is LOST/valid=0 and unreviewed: no lock was possible there, so
        # the coverage proof is unaffected and the band still carries +-1.
        spec = {19: ('LOST', '0', None)}
        spec[22] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec, margin=1)
        self.assertTrue(ep['latency_exact'])
        self.assertEqual(ep['status'], 'ok')
        self.assertEqual(ep['latency_lower_bound_frames'], 2)
        self.assertEqual(ep['latency_band_frames'], [1, 3])

    def test_unresolved_pre_onset_frame_without_lock_does_not_block(self):
        # Unresolved but not lock-capable: it cannot hide a lock, so no block.
        spec = {19: ('LOST', '0', self.NON_LOCK_UNCERTAIN)}
        spec[22] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec, margin=1)
        self.assertEqual(ep['status'], 'ok')
        self.assertEqual(ep['evidence']['blocking_ranges'], [])

    def test_large_margin_window_is_clamped_at_frame_zero(self):
        spec = self.capable_unreviewed([0])
        spec[22] = ('TRACKING', '1', self.CORRECT)
        res, ep = self.build(spec, margin=30)
        self.assertEqual(ep['evidence']['search_window_start'], 0)
        self.assertEqual(ep['evidence']['pre_onset_frames_considered'], 20)
        # margin 30 makes every frame boundary-uncertain, so nothing is resolved.
        self.assertEqual(ep['status'], 'correct_not_observed_censored')
        self.assertIsNone(ep['identity_latency_frames'])
        self.assertEqual(ep['latency_band_frames'][0], 0)

    def test_band_always_covers_observed_latency(self):
        for margin in (0, 1, 2, 5):
            with self.subTest(margin=margin):
                spec = self.capable_unreviewed([19])
                spec[22] = ('TRACKING', '1', self.CORRECT)
                res, ep = self.build(spec, margin=margin)
                band = ep['latency_band_frames']
                latency = ep['identity_latency_frames']
                if band is None or latency is None:
                    continue          # larger margins turn the frames unresolved
                self.assertLessEqual(band[0], latency)
                self.assertGreaterEqual(band[1], latency)
                if margin >= 1:
                    self.assertEqual(band[0], 0)


class SchemaValidationTests(unittest.TestCase):
    """Malformed log/review data must raise, never silently become 'no box'."""

    def setUp(self):
        self.gt = make_gt([(0, 9, 1, 'front'), (10, 19, 0, 'none'), (20, 29, 1, 'back')])

    def base_log(self, valid='0', state='LOST'):
        rows = []
        for frame in range(30):
            if frame == 22:
                rows.append({'frame': str(frame), 'state': 'TRACKING',
                             'target_track_id': '1', 'valid': '1'})
            else:
                rows.append({'frame': str(frame), 'state': state,
                             'target_track_id': '1', 'valid': valid})
        return rows

    def reviews(self, frame=22, state='TRACKING', presence='present',
                drawn=True, identity='correct'):
        return [{'frame': frame, 'state': state, 'presence': presence,
                 'drawn': drawn, 'identity': identity}]

    def evaluate(self, log, reviews):
        return ev.evaluate_identity_reviews(reviews, log, self.gt, margin_frames=0)

    # -- log schema --------------------------------------------------------
    def test_non_binary_valid_raises(self):
        for bad in ('invalid', '', 'true', '0.5', None, 2, -1, True):
            with self.subTest(value=bad):
                log = self.base_log()
                log[21] = {'frame': '21', 'state': 'TRACKING',
                           'target_track_id': '1', 'valid': bad}
                with self.assertRaises((ValueError, TypeError)):
                    self.evaluate(log, self.reviews())

    def test_binary_valid_forms_are_accepted(self):
        for good in ('0', '1', 0, 1):
            with self.subTest(value=good):
                log = self.base_log()
                log[21] = {'frame': '21', 'state': 'TRACKING',
                           'target_track_id': '1', 'valid': good}
                res = self.evaluate(log, self.reviews())
                self.assertEqual(len(res['reacquisition_episodes']), 1)

    def test_duplicate_log_frame_raises(self):
        log = self.base_log()
        log.append(dict(log[22]))
        with self.assertRaises(ValueError):
            self.evaluate(log, self.reviews())

    def test_unknown_state_raises(self):
        log = self.base_log()
        log[5]['state'] = 'FLYING'
        with self.assertRaises(ValueError):
            self.evaluate(log, self.reviews())

    def test_missing_log_column_raises(self):
        for column in ('frame', 'state', 'valid'):
            with self.subTest(column=column):
                log = self.base_log()
                del log[3][column]
                with self.assertRaises(ValueError):
                    self.evaluate(log, self.reviews())

    def test_bad_frame_values_raise(self):
        for bad in ('abc', '-1', 1.5):
            with self.subTest(frame=bad):
                log = self.base_log()
                log[3]['frame'] = bad
                with self.assertRaises(ValueError):
                    self.evaluate(log, self.reviews())

    def test_float_and_bool_frames_raise_without_coercion(self):
        # A single-row log is the counterexample: no duplicate-frame check can
        # mask the coercion. Old code turned 1.9->1, -0.5->0, True->1, 3.0->3.
        for bad in (1.9, -0.5, True, 3.0, 0.0):
            with self.subTest(frame=bad):
                with self.assertRaises(ValueError):
                    ev.validate_identity_log_rows([{'frame': bad, 'state': 'LOST', 'valid': '0'}])

    def test_review_float_and_bool_frames_raise_without_coercion(self):
        for bad in (1.9, -0.5, True, 3.0):
            with self.subTest(frame=bad):
                with self.assertRaises(ValueError):
                    ev.validate_identity_review_rows([{
                        'frame': bad, 'state': 'TRACKING', 'presence': 'present',
                        'drawn': True, 'identity': 'correct'}])

    def test_log_and_review_share_the_same_frame_contract(self):
        bad_values = (1.9, -0.5, True, 3.0, '1.9', '1e2', '+1', '1_0', None)
        for bad in bad_values:
            with self.subTest(frame=bad):
                with self.assertRaises(ValueError):
                    ev.validate_identity_log_rows([{'frame': bad, 'state': 'LOST', 'valid': '0'}])
                with self.assertRaises(ValueError):
                    ev.validate_identity_review_rows([{
                        'frame': bad, 'state': 'TRACKING', 'presence': 'present',
                        'drawn': True, 'identity': 'correct'}])

    def test_canonical_integer_frame_forms_still_accepted(self):
        for good in (0, 7, '0', '7', ' 7 '):
            with self.subTest(frame=good):
                out = ev.validate_identity_log_rows([{'frame': good, 'state': 'LOST', 'valid': '0'}])
                self.assertEqual(out[0]['frame'], str(int(str(good).strip())))
        out = ev.validate_identity_review_rows([{
            'frame': ' 7 ', 'state': 'TRACKING', 'presence': 'present',
            'drawn': True, 'identity': 'correct'}])
        self.assertEqual(out[0]['frame'], 7)

    def test_review_csv_loader_rejects_noncanonical_frames(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / 'reviews.csv'
        for bad in ('1.9', '1.0', '+1', '1_0', '1e2'):
            with self.subTest(frame=bad):
                path.write_text('frame,state,presence,drawn,identity\n'
                                f'{bad},TRACKING,present,1,correct\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    ev.load_identity_reviews(str(path))

    def test_caller_rows_are_not_mutated(self):
        log = self.base_log()
        log[21] = {'frame': '21', 'state': 'TRACKING', 'target_track_id': '1', 'valid': 1}
        before = [dict(row) for row in log]
        self.evaluate(log, self.reviews())
        self.assertEqual(log, before)

    # -- review schema -----------------------------------------------------
    def test_duplicate_review_frame_raises(self):
        reviews = self.reviews() + self.reviews()
        with self.assertRaises(ValueError):
            self.evaluate(self.base_log(), reviews)

    def test_conflicting_review_records_raise(self):
        reviews = self.reviews() + self.reviews(presence='uncertain', identity='uncertain',
                                               drawn=False)
        with self.assertRaises(ValueError):
            self.evaluate(self.base_log(), reviews)

    def test_invalid_review_fields_raise(self):
        cases = {
            'bad_presence': dict(presence='maybe'),
            'bad_identity': dict(identity='right'),
            'drawn_not_bool': dict(drawn='1'),
            'drawn_identity_contradiction': dict(identity='none', drawn=True),
            'absent_and_correct': dict(presence='absent'),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                with self.assertRaises((ValueError, TypeError)):
                    self.evaluate(self.base_log(), self.reviews(**overrides))

    def test_missing_review_column_raises(self):
        for column in ('frame', 'state', 'presence', 'drawn', 'identity'):
            with self.subTest(column=column):
                reviews = self.reviews()
                del reviews[0][column]
                with self.assertRaises(ValueError):
                    self.evaluate(self.base_log(), reviews)

    def test_validators_are_public_for_reuse(self):
        self.assertTrue(callable(ev.validate_identity_log_rows))
        self.assertTrue(callable(ev.validate_identity_review_rows))


class P1DefectRegressionTests(unittest.TestCase):
    """Permanent regression tests for all 4 Codex P1 defects."""

    def test_p1_01_invalid_evidence_bucket_raises_and_never_proves_negative(self):
        # 1. Invalid bucket 'uncertan' must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            ev.reacquisition_episode_evidence(
                start_f=10, end_f=15,
                bucket_by_frame={10: "uncertan"},
                lock_frames={10},
                log_frames={10, 11, 12, 13, 14, 15},
                onset_uncertainty_frames=0
            )
        self.assertIn("uncertan", str(ctx.exception))

        # 2. Other invalid buckets rejected
        for bad in ("unknown", "wrong", "", 123, None):
            with self.subTest(bucket=bad):
                with self.assertRaises((ValueError, TypeError)):
                    ev.reacquisition_episode_evidence(
                        10, 15, {10: bad}, lock_frames={10}, log_frames=set(range(10, 16))
                    )

        # 3. Valid 'unresolved' bucket remains uncertain, never a proven negative
        res = ev.reacquisition_episode_evidence(
            start_f=10, end_f=15,
            bucket_by_frame={10: "unresolved"},
            lock_frames={10},
            log_frames=set(range(10, 16)),
            onset_uncertainty_frames=0
        )
        self.assertFalse(res["no_wrong_lock_proven"])
        self.assertEqual(res["status"], "correct_not_observed_censored")
        self.assertEqual(res["evidence"]["resolved"], 0)
        self.assertEqual(res["evidence"]["unresolved"], 1)

        # 4. Empty evidence dictionary produces not_evaluated
        res_empty = ev.reacquisition_episode_evidence(
            10, 15, {}, lock_frames={10}, log_frames=set(range(10, 16))
        )
        self.assertEqual(res_empty["status"], "not_evaluated")
        self.assertTrue(res_empty["could_be_no_correct_lock"])
        self.assertFalse(res_empty["no_wrong_lock_proven"])

    def test_p1_02_viewpoint_changes_do_not_create_reacquisition_episodes(self):
        # View change front -> back: continuous visibility
        gt_view = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 10, "end_frame": 19, "target_visible": 1, "view": "back", "target_id": None},
        ]
        reacq = ev.build_reacquisition_episodes(gt_view)
        self.assertEqual(len(reacq["reacquisition_episodes"]), 0)
        self.assertEqual(len(reacq["viewpoint_changes"]), 1)
        self.assertEqual(reacq["viewpoint_changes"][0]["prev_view"], "front")

        # Occlusion recovery: partial (vis=2) -> visible (vis=1)
        gt_occl = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 10, "end_frame": 19, "target_visible": 2, "view": "front", "target_id": None},
            {"start_frame": 20, "end_frame": 29, "target_visible": 1, "view": "front", "target_id": None},
        ]
        reacq_occl = ev.build_reacquisition_episodes(gt_occl)
        self.assertEqual(len(reacq_occl["reacquisition_episodes"]), 0)
        self.assertEqual(len(reacq_occl["occlusion_recoveries"]), 1)

        # Unknown gap: [0-9, 1] -> unannotated gap [10-19] -> [20-29, 1]
        gt_gap = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 20, "end_frame": 29, "target_visible": 1, "view": "back", "target_id": None},
        ]
        reacq_gap = ev.build_reacquisition_episodes(gt_gap)
        self.assertEqual(len(reacq_gap["reacquisition_episodes"]), 0)
        self.assertEqual(len(reacq_gap["unclassifiable_reappearances"]), 1)

        # True out-of-view reacquisition: [0-9, 1] -> absent [10-19, 0] -> [20-29, 1]
        gt_true = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 10, "end_frame": 19, "target_visible": 0, "view": "none", "target_id": None},
            {"start_frame": 20, "end_frame": 29, "target_visible": 1, "view": "back", "target_id": None},
        ]
        reacq_true = ev.build_reacquisition_episodes(gt_true)
        self.assertEqual(len(reacq_true["reacquisition_episodes"]), 1)
        self.assertEqual(reacq_true["reacquisition_episodes"][0]["start_frame"], 20)
        self.assertEqual(reacq_true["reacquisition_episodes"][0]["prior_absence_start"], 10)
        self.assertEqual(reacq_true["reacquisition_episodes"][0]["prior_absence_end"], 19)

    def test_p1_03_missing_gt_coverage_is_unknown_not_absent(self):
        # Empty GT must return GT_UNKNOWN (-1), NOT absent (0)
        vis, view, tid = ev.get_gt_info_at_frame([], 5)
        self.assertEqual(vis, ev.GT_UNKNOWN)
        self.assertEqual(view, "unknown")
        self.assertIsNone(tid)

        # Frame outside annotated intervals returns GT_UNKNOWN
        gt = [{"start_frame": 0, "end_frame": 10, "target_visible": 1, "view": "front", "target_id": None}]
        vis_out, view_out, _ = ev.get_gt_info_at_frame(gt, 15)
        self.assertEqual(vis_out, ev.GT_UNKNOWN)

        # evaluate_log_gt: unannotated frames with TRACKING state are not false locks
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        log_path = Path(td.name) / "log.csv"
        gt_path = Path(td.name) / "gt.csv"

        # Log with frames 0..10, GT only covers 0..4 (vis=1). Frames 5..10 are unannotated!
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("frame,state,target_track_id,valid\n")
            for i in range(11):
                f.write(f"{i},TRACKING,1,1\n")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("start_frame,end_frame,target_visible,view\n")
            f.write("0,4,1,front\n")

        res = ev.evaluate_log_gt(str(log_path), str(gt_path), fps=30.0, margin_frames=0)
        self.assertEqual(res["false_lock_frames"], 0)
        self.assertEqual(res["unannotated_frames"], 6)
        self.assertEqual(res["unannotated_tracking_frames"], 6)

    def test_p1_04_legacy_path_strict_frame_validation(self):
        # get_tracking_segments rejects floats and booleans
        for bad in (1.9, -0.5, True, False, 3.0, "+1", "1_0", "1e2", "bad"):
            with self.subTest(frame=bad):
                with self.assertRaises(ValueError):
                    ev.get_tracking_segments([{"frame": bad, "state": "TRACKING", "target_track_id": "1"}])

        # parse_ground_truth rejects malformed start_frame, end_frame, target_visible
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        p = Path(td.name) / "gt_bad.csv"

        p.write_text("start_frame,end_frame,target_visible,view\n1.9,10,1,front\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ev.parse_ground_truth(str(p))

        p.write_text("start_frame,end_frame,target_visible,view\n0,10.5,1,front\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ev.parse_ground_truth(str(p))

        p.write_text("start_frame,end_frame,target_visible,view\n10,5,1,front\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ev.parse_ground_truth(str(p))

        # Contradictory overlapping GT intervals
        gt_overlap = [
            {"start_frame": 0, "end_frame": 10, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 8, "end_frame": 15, "target_visible": 0, "view": "none", "target_id": None},
        ]
        with self.assertRaises(ValueError):
            ev.validate_ground_truth_intervals(gt_overlap)


class ScientificMetricsContractTests(unittest.TestCase):
    """Tests verifying mathematical definitions, denominators, and metamorphic invariants."""

    def test_present_target_lock_recall_vs_conditional_recall(self):
        # When correct_output_not_lock or wrong_output_not_lock_present > 0,
        # unconditional recall is strictly <= conditional recall.
        # Use interior frames (1, 2, 3) so they are not boundary frames even at margin=0.
        gt = [{"start_frame": 0, "end_frame": 4, "target_visible": 1, "view": "front", "target_id": None}]
        log = [
            {"frame": "0", "state": "LOST", "target_track_id": "1", "valid": "0"},
            {"frame": "1", "state": "INITIALIZING", "target_track_id": "1", "valid": "1"},
            {"frame": "2", "state": "TRACKING", "target_track_id": "1", "valid": "1"},
            {"frame": "3", "state": "TRACKING", "target_track_id": "1", "valid": "1"},
            {"frame": "4", "state": "LOST", "target_track_id": "1", "valid": "0"},
        ]
        reviews = [
            {"frame": 1, "state": "INITIALIZING", "presence": "present", "drawn": True, "identity": "correct"},
            {"frame": 2, "state": "TRACKING", "presence": "present", "drawn": True, "identity": "correct"},
            {"frame": 3, "state": "TRACKING", "presence": "present", "drawn": True, "identity": "correct"},
        ]
        res = ev.evaluate_identity_reviews(reviews, log, gt, margin_frames=0)

        # correct_lock = 2 (frames 2, 3), correct_output_not_lock = 1 (frame 1)
        # conditional lock_recall = 2 / 2 = 1.0
        # unconditional present_target_lock_recall = 2 / 3 = 0.6667
        self.assertAlmostEqual(res["lock_recall"], 1.0)
        self.assertAlmostEqual(res["present_target_lock_recall"], 2 / 3)
        self.assertAlmostEqual(res["present_target_output_coverage"], 1.0)
        self.assertLess(res["present_target_lock_recall"], res["lock_recall"])

    def test_metamorphic_invariants(self):
        # Invariant 1: Splitting visible interval by view change must not change episode count
        gt_orig = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 10, "end_frame": 19, "target_visible": 0, "view": "none", "target_id": None},
            {"start_frame": 20, "end_frame": 29, "target_visible": 1, "view": "back", "target_id": None},
        ]
        gt_split = [
            {"start_frame": 0, "end_frame": 9, "target_visible": 1, "view": "front", "target_id": None},
            {"start_frame": 10, "end_frame": 19, "target_visible": 0, "view": "none", "target_id": None},
            {"start_frame": 20, "end_frame": 24, "target_visible": 1, "view": "back", "target_id": None},
            {"start_frame": 25, "end_frame": 29, "target_visible": 1, "view": "side", "target_id": None},
        ]
        reacq_orig = ev.build_reacquisition_episodes(gt_orig)
        reacq_split = ev.build_reacquisition_episodes(gt_split)
        self.assertEqual(len(reacq_orig["reacquisition_episodes"]), len(reacq_split["reacquisition_episodes"]))
        self.assertEqual(len(reacq_split["viewpoint_changes"]), 1)

        # Invariant 2: Replacing absent GT with unknown must never increase false-lock count
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        p_log = Path(td.name) / "log.csv"
        p_gt_abs = Path(td.name) / "gt_abs.csv"
        p_gt_unk = Path(td.name) / "gt_unk.csv"

        # 5 frames 0..4; frames 1, 2, 3 are interior frames outside margin 0
        log_lines = ["frame,state,target_track_id,valid\n"] + [f"{i},TRACKING,1,1\n" for i in range(5)]
        p_log.write_text("".join(log_lines), encoding="utf-8")
        p_gt_abs.write_text("start_frame,end_frame,target_visible,view\n0,4,0,none\n", encoding="utf-8")
        p_gt_unk.write_text("start_frame,end_frame,target_visible,view\n0,4,-1,unknown\n", encoding="utf-8")

        res_abs = ev.evaluate_log_gt(str(p_log), str(p_gt_abs), fps=30.0, margin_frames=0)
        res_unk = ev.evaluate_log_gt(str(p_log), str(p_gt_unk), fps=30.0, margin_frames=0)
        self.assertEqual(res_abs["false_lock_frames"], 3)
        self.assertEqual(res_unk["false_lock_frames"], 0)
        self.assertGreater(res_abs["false_lock_frames"], res_unk["false_lock_frames"])


if __name__ == '__main__':
    unittest.main()
