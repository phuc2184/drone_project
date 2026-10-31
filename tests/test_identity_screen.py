import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('identity_screen', Path(__file__).resolve().parents[1] / 'research_handoff/identity_screen.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def item(status='box', state='TRACKING', drawn='1', iou='0.05'):
    label = dict(status=status, view='front', x1='0', y1='0', x2='100', y2='200')
    obs = dict(status=status, view='front', state=state, drawn=drawn, iou_drawn=iou)
    log = dict(state=state, valid=drawn)
    return label, obs, log


class IdentityScreenTests(unittest.TestCase):
    def run_one(self, *args):
        a, b, c = item(*args)
        return m.screen({1:a}, {1:b}, {1:c})['frames'][0]['category']

    def test_target_present_wrong_box_is_review_candidate(self):
        self.assertEqual(self.run_one(), 'low_overlap_review')

    def test_overlap_thresholds_and_missing(self):
        self.assertEqual(self.run_one('box', 'TRACKING', '1', '0.1'), 'ambiguous_overlap')
        self.assertEqual(self.run_one('box', 'TRACKING', '1', '0.5'), 'overlap_agreement')
        self.assertEqual(self.run_one('box', 'TRACKING', '0', ''), 'no_output')

    def test_not_visible_is_not_proven_absent(self):
        self.assertEqual(self.run_one('not_visible', 'TRACKING', '1', ''), 'visibility_unknown_output')

    def test_initialization_is_separate(self):
        a,b,c = item('box', 'INITIALIZING', '1', '0.8')
        result = m.screen({1:a}, {1:b}, {1:c})['counts']
        self.assertEqual(result['INITIALIZING/front']['overlap_agreement'], 1)
        self.assertNotIn('TRACKING/front', result)

    def test_mismatch_and_invalid_values_rejected(self):
        a,b,c = item()
        c['valid'] = '0'
        with self.assertRaises(ValueError): m.screen({1:a}, {1:b}, {1:c})
        c['valid'] = '1'; b['iou_drawn'] = 'nan'
        with self.assertRaises(ValueError): m.screen({1:a}, {1:b}, {1:c})
        with self.assertRaises(ValueError): m.screen({1:a}, {}, {1:c})

    def test_reviewed_wrong_when_target_present_and_uncertainty(self):
        rows = [dict(frame='1', presence='present', drawn='1', identity='wrong'),
                dict(frame='2', presence='present', drawn='1', identity='correct'),
                dict(frame='3', presence='uncertain', drawn='1', identity='wrong'),
                dict(frame='4', presence='present', drawn='0', identity='none')]
        result = m.reviewed_identity_metrics(rows)
        self.assertEqual(result['counts']['wrong_present'], 1)
        self.assertEqual(result['counts']['unresolved_output'], 1)
        self.assertEqual(result['wrong_fraction_among_reviewed_outputs'], 0.5)

    def test_reviewed_identity_rejects_contradiction_and_empty_denominator(self):
        with self.assertRaises(ValueError):
            m.reviewed_identity_metrics([dict(frame='1', presence='absent', drawn='1', identity='correct')])
        self.assertIsNone(m.reviewed_identity_metrics([])['wrong_fraction_among_reviewed_outputs'])

    def test_duplicate_frames_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'duplicate.csv'
            path.write_text('frame\n1\n1\n')
            with self.assertRaises(ValueError): m.read_index(path)

if __name__ == '__main__': unittest.main()
