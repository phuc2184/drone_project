import csv, importlib.util, json, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
review=load('review',ROOT/'research_handoff/review/identity_review.py')
screen=load('screen',ROOT/'research_handoff/identity_screen.py')

class ReviewTests(unittest.TestCase):
    def label(self, **changes):
        r=dict(schema_version=review.SCHEMA,record_type='annotation',annotation_mode='blind',session_id='s',frame=0,label='TARGET',box=[1,2,10,20],reviewer_id='human',evidence='independent reference',notes='',recorded_at_utc='2026-10-29T00:00:00+00:00',video_sha256='hash')
        r.update(changes); return r
    def test_labels_and_bounds(self):
        review.validate_label(self.label(),100,100,2,'hash')
        for changes in [dict(box=[0,0,101,1]),dict(box=[0,0,float('nan'),1]),dict(label='UNKNOWN'),dict(label='TARGET_NOT_VISIBLE',box=[1,1,2,2]),dict(frame=True),dict(evidence=''),dict(video_sha256='other')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): review.validate_label(self.label(**changes),100,100,2,'hash')
        review.validate_label(self.label(label='UNKNOWN',box=None),100,100,2,'hash')
    def test_geometry_is_not_identity(self):
        a=self.label(); self.assertEqual(review.category(a,True,[1,2,10,20]),'overlap_agreement')
        self.assertEqual(review.category(a,True,[40,40,50,50]),'low_overlap_review')
        self.assertEqual(review.category(a,True,None),'output_box_unavailable')
        self.assertEqual(review.category(self.label(label='TARGET_NOT_VISIBLE',box=None),True,None),'visibility_unknown_output')
    def test_queue_priorities(self):
        rows=review.read_log(ROOT/'logs/tracking_log_20261026_023427.csv')
        self.assertEqual(review.queue(rows)[:6],[1061,1062,1066,1140,1141,1145])
        self.assertTrue(all(0 <= x < len(rows) for x in review.queue(rows)))
    def test_session_unknown_and_conflicts(self):
        video=ROOT/'videos/5.mp4'; log=ROOT/'logs/tracking_log_20261026_023427.csv'
        vh=review.digest(video); lh=review.digest(log)
        a=self.label(video_sha256=vh,label='UNKNOWN',box=None)
        c=dict(schema_version=review.SCHEMA,record_type='comparison',session_id='s',frame=0,reviewer_id='human',recorded_at_utc='now',video_sha256=vh,log_sha256=lh,manifest_sha256=None,drawn='1',state='INITIALIZING',identity='uncertain',geometry_category='visibility_unknown_output')
        lock=dict(schema_version=review.SCHEMA,record_type='annotation_lock',session_id='s',recorded_at_utc='now',video_sha256=vh,annotation_count=1)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'session.jsonl'
            def write(rows): p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            write([a,lock,c]); rows,audit=screen.load_review_session(p,video,log)
            self.assertEqual(rows[0]['presence'],'uncertain'); self.assertEqual(screen.reviewed_identity_metrics(rows)['counts']['unresolved_output'],1)
            self.assertEqual(audit['comparisons'],1)
            write([a,lock,dict(c,identity='wrong')])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)
            write([a,a])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)
            write([a,lock,dict(c,log_sha256='bad')])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)
            write([a,c])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)
            write([a,lock,a])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)
            write([c,a])
            with self.assertRaises(ValueError): screen.load_review_session(p,video,log)

if __name__ == '__main__': unittest.main()
