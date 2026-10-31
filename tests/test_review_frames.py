import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('review_frames', Path(__file__).resolve().parents[1] / 'research_handoff/review/review_frames.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

class ReviewToolTests(unittest.TestCase):
    def test_anonymous_boxes_sorted_clipped_deduplicated(self):
        rec = {'tracks':[{'track_id':204,'box':[4,3,20,30]},{'track_id':246,'box':[-1,1,3,5]}], 'output_box':[4,3,20,30]}
        self.assertEqual(r.candidate_boxes(rec, 10, 20), [(0,1,3,5),(4,3,10,20)])
        with self.assertRaises(ValueError): r.candidate_boxes({'tracks':[{'box':[12,1,15,3]}]}, 10,20)

    def test_record_validation_prevents_contradictory_claim(self):
        base=dict(reviewer='a', label_source='raw', video_sha256='v', manifest_sha256='m', recorded_at_utc='now',frame=1, scene_label='NOT_VISIBLE', certainty='HIGH',record_type='candidate', identity_label='TARGET',candidate_index=0,box=[0,0,1,1])
        with self.assertRaises(ValueError): r.validate_record(base)
        base['identity_label']='UNKNOWN'; r.validate_record(base)
        base['certainty']=''; self.assertRaises(ValueError,r.validate_record,base)

    def test_session_blinds_identity_and_saves_incrementally(self):
        class Capture:
            def __init__(self, path): self.n=0
            def isOpened(self): return True
            def read(self): self.n+=1; return True, np.zeros((20,20,3), dtype=np.uint8)
            def release(self): pass
        class FakeCV:
            WINDOW_NORMAL=0; FONT_HERSHEY_SIMPLEX=0
            VideoCapture=Capture
            def __init__(self): self.keys=iter(map(ord, 'p1n2'))
            def namedWindow(self,*a): pass
            def imshow(self,*a): pass
            def putText(self,*a): pass
            def rectangle(self,*a): pass
            def waitKey(self,*a): return next(self.keys)
            def destroyAllWindows(self): pass
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'video'; video.write_bytes(b'video')
            manifest=Path(d)/'manifest.json'; out=Path(d)/'review.jsonl'
            manifest.write_text(json.dumps({'video_sha256':r.digest(video),'fps_nominal':30,'frames':[{'frame':1,'state':'TRACKING','tracks':[{'track_id':204,'box':[1,1,8,8]}],'output_box':[1,1,8,8]}]}))
            args=['review','--manifest',str(manifest),'--video',str(video),'--reviewer','human','--label-source','raw and reference','--out',str(out)]
            with patch.dict(sys.modules, {'cv2':FakeCV()}), patch.object(sys,'argv',args): r.main()
            rows=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual([x['record_type'] for x in rows],['scene','candidate'])
            self.assertEqual(rows[1]['scene_label'],'TARGET')
            self.assertEqual(rows[1]['identity_label'],'NON_TARGET')
            self.assertNotIn('track_id',rows[1]); self.assertNotIn('state',rows[1]); self.assertNotIn('output_box',rows[1])
            self.assertEqual(rows[1]['video_sha256'],r.digest(video))
            with patch.dict(sys.modules, {'cv2':FakeCV()}), patch.object(sys,'argv',args), self.assertRaises(SystemExit): r.main()

class ReviewJoinTests(unittest.TestCase):
    def test_present_wrong_output_is_counted_and_unknown_stays_unknown(self):
        review_dir = str(Path(__file__).resolve().parents[1] / 'research_handoff/review')
        sys.path.insert(0, review_dir)
        try:
            import reviews_to_metrics as j
            manifest={'video_sha256':'vid','frames':[{'frame':1,'state':'TRACKING','tracks':[{'track_id':9,'box':[1,1,8,8]}],'output_box':[1,1,8,8]}]}
            base={'reviewer':'r','label_source':'video','video_sha256':'vid','manifest_sha256':'mh','recorded_at_utc':'now','frame':1,'scene_label':'TARGET','certainty':'HIGH'}
            scene={**base,'record_type':'scene'}
            candidate={**base,'record_type':'candidate','identity_label':'NON_TARGET','candidate_index':0,'box':[1,1,8,8]}
            rows=j.convert(manifest,'mh',[scene,candidate],10,10)
            self.assertEqual(rows[0]['presence'],'present')
            self.assertEqual(rows[0]['identity'],'wrong')
            screen_spec = importlib.util.spec_from_file_location('screen_check', Path(__file__).resolve().parents[1] / 'research_handoff/identity_screen.py')
            screen = importlib.util.module_from_spec(screen_spec); screen_spec.loader.exec_module(screen)
            self.assertEqual(screen.reviewed_identity_metrics(rows)['counts']['wrong_present'], 1)
            candidate['certainty']='LOW'
            self.assertEqual(j.convert(manifest,'mh',[scene,candidate],10,10)[0]['identity'],'uncertain')
            scene['scene_label']='NOT_VISIBLE'; candidate['scene_label']='NOT_VISIBLE'
            self.assertEqual(j.convert(manifest,'mh',[scene,candidate],10,10)[0]['presence'],'uncertain')
            candidate['box']=[1,1,9,9]
            with self.assertRaises(ValueError): j.convert(manifest,'mh',[scene,candidate],10,10)
        finally: sys.path.remove(review_dir)

if __name__ == '__main__': unittest.main()
