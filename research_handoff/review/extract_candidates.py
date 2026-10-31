#!/usr/bin/env python3
"""Replay locally to collect diagnostic boxes; output only after log consistency checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--frames', required=True, help='comma-separated zero-based frame indices')
    p.add_argument('--out', required=True, help='new JSON file, never overwrites')
    a = p.parse_args()
    frames = sorted(set(int(x) for x in a.frames.split(',')))
    if not frames or frames[0] < 0: p.error('frames must be nonnegative')
    if Path(a.out).exists(): p.error('output already exists')
    import cv2
    from follow_target import (DetectorTracker, TargetManager, PersonReIDExtractor,
                               QualityFilter, TargetState, REID_MODEL_NAME, REID_WEIGHTS_PATH,
                               YOLO_MODEL_PATH)
    with open(a.log, newline='') as f:
        rows = list(csv.DictReader(f))
    if any(int(r['frame']) != i for i, r in enumerate(rows)) or frames[-1] >= len(rows):
        raise ValueError('log must be sequential and cover requested frames')
    result = []
    with tempfile.TemporaryDirectory(prefix='uav_review_gallery_') as gallery:
        detector = DetectorTracker()
        extractor = PersonReIDExtractor(model_name=REID_MODEL_NAME, weights_path=REID_WEIGHTS_PATH)
        manager = TargetManager(feature_extractor=extractor, quality_filter=QualityFilter(),
                                use_view_chain=False, use_color_cue=False, gallery_debug_dir=gallery)
        manager.target_track_id = 1
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened(): raise RuntimeError('cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            for index in range(frames[-1] + 1):
                ok, frame = cap.read()
                if not ok: raise RuntimeError(f'decode failure at frame {index}')
                tracks = detector.track(frame)
                if manager.state == TargetState.IDLE and manager.target_track_id is not None:
                    for track in tracks:
                        if track.track_id == manager.target_track_id:
                            manager.select_target(track.track_id, track.box, index, frame)
                            break
                out = manager.update(tracks, frame, index, fps)
                log = rows[index]
                if (out['state'] != log['state'] or str(manager.target_track_id) != log['target_track_id']
                        or str(int(out['target_box'] is not None)) != log['valid']):
                    raise RuntimeError(f'replay differs from source log at frame {index}')
                if index in frames:
                    result.append({'frame': index, 'state': out['state'],
                                   'output_box': list(out['target_box']) if out['target_box'] is not None else None,
                                   'tracks': [{'track_id': t.track_id, 'box': list(t.box)} for t in tracks]})
                if index % 300 == 0: print(f'checked {index}/{frames[-1]}', file=sys.stderr, flush=True)
        finally:
            cap.release()
    record = {'purpose': 'diagnostic, targeted nonrepresentative sample; not held-out',
              'video': str(Path(a.video)), 'video_sha256': digest(a.video), 'log': str(Path(a.log)),
              'log_sha256': digest(a.log), 'fps_nominal': fps, 'frames_checked': frames[-1] + 1,
              'code_sha256': digest(ROOT / 'follow_target.py'), 'extractor_sha256': digest(__file__),
              'yolo_sha256': digest(YOLO_MODEL_PATH), 'reid_sha256': digest(REID_WEIGHTS_PATH),
              'config': {'target_initial_track_id': 1, 'view_chain': False, 'color_cue': False,
                         'state_and_target_id_and_valid_checked_each_frame': True}, 'frames': result}
    with open(a.out, 'x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
        f.flush(); os.fsync(f.fileno())
    print(f'wrote {a.out}; {len(result)} diagnostic frames')

if __name__ == '__main__': main()
