#!/usr/bin/env python3
"""Blind, local-only diagnostic review. Never modifies video or candidate manifest."""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidate_boxes(frame_record, width, height):
    """Return anonymous boxes in geometric order; output role and track ID stay hidden."""
    boxes = [t['box'] for t in frame_record['tracks']]
    if frame_record.get('output_box') is not None:
        boxes.append(frame_record['output_box'])
    valid = set()
    for b in boxes:
        if len(b) != 4 or any(not isinstance(n, int) or isinstance(n, bool) for n in b):
            raise ValueError('candidate box must contain four integers')
        x1, y1, x2, y2 = b
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('candidate box is empty or outside image')
        valid.add((x1, y1, x2, y2))
    return sorted(valid)


def validate_record(r):
    for k in ('reviewer', 'label_source', 'video_sha256', 'manifest_sha256', 'recorded_at_utc'):
        if not isinstance(r.get(k), str) or not r[k].strip():
            raise ValueError(f'missing {k}')
    if not isinstance(r.get('frame'), int) or r['frame'] < 0:
        raise ValueError('invalid frame')
    if r.get('scene_label') not in ('TARGET', 'NOT_VISIBLE', 'UNKNOWN'):
        raise ValueError('invalid scene label')
    if r.get('certainty') not in ('HIGH', 'MEDIUM', 'LOW'):
        raise ValueError('invalid certainty')
    if r['record_type'] == 'candidate':
        if r.get('identity_label') not in ('TARGET', 'NON_TARGET', 'UNKNOWN'):
            raise ValueError('invalid candidate identity')
        if r['scene_label'] != 'TARGET' and r['identity_label'] == 'TARGET':
            raise ValueError('candidate TARGET contradicts scene label')
        if not isinstance(r.get('candidate_index'), int) or r['candidate_index'] < 0:
            raise ValueError('invalid candidate index')
        if len(r.get('box', [])) != 4: raise ValueError('invalid candidate box')
    elif r['record_type'] != 'scene':
        raise ValueError('invalid record type')


def append_record(f, record):
    validate_record(record)
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    f.flush()
    os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--label-source', required=True, help='e.g. video plus independent target reference, version')
    p.add_argument('--out', required=True, help='new JSONL session; will not overwrite')
    p.add_argument('--frames', help='optional comma-separated subset')
    p.add_argument('--reference-frame', type=int, default=0, help='raw reference frame, independently chosen')
    a = p.parse_args()
    if not a.reviewer.strip() or not a.label_source.strip() or a.reference_frame < 0:
        p.error('reviewer, label source and valid reference frame required')
    if Path(a.out).exists(): p.error('output already exists; use a new session file')
    with open(a.manifest, encoding='utf-8') as f: manifest = json.load(f)
    if digest(a.video) != manifest['video_sha256']: raise ValueError('video hash mismatch')
    fps = float(manifest['fps_nominal'])
    if not math.isfinite(fps) or fps <= 0: raise ValueError('invalid nominal FPS')
    records = manifest['frames']
    by_frame = {r['frame']: r for r in records}
    if len(by_frame) != len(records): raise ValueError('duplicate manifest frame')
    selected = sorted(by_frame if not a.frames else set(int(x) for x in a.frames.split(',')))
    if not selected or any(i not in by_frame for i in selected): raise ValueError('requested frame missing')
    import cv2
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise RuntimeError('video cannot be opened')
    # Sequential decode: seeking is known to differ for this source.
    wanted = set(selected + [a.reference_frame])
    images = {}
    try:
        for i in range(max(wanted) + 1):
            ok, image = cap.read()
            if not ok: raise RuntimeError(f'decode failed at frame {i}')
            if i in wanted: images[i] = image
    finally: cap.release()
    manifest_hash = digest(a.manifest)
    cv2.namedWindow('Review', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Reference raw video', cv2.WINDOW_NORMAL)
    cv2.imshow('Reference raw video', images[a.reference_frame])
    print('Reference is raw image, not a verified identity label. Use independent evidence; see protocol.', flush=True)
    print('Scene: p=target visible, n=not visible (not necessarily absent), u=unknown, s=skip, q=quit. '
          'Candidate: t=TARGET, n=NON_TARGET, u=UNKNOWN, s=skip, q=quit. '
          'Certainty: 1=HIGH, 2=MEDIUM, 3=LOW, s=skip, q=quit.', flush=True)

    def key_choice(prompt, mapping):
        print(prompt, flush=True)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27): raise KeyboardInterrupt
            if key == ord('s'): return None
            if key in mapping: return mapping[key]

    certs = {ord('1'): 'HIGH', ord('2'): 'MEDIUM', ord('3'): 'LOW'}
    try:
        with open(a.out, 'x', encoding='utf-8') as f:
            for number in selected:
                image = images[number]
                h, w = image.shape[:2]
                boxes = candidate_boxes(by_frame[number], w, h)
                raw = image.copy()
                cv2.putText(raw, f'Frame {number}: p visible / n not visible / u unknown', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('Review', raw)
                scene = key_choice(f'Frame {number}: scene?', {ord('p'):'TARGET', ord('n'):'NOT_VISIBLE', ord('u'):'UNKNOWN'})
                if scene is None: continue
                certainty = key_choice('Scene certainty?', certs)
                if certainty is None: continue
                base = {'schema_version': 1, 'frame': number, 'timestamp_seconds_nominal': round(number / fps, 6),
                        'timestamp_basis': 'zero-based frame / nominal metadata fps; not source PTS',
                        'reviewer': a.reviewer, 'label_source': a.label_source, 'video_sha256': manifest['video_sha256'],
                        'manifest_sha256': manifest_hash, 'reference_frame': a.reference_frame,
                        'scene_label': scene, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
                append_record(f, {**base, 'record_type':'scene', 'certainty':certainty})
                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    canvas = image.copy()
                    cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,255), 3)
                    cv2.putText(canvas, f'Frame {number}, region {index+1}/{len(boxes)}: t/n/u', (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.imshow('Review', canvas)
                    cv2.imshow('Region', image[y1:y2, x1:x2])
                    label = key_choice(f'Frame {number}, region {index+1}: identity?',
                                       {ord('t'):'TARGET', ord('n'):'NON_TARGET', ord('u'):'UNKNOWN'})
                    if label is None: continue
                    if label == 'TARGET' and scene != 'TARGET':
                        print('Conflicts with scene: skipped; review frame again in a new session.', flush=True)
                        continue
                    confidence = key_choice('Candidate certainty?', certs)
                    if confidence is None: continue
                    append_record(f, {**base, 'record_type':'candidate', 'candidate_index':index,
                                      'box':list(box), 'identity_label':label, 'certainty':confidence,
                                      'recorded_at_utc':datetime.now(timezone.utc).isoformat()})
    except KeyboardInterrupt:
        print('Stopped; completed records have been saved.', flush=True)
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__': main()
