#!/usr/bin/env python3
"""Join completed blind reviews to output role; never infer absence from NOT_VISIBLE."""
import argparse
import csv
import json
from pathlib import Path
import sys
from review_frames import candidate_boxes, digest, validate_record


def convert(manifest, manifest_hash, reviews, width, height, state='TRACKING'):
    frames = {r['frame']: r for r in manifest['frames']}
    if len(frames) != len(manifest['frames']): raise ValueError('duplicate manifest frames')
    scenes, candidates, reviewer = {}, {}, None
    for r in reviews:
        validate_record(r)
        if r['manifest_sha256'] != manifest_hash or r['video_sha256'] != manifest['video_sha256']:
            raise ValueError('review provenance mismatch')
        if reviewer is None: reviewer = r['reviewer']
        if reviewer != r['reviewer']: raise ValueError('one reviewer per session')
        f = r['frame']
        if f not in frames: raise ValueError('review frame missing from manifest')
        if r['record_type'] == 'scene':
            if f in scenes: raise ValueError('duplicate scene review')
            scenes[f] = r
        else:
            k = (f, r['candidate_index'])
            if k in candidates: raise ValueError('duplicate candidate review')
            boxes = candidate_boxes(frames[f], width, height)
            if r['candidate_index'] >= len(boxes) or list(boxes[r['candidate_index']]) != r['box']:
                raise ValueError('candidate geometry mismatch')
            candidates[k] = r
    for (f, _), r in candidates.items():
        if f not in scenes or r['scene_label'] != scenes[f]['scene_label']:
            raise ValueError('missing or inconsistent scene review')
    rows = []
    for f, scene in sorted(scenes.items()):
        src = frames[f]
        if src['state'] != state: continue
        presence = 'present' if scene['scene_label'] == 'TARGET' and scene['certainty'] != 'LOW' else 'uncertain'
        output = src['output_box']
        if output is None:
            rows.append({'frame': f, 'state': state, 'presence': presence, 'drawn': '0', 'identity': 'none'})
            continue
        output_box = candidate_boxes({'tracks': [], 'output_box': output}, width, height)[0]
        boxes = candidate_boxes(src, width, height)
        index = boxes.index(output_box)
        review = candidates.get((f, index))
        identity = 'uncertain'
        if review and review['certainty'] != 'LOW':
            identity = {'TARGET':'correct', 'NON_TARGET':'wrong', 'UNKNOWN':'uncertain'}[review['identity_label']]
        rows.append({'frame': f, 'state': state, 'presence': presence, 'drawn': '1', 'identity': identity})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--reviews', required=True)
    p.add_argument('--video', required=True); p.add_argument('--out', required=True)
    p.add_argument('--state', default='TRACKING')
    a = p.parse_args()
    if Path(a.out).exists(): p.error('output already exists')
    with open(a.manifest, encoding='utf-8') as f: manifest = json.load(f)
    if digest(a.video) != manifest['video_sha256']: raise ValueError('video hash mismatch')
    with open(a.reviews, encoding='utf-8') as f: reviews = [json.loads(line) for line in f if line.strip()]
    if not reviews: raise ValueError('review session is empty')
    import cv2
    cap = cv2.VideoCapture(a.video)
    ok, frame = cap.read(); cap.release()
    if not ok: raise ValueError('cannot decode source video')
    height, width = frame.shape[:2]
    rows = convert(manifest, digest(a.manifest), reviews, width, height, a.state)
    if not rows: raise ValueError('no reviewed scenes for chosen state')
    with open(a.out, 'x', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=['frame','state','presence','drawn','identity'])
        writer.writeheader(); writer.writerows(rows)
    print(f'wrote {len(rows)} reviewed scene rows to {a.out}; missing candidates and LOW certainty stay uncertain')
    print('Diagnostic selection is nonrepresentative. Do not report population performance.')

if __name__ == '__main__': main()
