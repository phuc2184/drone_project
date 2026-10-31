#!/usr/bin/env python3
"""Screen labeled-frame output geometry; findings require independent identity review."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_index(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        frame = int(row['frame'])
        if frame < 0 or frame in result:
            raise ValueError(f'invalid or duplicate frame {frame} in {path}')
        result[frame] = row
    return result


def screen(labels, observations, logs, low=0.1, match=0.5):
    if not (0 <= low < match <= 1):
        raise ValueError('thresholds must satisfy 0 <= low < match <= 1')
    if set(labels) != set(observations):
        raise ValueError('label and observation frame sets differ')
    counts = defaultdict(Counter)
    details = []
    for frame, label in sorted(labels.items()):
        obs = observations[frame]
        if frame not in logs:
            raise ValueError(f'missing log frame {frame}')
        log = logs[frame]
        if label['status'] not in ('box', 'not_visible'):
            raise ValueError(f'unknown label status at {frame}')
        if (label['status'], label['view']) != (obs['status'], obs['view']):
            raise ValueError(f'label/observation mismatch at {frame}')
        if obs['state'] != log['state'] or obs['drawn'] not in ('0', '1') or obs['drawn'] != log['valid']:
            raise ValueError(f'observation/log mismatch at {frame}')
        drawn = obs['drawn'] == '1'
        iou = obs['iou_drawn'].strip()
        if not drawn and iou:
            raise ValueError(f'unexpected IoU without output at {frame}')
        if label['status'] == 'box':
            coords = [float(label[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            if not all(math.isfinite(v) for v in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError(f'invalid box at {frame}')
            if not drawn:
                category = 'no_output'
            else:
                value = float(iou)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'invalid IoU at {frame}')
                category = 'low_overlap_review' if value < low else 'overlap_agreement' if value >= match else 'ambiguous_overlap'
        else:
            if iou:
                raise ValueError(f'unexpected IoU for not_visible at {frame}')
            category = 'visibility_unknown_output' if drawn else 'visibility_unknown_no_output'
        state = obs['state']
        counts[f'{state}/{label["view"]}'][category] += 1
        details.append({'frame': frame, 'state': state, 'view': label['view'], 'category': category, 'iou': iou or None})
    return {'counts': {k: dict(v) for k, v in sorted(counts.items())}, 'frames': details}


def reviewed_identity_metrics(rows):
    """Count human-reviewed identity outcomes; no inference from tracker IDs/IoU.

    Rows require presence=present/absent/uncertain, drawn=0/1, and
    identity=correct/wrong/uncertain/none. Rows may be temporally correlated; no independence is assumed.
    """
    counts = Counter()
    seen = set()
    for r in rows:
        frame = int(r['frame'])
        if frame < 0 or frame in seen:
            raise ValueError('duplicate or invalid frame')
        seen.add(frame)
        presence, drawn, identity = r['presence'], r['drawn'], r['identity']
        if presence not in ('present', 'absent', 'uncertain') or drawn not in ('0', '1'):
            raise ValueError('invalid presence or drawn')
        if (drawn == '0' and identity != 'none') or (drawn == '1' and identity not in ('correct', 'wrong', 'uncertain')):
            raise ValueError('invalid identity for drawn status')
        if presence == 'absent' and identity == 'correct':
            raise ValueError('correct identity contradicts confirmed absence')
        counts['samples'] += 1
        if drawn == '0':
            counts['no_output'] += 1
        elif identity == 'uncertain' or presence == 'uncertain':
            counts['unresolved_output'] += 1
        else:
            counts['reviewed_output'] += 1
            if identity == 'wrong':
                counts['wrong_output'] += 1
                counts['wrong_present' if presence == 'present' else 'wrong_absent'] += 1
            else:
                counts['correct_output'] += 1
    denominator = counts['reviewed_output']
    return {'counts': dict(counts), 'wrong_fraction_among_reviewed_outputs': counts['wrong_output'] / denominator if denominator else None,
            'warning': 'Sample-specific; provenance and sampling are not verified here. Unresolved outputs are excluded; frames may be correlated. Do not infer population rates from diagnostic samples.'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_review_session(path, video, log, manifest=None):
    """Validate an append-only human review session; never infer identity from geometry."""
    import importlib.util
    review_path = Path(__file__).resolve().parent / 'review/identity_review.py'
    spec = importlib.util.spec_from_file_location('identity_review_schema', review_path)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    import cv2
    video_hash, log_hash = sha256(video), sha256(log)
    rows = review.read_log(log)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('cannot open video')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    manifests = review.read_manifest(manifest, video_hash, log_hash, rows, width, height)
    manifest_hash = sha256(manifest) if manifest else None
    annotations, comparisons = {}, {}
    session = None
    locked = False
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or r.get('schema_version') != review.SCHEMA: raise ValueError('invalid schema')
                if not isinstance(r.get('session_id'), str) or not r['session_id']: raise ValueError('missing session')
                if session is None: session = r['session_id']
                if r['session_id'] != session: raise ValueError('mixed sessions')
                if r.get('record_type') == 'annotation_lock':
                    if locked or not annotations or r.get('annotation_count') != len(annotations) or r.get('video_sha256') != video_hash or not r.get('recorded_at_utc'): raise ValueError('invalid annotation lock')
                    locked = True
                    continue
                n = r.get('frame')
                if type(n) is not int or not 0 <= n < len(rows): raise ValueError('invalid frame')
                if r.get('record_type') == 'annotation':
                    if locked: raise ValueError('annotation after lock')
                    review.validate_label(r, width, height, len(rows), video_hash)
                    if n in annotations: raise ValueError('duplicate or conflicting annotation')
                    annotations[n] = r
                elif r.get('record_type') == 'comparison':
                    if not locked or n not in annotations or n in comparisons: raise ValueError('comparison missing annotation or duplicate')
                    a = annotations[n]; row = rows[n]; m = manifests.get(n)
                    if (r.get('reviewer_id') != a['reviewer_id'] or not isinstance(r.get('recorded_at_utc'), str) or not r['recorded_at_utc'].strip()
                        or r.get('video_sha256') != video_hash or r.get('log_sha256') != log_hash or r.get('manifest_sha256') != manifest_hash
                        or r.get('state') != row['state'] or r.get('drawn') != row['valid']): raise ValueError('comparison provenance mismatch')
                    identity = r.get('identity'); drawn = row['valid']
                    if (drawn == '0' and identity != 'none') or (drawn == '1' and identity not in ('correct','wrong','uncertain')): raise ValueError('identity/drawn contradiction')
                    if a['label'] != 'TARGET' and identity in ('correct','wrong'): raise ValueError('identity/visibility contradiction')
                    if drawn == '1' and m is None and identity != 'uncertain': raise ValueError('missing output box')
                    expected = review.category(a, drawn == '1', m['output_box'] if m else None)
                    if r.get('geometry_category') != expected: raise ValueError('geometry mismatch')
                    comparisons[n] = r
                else: raise ValueError('unknown record type')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f'invalid review at line {line_no}: {exc}') from exc
    result = []
    for n, r in sorted(comparisons.items()):
        a = annotations[n]
        # NOT_VISIBLE and UNKNOWN both map to uncertain, never confirmed absence.
        result.append({'frame': n, 'presence': 'present' if a['label'] == 'TARGET' else 'uncertain', 'drawn': r['drawn'], 'identity': r['identity']})
    return result, {'annotations': len(annotations), 'comparisons': len(comparisons), 'uncompared_annotations': len(annotations)-len(comparisons)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels')
    p.add_argument('--observations')
    p.add_argument('--log')
    p.add_argument('--review-session', help='JSONL from local human review tool')
    p.add_argument('--video', help='original video for session verification')
    p.add_argument('--manifest', help='the exact replay manifest used during review, if any')
    p.add_argument('--reviews', help='CSV with frame,presence,drawn,identity; review must be independently documented')
    p.add_argument('--low', type=float, default=0.1)
    p.add_argument('--match', type=float, default=0.5)
    args = p.parse_args()
    if args.review_session:
        if not args.video or not args.log or args.reviews or args.labels or args.observations:
            p.error('--review-session requires --video and --log, optionally --manifest')
        rows, audit = load_review_session(args.review_session, args.video, args.log, args.manifest)
        result = reviewed_identity_metrics(rows)
        result['audit'] = audit
        result['input'] = {'path': args.review_session, 'sha256': sha256(args.review_session)}
        result['interpretation'] = 'Diagnostic, nonrepresentative sample; NOT_VISIBLE and UNKNOWN remain uncertain; human judgments are not independently verified.'
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.video or args.manifest: p.error('--video and --manifest require --review-session')
    if args.reviews:
        if args.labels or args.observations or args.log:
            p.error('--reviews cannot be combined with geometry inputs')
        rows = read_index(args.reviews)
        result = reviewed_identity_metrics(rows.values())
        result['input'] = {'path': args.reviews, 'sha256': sha256(args.reviews)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not all((args.labels, args.observations, args.log)):
        p.error('provide --reviews or all of --labels, --observations and --log')
    result = screen(read_index(args.labels), read_index(args.observations), read_index(args.log), args.low, args.match)
    result['configuration'] = {'low': args.low, 'match': args.match, 'source_iou_precision': 'as supplied; existing analysis rounds to 4 decimals'}
    result['inputs'] = {name: {'path': str(Path(path)), 'sha256': sha256(path)} for name, path in [('labels', args.labels), ('observations', args.observations), ('log', args.log)]}
    result['interpretation'] = 'Geometry screening only, not identity accuracy or confirmed false-lock rate; not_visible does not establish absence. Sampling is nonrepresentative.'
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
