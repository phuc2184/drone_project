#!/usr/bin/env python3
"""Re-run each reported defect against the pre-v3 backup and the v3 code."""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path('/home/foxsec/drone_project')
sys.path.insert(0, str(ROOT))
import evaluate as new

def load_old():
    spec = importlib.util.spec_from_file_location(
        'evaluate_v2_backup', ROOT / 'backups/gateA_evaluator_v3_20261107_032000/evaluate.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old = load_old()
out = {}

# 1. Unreviewed / zero-eligible false-lock rate
old_json = json.load(open(ROOT / 'research_handoff/reports/gemini_gateA_evaluator_20261107_024500/gatea_v2_evaluation_5_manual.json'))
out['p0_zero_eligible'] = {
    'v2_json_false_lock_rate_target_present': old_json['identity_metrics']['false_lock_rate_target_present'],
    'v2_json_judged_locks': old_json['identity_metrics']['judged_locks'],
    'v3_summary': 'run gate_a_evaluator_v3.py; false_lock_rate_target_present=None, status=NOT_ESTIMABLE',
}

# 2. Unknown gap after absence
gt_unknown = [
    {'start_frame': 0, 'end_frame': 9, 'target_visible': 1, 'view': 'front', 'target_id': None},
    {'start_frame': 10, 'end_frame': 19, 'target_visible': 0, 'view': 'none', 'target_id': None},
    {'start_frame': 20, 'end_frame': 29, 'target_visible': -1, 'view': 'unknown', 'target_id': None},
    {'start_frame': 30, 'end_frame': 39, 'target_visible': 1, 'view': 'front', 'target_id': None},
]
out['p1_unknown_gap'] = {
    'v2_reacquisition_episodes': len(old.build_reacquisition_episodes(gt_unknown)['reacquisition_episodes']),
    'v3_reacquisition_episodes': len(new.build_reacquisition_episodes(gt_unknown)['reacquisition_episodes']),
    'v3_unknown_onset': len(new.build_reacquisition_episodes(gt_unknown)['unknown_onset_reappearances']),
}

# 3. Nested GT overlap
gt_overlap = [
    {'start_frame': 0, 'end_frame': 100, 'target_visible': 1, 'view': 'front', 'target_id': None},
    {'start_frame': 10, 'end_frame': 20, 'target_visible': 1, 'view': 'front', 'target_id': None},
]
for name, mod in (('v2', old), ('v3', new)):
    try:
        cleaned = mod.validate_ground_truth_intervals(gt_overlap)
        out.setdefault('p1_gt_overlap', {})[name] = {'accepted': True, 'intervals': len(cleaned)}
    except Exception as exc:
        out.setdefault('p1_gt_overlap', {})[name] = {'accepted': False, 'error': str(exc)}

# 4. Duplicate canonical evidence keys
for name, mod in (('v2', old), ('v3', new)):
    try:
        ep = mod.reacquisition_episode_evidence(
            10, 15, {10: 'unresolved', '010': 'correct_lock'},
            lock_frames={10}, log_frames=set(range(10, 16)), onset_uncertainty_frames=0)
        out.setdefault('p1_duplicate_evidence_keys', {})[name] = {
            'accepted': True, 'status': ep['status'], 'first_correct_proven': ep['first_correct_proven']}
    except Exception as exc:
        out.setdefault('p1_duplicate_evidence_keys', {})[name] = {'accepted': False, 'error': str(exc)}

# 5. Lock evidence with empty lock-capable set
for name, mod in (('v2', old), ('v3', new)):
    try:
        ep = mod.reacquisition_episode_evidence(
            10, 15, {10: 'correct_lock'}, lock_frames=set(),
            log_frames=set(range(10, 16)), onset_uncertainty_frames=0)
        out.setdefault('p1_impossible_lock', {})[name] = {
            'accepted': True, 'status': ep['status'], 'first_correct_proven': ep['first_correct_proven']}
    except Exception as exc:
        out.setdefault('p1_impossible_lock', {})[name] = {'accepted': False, 'error': str(exc)}

# 6. Partial reappearance after absence
gt_partial = [
    {'start_frame': 0, 'end_frame': 9, 'target_visible': 1, 'view': 'front', 'target_id': None},
    {'start_frame': 10, 'end_frame': 19, 'target_visible': 0, 'view': 'none', 'target_id': None},
    {'start_frame': 20, 'end_frame': 29, 'target_visible': 2, 'view': 'back', 'target_id': None},
    {'start_frame': 30, 'end_frame': 39, 'target_visible': 1, 'view': 'back', 'target_id': None},
]
out['p1_partial_onset'] = {
    'v2_episodes': len(old.build_reacquisition_episodes(gt_partial)['reacquisition_episodes']),
    'v2_first_start': old.build_reacquisition_episodes(gt_partial)['reacquisition_episodes'][0]['start_frame']
        if old.build_reacquisition_episodes(gt_partial)['reacquisition_episodes'] else None,
    'v3_episodes': len(new.build_reacquisition_episodes(gt_partial)['reacquisition_episodes']),
    'v3_first_start': new.build_reacquisition_episodes(gt_partial)['reacquisition_episodes'][0]['start_frame'],
    'v3_onset_visibility': new.build_reacquisition_episodes(gt_partial)['reacquisition_episodes'][0]['onset_visibility'],
    'v3_first_fully_visible': new.build_reacquisition_episodes(gt_partial)['reacquisition_episodes'][0]['first_fully_visible_frame'],
}

# 7. View change
gt_view = [
    {'start_frame': 0, 'end_frame': 9, 'target_visible': 1, 'view': 'front', 'target_id': None},
    {'start_frame': 10, 'end_frame': 19, 'target_visible': 1, 'view': 'back', 'target_id': None},
]
out['p1_view_change'] = {
    'v2_episodes': len(old.build_reacquisition_episodes(gt_view)['reacquisition_episodes']),
    'v3_episodes': len(new.build_reacquisition_episodes(gt_view)['reacquisition_episodes']),
    'v3_viewpoint_changes': len(new.build_reacquisition_episodes(gt_view)['viewpoint_changes']),
}

print(json.dumps(out, ensure_ascii=False, indent=2))
