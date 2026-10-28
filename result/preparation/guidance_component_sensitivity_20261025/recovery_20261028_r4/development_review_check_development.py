"""Read-only development review; only OS temporary files are created."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

REPO = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components')
PREP = REPO / 'result/preparation/guidance_component_sensitivity_20261025'
sys.path.insert(0, str(PREP))
import analyze_sensitivity_local as a


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


plan_path = PREP / 'recovery_20261028_r4/controls/bundle/campaign_plan.json'
plan_sha = a.identity(plan_path)['sha256']
context = a.load_context(PREP, plan_path, plan_sha)
a.PLAN_SHA = plan_sha
plan, controller, _, _, _, registry, *_ = context
with zipfile.ZipFile(PREP / 'calibrate_evidence_drive_roundtrip.zip') as archive:
    calibration = json.loads(archive.read('calibration/calibration_report.json'))

# Run the actual old=True inspector on every physically available reused ZIP.
alignment, _ = a.baseline_labels(PREP, context)
paired, heads, lion, case_rows, raw_evidence = {}, {}, [], [], []
old = a.read_json(PREP / 'reused_baselines.json')
trials = {trial['family']: trial for trial in registry.values() if trial['reuse']}
old_assets = PREP.parent / 'guidance_component_study_20261025/assets_v1'
seen = set()
for raw in old['raw_archives']:
    path = Path(raw['canonical_path'])
    a.verify(path, raw)
    trial = trials[raw['component_arm']]
    job = dict(dataset=raw['dataset'], seed=0, trial=trial)
    config, scores, _, _, _ = a.inspect_raw(path, job, 'development', context, calibration,
                                           True, alignment, paired, heads, lion, old_assets)
    key = (raw['dataset'], config['backbone'], trial['id'])
    assert key not in seen and config['run_id'] == raw['run_id']
    seen.add(key)
    case_rows.extend(scores)
    raw_evidence.append(dict(dataset=key[0], backbone=key[1], trial_id=key[2], **a.identity(path)))
assert len(seen) == 24 and len(paired) == 30 and len(heads) == 8 and len(lion) == 1
baseline_scores = a.balanced_scores(case_rows, [trial['id'] for trial in trials.values()])

# These are the exact persisted preparations reused by development/calibration.
cache_root = PREP.parent / 'guidance_component_study_20261025/paired_cache_extracted'
physical = a.physical_caches(calibration['cache_records'], [cache_root])
assert len(physical) == 30 and all(row['local_verified'] for row in physical)
cache_evidence = [{key: row[key] for key in ('remote_path', 'bytes', 'sha256')} for row in physical]
jobs = controller.jobs_for(plan, registry, 'development',
                           '/content/studies/guidance-components/review-only-nonexistent', calibration, plan_sha)
assert len(jobs) == 32 and len({job['trial']['id'] for job in jobs}) == 16 and len(registry) == 19

# Use temporary local freeze/acceptance files; patch only filesystem ownership
# and remote acceptance I/O. Controller condition construction stays unchanged.
original_owned, original_accepted = controller.owned, controller.accepted_stage
freeze_evidence = []
try:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        local_plan = dict(plan, confirmation_freeze_path=str(root / 'freeze.json'),
                          accepted_development_path=str(root / 'accepted.json'))
        controller.owned = lambda path: Path(path)
        controller.accepted_stage = lambda path, plan_sha: a.read_json(path)
        spectral = [trial for trial in registry.values() if trial['family'] in ('combined', 'spectral')]
        chamfer = [trial for trial in registry.values() if trial['family'] == 'chamfer']
        for selected_spectral in spectral:
            for selected_chamfer in chamfer:
                winners = (selected_spectral['id'], selected_chamfer['id'])
                scores = [dict(trial_id=name, family=trial['family'], accuracy_pct=99. if name in winners else 1.)
                          for name, trial in registry.items()]
                accepted = dict(status='accepted', phase_plan_sha256=plan_sha, selection_policy='probability_mean',
                                selected_spectral_trial_id=winners[0], selected_chamfer_trial_id=winners[1])
                (root / 'accepted.json').write_text(json.dumps(accepted))
                freeze = a.freeze_selection(scores, registry, a.identity(root / 'accepted.json'), controller, calibration)
                (root / 'freeze.json').write_text(json.dumps(freeze))
                conditions = controller.frozen_conditions(local_plan, registry, calibration, plan_sha)
                expected = [dict(id=row['id'], role=row['role'], trial_id=row['trial']['id']) for row in conditions]
                assert freeze['conditions'] == expected
                freeze_evidence.append(dict(spectral=winners[0], chamfer=winners[1], conditions=expected))
finally:
    controller.owned, controller.accepted_stage = original_owned, original_accepted
assert len(freeze_evidence) == 72

result = dict(status='passed', numpy_version=a.np.__version__,
              plan_sha256=plan_sha,
              analyzer=a.identity(PREP / 'analyze_sensitivity_local.py'),
              controller=a.identity(plan_path.parent / 'sensitivity_campaign_controller.py'),
              reused_zip_count=len(seen), reused_zip_identity_rows_sha256=object_sha(raw_evidence),
              reused_pairing_cache_count=len(paired), frozen_head_contract_count=len(heads), lion_state_count=len(lion),
              reused_baseline_scores=baseline_scores,
              physical_cache_count=len(physical), physically_verified_cache_count=sum(row['local_verified'] for row in physical),
              physical_cache_identity_rows_sha256=object_sha(cache_evidence),
              new_development_jobs=len(jobs), new_development_trials=len({job['trial']['id'] for job in jobs}),
              total_registered_trials=len(registry), freeze_pairs=len(freeze_evidence),
              three_condition_pairs=sum(len(row['conditions']) == 3 for row in freeze_evidence),
              four_condition_pairs=sum(len(row['conditions']) == 4 for row in freeze_evidence),
              freeze_pair_rows_sha256=object_sha(freeze_evidence),
              limitation='Completed new development artifacts not inspected; all freeze scores are synthetic. No remote actions or repository writes.')
print(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
