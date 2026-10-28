"""CPU-only recovery: exact accepted calibration and cumulative approved budget."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

META = RECOVERY_META_LITERAL
ROOT = Path('/content/studies/guidance-components')
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
old = ROOT / 'phases'
sys.path.insert(0, str(old))
from deploy_assets_template import identity, verify, extract_checked, copy_checked, seal, utc
assert identity(old / 'deploy_assets_template.py')['sha256'] == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
package = Path('/content/agent_b_r4_controls.zip')
verify(package, META['control_package'])
output = ROOT / 'recovery_r4'
assert not output.exists(), 'Recovery directory exists; inspect partial state before reviewed retry'
assert not (DRIVE / 'sensitivity_campaign_r4_ledger.json').exists(), 'Existing migrated Drive ledger; do not reset'
extract_checked(package, output)
for name, expected in META['files'].items():
    verify(output / name, expected)
assert identity(output / 'campaign_plan.json')['sha256'] == META['plan_sha256']
assert identity(output / 'original_campaign_plan.json')['sha256'] == META['provenance']['original_plan_sha256']
for name in ('original_campaign_ledger.json', 'original_gpu_approval.json'):
    key = 'original_ledger' if name.startswith('original_campaign') else 'original_approval'
    verify(output / name, META['provenance'][key])
spec = importlib.util.spec_from_file_location('recovered_sensitivity_controller', output / 'sensitivity_campaign_controller.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
plan, registry, selections = c.plan_verified(output / 'campaign_plan.json', META['plan_sha256'])
fresh_cpu = c.cpu_preflight_verified(plan)
assert fresh_cpu['state']['session'] == META['session'] and fresh_cpu['state']['strict_loads'] == 16
assert not (ROOT / 'paired_sensitivity_smoke_v2_seed0').exists()
assert not (DRIVE / 'paired_sensitivity_smoke_v2_seed0').exists(), 'Old smoke cache exists; inspect before any new run'
receipt = json.loads((output / 'original_calibrate_receipt.json').read_text())
archive = receipt['archive']
assert archive['bytes'] == 5806352 and archive['sha256'] == '3431f2a15e67892429fcc663b7fd2c0d6e3d93083daee078dc4b3e884d61c8f2'
c.owned(archive['drive_path'], DRIVE)
copy_checked(archive['drive_path'], output / 'original_calibration_evidence.zip', archive)
extract_checked(output / 'original_calibration_evidence.zip', output / 'original_calibration')
assert (output / 'original_calibration/campaign_plan.json').read_bytes() == (output / 'original_campaign_plan.json').read_bytes()
assert json.loads((output / 'original_calibration/state.json').read_text()) == receipt['state']
acceptance = json.loads((output / 'calibration_local_acceptance.json').read_text())
assert acceptance['status'] == 'accepted_local_label_free_calibration'
assert acceptance['phase_plan_sha256'] == META['provenance']['original_plan_sha256']
assert acceptance['archive']['sha256'] == archive['sha256'] and acceptance['physical_cache_npz'] == 30
calibration_source = output / 'original_calibration/calibration'
report_path = calibration_source / 'calibration_report.json'
verify(report_path, plan['recovery']['calibration_report'])
report = json.loads(report_path.read_text())
assert report == receipt['state']['calibration']
assert report['session'] == META['provenance']['original_session']
assert report['coefficients'] == acceptance['coefficients'] and report['lion_state_before'] == report['lion_state_after']
assert not (ROOT / 'sensitivity_calibration_v2').exists()
shutil.copytree(calibration_source, ROOT / 'sensitivity_calibration_v2')
c.calibration_verified(ROOT / 'sensitivity_calibration_v2/calibration_report.json', plan)

# Existing development preparations are byte-identified historical data, never relabeled.
def install(source, target, expected=None):
    source, target = Path(source), Path(target)
    c.owned(target)
    expected = identity(source) if expected is None else expected
    if target.exists():
        verify(target, expected)
    else:
        copy_checked(source, target, expected)

for name in ('accepted_development.json', 'development_phase_receipt.json'):
    install(output / name, ROOT / name)
accepted = json.loads((output / 'accepted_development.json').read_text())
development = json.loads((output / 'development_phase_receipt.json').read_text())
assert identity(output / 'accepted_development.json')['sha256'] == 'e978c558f4ec70244c5ad14030fc09191bf78435541466866744366fb73a078c'
assert identity(output / 'development_phase_receipt.json')['sha256'] == '7ef2b62404b019a35027b38a65f146087a30d403cc260f129a3a1f83a17980d7'
phase = accepted['phase_archive']
assert phase == development['archive'] and phase['sha256'] == '8abd0ec00c71996407d0fabc3933a7984c2e9860c8844e2d2940bff1db15901d'
c.owned(phase['drive_path'], DRIVE)
c.owned(phase['local_path']); install(phase['drive_path'], phase['local_path'], phase)
import zipfile
with zipfile.ZipFile(phase['local_path']) as archive_stream:
    from deploy_assets_template import checked_members
    checked_members(archive_stream)
    assert json.loads(archive_stream.read('state.json')) == development['state']
cache_records = {}
for job in development['state']['jobs']:
    assert job['status'] == 'complete'
    for row in job['cache']:
        c.owned(row['local_path'])
        c.owned(row['drive_path'], DRIVE)
        assert row['local_path'] not in cache_records or cache_records[row['local_path']] == row
        cache_records[row['local_path']] = row
assert len(cache_records) == 30
for row in cache_records.values():
    install(row['drive_path'], row['local_path'], row)
(output / 'restored_development_cache_records.json').write_text(json.dumps(list(cache_records.values()), indent=2) + '\n')
phases = ROOT / 'sensitivity_phases'; phases.mkdir(exist_ok=True)
for name in plan['phase_sources']:
    target = phases / name
    if target.exists() and identity(target) != identity(output / name):
        assert name == 'sensitivity_campaign_controller.py' and identity(target)['sha256'] == 'e0244aa907540512727377df7cf341c21320b4e766bf62b541f8a783f86e1f17'
        shutil.copyfile(target, output / ('previous_' + name))
        target.unlink()
    install(output / name, target)
for source_name, target_name, previous_sha in [('campaign_plan.json', 'sensitivity_campaign_plan.json', META['provenance']['original_plan_sha256']), ('gpu_approval.json', 'sensitivity_gpu_approval.json', META['provenance']['original_approval']['sha256'])]:
    target = c.owned(ROOT / target_name)
    if target.exists() and identity(target) != identity(output / source_name):
        assert identity(target)['sha256'] == previous_sha, 'Foreign prior control; inspect before replacement'
        shutil.copyfile(target, output / ('previous_' + target_name)); target.unlink()
    install(output / source_name, target)
c.approval_verified(META['plan_sha256'])
ledger = json.loads((output / 'campaign_ledger.json').read_text())
c.ledger_valid(ledger, META['plan_sha256'])
assert ledger['spent_seconds'] == 155.86660684299568
c.campaign_budget_verified(plan, 'smoke', ledger['spent_seconds'])
assert not (ROOT / c.LEDGER_NAME).exists()
c.save_ledger(ledger)
shutil.copyfile(ROOT / 'prepared_sensitivity_v2/accepted_cpu_preflight_receipt.json', output / 'fresh_cpu_preflight_receipt.json')
recovery_state = dict(status='complete_CPU_recovery_no_GPU_repeated', session=META['session'], endpoint=META['endpoint'],
                     phase_plan_sha256=META['plan_sha256'], original_plan_sha256=META['provenance']['original_plan_sha256'],
                     manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                     calibration_report=identity(report_path), preserved_spent_seconds=ledger['spent_seconds'], restored_cache_npz=30,
                     metadata=META, completed_utc=datetime.now(timezone.utc).isoformat())
(output / 'recovery_state.json').write_text(json.dumps(recovery_state, indent=2) + '\n')
target = DRIVE / 'phase-evidence' / ('sensitivity_recovery_r4_' + utc()) / 'recovery_evidence.zip'
proof = dict(state=recovery_state, archive=seal(output, target))
(ROOT / 'sensitivity_recovery_r4_receipt.json').write_text(json.dumps(proof, indent=2) + '\n')
(target.parent / 'receipt.json').write_text(json.dumps(proof, indent=2) + '\n')
validation = dict(status='validated', stage='calibrate', phase_plan_sha256=META['plan_sha256'],
                  manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                  spent_seconds=ledger['spent_seconds'], evidence=receipt['archive'], recovered_from_original_session=True,
                  recovery_evidence=proof['archive'], independent_local_validation_pending=False)
(ROOT / 'sensitivity_calibrate_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print('AGENT_B_R4_RECOVERY_RECEIPT', json.dumps(proof['archive']), flush=True)
print('AGENT_B_R4_RECOVERY_CPU_COMPLETE', flush=True)
