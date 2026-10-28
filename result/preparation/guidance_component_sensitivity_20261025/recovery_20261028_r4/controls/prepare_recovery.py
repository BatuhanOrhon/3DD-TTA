"""Generate the ownership-only continuation of the approved Agent B campaign."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

SESSION = 'tta-guidance-components-a100-20261028-r4'
OLD_SESSION = 'tta-guidance-components-a100-20261025-r1'
OLD_PLAN = '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
SPENT = 155.86660684299568
ROOT = '/content/studies/guidance-components'
DRIVE = '/content/drive/MyDrive/thesis/result/guidance_component_study_20261025'
REPORT = dict(bytes=99029, sha256='82ebc1fdb2784a0663968a20d46909e3020e127310f237971382c78c40e211e0')
INPUTS = {
    'gpu_approval.json': '2f3e2a4f9efe9bdfe6a68e1345506f96f2a2f8aff848da849959f5e6005313da',
    'preserved_campaign_ledger_20261026T0556.json': '6c0e3cb8cebdf4a43bc3383f1599b782f4a1d4129690eb2de1a7de387a5915b3',
    'calibration_local_acceptance.json': '9023130dded72558e7631f1befffd99fd02353e2779033afa525b645a576c7fc',
    'calibrate_receipt.json': 'b5197775f0d9713043eb5860024c2b660301f56fab81e583cca6494f6c08ad27',
    'preflight_sensitivity_cpu.py': 'faee34c486723c478cb3aeff6d26c465401510f775a596768219cc8b12e7c6ea',
}


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replace_once(source, old, new):
    require(source.count(old) == 1, 'Unexpected source shape: ' + old[:80])
    return source.replace(old, new, 1)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(prep: Path, output: Path, endpoint: str):
    require(endpoint.startswith('gpu-a100-hm-') and '/' not in endpoint, 'Require actual A100 HighRAM endpoint')
    plan_raw = (prep / 'campaign_plan.json').read_bytes()
    require(identity(plan_raw)['sha256'] == OLD_PLAN, 'Original plan changed')
    plan = json.loads(plan_raw)
    for name, sha in INPUTS.items():
        require(identity((prep / name).read_bytes())['sha256'] == sha, 'Recovery input changed: ' + name)
    for name, expected in plan['phase_sources'].items():
        require(identity((prep / name).read_bytes()) == expected, 'Original phase source changed: ' + name)
    ledger = json.loads((prep / 'preserved_campaign_ledger_20261026T0556.json').read_text())
    require(ledger['active'] is None and ledger['spent_seconds'] == SPENT and ledger['cap_seconds'] == 19800
            and ledger['session'] == OLD_SESSION and ledger['phase_plan_sha256'] == OLD_PLAN
            and len(ledger['attempts']) == 1 and ledger['attempts'][0]['stage'] == 'calibrate'
            and ledger['attempts'][0]['status'] == 'complete', 'Unresolved or changed original campaign')
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'bundle'; bundle.mkdir()
    provenance = dict(original_session=OLD_SESSION, original_endpoint=plan['endpoint'], original_plan_sha256=OLD_PLAN,
                      original_approval=identity((prep / 'gpu_approval.json').read_bytes()),
                      original_ledger=identity((prep / 'preserved_campaign_ledger_20261026T0556.json').read_bytes()),
                      spent_floor_seconds=SPENT, new_session=SESSION, new_endpoint=endpoint,
                      scope='Ownership/control recovery only; sealed assets/objectives/trials/IDs/readouts/cap unchanged',
                      user_resume_instruction='Yeni oturum açıp çalışmaya devam et')
    controller = (prep / 'sensitivity_campaign_controller.py').read_text()
    controller = replace_once(controller, "SESSION = '" + OLD_SESSION + "'", "SESSION = '" + SESSION + "'")
    controller = replace_once(controller, "LEDGER_NAME = 'sensitivity_campaign_ledger.json'", "LEDGER_NAME = 'sensitivity_campaign_r4_ledger.json'")
    controller = replace_once(controller, '    report = json.loads(Path(path).read_text())\n',
                              '    verify(path, ' + repr(REPORT) + ')\n    report = json.loads(Path(path).read_text())\n')
    guard = "\n    require(ledger.get('spent_seconds', -1) >= %r and ledger.get('recovery', {}).get('original_plan_sha256') == %r\n            and ledger['recovery'].get('original_ledger_sha256') == %r, 'Recovered campaign floor/provenance missing')\n" % (SPENT, OLD_PLAN, INPUTS['preserved_campaign_ledger_20261026T0556.json'])
    controller = replace_once(controller, '\n\ndef load_ledger(plan_sha, local=None, remote=None):', guard + '\n\ndef load_ledger(plan_sha, local=None, remote=None):')
    controller = replace_once(controller, 'def load_ledger(plan_sha, local=None, remote=None):\n',
        "def load_ledger(plan_sha, local=None, remote=None):\n    if local is None and remote is None:\n        require((ROOT / LEDGER_NAME).exists() or (DRIVE / LEDGER_NAME).exists(), 'Recovery ledger must be installed before GPU work')\n")
    inherited = dict(original_plan_sha256=OLD_PLAN, original_ledger_sha256=INPUTS['preserved_campaign_ledger_20261026T0556.json'])
    controller = replace_once(controller, 'spent_seconds=0., active=None, attempts=[])',
                              'spent_seconds=%r, active=None, attempts=[], recovery=%r)' % (SPENT, inherited))
    controller = controller.replace("ledger['spent_seconds'] = 123.", "ledger['spent_seconds'] = %r" % (SPENT + 123.))
    controller = controller.replace("['spent_seconds'] == 123.", "['spent_seconds'] == %r" % (SPENT + 123.))
    (bundle / 'sensitivity_campaign_controller.py').write_text(controller)
    plan['phase_sources']['sensitivity_campaign_controller.py'] = identity(controller.encode())
    plan.update(session=SESSION, endpoint=endpoint, recovery=dict(provenance, calibration_report=REPORT,
                calibration_archive=dict(bytes=5806352, sha256='3431f2a15e67892429fcc663b7fd2c0d6e3d93083daee078dc4b3e884d61c8f2'),
                independent_acceptance=identity((prep / 'calibration_local_acceptance.json').read_bytes())))
    dump(bundle / 'campaign_plan.json', plan)
    plan_sha = identity((bundle / 'campaign_plan.json').read_bytes())['sha256']
    approval = json.loads((prep / 'gpu_approval.json').read_text())
    approval.update(session=SESSION, phase_plan_sha256=plan_sha, recovery=dict(provenance),
                    recovery_recorded_utc=datetime.now(timezone.utc).isoformat())
    dump(bundle / 'gpu_approval.json', approval)
    ledger.update(session=SESSION, phase_plan_sha256=plan_sha, recovery=inherited)
    receipt = json.loads((prep / 'calibrate_receipt.json').read_text())
    ledger['lion_state'] = receipt['state']['calibration']['lion_state_after']
    dump(bundle / 'campaign_ledger.json', ledger)
    for source, destination in [('gpu_approval.json', 'original_gpu_approval.json'),
                                ('preserved_campaign_ledger_20261026T0556.json', 'original_campaign_ledger.json'),
                                ('calibration_local_acceptance.json', 'calibration_local_acceptance.json'),
                                ('calibrate_receipt.json', 'original_calibrate_receipt.json'),
                                ('calibrate_spectral_losses.py', 'calibrate_spectral_losses.py')]:
        shutil.copyfile(prep / source, bundle / destination)
    (bundle / 'original_campaign_plan.json').write_bytes(plan_raw)
    old_prep = prep.parent / 'guidance_component_study_20261025'
    for name in ('accepted_development.json', 'development_phase_receipt.json'):
        shutil.copyfile(old_prep / name, bundle / name)
    fresh = replace_once((prep / 'preflight_sensitivity_cpu.py').read_text(), "SESSION = '" + OLD_SESSION + "'", "SESSION = '" + SESSION + "'")
    (output / 'preflight_sensitivity_cpu.py').write_text(fresh)
    for stage in ('smoke', 'development', 'confirmation'):
        wrapper = (prep / ('run_' + stage + '_phase.py')).read_text()
        require(wrapper.count(OLD_PLAN) == 2, 'Unexpected wrapper plan binding')
        wrapper = wrapper.replace(OLD_PLAN, plan_sha)
        (output / ('run_' + stage + '_phase.py')).write_text(wrapper)
    meta = dict(session=SESSION, endpoint=endpoint, plan_sha256=plan_sha, provenance=provenance,
                fresh_preflight=identity(fresh.encode()), files={f.name: identity(f.read_bytes()) for f in bundle.iterdir()})
    archive = output / 'r4_controls.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(bundle.iterdir()):
            stream.write(path, path.name)
    meta['control_package'] = identity(archive.read_bytes())
    template = (Path(__file__).parent / 'install_recovery_template.py').read_text()
    (output / 'install_recovery_cpu_phase.py').write_text(replace_once(template, 'RECOVERY_META_LITERAL', repr(meta)))
    dump(output / 'recovery_identity.json', meta)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prep, args.output, args.endpoint), indent=2))
