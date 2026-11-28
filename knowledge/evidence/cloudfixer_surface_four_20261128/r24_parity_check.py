"""Authenticate the r3 evidence and r24 calibration before resuming the four-head experiment."""
import copy
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import run_cloudfixer_surface_four as runner

CONDITIONS = ('uniform', 'background', 'upsampling', 'rotation')
OLD_CAMPAIGN = '9875e07edbcf0bda46cc3e3cca3fe10fda60a27f10c84f1c23a359000e618417'
OLD_RUNNER = '9a27805a6086dc10bdc08d0f727748ec751b82df429882aef52073fe3b3bcc5d'
OLD_PROFILES = {
    'uniform': 'f7aff2efbde2a3ab6c7699e2d67cb46b075a63211f7e5c569069720b97484d71',
    'background': '8ee85717e1dc49691eb05b03b4d563d5d93620ffd9a7d407d7543bfc0e1c095c',
    'upsampling': 'd4d9e10ee60eb27a19f6f473ebb39b7ed805959f3bb9a86ce551b824c868024a',
    'rotation': 'c6b2426e35f9023a9099702a15d7a50d09ff168560e4ac4334c3b19dc4e59f23',
}
SOURCE_SESSION = 'tta-surface-four-models-a100-20261128-r3'
TARGET_SESSION = 'tta-surface-four-models-a100-20261128-r24'


def require(test, description):
    if not test:
        raise ValueError('Cross-GPU parity: ' + description)


def normalized(campaign):
    stable = copy.deepcopy(campaign)
    del stable['runner_sha256']
    for key in ('gpu', 'total_bytes'):
        del stable['runtime']['hardware'][key]
    return stable


def verify_campaigns(old_read, new_read, runner_sha):
    source_raw = runner.checked(old_read, 'campaign.json')
    target_raw = runner.checked(new_read, 'campaign.json')
    source, target = map(json.loads, (source_raw, target_raw))
    require(runner.sha(source_raw) == OLD_CAMPAIGN and source['runner_sha256'] == OLD_RUNNER,
            'source campaign differs from authenticated recovery snapshot')
    require(target['runner_sha256'] == runner_sha and target['protocol'] == runner.protocol(),
            'replacement runner or method protocol differs from the deployed package')
    require(normalized(source) == normalized(target), 'labels, scientific settings, assets or runtime source differ')
    old_hw, new_hw = (c['runtime']['hardware'] for c in (source, target))
    require(old_hw['gpu'] == 'NVIDIA A100-SXM4-80GB' and old_hw['total_bytes'] == 85094825984,
            'unexpected source hardware')
    require(new_hw['gpu'] == 'NVIDIA A100-SXM4-40GB' and 38 * 2**30 <= new_hw['total_bytes'] <= 41 * 2**30,
            'unexpected replacement hardware')
    return source, target, runner.sha(target_raw)


def verify_profiles(old_read, new_read, target_sha, hardware):
    profiles, controls, native_differences = {}, {}, {}
    for condition in CONDITIONS:
        name = 'profiles/' + condition + '.json'
        old_data, new_data = [runner.checked(read, name) for read in (old_read, new_read)]
        old, new = [runner.read_profile(data, condition, digest) for data, digest in
                    ((old_data, OLD_CAMPAIGN), (new_data, target_sha))]
        require(runner.sha(old_data) == OLD_PROFILES[condition], 'old calibration receipt differs for ' + condition)
        require((old['total_bytes'], new['total_bytes']) == tuple(hw['total_bytes'] for hw in hardware),
                'profile device memory differs from campaign ' + condition)
        for field in ('condition', 'corruption', 'ids', 'reason_b1', 'rtol', 'atol', 'selected_batch_size',
                      'independent_b1', 'curvenet_historical_parity'):
            require(old[field] == new[field], f'{condition} differs at {field}')
        require(new['selected_batch_size'] == 1 and set(new['independent_b1']) == {'321', '1146'},
                'missing independent B1 control or unvalidated production batch')
        # Each authenticated r3 control includes input, cloud and logit SHA, decisions, angle and all three classes.
        for obj in new['independent_b1'].values():
            require(len({h['clouds_sha256'] for h in obj['heads'].values()}) == 1,
                    'trajectory hash disagrees between frozen heads')
        for size in (2, 4):
            row = new['curvenet_historical_parity'][str(size)]
            require(row['exact'] and row['ids'] == new['ids'][:size] and len(row['predictions']) == size
                    and all(re.fullmatch('[0-9a-f]{64}', row[k]) for k in ('logits_sha256', 'rng_after_sha256')),
                    'original/repaired CurveNet logits or RNG parity missing')
        pose_errors = {}
        for label, p in (('source', old), ('target', new)):
            require([t['size'] for t in p['trials']] == [1, 2, 4, 8], 'incomplete batch calibration')
            for trial in p['trials'][1:]:
                require(trial.get('input_equal') is True and trial.get('decisions_equal') is True
                        and trial.get('pose_close') is True and trial.get('angle_close') is True
                        and math.isfinite(trial['max_abs_pose_error']), 'B1 to batch pose parity failed')
            pose_errors[label] = {str(t['size']): t['max_abs_pose_error'] for t in p['trials'][1:]}
        a, b = old['native'], new['native']
        require(set(a) == set(b) and a, 'executed native modules differ')
        for filename in a:
            require(all(type(x[filename]['bytes']) is int and x[filename]['bytes'] > 0
                        and re.fullmatch('[0-9a-f]{64}', x[filename]['sha256']) for x in (a, b)),
                    'invalid native identity for ' + filename)
        native_differences[condition] = {n: {'source': a[n], 'target': b[n]} for n in a if a[n] != b[n]}
        profiles[condition] = dict(source_sha256=runner.sha(old_data), target_sha256=runner.sha(new_data))
        controls[condition] = dict(independent_b1=new['independent_b1'], curvenet_historical_parity=new['curvenet_historical_parity'],
                                   panel_ids=new['ids'], reason_b1=new['reason_b1'], selected_batch_size=1,
                                   within_device_pose_max_abs_error=pose_errors)
    return profiles, controls, native_differences


def uniform_raw_pose_checks(old_read, source, runtime):
    """Replay the first archived literal surface veto and acceptance, without using labels to select IDs."""
    name = 'profiles/uniform.json'
    profile = runner.read_profile(runner.checked(old_read, name), 'uniform', OLD_CAMPAIGN)
    result = {}
    for cond, first, last, name in runner.ranges():
        if cond != 'uniform':
            break
        raw = runner.checked(old_read, name)
        block = runner.validate(raw, cond, first, last, runtime.labels, profile, OLD_PROFILES[cond])
        for j, k in enumerate(block['ids']):
            reason = str(block['reason'][j])
            if reason not in ('surface_veto', 'accepted') or reason in result:
                continue
            k = int(k)
            actual = runtime.predict(cond, [k])
            require(block['labels'][j] == source['labels'][k] == runtime.labels[k], 'raw original ID label mismatch')
            for old_key, new_key in [('input_sha256', 'input_sha256'), ('trajectory_sha256', 'trajectory_sha256'),
                                     ('pose', 'pose'), ('reason', 'reason'), ('angle_deg', 'angle_degrees')]:
                require(np.array_equal(block[old_key][j], actual[new_key][0]), f'{cond}/{k} differs at {old_key}')
            require(bool(block['accepted'][j]) == (reason == 'accepted'), 'raw t50 decision differs')
            heads = {}
            for h, model in enumerate(runner.MODELS):
                a, b = block[model + '_logits'][j], actual['logits'][0, h]
                require(a.dtype == b.dtype == np.float32 and a.tobytes() == b.tobytes()
                        and block[model + '_pm'][j] == actual['pm'][0,h]
                        and block[model + '_terminal'][j] == actual['terminal'][0,h], f'{cond}/{k}/{model} differs')
                heads[model] = dict(logits_sha256=runner.sha(a.tobytes()), probes=a.argmax(-1).tolist(),
                                    pm=int(actual['pm'][0,h]), terminal=int(actual['terminal'][0,h]))
            result[reason] = dict(id=k, label=int(runtime.labels[k]), source_block=name, source_block_sha256=runner.sha(raw),
                                  input_sha256=str(actual['input_sha256'][0]),
                                  trajectory_sha256=str(actual['trajectory_sha256'][0]),
                                  pose=actual['pose'][0].tolist(), angle_degrees=float(actual['angle_degrees'][0]),
                                  exact=True, heads=heads)
        if len(result) == 2:
            break
    require({r: v['id'] for r, v in result.items()} == {'accepted': 149, 'surface_veto': 1},
            'expected uniform raw acceptance/veto controls unavailable')
    require(runtime.verify_frozen() == source['runtime']['frozen_states'], 'frozen states changed on replay')
    return result


def main():
    from scripts.cloudfixer_surface_four_runtime import SurfaceRuntime

    drive = Path('/content/drive')
    require(drive.is_mount(), 'physical Drive mount required')
    study = drive/'MyDrive/thesis/result/surface_four_models_20261128'
    owner = json.loads(Path('/content/surface_four_owner.json').read_bytes())
    require(owner['session'] == TARGET_SESSION and owner['owner'] == 'surface-four-models-20261128',
            'unexpected owned session')
    old_root, new_root = study/'campaign_fix1', study/'campaign_resume24'
    old_read, new_read = map(runner.disk_reader, (old_root, new_root))
    release = json.loads(Path('/content/surface_four_science_fix1_receipt.json').read_bytes())
    runner_sha = runner.file_sha(Path(runner.__file__))
    require(release['files']['scripts/run_cloudfixer_surface_four.py'] == runner_sha, 'executing runner outside release')
    source, target, target_sha = verify_campaigns(old_read, new_read, runner_sha)
    identity = target['runtime']
    for filename, digest in identity['source_files'].items():
        require(release['files'][filename] == digest, 'science source outside release: ' + filename)
    hardware = [x['runtime']['hardware'] for x in (source, target)]
    profiles, controls, natives = verify_profiles(old_read, new_read, target_sha, hardware)
    setup = Path('/content/surface_four_setup')
    runtime = SurfaceRuntime(SimpleNamespace(manifest=setup/'package/manifest.json', asset_root=setup/'assets',
        official_root=setup/'official', diffusion=setup/'official/checkpoints/diffusion_model_transformer_modelnet40.npy',
        dataset_root=Path('/content/3DD-TTA/data/modelnet40_c'), classifier=setup/'pointMAE_modelnet40.pth'))
    require(identity == runtime.identity, 'replay runtime differs from replacement calibration')
    raw = uniform_raw_pose_checks(old_read, source, runtime)
    evidence = dict(schema=1, check_version=1, status='passed', source_campaign_sha256=OLD_CAMPAIGN,
        target_campaign_sha256=target_sha, profiles=profiles, source_hardware=hardware[0], target_hardware=hardware[1],
        source_session=SOURCE_SESSION, target_session=TARGET_SESSION, target_endpoint=owner['endpoint'],
        source_runner_sha256=OLD_RUNNER, target_runner_sha256=runner_sha, target_science_release_sha256=release['sha256'],
        checker_sha256=runner.file_sha(Path(__file__)), scientific_identity_sha256=runner.sha(runner.canonical(normalized(source))),
        profile_controls=controls, changed_native_binaries=natives, uniform_raw_b1_replays=raw,
        raw_pose_scope='The two Uniform replays compare the full 3x3 matrices. The archived four-corruption profiles '
                       'record angles, reasons and resulting cloud hashes. No archived 3x3 matrix exists for Rotation ID 321.')
    payload = runner.canonical(evidence)
    for folder in (study/'setup'/TARGET_SESSION, new_root):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'resume_parity.json'
        runner.seal_identity(folder, path, payload)
        require(runner.checked(runner.disk_reader(folder), path.name) == payload, 'proof readback failed')
    print('FOUR_HEAD_RECOVERY_PARITY_PASSED', runner.sha(payload), 'uniform accepted=149 surface_veto=1', flush=True)


if __name__ == '__main__':
    main()
