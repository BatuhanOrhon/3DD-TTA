#!/usr/bin/env python3
"""Second-only depth follow-up from an immutable accepted N3 reference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_lion_reencode as parent

shared, depth = parent.shared, parent.depth


def selection(first, steps):
    result = copy.deepcopy(first)
    checkpoints = [1, 2, 3] if steps == 3 else depth.relative_checkpoints(steps)
    for scope in ('normal', 'background'):
        result[scope + '_loop_counts'] = checkpoints + [steps]
        result[scope + '_candidate_names'] = ['clean_%02d' % i for i in checkpoints] + ['terminal']
    return result


def validate_config(config):
    steps = config.get('reencode', {}).get('second_reverse_steps')
    replay = config.get('reencode_replay', {})
    if (type(steps) is not int or steps not in (3, 5, 8)
            or config.get('method') != '3dd_lion_reencode_n%d_replay' % steps
            or not replay.get('phase') or not replay.get('manifest_identity')
            or config.get('trajectory_selection') != selection(config['first_trajectory_selection'], steps)):
        raise ValueError('Invalid fixed second-pass replay condition')
    original = copy.deepcopy(config)
    original.update(method=parent.METHOD, reencode=parent.settings(),
                    trajectory_selection=selection(config['first_trajectory_selection'], 3))
    if config['reencode'] != dict(parent.settings(), second_reverse_steps=steps):
        raise ValueError('Replay changed a fixed reencoding setting')
    parent.validate_config(original)


def configurations(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--second-steps', required=True, type=int, choices=(3, 5, 8))
    parser.add_argument('--replay-phase', required=True)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    config, = parent.configurations(remaining)
    phase = Path(options.replay_phase).resolve()
    config.update(method='3dd_lion_reencode_n%d_replay' % options.second_steps,
        reencode=dict(parent.settings(), second_reverse_steps=options.second_steps),
        reencode_replay=dict(phase=str(phase), manifest_identity=shared.source.file_identity(phase/'evidence_manifest.json')),
        trajectory_selection=selection(config['first_trajectory_selection'], options.second_steps),
        display_name='Archived N8/B35 and z1/h1; second local N%d' % options.second_steps,
        shared_reconstruction='second local trajectory only; archived first/cycle/N3 references',
        timing_policy='second-only replay/cache/probes/geometry; excludes archived first/encoder work')
    validate_config(config)
    return [config]


def checked_file(root, manifest, name):
    name = str(name)
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or name not in manifest:
        raise ValueError('Unmanifested or escaping replay path')
    path = Path(root)/name
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Replay symlinks are forbidden')
    if shared.source.file_identity(path) != manifest[name]:
        raise ValueError('Replay file bytes changed: ' + name)
    return path


def load_reference(root, manifest, corruption, ids, device='cuda'):
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state_hash, tensor_hash
    root = Path(root)
    configs = list((root/'raw').glob('*/*/*/config.json'))
    if len(configs) != 1:
        raise ValueError('Need one accepted reference bundle')
    cfgpath = configs[0]
    cfg = json.loads(checked_file(root, manifest, cfgpath.relative_to(root)).read_text())
    parent.validate_config(cfg)
    ids = np.asarray(ids, dtype=np.int64)
    positions = [i for i in range(0, len(cfg['selected_original_ids']), 32)
                 if np.array_equal(ids, cfg['selected_original_ids'][i:i+32])]
    if len(positions) != 1 or corruption not in cfg['corruptions'] or cfg.get('failure'):
        raise ValueError('Replay batch is absent or mismatched')
    batch = cfgpath.parent/'batches'/corruption/('batch%04d' % (positions[0]//32))
    diagpath = checked_file(root, manifest, (batch/'diagnostics.json').relative_to(root))
    predpath = checked_file(root, manifest, (batch/'predictions.npz').relative_to(root))
    diag = json.loads(diagpath.read_text())
    cachepath = root/'paired/modelnet40_c'/corruption/(hashlib.sha256(ids.tobytes()).hexdigest()+'.npz')
    checked_file(root, manifest, cachepath.relative_to(root))
    proofpath = checked_file(root, manifest, cachepath.with_suffix('.reencode.npz').relative_to(root))
    with np.load(cachepath, allow_pickle=False) as saved:
        cache = {k: saved[k].copy() for k in saved.files if k != 'metadata'}
        metadata = json.loads(saved['metadata'].item())
    with np.load(proofpath, allow_pickle=False) as saved:
        proof = {k: saved[k].copy() for k in saved.files if k != 'metadata'}
        pm = json.loads(saved['metadata'].item())
    if set(proof) != set(pm['tensor_hashes']):
        raise ValueError('Incomplete posterior proof hash coverage')
    for key, value in proof.items():
        if not np.isfinite(value).all() or tensor_hash(torch.from_numpy(value)) != pm['tensor_hashes'][key]:
            raise ValueError('Changed posterior proof')
    for key in ('shape_latent', 'local_latent', 'noise'):
        if tensor_hash(torch.from_numpy(cache[key])) != metadata['tensor_hashes'][key]:
            raise ValueError('Changed original latent/noise')
    expected = dict(executed_source_sha256=cfg['executed_source_sha256'],
        asset_manifest_sha256=cfg['asset_manifest_sha256'], input_sha256=metadata['input_sha256'])
    if pm['binding'] != expected or not np.array_equal(proof['diffusion_noise'], cache['noise']):
        raise ValueError('Wrong proof source/input/noise binding')
    py, nr = metadata['python_rng'], metadata['numpy_rng']
    rng = ((py[0], tuple(py[1]), py[2]), (nr[0], cache['numpy_rng_keys'], *nr[1:]),
           torch.from_numpy(cache['torch_rng_cpu']),
           [torch.from_numpy(cache['torch_rng_cuda_%d' % i]) for i in range(metadata['cuda_devices'])])
    if _rng_state_hash(rng) != metadata['rng_after_sha256'] or _rng_state_hash(rng) != diag['trajectory']['second']['initial_rng_sha256']:
        raise ValueError('Archived second-pass RNG differs from recorded state')
    state = dict(shape_latent=torch.from_numpy(proof['selected_z']).to(device)[:, :, None, None],
        local_latent=torch.from_numpy(proof['selected_h']).to(device)[:, :, None, None],
        noise=torch.from_numpy(proof['diffusion_noise']).to(device), rng_after_prepare=rng)
    with np.load(predpath, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if not np.array_equal(arrays['index'], ids):
        raise ValueError('Archived logits belong to different original IDs')
    refs = {str(p.relative_to(root)): manifest[str(p.relative_to(root))]
            for p in (cfgpath, diagpath, predpath, cachepath, proofpath)}
    return state, proof, arrays, diag, refs


def replay_arrays(archive, second):
    import numpy as np
    from lion_reencode import readouts
    count = len(second['trajectory_logits'])
    if (archive['cycle_logits'].shape != (count, 40) or not np.isfinite(archive['cycle_logits']).all()
            or archive['trajectory_logits'].shape != (count, 4, 40) or not np.isfinite(archive['trajectory_logits']).all()):
        raise ValueError('Invalid archived cycle/N3 logits')
    return dict(second, first_trajectory_logits=archive['first_trajectory_logits'],
        cycle_logits=archive['cycle_logits'], reference_n3_trajectory_logits=archive['trajectory_logits'],
        **readouts(archive['first_trajectory_logits'], second['trajectory_logits']))


def validate_reference_host(old, current):
    for key in ('lion_state_before', 'classifier_state_before_sha256', 'dataset_provenance',
                'guidance_protocol', 'resolved_lion_config', 'resolved_model_config',
                'preprocessing', 'source_classifier_preprocessing'):
        if old[key] != current[key]:
            raise ValueError('Replay runtime differs: ' + key)
    previous, actual = old['executed_source_files'], current['executed_source_files']
    required = {'repo/' + name for name in ('tta.py', 'utilities_3dd_tta.py', 'shared_trajectory.py',
        'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py',
        'classifier_backbones.py', 'models_mate/Point_MAE.py')}
    if not required.issubset(previous.keys() & actual.keys()):
        raise ValueError('Missing shared reconstruction/classifier source identity')
    for name in previous.keys() & actual.keys():
        if name.startswith('repo/') and previous[name] != actual[name]:
            raise ValueError('Replay shared Python source differs: ' + name)
    for name, value in old['native_extensions'].items():
        now = current['native_extensions'].get(name, {})
        if any(now.get(k) != value.get(k) for k in ('bytes', 'sha256')):
            raise ValueError('Replay native extension differs: ' + name)


def worker(directories):
    import torch
    import scripts.analyze_legacy_trajectory
    import lion_reencode
    from legacy_trajectory import tensor_hash
    from shared_trajectory import SharedTrajectoryObserver
    if len(directories) != 1:
        raise ValueError('Replay requires one Point-MAE bundle')
    config = json.loads((Path(directories[0])/'config.json').read_text()); validate_config(config)
    replay = config['reencode_replay']; root = Path(replay['phase'])
    shared.source.verify_identity(root/'evidence_manifest.json', replay['manifest_identity'])
    manifest = json.loads((root/'evidence_manifest.json').read_text())
    steps = config['reencode']['second_reverse_steps']
    actual = []

    class Observer:
        def __init__(self, **options):
            self.state, self.proof, self.archive, self.old_diag, self.refs = load_reference(
                root, manifest, options.pop('corruption'), options.pop('original_ids'))
            old_cfg_path = next(k for k in self.refs if k.endswith('/config.json'))
            old = json.loads((root/old_cfg_path).read_text())
            validate_reference_host(old, actual[0])
            options.update(horizon=steps, checkpoint_steps=selection(config['first_trajectory_selection'], steps)['normal_loop_counts'][:-1])
            self.second = SharedTrajectoryObserver(**options); self.names = self.second.names
        def terminal(self, logits): self.second.terminal(logits)
        def arrays(self, backbone): return replay_arrays(self.archive, self.second.arrays(backbone))
        def diagnostics(self):
            return dict(second=self.second.diagnostics(), reference_files=self.refs,
                first_and_cycle='archived, not executed in this condition',
                reference_first=self.old_diag['trajectory']['first'], second_reverse_steps=steps)

    def runtime_loader(common, assets):
        actual.append(common)
        return shared.load_guidance_runtime(dict(common, method=shared.ORIGINAL_METHOD), assets)

    def geometry(raw, dataset, corruption, lion, graph, reconstruct, ops, *, observer_factory,
                 paired_state_path, scheduler_observer, protocol, **unused):
        _, center, scale = ops.normalize(raw)
        observer = observer_factory(center, scale)
        destination = Path(paired_state_path); destination.parent.mkdir(parents=True, exist_ok=True)
        original = root/'paired'/dataset/corruption/destination.name
        with original.open('rb') as src, destination.open('xb') as dst: shutil.copyfileobj(src, dst)
        bridge = torch.from_numpy(observer.proof['bridge_points']).to('cuda')
        def schedule(scheduler): scheduler_observer(depth.scheduler_record(scheduler, steps))
        decoded = reconstruct(x=bridge, lion=lion, steps_back_local=steps,
            prepared_state=observer.state, trajectory_observer=observer.second,
            scheduler_observer=schedule, **protocol)
        metrics = dict(replay=dict(reference_files=observer.refs,
            second_chamfer_reference_sha256=tensor_hash(observer.state['local_latent'].view(len(bridge),2048,4)[:,:,:3]),
            first_decoded_sha256=tensor_hash(bridge), second_decoded_sha256=tensor_hash(decoded)))
        return shared.decoded_to_geometry(decoded, dataset, scale, center, ops).detach(), metrics

    shared.worker(directories, runtime_loader=runtime_loader, geometry_reconstructor=geometry, observer_provider=Observer)


if __name__ == '__main__':
    shared.main(configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)
