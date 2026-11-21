#!/usr/bin/env python3
"""Run registered, resumable CloudFixer XYZ rotation comparisons on one CUDA GPU."""
import argparse
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from cloudfixer_diffusion import verify_gamma
from cloudfixer_rotation import ARMS, KABSCH_STEPS, axis_rotation, rotation_angle, rotation_policy
from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, sha, unit_key
from research_artifacts import CORRUPTIONS
from scripts.run_cloudfixer_point2vec import (POLICIES, load_head, model_hash, options_for,
                                            proximal_policy, timed)
from scripts.run_cloudfixer_transfer import file_hash, verify_runtime_files
from scripts.setup_cloudfixer_point2vec_colab import runtime_identity


REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_rotation_pilot.json').read_text())
CONDITIONS = ('clean', *sorted(CORRUPTIONS))
DEV_CONDITIONS = ('clean', 'rotation', 'background')
COMPONENTS = ('denoiser', 'classifier', 'proximal', 'local_matching', 'matching', 'svd')


class CudaComponents:
    def __init__(self):
        self.events = []

    def measure(self, component, function, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.events.append((component, start, end))
        return result

    def milliseconds(self):
        return {name: sum(start.elapsed_time(end) for key, start, end in self.events if key == name)
                for name in COMPONENTS}


def init_runtime(args, spec):
    official = verify_runtime_files(args, spec)
    clean_path = args.dataset_root / 'data_original.npy'
    receipt = json.loads(Path('/content/cli_setup_verification.json').read_text())
    clean_hash = file_hash(clean_path)
    expected_clean = receipt['assets'][str(clean_path.relative_to('/content/3DD-TTA'))]
    if receipt['status'] != 'passed' or clean_hash != expected_clean['sha256'] or clean_hash != REGISTRY['clean_sha256']:
        raise ValueError('Clean array differs from the canonical setup receipt')
    supplemental = runtime_identity()
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Registered for one A100 GPU')
    os.chdir(official)
    sys.path.insert(0, str(official))
    import adapt
    if Path(adapt.__file__).resolve() != official / 'adapt.py':
        raise ValueError('Unexpected author entry point')
    options = options_for(adapt, args)
    adapt.set_seed(0)
    model = adapt.get_model(options, torch.device('cuda'))
    intended = model.gamma.gamma.detach().clone()
    model.load_state_dict(torch.load(args.diffusion, map_location='cpu'), strict=True)
    gamma = verify_gamma(model.gamma.gamma, intended)
    model = model.cuda().eval().requires_grad_(False)
    head = torch.nn.DataParallel(load_head(args.classifier, spec['classifier'])).cuda().eval().requires_grad_(False)
    labels = np.load(args.dataset_root / 'label.npy', allow_pickle=False).reshape(-1)
    if len(labels) != 2468 or set(labels.tolist()) != set(range(40)):
        raise ValueError('Unexpected original-label mapping')
    hashes = [model_hash(m) for m in (model, head)]
    environment = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                       gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                       clean_sha256=clean_hash, gamma=gamma, model_hashes=hashes, supplemental=supplemental,
                       cudnn=dict(enabled=torch.backends.cudnn.enabled,
                                  deterministic=torch.backends.cudnn.deterministic,
                                  benchmark=torch.backends.cudnn.benchmark,
                                  allow_tf32=torch.backends.cudnn.allow_tf32),
                       tf32=torch.backends.cuda.matmul.allow_tf32,
                       cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    return adapt, options, model, head, labels, environment


def dataset_for(adapt, options, condition, labels, class_order):
    class CleanDataset(adapt.ModelNet40C):
        def load_modelnet40_c(self, data_path, corruption, severity):
            return super().load_modelnet40_c(data_path, 'original', severity)

    options.dataset = f'modelnet40c_{condition}_5'
    dataset = (CleanDataset if condition == 'clean' else adapt.ModelNet40C)(options, partition='test')
    if (not np.array_equal(dataset.label_list.reshape(-1), labels)
            or [dataset.idx_to_label[i] for i in range(40)] != class_order):
        raise ValueError('Official preprocessing has an unexpected object-label map')
    return dataset


def item(dataset, ident, seed, label):
    # Shared input selection and independent common RNG per object and seed.
    rng = seed * 100_003 + ident
    np.random.seed(rng)
    x, target, mask, indices = dataset[ident]
    if int(np.asarray(target).item()) != label:
        raise ValueError('Original label changed during preprocessing')
    xyz = torch.from_numpy(x).unsqueeze(0).cuda().float()
    valid = torch.from_numpy(mask).reshape(1, -1).cuda().bool()
    indices = torch.from_numpy(indices).reshape(1, -1).cuda()
    return rng, xyz, valid, indices


def run_arm(model, head, xyz, mask, indices, arm, params, *, rng, noise_rotation=None):
    components = CudaComponents()
    with torch.random.fork_rng(devices=[xyz.device.index]):
        torch.manual_seed(rng)
        result, seconds, memory = timed(lambda: rotation_policy(
            model, head, xyz, mask, indices, arm, max_degrees=REGISTRY['max_angle_degrees'],
            measure=components.measure, noise_rotation=noise_rotation, **params))
    expected_calls = (9, 3 if arm in KABSCH_STEPS else 9)
    observed_calls = (result.diagnostics['denoiser_calls'], result.diagnostics['classifier_calls'])
    if observed_calls != expected_calls:
        raise ValueError(f'{arm} query count {observed_calls} differs from {expected_calls}')
    if not torch.isfinite(result.probabilities).all() or not torch.isfinite(result.logits).all():
        raise ValueError(f'Nonfinite classifier outputs in {arm}')
    prob = result.probabilities[0]
    row = dict(prediction=int(prob.argmax()), confidence=float(prob.max()), probabilities=prob.tolist(),
               logits=result.logits[0].tolist(), pose=result.pose.tolist(), seconds=seconds,
               peak_mib=memory, cuda_ms=components.milliseconds(), diagnostics=result.diagnostics)
    return row, result


def check_identity(model, head, xyz, mask, indices, rng):
    with torch.random.fork_rng(devices=[xyz.device.index]):
        torch.manual_seed(rng)
        logits, pm, clouds, probes, calls = proximal_policy(model, head, xyz, mask, POLICIES['proximal_pm'])
        result = rotation_policy(model, head, xyz, mask, indices, 'identity')
    assert probes == [20, 10, 0] and calls == 9
    torch.testing.assert_close(result.clouds, clouds, atol=0, rtol=0)
    torch.testing.assert_close(result.logits, logits, atol=0, rtol=0)
    torch.testing.assert_close(result.probabilities, pm, atol=0, rtol=0)


def synthetic_diagnostics(model, head, xyz, mask, indices, rng):
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    rotations = {
        'identity': identity,
        'small_x': axis_rotation(0, np.deg2rad(5), xyz),
        'compound': (axis_rotation(0, np.deg2rad(15), xyz) @ axis_rotation(1, np.deg2rad(-15), xyz)
                     @ axis_rotation(2, np.deg2rad(15), xyz)),
        'large_y': axis_rotation(1, np.deg2rad(60), xyz),
    }
    rows = {}
    for name, applied in rotations.items():
        observed = xyz if name == 'identity' else xyz @ applied
        before = np.rad2deg(rotation_angle(applied))
        valid_points = xyz[mask]
        unposed = torch.cdist(observed[mask], valid_points).square()
        before_scd = float(unposed.amin(0).mean() + unposed.amin(1).mean())
        rows[name] = {}
        for arm in KABSCH_STEPS:
            record, result = run_arm(model, head, observed, mask, indices, arm, {},
                                     rng=rng, noise_rotation=None if name == 'identity' else applied)
            points = (observed @ result.pose)[mask]
            distance = torch.cdist(points, valid_points).square()
            scd = float(distance.amin(0).mean() + distance.amin(1).mean())
            updates = record['diagnostics'].pop('updates')
            for update in updates:
                pose = torch.tensor(update['pose'], device=xyz.device, dtype=xyz.dtype)
                update['residual_degrees'] = np.rad2deg(rotation_angle(applied @ pose))
            rows[name][arm] = dict(before_degrees=before, before_scd=before_scd, scd=scd,
                                   residual_degrees=np.rad2deg(rotation_angle(applied @ result.pose)),
                                   updates=updates, pose=record['pose'])
    return rows


def search_lock(rows):
    from statistics import mean

    def delta(name, condition):
        return 100 * mean(
            int(row['arms'][name]['prediction'] == row['label'])
            - int(row['arms']['identity']['prediction'] == row['label'])
            for row in rows if row['condition'] == condition)

    grid = REGISTRY['search_grid']
    outcomes = [dict(params=params, deltas={c: delta(f'search_{i}', c) for c in DEV_CONDITIONS})
                for i, params in enumerate(grid)]
    eligible = [x for x in outcomes if min(x['deltas']['clean'], x['deltas']['background']) >= -2]
    winner = max(eligible, key=lambda x: (x['deltas']['rotation'], -x['params']['theta'],
                                         x['params']['threshold'], x['params']['penalty'])) if eligible else outcomes[-1]
    return dict(params=winner['params'], development_passed=bool(eligible), development_scores=outcomes)


def phase_scope(phase, freeze, decision):
    if phase == 'development':
        return DEV_CONDITIONS, REGISTRY['development_ids'], REGISTRY['seeds'], {}
    if phase == 'smoke':
        return ('background', 'rotation'), REGISTRY['development_ids'][:1], [0], dict.fromkeys(ARMS, {})
    if phase == 'validation':
        return CONDITIONS, REGISTRY['validation_ids'], REGISTRY['seeds'], {
            arm: freeze['params'] if arm not in KABSCH_STEPS else {} for arm in ARMS}
    selected = decision.get('next_arm')
    if selected not in (*KABSCH_STEPS, 'output', 'feedback') or selected == 'identity':
        raise ValueError('A validated candidate must pass the continuation gates before full scoring')
    return CONDITIONS[1:], list(range(2468)), [REGISTRY['seed_full']], {
        'identity': {}, selected: freeze['params'] if selected not in KABSCH_STEPS else {}}


def pack(rows):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, records=np.frombuffer(canonical(rows), dtype=np.uint8))
    return buffer.getvalue()


def unpack(data, unit, labels, expected_arms=None):
    with np.load(io.BytesIO(data), allow_pickle=False) as payload:
        rows = json.loads(payload['records'].tobytes())
    if [row['id'] for row in rows] != unit['ids']:
        raise ValueError('Recovered IDs do not match the block')
    for row in rows:
        if (row['seed'] != unit['seed'] or row['condition'] != unit['condition']
                or row['label'] != labels[row['id']]
                or (expected_arms is not None and set(row['arms']) != set(expected_arms))):
            raise ValueError('Recovered record does not match the paired protocol')
    return rows


def evaluate(args, spec, store, budget):
    adapt, options, model, head, labels, environment = init_runtime(args, spec)
    config = json.loads((args.output / 'config.json').read_text())
    signature = sha(canonical(environment))
    root = args.output / 'raw'
    root.mkdir()
    (args.output / 'environment.txt').write_text(json.dumps(environment, indent=2) + '\n')
    store.put('execution.json', canonical(dict(spec=spec, registry=REGISTRY)))
    if args.phase != 'smoke' and json.loads(store.get('smoke_ready.json'))['runtime'] != signature:
        raise ValueError('Smoke verification does not match the live environment')
    freeze = json.loads(store.get('search_lock.json')) if args.phase in ('validation', 'full') else {}
    decision = json.loads(store.get('validation_decision.json')) if args.phase == 'full' else {}
    conditions, ids, seeds, params = phase_scope(args.phase, freeze, decision)
    campaign = sha(canonical(dict(spec=spec, phase=args.phase, freeze=freeze, decision=decision,
                                  registry=REGISTRY, runtime=signature)))
    config.update(campaign=campaign, freeze=freeze, decision=decision, conditions=conditions,
                  ids=ids, seeds=seeds, params=params, environment_sha256=signature)
    (args.output / 'config.json').write_bytes(canonical(config))
    cache = MountStore(store.folder / f'{args.phase}_{campaign[:16]}')
    cached = load_units(cache, campaign)
    length = 1 if args.phase == 'smoke' else 8 if args.phase == 'development' else 16
    slots = [dict(campaign=campaign, condition=cond, seed=seed, ids=ids[i:i+length])
             for cond in conditions for seed in seeds for i in range(0, len(ids), length)]
    keys = [unit_key(unit) for unit in slots]
    if set(cached) != set(keys[:len(cached)]):
        raise ValueError('Restored units must be a contiguous prefix of the registered campaign')
    rows = []
    warmed = set()
    for block, (key, unit) in enumerate(zip(keys, slots), 1):
        cond, seed = unit['condition'], unit['seed']
        if cond not in warmed:
            dataset = dataset_for(adapt, options, cond, labels, spec['classifier']['class_order'])
            warmed.add(cond)
            if args.phase == 'development':
                arms = {'identity': ('identity', {})}
                arms.update({f'search_{i}': ('output', p) for i, p in enumerate(REGISTRY['search_grid'])})
            else:
                arms = {arm: (arm, p) for arm, p in params.items()}
            # A shape-matched untimed pass first, using the same per-object RNG.
            rng, *batch = item(dataset, ids[0], seed, int(labels[ids[0]]))
            if args.phase == 'smoke':
                check_identity(model, head, *batch, rng)
            for arm, p in arms.values():
                run_arm(model, head, *batch, arm, p, rng=rng)
        with budget.operation(1200):
            if key in cached:
                data = cached[key]['data']
            else:
                found = []
                for ident in unit['ids']:
                    label = int(labels[ident])
                    rng, *batch = item(dataset, ident, seed, label)
                    order = list(arms)
                    offset = rng % len(order)
                    order = order[offset:] + order[:offset]
                    scores = {name: run_arm(model, head, *batch, *arms[name], rng=rng)[0] for name in order}
                    if 'feedback' in scores and 'output' in scores:
                        left, right = scores['feedback'], scores['output']
                        if (left['diagnostics']['choice'] != right['diagnostics']['choice']
                                or left['logits'][0] != right['logits'][0]):
                            raise ValueError('The t20 choice differs between paired search paths')
                    row = dict(condition=cond, seed=seed, id=ident, label=label, arms=scores,
                               timing_order=order, rng=rng)
                    if args.phase == 'development' and seed == 0 and cond != 'rotation':
                        row['synthetic'] = synthetic_diagnostics(model, head, *batch, rng)
                    found.append(row)
                data = pack(found)
                commit_unit(cache, unit, data)
            restored = unpack(data, unit, labels, arms)
            if args.phase != 'full':
                rows.extend(restored)
            payload_name = f'block_{block:04d}.npz'
            (root / payload_name).write_bytes(data)
            (root / f'block_{block:04d}.json').write_bytes(
                canonical(dict(unit=unit, bytes=len(data), sha256=sha(data))))
            print('ROTATION_BLOCK', block, len(slots), cond, seed, key, flush=True)
    if [model_hash(m) for m in (model, head)] != environment['model_hashes']:
        raise ValueError('Frozen model state changed during inference')
    if args.phase == 'smoke':
        store.put('smoke_ready.json', canonical(dict(runtime=signature, campaign=campaign)))
    if args.phase == 'development':
        lock = dict(search_lock(rows), campaign=campaign)
        store.put('search_lock.json', canonical(lock))
        (args.output / 'search_lock.json').write_bytes(canonical(lock))
    config.update(execution_status='complete', blocks=len(slots))
    (args.output / 'config.json').write_bytes(canonical(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'development', 'validation', 'full'), required=True)
    for arg in ('manifest', 'output', 'drive-folder'):
        parser.add_argument(f'--{arg}', type=Path, required=True)
    for arg, default in (
        ('official-root', '/content/CloudFixer'), ('dataset-root', '/content/3DD-TTA/data/modelnet40_c'),
        ('classifier', '/content/CloudFixer/checkpoints/point2vec_modelnet40.ckpt'),
        ('diffusion', '/content/CloudFixer/checkpoints/diffusion_model_transformer_modelnet40.npy'),
    ):
        parser.add_argument(f'--{arg}', type=Path, default=Path(default))
    args = parser.parse_args()
    if not Path('/content/drive').is_mount() or Path('/content/drive/MyDrive/thesis') not in args.drive_folder.parents:
        raise ValueError('Owned CloudFixer rotation Drive mount is required')
    for name in ('output', 'manifest', 'official_root', 'dataset_root', 'classifier', 'diffusion'):
        setattr(args, name, getattr(args, name).resolve())
    spec = json.loads(args.manifest.read_text())
    if spec['registration'] != REGISTRY:
        raise ValueError('Source manifest and pilot registry disagree')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (args.output / 'notes.md').write_text(
        'Frozen Point2Vec + CloudFixer XYZ. Gradient-free rotation pilot, B1. '
        'GPU component values are CUDA-event ms. Wall time is synchronized; peak allocated memory is MiB. '
        'Registry ID-addressed draws and author pre-processing. See the source registration for selection and coverage.\n')
    path = args.output / 'config.json'
    path.write_bytes(canonical(dict(run_id=args.output.name, phase=args.phase, spec=spec, execution_status='running')))
    store = MountStore(args.drive_folder)
    budget = Budget(MountStore(store.folder / ('full_budget' if args.phase == 'full' else 'pilot_budget')),
                    cap=REGISTRY['full_pair_budget_seconds' if args.phase == 'full' else 'pilot_budget_seconds'])
    try:
        evaluate(args, spec, store, budget)
    except BaseException:
        path.write_bytes(canonical(dict(json.loads(path.read_text()), execution_status='failed')))
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    print('CLOUDFIXER_ROTATION_PHASE_DONE', args.phase, flush=True)


if __name__ == '__main__':
    main()
