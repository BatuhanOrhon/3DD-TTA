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

from cloudfixer_background import BACKGROUND_ARMS, EXPERIMENT_ARMS, anchor_weights
from cloudfixer_background_checkpoint import block_name, mirror_group, restore_or_score
from cloudfixer_diffusion import verify_gamma
from cloudfixer_rotation import (ARMS, OUTPUT_ARMS, SURFACE_ARM, GRAPH_K, GRAPH_STEPS, KABSCH_STEPS, SCD_SEARCH_STEPS, SPECTRAL_STEPS,
                                 axis_rotation, rotation_angle, rotation_policy)
from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, sha, unit_key
from research_artifacts import CORRUPTIONS
from scripts.run_cloudfixer_point2vec import (POLICIES, load_head, model_hash, options_for,
                                            proximal_policy, timed)
from scripts.run_cloudfixer_transfer import file_hash, verify_runtime_files
from scripts.setup_cloudfixer_point2vec_colab import runtime_identity


REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_rotation_pilot.json').read_text())
SPECTRAL_REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_spectral_rotation_pilot.json').read_text())
SCD_REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_scd_rotation_pilot.json').read_text())
BG_REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_background_pilot.json').read_text())
FOLLOW_REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_background_followup.json').read_text())
OUTPUT_REGISTRY = json.loads((ROOT / 'cfgs/cloudfixer_output_rotation.json').read_text())
OUTPUT_PHASES = ('feedback_smoke', 'feedback_focus', 'feedback_other', 'feedback_all15_smoke', 'feedback_all15')
SURFACE_PHASES = ('feedback_surface_smoke', 'feedback_surface512', 'feedback_surface_full')
SURFACE_CONFIG = json.loads((ROOT / 'cfgs/cloudfixer_surface512.json').read_text())
SURFACE_SMOKE = json.loads((ROOT / 'cfgs/cloudfixer_surface_smoke.json').read_text())
FOLLOW_PHASES = ('background_follow_smoke', 'background_follow_focus', 'background_follow_scd', 'background_follow_other')
CONDITIONS = ('clean', *sorted(CORRUPTIONS))
DEV_CONDITIONS = ('clean', 'rotation', 'background')
COMPONENTS = ('denoiser', 'classifier', 'proximal', 'local_matching', 'matching', 'svd',
              'spectral_graph', 'spectral_filter', 'scd_search')


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
    three_probes = arm in KABSCH_STEPS or arm in SPECTRAL_STEPS or arm in BACKGROUND_ARMS or arm == 'scd_search' or arm in (*OUTPUT_ARMS, SURFACE_ARM)
    expected_calls = (9, 3 if three_probes else 9)
    if arm in BACKGROUND_ARMS and sum(key == 'denoiser' for key, _, _ in components.events) != 9:
        raise ValueError('Unexpected measured denoiser count')
    observed_calls = (result.diagnostics['denoiser_calls'], result.diagnostics['classifier_calls'])
    if observed_calls != expected_calls:
        raise ValueError(f'{arm} query count {observed_calls} differs from {expected_calls}')
    if not torch.isfinite(result.probabilities).all() or not torch.isfinite(result.logits).all():
        raise ValueError(f'Nonfinite classifier outputs in {arm}')
    prob = result.probabilities[0]
    row = dict(prediction=int(prob.argmax()), confidence=float(prob.max()), probabilities=prob.tolist(),
               logits=result.logits[0].tolist(), pose=result.pose.tolist(), seconds=seconds,
               peak_mib=memory, cuda_ms=components.milliseconds(), diagnostics=result.diagnostics,
               graph_builds=sum(key == 'spectral_graph' for key, _, _ in components.events))
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


def synthetic_diagnostics(model, head, xyz, mask, indices, rng, *, arms=KABSCH_STEPS):
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
        for arm in arms:
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
    if phase in SURFACE_PHASES:
        arms = dict.fromkeys(('identity', SURFACE_ARM), {})
        if phase == 'feedback_surface_smoke':
            return tuple(dict.fromkeys(r['condition'] for r in SURFACE_SMOKE)), list(dict.fromkeys(r['id'] for r in SURFACE_SMOKE)), [0], arms
        if phase == 'feedback_surface_full':
            if not decision['pass']:
                raise ValueError('Full evaluation requires a passing surface512 decision')
            return CONDITIONS[1:], list(range(2468)), [SURFACE_CONFIG['full_seed']], {SURFACE_ARM: {}}
        return CONDITIONS, SURFACE_CONFIG['ids'], SURFACE_CONFIG['seeds'], arms
    if phase in ('feedback_all15_smoke', 'feedback_all15'):
        arms = dict.fromkeys(('identity', 'pose_pca_t50', 'pose_pca_all'), {})
        if phase == 'feedback_all15_smoke':
            return ('rotation', 'clean'), OUTPUT_REGISTRY['smoke_ids'], [0], arms
        return tuple(c for c in CONDITIONS if c != 'clean'), OUTPUT_REGISTRY['other_ids'], [0], arms
    if phase in OUTPUT_PHASES:
        conditions, seeds = ('rotation', 'clean'), OUTPUT_REGISTRY['seeds']
        arms = ('identity', *OUTPUT_ARMS)
        ids = OUTPUT_REGISTRY['ids']
        if phase == 'feedback_smoke':
            ids, seeds = OUTPUT_REGISTRY['smoke_ids'], [0]
        if phase == 'feedback_other':
            ids, seeds = OUTPUT_REGISTRY['other_ids'], [0]
            conditions = tuple(c for c in CONDITIONS if c != 'clean')
            selected = decision['eligible']
            if not selected or not set(selected).issubset(OUTPUT_ARMS):
                raise ValueError('Expansion requires a passing focus candidate')
            arms = ('identity', *selected)
        return conditions, ids, seeds, dict.fromkeys(arms, {})
    if phase in FOLLOW_PHASES:
        params = dict(BG_REGISTRY, wide_k=FOLLOW_REGISTRY['wide_k'])
        view_arms = ('identity', 'bg_view', 'bg_random_view')
        scd_arms = ('identity', 'bg_weighted', 'bg_multiscale', 'bg_multiscale_scalar')
        conditions = ('background', 'rotation', 'clean')
        ids, seeds, names = FOLLOW_REGISTRY['ids'], FOLLOW_REGISTRY['seeds'], view_arms
        if phase == 'background_follow_smoke':
            conditions, ids, seeds = conditions[:2], FOLLOW_REGISTRY['geometry_ids'][:1], [0]
            names = view_arms + scd_arms[1:]
        elif phase == 'background_follow_scd':
            ids, names = FOLLOW_REGISTRY['scd_ids'], scd_arms
        elif phase == 'background_follow_other':
            conditions, seeds = tuple(c for c in CONDITIONS if c not in conditions), [0]
        return conditions, ids, seeds, {arm: {} if arm == 'identity' else {'background_config': params} for arm in names}
    if phase.startswith('background'):
        arms = {arm: {} for arm in ('identity', 'kabsch_progressive')}
        arms.update({arm: {'background_config': BG_REGISTRY} for arm in EXPERIMENT_ARMS})
        if phase == 'background_smoke':
            return ('background', 'rotation'), BG_REGISTRY['dev_ids'][:1], [0], arms
        if phase == 'background_pm':
            comparisons = {a: arms[a] for a in ('identity', 'bg_mid', 'bg_high', 'bg_start_only')}
            return ('background', 'rotation', 'clean'), BG_REGISTRY['pm_ids'], BG_REGISTRY['seeds'], comparisons
        if phase == 'background_dev':
            return ('clean', 'background'), BG_REGISTRY['dev_ids'], [0], arms
        return ('background', 'rotation', 'clean', *(c for c in CONDITIONS if c not in DEV_CONDITIONS)), BG_REGISTRY['ids'], BG_REGISTRY['seeds'], arms
    if phase.startswith(('scd', 'spectral')):
        if phase.startswith('scd'):
            ids = SCD_REGISTRY['ids']
            arms = dict.fromkeys(('identity', 'kabsch_fixed', 'scd_search'), {})
        else:
            ids = SPECTRAL_REGISTRY['ids']
            arms = dict.fromkeys((*KABSCH_STEPS, *SPECTRAL_STEPS), {})
        if phase.endswith('_smoke'):
            return ('background', 'rotation'), ids[:1], [0], arms
        return CONDITIONS, ids, REGISTRY['seeds'], arms
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


@torch.no_grad()
def audit_start_noise(store, model, head, batch, rng, arm, params, condition, signature):
    from unittest.mock import patch
    x = batch[0]
    with torch.random.fork_rng(devices=[x.device.index]):
        torch.manual_seed(rng)
        common_noise = torch.randn_like(x)
    observed = []
    predictor = model.phi
    def verify(state, t, node_mask):
        index = int((t * 500).round().item())
        if not observed:
            gamma = model.gamma(t)
            alpha = model.alpha(gamma, target_tensor=x)
            sigma = model.sigma(gamma, target_tensor=x)
            torch.testing.assert_close(state, alpha * x + sigma * common_noise, rtol=0, atol=0)
        observed.append(dict(index=index, logsnr=float(-model.gamma(t).item())))
        return predictor(state, t, node_mask)
    with patch.object(model, 'phi', verify):
        run_arm(model, head, *batch, arm, params, rng=rng)
    registered = {'identity': list(range(80, -1, -10)), 'bg_mid': BG_REGISTRY['mid_grid'],
                  'bg_start_only': BG_REGISTRY['start_only_grid'], 'bg_high': BG_REGISTRY['high_grid']}
    if [r['index'] for r in observed] != registered[arm]:
        raise ValueError('Denoiser did not execute the registered noise grid')
    store.put(f'noise_{signature}_{condition}_{arm}.json', canonical(dict(grid=observed, rng=rng,
              noise_sha256=sha(common_noise.cpu().contiguous().numpy().tobytes()))))


@torch.no_grad()
def synthetic_support(x, mask, seed):
    points = torch.unique(x[0, mask.reshape(-1)], dim=0)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    amount = max(1, len(points) // 10)
    extent = points.amax(0) - points.amin(0)
    uniform = points.amin(0) + torch.rand((amount, 3), generator=rng).to(x) * extent
    clustered = points.amax(0) + extent * (.1 + .02 * torch.randn((amount, 3), generator=rng).to(x))
    result = {}
    for name, extra in [('uniform', uniform), ('clustered', clustered)]:
        xyz = torch.cat((points, extra)).unsqueeze(0)
        valid = torch.ones_like(xyz[..., 0], dtype=torch.bool)
        weights = anchor_weights(xyz, valid, BG_REGISTRY['k'])
        weak = weights[0] < BG_REGISTRY['filter_below']
        rotated = xyz[..., [1, 2, 0]]
        delta = (anchor_weights(rotated, valid, BG_REGISTRY['k']) - weights).abs().max()
        result[name] = dict(inliers=len(points), outliers=amount, weak_inliers=int(weak[:len(points)].sum()),
                            weak_outliers=int(weak[len(points):].sum()), rotation_max_abs=float(delta))
    return result


def evaluate(args, spec, store, budget):
    adapt, options, model, head, labels, environment = init_runtime(args, spec)
    config = json.loads((args.output / 'config.json').read_text())
    signature = sha(canonical(environment))
    root = args.output / 'raw'
    root.mkdir()
    (args.output / 'environment.txt').write_text(json.dumps(environment, indent=2) + '\n')
    store.put('execution.json', canonical(dict(spec=spec, registry=REGISTRY)))
    bg = args.phase.startswith('background') or args.phase in (*OUTPUT_PHASES, *SURFACE_PHASES)
    is_smoke = args.phase in ('feedback_surface_smoke', 'feedback_all15_smoke', 'feedback_smoke', 'background_follow_smoke', 'background_smoke', 'smoke', 'spectral_smoke', 'scd_smoke')
    smoke_key = f'smoke_{signature}.json' if bg else 'smoke_ready.json'
    if not is_smoke and json.loads(store.get(smoke_key))['runtime'] != signature:
        raise ValueError('Smoke verification does not match the live environment')
    freeze = json.loads(store.get('search_lock.json')) if args.phase in ('validation', 'full') else {}
    decision = json.loads(store.get('validation_decision.json')) if args.phase == 'full' else {}
    if args.phase == 'feedback_surface_full':
        decision = json.loads(store.get('surface512_decision.json'))
    if args.phase == 'feedback_other':
        decision = json.loads(store.get('feedback_decision.json'))
    conditions, ids, seeds, params = phase_scope(args.phase, freeze, decision)
    campaign = sha(canonical(dict(spec=spec, phase=args.phase, freeze=freeze, decision=decision,
                                  registry=REGISTRY, runtime=('verified-background-pilot' if bg and not is_smoke else signature))))
    config.update(campaign=campaign, freeze=freeze, decision=decision, conditions=conditions,
                  ids=ids, seeds=seeds, params=params, environment_sha256=signature)
    (args.output / 'config.json').write_bytes(canonical(config))
    if bg:
        store.put('environment_' + signature + '.json', canonical(environment))
        store.put('protocol_' + campaign + '.json', canonical(dict(
            phase=args.phase, spec=spec, seeds=seeds, ids=ids, conditions=conditions, arms=params)))
    cache = MountStore(store.folder / f'{args.phase}_{campaign[:16]}')
    cached = set(cache.keys('unit_')) if bg else load_units(cache, campaign)
    length = 1 if is_smoke else 8 if args.phase == 'development' else 16
    if bg and not is_smoke:
        length = 4
    shards = [ids[:128], ids[128:]] if args.phase == 'feedback_surface512' else [ids]
    slots = [dict(campaign=campaign, condition=cond, seed=seed, ids=shard[i:i+length])
             for shard in shards for cond in conditions for seed in seeds for i in range(0, len(shard), length)]
    keys = [unit_key(unit) for unit in slots]
    if bg and not cached.issubset({block_name(u) for u in slots}):
        raise ValueError('Unexpected checkpoint in Background campaign')
    if not bg and set(cached) != set(keys[:len(cached)]):
        raise ValueError('Restored units must be a contiguous prefix of the registered campaign')
    rows = []
    pending_mirror = []
    active_condition = None
    previous_condition = None
    carried = {}
    if args.phase == 'feedback_surface_full':
        validation_campaign = decision['campaign']
        validation_store = MountStore(store.folder / ('feedback_surface512_' + validation_campaign[:16]))
        validation_protocol = json.loads(store.get('protocol_' + validation_campaign + '.json'))
        if validation_protocol['spec'] != spec or validation_protocol['ids'] != SURFACE_CONFIG['ids']:
            raise ValueError('Validation data do not share the exact full-evaluation policy')
    for block, (key, unit) in enumerate(zip(keys, slots), 1):
        cond, seed = unit['condition'], unit['seed']
        already_scored = bg and block_name(unit) in cached
        if bg:
            arms = {arm: (arm, p) for arm, p in params.items()}
        if cond != active_condition and (not already_scored or is_smoke):
            dataset = dataset_for(adapt, options, cond, labels, spec['classifier']['class_order'])
            active_condition = cond
            if args.phase == 'development':
                arms = {'identity': ('identity', {})}
                arms.update({f'search_{i}': ('output', p) for i, p in enumerate(REGISTRY['search_grid'])})
            else:
                arms = {arm: (arm, p) for arm, p in params.items()}
            # A shape-matched untimed pass first, using the same per-object RNG.
            rng, *batch = item(dataset, ids[0], seed, int(labels[ids[0]]))
            if is_smoke:
                check_identity(model, head, *batch, rng)
            for arm, p in arms.values():
                warm, _ = run_arm(model, head, *batch, arm, p, rng=rng)
                if bg and is_smoke:
                    if arm in ('identity', 'bg_mid', 'bg_high', 'bg_start_only'):
                        audit_start_noise(store, model, head, batch, rng, arm, p, cond, signature)
                    name = f'golden_{cond}_{arm}.json'
                    golden = dict(logits=warm['logits'], prediction=warm['prediction'], label=int(labels[ids[0]]))
                    if name in store.keys('golden_'):
                        saved = json.loads(store.get(name))
                        if (saved['label'] != golden['label'] or saved['prediction'] != golden['prediction']
                                or not np.allclose(saved['logits'], golden['logits'], rtol=1e-4, atol=1e-4)):
                            raise ValueError('Cross-runtime smoke replay differs for ' + name)
                    else:
                        store.put(name, canonical(golden))
        if args.phase == 'feedback_surface_full' and cond != previous_condition and not already_scored:
            carried = {}
            previous_condition = cond
            pilot_ids = SURFACE_CONFIG['ids']
            for j in range(0, len(pilot_ids), 4):
                prior_unit = dict(condition=cond, seed=seed, ids=pilot_ids[j:j+4], campaign=validation_campaign)
                saved = restore_or_score(validation_store, prior_unit, {block_name(prior_unit)}, None,
                                         lambda blob: unpack(blob, prior_unit, labels, ('identity', SURFACE_ARM)))
                for r in unpack(saved, prior_unit, labels, ('identity', SURFACE_ARM)):
                    if r['rng'] != seed * 100_003 + r['id']:
                        raise ValueError('Candidate RNG does not match the full protocol')
                    carried[r['id']] = dict(r, arms={SURFACE_ARM: r['arms'][SURFACE_ARM]},
                                            timing_order=[SURFACE_ARM], reused_from=validation_campaign)
            if len(carried) != len(pilot_ids):
                raise ValueError('Incomplete seed-matched validation records')
        with budget.operation(180 if bg else 1200):
            def score_block():
                found = []
                for ident in unit['ids']:
                    if ident in carried:
                        found.append(carried[ident])
                        continue
                    label = int(labels[ident])
                    rng, *batch = item(dataset, ident, seed, label)
                    order = list(arms)
                    offset = rng % len(order)
                    order = order[offset:] + order[:offset]
                    if args.phase == 'feedback_surface512':
                        timing = ident in SURFACE_CONFIG['timing_ids']
                        order = order if timing else [SURFACE_ARM]
                        if timing and ((SURFACE_CONFIG['timing_ids'].index(ident) + seed + conditions.index(cond)) % 2 == 0):
                            order = ['identity', SURFACE_ARM]
                        elif timing:
                            order = [SURFACE_ARM, 'identity']
                    scores = {name: run_arm(model, head, *batch, *arms[name], rng=rng)[0] for name in order}
                    if args.phase in ('feedback_surface512', 'feedback_surface_smoke'):
                        proposed = scores[SURFACE_ARM]
                        update, = proposed['diagnostics']['updates']
                        applied = update['reason'] == 'accepted'
                        if 'identity' not in scores:
                            if applied:
                                scores['identity'] = run_arm(model, head, *batch, 'identity', {}, rng=rng)[0]
                                order.append('identity')
                            else:
                                scores['identity'] = {key: proposed[key] for key in ('prediction', 'confidence', 'probabilities', 'logits')}
                                scores['identity']['from_noop'] = True
                        if not applied and proposed['logits'] != scores['identity']['logits']:
                            raise ValueError('Rejected surface rotation changed baseline logits')
                        if args.phase == 'feedback_surface_smoke':
                            for known in SURFACE_SMOKE:
                                if (cond, ident) == (known['condition'], known['id']):
                                    if applied != known['apply'] or label != known['label']:
                                        raise ValueError('Surface decision differs from archived geometry')
                                    for arm, key in ((SURFACE_ARM, 'logits'), ('identity', 'base_logits')):
                                        np.testing.assert_allclose(scores[arm]['logits'], known[key], rtol=1e-4, atol=1e-4)
                    if args.phase in FOLLOW_PHASES and 'bg_view' in scores:
                        base = scores['identity']['logits']
                        view, random = scores['bg_view'], scores['bg_random_view']
                        if not all(e['logits'][i] == base[i] for e in (view, random) for i in (0, 2)):
                            raise ValueError('Filtered PM changed a shared t20 or terminal prediction')
                        if view['diagnostics']['t10_view_points'] != random['diagnostics']['t10_view_points']:
                            raise ValueError('Random t10 control has a different point count')
                    if 'feedback' in scores and 'output' in scores:
                        left, right = scores['feedback'], scores['output']
                        if (left['diagnostics']['choice'] != right['diagnostics']['choice']
                                or left['logits'][0] != right['logits'][0]):
                            raise ValueError('The t20 choice differs between paired search paths')
                    row = dict(condition=cond, seed=seed, id=ident, label=label, arms=scores,
                               timing_order=order, rng=rng)
                    if args.phase == 'feedback_focus' and cond == 'clean' and seed == 0 and ident in ids[:8]:
                        row['synthetic'] = synthetic_diagnostics(model, head, *batch, rng, arms=arms)
                    if args.phase == 'development' and seed == 0 and cond != 'rotation':
                        row['synthetic'] = synthetic_diagnostics(model, head, *batch, rng)
                    if args.phase in ('spectral', 'scd') and seed == 0 and cond in ('clean', 'background'):
                        row['synthetic'] = synthetic_diagnostics(model, head, *batch, rng, arms=arms)
                    if bg:
                        row['runtime_sha256'] = signature
                        if args.phase == 'background_dev' and cond == 'clean':
                            row['support_diagnostic'] = synthetic_support(batch[0], batch[1], rng)
                    found.append(row)
                return pack(found)

            if bg:
                data = restore_or_score(cache, unit, cached, score_block,
                                        lambda blob: unpack(blob, unit, labels, arms))
            elif key in cached:
                data = cached[key]['data']
            else:
                data = score_block()
                commit_unit(cache, unit, data)
            restored = unpack(data, unit, labels, arms)
            if args.phase != 'full' and args.phase not in SURFACE_PHASES:
                rows.extend(restored)
            payload_name = f'block_{block:04d}.npz'
            (root / payload_name).write_bytes(data)
            (root / f'block_{block:04d}.json').write_bytes(
                canonical(dict(unit=unit, bytes=len(data), sha256=sha(data))))
            if bg:
                pending_mirror.append(unit)
                mirror_blocks = 32 if args.phase in SURFACE_PHASES else 4
                if len(pending_mirror) == mirror_blocks or block == len(slots):
                    target = args.output.parent / f'{args.output.name}_mirror_{block:04d}.zip'
                    digest = mirror_group(cache, pending_mirror, target)
                    print('BACKGROUND_MIRROR', target, digest, flush=True)
                    pending_mirror.clear()
                print('BACKGROUND_CHECKPOINT', 'restored' if already_scored else 'committed',
                      block, len(slots), cache.folder / block_name(unit), sha(data), flush=True)
            print('ROTATION_BLOCK', block, len(slots), cond, seed, key, flush=True)
    if [model_hash(m) for m in (model, head)] != environment['model_hashes']:
        raise ValueError('Frozen model state changed during inference')
    if is_smoke:
        store.put(smoke_key, canonical(dict(runtime=signature, campaign=campaign)))
    if args.phase == 'development':
        lock = dict(search_lock(rows), campaign=campaign)
        store.put('search_lock.json', canonical(lock))
        (args.output / 'search_lock.json').write_bytes(canonical(lock))
    config.update(execution_status='complete', blocks=len(slots))
    (args.output / 'config.json').write_bytes(canonical(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'development', 'validation', 'full',
                                            'spectral_smoke', 'spectral', 'scd_smoke', 'scd',
                                            'background_smoke', 'background_pm', 'background_dev', 'background', *FOLLOW_PHASES, *OUTPUT_PHASES, *SURFACE_PHASES), required=True)
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
    if args.phase.startswith('spectral') and (
        spec['spectral_registration'] != SPECTRAL_REGISTRY
        or (SPECTRAL_REGISTRY['neighbors'], SPECTRAL_REGISTRY['sweeps']) != (GRAPH_K, GRAPH_STEPS)
        or set(SPECTRAL_REGISTRY['ids']) & set(REGISTRY['development_ids'] + REGISTRY['validation_ids'])
    ):
        raise ValueError('Spectral manifest or reserved cohort differs from registration')
    if args.phase.startswith('scd') and (
        spec['scd_registration'] != SCD_REGISTRY or list(SCD_SEARCH_STEPS) != SCD_REGISTRY['step_degrees']
        or len(set(SCD_REGISTRY['ids'])) != 12
        or set(SCD_REGISTRY['ids']) & set(REGISTRY['development_ids'] + REGISTRY['validation_ids'] + SPECTRAL_REGISTRY['ids'])
    ):
        raise ValueError('SCD manifest or reserved cohort differs from registration')
    if args.phase.startswith('background') and spec['background_registration'] != BG_REGISTRY:
        raise ValueError('Background study differs from registered parameters')
    if args.phase in OUTPUT_PHASES and spec['output_rotation'] != OUTPUT_REGISTRY:
        raise ValueError('Output pose manifest differs from registration')
    if args.phase in FOLLOW_PHASES and spec['background_followup'] != FOLLOW_REGISTRY:
        raise ValueError('Follow-up manifest differs from registration')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    (args.output / 'notes.md').write_text(
        'Frozen Point2Vec + CloudFixer XYZ. Gradient-free rotation pilot, B1. '
        'GPU component values are CUDA-event ms. Wall time is synchronized; peak allocated memory is MiB. '
        'Registry ID-addressed draws and author pre-processing. See the source registration for selection and coverage.\n')
    path = args.output / 'config.json'
    path.write_bytes(canonical(dict(run_id=args.output.name, phase=args.phase, spec=spec, execution_status='running')))
    store = MountStore(args.drive_folder)
    if args.phase in SURFACE_PHASES:
        scope, cap = 'surface', 3600 * SURFACE_CONFIG['duration_hours']
    elif args.phase.startswith('feedback_all15'):
        scope, cap = 'feedback', 4*3600
    elif args.phase in OUTPUT_PHASES:
        scope, cap = 'feedback', OUTPUT_REGISTRY['budget_seconds']
    elif args.phase.startswith('background'):
        scope, cap = 'background', (FOLLOW_REGISTRY if args.phase in FOLLOW_PHASES else BG_REGISTRY)['budget_seconds']
    elif args.phase.startswith(('scd', 'spectral')):
        scope = args.phase.split('_')[0]
        cap = (SCD_REGISTRY if scope == 'scd' else SPECTRAL_REGISTRY)['budget_seconds']
    else:
        scope = 'full_pair' if args.phase == 'full' else 'pilot'
        cap = REGISTRY[scope + '_budget_seconds']
    budget = Budget(MountStore(store.folder / (scope + '_budget')), cap=cap)
    try:
        evaluate(args, spec, store, budget)
    except BaseException:
        path.write_bytes(canonical(dict(json.loads(path.read_text()), execution_status='failed')))
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    print('CLOUDFIXER_ROTATION_PHASE_DONE', args.phase, flush=True)


if __name__ == '__main__':
    main()
