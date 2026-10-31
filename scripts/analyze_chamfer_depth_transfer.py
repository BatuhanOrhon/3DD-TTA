"""CPU acceptance of sealed frozen-N8 transfer bundles and paired readouts.

``analyze_bundle(path, binding, cache_root)`` requires a binding containing
``expected_config`` (dataset/head/seed/IDs, source/assets/checkpoint identities)
and ``required_source_files``. Physical caches retain paired/seedN/dataset/
corruption/basename beneath cache_root. Raw bundles are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research_artifacts import CORRUPTIONS
from scripts.analyze_legacy_trajectory import policies
from scripts.analyze_scan_trajectory import Bundle, require
from scripts.eval_chamfer_depth import validate_transfer_config

CRITICAL_SOURCES = ('tta.py', 'shared_trajectory.py', 'legacy_trajectory.py', 'research_artifacts.py',
                    'classifier_backbones.py', 'scripts/eval_chamfer_depth.py',
                    'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py',
                    'scripts/analyze_legacy_trajectory.py')
POLICIES = ('terminal', 'minimum_entropy', 'probability_mean')


def _depth_validators():
    """Reuse accepted physical RNG verification without a global campaign collision."""
    name = '_chamfer_depth_20261030_analysis'
    if name not in sys.modules:
        prep = REPO / 'result/preparation/chamfer_depth_tuning_20261030'
        previous = sys.modules.get('campaign')
        original_path = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location('_chamfer_depth_20261030_campaign', prep / 'campaign.py')
            campaign = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(campaign)
            sys.modules['campaign'] = campaign
            spec = importlib.util.spec_from_file_location(name, prep / 'analyze.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
        finally:
            sys.path[:] = original_path
            if previous is None:
                sys.modules.pop('campaign', None)
            else:
                sys.modules['campaign'] = previous
    return sys.modules[name]


def _identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _subset(expected, actual, context='config'):
    require(isinstance(actual, dict), context + ' must be an object')
    for key, value in expected.items():
        require(key in actual, context + ' missing ' + key)
        if isinstance(value, dict):
            _subset(value, actual[key], context + '.' + key)
        else:
            require(value == actual[key], context + ' mismatch ' + key)


def _sources(bundle, cfg, required):
    require(isinstance(required, dict), 'Required source binding missing')
    for name in CRITICAL_SOURCES:
        require('assets/source_snapshot/' + name in required or 'repo/' + name in required,
                'Missing critical source binding: ' + name)
    _subset(required, cfg.get('executed_source_files'), 'executed sources')
    raw = bundle.read('source_snapshot.zip')
    require(_identity(raw)['sha256'] == cfg['source_snapshot_sha256'], 'Source snapshot bytes mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, 'Source snapshot duplicate/CRC mismatch')
        for name, expected in required.items():
            if name.startswith('assets/source_snapshot/'):
                value = archive.read(name[len('assets/'):])
            elif name.startswith('repo/'):
                value = bundle.read('source_snapshot/' + name[len('repo/'):])
            else:
                raise ValueError('Unsupported required source namespace: ' + name)
            require(_identity(value) == expected, 'Executed source bytes mismatch: ' + name)
    actual = hashlib.sha256(json.dumps(cfg['executed_source_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(actual == cfg.get('executed_source_sha256'), 'Executed source map digest mismatch')


def analyze_bundle(path: str | Path, expected_config: dict, cache_root: str | Path) -> dict:
    """Accept one full selected cohort or explicitly bound technical smoke; fail closed."""
    binding = expected_config
    expected = binding.get('expected_config', binding)
    for key in ('dataset', 'backbone', 'seed', 'selected_original_ids', 'source_snapshot_sha256',
                'asset_manifest_sha256', 'checkpoint_load'):
        require(key in expected, 'Expected binding missing ' + key)
    require(_is_sha(expected['source_snapshot_sha256']) and _is_sha(expected['asset_manifest_sha256']) and
            _is_sha(expected['checkpoint_load'].get('checkpoint_sha256')), 'Unpinned source/assets/checkpoint binding')
    bundle = Bundle(path)
    try:
        cfg = bundle.json('config.json')
        _subset({k: v for k, v in expected.items() if k != 'required_source_files'}, cfg)
        validate_transfer_config(cfg)
        smoke = cfg.get('technical_smoke', False)
        require(cfg.get('status') in (('partial', 'complete') if smoke else ('complete',)), 'Unavailable failed/incomplete endpoint')
        require(type(cfg.get('seed')) is int and cfg['seed'] in (0, 1, 2) and cfg.get('severity') == 5,
                'Expected fixed seeds0/1/2 and severity5')
        require(bool(cfg.get('reconstruction_group_id')), 'Shared reconstruction group missing')
        require(cfg.get('guidance_rate_mapping') == dict(gamma='local_latent', eta='conditioning_style'), 'Rate mapping changed')
        _subset(dict(style='original encoded shape_latent for every clean and terminal decode',
                     candidate_weighting='equal probability per unique clean checkpoint and terminal'), cfg['trajectory_selection'])
        selection = bundle.read('selection_manifest.json')
        require(json.loads(selection) == cfg.get('selection_manifest') and _identity(selection) == cfg.get('selection_manifest_identity'),
                'Selection manifest bytes/content mismatch')
        ids = cfg['selected_original_ids']
        require(_is_sha(cfg.get('classifier_state_before_sha256')) and
                cfg['classifier_state_before_sha256'] == cfg.get('classifier_state_after_sha256'), 'Frozen classifier state missing/changed')
        require(isinstance(cfg.get('lion_state_before'), dict) and set(cfg['lion_state_before']) == {'vae', 'priors'} and
                all(_is_sha(v) for v in cfg['lion_state_before'].values()) and cfg['lion_state_before'] == cfg.get('lion_state_after'),
                'Frozen LION state missing/changed')
        _subset(dict(strict=True, missing_keys=[], unexpected_keys=[]), cfg['checkpoint_load'], 'checkpoint')
        _sources(bundle, cfg, binding.get('required_source_files'))
        classes = cfg['num_classes']
        corruptions = cfg['corruptions']
        outcomes = {p: [] for p in POLICIES}
        rows, guards, caches = [], [], []
        used = set()
        label_reference = None
        for corruption in corruptions:
            depth, steps = (35, [9, 18, 27]) if corruption == 'background' else (8, [2, 4, 6])
            names = ['clean_%02d' % s for s in steps] + ['terminal']
            parts, labels = [], []
            for batch, start in enumerate(range(0, len(ids), 32)):
                prefix = 'batches/%s/batch%04d/' % (corruption, batch)
                member = prefix + 'predictions.npz'
                used.add(member)
                with np.load(io.BytesIO(bundle.read(member)), allow_pickle=False) as data:
                    values = {k: data[k].copy() for k in data.files}
                selected_ids = ids[start:start + 32]
                n = len(selected_ids)
                require(values['index'].dtype.kind in 'iu' and values['index'].tolist() == selected_ids, 'Original IDs/order/remainder mismatch')
                require(values['label'].shape == (n,) and values['label'].dtype.kind in 'iu' and
                        ((values['label'] >= 0) & (values['label'] < classes)).all(), 'Labels outside dataset class contract')
                require(values['candidate_names'].tolist() == names and values['candidate_loop_counts'].tolist() == steps + [depth], 'Candidate readout mismatch')
                logits = values['trajectory_logits']
                require(logits.shape == (n, 4, classes) and np.isfinite(logits).all(), 'Invalid/nonfinite candidate logits')
                require(values['logits'].shape == (n, classes) and np.array_equal(values['logits'], logits[:, -1]) and
                        values['prediction'].shape == (n,) and values['prediction'].dtype.kind in 'iu' and np.array_equal(values['prediction'], logits[:, -1].argmax(1)), 'Terminal logits/prediction mismatch')
                computed = policies(logits, 3)
                for key, result in computed.items():
                    require(key in values and values[key].shape == result.shape and
                            np.allclose(values[key], result, rtol=1e-10, atol=1e-12), 'Saved/recomputed readout mismatch: ' + key)
                d = bundle.json(prefix + 'diagnostics.json')
                t, scheduler = d['trajectory'], d['scheduler']
                require(t.get('complete') is True and t.get('horizon') == depth and t.get('candidate_names') == names and
                        t.get('candidate_loop_counts') == steps + [depth], 'Actual horizon/readout mismatch')
                _subset(dict(active_timesteps=list(range((depth - 1) * 10, -1, -10)), reverse_steps=depth,
                             initial_timestep=(depth - 1) * 10, config=dict(set_alpha_to_one=True), final_alpha_cumprod=1.), scheduler, 'actual scheduler')
                initial = t.get('initial_hashes', {})
                require(all(_is_sha(initial.get(k)) for k in ('input_points', 'shape_latent', 'local_latent', 'noise', 'noisy_local', 'style')) and
                        _is_sha(t.get('initial_rng_sha256')) and _is_sha(t.get('final_hashes', {}).get('latent')) and
                        t['final_hashes'].get('style') == initial['shape_latent'], 'Latent/RNG/original-style hashes missing/changed')
                probes = t.get('probes', [])
                require([p.get('loop_count') for p in probes] == steps, 'Missing/reordered probes')
                for s, probe in zip(steps, probes):
                    require(probe.get('timestep') == (depth - s) * 10 and
                            all(probe.get(k) is True for k in ('rng_unchanged', 'buffers_unchanged', 'live_inputs_unchanged')) and
                            _is_sha(probe.get('rng_before_sha256')) and probe['rng_before_sha256'] == probe.get('rng_after_sha256') and
                            probe.get('buffers_before') and probe['buffers_before'] == probe.get('buffers_after') and
                            probe.get('style_sha256') == initial['shape_latent'] and
                            all(_is_sha(probe.get(k)) for k in ('clean_latent_sha256', 'decoded_geometry_sha256')), 'Probe purity/style/RNG guard failed')
                require(d.get('rng_unchanged') is True and _is_sha(d.get('rng_before_sha256')) and
                        d['rng_before_sha256'] == d.get('rng_after_sha256') == d.get('reconstruction_rng_after_sha256') and
                        _is_sha(d.get('reconstruction_rng_before_sha256')) and
                        _is_sha(d.get('geometry_sha256')) and d.get('geometry_shape') == [n, 2048, 3], 'Terminal RNG/geometry guard failed')
                for key in ('executed_source_sha256', 'source_snapshot_sha256', 'asset_manifest_sha256', 'checkpoint_load'):
                    require(d.get(key) == cfg[key], 'Batch identity mismatch: ' + key)
                recorded = d['paired_state']
                remote_root = Path(cfg['paired_state_root'])
                require(remote_root.name == 'seed%d' % cfg['seed'] and remote_root.parent.name == ('paired-smoke' if smoke else 'paired') and
                        Path(recorded['path']).parent == remote_root / cfg['dataset'] / corruption,
                        'Paired cache outside fixed seed/dataset/corruption namespace')
                expected_name = hashlib.sha256(np.asarray(selected_ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
                require(Path(recorded['path']).name == expected_name, 'Paired cache ID-derived filename mismatch')
                relative = Path(remote_root.parent.name) / remote_root.name / cfg['dataset'] / corruption / Path(recorded['path']).name
                physical = Path(cache_root) / relative
                require(physical.is_file() and not physical.is_symlink(), 'Physical paired cache missing: ' + str(relative))
                _depth_validators().verify_cache(physical, recorded, initial, n)
                with np.load(physical, allow_pickle=False) as cache_values:
                    cached_rng = json.loads(str(cache_values['metadata'].item()))['rng_after_sha256']
                require(cached_rng == t['initial_rng_sha256'], 'Physical cache RNG differs from trajectory initial RNG')
                cache = dict(relative_path=relative.as_posix(), **{k: recorded[k] for k in ('bytes', 'sha256')})
                caches.append(cache)
                guards.append(dict(corruption=corruption, batch=batch, ids=selected_ids, initial_hashes=initial,
                                   final_hashes=t['final_hashes'], initial_rng_sha256=t['initial_rng_sha256'],
                                   geometry_sha256=d['geometry_sha256'], probes=probes, scheduler=scheduler, cache=cache))
                parts.append(logits)
                labels.append(values['label'])
            labels = np.concatenate(labels)
            if label_reference is None:
                label_reference = labels
            require(np.array_equal(labels, label_reference), 'Object labels vary across corruptions')
            selected = policies(np.concatenate(parts), 3)
            terminal = selected['terminal'] == labels
            for policy in POLICIES:
                correct = selected[policy] == labels
                outcomes[policy].append(correct.tolist())
                rows.append(dict(dataset=cfg['dataset'], backbone=cfg['backbone'], seed=cfg['seed'], corruption=corruption,
                                 policy=policy, n_objects=len(ids), accuracy_pct=float(correct.mean() * 100),
                                 delta_terminal_pp=float((correct.astype(float) - terminal).mean() * 100),
                                 corrected=int((correct & ~terminal).sum()), harmed=int((~correct & terminal).sum())))
        require(used == {n for n in bundle.names if n.startswith('batches/') and n.endswith('/predictions.npz')}, 'Undeclared prediction shard')
        return dict(status='accepted', input=str(path), dataset=cfg['dataset'], backbone=cfg['backbone'], seed=cfg['seed'],
                    technical_smoke=smoke, object_ids=ids, labels=label_reference.tolist(), corruptions=corruptions,
                    bundle_identity=_identity(Path(path).read_bytes()) if Path(path).is_file() else None,
                    sealed_manifest_identity=_identity(bundle.read('bundle_manifest.json')),
                    reconstruction_group_id=cfg.get('reconstruction_group_id'), source_snapshot_sha256=cfg['source_snapshot_sha256'],
                    asset_manifest_sha256=cfg['asset_manifest_sha256'], outcomes=outcomes, per_corruption=rows,
                    aggregate=[dict(policy=p, accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rows if r['policy'] == p]))) for p in POLICIES],
                    batch_guards=guards, cache_proofs=caches,
                    limitations=['state and probe guards are recorded evidence bound to sealed executed source bytes',
                                 'physical cache verified; intermediate/terminal geometry is hash-only, not replayed on CPU'])
    finally:
        bundle.close()


def analyze_reports(reports: list[dict], repetitions: int = 10000) -> dict:
    """Check four-head sharing and form separate per-head PM-versus-own-terminal CIs."""
    groups, sharing = {}, {}
    for report in reports:
        require(report.get('status') == 'accepted', 'Only accepted endpoints enter summaries')
        key = (report['dataset'], report['backbone'], report['technical_smoke'])
        seeds = groups.setdefault(key, {})
        require(report['seed'] not in seeds, 'Duplicate dataset/head/seed endpoint')
        seeds[report['seed']] = report
        pair = (report['dataset'], report['seed'], report['technical_smoke'])
        signature = {k: report[k] for k in ('object_ids', 'labels', 'corruptions', 'reconstruction_group_id',
                                           'source_snapshot_sha256', 'asset_manifest_sha256', 'batch_guards')}
        if pair in sharing:
            require(sharing[pair] == signature, 'Cross-head shared geometry/cache/source/ID guards differ')
        sharing[pair] = signature
    comparisons = []
    for (dataset, head, smoke), seeds in sorted(groups.items()):
        context = dict(dataset=dataset, backbone=head, technical_smoke=smoke)
        if smoke or set(seeds) != {0, 1, 2}:
            comparisons.append(dict(**context, status='unavailable', reason='complete fixed three-seed research endpoint required; no surviving-seed substitution'))
            continue
        first = seeds[0]
        require(all(all(r[k] == first[k] for k in ('object_ids', 'labels', 'corruptions', 'source_snapshot_sha256', 'asset_manifest_sha256')) for r in seeds.values()), 'Cross-seed ID/label/source mismatch')
        for scope in ('all15', 'macro14'):
            mask = [i for i, c in enumerate(first['corruptions']) if scope == 'all15' or c != 'background']
            a = np.asarray([seeds[s]['outcomes']['probability_mean'] for s in (0, 1, 2)])[:, mask]
            b = np.asarray([seeds[s]['outcomes']['terminal'] for s in (0, 1, 2)])[:, mask]
            for seed in (None, 0, 1, 2):
                comparisons.append(dict(**context, status='complete', scope=scope, seed=seed,
                    left_policy='probability_mean', right_policy='own_same_run_terminal',
                    **_depth_validators().compare(a if seed is None else a[seed:seed + 1], b if seed is None else b[seed:seed + 1], repetitions)))
    return dict(status='derived_CPU_analysis', bundles=reports, comparisons=comparisons,
                comparison_scope='per-head paired readout effects; conditional object-cluster intervals; no historical baseline or model ensemble')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, type=Path, help='JSON list of {path, binding} acceptance requests')
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    requests = json.loads(args.requests.read_text())
    reports, unavailable = [], []
    for request in requests:
        try:
            reports.append(analyze_bundle(request['path'], request['binding'], args.cache_root))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            unavailable.append(dict(path=request['path'], status='unavailable', reason=str(error)))
    result = analyze_reports(reports, args.bootstrap_repetitions)
    result['unavailable'] = unavailable
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    main()
