#!/usr/bin/env python3
"""Frozen Point-MAE confirmation of CloudFixer probe and correction hypotheses."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shlex
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import Budget, MountStore, canonical, record_setup, sha, unit_key
from scripts import run_cloudfixer_exploration as exploration
from scripts import run_cloudfixer_transfer as transfer

PRIOR_MANIFEST = 'b09f8b7730d1b153f9ac4633b20ead8d002e59b71dd4a9ef07e9376436aa7e38'
PRIOR_DENSE = '509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df'
LATE = {8: [20, 10, 0], 12: [27, 0]}
GEOMETRY = ('logsnr', 'correction_rms', 'pre_to_input_rms', 'post_to_input_rms',
            'state_to_input_rms', 'pre_to_previous_pre_rms')


NATIVE_BINDINGS = ('knn_cuda._knn', 'pointnet2_ops._ext', 'chamfer', 'pytorch3d._C')


def native_identity() -> dict:
    # The JIT KNN library lives on knn_cuda._knn and need not be in sys.modules itself.
    import chamfer
    from knn_cuda import _knn
    from pointnet2_ops import _ext
    from pytorch3d import _C
    identity = {}
    for name, module in zip(NATIVE_BINDINGS, (_knn, _ext, chamfer, _C)):
        path = Path(module.__file__).resolve(strict=True)
        with path.open('rb') as stream:
            if path.suffix != '.so' or stream.read(4) != b'\x7fELF':
                raise ValueError(f'Expected loaded ELF extension for {name}: {path}')
        identity[name] = dict(path=str(path), sha256=transfer.file_hash(path))
    return identity


def read_pilot_environment(read, digest: str, campaign: str, reg: dict, *, output=None) -> bytes:
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid environment SHA')
    environment_name = 'environment_'+digest+'.json'
    environment = read(environment_name)
    extensions = json.loads(environment).get('native_extensions', {})
    if (sha(environment) != digest or set(extensions) != set(NATIVE_BINDINGS)
            or any(not Path(v['path']).is_absolute() or not v['path'].endswith('.so')
                   or not re.fullmatch('[0-9a-f]{64}', v['sha256']) for v in extensions.values())):
        raise ValueError('Invalid native environment receipt')
    gate_name = 'pilot_gate_'+digest+'.json'
    data = read(gate_name)
    gate = json.loads(data)
    expected = dict(status='passed', campaign=campaign, seed=0, corruption=reg['pilot_corruption'],
                    historical_dense_sha256=reg['historical_dense_sha256'], environment_sha256=digest)
    size = reg['batch_size']
    pilot_ids = reg['pilot_ids']
    batches = gate.get('batches', [])
    if (any(gate.get(k) != v for k, v in expected.items())
            or [b['ids'] for b in batches] != [pilot_ids[i:i+size] for i in range(0, len(pilot_ids), size)]):
        raise ValueError('Environment historical pilot gate mismatch')
    for batch in batches:
        old = batch['old_canonical']
        production = batch['production']
        checks = (old, batch['old_dense'], production['sparse'], production['terminal'])
        if (old.get('status') != 'passed' or old.get('old_archive_sha256') != reg['historical_development_sha256']
                or any(check.get('argmax_changes') != 0 or check.get('pm_argmax_changes') != 0 for check in checks)):
            raise ValueError('Environment historical pilot gate lacks prediction parity')
    if output is not None:
        (output/environment_name).write_bytes(environment)
        (output/gate_name).write_bytes(data)
    return environment


def registration() -> dict:
    old = transfer.registration()
    return dict(schema=1, role='cloudfixer_pointmae_hypotheses_b', dataset=old['dataset'],
        classifier='Point-MAE', severity=5, seeds=[0, 1, 2], batch_size=32,
        corruptions=old['corruptions'], confirmation_ids=old['confirmation_ids'],
        pilot_ids=old['development_ids'], pilot_corruption=old['corruptions'][0],
        schedules=[dict(method='proximal', steps=n, start=80, tau=tau, retained=.95)
                   for n, tau in ((8, .1), (12, .1), (12, .8/12))],
        late_native_indices={str(n): ts for n, ts in LATE.items()},
        windows=['canonical', 'all', 'late'], entropy_beta=[0, .5, 1, 2, 4, 8],
        entropy_linear='1-H/log40 with equal fallback', confidence_threshold=.9,
        geometry='per-object RMS Euclidean displacement in input point order; previous pre at first probe is observed input',
        observation='pre, post, pre-extra-t0 noisy state and true t0; head observations after transport with common RNG',
        primary='equal late PM vs own true terminal; object-cluster paired 95% bootstrap, conditional on 3 seeds',
        bootstrap_seed=20261118, bootstrap_draws=10000, budget_seconds=28800,
        historical_development_sha256=exploration.PRIOR_ARCHIVE_SHA, historical_dense_sha256=PRIOR_DENSE)


def units(campaign: str, reg: dict) -> list:
    return [dict(campaign=campaign, phase='hypothesis_confirmation', config=config,
                 config_id='proximal_'+sha(canonical(config))[:12], seed=seed,
                 corruption=c, ids=reg['confirmation_ids'][j:j+reg['batch_size']])
            for c in reg['corruptions'] for seed in reg['seeds']
            for j in range(0, len(reg['confirmation_ids']), reg['batch_size'])
            for config in reg['schedules']]


def probe_evaluator(model, classifier, x, mask, noise, config):
    """Finish transport before classifying any pre/post XYZ. Each view uses the same head RNG."""
    import torch
    from cloudfixer_diffusion import sample

    mode = config.get('observation', 'dense')
    if mode not in ('dense', 'sparse', 'terminal'):
        raise ValueError('Unknown observation mode')
    diagnostics = {name: [] for name in GEOMETRY}
    previous = x
    post = []
    endpoint = []
    def rms(difference):
        return difference.square().sum(-1).mean(-1).sqrt()
    def observe(index, state, clean, corrected, alpha, sigma):
        nonlocal previous
        diagnostics['logsnr'].append((2*(alpha.reshape(len(x)).log()-sigma.reshape(len(x)).log())))
        diagnostics['correction_rms'].append(rms(corrected-clean))
        diagnostics['pre_to_input_rms'].append(rms(clean-x))
        diagnostics['post_to_input_rms'].append(rms(corrected-x))
        diagnostics['state_to_input_rms'].append(rms(state-x))
        diagnostics['pre_to_previous_pre_rms'].append(rms(clean-previous))
        previous = clean
        if index == 0:
            endpoint.append(state)
        else:
            post.append(corrected)

    params = {k: config[k] for k in ('steps', 'start', 'tau', 'retained')}
    probes = None if mode == 'dense' else (LATE[config['steps']] if mode == 'sparse' else [0])
    fingerprint = config.get('pilot', False)
    with torch.no_grad():
        candidates, meta = sample(model, x, mask, noise, **params,
            probe_indices=probes, trace_all=mode=='dense',
            observe=observe if mode=='dense' else None, fingerprint=fingerprint)
        py_rng, np_rng = random.getstate(), np.random.get_state()
        devices = [x.device.index] if x.is_cuda else []
        def rng_hash():
            import pickle
            h = torch.get_rng_state().numpy().tobytes()
            if x.is_cuda:
                h += torch.cuda.get_rng_state(x.device).cpu().numpy().tobytes()
            return sha(h+pickle.dumps((random.getstate(), np.random.get_state()), protocol=4))
        if fingerprint:
            meta['pre_head_rng_sha256'] = rng_hash()
            meta['terminal_xyz_sha256'] = sha(candidates[:, -1].contiguous().cpu().numpy().tobytes())
        def classify(cloud):
            try:
                with torch.random.fork_rng(devices=devices):
                    return classifier(cloud.contiguous())
            finally:
                random.setstate(py_rng)
                np.random.set_state(np_rng)
        logits = torch.stack([classify(cloud) for cloud in candidates.unbind(1)], 1)
        if mode == 'dense':
            # Post observations omit terminal: there is no proximal update at index zero.
            meta['post_logits'] = torch.stack([classify(cloud) for cloud in post], 1).cpu().tolist()
            meta['pre_extra_t0_logits'] = classify(endpoint[0]).cpu().tolist()
            meta['pre_extra_t0_to_terminal_rms'] = rms(endpoint[0]-candidates[:, -1]).cpu().tolist()
            for name, values in diagnostics.items():
                meta[name] = torch.stack(values, 1).cpu().tolist()
        if fingerprint:
            meta['post_head_rng_sha256'] = rng_hash()
            if meta['post_head_rng_sha256'] != meta['pre_head_rng_sha256']:
                raise ValueError('Observing classifiers changed RNG')
        return logits, meta


def add_diagnostics(data: bytes, source_sha: str, environment_sha: str) -> bytes:
    p = exploration.unpack(data)
    meta = json.loads(str(p['sampler']))
    for name in (*GEOMETRY, 'pre_extra_t0_to_terminal_rms', 'post_logits', 'pre_extra_t0_logits'):
        p[name] = np.asarray(meta.pop(name), dtype=np.float32)
    p.update(exploration.probability_stats(p['logits']))
    for prefix in ('post_', 'pre_extra_t0_'):
        stats = exploration.probability_stats(p[prefix+'logits'])
        for field in ('probabilities', 'entropy'):
            p[prefix+field] = stats[field]
    p.update(sampler=json.dumps(meta, sort_keys=True), source_sha256=source_sha, environment_sha256=environment_sha)
    return exploration.pack(p)


def validate(data: bytes, unit: dict, labels, manifest: dict, environment: bytes) -> dict:
    p = exploration.unpack(data)
    n, steps = len(unit['ids']), unit['config']['steps']
    meta = json.loads(str(p['sampler']))
    times = [round(80*(steps-i)/steps) for i in range(steps+1)]
    if (json.loads(str(p['unit'])) != unit or not np.array_equal(p['ids'], unit['ids'])
            or p['ids'].dtype.kind not in 'iu' or p['labels'].dtype.kind not in 'iu'
            or p['labels'].shape != (n,) or not np.array_equal(p['labels'], labels[unit['ids']])
            or meta['candidate_indices'] != times or meta['indices'] != times
            or meta['candidate_loops'] != list(range(1, steps+1))+[steps]
            or int(p['denoiser_calls']) != steps+1 or meta['denoiser_calls'] != steps+1
            or not re.fullmatch('[0-9a-f]{64}', meta['input_rng_sha256'])
            or not np.isfinite([p['seconds'], p['peak_mb']]).all() or min(p['seconds'], p['peak_mb']) <= 0):
        raise ValueError('Invalid confirmation unit identity, timing or trajectory')
    exploration.validate_provenance(p, manifest, environment)
    if json.loads(environment)['classifier'].find('Point-MAE') == -1:
        raise ValueError('Confirmation requires the frozen author Point-MAE')
    shape, counts = p['input_shape'].tolist(), meta['valid_reference_counts']
    if (len(shape) != 3 or shape[0] != n or shape[2] != 3 or len(counts) != n
            or any(type(c) is not int or not 1 <= c <= shape[1] for c in counts)
            or meta['retained_forward_count'] != int(shape[1]*unit['config']['retained'])
            or meta['retained_reference_counts'] != [int(c*unit['config']['retained']) for c in counts]):
        raise ValueError('Anchor mask or retained counts differ')
    for prefix, shape in (('', (n, steps+1, 40)), ('post_', (n, steps, 40)), ('pre_extra_t0_', (n, 40))):
        logits = p[prefix+'logits']
        if logits.shape != shape or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError('Invalid logits for '+prefix)
        stats = exploration.probability_stats(logits)
        for name in (('probabilities', 'entropy') if prefix else stats):
            saved = p[prefix+name]
            if (saved.dtype != np.float64 or saved.shape != stats[name].shape
                    or not np.allclose(stats[name], saved, rtol=1e-12, atol=1e-12)):
                raise ValueError('Saved softmax/entropy differs from raw logits')
    for name in (*GEOMETRY, 'pre_extra_t0_to_terminal_rms'):
        expected = (n,) if name == 'pre_extra_t0_to_terminal_rms' else (n, steps+1)
        if (p[name].shape != expected or not np.isfinite(p[name]).all()
                or (name != 'logsnr' and (p[name] < 0).any())):
            raise ValueError('Invalid geometry '+name)
    if np.any(p['correction_rms'][:, -1]) or not (np.diff(p['logsnr'], axis=1)>0).all():
        raise ValueError('Unexpected t0 correction or nonmonotone logSNR')
    return p


def check_readout(actual, expected) -> dict:
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, atol=1e-4, rtol=1e-5)
            or np.any(actual.argmax(-1) != expected.argmax(-1))):
        raise ValueError('Same-trace classifier logit parity failed')
    a = exploration.probability_stats(actual)['probabilities'].mean(1)
    b = exploration.probability_stats(expected)['probabilities'].mean(1)
    if np.any(a.argmax(-1) != b.argmax(-1)):
        raise ValueError('Paired PM prediction parity failed')
    return dict(max_abs=float(np.abs(actual-expected).max()), pm_max_abs=float(np.abs(a-b).max()),
                argmax_changes=0, pm_argmax_changes=0)


def pilot(args, reg, manifest, campaign, infer, verify, store, env_sha, budget):
    pilot_out = args.output/'pilot'
    pilot_out.mkdir()
    prior = exploration.PriorArchive(args.prior_archive, manifest)
    if transfer.file_hash(args.prior_dense_archive) != PRIOR_DENSE:
        raise ValueError('Historical dense ZIP identity differs')
    dense_records = {}
    with zipfile.ZipFile(args.prior_dense_archive) as archive:
        if archive.testzip() or len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('Damaged historical dense ZIP')
        for name in archive.namelist():
            if '/pending_unit_' in name and name.endswith('.npz'):
                p = exploration.unpack(archive.read(name))
                u = json.loads(str(p['unit']))
                if u['corruption'] == reg['pilot_corruption']:
                    dense_records[tuple(u['ids'])] = p
    rows = []
    totals = {mode:dict(n=0,correct=0,seconds=0.,peak=0.) for mode in ('dense','sparse','terminal')}
    config = reg['schedules'][0]
    for j in range(0, len(reg['pilot_ids']), 32):
        ids = reg['pilot_ids'][j:j+32]
        base = dict(campaign=campaign, phase='pilot', config=config, config_id='proximal_'+sha(canonical(config))[:12],
                    corruption=reg['pilot_corruption'], seed=0, ids=ids)
        outputs = {}
        with budget.operation(600):
            for mode in ('dense', 'sparse', 'terminal'):
                u = dict(base, config=dict(config, observation=mode, pilot=True))
                receipt_name = 'pilot_'+env_sha+'_'+unit_key(u)
                if receipt_name in store.keys(receipt_name):
                    receipt = json.loads(store.get(receipt_name))
                    data = store.get('pilot_'+receipt['sha256']+'.npz')
                    if receipt['unit'] != u or sha(data) != receipt['sha256']:
                        raise ValueError('Restored pilot receipt differs')
                else:
                    data = infer(u)
                    store.put('pilot_'+sha(data)+'.npz', data)
                    store.put(receipt_name, canonical(dict(unit=u, sha256=sha(data))))
                    (args.output/('pending_'+unit_key(u)+'.npz')).unlink()
                (pilot_out/('pilot_'+unit_key(u)+'.npz')).write_bytes(data)
                outputs[mode] = exploration.unpack(data)
                if json.loads(str(outputs[mode]['unit'])) != u:
                    raise ValueError('Pilot payload unit mismatch')
            dense = outputs['dense']
            historical = dense_records[tuple(ids)]
            old_canonical = prior.check(dense, base)
            for field in ('labels', 'ids', 'model_hashes', 'input_shape'):
                if any(not np.array_equal(dense[field], p[field]) for p in (*outputs.values(), historical)):
                    raise ValueError('Historical or technical pilot identity differs: '+field)
            metadata = {k: json.loads(str(v['sampler'])) for k, v in outputs.items()}
            if json.loads(str(historical['sampler']))['input_rng_sha256'] != metadata['dense']['input_rng_sha256']:
                raise ValueError('Historical dense input/RNG differs')
            identity = {}
            for field in ('input_rng_sha256', 'denoiser_input_trajectory_sha256', 'terminal_xyz_sha256',
                          'pre_head_rng_sha256', 'post_head_rng_sha256'):
                digest = metadata['dense'][field]
                if not re.fullmatch('[0-9a-f]{64}', digest) or any(m[field] != digest for m in metadata.values()):
                    raise ValueError('Observation changed pilot '+field)
                identity[field] = digest
            times = metadata['dense']['candidate_indices']
            sparsity = {}
            for mode in ('sparse', 'terminal'):
                slots = [times.index(t) for t in metadata[mode]['candidate_indices']]
                sparsity[mode] = check_readout(outputs[mode]['logits'], dense['logits'][:, slots])
            rows.append(dict(ids=ids, identity=identity, old_canonical=old_canonical,
                old_dense=check_readout(dense['logits'], historical['logits']), production=sparsity))
            for mode,payload in outputs.items():
                terminal = payload['logits'][:,-1].argmax(-1)
                counts = totals[mode]
                counts['n'] += len(ids)
                counts['correct'] += int((terminal==payload['labels']).sum())
                counts['seconds'] += float(payload['seconds'])
                counts['peak'] = max(counts['peak'],float(payload['peak_mb']))
            verify()
    prior.archive.close()
    report_dir = args.output if args.smoke_only else pilot_out
    for file in ('summary','per_corruption'):
        pilot_rows = []
        for mode,counts in totals.items():
            accuracy = counts['correct']/counts['n']
            item = dict(run_id=campaign,dataset=reg['dataset'],severity=5,method='pilot_'+mode+'_true_terminal',
                        seed=0,status='technical_pilot')
            if file=='summary':
                item.update(n_corruptions=1,macro_accuracy=accuracy,total_examples=counts['n'],
                    total_correct=counts['correct'],micro_accuracy=accuracy,total_runtime_seconds=counts['seconds'])
            else:
                item.update(corruption=reg['pilot_corruption'],n_examples=counts['n'],n_correct=counts['correct'],
                    accuracy=accuracy,runtime_seconds=counts['seconds'],peak_gpu_memory_mb=counts['peak'])
            pilot_rows.append(item)
        with (report_dir/(file+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pilot_rows[0]))
            writer.writeheader()
            writer.writerows(pilot_rows)
    gate = dict(status='passed', campaign=campaign, corruption=reg['pilot_corruption'], seed=0,
                historical_dense_sha256=PRIOR_DENSE, environment_sha256=env_sha, batches=rows)
    store.put('pilot_gate_'+env_sha+'.json', canonical(gate))
    (args.output/'pilot_gate.json').write_bytes(canonical(gate))


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--register', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prior-archive', type=Path)
    parser.add_argument('--prior-dense-archive', type=Path)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--drive-folder', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--setup-receipt', type=Path)
    parser.add_argument('--official-root', default='/content/CloudFixer')
    parser.add_argument('--dataset-root', default='/content/3DD-TTA/data/modelnet40_c')
    parser.add_argument('--classifier', default='/content/CloudFixer/checkpoints/pointMAE_modelnet40.pth')
    parser.add_argument('--diffusion', default='/content/CloudFixer/checkpoints/diffusion_model_transformer_modelnet40.npy')
    args = parser.parse_args(argv)
    reg = registration()
    if args.register:
        args.register.write_bytes(canonical(reg))
        return
    if not args.manifest:
        parser.error('--manifest is required')
    manifest = json.loads(args.manifest.read_text())
    if (manifest['registration'] != reg or manifest['prior_manifest_sha256'] != PRIOR_MANIFEST
            or 'scripts/run_cloudfixer_hypotheses.py' not in manifest['overlay_files']):
        raise ValueError('B must have its own preregistered source and historical model identity')
    identity = dict(manifest=manifest, role=reg['role'])
    campaign = sha(canonical(identity))
    wanted = units(campaign, reg)
    if args.preview:
        print(json.dumps(dict(campaign=campaign, units=len(wanted), registration=reg), indent=2))
        return
    if not all((args.output, args.drive_folder, args.prior_archive, args.prior_dense_archive, args.setup_receipt)):
        parser.error('Execution requires output, Drive folder, both prior archives and setup receipt')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.prior_archive = args.prior_archive.resolve()
    args.prior_dense_archive = args.prior_dense_archive.resolve()
    if not Path('/content/drive').is_mount() or Path('/content/drive/MyDrive/thesis/result').resolve() not in args.drive_folder.resolve().parents:
        raise ValueError('Expected mounted task folder under MyDrive/thesis/result')
    budget_store = MountStore(args.drive_folder.resolve())
    budget_store.put('task.json', canonical(manifest))
    record_setup(budget_store, json.loads(args.setup_receipt.read_text()), cap=reg['budget_seconds'])
    budget = Budget(budget_store, cap=reg['budget_seconds'], started=started)
    store = MountStore(budget_store.folder/('b_'+campaign[:12]))
    store.put('campaign.json', canonical(identity))
    source_sha = sha(canonical(manifest))
    (args.output/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    (args.output/'notes.md').write_text(
        'Point-MAE severity5 confirmation on 256 fixed objects, all15 corruptions, seeds0/1/2. '
        'Native pre/post clean estimates share transport and a common head RNG. The pre-extra-t0 state still contains noise. '
        'Runtime includes all diagnostic calls. Intervals condition on three seeds and cluster across corruptions by original ID.\n')
    def status(value):
        (args.output/'config.json').write_bytes(canonical(dict(identity, campaign=campaign, execution_status=value)))
    status('running')
    try:
        with budget.operation(1800):
            label_path = Path(args.dataset_root)/'label.npy'
            if transfer.file_hash(label_path) != manifest['dataset_files']['label.npy']:
                raise ValueError('B original label SHA mismatch')
            labels = np.load(label_path, allow_pickle=False).reshape(-1)
            raw, verify, env = transfer.runtime(args, manifest, proximal_evaluator=probe_evaluator)
            env['native_extensions'] = native_identity()
            environment = canonical(env)
            env_sha = sha(environment)
            store.put('environment_'+env_sha+'.json', environment)
            (args.output/'environment.txt').write_bytes(environment)
        pilot(args, reg, manifest, campaign, raw, verify, store, env_sha, budget)
        environments = {env_sha: read_pilot_environment(store.get, env_sha, campaign, reg, output=args.output)}
        if args.smoke_only:
            status('pilot_complete')
            print('CF_HYPOTHESES_DONE pilot', flush=True)
            return
        def check(data, unit):
            # Resolve each prior environment on resume; model/head identity remains pinned.
            p = exploration.unpack(data)
            digest = str(p['environment_sha256'])
            if digest not in environments:
                environments[digest] = read_pilot_environment(store.get, digest, campaign, reg, output=args.output)
            return validate(data, unit, labels, manifest, environments[digest])
        def infer(unit):
            data = add_diagnostics(raw(unit), source_sha, env_sha)
            (args.output/('pending_'+unit_key(unit)+'.npz')).write_bytes(data)
            return data
        def progress(value):
            print('CF_HYPOTHESES_PROGRESS', json.dumps(value), flush=True)
            (args.output/'progress.json').write_bytes(canonical(value))
        records = exploration.execute_pending(store, campaign, wanted, infer, cap=reg['budget_seconds'],
                                              budget=budget, validate=check, progress=progress)
        with budget.operation(1800):
            verify()
            for record in records:
                key = unit_key(record['unit'])
                (args.output/('pending_'+key+'.npz')).write_bytes(record['data'])
                receipt = store.get(key)
                if json.loads(receipt)['sha256'] != sha(record['data']):
                    raise ValueError('Final unit SHA receipt mismatch')
                (args.output/key).write_bytes(receipt)
            (args.output/'verified_label.npy').write_bytes(label_path.read_bytes())
            from scripts.analyze_cloudfixer_hypotheses import write_reports
            write_reports(args.output, records, wanted, reg, campaign)
            status('complete')
            for name in ('summary.csv','per_corruption.csv','paired.csv','diagnostics.json','config.json','notes.md'):
                store.put('report_'+args.output.name+'_'+name, (args.output/name).read_bytes())
        print('CF_HYPOTHESES_DONE confirmation', flush=True)
    except BaseException:
        status('failed')
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
