#!/usr/bin/env python3
"""Resumable XYZ diffusion search around copied, pinned official CloudFixer code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import canonical, sha, unit_key, commit_unit, load_units, Budget, record_setup
from research_artifacts import CORRUPTIONS


def registration() -> dict:
    ids = list(range(2468))
    random.Random(20261114).shuffle(ids)
    return dict(schema=1, dataset='modelnet40_c', severity=5, selection_seed=20261114,
        smoke_ids=sorted(ids[:8]), development_ids=sorted(ids[8:136]),
        confirmation_ids=sorted(ids[136:392]), corruptions=list(CORRUPTIONS),
        batch_size=32, smoke_batch_size=2, retained=.95, budget_seconds=86400,
        native_reference_repeats=3,repeat_policy="strict_source_proximal; original_CF_native_variance",
        exposure='Prospective object-disjoint subsets of an already exposed official test benchmark')


def conditions(phase: str) -> list:
    values = [dict(method='source')] + [dict(method='cloudfixer',operational_repeat=r)
        for r in range(1 if phase=='profile' else 3)]
    if phase == 'profile':
        combinations = [(8, 80, 0), (8, 80, .1)]
    elif phase == 'smoke':
        combinations = [(4, 40, 0), (8, 80, .1)]
    else:
        combinations = itertools.product((4, 8, 16, 32), (40, 80, 160), (0, .03, .1, .3))
    return values + [dict(method='proximal', steps=n, start=k, tau=tau, retained=.95)
                     for n, k, tau in combinations]


def seed_for(unit: dict) -> int:
    # Trial-independent draws: restoring or reordering trials cannot shift RNG state.
    return int(sha(canonical({k: unit[k] for k in ('seed', 'corruption', 'ids')}))[:8], 16)


def units(campaign: str, reg: dict, phase: str, selected=None) -> list:
    ids = reg['development_ids'][:32] if phase == 'profile' else reg[phase + '_ids']
    size = reg['smoke_batch_size'] if phase == 'smoke' else reg['batch_size']
    configs = conditions(phase) if selected is None else [dict(method='source'), dict(method='cloudfixer'), selected]
    result = []
    for corruption in reg['corruptions']:
        for seed in ((0, 1, 2) if phase == 'confirmation' else (0,)):
            for start in range(0, len(ids), size):
                for config in configs:
                    result.append(dict(campaign=campaign, phase=phase, config=config,
                        config_id=config['method']+'_'+sha(canonical(config))[:12],
                        corruption=corruption, seed=seed, ids=ids[start:start+size]))
    return result


def validate_payload(data: bytes, unit: dict, labels=None) -> dict:
    import numpy as np
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if (len(z.namelist()) != len(set(z.namelist()))
                or sum(i.file_size for i in z.infolist()) > 100_000_000):
            raise ValueError('Unsafe prediction archive')
    with np.load(io.BytesIO(data), allow_pickle=False) as p:
        n = len(unit['ids'])
        k = 4 if unit['config']['method'] == 'proximal' else 1
        expected_calls = unit['config']['steps']+1 if k == 4 else 30 if unit['config']['method'] == 'cloudfixer' else 0
        if (json.loads(str(p['unit'])) != unit or not np.array_equal(p['ids'], unit['ids'])
                or p['logits'].shape != (n, k, 40) or not np.isfinite(p['logits']).all()
                or p['labels'].shape != (n,) or p['labels'].dtype.kind not in 'iu'
                or not ((p['labels'] >= 0) & (p['labels'] < 40)).all()
                or int(p['denoiser_calls']) != expected_calls
                or not math.isfinite(float(p['seconds'])) or float(p['seconds']) <= 0
                or not math.isfinite(float(p['peak_mb'])) or float(p['peak_mb']) < 0):
            raise ValueError('Prediction unit identity, shapes, counts or timing mismatch')
        if labels is not None and not np.array_equal(p['labels'], labels[unit['ids']]):
            raise ValueError('Prediction labels differ from the pinned original IDs')
        if k==4:
            shape=p['input_shape'].tolist();meta=json.loads(str(p['sampler']))
            counts=meta.get('valid_reference_counts',[])
            if (len(shape)!=3 or shape[0]!=n or shape[2]!=3 or shape[1]<1 or len(counts)!=n
                    or any(type(c) is not int or not 1<=c<=shape[1] for c in counts)
                    or meta.get('retained_forward_count')!=int(shape[1]*unit['config']['retained'])
                    or meta.get('retained_reference_counts')!=[int(c*unit['config']['retained']) for c in counts]):
                raise ValueError('Invalid valid-anchor or retained-count diagnostics')
        return {key: p[key].copy() for key in p.files}


def execute_pending(store, campaign: str, wanted: list, infer, *, cap: float, labels=None,
                    progress=None, budget_store=None, accepted=None, budget=None) -> list:
    budget=budget or Budget(store if budget_store is None else budget_store,cap=cap)
    with budget.operation(1800):
        accepted = load_units(store, campaign) if accepted is None else accepted
        for record in accepted.values():validate_payload(record['data'],record['unit'],labels)
    expected = {unit_key(u): u for u in wanted}
    if len(expected) != len(wanted) or not set(accepted) <= set(expected):
        raise ValueError('Duplicate requested units or incompatible committed coverage')
    for unit in wanted:
        key = unit_key(unit)
        if key in accepted:
            continue
        started = time.monotonic()
        with budget.operation(300):
            data = infer(unit)
            validate_payload(data, unit, labels)
            commit_unit(store, unit, data)
            accepted[key] = dict(unit=unit, data=data)
            if unit['phase']=='profile':
                store.put('wall_'+sha(canonical(unit))+'.json',canonical(dict(unit_key=key,
                    wall_seconds=time.monotonic()-started)))
        elapsed = time.monotonic()-started
        if progress:
            progress(dict(completed=len(accepted), total=len(wanted), last=unit,
                          spent_seconds=cap-budget.remaining(), cap_seconds=cap, wall_seconds=elapsed))
    return list(accepted.values())


def execute_smoke(store, reopen, campaign: str, wanted: list, infer, **kwargs) -> list:
    if not store.keys('unit_'):
        first=next(u for u in wanted if u['config']['method']=='proximal')
        execute_pending(store,campaign,[first],infer,**kwargs)
    # Reopen even on an already-completed or partially-completed attempt.
    print('CF_TRANSFER_FRESH_MOUNT_STORE_RESUME',flush=True)
    return execute_pending(reopen(),campaign,wanted,infer,**kwargs)


def summarize(records: list, wanted: list) -> list:
    import numpy as np
    expected = {}
    for unit in wanted:
        expected.setdefault((unit['config_id'], unit['seed']), set()).add(unit_key(unit))
    groups = {}
    for record in records:
        u = record['unit']
        p = validate_payload(record['data'], u)
        logits = p['logits'].astype(np.float64)
        exp = np.exp(logits-logits.max(-1, keepdims=True))
        probs = (exp/exp.sum(-1, keepdims=True)).mean(1)
        for readout, prediction in (('pm', probs.argmax(-1)), ('terminal', logits[:, -1].argmax(-1))):
            key = (u['config_id'], u['seed'], readout)
            g = groups.setdefault(key, dict(config=u['config'], config_id=u['config_id'],
                seed=u['seed'], readout=readout, n_examples=0, n_correct=0, runtime_seconds=0.,
                peak_gpu_memory_mb=0., corruptions={}, seen=set(), keys=set()))
            ids = {(u['corruption'], i) for i in u['ids']}
            if ids & g['seen']:
                raise ValueError('Duplicate original-object coverage')
            g['seen'].update(ids)
            g['keys'].add(unit_key(u))
            n, correct = len(prediction), int((prediction == p['labels']).sum())
            g['n_examples'] += n
            g['n_correct'] += correct
            g['runtime_seconds'] += float(p['seconds'])
            g['peak_gpu_memory_mb'] = max(g['peak_gpu_memory_mb'], float(p['peak_mb']))
            c = g['corruptions'].setdefault(u['corruption'], dict(n_examples=0, n_correct=0, runtime_seconds=0.))
            c['n_examples'] += n
            c['n_correct'] += correct
            c['runtime_seconds'] += float(p['seconds'])
    for g in groups.values():
        g['macro_accuracy'] = float(np.mean([c['n_correct']/c['n_examples'] for c in g['corruptions'].values()]))
        g['status'] = 'complete' if g.pop('keys') == expected[(g['config_id'], g['seed'])] else 'partial'
        g.pop('seen')
    return list(groups.values())


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_gate(store, key: str, task: str) -> dict:
    if key not in store.keys(key):
        raise ValueError('Required technical gate has not passed: '+key)
    gate=json.loads(store.get(key))
    if gate.get('status')!='passed' or gate.get('task')!=task:
        raise ValueError('Technical gate is failed or belongs to another source/assets identity')
    return gate


def project_grid(profile: list, reg: dict, *, cloud_overhead: float, pending=None) -> float:
    measured={}
    for record in profile:
        unit=record['unit'];config=unit['config']
        family=(config['method'],config.get('tau',0)>0)
        measured[(unit['corruption'],family)]=float(record['seconds'])
    total=0.
    pending=units('projection',reg,'development') if pending is None else pending
    for unit in pending:
        config=unit['config']
        family=(config['method'],config.get('tau',0)>0)
        scale=max(1.,(config['steps']+1)/9) if config['method']=='proximal' else 1.
        total+=measured[(unit['corruption'],family)]*scale+cloud_overhead
    return total*1.25


def profile_gate(budget_store, store, task: str, records: list, reg: dict) -> dict:
    if 'profile_gate.json' in budget_store.keys('profile_gate.json'):
        return require_gate(budget_store,'profile_gate.json',task)
    prof=[dict(unit=r['unit'],seconds=float(validate_payload(r['data'],r['unit'])['seconds'])) for r in records]
    measured={unit_key(r['unit']):r['seconds'] for r in prof}
    overhead=[]
    for key in store.keys('wall_'):
        value=json.loads(store.get(key))
        if value['unit_key'] not in measured or not math.isfinite(value['wall_seconds']) or value['wall_seconds']<0:
            raise ValueError('Invalid persisted profile duration evidence')
        overhead.append(max(0.,value['wall_seconds']-measured[value['unit_key']]))
    if not overhead:
        raise ValueError('Profile cloud overhead is unmeasured; cannot approve the grid')
    median=statistics.median(overhead)
    gate=dict(status='passed',task=task,cloud_overhead_seconds=median,profile=prof,
        projected_grid_seconds=project_grid(prof,reg,cloud_overhead=median),completed_units=len(records))
    budget_store.put('profile_gate.json',canonical(gate))
    return gate


def check_repeat(first, repeated, method, input_identity, repeat_identity):
    import torch
    if first.shape!=repeated.shape or not torch.isfinite(first).all() or not torch.isfinite(repeated).all():
        raise FloatingPointError('Invalid repeat logits')
    if input_identity!=repeat_identity:raise ValueError('Repeated input/RNG identity changed')
    exact=bool(torch.allclose(first,repeated,atol=1e-4,rtol=1e-5))
    changes=int((first.argmax(-1)!=repeated.argmax(-1)).sum())
    if method!='cloudfixer' and (not exact or changes):
        raise ValueError('Same-input/RNG deterministic repeat gate failed')
    return dict(mode='original_native_variation' if method=='cloudfixer' else 'strict_deterministic',
                max_abs=float((first-repeated).abs().max()),argmax_changes=changes,within_tolerance=exact)


def verify_runtime_files(args, manifest):
    official = Path(args.official_root).resolve()
    for name, digest in manifest['official_files'].items():
        if file_hash(official/name) != digest:
            raise ValueError('Official source differs: ' + name)
    for name, digest in manifest['overlay_files'].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError('Executed overlay differs: ' + name)
    for name, digest in manifest['dataset_files'].items():
        if file_hash(Path(args.dataset_root)/name) != digest:
            raise ValueError('Dataset differs: ' + name)
    if file_hash(args.classifier) != manifest['classifier_sha256'] or file_hash(args.diffusion) != manifest['diffusion_sha256']:
        raise ValueError('Checkpoint differs from the pinned author assets')
    return official


def runtime(args, manifest: dict, *, trace_all: bool = False, proximal_evaluator=None):
    """Load the COPIED author entrypoints; no model/data/classifier reimplementation."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import numpy as np
    import torch
    from types import SimpleNamespace
    from cloudfixer_diffusion import sample, verify_gamma
    official = verify_runtime_files(args, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA is required for author-model execution')
    os.chdir(official)
    sys.path.insert(0, str(official))
    import adapt
    from classifier.pointMAE import build_model_from_cfg
    from utils.config import cfg_from_yaml_file
    from torch.utils.data._utils.collate import default_collate
    arguments = SimpleNamespace(model='transformer', probabilistic_model='diffusion', diffusion_steps=500,
        diffusion_noise_schedule='polynomial_2', diffusion_noise_precision=1e-5, diffusion_loss_type='l2')
    model = adapt.get_model(arguments, torch.device('cuda'))
    intended = model.gamma.gamma.detach().clone()
    model.load_state_dict(torch.load(args.diffusion, map_location='cpu'), strict=True)
    gamma_validation=verify_gamma(model.gamma.gamma,intended)
    model = model.cuda().eval().requires_grad_(False)
    classifier = build_model_from_cfg(cfg_from_yaml_file('cfgs/cfgs_mate/tta/tta_modelnet.yaml').model)
    ckpt = torch.load(args.classifier, map_location='cpu')['base_model']
    classifier.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}, strict=True)
    classifier.rotate = True
    classifier = classifier.cuda().eval().requires_grad_(False)
    argv = sys.argv
    try:
        sys.argv = ['adapt.py', '--mode', 'eval', '--method', 'cloudfixer', '--classifier', 'pointMAE',
            '--classifier_dir', args.classifier, '--dataset_dir', args.dataset_root,
            '--input_lr', '.2', '--rotation', '.02', '--lam_h', '10', '--lam_l', '1',
            '--n_update', '30', '--vote', '1', '--random_seed', '0', '--subsample', '2048']
        options = adapt.parse_arguments()
    finally:
        sys.argv = argv
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)

    def model_hash(m):
        h = hashlib.sha256()
        for key, value in sorted(m.state_dict().items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    original_hashes = [model_hash(m) for m in (model, classifier)]
    dataset, last_corruption, warmed = None, None, set()

    def infer(unit):
        nonlocal dataset, last_corruption
        if unit['corruption'] != last_corruption:
            options.dataset = 'modelnet40c_'+unit['corruption']+'_5'
            dataset = adapt.ModelNet40C(options, partition='test')
            last_corruption = unit['corruption']
        seed = seed_for(unit)

        def evaluate(config):
            torch.use_deterministic_algorithms(config['method']!='cloudfixer')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            batch = default_collate([dataset[i] for i in unit['ids']])
            input_identity=sha(b''.join(batch[j].contiguous().numpy().tobytes() for j in (0,2,3)))
            x, mask, ind = (batch[j].cuda() for j in (0, 2, 3))
            rng_identity=sha(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes())
            if config['method'] == 'source':
                candidates, calls, metadata = x[:, None], 0, {}
            elif config['method'] == 'cloudfixer':
                repaired = adapt.cloudfixer(options, torch.nn.DataParallel(model), x, mask, ind).detach()
                candidates, calls, metadata = repaired[:, None], 30, {}
            else:
                generator = torch.Generator(device=x.device).manual_seed(seed)
                noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                if proximal_evaluator is not None:
                    logits, metadata = proximal_evaluator(model, classifier, x, mask, noise, config)
                    metadata['input_rng_sha256'] = sha(canonical(dict(input=input_identity, rng=rng_identity)))
                    return logits, batch[1].reshape(-1), metadata['denoiser_calls'], metadata, tuple(x.shape)
                candidates, metadata = sample(model, x, mask, noise,
                    **{k: config[k] for k in ('steps', 'start', 'tau', 'retained')}, trace_all=trace_all)
                calls = metadata['denoiser_calls']
            metadata['input_rng_sha256']=sha(canonical(dict(input=input_identity,rng=rng_identity)))
            if 'valid_reference_counts' not in metadata:
                metadata['valid_reference_counts']=mask.reshape(len(x),-1).sum(1).long().cpu().tolist()
            with torch.no_grad(), torch.random.fork_rng(devices=[x.device.index]):
                views = candidates.unbind(1)
                order = list(range(len(views)))
                if trace_all and config['method'] == 'proximal':
                    n = config['steps']
                    # Preserve the original classifier RNG draws for the canonical four views.
                    canonical_views = [n//4-1, n//2-1, 3*n//4-1, n]
                    order = canonical_views + [i for i in order if i not in canonical_views]
                observations = [None] * len(views)
                for i in order:
                    observations[i] = classifier(views[i].contiguous())
                logits = torch.stack(observations, 1)
                # Include the actual PM reduction in pipeline timing; saved logits are authoritative.
                logits.double().softmax(-1).mean(1)
            return logits, batch[1].reshape(-1), calls, metadata, tuple(x.shape)

        shape = (len(unit['ids']), dataset.pc_list.shape[1], unit['config']['method'])
        if shape not in warmed:
            warm_config = dict(unit['config'])
            if warm_config['method'] == 'proximal' and proximal_evaluator is None:
                warm_config.update(steps=4, start=40, tau=.1)
            for _ in range(2):
                evaluate(warm_config)
            torch.cuda.synchronize()
            warmed.add(shape)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        logits, labels, calls, metadata, shape = evaluate(unit['config'])
        torch.cuda.synchronize()
        elapsed, peak = time.perf_counter()-started, torch.cuda.max_memory_allocated()/1024**2
        repeated = None
        if unit['phase'] == 'smoke':
            repeated, _, _, repeat_metadata, _ = evaluate(unit['config'])
            try:
                metadata['repeat_check']=check_repeat(logits,repeated,unit['config']['method'],
                    metadata['input_rng_sha256'],repeat_metadata['input_rng_sha256'])
            except (ValueError,FloatingPointError):
                np.savez_compressed(Path(args.output)/('repeat_failure_'+unit_key(unit)+'.npz'),
                    first=logits.detach().cpu().numpy(), repeated=repeated.detach().cpu().numpy())
                raise
        verify_models()
        stream = io.BytesIO()
        np.savez_compressed(stream, unit=json.dumps(unit, sort_keys=True), ids=unit['ids'], labels=labels.numpy(),
            logits=logits.detach().cpu().numpy(), seconds=elapsed, peak_mb=peak, denoiser_calls=calls,
            sampler=json.dumps(metadata, sort_keys=True), input_shape=shape,
            model_hashes=original_hashes,
            repeated_logits=(repeated.detach().cpu().numpy() if repeated is not None else np.empty((0,))))
        data = stream.getvalue()
        (Path(args.output)/('pending_'+unit_key(unit)+'.npz')).write_bytes(data)
        return data

    def verify_models():
        if [model_hash(m) for m in (model, classifier)] != original_hashes:
            raise ValueError('Frozen model state changed')
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), model_hashes=original_hashes, gamma_verified=True,gamma_validation=gamma_validation,
        classifier='author Point-MAE32 groups, all tokens, no external FPS1024', lion_loaded=False,
        repeat_policy='strict source/proximal; original CF native KNN backward variation',
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    return infer, verify_models, environment


def write_reports(output: Path, rows: list, campaign: str, expected_count: int, actual_count: int, *, select=True):
    (output/'analysis.json').write_text(json.dumps(dict(campaign=campaign, rows=rows,
        completed_units=actual_count, expected_units=expected_count,
        coverage='registered subset; not full benchmark'), indent=2)+'\n')
    with (output/'per_corruption.csv').open('w') as f:
        fields = ['run_id','dataset','severity','method','seed','corruption','n_examples','n_correct',
                  'accuracy','runtime_seconds','peak_gpu_memory_mb','status']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            for corruption, c in row['corruptions'].items():
                writer.writerow(dict(run_id=campaign, dataset='modelnet40_c', severity=5,
                    method=row['config_id']+'_'+row['readout'], seed=row['seed'], corruption=corruption,
                    **c, accuracy=c['n_correct']/c['n_examples'], peak_gpu_memory_mb=row['peak_gpu_memory_mb'], status='partial'))
    with (output/'summary.csv').open('w') as f:
        fields=['run_id','dataset','severity','method','seed','n_corruptions','macro_accuracy','total_examples',
                'total_correct','micro_accuracy','total_runtime_seconds','status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow(dict(run_id=campaign,dataset='modelnet40_c',severity=5,method=row['config_id']+'_'+row['readout'],
                seed=row['seed'],n_corruptions=len(row['corruptions']),macro_accuracy=row['macro_accuracy'],
                total_examples=row['n_examples'],total_correct=row['n_correct'],micro_accuracy=row['n_correct']/row['n_examples'],
                total_runtime_seconds=row['runtime_seconds'],status='partial'))
    if select and actual_count == expected_count:
        candidates=[r for r in rows if r['config']['method']=='proximal' and r['readout']=='pm' and r['status']=='complete']
        if candidates:
            best=min(candidates,key=lambda r:(-r['macro_accuracy'],r['runtime_seconds'],r['config_id']))
            frontier=[r for r in candidates if not any(s['macro_accuracy']>=r['macro_accuracy'] and
                s['runtime_seconds']<=r['runtime_seconds'] and (s['macro_accuracy']>r['macro_accuracy'] or
                s['runtime_seconds']<r['runtime_seconds']) for s in candidates)]
            (output/'selection.json').write_text(json.dumps(dict(best_config=best['config'],best=best,
                frontier=frontier,claim='Best tested registered subset; not a global optimum'),indent=2)+'\n')


def main(argv=None):
    process_started=time.monotonic()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--register',type=Path)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--phase',choices=('smoke','profile','development','confirmation'),default='smoke')
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--drive-folder',required=False,help='Task folder under mounted MyDrive/thesis/result')
    parser.add_argument('--official-root',default='/content/CloudFixer')
    parser.add_argument('--dataset-root',default='/content/3DD-TTA/data/modelnet40_c')
    parser.add_argument('--classifier',default='/content/CloudFixer/checkpoints/pointMAE_modelnet40.pth')
    parser.add_argument('--diffusion',default='/content/CloudFixer/checkpoints/diffusion_model_transformer_modelnet40.npy')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--setup-receipt',type=Path,default=Path('/content/cf_transfer_setup_receipt.json'))
    args=parser.parse_args(argv)
    if args.register:
        args.register.write_bytes(canonical(registration()));return
    if not args.manifest:
        parser.error('--manifest is required')
    manifest=json.loads(args.manifest.read_text())
    reg=manifest['registration']
    if reg != registration():
        raise ValueError('Registration differs from the approved search')
    selected=None
    if args.phase=='confirmation':
        if not args.selection:parser.error('Confirmation requires a frozen selection')
        selected=json.loads(args.selection.read_text())['best_config']
        if selected not in conditions('development'):raise ValueError('Invalid selected configuration')
    identity=dict(manifest=manifest,phase=args.phase,selected=selected)
    campaign=sha(canonical(identity))
    wanted=units(campaign,reg,args.phase,selected)
    if args.preview:
        print(json.dumps(dict(campaign=campaign,phase=args.phase,n_units=len(wanted),
            n_configs=len(conditions(args.phase)) if selected is None else 3,registration=reg),indent=2));return
    if not args.output or not args.drive_folder:
        parser.error('Execution/report requires --output and --drive-folder')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    from cloudfixer_store import MountStore
    mounted=Path('/content/drive')
    folder_root=Path(args.drive_folder).resolve()
    if not mounted.is_mount() or Path('/content/drive/MyDrive/thesis/result').resolve() not in folder_root.parents:
        raise ValueError('Expected mounted task folder under MyDrive/thesis/result')
    budget_store=MountStore(folder_root)
    budget_store.put('task.json',canonical(manifest))
    if not args.report_only:
        record_setup(budget_store,json.loads(args.setup_receipt.read_text()),cap=reg['budget_seconds'])
    budget=Budget(budget_store,cap=reg['budget_seconds'],started=process_started)
    task=sha(canonical(manifest))
    if not args.report_only and args.phase!='smoke':
        require_gate(budget_store,'smoke_gate.json',task)
    folder=folder_root/(args.phase+'_'+campaign[:12])
    store=MountStore(folder)
    store.put('campaign.json',canonical(identity))
    with budget.operation(1800):
        restored=load_units(store,campaign)
        expected={unit_key(u):u for u in wanted}
        if not set(restored)<=set(expected):
            raise ValueError('Recovered coverage differs from the registered units')
        for key,record in restored.items():validate_payload(record['data'],expected[key])
    if not args.report_only and args.phase=='development':
        gate=require_gate(budget_store,'profile_gate.json',task)
        remaining=[u for u in wanted if unit_key(u) not in restored]
        projection=project_grid(gate['profile'],reg,cloud_overhead=gate['cloud_overhead_seconds'],pending=remaining)
        if projection>budget.remaining():
            raise TimeoutError('Measured remaining-grid projection exceeds the remaining campaign budget')
    labels=__import__('numpy').load(Path(args.dataset_root)/'label.npy',allow_pickle=False).reshape(-1)
    if file_hash(Path(args.dataset_root)/'label.npy')!=manifest['dataset_files']['label.npy']:
        raise ValueError('Pinned label array mismatch')
    (args.output/'config.json').write_bytes(canonical(dict(identity,campaign=campaign,execution_status='running')))
    (args.output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    (args.output/'notes.md').write_text('Frozen author XYZ models; no LION. Registered subset only. Terminal readout uses PM trace timing, not terminal-only latency. Drive mount fsync/rename/readback; remote sync is not independently proven.\n')
    records=[]
    measurements=[]
    try:
        if args.report_only:
            records=list(restored.values())
        else:
            with budget.operation(1800):
                infer,verify,environment=runtime(args,manifest)
            (args.output/'environment.txt').write_text(json.dumps(environment,indent=2)+'\n')
            store.put('environment_'+sha(canonical(environment))+'.json',canonical(environment))
            def progress(value):
                print('CF_TRANSFER_PROGRESS',json.dumps(value),flush=True)
                (args.output/'progress.json').write_bytes(canonical(value))
                measurements.append(value)
            options=dict(cap=reg['budget_seconds'],labels=labels,progress=progress,budget=budget)
            if args.phase=='smoke':
                records=execute_smoke(store,lambda:MountStore(folder),campaign,wanted,infer,**options)
            else:
                records=execute_pending(store,campaign,wanted,infer,accepted=restored,**options)
        with budget.operation(300):
            if not args.report_only:
                verify()
                if args.phase=='smoke':
                    fresh=load_units(MountStore(folder),campaign)
                    if len(fresh)!=len(wanted):raise ValueError('Mounted smoke recovery coverage mismatch')
                    budget_store.put('smoke_gate.json',canonical(dict(status='passed',task=task,
                        campaign=campaign,completed_units=len(fresh),repeat_tolerance=dict(atol=1e-4,rtol=1e-5),
                        repeat_policy=reg['repeat_policy'],native_reference_repeats=3)))
                if args.phase=='profile':
                    gate=profile_gate(budget_store,store,task,records,reg)
                    (args.output/'projection.json').write_bytes(canonical(gate))
                    print('CF_TRANSFER_GRID_PROJECTION',json.dumps(gate),flush=True)
            write_reports(args.output,summarize(records,wanted),campaign,len(wanted),len(records),select=args.phase=='development')
            status='complete' if len(records)==len(wanted) else 'partial'
            (args.output/'config.json').write_bytes(canonical(dict(identity,campaign=campaign,execution_status=status)))
            for name in ('config.json','notes.md','command.txt','summary.csv','per_corruption.csv','analysis.json'):
                store.put('report_'+args.output.name+'_'+name,(args.output/name).read_bytes())
        print('CF_TRANSFER_REPORT_DONE' if args.report_only else 'CF_TRANSFER_PHASE_DONE',args.phase,flush=True)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        # Closed cloud units remain authoritative even if this attempt's report cannot be finalized.
        raise


if __name__=='__main__':
    main()
