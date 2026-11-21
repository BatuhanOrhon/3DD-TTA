#!/usr/bin/env python3
"""Frozen Point2Vec full benchmark with the released CloudFixer seed stream."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import csv
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, sha, unit_key
from research_artifacts import CORRUPTIONS
from scripts.run_cloudfixer_transfer import require_gate, verify_runtime_files


ARMS = ('source', 'proximal_pm', 'proximal_terminal', 'cloudfixer')
POLICIES = {
    'proximal_pm': dict(steps=8, start=80, tau=.1, retained=.95, probe_indices=[20, 10, 0]),
    'proximal_terminal': dict(steps=32, start=160, tau=.03, retained=.95, probe_indices=[0]),
}
CF_OPTIONS = dict(t_min=.02, t_len=.1, warmup=.2, input_lr=.2, rotation=.02, pow=1,
                  knn=5, weighted_reg=True, optim='adamax', n_update=30, weight_decay=0,
                  beta1=.9, beta2=.999, optim_end_factor=.05, lam_h=10, lam_l=1, vote=1,
                  subsample=2048, diffusion_steps=500, diffusion_noise_schedule='polynomial_2',
                  scenario='normal', imb_ratio=1, cls_scale_mode='unit_norm', scale_mode='unit_std')


def registration():
    return dict(task='cloudfixer_point2vec_20261118', dataset='modelnet40_c', severity=5,
                corruptions=sorted(CORRUPTIONS), objects=2468, seed=2, batch_size=1,
                arms=list(ARMS), policies=POLICIES, author_options=CF_OPTIONS,
                order='author shuffled DataLoader after diffusion and classifier construction; restart seed per corruption',
                proximal_rng='fork after source; common noise; independent index-addressed FPS draws',
                block_size=32, smoke_corruptions=['background', 'lidar'], smoke_objects=2,
                budget_seconds=30*3600)


def rng_state():
    import numpy as np
    import torch
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=[kind, keys.tolist(), pos, has_gauss, cached],
                cpu=torch.get_rng_state().tolist(), cuda=torch.cuda.get_rng_state().tolist())


def restore_rng(state):
    import numpy as np
    import torch
    version, keys, gauss = state['python']
    random.setstate((version, tuple(keys), gauss))
    np.random.set_state(tuple(state['numpy']))
    torch.set_rng_state(torch.tensor(state['cpu'], dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(state['cuda'], dtype=torch.uint8))


def load_head(checkpoint_path, head_spec):
    from classifier.point2vec.models import Point2VecClassification
    from classifier.point2vec.utils.checkpoint import extract_model_checkpoint
    head = Point2VecClassification().cuda()
    head.setup()
    checkpoint = extract_model_checkpoint(str(checkpoint_path))
    if (len(checkpoint) != head_spec['state_keys']
            or tuple(checkpoint['cls_head.8.weight'].shape) != (40, 256)
            or tuple(checkpoint['cls_head.8.bias'].shape) != (40,)):
        raise ValueError('Unexpected Point2Vec state or number of classes')
    head.load_state_dict(checkpoint, strict=True)
    return head.cuda().eval().requires_grad_(False)


def model_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def options_for(adapt, args):
    argv = sys.argv
    try:
        sys.argv = ['adapt.py', '--method', 'cloudfixer', '--mode', 'eval', '--use_best_hparam',
                    '--classifier', 'point2vec', '--classifier_dir', str(args.classifier),
                    '--diffusion_dir', str(args.diffusion), '--dataset_dir', str(args.dataset_root),
                    '--batch_size', '1', '--random_seed', '2', '--params_to_adapt', 'LN', 'GN', 'BN']
        for key, value in CF_OPTIONS.items():
            sys.argv.extend(['--'+key, str(value)])
        options = adapt.parse_arguments()
    finally:
        sys.argv = argv
    if any(getattr(options, k) != v for k, v in CF_OPTIONS.items()):
        raise ValueError('Parsed author CloudFixer recipe differs from registration')
    return options


@contextmanager
def deterministic_proximal():
    import torch
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def proximal_policy(model, classifier, x, mask, config, *, dense=False):
    import torch
    from cloudfixer_diffusion import sample
    device_ids = [x.device.index] if x.is_cuda else []
    with deterministic_proximal(), torch.no_grad(), torch.random.fork_rng(devices=device_ids):
        noise = torch.randn_like(x)
        # Index-addressed FPS draws let a sparse run omit all other classifier forwards.
        seeds = torch.randint(2**31, (501,), device='cpu').tolist()
        settings = {k: v for k, v in config.items() if k != 'probe_indices'}
        clouds, meta = sample(model, x, mask, noise, **settings, trace_all=dense,
                              probe_indices=None if dense else config['probe_indices'])
        indices = meta['candidate_indices']
        logits = []
        for index, cloud in zip(indices, clouds.unbind(1)):
            torch.random.default_generator.manual_seed(seeds[index])
            if x.is_cuda:
                torch.cuda.manual_seed(seeds[index])
            logits.append(classifier(cloud.contiguous()))
        logits = torch.stack(logits, dim=1)
        probabilities = logits.double().softmax(dim=-1)
        readout = probabilities.mean(dim=1) if len(indices) > 1 else probabilities[:, 0]
        return logits, readout, clouds, indices, meta['denoiser_calls']


def timed(function):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start, torch.cuda.max_memory_allocated()/2**20


@contextmanager
def audit_native(adapt, records, production=None):
    """Instrument the pinned author entrypoints for a smoke object without advancing its RNG."""
    from unittest.mock import patch
    import numpy as np
    import torch

    original_forward = adapt.forward_and_adapt
    original_cf = adapt.cloudfixer

    def checked_forward(args, classifier, optimizer, diffusion_model, x, mask, ind):
        record = {}
        adapted = None
        calls = []

        def checked_cf(options, model, cloud, cloud_mask, indices):
            nonlocal adapted
            if adapted is not None or options is not args or model is not diffusion_model:
                raise ValueError('Unexpected native CloudFixer invocation')
            tensors = (cloud, cloud_mask, indices)
            if any(not torch.equal(actual, expected) for actual, expected in zip(tensors, (x, mask, ind))):
                raise ValueError('Native CloudFixer inputs differ from forward_and_adapt')
            record['native_input_sha256'] = sha(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors))
            settings = {k: v for k, v in vars(options).items() if k not in ('exp_name', 'out_path')}
            record['native_options_sha256'] = sha(canonical(settings))
            record['native_rng_sha256'] = sha(canonical(rng_state()))
            result = original_cf(options, model, cloud, cloud_mask, indices)
            adapted = result.detach().clone()
            return result

        def capture_input(module, inputs):
            if adapted is None or not torch.equal(inputs[0], adapted):
                raise ValueError('Classifier input differs from the native adapted XYZ')
            calls.append([rng_state(), torch.is_grad_enabled()])

        def capture_output(module, inputs, logits):
            calls[-1].append(logits.detach().clone())

        with ExitStack() as hooks:
            hooks.enter_context(patch.object(adapt, 'cloudfixer', checked_cf))
            hooks.enter_context(classifier.register_forward_pre_hook(capture_input))
            hooks.enter_context(classifier.register_forward_hook(capture_output))
            plain, vote = original_forward(args, classifier, optimizer, diffusion_model, x, mask=mask, ind=ind)
        if adapted is None or len(calls) != 2:
            raise ValueError('Expected original CloudFixer once and both plain/vote classifier calls')
        torch.testing.assert_close(plain, calls[0][2], atol=0, rtol=0)
        torch.testing.assert_close(vote, calls[1][2].softmax(-1).log(), atol=0, rtol=0)
        end = rng_state()
        try:
            for pos, (before, grad, expected) in enumerate(calls):
                restore_rng(before)
                with torch.set_grad_enabled(grad):
                    # Bypass only DataParallel's observer hooks on this extra replay.
                    replay = classifier.forward(adapted)
                torch.testing.assert_close(replay, expected, atol=0, rtol=0)
                if production is not None:
                    # Recount the *saved* production outputs with the independently loaded author head.
                    i = len(records)
                    xyz = torch.as_tensor(production['native_xyz'][i], device=x.device).unsqueeze(0)
                    restore_rng(before)
                    with torch.set_grad_enabled(grad):
                        replay = classifier.forward(xyz)
                    if pos == 0:
                        saved = production['cloudfixer_logits'][i]
                    else:
                        replay = replay.softmax(-1).log()
                        saved = production['vote_log_probs'][i:i+1]
                    np.testing.assert_array_equal(replay.detach().cpu().numpy(), saved)
        finally:
            restore_rng(end)
        record['native_xyz'] = adapted.cpu().numpy()[0].copy()
        record['native_head_rng_sha256'] = np.array([sha(canonical(c[0])) for c in calls])
        records.append(record)
        return plain, vote

    with patch.object(adapt, 'forward_and_adapt', checked_forward):
        yield


def score_object(adapt, options, model, head, batch, *, smoke):
    import numpy as np
    import torch
    x, mask, ind = (batch[i].cuda() for i in (0, 2, 3))
    raw = {}
    per_arm = {}
    with torch.no_grad():
        if smoke:
            with torch.random.fork_rng(devices=[x.device.index]):
                direct = head.module(x)
        (source, seconds, memory) = timed(lambda: head(x))
        if smoke:
            torch.testing.assert_close(source, direct, atol=1e-5, rtol=1e-5)
        raw['source'] = source[:, None]
        per_arm['source'] = (seconds, memory)
    for arm, config in POLICIES.items():
        before = canonical(rng_state()) if smoke else None
        (logits, probs, clouds, indices, calls), seconds, memory = timed(
            lambda: proximal_policy(model.module, head, x, mask, config))
        if indices != config['probe_indices'] or calls != config['steps'] + 1:
            raise ValueError('Incorrect production probe schedule')
        if smoke:
            dense_logits, _, dense_clouds, dense_indices, _ = proximal_policy(model.module, head, x, mask, config, dense=True)
            columns = [dense_indices.index(i) for i in indices]
            torch.testing.assert_close(clouds, dense_clouds[:, columns], atol=0, rtol=0)
            torch.testing.assert_close(logits, dense_logits[:, columns], atol=1e-5, rtol=1e-5)
            if canonical(rng_state()) != before:
                raise ValueError('Proximal observation changed author RNG state')
        raw[arm] = logits
        per_arm[arm] = (seconds, memory)
        del clouds, logits, probs
        if smoke:
            del dense_logits, dense_clouds

    # Record end of the author's *first* classifier forward. Let the second (vote)
    # forward execute normally to preserve the RNG state seen by the next object.
    native = []
    def observe_plain(module, inputs, output):
        if not native:
            torch.cuda.synchronize()
            native.extend((time.perf_counter()-started, torch.cuda.max_memory_allocated()/2**20))
    hook = head.register_forward_hook(observe_plain)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        evidence = []
        with audit_native(adapt, evidence) if smoke else nullcontext():
            plain, vote_log_probs = adapt.forward_and_adapt(options, head, None, model, x, mask=mask, ind=ind)
    finally:
        hook.remove()
    raw['cloudfixer'] = plain.detach()[:, None]
    per_arm['cloudfixer'] = native
    # All timings exclude the comparison gate, file I/O and the diagnostic vote forward.
    result = dict(vote_log_probs=vote_log_probs.detach().cpu().numpy()[0],
                  seconds=np.array([per_arm[a][0] for a in ARMS]),
                  peak_mb=np.array([per_arm[a][1] for a in ARMS]),
                  input_sha256=sha(b''.join(batch[i].contiguous().numpy().tobytes() for i in (0, 2, 3))))
    if smoke:
        result.update(evidence[0])
    for arm in ARMS:
        logits = raw[arm].detach()[0]
        probs = logits.double().softmax(dim=-1)
        result[arm+'_logits'] = logits.cpu().numpy()
        result[arm+'_probs'] = probs.cpu().numpy()
        result[arm+'_entropy'] = (-torch.special.xlogy(probs, probs).sum(-1)).cpu().numpy()
    return result


def validate(data, unit, labels):
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as f:
        p = {k: f[k] for k in f.files}
    if json.loads(str(p['unit'])) != unit or not np.array_equal(p['ids'], unit['ids']):
        raise ValueError('Incorrect prediction block identity')
    ids, n = p['ids'], len(unit['ids'])
    if not np.array_equal(p['labels'], labels[ids]):
        raise ValueError('Incorrect original-ID labels')
    if (p['vote_log_probs'].shape != (n,40) or not np.isfinite(p['vote_log_probs']).all()
            or p['input_sha256'].shape != (n,) or any(len(x)!=64 for x in p['input_sha256'])
            or any(p[k].shape != (n,4) or not np.isfinite(p[k]).all() or np.any(p[k] <= 0)
                   for k in ('seconds', 'peak_mb'))):
        raise ValueError('Invalid vote, input or timing evidence')
    for arm in ARMS:
        indices = POLICIES[arm]['probe_indices'] if arm in POLICIES else []
        count = len(indices) or 1
        logits = p[arm+'_logits']
        if logits.shape != (n,count,40) or not np.isfinite(logits).all():
            raise ValueError('Invalid logits for '+arm)
        shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
        e = np.exp(shifted)
        prob = e / e.sum(-1, keepdims=True)
        if (p[arm+'_probs'].shape != prob.shape or p[arm+'_entropy'].shape != (n,count)
                or not np.allclose(prob, p[arm+'_probs'], atol=1e-12, rtol=1e-10)
                or not np.allclose(-(prob*np.log(prob.clip(1e-300))).sum(-1), p[arm+'_entropy'])):
            raise ValueError('Incorrect saved probability/entropy for '+arm)
        if indices and (p[arm+'_indices'].tolist() != indices or
                        p[arm+'_logsnr'].shape != (count,) or not np.isfinite(p[arm+'_logsnr']).all()):
            raise ValueError('Incorrect native probe/logSNR evidence')
    return p


def restore_block(data, unit, labels, before_sha):
    p = validate(data, unit, labels)
    if str(p['rng_before_sha256']) != before_sha:
        raise ValueError('Recovered block does not continue the author RNG stream')
    restore_rng(json.loads(str(p['rng_after'])))
    return p


def counts(p):
    pred = {}
    for i, arm in enumerate(ARMS):
        prob = p[arm+'_probs'].mean(axis=1)
        pred[arm] = dict(n_examples=len(prob), n_correct=int((prob.argmax(-1)==p['labels']).sum()),
                         runtime_seconds=float(p['seconds'][:,i].sum()),
                         peak_gpu_memory_mb=float(p['peak_mb'][:,i].max()))
    return pred


def author_prefix(adapt, options, payload, output):
    """Compare the ordinary author main (including model setup + shuffled loader) on the smoke prefix."""
    from copy import copy
    from itertools import islice
    from unittest.mock import patch
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    observed = []
    native = []
    logits = []
    n = len(payload['ids'])

    def loader(dataset, **kwargs):
        class Indexed:
            def __len__(self):
                return len(dataset)

            def __getitem__(self, index):
                return (*dataset[index], index)

        for batch in islice(DataLoader(Indexed(), **kwargs), n):
            identity = sha(b''.join(batch[i].contiguous().numpy().tobytes() for i in (0, 2, 3)))
            observed.append((int(batch[-1].item()), int(batch[1].item()), identity))
            yield batch[:-1]

    def capture(module, inputs, logit):
        if isinstance(module, torch.nn.DataParallel) and type(module.module).__name__ == 'Point2VecClassification':
            logits.append(logit.detach().cpu().numpy().copy())

    class Prefix:
        def __init__(self, dataset, **kwargs):
            self.dataset, self.kwargs = dataset, kwargs

        def __iter__(self):
            return loader(self.dataset, **self.kwargs)

        def __len__(self):
            return n

    options = copy(options)
    options.out_path, options.exp_name = str(output), 'author_'+options.dataset
    state, cwd = rng_state(), Path.cwd()
    hook = torch.nn.modules.module.register_module_forward_hook(capture)
    try:
        os.chdir(output)
        with patch.object(adapt, 'DataLoader', Prefix), audit_native(adapt, native, payload):
            adapt.main(options)
        final_state = rng_state()
    finally:
        hook.remove()
        os.chdir(cwd)
        restore_rng(state)
    expected = list(zip(payload['ids'], payload['labels'], payload['input_sha256']))
    if observed != expected or len(logits) != 3*n:
        raise ValueError('Author prefix differs in shuffled IDs, labels, inputs or classifier calls')
    if json.loads(str(payload['rng_after'])) != json.loads(canonical(final_state)):
        raise ValueError('Final RNG differs from the author prefix')
    raw = np.concatenate(logits).reshape(n,3,40)
    np.testing.assert_allclose(raw[:,0], payload['source_logits'][:,0], atol=1e-5,rtol=1e-5)
    if len(native) != n:
        raise ValueError('Missing audited native CloudFixer calls')
    fields = {k: np.stack([record[k] for record in native]) for k in native[0]}
    for key in ('native_input_sha256', 'native_options_sha256', 'native_rng_sha256', 'native_head_rng_sha256'):
        if not np.array_equal(fields[key], payload[key]):
            raise ValueError('Author native call differs from production: '+key)
    if not np.array_equal(fields['native_input_sha256'], payload['input_sha256']):
        raise ValueError('Author native inputs differ from the shuffled dataset inputs')
    # Original Chamfer KNN backward may vary at identical RNG; retain each invocation's adapted XYZ.
    report = dict(ids=payload['ids'].tolist(), ordered_inputs_equal=True, end_rng_equal=True,
                native_calls_and_rng_equal=True, native_xyz_plain_vote_replay_exact=True,
                source_max_abs=float(np.abs(raw[:,0]-payload['source_logits'][:,0]).max()),
                plain_cf_max_abs=float(np.abs(raw[:,1]-payload['cloudfixer_logits'][:,0]).max()),
                plain_cf_argmax_changes=int((raw[:,1].argmax(-1)!=payload['cloudfixer_logits'][:,0].argmax(-1)).sum()))
    np.savez_compressed(output/(options.dataset+'_author_reference.npz'), ids=payload['ids'], logits=raw,
                        report_json=canonical(report).decode(), **fields)
    return report


def saved_author_prefix(adapt, options, payload, output, cache, campaign, block_sha):
    """Resume the passed original-author audit, whose native KNN optimization can vary on repetition."""
    import numpy as np
    store = MountStore(cache.folder/'author_prefix')
    unit = dict(campaign=campaign, prediction_sha256=block_sha)
    key = unit_key(unit)
    restored = load_units(store, campaign)
    path = output/(options.dataset+'_author_reference.npz')
    if not restored:
        report = author_prefix(adapt, options, payload, output)
        commit_unit(store, unit, path.read_bytes())
        return report
    if set(restored) != {key}:
        raise ValueError('Recovered author audit differs from the smoke prediction block')
    data = restored[key]['data']
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        report = json.loads(str(archive['report_json']))
    path.write_bytes(data)
    return report



def reports(output, collected, phase, campaign):
    per = []
    for (arm, corr), rows in collected.items():
        n = sum(r['n_examples'] for r in rows)
        good = sum(r['n_correct'] for r in rows)
        per.append(dict(run_id=campaign, dataset='modelnet40_c', severity=5, method=arm, seed=2,
                        corruption=corr, n_examples=n, n_correct=good, accuracy=good/n,
                        runtime_seconds=sum(r['runtime_seconds'] for r in rows),
                        peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in rows),
                        status='complete' if phase=='full' and n==2468 else 'partial'))
    summary=[]
    for arm in ARMS:
        rows=[r for r in per if r['method']==arm]
        if not rows:
            continue
        n=sum(r['n_examples'] for r in rows)
        good=sum(r['n_correct'] for r in rows)
        summary.append(dict(run_id=campaign, dataset='modelnet40_c', severity=5, method=arm, seed=2,
                            n_corruptions=len(rows), macro_accuracy=sum(r['accuracy'] for r in rows)/len(rows),
                            total_examples=n,total_correct=good,micro_accuracy=good/n,
                            total_runtime_seconds=sum(r['runtime_seconds'] for r in rows),
                            peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in rows),
                            status='complete' if len(rows)==15 and all(r['status']=='complete' for r in rows) else 'partial'))
    for name, rows in [('summary.csv',summary),('per_corruption.csv',per)]:
        if rows:
            with (output/name).open('w', newline='') as out:
                writer=csv.DictWriter(out, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke','full'), required=True)
    parser.add_argument('--drive-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, default=Path('/content/CloudFixer'))
    parser.add_argument('--classifier', type=Path, default=Path('/content/CloudFixer/checkpoints/point2vec_modelnet40.ckpt'))
    parser.add_argument('--diffusion', type=Path, default=Path('/content/CloudFixer/checkpoints/diffusion_model_transformer_modelnet40.npy'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/content/3DD-TTA/data/modelnet40_c'))
    args=parser.parse_args(argv)
    for key in ('manifest','drive_folder','output','official_root','classifier','diffusion','dataset_root'):
        setattr(args,key,getattr(args,key).resolve())
    spec=json.loads(args.manifest.read_text())
    head_spec=json.loads((ROOT/'cfgs/cloudfixer_point2vec_assets.json').read_text())
    reg=registration()
    if (spec.get('schema') != 2 or spec.get('classifier') != head_spec or spec.get('registration') != reg
            or spec.get('classifier_sha256')!=head_spec['classifier_sha256']):
        raise ValueError('Point2Vec manifest or locked arm registration mismatch')
    if not Path('/content/drive').is_mount() or Path('/content/drive/MyDrive/thesis/result').resolve() not in args.drive_folder.parents:
        raise ValueError('Expected a mounted MyDrive/thesis/result task folder')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'raw').mkdir()
    store=MountStore(args.drive_folder)
    store.put('task.json', canonical(spec))
    campaign=sha(canonical(dict(manifest=spec,phase=args.phase)))
    task=sha(canonical(spec))
    if args.phase=='full':
        require_gate(store,'smoke_gate.json',task)
    config=dict(manifest=spec,campaign=campaign,phase=args.phase,execution_status='running')
    (args.output/'config.json').write_bytes(canonical(config))
    (args.output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    (args.output/'notes.md').write_text(
        'Four registered rows include source-only and three full policies. Frozen Point2Vec, paper B1, '
        'author shuffled seed2 per corruption (example shell uses B128). PM window selected on old Point-MAE development labels. '
        'Proximal branches borrow the RNG after the source forward and restore it before published CF. '
        'Proximal uses strict deterministic algorithms for the colliding CUDA scatter and restores '
        'the inherited algorithm and warn-only flags before native CF. Independent preassigned FPS '
        'seeds by native index. CF primary is logits_after. The second '
        'vote1 forward stays in the author stream; timed native ends at first forward. '
        'Smoke native timings include geometry/RNG audit overhead. The author pc_utils.set_seed '
        'shadows utils.utils.set_seed and leaves inherited cuDNN and deterministic flags intact. Single seed.\n')
    budget=Budget(store, cap=reg['budget_seconds'])
    try:
        run(args,spec,head_spec,reg,store,campaign,task,budget)
    except BaseException:
        (args.output/'config.json').write_bytes(canonical(dict(config,execution_status='failed')))
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


def run(args,spec,head_spec,reg,store,campaign,task,budget):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from cloudfixer_diffusion import verify_gamma
    official=verify_runtime_files(args,spec)
    from scripts.setup_cloudfixer_point2vec_colab import runtime_identity
    environment_identity = runtime_identity()
    os.chdir(official)
    sys.path.insert(0,str(official))
    import adapt
    if Path(adapt.__file__).resolve() != official/'adapt.py':
        raise ValueError('The loaded author entrypoint differs from the pinned checkout')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Point2Vec registered for a single visible CUDA device')
    labels=np.load(args.dataset_root/'label.npy',allow_pickle=False).reshape(-1)
    if len(labels)!=2468 or set(labels.tolist())!=set(range(40)):
        raise ValueError('Unexpected ModelNet40-C label space')
    options=options_for(adapt,args)
    smoke=args.phase=='smoke'
    corruptions=reg['smoke_corruptions'] if smoke else reg['corruptions']
    collected={}
    reference={}
    initialized_rng=None
    for corr in corruptions:
        adapt.set_seed(reg['seed'])
        options.dataset='modelnet40c_'+corr+'_5'
        # adapt.main constructs the ModelNet40C test dataset twice before loading models.
        dataset=adapt.ModelNet40C(options,partition='test')
        dataset=adapt.ModelNet40C(options,partition='test')
        if (not np.array_equal(dataset.label_list.reshape(-1),labels) or
                [dataset.idx_to_label[i] for i in range(40)]!=head_spec['class_order']):
            raise ValueError('Invalid author class order or object-label map')
        loader=DataLoader(range(len(dataset)),batch_size=1,shuffle=True,num_workers=0,drop_last=False)
        if initialized_rng is None:
            model=adapt.get_model(options,torch.device('cuda'))
            intended=model.gamma.gamma.detach().clone()
            model.load_state_dict(torch.load(args.diffusion,map_location='cpu'),strict=True)
            gamma=verify_gamma(model.gamma.gamma,intended)
            model=torch.nn.DataParallel(model).cuda().eval().requires_grad_(False)
            head=torch.nn.DataParallel(load_head(args.classifier,head_spec)).cuda().eval().requires_grad_(False)
            hashes=[model_hash(m) for m in (model,head)]
            initialized_rng=rng_state()
        else:
            # Every official corruption starts in a fresh process with the same model init.
            restore_rng(initialized_rng)
        adapt.adaptation_time=0
        # Range items expose original IDs. With zero workers, getitem only uses NumPy.
        order=[int(i.item()) for i in loader]
        if sorted(order)!=list(range(len(labels))):
            raise ValueError('Shuffled official order lacks complete original-ID coverage')
        scope=order[:reg['smoke_objects']] if smoke else order
        order_sha=sha(canonical(order))
        if corr==corruptions[0]:
            env=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),gamma=gamma,model_hashes=hashes,
                     cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                     cudnn_enabled=torch.backends.cudnn.enabled,tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32, deterministic=torch.are_deterministic_algorithms_enabled(),
                     deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                     proximal_deterministic=True, proximal_warn_only=False, supplemental_setup=environment_identity,
                     options=vars(options))
            (args.output/'environment.txt').write_text(json.dumps(env,indent=2)+'\n')
            store.put('runtime_signature.json',canonical({k:v for k,v in env.items() if k!='options'}))
        (args.output/(corr+'_order.json')).write_bytes(canonical(order))
        cache=MountStore(store.folder/(args.phase+'_'+campaign[:16])/corr)
        restored=load_units(cache,campaign)
        units=[dict(campaign=campaign,corruption=corr,seed=2,batch_size=1,offset=i,order_sha256=order_sha,
                    ids=scope[i:i+reg['block_size']]) for i in range(0,len(scope),reg['block_size'])]
        keys=[unit_key(unit) for unit in units]
        if set(restored)!=set(keys[:len(restored)]):
            raise ValueError('Recovered blocks must be a contiguous prefix of the registered author stream')
        for block,unit in enumerate(units):
            key=keys[block]
            with budget.operation(1800):
                before_sha=sha(canonical(rng_state()))
                if key in restored:
                    data=restored[key]['data']
                    p=restore_block(data,unit,labels,before_sha)
                else:
                    rows=[]
                    for ident in unit['ids']:
                        batch=default_collate([dataset[ident]])
                        if int(batch[1].item())!=labels[ident]:
                            raise ValueError('Dataset object has unexpected label')
                        if block==len(restored) and not rows:
                            state=rng_state()
                            score_object(adapt,options,model,head,batch,smoke=False)
                            restore_rng(state)
                        rows.append(score_object(adapt,options,model,head,batch,smoke=smoke))
                    values={key_:np.stack([r[key_] for r in rows]) for key_ in rows[0]}
                    for arm,config in POLICIES.items():
                        ii=config['probe_indices']
                        t=torch.tensor(ii,dtype=torch.float32,device='cuda').reshape(-1,1)/500
                        values[arm+'_indices']=ii
                        values[arm+'_logsnr']=(-model.module.gamma(t)).detach().cpu().numpy().reshape(-1)
                    stream=io.BytesIO()
                    np.savez_compressed(stream,unit=json.dumps(unit,sort_keys=True),ids=unit['ids'],
                                        labels=labels[unit['ids']],rng_before_sha256=before_sha,rng_after=canonical(rng_state()).decode(),**values)
                    data=stream.getvalue()
                    p=validate(data,unit,labels)
                    if [model_hash(m) for m in (model,head)]!=hashes:
                        raise ValueError('Frozen model state changed')
                    commit_unit(cache,unit,data)
                receipt=cache.get(key)
                payload_name=json.loads(receipt)['payload']
                (args.output/'raw'/payload_name).write_bytes(data)
                (args.output/'raw'/key).write_bytes(receipt)
                for arm,record in counts(p).items():
                    collected.setdefault((arm,corr),[]).append(record)
                print('POINT2VEC_BLOCK',corr,block+1,len(units),key,flush=True)
        if smoke:
            with budget.operation(1800):
                reference[corr]=saved_author_prefix(adapt,options,p,args.output,cache,campaign,sha(data))
                (args.output/'author_prefix.json').write_bytes(canonical(reference))
        del dataset
    reports(args.output,collected,args.phase,campaign)
    (args.output/'config.json').write_bytes(canonical(dict(manifest=spec,campaign=campaign,phase=args.phase,execution_status='complete')))
    for name in ('summary.csv','per_corruption.csv','config.json','command.txt','environment.txt','notes.md'):
        store.put('report_'+args.output.name+'_'+name,(args.output/name).read_bytes())
    if smoke:
        store.put('smoke_gate.json',canonical(dict(task=task,status='passed',campaign=campaign,
                  sparse_dense_xyz='exact',sparse_dense_logits='atol=1e-5,rtol=1e-5',
                  unchanged_author_rng=True,source_direct_checked=True,author_prefix=reference)))
    print('CF_POINT2VEC_PHASE_DONE',args.phase,flush=True)


if __name__=='__main__':
    main()
