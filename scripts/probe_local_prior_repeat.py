#!/usr/bin/env python3
"""Three exact-input/RNG native prior forwards per first Gaussian/Background state."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LEAF_BYTES = 64*1024**2


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def paired_rng(path, expected, initial_sha, prior_sha):
    from legacy_trajectory import _rng_state_hash
    if identity(path) != {k: expected[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Paired cache identity mismatch')
    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z['metadata'].item())); py = m['python_rng']
        state = ((py[0], tuple(py[1]), py[2]), (m['numpy_rng'][0], z['numpy_rng_keys'].copy(), *m['numpy_rng'][1:]),
                 torch.from_numpy(z['torch_rng_cpu'].copy()),
                 [torch.from_numpy(z['torch_rng_cuda_%d'%i].copy()) for i in range(m['cuda_devices'])])
    if not _rng_state_hash(state) == m['rng_after_sha256'] == initial_sha == prior_sha:
        raise ValueError('Exact initial/cache/prior RNG chain mismatch; no seed-only fallback')
    return state


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Output shape/dtype mismatch')
    delta = a.double()-b.double()
    denominator = max(float(a.double().norm()), float(b.double().norm()))
    return dict(max_absolute=float(delta.abs().max()) if delta.numel() else 0.,
        relative_l2=float(delta.norm())/denominator if denominator else 0.,
        per_object_rms=delta.reshape(len(a), -1).square().mean(1).sqrt().tolist() if a.ndim and a.numel() else None)


def tensor_tree(value):
    """Hash actual pre-forward tensor/scalar trees; unsupported objects stay unavailable."""
    from legacy_trajectory import tensor_hash
    tensors, supported = {}, [True]
    def visit(item, key):
        if isinstance(item, torch.Tensor):
            tensors[key] = item
            return dict(tensor_sha256=tensor_hash(item), shape=list(item.shape), dtype=str(item.dtype), requires_grad=item.requires_grad)
        if item is None or type(item) in (str, int, float, bool):
            return item
        if isinstance(item, (list, tuple)):
            return [visit(v, key+'/'+str(i)) for i, v in enumerate(item)]
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            return {k: visit(v, key+'/'+k) for k, v in sorted(item.items())}
        supported[0] = False
        return dict(unsupported_type=type(item).__module__+'.'+type(item).__name__)
    tree = visit(value, 'root')
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest(), tensors, supported[0]


class LeafRecorder:
    """Bounded CPU reference outputs; hooks add synchronization, never a forward."""
    def __init__(self):
        self.reference, self.bytes = {}, 0

    def install(self, model, repeat):
        from legacy_trajectory import rng_hash
        events, pending, counts, hooks = [], {}, {}, []
        def before(name, module, args, kwargs):
            count = counts.get(name, 0); counts[name] = count+1
            sha, _, supported = tensor_tree((args, kwargs))
            pending[name] = dict(module=name, call_index=count, inputs_sha256=sha, inputs_supported=supported,
                                 rng_before_sha256=rng_hash(), grad_enabled=torch.is_grad_enabled())
        def after(name, module, args, kwargs, output):
            row = pending.pop(name); key = (name, row['call_index'])
            sha, tensors, supported = tensor_tree(output)
            row.update(outputs_sha256=sha, outputs_supported=supported,
                       outputs_finite=all(bool(torch.isfinite(v).all()) for v in tensors.values()))
            if repeat == 0:
                size = sum(v.numel()*v.element_size() for v in tensors.values())
                cpu = {k: v.detach().cpu().clone() for k, v in tensors.items()} if self.bytes+size <= LEAF_BYTES else None
                if cpu is not None:
                    self.bytes += size
                self.reference[key] = dict(row=row.copy(), outputs=cpu)
                row['numeric_reference_available'] = cpu is not None
            elif key in self.reference:
                ref = self.reference[key]; base = ref['row']
                exact = (row['inputs_supported'] and base['inputs_supported'] and row['inputs_sha256'] == base['inputs_sha256']
                         and row['rng_before_sha256'] == base['rng_before_sha256'] and row['grad_enabled'] == base['grad_enabled'])
                row.update(inputs_exact=exact, output_hash_equal=sha == base['outputs_sha256'], numeric_reference_available=ref['outputs'] is not None)
                if ref['outputs'] is not None and set(ref['outputs']) == set(tensors):
                    row['numeric_differences'] = {k: difference(ref['outputs'][k], v.detach().cpu()) for k, v in tensors.items()}
                    row['exact_input_numeric_output_drift'] = bool(exact and any(v['max_absolute'] > 0 for v in row['numeric_differences'].values()))
            events.append(row)
        for name, module in model.named_modules():
            if name and not any(module.children()):
                hooks.append(module.register_forward_pre_hook(lambda m, a, k, n=name: before(n, m, a, k), with_kwargs=True))
                hooks.append(module.register_forward_hook(lambda m, a, k, o, n=name: after(n, m, a, k, o), with_kwargs=True))
        return events, hooks


def repeat_prior(prior, h, condition, prior_t, state, expected_rng, directory):
    from legacy_trajectory import _restore_rng, rng_hash, tensor_hash, model_buffer_hashes
    from semantic_trajectory import gradient_probe
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    original = [tensor_hash(v) for v in (h, condition, prior_t)]
    entry_rng, buffers = rng_hash(), model_buffer_hashes({'prior': prior})
    recorder, outputs, records = LeafRecorder(), [], []
    for repeat in range(3):
        events, hooks = recorder.install(prior, repeat)
        try:
            with gradient_probe([prior]):
                local = h.detach().clone().requires_grad_(True)
                style = condition.detach().clone().requires_grad_(True)
                t = prior_t.detach().clone()
                _restore_rng(state)
                if rng_hash() != expected_rng:
                    raise ValueError('Actual restored prior RNG differs')
                input_hashes = [tensor_hash(v) for v in (local, style, t)]
                if input_hashes != original:
                    raise ValueError('Actual prior input clone differs')
                started = time.perf_counter()
                result = prior(x=local, t=t, condition_input=style, clip_feat=None)
                if result.shape != h.shape or result.dtype != h.dtype or not bool(torch.isfinite(result).all()):
                    raise ValueError('Invalid finite actual prior output')
                output = result.detach().cpu().clone()
                row = dict(repeat=repeat, output_sha256=tensor_hash(output), finite=True,
                    input_sha256=input_hashes, rng_before_sha256=expected_rng,
                    rng_after_forward_sha256=rng_hash(), elapsed_with_hooks_seconds=time.perf_counter()-started,
                    grad_enabled=torch.is_grad_enabled(), local_leaf=local.is_leaf, style_leaf=style.is_leaf,
                    local_requires_grad=local.requires_grad, style_requires_grad=style.requires_grad)
                with (directory/('repeat%d_output.npz'%repeat)).open('xb') as stream:
                    np.savez_compressed(stream, noise_pred=output.numpy())
                records.append(row); outputs.append(output)
                if input_hashes != [tensor_hash(v) for v in (local, style, t)]:
                    raise RuntimeError('Actual prior mutated its fixed inputs')
                del result
        finally:
            for hook in hooks:
                hook.remove()
            write(directory/('repeat%d_leaf_trace.json'%repeat), events)
            write(directory/'partial.json', dict(repeats=records, numeric_leaf_reference_bytes=recorder.bytes))
        if rng_hash() != entry_rng or model_buffer_hashes({'prior': prior}) != buffers:
            raise RuntimeError('Prior repeat changed caller RNG/buffers')
    if original != [tensor_hash(v) for v in (h, condition, prior_t)]:
        raise RuntimeError('Prior repeat mutated archived inputs')
    report = dict(status='complete', exact_rng_verified=True, forward_calls=3, backward_calls=0, repeats=records,
        pair_differences=[dict(repeats=[a,b], **difference(outputs[a], outputs[b])) for a,b in itertools.combinations(range(3),2)],
        numeric_leaf_reference_bytes=recorder.bytes, leaf_reference_limit_bytes=LEAF_BYTES,
        limitation='Instrumented network forwards only; leaf hooks synchronize and perturb scheduling. Bounded reference coverage, unsupported input trees or unequal module inputs cannot identify a kernel. No speed or trajectory claim.')
    write(directory/'summary.json', report)
    return report


def load_capture(root, config_sha, corruption):
    from scripts.analyze_scan_trajectory import Bundle
    from scripts import eval_multibackbone_source as source
    from legacy_trajectory import tensor_hash
    root = Path(root); bundle = Bundle(root)
    try:
        config = bundle.json('config.json')
        if identity(root/'config.json')['sha256'] != config_sha:
            raise ValueError('Pinned capture config mismatch')
        ids = [44,91] if corruption == 'gaussian' else [495,512]
        if not (config['diagnostic_complete'] is True and config['arm']=='control' and config['weight']==0
                and config['capture_native_trace'] is True and config['count']==2 and config['selected_original_ids']==ids
                and config['corruptions']==[corruption] and config['seed']==0 and config['backend_policy']=='controlled'
                and config['lion_state_before']==config['lion_state_after']):
            raise ValueError('Wrong completed native-trace control scope')
        prefix='batches/'+corruption+'/batch0000/'
        diagnostic=bundle.json(prefix+'diagnostics.json'); trace=bundle.json(prefix+'native_trace.json')
        source.verify_identity(root/prefix/'native_trace.json', diagnostic['native_trace'])
        first=trace['snapshots'][0]
        if (first['loop_count'],first['boundary'])!=(1,'prior_input') or first['rng_before_sha256']!=first['rng_after_sha256']:
            raise ValueError('Wrong first prior boundary')
        path=root/prefix/first['file'];source.verify_identity(path,first['identity'])
        with np.load(path,allow_pickle=False) as z:
            values={k:torch.from_numpy(z[k].copy()) for k in first['tensors']}
            if z['index'].tolist()!=ids:raise ValueError('Prior input IDs differ')
        for key,value in values.items():
            if tensor_hash(value)!=first['tensors'][key]['sha256'] or not bool(torch.isfinite(value).all()):
                raise ValueError('Prior input tensor hash/finite mismatch')
        expected_shapes={'noisy_local':(2,8192,1,1),'conditioning_style':(2,128,1,1),'prior_t':(2,)}
        if any(values[k].shape!=shape or values[k].dtype!=torch.float32 for k,shape in expected_shapes.items()):
            raise ValueError('Actual prior shape/dtype mismatch')
        if not bool((values['prior_t']==first['timestep']+1).all()):raise ValueError('Actual prior timestep differs')
        initial=diagnostic['trajectory']['initial_hashes']
        if tensor_hash(values['noisy_local'])!=initial['noisy_local'] or tensor_hash(values['conditioning_style'])!=initial['style']:
            raise ValueError('Initial noisy local/optimized condition differs')
        state=paired_rng(diagnostic['paired_state']['path'],diagnostic['paired_state'],diagnostic['trajectory']['initial_rng_sha256'],first['rng_before_sha256'])
        first_output=bundle.read(prefix+'native_trace_loop01_scheduler.npz')
        import io
        with np.load(io.BytesIO(first_output),allow_pickle=False) as z:archived=torch.from_numpy(z['noise_pred'].copy())
        return dict(config=config,inputs=values,rng=state,rng_sha=first['rng_before_sha256'],archived_output=archived,
                    input_identity=identity(path),paired=diagnostic['paired_state'],ids=ids)
    finally:bundle.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture-gaussian','capture-gaussian-sha256','capture-background','capture-background-sha256',
                 'asset-root','asset-manifest-sha256','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv); output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    # Import model/runtime helpers only from the immutable deployed source tree.
    frozen=Path(args.asset_root)/'source_snapshot';sys.path.insert(0,str(frozen))
    from scripts import eval_multibackbone_source as source, eval_multibackbone_guidance as shared
    from scripts.diagnose_decoded_spectral import apply_backend_policy,verify_setup
    from legacy_trajectory import _rng_state,_restore_rng,rng_hash
    report=dict(status='running',operator_source=identity(__file__),cases={},forward_calls_planned=6,backward_calls=0,
                scope='Fixed loop1 local-prior repeat only; no decoder, classifier, Chamfer forward or trajectory')
    lion,before,entry=None,None,None
    try:
        captures={c:load_capture(getattr(args,'capture_'+c),getattr(args,'capture_'+c+'_sha256'),c) for c in ('gaussian','background')}
        config=captures['gaussian']['config']
        for key in ('native_extensions','native_bootstrap','executed_source_files','lion_state_before','resolved_lion_config','runtime','asset_manifest_sha256'):
            if captures['background']['config'][key]!=config[key]:raise ValueError('Captured runtime identities differ: '+key)
        if config['asset_manifest_sha256']!=args.asset_manifest_sha256:raise ValueError('Pinned asset manifest differs')
        report['backend']=apply_backend_policy(config['backend_policy'])
        if not torch.cuda.is_available() or str(torch.__version__)!='2.1.2+cu121' or torch.version.cuda!='12.1':
            raise RuntimeError('Requires actual pinned native CUDA runtime; no CPU/toy production fallback')
        torch.cuda.get_rng_state_all();entry=_rng_state();report['entry_rng_before_sha256']=rng_hash()
        manifest=source.read_manifest(Path(args.asset_root),args.asset_manifest_sha256)
        report['setup']=verify_setup(Path(args.asset_root),manifest,config)
        for name,expected in config['executed_source_files'].items():
            if name.startswith('repo/'):path=source.asset_path(frozen,name[5:])
            elif name.startswith('assets/'):path=source.asset_path(Path(args.asset_root),name[7:])
            else:raise ValueError('Unknown executed source origin')
            source.verify_identity(path,expected)
        report['executed_source_guards_passed']=True
        runtime_config=dict(config,method=shared.ORIGINAL_METHOD,asset_root=args.asset_root)
        lion,graph,_,_,native,resolved=shared.load_guidance_runtime(runtime_config,manifest)
        if graph is not None or native.get('no_build') is not True or resolved!=config['resolved_lion_config']:
            raise ValueError('Wrong original raw/eval/no-build runtime')
        if any(native.get(k)!=config['native_bootstrap'].get(k) for k in ('bindings','native_audit_sha256')):
            raise ValueError('Native bootstrap binding differs')
        from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        from pointnet2_ops import pointnet2_utils
        for name,expected in config['native_extensions'].items():
            module=sys.modules.get(name)
            if module is None or Path(module.__file__).resolve()!=Path(expected['path']).resolve():raise ValueError('Loaded native path differs: '+name)
            source.verify_identity(Path(expected['path']),expected)
        report['native_guards_passed']=True
        before=dict(vae=source.state_hash(lion.vae),priors=source.state_hash(lion.priors))
        if before!=config['lion_state_before']:raise ValueError('Raw model state differs')
        if any(m.training for model in (lion.vae,lion.priors) for m in model.modules()) or any(p.requires_grad for model in (lion.vae,lion.priors) for p in model.parameters()):raise ValueError('Models must be eval and frozen')
        actual=dict(torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        if any(actual[k]!=config['runtime'][k] for k in actual):raise ValueError('Runtime version/hardware differs')
        if any(report['backend'][k]!=config['runtime'][k] for k in report['backend']):raise ValueError('Actual backend differs')
        report.update(native=native,model_before=before,runtime=actual)
        write(output/'runtime.json',report)
        for corruption,capture in captures.items():
            inputs=capture['inputs'];directory=output/corruption
            result=repeat_prior(lion.priors[1],inputs['noisy_local'].cuda(),inputs['conditioning_style'].cuda(),inputs['prior_t'].cuda(),capture['rng'],capture['rng_sha'],directory)
            with np.load(directory/'repeat0_output.npz',allow_pickle=False) as z:fresh=torch.from_numpy(z['noise_pred'].copy())
            result['archived_output_difference']=difference(capture['archived_output'],fresh)
            result.update(archived_input=capture['input_identity'],paired_cache=capture['paired'],rng_sha256=capture['rng_sha'])
            write(directory/'summary.json',result);report['cases'][corruption]=result;write(output/'report.json',report)
            print('PRIOR_REPEAT_CASE_COMPLETE',corruption,flush=True)
        report['status']='complete'
    except BaseException:
        report.update(status='failed',failure=traceback.format_exc());raise
    finally:
        if lion is not None and before is not None:
            after=dict(vae=source.state_hash(lion.vae),priors=source.state_hash(lion.priors));report['model_after']=after
            report['model_unchanged']=after==before
            if after!=before:report['status']='failed_model_state_changed'
        if entry is not None:
            _restore_rng(entry);report['entry_rng_after_sha256']=rng_hash()
            report['entry_rng_restored']=report['entry_rng_after_sha256']==report['entry_rng_before_sha256']
            if not report['entry_rng_restored']:report['status']='failed_rng_changed'
        write(output/'report.json',report)
        write(output/'artifact_manifest.json',{str(p.relative_to(output)):identity(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    return report


if __name__=='__main__':main()
