#!/usr/bin/env python3
"""Bounded five-arm terminal/PM pilot. Production baseline defaults are unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = dict(T5=dict(steps=5, gamma=.01, eta=.01, cheap=False, pm=False),
            P5=dict(steps=5, cheap=True, pm=False),
            T8=dict(steps=8, gamma=.03, eta=.003, cheap=False, pm=False),
            P8=dict(steps=8, cheap=True, pm=False),
            **{'P8-PM': dict(steps=8, cheap=True, pm=True)})
CANDIDATES = ['clean_02','clean_04','clean_06','terminal']


def registration():
    import numpy as np
    from research_artifacts import CORRUPTIONS
    permutation = np.random.default_rng(20261108).permutation(2468).tolist()
    return dict(schema_version=1, dataset='modelnet40_c', severity=5, backbone='pointmae',
                selection_seed=20261108, smoke_ids=sorted(permutation[:4]), pilot_ids=sorted(permutation[4:36]),
                inference_seed=0, operational_repeats=3, batch_size=32, smoke_batch_size=2,
                corruptions=list(CORRUPTIONS), smoke_corruptions=['gaussian','background','density','lidar'],
                arms=ARMS, tau=.1, retained=.95, warmups=2, budget_seconds=2700,
                bootstrap_draws=10000, bootstrap_seed=20261108,
                accuracy_primary_repeat=0, reduction_precision='float64 offline; archived GPU PM float32 separately',
                exposure='Official test development data; neither unseen confirmation nor noninferiority proof')


def validate_registration(value):
    if value != registration():
        raise ValueError('Pilot registration differs from the frozen protocol')


def prepare(raw_array, lion, ops, device='cuda'):
    """Shared fresh preprocessing; encoding is owned by the selected sampler."""
    import numpy as np
    import torch
    raw = torch.from_numpy(np.array(raw_array, dtype=np.float32, copy=True))
    normalized, center, scale = ops.normalize(raw)
    prepared = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().to(device)
    return ops.rotate_pointcloud(prepared * 3.3885), center, scale


def encode(prepared, lion):
    import torch
    from legacy_trajectory import _rng_state
    with torch.no_grad():
        latents = lion.vae.encode(prepared)
        z, h = (latents[2][i][0].unsqueeze(2).unsqueeze(3) for i in (0,1))
        noise = torch.randn_like(h)
    return dict(shape_latent=z, local_latent=h, noise=noise, rng_after_prepare=_rng_state())


def infer(raw_array, arm, lion, original, ops, classify_geometry, models, *, cached=None, audit=None, device='cuda'):
    """One fresh operational inference, or an explicitly untimed same-state smoke."""
    import torch
    from legacy_trajectory import isolated_probe
    from lion_guidance_pilot import sample
    from scripts.eval_multibackbone_guidance import decoded_to_geometry
    from scripts.run_lion_guidance_speed import pm_readout
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    policy = ARMS[arm]
    prepared, center, scale = prepare(raw_array, lion, ops, device) if cached is None else cached[:3]
    def classify(decoded):
        return classify_geometry(decoded_to_geometry(decoded, 'modelnet40_c', scale, center, ops))
    if not policy['cheap']:
        options = {} if cached is None else dict(prepared_state=cached[3])
        if audit is not None:
            options['scheduler_observer'] = lambda s:audit.update(timesteps=s.timesteps[-policy['steps']:].cpu().tolist())
        decoded = original(prepared, lion, policy['steps'], policy['gamma'], policy['eta'], .95, **options)
        with isolated_probe(models):
            logits = classify(decoded).detach()[:,None,:]
    else:
        state = encode(prepared, lion) if cached is None else cached[3]
        saved = []
        def decode(h,z):
            return lion.vae.decoder(None,beta=None,context=h.reshape(len(h),-1),style=z.reshape(len(z),-1))
        def readout(h,z,loop):
            saved.append(pm_readout(h,z,loop == policy['steps'],decode,classify,models))
        observer = None if audit is None else lambda s:audit.update(timesteps=s.timesteps[-policy['steps']:].cpu().tolist())
        snapshots, records = sample(lion,state,policy['steps'],'cheap',tau=.1,
            terminal_only=not policy['pm'],callback=readout,scheduler_observer=observer)
        if audit is not None:
            audit.update(snapshots=len(snapshots), updates=len(records), backward_steps=sum(r['full_backward'] for r in records))
        logits = torch.stack(saved,1)
    return logits, logits.softmax(-1).mean(1)


def validate_executed_source(executed, manifest):
    """A snapshot hash cannot excuse running a different checkout's code."""
    for name,observed in executed.items():
        prefix,relative=name.split('/',1)
        declared='source_snapshot/'+relative if prefix=='repo' else relative
        expected=manifest['files'].get(declared)
        if expected is None or any(observed[k]!=expected[k] for k in ('bytes','sha256')):
            raise ValueError('Executed source is not the sealed source: '+name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', type=Path)
    for name in ('asset-root','manifest-sha256','registration','output','progress-drive','smoke-gate'):
        parser.add_argument('--'+name)
    parser.add_argument('--phase',choices=('smoke','pilot'))
    args = parser.parse_args(argv)
    from scripts.run_lion_guidance_speed import write, checkpoint_case, checkpoint_integrity, arm_order, iter_batches
    if args.register:
        args.register.parent.mkdir(parents=True,exist_ok=True)
        write(args.register, registration())
        return
    if not all(getattr(args,n) for n in ('asset_root','manifest_sha256','registration','output','progress_drive','phase')):
        parser.error('Execution requires assets, registration, output, Drive progress and phase')
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state, _restore_rng
    from scripts import eval_multibackbone_source as source, eval_multibackbone_guidance as shared
    from scripts.diagnose_decoded_spectral import configuration, apply_backend_policy, verify_setup
    from scripts.prepare_chamfer_variants_assets import identity
    # Load lazy scientific code before capturing the executed source inventory.
    from lion_guidance_pilot import sample
    reg = json.loads(Path(args.registration).read_text())
    validate_registration(reg)
    asset, output, progress = map(Path,(args.asset_root,args.output,args.progress_drive))
    if not __import__('os').path.ismount('/content/drive') or Path('/content/drive/MyDrive') not in progress.parents:
        raise ValueError('Verified Drive mount and fresh task checkpoint path required')
    if args.phase == 'pilot':
        if not args.smoke_gate:
            raise ValueError('Pilot requires a completed technical smoke gate')
        gate = json.loads(Path(args.smoke_gate).read_text())
        if (gate['status'] != 'technical_pass' or gate['registration'] != identity(args.registration)
                or gate['manifest_sha256'] != args.manifest_sha256):
            raise ValueError('Smoke gate/source/registration mismatch')
    output.mkdir(parents=True,exist_ok=False)
    progress.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',dict(vars(args),run_id=output.name,protocol=reg))
    (output/'command.txt').write_text(__import__('shlex').join(sys.argv)+'\n')
    started = time.monotonic()
    try:
        manifest = source.read_manifest(asset,args.manifest_sha256)
        corruptions = reg['smoke_corruptions'] if args.phase == 'smoke' else reg['corruptions']
        ids = reg['smoke_ids'] if args.phase == 'smoke' else reg['pilot_ids']
        size = 2 if args.phase == 'smoke' else 32
        base = configuration(['--asset-root',str(asset),'--asset-manifest-sha256',args.manifest_sha256,
            '--result-root',str(output),'--paired-state-root',str(output/'unused'),'--arm','control',
            '--count','2','--backend-policy','controlled','--corruptions',*corruptions])
        base.update(batch_size=size,selected_original_ids=ids,count=len(ids))
        backend = apply_backend_policy('controlled')
        setup = verify_setup(asset,manifest,base)
        dataset = source.dataset_manifest(manifest,base)
        base['dataset_root'] = dataset['root']
        arrays = {c:source.load_array(Path(dataset['root'])/('data_'+c+'_5.npy'),
            dataset['files']['data_'+c+'_5.npy'],base) for c in corruptions}
        labels = source.load_array(Path(dataset['root'])/'label.npy',dataset['files']['label.npy'],base,labels=True)
        lion,_,original,ops,native,resolved = shared.load_guidance_runtime(dict(base,method=shared.ORIGINAL_METHOD),manifest)
        classifier = source.load_classifier(base,manifest)
        from pointnet2_ops import pointnet2_utils
        from types import SimpleNamespace
        captured={"asset_root":str(asset)}
        source.snapshot_sources(SimpleNamespace(path=output),captured,manifest)
        executed_source=captured['executed_source_files']
        validate_executed_source(executed_source,manifest)
        native_extensions=captured['native_extensions']
        models = [lion.vae,lion.priors,classifier]
        hashes = [source.state_hash(m) for m in models]
        if any(m.training for model in models for m in model.modules()) or any(p.requires_grad for m in models for p in m.parameters()):
            raise ValueError('All weights and evaluation modes must be frozen')
        if 'A100' not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory < 70*1024**3:
            raise ValueError('Pilot requires the requested High-RAM A10080 runtime')
        if args.phase == 'pilot' and gate['model_hashes'] != hashes:
            raise ValueError('Smoke model identity changed')
        if args.phase == 'pilot' and (gate['source_snapshot_sha256'] != manifest['source_snapshot_sha256']
                or gate['native_extensions'] != native_extensions):
            raise ValueError('Smoke source/native identity changed')
        write(output/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            native=native,backend=backend,setup=setup,dataset=dataset,model_hashes=hashes,lion_config=resolved,
            manifest_sha256=args.manifest_sha256,registration=identity(args.registration),
            source_snapshot_sha256=manifest['source_snapshot_sha256'],executed_source_files=executed_source,
            native_extensions=native_extensions,source_overlay=manifest.get('source_overlay_receipt')))
        (output/'environment.txt').write_text((output/'environment.json').read_text())
        def fps(points,n):
            idx=pointnet2_utils.furthest_point_sample(points.contiguous(),n)
            return pointnet2_utils.gather_operation(points.transpose(1,2).contiguous(),idx).transpose(1,2).contiguous()
        def classify(geometry):
            points,features=source.preprocess(geometry,base,fps)
            return classifier(points,features=features)
        def run(raw,arm,**kw):return infer(raw,arm,lion,original,ops,classify,models,**kw)
        repeat_reports, durations = [], []
        checkpoint_case(output,progress/'header')
        for ci,corruption in enumerate(corruptions):
            case = output/corruption
            case.mkdir()
            warm_ids=(reg['smoke_ids']*((size+3)//4))[:size]
            for warm in range(2):
                for arm in arm_order(ARMS,ci*2+warm):
                    shared.seed_inference(90000+ci)
                    values=run(arrays[corruption][warm_ids],arm)
                    torch.cuda.synchronize()
                    del values
                print('TERMINAL_PILOT_WARMUP',corruption,warm+1,flush=True)
            for bi,batch_ids in iter_batches(ids,size):
                raw=arrays[corruption][batch_ids]
                cached = None
                if args.phase == 'smoke':
                    shared.seed_inference(ci*1000+bi*10)
                    prepared,center,scale=prepare(raw,lion,ops)
                    cached=(prepared,center,scale,encode(prepared,lion))
                first={}
                for repeat in range(3):
                    draw_seed=ci*1000+bi*10
                    for position,arm in enumerate(arm_order(ARMS,ci*3+repeat+bi)):
                        shared.seed_inference(draw_seed)
                        audit={} if args.phase=='smoke' else None
                        counts=dict(prior=0,decoder=0,head=0)
                        hooks=[]
                        if audit is not None:
                            def counter(key):
                                def hook(*unused):counts[key]+=1
                                return hook
                            for key,model in (('prior',lion.priors[1]),('decoder',lion.vae.decoder),('head',classifier)):
                                hooks.append(model.register_forward_hook(counter(key)))
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        tick=time.perf_counter()
                        try:logits,pm=run(raw,arm,cached=cached,audit=audit)
                        finally:
                            for hook in hooks:hook.remove()
                        torch.cuda.synchronize()
                        seconds=time.perf_counter()-tick
                        peak=torch.cuda.max_memory_allocated()/1024**2
                        reserved=torch.cuda.max_memory_reserved()/1024**2
                        policy=ARMS[arm]
                        if logits.shape != (len(batch_ids),4 if policy['pm'] else 1,40) or not torch.isfinite(logits).all():
                            raise ValueError('Nonfinite or malformed class logits')
                        saved=dict(logits=logits.cpu().numpy(),probability_mean=pm.cpu().numpy(),prediction=pm.argmax(-1).cpu().numpy(),
                            index=np.asarray(batch_ids),label=np.asarray(labels[batch_ids]).reshape(-1),
                            candidates=np.asarray(CANDIDATES if policy['pm'] else ['terminal']))
                        meta=dict(seconds=seconds,peak_mib=peak,reserved_mib=reserved,order=position,
                            draw_seed=draw_seed,inference_seed=0,operational_repeat=repeat,ids=batch_ids,audit=audit,counts=counts)
                        if audit is not None:
                            expected=4 if policy['pm'] else 1
                            if counts != dict(prior=policy['steps'],decoder=expected,head=expected) or audit['timesteps'] != list(range((policy['steps']-1)*10,-1,-10)):
                                raise ValueError('Actual scheduler/readout count gate failed')
                            if policy['cheap'] and (audit['updates']!=policy['steps'] or audit['backward_steps']!=0 or audit['snapshots']!=expected):
                                raise ValueError('Cheap update/snapshot gate failed')
                            if arm in first:
                                old=first[arm]
                                diff=saved['logits'].astype(np.float64)-old['logits']
                                repeat_reports.append(dict(corruption=corruption,batch=bi,arm=arm,repeat=repeat,
                                    logit_max_abs=float(abs(diff).max()),logit_rms=float(np.sqrt((diff**2).mean())),
                                    prediction_changes=int((saved['prediction']!=old['prediction']).sum()),
                                    fixed_tolerance=1e-5,passed=bool(abs(diff).max()<=1e-5)))
                            else:first[arm]=saved
                        stem='%s_b%d_r%d'%(arm,bi,repeat)
                        np.savez_compressed(case/(stem+'.npz'),**saved)
                        write(case/(stem+'.json'),meta)
                        durations.append(seconds)
                        del logits,pm,saved
                        print('TERMINAL_PILOT_PROGRESS',args.phase,corruption,bi,repeat,arm,round(seconds,3),flush=True)
            checkpoint_case(case,progress/corruption)
            checkpoint_integrity(models,hashes,output/'integrity'/corruption,progress/'integrity'/corruption)
            if args.phase=='pilot' and time.monotonic()-started > reg['budget_seconds']:
                raise TimeoutError('45 minute scientific soft budget reached after closed corruption; no expansion/resume')
        for name,expected in executed_source.items():
            prefix,relative=name.split('/',1)
            path=asset/relative if prefix=='assets' else source.REPO/relative
            source.verify_identity(path,expected)
        if args.phase=='smoke':
            write(output/'smoke_gate.json',dict(status='technical_pass',registration=identity(args.registration),
                manifest_sha256=args.manifest_sha256,model_hashes=hashes,native_repeat_reports=repeat_reports,
                source_snapshot_sha256=manifest['source_snapshot_sha256'],native_extensions=native_extensions,
                repeat_passes=sum(r['passed'] for r in repeat_reports),repeat_cases=len(repeat_reports),
                interpretation='Native repeat failure prohibits causal/parity claims; descriptive fresh-repeat pilot only',
                smoke_seconds=time.monotonic()-started))
        write(output/'completion.json',dict(status='complete',phase=args.phase,seconds=time.monotonic()-started,
            cases=len(durations),registration=identity(args.registration)))
        (output/'notes.md').write_text('Official-test development pilot, inference seed0, three operational repeats.\n'
            'Timing: CPU raw batch ready through fresh preprocessing/encode/sampling/required readouts/GPU PM; CUDA synchronized.\n'
            'Excluded equally: data/model load, warmup, seeding, hashes, D2H audit, NPZ/Drive I/O. Smoke uses cached state and hooks; its time is diagnostic.\n'
            'Native repeat floor is separate; equal seed does not imply equal encoded states. B32 amortized ms is not B1 latency.\n')
        print('TERMINAL_PILOT_PHASE_COMPLETE',args.phase,flush=True)
    except BaseException:
        write(output/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
