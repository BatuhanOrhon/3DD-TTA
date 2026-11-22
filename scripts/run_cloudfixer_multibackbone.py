#!/usr/bin/env python3
"""Original author CloudFixer geometry, shared by three accepted frozen heads."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import signal
import threading
import time
import zipfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import Budget, MountStore, canonical, commit_unit, load_units, sha, unit_key
from research_artifacts import CORRUPTIONS
from scripts.run_cloudfixer_point2vec import CF_OPTIONS, rng_state, restore_rng
from scripts import eval_multibackbone_source as source

HEADS = ('dgcnn', 'curvenet', 'pointnext')
METHODS = ('source_raw', 'source_cf_frame', 'cloudfixer_plain')


def units(campaign, phase, batch_size=32):
    if type(batch_size) is not int or batch_size not in (32,64):
        raise ValueError('Registered batch size must be32 or64')
    if phase not in ('full', 'smoke', 'profile'):
        raise ValueError('Unknown registered phase')
    corruptions = list(CORRUPTIONS) if phase == 'full' else ['gaussian'] if phase == 'profile' else ['background', 'rotation', 'gaussian']
    ids = list(range(2468)) if phase == 'full' else list(range(batch_size if phase == 'profile' else 8))
    size=4 if phase=='smoke' else batch_size
    return [dict(campaign=campaign, phase=phase, corruption=c, seed=0,
                 ids=ids[i:i+size], batch_size=size)
            for c in corruptions for i in range(0, len(ids), size)]


def seed_for(unit):
    # Classifier-only substream; native geometry uses the continuous corruption seed0.
    return int(sha(canonical({k: unit[k] for k in ('corruption', 'seed', 'ids')}))[:8], 16)


def require_drive(folder):
    from scripts.setup_cloudfixer_multibackbone_colab import require_task_drive
    return require_task_drive(folder)


def validate_full_gate(gate, campaign, profile_sha, *, now=None):
    now = time.time() if now is None else now
    fields = ('account_read_epoch','balance_cu','all_assignments_cu_per_h','remaining_eta_seconds','execution_window_seconds','reserve_cu')
    if any(type(gate.get(k)) not in (int,float) or not math.isfinite(gate[k]) for k in fields):
        raise ValueError('Finite current resource receipt required')
    if (gate.get('status')!='passed' or gate.get('campaign')!=campaign
            or gate.get('profile_payload_sha256')!=profile_sha
            or not 0 <= now-gate['account_read_epoch'] <= 600
            or gate['reserve_cu']<5 or gate['all_assignments_cu_per_h']<=0
            or gate['remaining_eta_seconds']<=0 or gate['execution_window_seconds']<120
            or gate['execution_window_seconds']-(now-gate['account_read_epoch'])<120
            or gate['execution_window_seconds']>gate['remaining_eta_seconds']
            or gate['balance_cu'] < gate['reserve_cu'] + gate['all_assignments_cu_per_h']*gate['execution_window_seconds']/3600):
        raise ValueError('Profile/current shared-balance gate failed')


def read_full_gate(store, key, campaign, profile_sha, *, now=None):
    if not isinstance(key,str) or not re.fullmatch('full_gate_[0-9a-f]{64}.json',key):
        raise ValueError('Attempt-specific immutable resource receipt key required')
    data=store.get(key)
    if key!='full_gate_'+sha(data)+'.json':
        raise ValueError('Resource receipt key/content SHA mismatch')
    gate=json.loads(data)
    validate_full_gate(gate,campaign,profile_sha,now=now)
    return sha(data)


def validate(data, unit, labels):
    import numpy as np
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if len(z.namelist()) != len(set(z.namelist())) or sum(v.file_size for v in z.infolist()) > 100_000_000:
            raise ValueError('Unsafe prediction payload')
    with np.load(io.BytesIO(data), allow_pickle=False) as p:
        values = {k: p[k].copy() for k in p.files}
    n = len(unit['ids'])
    if (json.loads(str(values['unit'])) != unit or not np.array_equal(values['ids'], unit['ids'])
            or values['labels'].shape != (n,) or values['labels'].dtype.kind not in 'iu'
            or not np.array_equal(values['labels'], labels[unit['ids']])
            or int(values['denoiser_calls']) != 30):
        raise ValueError('Prediction identity/labels/native call count mismatch')
    for name in ('source_logits', 'frame_logits', 'cf_logits'):
        logits = values[name]
        if logits.shape != (n, 3, 40) or not np.isfinite(logits).all():
            raise ValueError('Invalid classifier predictions: '+name)
    if (values['seconds'].shape != (3,) or not np.isfinite(values['seconds']).all()
            or not (values['seconds'] > 0).all()
            or not math.isfinite(float(values['peak_mib'])) or float(values['peak_mib']) < 0):
        raise ValueError('Invalid measured timing/memory')
    if ('head_seconds' in values and (values['head_seconds'].shape != (3, 3)
            or not np.isfinite(values['head_seconds']).all() or (values['head_seconds'] < 0).any())):
        raise ValueError('Invalid head timing')
    for key in ('input_sha256', 'output_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(values[key])):
            raise ValueError('Invalid geometry identity')
    if (not re.fullmatch('[0-9a-f]{64}',str(values['rng_before_sha256']))
            or set(json.loads(str(values['rng_after']))) != {'python','numpy','cpu','cuda'}):
        raise ValueError('Full native RNG continuation required')
    return values


def recover(store, campaign, wanted, labels, ignored=()):
    class Remaining:
        def keys(self,prefix):return [k for k in store.keys(prefix) if k not in ignored]
        def get(self,key):return store.get(key)
    records = load_units(Remaining(), campaign)
    expected = {unit_key(u): u for u in wanted}
    if not set(records) <= set(expected):
        raise ValueError('Unexpected recovered work')
    for key, record in records.items():
        record['values'] = validate(record['data'], expected[key], labels)
    for corruption in {u['corruption'] for u in wanted}:
        keys = [unit_key(u) for u in wanted if u['corruption']==corruption]
        recovered = [k for k in keys if k in records]
        if recovered != keys[:len(recovered)]:
            raise ValueError('Recovered native stream must be a contiguous prefix')
    return records


def completed_corruptions(store,campaign):
    result={}
    for key in store.keys('completed_'):
        receipt=json.loads(store.get(key))
        corr=receipt['corruption']
        data=store.get(receipt['archive'])
        if (key!='completed_'+corr+'.json' or corr not in CORRUPTIONS
                or receipt['campaign']!=campaign or receipt['objects']!=2468
                or receipt['heads']!=list(HEADS) or sha(data)!=receipt['archive_sha256']):
            raise ValueError('Completed corruption archive/cursor mismatch')
        result[corr]=receipt
    return result


def report_rows(records):
    import numpy as np
    groups = {}
    for r in records:
        u, p = r['unit'], r['values']
        for mode, (method, key) in enumerate(zip(METHODS, ('source_logits', 'frame_logits', 'cf_logits'))):
            for head, name in enumerate(HEADS):
                g = groups.setdefault((method, name, u['corruption']),
                    dict(method=method, backbone=name, corruption=u['corruption'], n_examples=0,
                         n_correct=0, runtime_seconds=0., reconstruction_seconds_shared=0., peak_gpu_memory_mb=0.))
                g['n_examples'] += len(u['ids'])
                g['n_correct'] += int((p[key][:, head].argmax(-1) == p['labels']).sum())
                native = float(p['seconds'][2]) if mode == 2 else 0.
                g['reconstruction_seconds_shared'] += native
                g['runtime_seconds'] += native + float(p.get('head_seconds', np.zeros((3, 3)))[mode, head])
                g['peak_gpu_memory_mb'] = max(g['peak_gpu_memory_mb'], float(p['peak_mib']))
    return [dict(v, accuracy=v['n_correct']/v['n_examples'], seed=0, severity=5,
                 dataset='modelnet40_c', status='complete' if v['n_examples']==2468 else 'partial')
            for v in groups.values()]


def write_reports(output, records, campaign):
    rows = report_rows(records)
    if not rows:
        return
    def write(name, records_):
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_id']+list(records_[0]))
            writer.writeheader(); writer.writerows(dict(run_id=campaign, **r) for r in records_)
    write('per_corruption.csv', rows)
    summary = []
    for method in METHODS:
        for head in HEADS:
            group = [r for r in rows if r['method']==method and r['backbone']==head]
            count = sum(r['n_examples'] for r in group); correct = sum(r['n_correct'] for r in group)
            summary.append(dict(dataset='modelnet40_c', severity=5, method=method, backbone=head,
                seed=0, n_corruptions=len(group), total_examples=count, total_correct=correct,
                micro_accuracy=correct/count, macro_accuracy=sum(r['accuracy'] for r in group)/len(group),
                total_runtime_seconds=sum(r['runtime_seconds'] for r in group),
                status='complete' if len(group)==15 and count==37020 else 'partial'))
    write('summary.csv', summary)


def checked_runtime(args, spec):
    import numpy as np
    import torch
    from scripts.setup_cloudfixer_multibackbone_colab import gpu_architecture,preload_head_native
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Single registered CUDA device required')
    accelerator=spec.get('accelerator','A100')
    gpu_architecture(accelerator,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
    setup = json.loads(Path('/content/cli_setup_verification.json').read_text())
    if setup.get('status') != 'passed' or sys.version_info[:2] != (3, 8):
        raise ValueError('Canonical10 setup and pinned Python3.8 required')
    for name, identity in spec['files'].items():
        source.verify_identity(source.asset_path(args.asset_root, name), identity)
    for name, identity in spec['official_files'].items():
        source.verify_identity(source.asset_path(args.official_root, name), identity)
    for name, identity in spec['overlay_files'].items():
        source.verify_identity(source.asset_path(ROOT, name), identity)
    for filename, digest in spec['dataset_files'].items():
        path = args.dataset_root/filename
        if source.file_identity(path)['sha256'] != digest:
            raise ValueError('Dataset identity mismatch: '+filename)
    if source.file_identity(args.diffusion)['sha256'] != spec['diffusion_sha256']:
        raise ValueError('Diffusion checkpoint identity mismatch')
    os.chdir(args.official_root); sys.path.insert(0, str(args.official_root))
    import adapt
    from cloudfixer_diffusion import verify_gamma
    from utils.pc_utils import scale_to_unit_cube_torch, rotate_shape_tensor
    from pointnet2_ops import pointnet2_utils
    from types import SimpleNamespace
    model_args = SimpleNamespace(model='transformer', probabilistic_model='diffusion', diffusion_steps=500,
        diffusion_noise_schedule='polynomial_2', diffusion_noise_precision=1e-5, diffusion_loss_type='l2')
    model = adapt.get_model(model_args, torch.device('cuda'))
    intended = model.gamma.gamma.detach().clone()
    model.load_state_dict(torch.load(args.diffusion, map_location='cpu'), strict=True)
    gamma = verify_gamma(model.gamma.gamma, intended)
    model = model.cuda().eval().requires_grad_(False)
    preload_head_native(args.asset_root.parent,accelerator)
    heads = {}
    for name in HEADS:
        config = dict(spec['head_configs'][name], asset_root=str(args.asset_root))
        heads[name] = source.load_classifier(config, spec)
        if source.state_hash(heads[name]) != spec['head_state_hashes'][name]:
            raise ValueError('Loaded head differs from accepted source: '+name)
    options = SimpleNamespace(**dict(CF_OPTIONS, rotate=True, dataset='modelnet40c_gaussian_5',
        dataset_dir=str(args.dataset_root), random_seed=0, batch_size=spec['scientific_protocol']['batch_size']))
    def fps(points, number):
        points = points.contiguous()
        idx = pointnet2_utils.furthest_point_sample(points, number)
        return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    def geometry(points):
        return rotate_shape_tensor(scale_to_unit_cube_torch(points), 'x', np.pi/2).contiguous()
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    # Force lazy native loads before sealing the runtime signature.
    from pytorch3d.ops import knn_points
    probe=(torch.arange(96,device='cuda',dtype=torch.float32).reshape(1,32,3)/96).requires_grad_(True)
    d1,d2,_,_=adapt.chamfer_dist_fn(probe,probe.detach()+.01)
    (d1.mean()+d2.mean()).backward()
    if not torch.isfinite(probe.grad).all() or not torch.isfinite(knn_points(probe.detach(),probe.detach(),K=5).dists).all():
        raise ValueError('Official CF native forward/backward probe failed')
    if fps(probe.detach(),8).shape!=(1,8,3):
        raise ValueError('Matched classifier FPS native probe failed')
    torch.cuda.synchronize()
    return adapt, model, heads, options, fps, geometry, gamma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','asset-root','official-root','diffusion','dataset-root','drive-folder','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke','profile','full'), required=True)
    parser.add_argument('--full-gate-key')
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text()); campaign = sha(canonical(spec))
    batch_size=spec['scientific_protocol']['batch_size']
    args.drive_folder=require_drive(args.drive_folder)
    store = MountStore(args.drive_folder)
    store.put('task.json', canonical(spec))
    if args.phase != 'smoke':
        gate = json.loads(store.get('smoke_gate.json'))
        if gate.get('campaign') != campaign or gate.get('status') != 'passed':
            raise ValueError('Current campaign smoke gate required')
    full_gate_sha=None
    if args.phase=='full':
        profile=load_units(MountStore(store.folder/'profile'),campaign)
        profile_key=unit_key(units(campaign,'profile',batch_size)[0])
        if set(profile)!={profile_key}:
            raise ValueError('Exactly one current registered-batch profile required')
        import numpy as np
        validate(profile[profile_key]['data'],units(campaign,'profile',batch_size)[0],
                 np.load(args.dataset_root/'label.npy',allow_pickle=False).reshape(-1))
        full_gate_sha=read_full_gate(store,args.full_gate_key,campaign,profile[profile_key]['sha256'])
    if args.output.exists():
        raise FileExistsError('Fresh output required; recovery uses immutable Drive units')
    args.output.mkdir(parents=True); (args.output/'raw').mkdir()
    for name, data in [('command.txt', canonical(sys.argv)), ('config.json', canonical(dict(manifest=spec, phase=args.phase, campaign=campaign,full_gate_key=args.full_gate_key,full_gate_sha256=full_gate_sha)))]:
        (args.output/name).write_bytes(data)
    (args.output/'notes.md').write_text('Frozen accepted alternative heads. Original author geometry; plainCF/vote1.\n'
        'Geometry input: author rotateTrue/unitSTD/mask/ind. Output: author unit-sphere/inverse-axis, then accepted headFPS1024.\n'
        'Source_raw uses original arrays; source_cf_frame is separate. Different LION prior/output preprocessing and RNG: whole pipelines only.\n'
        'Native optimizer is nondeterministic. Shared geometry/head order is fixed; classifier RNG is forked.\n'
        'Peak field is MiB. Per-head cost includes the shared reconstruction; do not sum3heads as3independent CF runs.\n')
    import numpy as np
    import torch
    budget = Budget(store, cap=spec['budget_seconds'])
    if args.phase=='full':
        gate=json.loads(store.get(args.full_gate_key))
        store.put('used_'+full_gate_sha+'.json',canonical(dict(output=str(args.output.resolve()))))
        remaining_window=gate['execution_window_seconds']-(time.time()-gate['account_read_epoch'])
        budget.cap=min(budget.cap,budget.base+remaining_window)
    deadline=threading.Timer(max(.001,budget.remaining()),lambda:os.kill(os.getpid(),signal.SIGKILL))
    deadline.daemon=True;deadline.start()
    adapt, model, heads, options, fps, geometry, gamma = checked_runtime(args, spec)
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0), gamma=gamma,
        frozen_head_hashes={n:source.state_hash(h) for n,h in heads.items()},
        native_optimizer='non-deterministic KNN backward; no exact cross-VM equality claim')
    native = {};native_paths={}
    for name, module in sorted(sys.modules.copy().items()):
        path = str(getattr(module,'__file__','') or '')
        if path.endswith('.so') and any(s in path for s in ('pytorch3d','knn_cuda','pointnet2','chamfer','torch_extensions')):
            native[name]=source.file_identity(Path(path))
            native_paths[name]=path
    if not native:
        raise ValueError('Loaded native ELF identities required')
    environment['native_elf_identities']=native
    from scripts.setup_cloudfixer_multibackbone_colab import native_snapshot
    native_snapshot(store,native_paths)
    store.put('runtime_signature.json',canonical(environment))
    (args.output/'environment.txt').write_bytes(canonical(environment))
    labels = np.load(args.dataset_root/'label.npy', allow_pickle=False).reshape(-1)
    all_wanted = units(campaign, args.phase,batch_size); cache = MountStore(store.folder/args.phase)
    completed=completed_corruptions(store,campaign) if args.phase=='full' else {}
    wanted=[u for u in all_wanted if u['corruption'] not in completed]
    ignored={unit_key(u) for u in all_wanted if u['corruption'] in completed}
    accepted = recover(cache, campaign, wanted, labels,ignored)
    recovered_at_start=len(accepted)
    original_model = source.state_hash(model)
    current = None
    from torch.utils.data._utils.collate import default_collate
    def timed(fn):
        torch.cuda.synchronize(); start=time.perf_counter(); value=fn(); torch.cuda.synchronize()
        return value, time.perf_counter()-start
    def classify(points, seed):
        logits, seconds = [], []
        state=rng_state()
        try:
            with torch.no_grad():
                for i, name in enumerate(HEADS):
                    torch.manual_seed(seed+i); torch.cuda.manual_seed_all(seed+i)
                    def forward():
                        x, features = source.preprocess(points, spec['head_configs'][name], fps)
                        if args.phase!='smoke':
                            return heads[name](x, features=features)
                        reference_rng=rng_state()
                        wrapped=heads[name](x, features=features)
                        wrapped_rng=sha(canonical(rng_state()))
                        restore_rng(reference_rng)
                        direct=heads[name].model({'pos':x,'x':features}) if name=='pointnext' else heads[name].model(x)
                        if not torch.equal(wrapped,direct) or sha(canonical(rng_state()))!=wrapped_rng:
                            raise ValueError('Frozen source-wrapper/direct model or RNG parity failed: '+name)
                        return wrapped
                    values, elapsed = timed(forward); logits.append(values.cpu().numpy()); seconds.append(elapsed)
        finally:
            restore_rng(state)
        return np.stack(logits, axis=1), seconds
    for unit in wanted:
        key = unit_key(unit)
        if key not in accepted and args.phase=='full' and budget.remaining()<90:
            write_reports(args.output,list(accepted.values()),campaign)
            print('CF_MULTIBACKBONE_PARTIAL',len(accepted),len(wanted),'funded window ended; Drive receipts retained',flush=True)
            return
        if current != unit['corruption']:
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            options.dataset='modelnet40c_'+unit['corruption']+'_5'
            dataset=adapt.ModelNet40C(options, partition='test'); current=unit['corruption']
            raw=np.load(args.dataset_root/('data_'+current+'_5.npy'), mmap_mode='r', allow_pickle=False)
        before_sha=sha(canonical(rng_state()))
        if key in accepted:
            if str(accepted[key]['values']['rng_before_sha256'])!=before_sha:
                raise ValueError('Recovered native input RNG differs from continuous seed0 stream')
            restore_rng(json.loads(str(accepted[key]['values']['rng_after'])))
        if key not in accepted:
            with budget.operation(1800):
                seed = seed_for(unit)
                batch=default_collate([dataset[i] for i in unit['ids']])
                x, mask, indices = [batch[i].cuda() for i in (0,2,3)]
                if not np.array_equal(batch[1].numpy().reshape(-1), labels[unit['ids']]):
                    raise ValueError('Official loader original-ID labels disagree')
                raw_points=torch.as_tensor(np.asarray(raw[unit['ids']]).copy(), device='cuda', dtype=torch.float32)
                if args.phase=='smoke':
                    reference=raw_points-raw_points.mean(1,keepdim=True)
                    reference=reference/reference.norm(dim=-1,keepdim=True).amax(1,keepdim=True)
                    if not torch.allclose(geometry(x),reference,atol=2e-5,rtol=2e-5):
                        raise ValueError('Author/input/classifier coordinate round-trip failed')
                torch.cuda.reset_peak_memory_stats()
                classification_rng=sha(canonical(rng_state()))
                (src, src_s), source_wall = timed(lambda: classify(raw_points, seed+1000))
                (frame, frame_s), frame_wall = timed(lambda: classify(geometry(x), seed+1000))
                if sha(canonical(rng_state()))!=classification_rng:
                    raise ValueError('Classifier controls shifted geometry RNG')
                repaired, native_s=timed(lambda:adapt.cloudfixer(options, torch.nn.DataParallel(model), x, mask, indices).detach())
                classification_rng=sha(canonical(rng_state()))
                (cf, cf_s), _=timed(lambda: classify(geometry(repaired), seed+1000))
                if sha(canonical(rng_state()))!=classification_rng:
                    raise ValueError('Classifier after adaptation shifted native RNG')
                if source.state_hash(model)!=original_model or any(source.state_hash(h)!=spec['head_state_hashes'][n] for n,h in heads.items()):
                    raise ValueError('Frozen model state changed before unit commit')
                stream=io.BytesIO()
                np.savez_compressed(stream, unit=json.dumps(unit,sort_keys=True), ids=unit['ids'], labels=labels[unit['ids']],
                    source_logits=src, frame_logits=frame, cf_logits=cf,
                    seconds=[source_wall,frame_wall,native_s], head_seconds=[src_s,frame_s,cf_s],
                    peak_mib=torch.cuda.max_memory_allocated()/2**20, denoiser_calls=30,
                    input_sha256=sha(x.cpu().contiguous().numpy().tobytes()),
                    output_sha256=sha(repaired.cpu().contiguous().numpy().tobytes()),
                    rng_before_sha256=before_sha,rng_after=canonical(rng_state()).decode())
                data=stream.getvalue(); values=validate(data,unit,labels);commit_unit(cache,unit,data)
                accepted[key]=dict(unit=unit,data=data,values=values)
        record=accepted[key];receipt=json.loads(cache.get(key))
        (args.output/'raw'/key).write_bytes(canonical(receipt))
        (args.output/'raw'/receipt['payload']).write_bytes(record['data'])
        print('CF_MULTIBACKBONE_BLOCK',unit['corruption'],unit['ids'][0],len(accepted),len(wanted),flush=True)
        if unit==wanted[-1] or unit['ids'][-1]==2467:
            if source.state_hash(model)!=original_model or any(source.state_hash(h)!=spec['head_state_hashes'][n] for n,h in heads.items()):
                raise ValueError('Frozen model state changed')
            write_reports(args.output,list(accepted.values()),campaign)
            for name in ('summary.csv','per_corruption.csv'):
                store.put('report_'+args.output.name+'_'+str(len(accepted))+'_'+name,(args.output/name).read_bytes())
            if args.phase=='full':
                scoped=[r for r in accepted.values() if r['unit']['corruption']==unit['corruption']]
                scoped_dir=args.output/('scope_'+unit['corruption']);scoped_dir.mkdir()
                write_reports(scoped_dir,scoped,campaign)
                archive=args.output/(unit['corruption']+'.zip')
                with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                    for name in ('command.txt','environment.txt','notes.md'):
                        z.write(args.output/name,name)
                    z.writestr('config.json',canonical(dict(manifest=spec,campaign=campaign,phase=args.phase,
                        artifact_scope='single_corruption',corruption=unit['corruption'],expected_objects=2468,
                        campaign_complete=False,full_gate_key=args.full_gate_key,full_gate_sha256=full_gate_sha)))
                    for name in ('summary.csv','per_corruption.csv'):
                        z.write(scoped_dir/name,name)
                    for r in scoped:
                        receipt=cache.get(unit_key(r['unit']))
                        z.writestr('raw/'+unit_key(r['unit']),receipt)
                        z.writestr('raw/'+json.loads(receipt)['payload'],r['data'])
                data=archive.read_bytes()
                destination='corruption_'+args.output.name+'_'+unit['corruption']+'.zip'
                store.put(destination,data)
                print('CF_MULTIBACKBONE_CORRUPTION_ARCHIVE',json.dumps(dict(path=str(archive),drive=str(store.folder/destination),sha256=sha(data),bytes=len(data))),flush=True)
                for head in HEADS:
                    result=dict(campaign=campaign,backbone=head,corruption=unit['corruption'],
                        status='complete',archive=destination,archive_sha256=sha(data),
                        rows=[r for r in report_rows(scoped) if r['backbone']==head])
                    name='head_'+args.output.name+'_'+unit['corruption']+'_'+head+'.json'
                    store.put(name,canonical(result))
                    print('CF_MULTIBACKBONE_HEAD_COMPLETE',str(store.folder/name),flush=True)
                cursor=dict(campaign=campaign,corruption=unit['corruption'],objects=2468,heads=list(HEADS),
                    archive=destination,archive_sha256=sha(data))
                store.put('completed_'+unit['corruption']+'.json',canonical(cursor))
                completed[unit['corruption']]=cursor
    if args.phase=='smoke':
        key='smoke_gate.json' if recovered_at_start==len(wanted) else 'smoke_produced.json'
        store.put(key,canonical(dict(campaign=campaign,status='passed',units=len(accepted),environment=environment)))
    if args.phase=='full':
        print('CF_MULTIBACKBONE_DONE',args.phase,'completed_corruptions',len(completed),15,flush=True)
    else:
        print('CF_MULTIBACKBONE_DONE',args.phase,len(accepted),flush=True)
    deadline.cancel()


if __name__=='__main__':
    main()
