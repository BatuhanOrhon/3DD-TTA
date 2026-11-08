"""Standalone existing CPU fixture helpers; no old PREP dependency."""
import importlib.util
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import pickle
import random
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name):
    path = HERE/(name+'.py')
    assert path.is_file(), 'Missing fast617 implementation: '+name
    spec = importlib.util.spec_from_file_location(name+'_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def raw_identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return raw_identity(path.read_bytes())


def npz(**values):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def csv_raw(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def context(root, batch=8):
    operator = root/'operator'
    operator.mkdir(parents=True)
    evaluation = json.loads((HERE/'evaluation617.json').read_text())
    (operator/'evaluation617.json').write_bytes((HERE/'evaluation617.json').read_bytes())
    calibration = dict(schema_version=1, dataset='modelnet40_c', indices=list(range(64)))
    calibration_identity = write_json(operator/'calibration64.json', calibration)
    capsule = b'Synthetic source fixture; never GPU evidence'
    (root/'source').mkdir()
    (root/'source/source_snapshot.zip').write_bytes(capsule)
    source = b'# fixture source\n'
    files = {'source_snapshot/fixture.py': raw_identity(source)}
    manifest = dict(files=files, source_snapshot_sha256=hashlib.sha256(capsule).hexdigest())
    manifest_identity = write_json(root/'source/asset_manifest.json', manifest)
    plan = dict(dataset='modelnet40_c', corruptions=list(module('accept_fast617').CORRUPTIONS),
        asset_manifest_sha256=manifest_identity['sha256'], source_snapshot=raw_identity(capsule),
        checkpoint_sha256='507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75',
        evaluation_file='evaluation617.json', evaluation_manifest_sha256=raw_identity((operator/'evaluation617.json').read_bytes())['sha256'],
        calibration_file='calibration64.json', calibration_manifest_sha256=calibration_identity['sha256'], shards=[])
    for i in range(10):
        ids = evaluation['indices'][64*i:64*(i+1)]
        name = 'shard%03d' % i
        write_json(operator/(name+'.json'), dict(schema_version=1, dataset='modelnet40_c', indices=ids))
        plan['shards'].append(dict(number=i, label=name, count=len(ids), selection_file=name+'.json', arms=['control','half0','M100']))
    write_json(operator/'plan.json', plan)
    batch_identity=write_json(root/'batch_selection.json', dict(status='selected', batch_size=batch,
        asset_manifest_sha256=plan['asset_manifest_sha256'], source_snapshot=plan['source_snapshot'], profiles=[]))
    rows = [dict(id=i, corruption=c, checkpoint=t, scd_norm=2., gsd_norm=1., ratio=2.)
        for i in calibration['indices'] for c in ('gaussian','background')
        for t in ([9,18,27] if c=='background' else [2,4,6])]
    write_json(root/'calibration_frozen.json', dict(status='frozen', dataset='modelnet40_c', batch_size=batch,
        alpha={'100':2.}, rows=rows, asset_manifest_sha256=plan['asset_manifest_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'], calibration_manifest_sha256=plan['calibration_manifest_sha256'],
        source_snapshot=plan['source_snapshot'],batch_selection=batch_identity,ratios_per_mode=384))
    native = root/'native_payload/native/fixture.so'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'\x7fELF synthetic fixture')
    return plan, manifest, source, capsule, dict(path='/native/fixture.so', **raw_identity(native.read_bytes()))


def cache_raw(actual, prepared):
    tensors = dict(shape_latent=np.zeros((actual,128,1,1), dtype=np.float32),
        local_latent=np.zeros((actual,8192,1,1), dtype=np.float32), noise=np.zeros((actual,8192,1,1), dtype=np.float32))
    tensor_id = lambda a: hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
    py = random.Random(0).getstate()
    ns = np.random.RandomState(0).get_state()
    cpu = np.arange(8, dtype=np.uint8)
    cuda = [np.arange(8, dtype=np.uint8)]
    rng_sha = hashlib.sha256(pickle.dumps((py, ns, cpu, cuda), protocol=4)).hexdigest()
    metadata = dict(schema_version=1, input_sha256=tensor_id(prepared), rng_before_sha256='a'*64,
        rng_after_sha256=rng_sha, python_rng=py, numpy_rng=[ns[0],*ns[2:]], cuda_devices=1,
        tensor_hashes={k:tensor_id(v) for k,v in tensors.items()})
    return npz(**tensors, metadata=np.asarray(json.dumps(metadata)), numpy_rng_keys=ns[1],
        torch_rng_cpu=cpu, torch_rng_cuda_0=cuda[0]), metadata


def sealed_run(root, batch=8, count=9, rank=257):
    accept = module('accept_fast617')
    plan, manifest, source, capsule, native = context(root, batch)
    ids = json.loads((root/'operator/evaluation617.json').read_text())['indices'][:count]
    selection = dict(schema_version=1, dataset='modelnet40_c', indices=ids)
    selection_sha = write_json(root/'operator/fixture.json', selection)['sha256']
    config = dict(status='partial', diagnostic_complete=True, dataset='modelnet40_c', backbone='pointmae',
        split='official_modelnet40_test', n_examples=2468, num_classes=40, seed=0, severity=5, batch_size=batch,
        count=count, selected_original_ids=ids, benchmark_shard=True, corruptions=list(accept.CORRUPTIONS),
        completed_batches={c:(count+batch-1)//batch for c in accept.CORRUPTIONS},
        guidance_protocol=dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False),
        normal_reverse_steps=8, background_reverse_steps=35, backend_policy='controlled',
        runtime=dict(cudnn_benchmark=False,cudnn_deterministic=True,cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,deterministic_algorithms=False),
        lion_state_before={'same':'model'}, lion_state_after={'same':'model'}, classifier_state_before_sha256='same', classifier_state_after_sha256='same',
        freeze_decoder_residual=False, update_norm_probes=False, native_decoder_audit=False,
        capture_h5=False, capture_native_trace=False, capture_shared_states=False, skip_dense_filter_witness=True,
        arm='physical', weight=1., checkpoint_scd_weight=.5, half_mix_alpha=2., lowband_modes=100,
        physical_graph=dict(k=10,delta=.1,graph_gamma=.6,modes=100,profile='hard',beta=None),
        pointmae_checkpoint='checkpoints/modelnet_jt.pth', checkpoint_load=dict(strict=True,missing_keys=[],unexpected_keys=[],checkpoint_sha256=plan['checkpoint_sha256']),
        asset_manifest_sha256=plan['asset_manifest_sha256'], asset_manifest=manifest, dataset_provenance={'same':'data'},
        executed_source_files={'repo/fixture.py':raw_identity(source)}, native_extensions={'fixture':native}, native_bootstrap={'bindings':{'fixture':native}},
        index_manifest=str(accept.REMOTE/'operator/fixture.json'), index_manifest_sha256=selection_sha, selection_manifest=selection,
        paired_state_root=str(accept.REMOTE/('paired_states/B%d'%batch)), wall_seconds=1.)
    contents = {k:b'CPU fixture\n' for k in ('command.txt','environment.txt','stdout.log','notes.md')}
    contents.update({'config.json':json.dumps(config).encode(), 'selection_manifest.json':(root/'operator/fixture.json').read_bytes(),
        'source_snapshot.zip':capsule, 'source_snapshot/fixture.py':source})
    counts=[]
    for c in accept.CORRUPTIONS:
        parts=[]
        for start in range(0,count,batch):
            selected=ids[start:start+batch]; actual=len(selected); ordinal=start//batch
            prefix='batches/%s/batch%04d/'%(c,ordinal)
            prepared=np.zeros((actual,2048,3),dtype=np.float32)
            raw,metadata=cache_raw(actual,prepared)
            relative=Path('paired_states/B%d/modelnet40_c'%batch)/c/(hashlib.sha256(np.asarray(selected,dtype=np.int64).tobytes()).hexdigest()+'.npz')
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            logits=np.zeros((actual,4,40),dtype=np.float32);labels=np.zeros(actual,dtype=np.int64)
            p=dict(index=np.asarray(selected),label=labels,prediction=labels,probability_mean=labels,terminal_prediction=labels,
                logits=logits[:,-1],trajectory_logits=logits,candidate_loop_counts=np.asarray([9,18,27,35] if c=='background' else [2,4,6,8]))
            contents[prefix+'predictions.npz']=npz(**p);parts.append(p)
            contents[prefix+'reference.npz']=npz(prepared_x=prepared,initial_h0_xyz=prepared)
            contents[prefix+'decoded_first_checkpoint.npz']=npz(decoded_xyz=prepared)
            records=[dict(loop_count=t,raw_loss=0.,decoded_geometry_reused=True,classifier_gradients=False,
                decoder_and_hash_seconds=.1,spectral_loss_seconds=.01) for t in ([9,18,27] if c=='background' else [2,4,6])]
            d=dict(original_ids=selected,prepared_x_sha256=metadata['input_sha256'], paired_state=dict(path=str(accept.REMOTE/relative),**raw_identity(raw)),
                scheduler=dict(final_alpha_cumprod=1.,config=dict(set_alpha_to_one=True)), input_rng_before_sha256='a'*64,reconstruction_rng_after_sha256='b'*64,
                timing=dict(graph_build_seconds=.1), trajectory=dict(complete=True,probe_rng_unchanged=True,probe_buffers_unchanged=True,
                initial_hashes=metadata['tensor_hashes'],initial_rng_sha256=metadata['rng_after_sha256'],decoded_spectral=dict(
                loss_gradient_mode='full_decoder_gradient',decoder_jacobian_in_loss=True,reference_xyz_sha256=metadata['input_sha256'],records=records)))
            contents[prefix+'diagnostics.json']=json.dumps(d).encode()
            u=np.eye(2048,rank,dtype=np.float32)
            contents[prefix+'physical_bases.npz']=npz(**{'basis_%d'%i:u for i in range(actual)})
            contents[prefix+'graph.json']=json.dumps(dict(reference_xyz_sha256=metadata['input_sha256'],
                diagnostics=[dict(actual_rank=rank,requested_rank=100,spectral_profile='hard') for _ in selected],
                basis_sha256=[accept.tensor_identity(u)]*actual,filter_sha256=['c'*64]*actual,active_vertex_indices=[list(range(2048))]*actual)).encode()
        contents[c+'_predictions.npz']=npz(**{k:np.concatenate([p[k] for p in parts]) for k in ('index','label','prediction','terminal_prediction','logits','trajectory_logits')})
        counts.append(dict(corruption=c,n_examples=count,n_correct=count,accuracy=1.,runtime_seconds=1.,peak_gpu_memory_mb=1.))
    contents['per_corruption.csv']=csv_raw(counts)
    contents['summary.csv']=csv_raw([dict(total_examples=count*15,total_correct=count*15,n_corruptions=15,macro_accuracy=1.)])
    return seal(root/'run.zip',contents),contents


def seal(path, contents):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in contents.items(): z.writestr('bundle/'+name,raw)
        z.writestr('bundle/bundle_manifest.json',json.dumps({n:raw_identity(v) for n,v in contents.items()}))
    return path
