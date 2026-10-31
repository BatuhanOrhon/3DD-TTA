#!/usr/bin/env python3
"""Small global-DDIM clean-estimate guidance study with immutable imported states."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth
shared=depth.shared
ARMS=('original','mean','prior3','proxy3','oracle3')
CORRUPTIONS=['gaussian','background','rotation','lidar']
METHOD='3dd_shape_proximal_pilot'


def settings(arm):
    if arm not in ARMS:raise ValueError('Unknown fixed global arm')
    return dict(arm=arm,global_forward_calls=3 if arm.endswith('3') else 0,
                global_backward_calls=0,global_timesteps=[20,10,0] if arm.endswith('3') else [],
                guidance_target_weight=1. if arm in ('proxy3','oracle3') else 0.,
                clean_information_used=arm=='oracle3',
                target='none' if arm in ('original','prior3') else ('clean oracle' if arm=='oracle3' else 'denoised encoder mean with old sigma*eps'),
                local_state='imported original h0/noise/postprepare RNG and h0 XYZ anchor')


def validate_config(config):
    spec=config.get('shape_proximal',{});arm=spec.get('arm')
    if (config.get('method')!=METHOD or spec!=settings(arm)
            or config.get('dataset')!='modelnet40_c' or config.get('backbone')!='pointmae'
            or config.get('seed')!=0 or config.get('batch_size')!=32
            or config.get('evaluation_n_examples')!=32 or len(config.get('selected_original_ids',[]))!=32
            or config.get('selection_manifest',{}).get('stage')!='development'
            or config.get('corruptions')!=(['gaussian','background'] if config.get('technical_smoke') else CORRUPTIONS)
            or config.get('normal_reverse_steps')!=8 or config.get('background_reverse_steps')!=35
            or config.get('guidance_protocol')!=dict(gamma=.03,eta=.003,p=.95,total=100,scd_normalize=False)
            or config.get('decoder_style')!='original' or config.get('lion_ema') is not False
            or config.get('max_batches')!=0 or config.get('primary_policy')!='probability_mean'
            or any(k in config for k in ('shape_latent','shape_refresh','shape_oracle'))
            or bool(config.get('oracle_target_root'))!=(arm=='oracle3')
            or not config.get('global_state_root')
            or Path(config['global_state_root']).resolve()==Path(config['paired_state_root']).resolve()):
        raise ValueError('Global pilot differs from fixed scope/clean-information contract')
    if (config['trajectory_selection']['normal_loop_counts']!=[2,4,6,8]
            or config['trajectory_selection']['background_loop_counts']!=[9,18,27,35]):
        raise ValueError('Global pilot changed PM checkpoints')


def configurations(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--global-arm',choices=ARMS,required=True)
    parser.add_argument('--global-state-root',required=True)
    parser.add_argument('--oracle-target-root')
    parser.add_argument('--global-technical-smoke',action='store_true')
    options,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    config,=depth.configurations(['--normal-reverse-steps','8','--gamma','.03','--eta','.003','--p','.95']+rest)
    config.update(method=METHOD,shape_proximal=settings(options.global_arm),
        global_state_root=str(Path(options.global_state_root).absolute()),
        oracle_target_root=str(Path(options.oracle_target_root).absolute()) if options.oracle_target_root else None,
        technical_smoke=options.global_technical_smoke,
        corruptions=['gaussian','background'] if options.global_technical_smoke else list(CORRUPTIONS),
        display_name='Global proximal shape pilot: '+options.global_arm,
        shared_reconstruction='fixed imported initial state; own initial z at clean/final decodes',
        comparison_scope='32exposed development objects, four fixed corruptions, seed0; oracle3 diagnostic only',
        randomness_policy='frozen imported postprepare RNG; private common global Gaussian; original local noise',
        timing_policy='replay excludes previous first local pass/proxy encoding; oracle not deployable')
    validate_config(config);return [config]


def make_scheduler(device):
    from diffusers import DDIMScheduler
    scheduler=DDIMScheduler(beta_start=.0001,beta_end=.02,beta_schedule='linear',
        num_train_timesteps=1000,prediction_type='epsilon',clip_sample=False)
    scheduler.set_timesteps(100,device=device)
    return scheduler


def reconstruct_geometry(raw,dataset,corruption,lion,graph,reconstruct,ops,*,global_config,
                         device='cuda',method=METHOD,scheduler_observer=None,
                         observer_factory=None,protocol=None,paired_state_path=None):
    import numpy as np
    import torch
    from legacy_trajectory import tensor_hash
    from shape_latent_proximal import load_reference,propose
    from shape_latent_refresh import load_proof,save_proof
    if method!=METHOD or graph is not None or paired_state_path is None or observer_factory is None:
        raise ValueError('Global pilot requires frozen paired original host')
    root=Path(global_config['asset_root']);manifest_path=root/'references/manifest.json'
    shared.source.verify_identity(manifest_path,global_config['asset_manifest']['files']['references/manifest.json'])
    inputs=json.loads(manifest_path.read_text());ids=global_config['selected_original_ids']
    name=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()+'.npz'
    if (inputs['original_ids']!=ids or Path(paired_state_path).name!=name
            or inputs['lion_state']!=global_config['lion_state_before']
            or inputs['classifier_state']!=global_config['classifier_state_before_sha256']
            or inputs['dataset_provenance']!=global_config['dataset_provenance']):
        raise ValueError('Imported reference IDs/model/dataset identities differ')
    entry=inputs['cases'][corruption]
    base,base_meta=load_reference(root/entry['common']['path'],entry['common'],device)
    imported=load_proof(root/entry['proof']['path'],device,entry['proof_binding'])
    shared.source.verify_identity(root/entry['proof']['path'],entry['proof'])
    x=imported['prepared_input']
    if (tensor_hash(x)!=base_meta['input_sha256'] or len(x)!=32
            or entry['proof_binding'].get('common_state')!=shared.source.file_identity(root/entry['common']['path'])
            or entry['proof_binding'].get('input_sha256')!=base_meta['input_sha256']):
        raise ValueError('Imported reference binding/input differs')
    arm=global_config['shape_proximal']['arm'];oracle=None
    if arm=='oracle3':
        path=Path(global_config['oracle_target_root'])/entry['oracle']['filename']
        shared.source.verify_identity(path,entry['oracle'])
        with np.load(path,allow_pickle=False) as saved:oracle=torch.from_numpy(saved['clean_z'].copy()).to(device)[:,:,None,None]
    output=Path(global_config['global_state_root'])/arm/dataset/corruption/(Path(name).stem+'.global.npz')
    if output.exists():raise ValueError('Preserve prior global proof; never overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    seed=int.from_bytes(hashlib.sha256((str(global_config['seed'])+':'+name+':proximal').encode()).digest()[:8],'big')%(2**63)
    z,proof=propose(base['shape_latent'],imported['mean_z'][:,:,None,None],arm,lion.priors[0],
                    make_scheduler(device),noise_seed=seed,oracle_target=oracle)
    selected=dict(base,shape_latent=z)
    cached,record=shared.paired_preparation(paired_state_path,x)
    if cached is not None:raise ValueError('Fresh global phase selected-cache already exists')
    binding=dict(executed_source_sha256=global_config['executed_source_sha256'],
                 input_manifest_sha256=shared.source.file_identity(manifest_path)['sha256'],arm=arm,
                 imported_common=entry['common'],imported_proof=entry['proof'],
                 oracle=entry['oracle'] if arm=='oracle3' else None)
    save_proof(output,proof,binding)
    _,center,scale=ops.normalize(raw);steps=35 if corruption=='background' else 8;seen=[]
    def observe(scheduler):
        record=depth.scheduler_record(scheduler,steps);seen.append(record)
        if scheduler_observer is not None:scheduler_observer(record)
    reference=base['local_latent'].view(32,2048,4)[:,:,:3]
    decoded=reconstruct(x=x,lion=lion,steps_back_local=steps,scheduler_observer=observe,
        prepared_state=selected,state_observer=record,trajectory_observer=observer_factory(center,scale),
        chamfer_reference=reference,**protocol)
    if len(seen)!=1:raise ValueError('Local scheduler record missing')
    geometry=shared.decoded_to_geometry(decoded,dataset,scale,center,ops)
    if geometry.shape!=(32,2048,3) or not torch.isfinite(geometry).all():raise ValueError('Invalid global pilot geometry')
    metrics=dict(arm=arm,global_forward_calls=global_config['shape_proximal']['global_forward_calls'],
        global_backward_calls=0,clean_information_used=arm=='oracle3',global_noise_seed=seed,
        global_proof=dict(path=str(output),**shared.source.file_identity(output)),
        imported_common=entry['common'],imported_proof=entry['proof'],original_ids=ids,
        input_sha256=tensor_hash(x),chamfer_reference_sha256=tensor_hash(reference),
        selected_z_sha256=tensor_hash(z),executed_source_sha256=global_config['executed_source_sha256'])
    return geometry.detach(),dict(shape_proximal=metrics)


def worker(directories):
    import shape_latent_proximal
    import shape_latent_refresh
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories)!=1:raise ValueError('Global pilot uses one frozen Point-MAE bundle')
    config=json.loads((Path(directories[0])/'config.json').read_text());validate_config(config);actual=[]
    def runtime_loader(common,manifest):
        actual.append(common);original=copy.deepcopy(common);original['method']=shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original,manifest)
    def geometry(*args,**kwargs):return reconstruct_geometry(*args,global_config=actual[0],**kwargs)
    shared.worker(directories,runtime_loader=runtime_loader,geometry_reconstructor=geometry)


def main(argv=None):
    shared.main(argv,configuration_factory=configurations,worker_entrypoint=__file__,worker_fn=worker)


if __name__=='__main__':main()
