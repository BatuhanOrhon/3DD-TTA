#!/usr/bin/env python3
"""Three paired frozen-global-encoder refresh arms on the original N8/B35 host."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import eval_shape_latent as parent
from scripts import eval_chamfer_depth as depth

shared=parent.shared
ARMS=('baseline','mean_refresh','posterior_refresh')
METHOD='3dd_shape_posterior_refresh_pilot'


def settings(arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed refresh arm')
    return dict(arm=arm,global_prior_calls=0,posterior_epsilon='same reference eps_z',
                local_state='original h0 and diffusion noise',
                observation='one shared baseline decoder output, LION frame',
                clean_information_used=False,chamfer_target='baseline h0 XYZ')


def validate_config(config):
    refresh=config.get('shape_refresh',{})
    if (config.get('method') != METHOD or refresh != settings(refresh.get('arm'))
            or 'shape_latent' in config or 'shape_oracle' in config
            or config.get('max_batches') != 0 or config.get('csv_policy') != 'terminal'
            or config.get('lion_ema') is not False or not config.get('refresh_state_root')
            or Path(config['refresh_state_root']).resolve()==Path(config['paired_state_root']).resolve()):
        raise ValueError('Invalid fixed refresh payload/cache roots')
    parent.validate_config(dict(config,method=parent.METHOD,shape_latent=parent.shape_settings('baseline')))


def configurations(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--refresh-arm',choices=ARMS,required=True)
    parser.add_argument('--refresh-state-root',required=True)
    parser.add_argument('--refresh-technical-smoke',action='store_true')
    options,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if any(v in rest for v in ('--shape-arm','--shape-technical-smoke')):
        raise ValueError('Refresh cannot combine global guidance')
    config,=parent.configurations(['--shape-arm','baseline']+
        (['--shape-technical-smoke'] if options.refresh_technical_smoke else [])+rest)
    config.pop('shape_latent')
    config.update(method=METHOD,shape_refresh=settings(options.refresh_arm),
        refresh_state_root=str(Path(options.refresh_state_root).absolute()),
        display_name='Frozen global posterior refresh: '+options.refresh_arm,
        shared_reconstruction='original N8/B35 with common baseline-denoised X and posterior epsilon',
        comparison_scope='64 exposed development objects, seed0 exploratory three-arm pilot',
        randomness_policy='same eps_z, initial h0, local diffusion noise and postprepare RNG',
        timing_policy='replay timing excludes shared baseline first pass; deployment cost must add it')
    validate_config(config)
    return [config]


def validate_baseline(path):
    config=json.loads(Path(path).read_text());validate_config(config)
    batches=(config['evaluation_n_examples']+31)//32
    if (config['shape_refresh']['arm']!='baseline'
            or config.get('status') not in (('complete','partial') if config['technical_smoke'] else ('complete',))
            or config.get('completed_batches')!={c:batches for c in config['corruptions']}
            or not config.get('executed_source_sha256') or config.get('failure')
            or not config.get('lion_state_before') or config['lion_state_before']!=config.get('lion_state_after')
            or not config.get('classifier_state_before_sha256')
            or config['classifier_state_before_sha256']!=config.get('classifier_state_after_sha256')):
        raise ValueError('Need completed baseline with final frozen-state/source guards')
    return config


def completed_baseline(root,config):
    receipt_path=Path(root)/'baseline_complete.json'
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError('Need completed baseline receipt before candidate inference')
    receipt=json.loads(receipt_path.read_text());path=Path(receipt['config_path'])
    if path.is_symlink() or shared.source.file_identity(path)!=receipt['config_identity']:
        raise ValueError('Completed baseline config identity changed')
    baseline=validate_baseline(path)
    for key in ('asset_manifest_sha256','executed_source_sha256','selected_original_ids','corruptions','seed'):
        if baseline.get(key)!=config.get(key):
            raise ValueError('Completed baseline source/protocol differs: '+key)


def reconstruct_geometry(*args,refresh_config,device='cuda',method=METHOD,**kwargs):
    from legacy_trajectory import _rng_state_hash, tensor_hash
    from shape_latent_refresh import prepare_reference, refresh_proof, save_proof, load_proof
    raw,dataset,corruption,lion,graph,reconstruct,ops=args
    selected_path=kwargs.get('paired_state_path')
    if method != METHOD or selected_path is None or graph is not None:
        raise ValueError('Refresh needs original host and paired cache')
    arm=refresh_config['shape_refresh']['arm'];metrics={}
    if not refresh_config.get('executed_source_sha256'):
        raise ValueError('Actual executed source signature required')
    if arm!='baseline':completed_baseline(refresh_config['refresh_state_root'],refresh_config)
    def sample(**parameters):
        x=parameters['x']
        common=Path(refresh_config['refresh_state_root'])/dataset/corruption/Path(selected_path).name
        proof_path=common.with_suffix('.refresh.npz')
        base,record=shared.paired_preparation(common,x)
        if arm=='baseline':
            if base is not None or proof_path.exists():
                raise ValueError('Baseline state/proof exists; preserve failed or completed attempt')
            base,initial=prepare_reference(x,lion.vae)
            record({k:base[k] for k in ('shape_latent','local_latent','noise')})
        elif base is None or not proof_path.is_file():
            raise ValueError('Refresh candidate needs completed common baseline proof')
        binding=dict(input_sha256=tensor_hash(x),common_state=shared.source.file_identity(common),
                     asset_manifest_sha256=refresh_config.get('asset_manifest_sha256'),
                     executed_source_sha256=refresh_config['executed_source_sha256'])
        proof=None if arm=='baseline' else load_proof(proof_path,device,binding)
        selected=dict(base)
        if arm != 'baseline':
            selected['shape_latent']=proof['mean_z' if arm=='mean_refresh' else 'posterior_z'][:,:,None,None]
        cached=parameters.get('prepared_state')
        if cached is not None and (any(tensor_hash(cached[k])!=tensor_hash(selected[k])
                for k in ('shape_latent','local_latent','noise'))
                or _rng_state_hash(cached['rng_after_prepare'])!=_rng_state_hash(selected['rng_after_prepare'])):
            raise ValueError('Selected refresh cache differs from common proof')
        parameters['prepared_state']=selected
        reference=base['local_latent'].view(len(x),2048,4)[:,:,:3]
        parameters['chamfer_reference']=reference
        decoded=reconstruct(**parameters)
        if arm=='baseline':
            proof=refresh_proof(decoded,base,initial,lion.vae)
            save_proof(proof_path,proof,binding)
        metrics.update(arm=arm,global_prior_calls=0,common_state=dict(path=str(common),**shared.source.file_identity(common)),
                       posterior_proof=dict(path=str(proof_path),**shared.source.file_identity(proof_path)),
                       input_sha256=tensor_hash(x),chamfer_reference_sha256=tensor_hash(reference),
                       proposal_sha256=tensor_hash(selected['shape_latent']),
                       executed_source_sha256=refresh_config['executed_source_sha256'])
        return decoded
    geometry,_=depth.reconstruct_geometry(*args[:5],sample,ops,depth_config=refresh_config,
                                          device=device,method=depth.METHOD,**kwargs)
    return geometry,dict(shape_refresh=metrics)


def worker(directories):
    import shape_latent_refresh
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories)!=1:
        raise ValueError('Refresh requires one frozen Point-MAE bundle')
    config=json.loads((Path(directories[0])/'config.json').read_text());validate_config(config)
    runtime_configs=[]
    def runtime_loader(common,manifest):
        runtime_configs.append(common)
        original=copy.deepcopy(common);original['method']=shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original,manifest)
    def geometry(*args,**kwargs):
        return reconstruct_geometry(*args,refresh_config=runtime_configs[0],**kwargs)
    shared.worker(directories,runtime_loader=runtime_loader,geometry_reconstructor=geometry)


def main(argv=None):
    shared.main(argv,configuration_factory=configurations,worker_entrypoint=__file__,worker_fn=worker)


if __name__=='__main__':main()
