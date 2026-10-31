"""Fixed small global-guidance pilot scope, without pretrained models."""
import importlib
import json

import pytest

import test_eval_shape_latent as parent


def feature():return importlib.import_module('scripts.eval_shape_proximal')


def args(root):
    return parent.args(root,'development',32)+['--global-state-root',str(root/'global')]


@pytest.mark.parametrize('arm',['original','mean','prior3','proxy3','oracle3'])
def test_fixed_small_arms_and_explicit_clean_target_policy(tmp_path,arm):
    m=feature();extra=['--oracle-target-root',str(tmp_path/'oracle')] if arm=='oracle3' else []
    cfg,=m.configurations(args(tmp_path)+['--global-arm',arm]+extra)
    assert cfg['corruptions']==['gaussian','background','rotation','lidar']
    assert cfg['evaluation_n_examples']==32 and cfg['decoder_style']=='original'
    assert cfg['shape_proximal']['global_forward_calls']==(3 if arm.endswith('3') else 0)
    assert cfg['shape_proximal']['clean_information_used']==(arm=='oracle3')
    assert cfg['shape_proximal']['guidance_target_weight']==(1. if arm in ('proxy3','oracle3') else 0.)
    if arm in ('original','prior3'):assert cfg['shape_proximal']['target']=='none'
    for key,value in [('seed',1),('normal_reverse_steps',5),('evaluation_n_examples',64),('decoder_style','updated')]:
        with pytest.raises(ValueError):m.validate_config(dict(cfg,**{key:value}))


def test_oracle_path_cannot_enter_deployed_arms_and_smoke_is_exact_gb(tmp_path):
    m=feature()
    with pytest.raises(ValueError):m.configurations(args(tmp_path)+['--global-arm','proxy3','--oracle-target-root',str(tmp_path/'oracle')])
    with pytest.raises(ValueError):m.configurations(args(tmp_path)+['--global-arm','oracle3'])
    cfg,=m.configurations(args(tmp_path)+['--global-arm','prior3','--global-technical-smoke'])
    assert cfg['corruptions']==['gaussian','background'] and cfg['technical_smoke']
    cfg['global_state_root']=cfg['paired_state_root']
    with pytest.raises(ValueError):m.validate_config(cfg)


def test_five_arm_routing_replays_original_h_and_rejects_mispaired_reference(tmp_path):
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state,_restore_rng,tensor_hash
    from shape_latent_refresh import prepare_reference,refresh_proof,save_proof
    from scripts.eval_multibackbone_guidance import paired_preparation
    from scripts.eval_multibackbone_source import file_identity
    from test_shape_latent_refresh import RefreshVAE
    from test_shape_latent_proximal import Prior,Scheduler
    m=feature();cfg,=m.configurations(args(tmp_path)+['--global-arm','original'])
    root=Path(cfg['asset_root']);(root/'references').mkdir(parents=True)
    x=torch.ones(32,2048,3);name=hashlib.sha256(np.asarray(cfg['selected_original_ids'],dtype=np.int64).tobytes()).hexdigest()+'.npz'
    common=root/'references'/name;proof_file=common.with_suffix('.refresh.npz')
    _,record=paired_preparation(common,x);state,initial=prepare_reference(x,RefreshVAE().eval())
    record({k:state[k] for k in ('shape_latent','local_latent','noise')})
    proof=refresh_proof(x*.2,state,initial,RefreshVAE().eval())
    binding=dict(input_sha256=tensor_hash(x),common_state=file_identity(common),source='old-published')
    save_proof(proof_file,proof,binding)
    oracle_dir=tmp_path/'oracle';oracle_dir.mkdir();oracle=oracle_dir/'clean.npz'
    np.savez_compressed(oracle,clean_z=state['shape_latent'].flatten(1).numpy()+.1)
    entry=dict(common=dict(path=str(common.relative_to(root)),**file_identity(common)),
        proof=dict(path=str(proof_file.relative_to(root)),**file_identity(proof_file)),proof_binding=binding,
        oracle=dict(filename=oracle.name,**file_identity(oracle)))
    manifest_path=root/'references/manifest.json'
    inputs=dict(original_ids=cfg['selected_original_ids'],lion_state={'vae':'v','priors':'p'},classifier_state='c',dataset_provenance={'files':{'data':'published-sha'}},cases={'gaussian':entry})
    manifest_path.write_text(json.dumps(inputs))
    cfg.update(asset_manifest={'files':{'references/manifest.json':file_identity(manifest_path)}},
        lion_state_before=inputs['lion_state'],classifier_state_before_sha256='c',executed_source_sha256='new-source',dataset_provenance=inputs['dataset_provenance'])
    seen=[];lion=SimpleNamespace(priors=[Prior().eval()])
    ops=SimpleNamespace(normalize=lambda raw:(raw,torch.zeros(32,1,3),torch.ones(32,1,1)))
    def reconstruct(**kw):
        _restore_rng(kw['prepared_state']['rng_after_prepare']);kw['scheduler_observer'](None)
        kw['state_observer']({k:kw['prepared_state'][k] for k in ('shape_latent','local_latent','noise')})
        seen.append(kw);return kw['x']
    start=_rng_state()
    with patch.object(m,'make_scheduler',lambda device:Scheduler()),patch.object(m.depth,'scheduler_record',lambda *a:{}),patch.object(m.shared,'decoded_to_geometry',lambda decoded,*a:decoded):
        for arm in m.ARMS:
            _restore_rng(start);cfg['shape_proximal']=m.settings(arm);cfg['oracle_target_root']=str(oracle_dir) if arm=='oracle3' else None
            _,metrics=m.reconstruct_geometry(x,'modelnet40_c','gaussian',lion,None,reconstruct,ops,
                global_config=cfg,device='cpu',paired_state_path=tmp_path/'selected'/arm/name,
                observer_factory=lambda *a:object(),protocol=cfg['guidance_protocol'])
            assert metrics['shape_proximal']['clean_information_used']==(arm=='oracle3')
        assert torch.equal(seen[0]['prepared_state']['shape_latent'],state['shape_latent'])
        assert torch.equal(seen[1]['prepared_state']['shape_latent'].flatten(1),proof['mean_z'])
        for item in seen:
            for key in ('local_latent','noise'):assert torch.equal(item['prepared_state'][key],state[key])
            assert torch.equal(item['chamfer_reference'],state['local_latent'].view(32,2048,4)[:,:,:3])
        # A separately valid cache from another latent draw must not pair with this proof.
        with np.load(common,allow_pickle=False) as saved:fields={k:saved[k].copy() for k in saved.files}
        fields['shape_latent']+=1;meta=json.loads(fields['metadata'].item())
        meta['tensor_hashes']['shape_latent']=tensor_hash(torch.from_numpy(fields['shape_latent']))
        fields['metadata']=np.asarray(json.dumps(meta));other=root/'references/other.npz'
        np.savez_compressed(other,**fields)
        inputs['cases']['gaussian']['common']=dict(path=str(other.relative_to(root)),**file_identity(other))
        manifest_path.write_text(json.dumps(inputs));cfg['asset_manifest']['files']['references/manifest.json']=file_identity(manifest_path)
        cfg['shape_proximal']=m.settings('original');cfg['oracle_target_root']=None;cfg['global_state_root']=str(tmp_path/'bad_global')
        _restore_rng(start)
        with pytest.raises(ValueError,match='reference binding'):
            m.reconstruct_geometry(x,'modelnet40_c','gaussian',lion,None,reconstruct,ops,
                global_config=cfg,device='cpu',paired_state_path=tmp_path/'bad_selected'/name,
                observer_factory=lambda *a:object(),protocol=cfg['guidance_protocol'])
