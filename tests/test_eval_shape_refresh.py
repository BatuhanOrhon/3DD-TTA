"""Fixed refresh protocol and actual original-host routing."""
import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import test_eval_shape_latent as parent
from test_shape_latent_refresh import RefreshVAE


def feature():
    return importlib.import_module('scripts.eval_shape_refresh')


def args(root, stage='development', count=64):
    return parent.args(root,stage,count)+['--refresh-state-root',str(root/'refresh')]


@pytest.mark.parametrize('arm',['baseline','mean_refresh','posterior_refresh'])
def test_three_fixed_arms_and_no_clean_or_global_diffusion(tmp_path,arm):
    m=feature();cfg,=m.configurations(args(tmp_path)+['--refresh-arm',arm])
    assert cfg['shape_refresh']==m.settings(arm)
    assert cfg['decoder_style']=='original' and 'shape_latent' not in cfg and 'shape_oracle' not in cfg
    assert cfg['shape_refresh']['global_prior_calls']==0
    for key,value in [('seed',1),('normal_reverse_steps',5),('decoder_style','updated'),('max_batches',1)]:
        with pytest.raises(ValueError):m.validate_config(dict(cfg,**{key:value}))
    bad=copy.deepcopy(cfg);bad['shape_refresh']['posterior_epsilon']='fresh'
    with pytest.raises(ValueError):m.validate_config(bad)


def test_explicit_smoke_and_separate_common_cache(tmp_path):
    m=feature();cfg,=m.configurations(args(tmp_path,'smoke',32)+['--refresh-arm','baseline','--refresh-technical-smoke'])
    assert cfg['technical_smoke'] and cfg['corruptions']==['gaussian','background']
    cfg['refresh_state_root']=cfg['paired_state_root']
    with pytest.raises(ValueError):m.validate_config(cfg)


def test_actual_routing_keeps_h_noise_anchor_and_same_proposal(tmp_path):
    m=feature();vae=RefreshVAE().eval()
    lion=type('Lion',(),{'vae':vae})();x=torch.ones(2,2048,3)
    cfg,=m.configurations(args(tmp_path)+['--refresh-arm','baseline'])
    cfg['executed_source_sha256']='actual-source-a'
    # Exercise the wrapper's callback using real state preparation and disk proofs.
    seen=[]
    def reconstruct(**kw):
        state=kw['prepared_state'];seen.append(kw)
        from legacy_trajectory import _restore_rng
        _restore_rng(state['rng_after_prepare'])
        if kw.get('state_observer'):kw['state_observer']({k:state[k] for k in ('shape_latent','local_latent','noise')})
        return x*.2
    def host(*args,**kwargs):
        points=args[5](x=x,lion=lion,prepared_state=None,state_observer=None)
        return points,{}
    with patch.object(m.depth,'reconstruct_geometry',host):
        from legacy_trajectory import _rng_state, _restore_rng
        start=_rng_state()
        for arm in m.ARMS:
            _restore_rng(start)
            cfg['shape_refresh']=m.settings(arm)
            _,metrics=m.reconstruct_geometry(x,'modelnet40_c','gaussian',lion,None,reconstruct,None,
                refresh_config=cfg,device='cpu',paired_state_path=tmp_path/'selected'/arm/'batch.npz')
            assert Path(metrics['shape_refresh']['common_state']['path']).is_file()
            assert Path(metrics['shape_refresh']['posterior_proof']['path']).is_file()
            if arm=='baseline':
                # A proof exists before worker completion; standalone candidates must reject it.
                cfg['shape_refresh']=m.settings('mean_refresh');_restore_rng(start)
                with pytest.raises(ValueError,match='completed baseline'):
                    m.reconstruct_geometry(x,'modelnet40_c','gaussian',lion,None,reconstruct,None,
                        refresh_config=cfg,device='cpu',paired_state_path=tmp_path/'selected'/'mean_refresh'/'batch.npz')
                cfg['shape_refresh']=m.settings('baseline')
                complete=dict(cfg,status='complete',completed_batches={c:2 for c in cfg['corruptions']},
                    classifier_state_before_sha256='c',classifier_state_after_sha256='c',
                    lion_state_before={'vae':'v'},lion_state_after={'vae':'v'})
                baseline_file=tmp_path/'completed_baseline_config.json'
                baseline_file.write_text(json.dumps(complete))
                root=tmp_path/'refresh';identity=m.shared.source.file_identity(baseline_file)
                (root/'baseline_complete.json').write_text(json.dumps(dict(config_path=str(baseline_file),config_identity=identity)))
    assert len(seen)==3
    for other in seen[1:]:
        for key in ('local_latent','noise'):assert torch.equal(seen[0]['prepared_state'][key],other['prepared_state'][key])
        assert torch.equal(seen[0]['chamfer_reference'],other['chamfer_reference'])
    from shape_latent_refresh import load_proof
    proof_file=tmp_path/'refresh/modelnet40_c/gaussian/batch.refresh.npz'
    import numpy as np
    with np.load(proof_file,allow_pickle=False) as saved:binding=json.loads(saved['metadata'].item())['binding']
    proof=load_proof(proof_file,'cpu',binding)
    assert torch.equal(seen[1]['prepared_state']['shape_latent'].flatten(1),proof['mean_z'])
    assert torch.equal(seen[2]['prepared_state']['shape_latent'].flatten(1),proof['posterior_z'])
    assert torch.equal(proof['decoded_points'],x*.2)
    cfg['executed_source_sha256']='changed-source';cfg['shape_refresh']=m.settings('mean_refresh')
    _restore_rng(start)
    with patch.object(m.depth,'reconstruct_geometry',host),pytest.raises(ValueError,match='source'):
        m.reconstruct_geometry(x,'modelnet40_c','gaussian',lion,None,reconstruct,None,
            refresh_config=cfg,device='cpu',paired_state_path=tmp_path/'selected'/'mean_refresh'/'batch.npz')


def test_candidate_cannot_create_missing_baseline(tmp_path):
    m=feature();cfg,=m.configurations(args(tmp_path)+['--refresh-arm','mean_refresh'])
    def host(*args,**kw):return args[5](x=torch.ones(2,2048,3),prepared_state=None),{}
    with patch.object(m.depth,'reconstruct_geometry',host),pytest.raises(ValueError):
        m.reconstruct_geometry(None,'modelnet40_c','gaussian',type('Lion',(),{'vae':RefreshVAE().eval()})(),None,None,None,
            refresh_config=cfg,device='cpu',paired_state_path=tmp_path/'batch.npz')
