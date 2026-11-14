"""Contracts for the promoted fixed proximal baseline; CPU only, no new benchmark."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import eval_chamfer_baseline as baseline
from scripts import eval_chamfer_depth as depth


def arguments(root, dataset='modelnet40_c', heads=('pointmae',)):
    (root/'ids.json').write_text(json.dumps(dict(dataset=dataset,n_examples=2468 if dataset=='modelnet40_c' else 581,
        stage='development',indices=[0,2,4])))
    return ['--dataset',dataset,'--dataset-root','/data','--label-path','/data/label.npy',
        '--asset-root','/assets','--result-root',str(root/'results'),'--index-manifest',str(root/'ids.json'),
        '--paired-state-root',str(root/'states'),'--pointmae-config','cfgs/tta_modelnet.yaml',
        '--pointmae-checkpoint','/checkpoint.pth','--backbones',*heads,'--preview']


def test_default_profile_is_frozen_proximal_common_N8_and_preserves_legacy(tmp_path):
    config,=baseline.configurations(arguments(tmp_path))
    assert config['operational_baseline']=='lion_proximal_pm_n8_v1'
    assert config['method']=='3dd_proximal_pm_baseline'
    assert config['normal_reverse_steps']==config['background_reverse_steps']==8
    assert config['batch_size']==128 and config['backend_policy']=='controlled'
    assert config['proximal_guidance']==dict(arm='cheap',tau=.1,retained_fraction=.95,style_updates=False,denoiser_backward=False)
    assert config['csv_policy']==config['primary_policy']=='probability_mean'
    assert config['trajectory_selection']['normal_loop_counts']==config['trajectory_selection']['background_loop_counts']==[2,4,6,8]
    legacy,=baseline.configurations(arguments(tmp_path)+['--baseline-profile','chamfer_pm_n8_v1'])
    assert legacy['operational_baseline']=='chamfer_pm_n8_v1' and legacy['background_reverse_steps']==35 and legacy['batch_size']==32
    depth.validate_transfer_config(legacy)


@pytest.mark.parametrize('dataset,heads',[('scanobjectnn_c',('pointmae',)),('modelnet40_c',('pointmae','dgcnn'))])
def test_other_supported_scopes_keep_conservative_batch_and_explicit_empirical_limits(tmp_path,dataset,heads):
    configs=baseline.configurations(arguments(tmp_path,dataset,heads))
    assert all(c['batch_size']==32 and c['background_reverse_steps']==8 for c in configs)
    assert all('Point-MAE' in c['baseline_evidence_scope'] and 'seed0' in c['baseline_evidence_scope'] for c in configs)


def test_new_profile_preview_is_native_free(tmp_path):
    script=Path(baseline.__file__)
    result=subprocess.run([sys.executable,'-S',str(script),*arguments(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)[0]['operational_baseline']=='lion_proximal_pm_n8_v1'


@pytest.mark.parametrize('dataset,count',[('modelnet40_c',2468),('scanobjectnn_c',581)])
def test_full_corruption_shard_keeps_policy_and_requires_all_test_ids(tmp_path,dataset,count):
    args=arguments(tmp_path,dataset=dataset,heads=('pointmae','dgcnn','curvenet','pointnext'))
    (tmp_path/'ids.json').write_text(json.dumps(dict(dataset=dataset,n_examples=count,
        stage='development',indices=list(range(count)))))
    configs=baseline.configurations(args+['--corruption-shard','background'])
    assert len(configs)==4
    for config in configs:
        assert config['corruptions']==['background']
        assert config['normal_reverse_steps']==config['background_reverse_steps']==8
        assert config['corruption_seed_policy']=='fresh_worker_seed_v1'
        baseline.validate_config(config)
    with pytest.raises(ValueError):
        baseline.configurations(args+['--baseline-profile','chamfer_pm_n8_v1','--corruption-shard','background'])
    with pytest.raises(ValueError):
        baseline.configurations(arguments(tmp_path)+['--corruption-shard','background'])


def test_worker_loads_policy_source_before_shared_source_freeze(tmp_path):
    config, = baseline.configurations(arguments(tmp_path))
    directory = tmp_path/'worker'
    directory.mkdir()
    (directory/'config.json').write_text(json.dumps(config))
    check = """
import sys
from scripts import eval_chamfer_baseline as baseline
assert 'scripts.analyze_legacy_trajectory' not in sys.modules
def source_freeze(*args, **kwargs):
    assert 'scripts.analyze_legacy_trajectory' in sys.modules, 'policy loaded after source freeze'
baseline.depth.shared.worker = source_freeze
baseline.worker([sys.argv[1]])
"""
    result = subprocess.run([sys.executable, '-c', check, str(directory)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('changed',[dict(operational_baseline='unknown'),dict(csv_policy='terminal'),
    dict(background_reverse_steps=35),dict(normal_reverse_steps=5),dict(backend_policy='legacy'),
    dict(batch_size=True),dict(batch_size=7),dict(proximal_guidance=dict(arm='hybrid',tau=.1))])
def test_corrupt_new_worker_policy_is_rejected_before_assets(tmp_path,monkeypatch,changed):
    config,=baseline.configurations(arguments(tmp_path));config.update(changed)
    directory=tmp_path/'worker';directory.mkdir();(directory/'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(depth.shared.source,'read_manifest',lambda *a:pytest.fail('asset access before validation'))
    with pytest.raises(ValueError):baseline.worker([str(directory)])


def test_mixed_profile_or_candidate_bundles_are_rejected(tmp_path,monkeypatch):
    configs=baseline.configurations(arguments(tmp_path,heads=('pointmae','dgcnn')))
    configs[1]['seed']=1
    directories=[]
    for i,config in enumerate(configs):
        directory=tmp_path/str(i);directory.mkdir();(directory/'config.json').write_text(json.dumps(config));directories.append(str(directory))
    monkeypatch.setattr(depth.shared.source,'read_manifest',lambda *a:pytest.fail('asset access before validation'))
    with pytest.raises(ValueError):baseline.worker(directories)


def test_proximal_adapter_preserves_state_observers_style_and_terminal_decoder_grad_branch(monkeypatch):
    from legacy_trajectory import _rng_state,rng_hash
    import lion_guidance_pilot
    class VAE:
        def __init__(self):self.encodes=0;self.decode=[]
        def encode(self,x):
            self.encodes+=1
            return None,None,[[torch.ones(len(x),128)],[torch.ones(len(x),8192)]]
        def global2style(self,z):return z*2
        def decoder(self,unused,beta,context,style):
            self.decode.append((torch.is_grad_enabled(),context.requires_grad,style.detach().clone()))
            return context.reshape(len(context),2048,4)[...,:3]
    vae=VAE();lion=SimpleNamespace(vae=vae);calls=[]
    def sample(lion,state,steps,arm,*,tau,callback,scheduler_observer):
        assert steps==8 and arm=='cheap' and tau==.1
        scheduler_observer('actual-scheduler')
        values=[]
        for loop in (2,4,6,8):
            h=state['local_latent']+loop;callback(h,state['shape_latent'],loop);values.append(h)
        return values,[dict(full_backward=False) for _ in range(8)]
    monkeypatch.setattr(lion_guidance_pilot,'sample',sample)
    class Observer:
        def initial(self,**state):calls.append('initial')
        def clean(self,h,z,i,t):calls.append(('clean',i+1,t))
        def final_state(self,h,z,t):calls.append(('final',t))
    observed=[];schedules=[]
    protocol=dict(steps_back_local=8,gamma=.03,eta=.003,p=.95,total=100,scd_normalize=False)
    decoded=baseline.proximal_reconstruct(torch.ones(1,2048,3),lion,**protocol,
        trajectory_observer=Observer(),state_observer=observed.append,scheduler_observer=schedules.append)
    assert decoded.shape==(1,2048,3) and vae.encodes==1 and schedules==['actual-scheduler']
    assert calls==['initial',('clean',2,60),('clean',4,40),('clean',6,20),('final',0)]
    assert set(observed[0])=={'shape_latent','local_latent','noise'}
    assert vae.decode[0][0:2]==(True,True)
    torch.testing.assert_close(vae.decode[0][2],torch.ones(1,128))
    cached=dict(observed[0],rng_after_prepare=_rng_state());before=rng_hash()
    baseline.proximal_reconstruct(torch.ones(1,2048,3),lion,**protocol,prepared_state=cached,
        trajectory_observer=Observer(),scheduler_observer=schedules.append)
    assert vae.encodes==1 and rng_hash()==before
    with pytest.raises(ValueError):
        baseline.proximal_reconstruct(torch.ones(1,2048,3),lion,**dict(protocol,steps_back_local=35))
