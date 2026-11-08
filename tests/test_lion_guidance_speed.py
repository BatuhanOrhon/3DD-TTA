"""Guard the repeated operational comparison, not implementation details."""
import pytest


def records():
    # Two objects, two repetitions: candidate has SAME mean accuracy although
    # averaging probabilities across repeats could produce a different score.
    return [dict(arm=arm, object_id=i, corruption='gaussian', seed=0,
                 repeat=r, batch=0, label=i, correct=int((i+r)%2 == (arm=='original')),
                 seconds=2. if arm=='original' else 1., batch_size=2)
            for arm in ('original','cheap') for r in range(2) for i in range(2)]


def test_repeated_accuracy_is_per_inference_and_latency_counts_each_batch_once():
    from scripts.analyze_lion_guidance_speed import paired_summary
    result=paired_summary(records(),'original','cheap',draws=100)
    assert result['objects']==2 and result['paired_inferences']==4
    assert result['reference_accuracy_percent']==50
    assert result['candidate_accuracy_percent']==50
    assert result['delta_pp']==0
    assert result['reference_seconds']==4 and result['candidate_seconds']==2
    assert result['saving_percent']==50
    assert result['delta_ci95_pp']==[0.,0.]
    assert result['degenerate_object_differences'] is True


def test_missing_or_duplicate_pair_cannot_produce_a_partial_success():
    from scripts.analyze_lion_guidance_speed import paired_summary
    data=records()
    with pytest.raises(ValueError,match='pair'):
        paired_summary(data[:-1],'original','cheap')
    with pytest.raises(ValueError,match='duplicate'):
        paired_summary(data+[data[0]],'original','cheap')


def test_inconsistent_label_or_batch_timing_is_rejected():
    from scripts.analyze_lion_guidance_speed import paired_summary
    data=records();data[-1]['label']=999
    with pytest.raises(ValueError,match='label'):
        paired_summary(data,'original','cheap')
    data=records();data[-1]['seconds']=99
    with pytest.raises(ValueError,match='timing'):
        paired_summary(data,'original','cheap')


def test_paired_timing_rejects_changed_object_batch_membership():
    from scripts.analyze_lion_guidance_speed import paired_summary
    data=records()
    data += [dict(r, object_id=r['object_id']+2, label=r['label']+2, batch=1) for r in records()]
    for r in data:
        if r['arm']=='cheap':
            r['batch']=1-r['batch']
    with pytest.raises(ValueError,match='membership'):
        paired_summary(data,'original','cheap')


def test_order_balances_positions_over_blocks():
    from scripts.run_lion_guidance_speed import arm_order
    arms=['original','full','cheap','hybrid','sparse']
    orders=[arm_order(arms,i) for i in range(10)]
    for position in range(5):
        assert sorted(row[position] for row in orders)==sorted(arms*2)


def test_light_observer_observes_only_declared_steps():
    import torch
    from scripts.run_lion_guidance_speed import SpeedObserver
    seen=[]
    def readout(h,z,terminal):
        seen.append((float(h.detach().flatten()[0]),terminal,torch.is_grad_enabled()))
        return torch.zeros(len(h),40)
    obs=SpeedObserver([2,4,6],readout)
    h=torch.ones(2,3,requires_grad=True)
    for i in range(8):obs.clean(h*i,h,i,i)
    obs.final_state(h*8,h,0)
    obs.terminal(torch.ones(2,40))
    assert [x[0] for x in seen]==[1,3,5]
    assert obs.logits().shape==(2,4,40)
    assert len(obs.latents)==4
    assert all(not x.requires_grad for x in obs.latents)


def test_pm_readout_preserves_decoder_grad_branch_and_probe_rng():
    import torch
    from scripts.run_lion_guidance_speed import pm_readout
    model = torch.nn.Linear(3, 3).eval().requires_grad_(False)
    seen = []
    def decode(h, z):
        seen.append(('decode', torch.is_grad_enabled(), h.requires_grad))
        return model(h)
    def classify(x):
        seen.append(('classify', torch.is_grad_enabled(), x.requires_grad))
        return x + torch.rand_like(x)
    h = torch.ones(2, 3, requires_grad=True)
    before = torch.random.get_rng_state()
    pm_readout(h, h, False, decode, classify, [model])
    assert torch.equal(before, torch.random.get_rng_state())
    assert seen == [('decode', False, False), ('classify', False, False)]
    seen.clear()
    pm_readout(h.detach(), h, True, decode, classify, [model])
    assert torch.equal(before, torch.random.get_rng_state())
    assert seen == [('decode', True, True), ('classify', False, True)]


def test_cross_controls_include_every_repeat_combination(tmp_path):
    import numpy as np
    from scripts.analyze_lion_guidance_speed import cross_control_differences
    for arm in ('original','full'):
        for repeat in range(3):
            value=repeat+(arm=='full')
            np.savez(tmp_path/('%s_r%d.npz'%(arm,repeat)),
                     latents=np.full((2,4,3),value),logits=np.full((2,4,40),value),
                     prediction=np.array([0,value]),index=np.array([8,9]))
    result=cross_control_differences(tmp_path,'original','full',3)
    assert len(result)==9
    first=result[0]
    assert first['latent_rms']==1 and first['logit_rms']==1 and first['prediction_changes']==1
    (tmp_path/'full_r2.npz').unlink()
    with pytest.raises(FileNotFoundError):
        cross_control_differences(tmp_path,'original','full',3)


def test_completed_case_checkpoint_is_verified_and_never_overwritten(tmp_path):
    import json,hashlib
    from scripts.run_lion_guidance_speed import checkpoint_case
    case=tmp_path/'case';case.mkdir()
    (case/'prediction.bin').write_bytes(b'fixed evidence')
    target=tmp_path/'drive'/'case'
    checkpoint_case(case,target)
    receipt=json.loads((target/'checkpoint_manifest.json').read_text())
    assert (target/'prediction.bin').read_bytes()==b'fixed evidence'
    assert receipt['prediction.bin']['sha256']==hashlib.sha256(b'fixed evidence').hexdigest()
    with pytest.raises(FileExistsError):checkpoint_case(case,target)


def test_checkpoint_rejects_lost_drive_before_creating_local_directories(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.run_lion_guidance_speed import checkpoint_case
    monkeypatch.setattr('os.path.ismount', lambda path: False)
    def forbidden_mkdir(*args, **kwargs):
        raise AssertionError('Must reject unmounted Drive before creating directories')
    monkeypatch.setattr(Path, 'mkdir', forbidden_mkdir)
    with pytest.raises(OSError, match='Drive'):
        checkpoint_case(tmp_path, '/content/drive/MyDrive/study/case')


def test_unguided_matches_zero_proximal_without_guidance_work(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from legacy_trajectory import _rng_state
    from lion_guidance_pilot import sample
    calls, grad_modes, readouts = [], [], []
    class Scheduler:
        def __init__(self, **kwargs):
            self.alphas_cumprod = torch.linspace(.99, .5, 1000)
            self.final_alpha_cumprod = torch.tensor(1.)
        def set_timesteps(self, count, device):
            self.timesteps = torch.arange(999, -1, -10, device=device)
        def step(self, eps, t, x):
            return SimpleNamespace(pred_original_sample=x-.2*eps, prev_sample=x-.1*eps)
    class Chamfer:
        def __call__(self, q, reference):
            calls.append('chamfer')
            distance=(q-reference).square().sum(-1)
            indices=torch.arange(q.shape[1]).expand(q.shape[:2])
            return distance, distance, indices, indices
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(DDIMScheduler=Scheduler))
    monkeypatch.setitem(sys.modules, 'third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D',
                        SimpleNamespace(chamfer_3DDist=Chamfer))
    def prior(x, t, condition_input, clip_feat):
        grad_modes.append(torch.is_grad_enabled())
        return .1*x+.01*condition_input.mean()
    lion=SimpleNamespace(vae=SimpleNamespace(global2style=lambda z:z.clone()), priors=[None,prior])
    h=torch.arange(16).float().reshape(1,16,1,1)/10
    state=dict(local_latent=h, shape_latent=torch.ones(1,2,1,1),
               noise=torch.full_like(h,.01), rng_after_prepare=_rng_state())
    scheduler_seen=[]
    zero,_=sample(lion,state,8,'cheap',tau=0,
                  scheduler_observer=lambda scheduler:scheduler_seen.append(scheduler.timesteps[-8:].tolist()))
    assert scheduler_seen==[[79,69,59,49,39,29,19,9]]
    assert len(calls)==8
    calls.clear();grad_modes.clear()
    unguided,records=sample(lion,state,8,'unguided',callback=lambda h,z,i:readouts.append(i))
    assert calls==[] and grad_modes==[False]*8
    assert readouts==[2,4,6,8] and all(not r['full_backward'] for r in records)
    assert len(unguided)==len(zero)==4
    for actual,expected in zip(unguided,zero):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_integrity_checkpoint_rejects_mutation_before_persisting_success(tmp_path):
    import json
    import torch
    from scripts.eval_multibackbone_source import state_hash
    from scripts.run_lion_guidance_speed import checkpoint_integrity
    model=torch.nn.Linear(2,2).eval().requires_grad_(False)
    before=[state_hash(model)]
    checkpoint_integrity([model],before,tmp_path/'local',tmp_path/'backup')
    record=json.loads((tmp_path/'backup'/'integrity.json').read_text())
    assert record['before']==record['after']==before and record['status']=='verified'
    with torch.no_grad():model.weight.add_(1)
    with pytest.raises(ValueError,match='Model state'):
        checkpoint_integrity([model],before,tmp_path/'failed',tmp_path/'bad_backup')
    assert not (tmp_path/'failed').exists() and not (tmp_path/'bad_backup').exists()


def test_full14_plan_keeps_every_object_and_final_partial_batch():
    from scripts.run_lion_guidance_speed import phase_layout, iter_batches
    selection=dict(calibration=[1,2,3,4],screen=list(range(32)),confirm=list(range(256)))
    plan=phase_layout('full14',selection,128)
    assert plan['ids']==list(range(2468))
    assert len(plan['corruptions'])==14 and 'background' not in plan['corruptions']
    assert plan['seeds']==[0] and plan['repetitions']==1
    batches=list(iter_batches(plan['ids'],128))
    assert [i for _,ids in batches for i in ids]==list(range(2468))
    assert len(batches)==20 and len(batches[-1][1])==36
    profile=phase_layout('profile',selection,1)
    assert profile['ids']==selection['calibration'] and profile['repetitions']==3


def test_profile_warms_the_same_remainder_shapes_as_full_evaluation():
    from scripts.run_lion_guidance_speed import warm_batch_sizes
    for size in (32,64,128):
        assert warm_batch_sizes('profile',list(range(size)),size)==warm_batch_sizes('full14',list(range(2468)),size)
    assert warm_batch_sizes('profile',[140,1557,1808,2312],1)==[1]


def test_original_terminal_has_one_readout_and_preserves_classifier_rng(monkeypatch):
    import numpy as np
    import torch
    from scripts import eval_multibackbone_guidance as shared
    from scripts.run_lion_guidance_speed import terminal_inference
    model=torch.nn.Linear(3,3).eval().requires_grad_(False)
    calls=[]
    def reconstruct(raw,dataset,corruption,lion,graph,original,ops,**kwargs):
        calls.append(('reconstruct',torch.is_grad_enabled(),kwargs))
        return raw+1,{}
    def classify(x):
        calls.append(('classify',torch.is_grad_enabled()))
        return model(x.mean(1))+torch.rand(len(x),3)
    monkeypatch.setattr(shared,'reconstruct_geometry',reconstruct)
    before=torch.random.get_rng_state()
    logits,prob=terminal_inference(np.zeros((2,4,3),dtype=np.float32),'uniform',None,None,None,classify,[model])
    assert logits.shape==(2,1,3) and prob.shape==(2,3)
    assert len(calls)==2 and calls[0]==('reconstruct',True,{'method':shared.ORIGINAL_METHOD})
    assert calls[1]==('classify',False)
    assert torch.equal(before,torch.random.get_rng_state())
    torch.testing.assert_close(prob,logits[:,0].softmax(-1))


def test_batch_selection_uses_complete_safe_throughput_not_accuracy():
    from scripts.analyze_lion_guidance_speed import choose_profile
    profiles=[dict(status='accepted',phase='profile',batch_size=b,rows=18*b,
                   source_overlay_sha256='same-source',asset_manifest_sha256='same-assets',
                   profile_metrics=dict(objects_per_second=b,max_allocated_mib=10000,
                                        max_reserved_mib=12000,gpu_total_mib=81920)) for b in (32,64,128)]
    profiles[0]['accuracy']=100
    profiles[-1]['profile_metrics']['max_reserved_mib']=70000
    assert choose_profile(profiles)['batch_size']==64
    profiles[-1]['profile_metrics']['max_reserved_mib']=12000
    assert choose_profile(profiles)['batch_size']==128
    with pytest.raises(ValueError,match='grid'):choose_profile(profiles[:-1])
    profiles[-1]['rows']-=1
    with pytest.raises(ValueError,match='coverage'):choose_profile(profiles)


def test_continuation_preserves_canonical_seeds_order_and_final_partial_batch():
    from scripts.run_lion_guidance_speed import corruption_schedule, phase_layout, iter_batches
    selection={'calibration':[140,1557,1808,2312]}
    layout=phase_layout('continue14',selection,128)
    assert layout['ids']==list(range(2468)) and layout['seeds']==[0] and layout['repetitions']==1
    work=list(corruption_schedule(layout['corruptions'],['uniform','gaussian'],20))
    assert len(work)==12 and work[0]==(2,'impulse',40)
    batches=list(iter_batches(layout['ids'],128))
    assert len(batches)==20 and batches[-1]==(19,list(range(2432,2468)))
    ci,_,block=work[0]
    assert ci*1000+batches[0][0]*10==2000
    assert block+batches[-1][0]==59
    assert 2468*len(work)*3==88848


def test_continuation_rejects_unknown_duplicate_or_all_skips():
    import pytest
    from scripts.run_lion_guidance_speed import corruption_schedule
    for skip in (['unknown'],['a','a'],['a','b']):
        with pytest.raises(ValueError):list(corruption_schedule(['a','b'],skip,20))
    assert list(corruption_schedule(['a','b'],[],20))==[(0,'a',0),(1,'b',20)]


def test_background_is_a_separate_complete_single_pass_with_ragged_batch():
    from scripts.run_lion_guidance_speed import phase_layout, iter_batches, warm_batch_sizes, corruption_schedule
    layout=phase_layout('background',{},128)
    assert layout==dict(ids=list(range(2468)),corruptions=['background'],seeds=[0],repetitions=1)
    batches=list(iter_batches(layout['ids'],128))
    assert len(batches)==20 and batches[-1]==(19,list(range(2432,2468)))
    assert warm_batch_sizes('background',layout['ids'],128)==[36,128]
    assert corruption_schedule(layout['corruptions'],[],20)==[(0,'background',0)]


def test_background_arm_metadata_matches_actual_35_step_policy():
    from scripts.run_lion_guidance_speed import arm_protocols
    arms=['terminal','original','cheap']
    protocols=arm_protocols('background',arms)
    assert protocols['terminal']==dict(steps=35,gamma=.01,eta=.01,p=.95,readouts=1,method='repository original eval/raw with common controlled backend')
    for arm in ('original','cheap'):
        assert protocols[arm]==dict(steps=35,gamma=.03,eta=.003,p=.95,readouts=4)
    for phase in ('full14','continue14','profile'):
        assert arm_protocols(phase,arms)['terminal']['steps']==5
        assert arm_protocols(phase,arms)['original']['steps']==8
    assert arm_protocols('screen',arms) is None


@pytest.mark.parametrize('skips,expected', [([], 'Verified Drive progress path required'),
                                          (['uniform'], 'Corruption skips require')])
def test_background_parser_accepts_terminal_but_rejects_continuation_skips(monkeypatch,capsys,skips,expected):
    import sys
    from scripts import run_lion_guidance_speed as run
    arguments=['run','--asset-root','unused','--manifest-sha256','0'*64,
               '--selection','unused','--output','unused','--progress-drive','/content/drive/MyDrive/test',
               '--phase','background','--arms','terminal','original','cheap']
    if skips:arguments+=['--skip-corruptions',*skips]
    monkeypatch.setattr(sys,'argv',arguments)
    monkeypatch.setattr(run.os.path,'ismount',lambda path:False)
    with pytest.raises(SystemExit):run.main()
    assert expected in capsys.readouterr().err


def test_background_uniform_uses_full_data_and_normal_arm_horizons():
    from scripts.run_lion_guidance_speed import phase_layout, arm_protocols, sampler_steps
    assert phase_layout('background_uniform',{},128)==dict(ids=list(range(2468)),corruptions=['background'],seeds=[0],repetitions=1)
    actual=arm_protocols('background_uniform',['terminal','original','cheap'])
    assert actual==arm_protocols('full14',['terminal','original','cheap'])
    assert [actual[a]['steps'] for a in ('terminal','original','cheap')]==[5,8,8]
    for corruption in ('background','uniform','gaussian'):
        assert sampler_steps('background_uniform',corruption)==8
    assert sampler_steps('background','background')==35
    assert sampler_steps('full14','uniform')==8


def test_background_uniform_terminal_forwards_explicit_horizon_without_relabelling(monkeypatch):
    import numpy as np
    import torch
    from scripts import eval_multibackbone_guidance as shared
    from scripts.run_lion_guidance_speed import terminal_inference
    seen=[]
    def reconstruct(raw,dataset,corruption,*args,**kwargs):
        seen.append((corruption,kwargs));return raw,{}
    monkeypatch.setattr(shared,'reconstruct_geometry',reconstruct)
    logits,pm=terminal_inference(np.zeros((2,4,3),dtype=np.float32),'background',None,None,None,
        lambda x:torch.zeros(len(x),40),[],steps=5)
    assert seen==[('background',{'method':shared.ORIGINAL_METHOD,'original_steps':5})]
    assert logits.shape==(2,1,40) and pm.shape==(2,40)


def test_background_uniform_parser_accepts_terminal(monkeypatch,capsys):
    import sys
    from scripts import run_lion_guidance_speed as run
    monkeypatch.setattr(sys,'argv',['run','--asset-root','unused','--manifest-sha256','0'*64,
        '--selection','unused','--output','unused','--progress-drive','/content/drive/MyDrive/test',
        '--phase','background_uniform','--arms','terminal','original','cheap'])
    monkeypatch.setattr(run.os.path,'ismount',lambda path:False)
    with pytest.raises(SystemExit):run.main()
    assert 'Verified Drive progress path required' in capsys.readouterr().err


@pytest.mark.parametrize('override,method', [(True,'3dd_original'),(0,'3dd_original'),
    (5.0,'3dd_original'),(8,'3dd_original'),(101,'3dd_original'),(5,'gsd_latent_spectral_v1')])
def test_original_horizon_override_rejects_invalid_values_before_preprocessing(override,method):
    from scripts.eval_multibackbone_guidance import reconstruct_geometry
    with pytest.raises(ValueError,match='original horizon'):
        reconstruct_geometry(None,'modelnet40_c','background',None,None,None,None,
                             original_steps=override,method=method)


def test_original_horizon_override_preserves_background_name_and_old_default():
    import torch
    from types import SimpleNamespace
    from scripts import eval_multibackbone_guidance as shared
    seen=[]
    ops=SimpleNamespace(normalize=lambda x:(x,None,None),upsample_all=lambda x,n:x,
        rotate_pointcloud=lambda x:x,rotateback_pointcloud=lambda x:x)
    scheduler=SimpleNamespace(config=dict(beta_end=.02,beta_start=.0001,beta_schedule='linear',
        clip_sample=False,num_train_timesteps=1000,prediction_type='epsilon',set_alpha_to_one=True),
        timesteps=torch.arange(990,-1,-10),final_alpha_cumprod=torch.tensor(1.))
    def reconstruct(**kwargs):
        seen.append(kwargs['steps_back_local']);kwargs['scheduler_observer'](scheduler)
        return kwargs['x']
    for override in (None,5,35):
        geometry,_=shared.reconstruct_geometry(torch.ones(1,2048,3),'modelnet40_c','background',
            None,None,reconstruct,ops,method=shared.ORIGINAL_METHOD,device='cpu',original_steps=override)
        assert geometry.shape==(1,2048,3)
    assert seen==[35,5,35]


def test_uniform_phase_metadata_uses_normal_checkpoints_and_preserves_old_phase():
    from scripts.run_lion_guidance_speed import configure_phase
    for phase in ('background','background_uniform'):
        base=dict(normal_reverse_steps=8,background_reverse_steps=35,
                  trajectory_selection=dict(normal=[2,4,6],background=[9,18,27]))
        configure_phase(base,phase)
        assert base['normal_reverse_steps']==8
        assert base['background_reverse_steps']==(8 if phase=='background_uniform' else 35)
        assert base['trajectory_selection']['background']==([2,4,6] if phase=='background_uniform' else [9,18,27])
