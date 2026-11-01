"""Contracts for second-only inference from a sealed first-pass reference."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def runner():
    return importlib.import_module('scripts.eval_lion_reencode_depth')


@pytest.mark.parametrize('steps,counts,names', [
    (5,[2,3,4,5],['clean_02','clean_03','clean_04','terminal']),
    (8,[2,4,6,8],['clean_02','clean_04','clean_06','terminal'])])
def test_depth_config_preserves_first_reference_and_four_pm_candidates(tmp_path,steps,counts,names):
    from test_eval_shape_latent import args
    m=runner()
    phase=tmp_path/'reference';phase.mkdir()
    (phase/'evidence_manifest.json').write_text('{}')
    cfg,=m.configurations(args(tmp_path)+['--second-steps',str(steps),'--replay-phase',str(phase)])
    assert cfg['normal_reverse_steps']==8 and cfg['background_reverse_steps']==35
    assert cfg['trajectory_selection']['normal_loop_counts']==counts
    assert cfg['trajectory_selection']['background_candidate_names']==names
    assert cfg['first_trajectory_selection']['normal_loop_counts']==[2,4,6,8]
    assert cfg['reencode']['second_reverse_steps']==steps and cfg['csv_policy']=='probability_mean'
    for changed in [dict(seed=1),dict(csv_policy='terminal'),dict(normal_reverse_steps=5)]:
        with pytest.raises(ValueError):m.validate_config(dict(cfg,**changed))


def test_checked_replay_file_rejects_tampering_and_path_escape(tmp_path):
    m=runner();p=tmp_path/'state.npz';p.write_bytes(b'original')
    manifest={'state.npz':m.shared.source.file_identity(p)}
    assert m.checked_file(tmp_path,manifest,'state.npz')==p
    p.write_bytes(b'tampered')
    with pytest.raises(ValueError):m.checked_file(tmp_path,manifest,'state.npz')
    with pytest.raises(ValueError):m.checked_file(tmp_path,manifest,'../other.npz')


def test_replay_arrays_reject_nonfinite_logits_and_keep_archived_first_weights():
    m=runner()
    first=np.tile(np.arange(40,dtype=np.float32),(2,4,1))
    second=-first
    archive=dict(first_trajectory_logits=first,cycle_logits=first[:,0],trajectory_logits=first+1)
    result=m.replay_arrays(archive,dict(trajectory_logits=second))
    assert np.array_equal(result['first_trajectory_logits'],first)
    assert np.array_equal(result['reference_n3_trajectory_logits'],first+1)
    assert np.allclose(result['combined_probabilities'],.5*(result['first_probabilities']+result['second_probabilities']))
    for key in ('cycle_logits','first_trajectory_logits','trajectory_logits'):
        bad=dict(archive);bad[key]=np.full_like(archive[key],np.nan)
        with pytest.raises(ValueError):m.replay_arrays(bad,dict(trajectory_logits=second))


def test_archived_real_cache_preserves_selected_posterior_and_rng_when_available():
    # Local optional audit supplements portable contracts, not a substitute for them.
    root=Path(__file__).resolve().parents[1]/'result/preparation/lion_reencode_20261101/operator/pilot_extracted'
    if not root.exists():pytest.skip('Immutable raw reference is outside Git')
    m=runner();manifest=json.loads((root/'evidence_manifest.json').read_text())
    cfgpath=next((root/'raw').glob('*/*/*/config.json'))
    cfg=json.loads(cfgpath.read_text())
    original=np.asarray(cfg['selected_original_ids'][:32],dtype=np.int64)
    state,proof,arrays,diag,refs=m.load_reference(root,manifest,'gaussian',original,device='cpu')
    assert torch.equal(state['shape_latent'].flatten(1),torch.from_numpy(proof['selected_z']))
    assert torch.equal(state['local_latent'].flatten(1),torch.from_numpy(proof['selected_h']))
    assert np.array_equal(state['noise'].numpy(),proof['diffusion_noise'])
    from legacy_trajectory import _rng_state_hash
    assert _rng_state_hash(state['rng_after_prepare'])==diag['trajectory']['second']['initial_rng_sha256']
    assert arrays['first_trajectory_logits'].shape==(32,4,40) and refs


def test_independent_noisy_state_acceptance_rejects_wrong_alpha_or_actual_hash():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'knowledge/evidence/lion_reencode_depth_20261101/analyze_phase.py'
    spec=importlib.util.spec_from_file_location('depth_acceptance',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proof=dict(selected_h=np.ones((1,8192),dtype=np.float32),diffusion_noise=np.zeros((1,8192,1,1),dtype=np.float32))
    expected=np.full((1,8192,1,1),.5,dtype=np.float32)
    digest=module.tensor_hash(expected)
    module.validate_noisy_state(proof,.25,digest)
    for alpha,actual in [(.36,digest),(.25,'wrong')]:
        with pytest.raises(ValueError):module.validate_noisy_state(proof,alpha,actual)


def test_replay_rejects_changed_host_source_or_resolved_behavior():
    import copy
    m=runner()
    source_names=['tta.py','utilities_3dd_tta.py','shared_trajectory.py','scripts/eval_multibackbone_guidance.py',
        'scripts/eval_multibackbone_source.py','classifier_backbones.py','models_mate/Point_MAE.py']
    old={key:'same' for key in ('lion_state_before','classifier_state_before_sha256','dataset_provenance',
        'guidance_protocol','resolved_lion_config','resolved_model_config','preprocessing','source_classifier_preprocessing')}
    old.update(executed_source_files={'repo/'+name:dict(bytes=1,sha256='hash') for name in source_names},native_extensions={})
    current=copy.deepcopy(old);current['executed_source_files']['repo/scripts/eval_lion_reencode_depth.py']={'sha256':'new'}
    m.validate_reference_host(old,current)
    for key in ('resolved_lion_config','resolved_model_config','preprocessing'):
        bad=copy.deepcopy(current);bad[key]='changed'
        with pytest.raises(ValueError):m.validate_reference_host(old,bad)
    bad=copy.deepcopy(current);bad['executed_source_files']['repo/tta.py']['sha256']='changed'
    with pytest.raises(ValueError):m.validate_reference_host(old,bad)
