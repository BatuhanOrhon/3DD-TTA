import json
import random

import numpy as np
import pytest
import torch

from cloudfixer_diffusion import sample
from cloudfixer_store import canonical, sha
from scripts import run_cloudfixer_hypotheses as runner
from scripts.analyze_cloudfixer_hypotheses import entropy_diagnostics, paired, write_reports
from scripts.run_cloudfixer_exploration import pack, unpack


class Denoiser:
    T = 500

    def gamma(self, t):
        return -4 + t*5

    def alpha(self, g, target_tensor):
        return torch.sigmoid(-g).sqrt().unsqueeze(-1)

    def sigma(self, g, target_tensor):
        return torch.sigmoid(g).sqrt().unsqueeze(-1)

    def phi(self, x, t, mask):
        return x.sin() + .01*torch.rand_like(x)


def head(x):
    # Each source of RNG deliberately changes logits, including class ordering.
    offsets = random.random()+np.random.rand()+torch.rand(len(x),40)
    return x.mean((1,2))[:,None]+offsets


def evaluate(config):
    torch.manual_seed(1989)
    np.random.seed(1989)
    random.seed(1989)
    x = torch.tensor([[[0.,1.,2.],[4.,3.,1.],[2.,1.,2.]],[[0.,1.,0.],[2.,0.,1.],[2.,3.,1.]]])
    return runner.probe_evaluator(Denoiser(),head,x,torch.ones(x.shape[:2]),torch.ones_like(x),config)


def test_actual_observer_uses_common_head_rng_preserves_transport_and_separates_endpoints():
    config = runner.registration()['schedules'][0]
    dense, first = evaluate(dict(config,pilot=True))
    for mode, slots in [('sparse',[6,7,8]),('terminal',[8])]:
        logits, meta = evaluate(dict(config,pilot=True,observation=mode))
        assert torch.equal(logits,dense[:,slots])
        for field in ('denoiser_input_trajectory_sha256','terminal_xyz_sha256','pre_head_rng_sha256','post_head_rng_sha256'):
            assert first[field] == meta[field]
    assert first['pre_head_rng_sha256']==first['post_head_rng_sha256']
    assert np.asarray(first['post_logits']).shape == (2,8,40)
    assert np.asarray(first['pre_extra_t0_logits']).shape == (2,40)
    assert np.asarray(first['pre_extra_t0_to_terminal_rms']).min()>0
    assert np.diff(first['logsnr'],axis=1).min()>0
    assert np.array_equal(np.array(first['correction_rms'])[:,-1],[0.,0.])


@pytest.mark.parametrize('probes', [[],[20,10],[10,20,0],[20,20,0],[5,0],[True,0]])
def test_sparse_indices_reject_ambiguous_or_non_native(probes):
    x = torch.ones(1,2,3)
    with pytest.raises(ValueError):
        sample(Denoiser(),x,torch.ones(1,2),x,steps=8,start=80,tau=0,probe_indices=probes)


def test_heldout_ids_predeclared_once_and_all_three_arms_share_noise():
    from scripts.run_cloudfixer_transfer import registration,seed_for
    reg = runner.registration()
    assert set(reg['confirmation_ids']).isdisjoint(set(reg['pilot_ids'])|set(registration()['smoke_ids']))
    assert sha(canonical(reg['confirmation_ids']))=='6793dfb437afdb6b90c491809d3f738df0713392eb1f239164b6b52cbaf622f9'
    registered=runner.units('b',reg)
    assert len(registered)==1080 and len({canonical(u) for u in registered})==1080
    assert len({seed_for(u) for u in registered[:3]})==1
    assert [u['config']['steps']*u['config']['tau'] for u in registered[:3]]==pytest.approx([.8,1.2,.8])


def test_cluster_pairing_and_within_example_entropy_ties():
    draws=np.array([[0,0],[1,1],[0,1]])
    accuracy=paired(np.array([True, True, False, False]),np.array([False,True,True,False]),[8,8,9,9],draws)
    assert accuracy['repair']==accuracy['harm']==1 and accuracy['delta_pp']==0
    p=np.array([[[.9,.1],[.1,.9]],[[.57,.43],[.57,.43]]])
    value=entropy_diagnostics(p,np.array([[.1,.1],[.8,.8]]),np.array([0,1]),.9)
    assert value['within_example_pair_auc']==value['within_example_mean_auc']==.5
    assert value['mixed_examples']==value['within_example_pairs']==value['wrong_confident']==1


def test_small_complete_analysis_uses_immutable_logits_and_own_terminal(tmp_path):
    reg=runner.registration()
    reg.update(confirmation_ids=[1,2],corruptions=['uniform'],seeds=[0],bootstrap_draws=50)
    manifest={'registration':reg}
    env=canonical(dict(model_hashes=['a'*64,'b'*64],gamma_verified=True,lion_loaded=False,classifier='Point-MAE'))
    label=np.array([0,2,3])
    wanted=runner.units('c',reg)
    records=[]
    for u in wanted:
        logits,meta=evaluate(u['config'])
        meta['correction_rms'] = [[1.] * u['config']['steps'] + [0.]] * len(u['ids'])
        meta['input_rng_sha256']='c'*64
        raw=pack(dict(unit=json.dumps(u),ids=u['ids'],labels=label[u['ids']],logits=logits.numpy(),
            seconds=1.,peak_mb=4.,input_shape=[2,3,3],model_hashes=['a'*64,'b'*64],sampler=json.dumps(meta),
            denoiser_calls=meta['denoiser_calls']))
        data=runner.add_diagnostics(raw,sha(canonical(manifest)),sha(env))
        runner.validate(data,u,label,manifest,env)
        bad=unpack(data)
        bad['post_entropy'][0,0]+=.1
        with pytest.raises(ValueError,match='entropy'):
            runner.validate(pack(bad),u,label,manifest,env)
        records.append(dict(unit=u,data=data))
    write_reports(tmp_path,records,wanted,reg,'c')
    import csv
    rows=list(csv.DictReader((tmp_path/'paired.csv').open()))
    baseline=[r for r in rows if r['method']=='terminal' and r['reference']=='terminal']
    assert len(baseline)==6 and all(float(r['delta_pp'])==float(r['ci95_low_pp'])==float(r['ci95_high_pp'])==0 for r in baseline)
    assert all(float(r['delta_pp'])==0 for r in rows if r['method'].endswith('exp_beta_0') and r['reference'].endswith('equal_pm'))
    diagnostics = json.loads((tmp_path/'diagnostics.json').read_text())
    assert diagnostics['n_units']==3
    for name, config in diagnostics['configurations'].items():
        expected = 2. if name.startswith('N8_') else 4.
        for group in config['late_to_terminal_groups'].values():
            if group['n']:
                assert group['correction_rms_after_late'] == expected
    with pytest.raises(ValueError,match='coverage'):
        write_reports(tmp_path,records[:-1],wanted,reg,'c')


def test_failed_asset_check_records_failed_execution(tmp_path, monkeypatch):
    from pathlib import Path
    from cloudfixer_store import MountStore
    manifest = dict(registration=runner.registration(), prior_manifest_sha256=runner.PRIOR_MANIFEST,
                    overlay_files={'scripts/run_cloudfixer_hypotheses.py': 'a'*64},
                    dataset_files={'label.npy': 'b'*64})
    source = tmp_path/'manifest.json'
    source.write_text(json.dumps(manifest))
    setup = tmp_path/'setup.json'
    setup.write_text('{"attempts": []}')
    monkeypatch.setattr(Path, 'is_mount', lambda path: str(path) == '/content/drive')
    monkeypatch.setattr(runner, 'MountStore', lambda folder: MountStore(tmp_path/'store'))
    output = tmp_path/'failed_run'
    with pytest.raises(FileNotFoundError):
        runner.main(['--manifest', str(source), '--output', str(output),
                     '--drive-folder', '/content/drive/MyDrive/thesis/result/fixture',
                     '--prior-archive', str(tmp_path/'prior.zip'),
                     '--prior-dense-archive', str(tmp_path/'dense.zip'),
                     '--setup-receipt', str(setup), '--dataset-root', str(tmp_path/'missing')])
    assert json.loads((output/'config.json').read_text())['execution_status'] == 'failed'
    assert 'FileNotFoundError' in (output/'failure.txt').read_text()


def test_native_identity_tracks_exact_loaded_elf_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    bindings = {}
    for name in ('knn_cuda._knn', 'pointnet2_ops._ext', 'chamfer', 'pytorch3d._C'):
        path = tmp_path / (name + '.so')
        path.write_bytes(b'\x7fELF' + name.encode())
        binary = SimpleNamespace(__file__=str(path))
        package, _, member = name.partition('.')
        monkeypatch.setitem(sys.modules, package, SimpleNamespace(**{member: binary}) if member else binary)
        bindings[name] = dict(path=str(path), sha256=sha(path.read_bytes()))
    first = runner.native_identity()
    assert first == bindings
    path = tmp_path / 'knn_cuda._knn.so'
    path.write_bytes(b'\x7fELF rebuilt')
    assert sha(canonical(runner.native_identity())) != sha(canonical(first))
    path.write_bytes(b'python source')
    with pytest.raises(ValueError, match='ELF'):
        runner.native_identity()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.native_identity()


def test_mixed_build_recovery_archives_and_checks_each_pilot(tmp_path):
    import zipfile
    reg = runner.registration()
    campaign = 'campaign'
    store = {}
    builds = []
    for revision in ('one', 'two'):
        native = {name: dict(path='/env/'+name+'.so', sha256=sha((name+revision).encode()))
                  for name in ('knn_cuda._knn', 'pointnet2_ops._ext', 'chamfer', 'pytorch3d._C')}
        env = canonical(dict(native_extensions=native))
        digest = sha(env)
        exact = dict(argmax_changes=0, pm_argmax_changes=0)
        batches = [dict(ids=reg['pilot_ids'][i:i+32], old_canonical=dict(exact, status='passed',
                   old_archive_sha256=reg['historical_development_sha256']), old_dense=exact,
                   production=dict(sparse=exact, terminal=exact))
                   for i in range(0, len(reg['pilot_ids']), 32)]
        gate = dict(environment_sha256=digest, campaign=campaign, status='passed', seed=0,
                    corruption=reg['pilot_corruption'], historical_dense_sha256=reg['historical_dense_sha256'],
                    batches=batches)
        store['environment_'+digest+'.json'] = env
        store['pilot_gate_'+digest+'.json'] = canonical(gate)
        builds.append(digest)
        assert runner.read_pilot_environment(store.__getitem__, digest, campaign, reg, output=tmp_path) == env
    archive = tmp_path/'builds.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for p in tmp_path.glob('*.json'):
            z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(store)
        for digest in builds:
            assert runner.read_pilot_environment(z.read, digest, campaign, reg) == store['environment_'+digest+'.json']
    gate_key = 'pilot_gate_'+builds[0]+'.json'
    changed = json.loads(store[gate_key])
    changed['batches'][0]['old_dense']['argmax_changes'] = 1
    store[gate_key] = canonical(changed)
    with pytest.raises(ValueError, match='pilot gate'):
        runner.read_pilot_environment(store.__getitem__, builds[0], campaign, reg)
    store[gate_key] = store['pilot_gate_'+builds[1]+'.json']
    with pytest.raises(ValueError, match='pilot gate'):
        runner.read_pilot_environment(store.__getitem__, builds[0], campaign, reg)
    del store[gate_key]
    with pytest.raises(KeyError):
        runner.read_pilot_environment(store.__getitem__, builds[0], campaign, reg)
