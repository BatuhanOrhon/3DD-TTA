import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts import run_cloudfixer_point2vec as worker


@pytest.mark.parametrize(('inherited', 'expected'), [(None, ':4096:8'), (':16:8', ':16:8')])
def test_runtime_preserves_inherited_cuda_workspace(monkeypatch, inherited, expected):
    if inherited is None:
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    else:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', inherited)
    class StopBeforeCuda(Exception):
        pass
    def stop(args, manifest):
        assert worker.os.environ.get('CUBLAS_WORKSPACE_CONFIG') == expected
        raise StopBeforeCuda
    monkeypatch.setattr(worker, 'verify_runtime_files', stop)
    with pytest.raises(StopBeforeCuda):
        worker.run(None, {}, {}, {}, None, None, None, None)


def test_manifest_binds_classifier_and_four_locked_policies():
    from scripts.prepare_cloudfixer_transfer import manifest
    a, old = manifest({}, 'point2vec'), manifest({})
    assert a['classifier_sha256'] != old['classifier_sha256']
    assert a['classifier']['class_order'][0::39] == ['airplane', 'xbox']
    assert a['registration']['batch_size'] == 1 and a['registration']['seed'] == 2
    assert a['registration']['corruptions'] == sorted(old['registration']['corruptions'])
    assert a['registration']['policies']['proximal_pm']['probe_indices'] == [20, 10, 0]
    assert a['registration']['policies']['proximal_terminal']['probe_indices'] == [0]
    assert 'scripts/run_cloudfixer_point2vec.py' in a['overlay_files']
    assert 'scripts/run_cloudfixer_point2vec.py' not in old['overlay_files']


@pytest.mark.parametrize('arm', list(worker.POLICIES))
def test_production_spends_only_selected_forwards_and_preserves_dense_readout_rng(arm):
    from cloudfixer_diffusion import grid
    class Diffusion:
        T = 500
        gamma = staticmethod(lambda t: t)
        alpha = staticmethod(lambda gamma,target_tensor: torch.ones_like(gamma)[:, :, None]*.8)
        sigma = staticmethod(lambda gamma,target_tensor: torch.ones_like(gamma)[:, :, None]*.6)
        phi = staticmethod(lambda state,t,mask: state*.25)

    torch.manual_seed(2)
    x = torch.randn(1,12,3)
    mask = torch.ones(1,12)
    def classify(cloud):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        return cloud.mean((1,2))[:,None] + torch.rand(len(x),40)
    head = MagicMock(side_effect=classify)
    state = torch.get_rng_state().clone()
    config = worker.POLICIES[arm]
    result = worker.proximal_policy(Diffusion(), head, x, mask, config)
    assert result[3] == config['probe_indices']
    assert head.call_count == len(config['probe_indices'])
    assert result[4] == config['steps']+1
    dense = worker.proximal_policy(Diffusion(), head, x, mask, config, dense=True)
    columns = [dense[3].index(i) for i in result[3]]
    assert dense[3] == grid(steps=config['steps'],start=config['start'])
    assert torch.equal(result[0],dense[0][:,columns])
    assert torch.equal(result[2],dense[2][:,columns])
    assert torch.equal(result[1],result[0].double().softmax(-1).mean(1))
    assert torch.equal(state,torch.get_rng_state())


@pytest.mark.parametrize(('enabled', 'warn_only'), [(False, False), (True, True)])
def test_proximal_failure_restores_algorithm_flags_and_rng(enabled, warn_only):
    old = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    def fail(*args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        torch.rand(10)
        raise ValueError('interrupted')
    from unittest.mock import patch
    x = torch.ones(1, 4, 3)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        state = torch.get_rng_state()
        with patch('cloudfixer_diffusion.sample', fail), pytest.raises(ValueError, match='interrupted'):
            worker.proximal_policy(None, None, x, None, worker.POLICIES['proximal_pm'])
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
        assert torch.equal(state, torch.get_rng_state())
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])


def test_rng_checkpoint_resumes_all_streams(monkeypatch):
    state = torch.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:state)
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda t:None)
    snap = json.loads(worker.canonical(worker.rng_state()))
    def draws():
        import random
        return random.random(), np.random.randn(), torch.randn(()).item()
    original = draws()
    worker.restore_rng(snap)
    assert draws() == original


def test_checkpoint_head_setup_and_strict_load(monkeypatch):
    import sys
    from types import ModuleType
    head=MagicMock()
    head.cuda.return_value=head
    model=ModuleType('classifier.point2vec.models')
    model.Point2VecClassification=lambda:head
    ckpt=ModuleType('classifier.point2vec.utils.checkpoint')
    state={'cls_head.8.weight':torch.zeros(40,256),'cls_head.8.bias':torch.zeros(40)}
    ckpt.extract_model_checkpoint=lambda path:state
    monkeypatch.setitem(sys.modules,model.__name__,model)
    monkeypatch.setitem(sys.modules,ckpt.__name__,ckpt)
    worker.load_head('weights.ckpt',{'state_keys':2})
    head.setup.assert_called_once_with()
    head.load_state_dict.assert_called_once_with(state,strict=True)
    state['cls_head.8.weight']=torch.zeros(10,256)
    with pytest.raises(ValueError,match='number of classes'):
        worker.load_head('weights.ckpt',{'state_keys':2})


def block_payload(unit, labels):
    ids = unit['ids']
    n = len(ids)
    data = dict(unit=json.dumps(unit), ids=ids, labels=labels[ids],
                vote_log_probs=np.zeros((n,40)), input_sha256=['a'*64]*n,
                seconds=np.ones((n,4)), peak_mb=np.ones((n,4)))
    for arm in worker.ARMS:
        idx = worker.POLICIES[arm]['probe_indices'] if arm in worker.POLICIES else []
        width = len(idx) or 1
        logits = np.zeros((n,width,40))
        logits[np.arange(n), :, labels[ids]] = 1
        p = np.exp(logits)/np.exp(logits).sum(-1,keepdims=True)
        data[arm+'_logits'] = logits
        data[arm+'_probs'] = p
        data[arm+'_entropy'] = -(p*np.log(p)).sum(-1)
        if idx:
            data[arm+'_indices'] = idx
            data[arm+'_logsnr'] = np.arange(width)
    return data


def encoded_block(data):
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    return buf.getvalue()


def test_saved_block_recount_rejects_wrong_original_label_and_changed_entropy():
    labels = np.array([2,4])
    unit = dict(campaign='test',ids=[1,0])
    data = block_payload(unit, labels)
    verified = worker.validate(encoded_block(data),unit,labels)
    assert all(r['n_correct']==2 for r in worker.counts(verified).values())
    with pytest.raises(ValueError,match='labels'):
        worker.validate(encoded_block(data),unit,labels[::-1])
    data['proximal_pm_entropy']+=1
    with pytest.raises(ValueError,match='entropy'):
        worker.validate(encoded_block(data),unit,labels)


def test_fresh_interpreter_reopens_committed_block_and_continues_rng(tmp_path, monkeypatch):
    import random
    import subprocess
    import sys
    gen = torch.Generator().manual_seed(99)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', gen.get_state)
    n = worker.registration()['block_size']
    unit = dict(campaign='test', ids=list(range(n)))
    data = block_payload(unit, np.arange(n))
    before = worker.sha(worker.canonical(worker.rng_state()))
    for _ in unit['ids']:
        random.random(), np.random.random(), torch.rand(()), torch.rand((), generator=gen)
    data.update(rng_before_sha256=before, rng_after=worker.canonical(worker.rng_state()).decode())
    worker.commit_unit(worker.MountStore(tmp_path), unit, encoded_block(data))
    expected = [random.random(), np.random.random(), torch.rand(()).item(), torch.rand((), generator=gen).item()]
    code = """
import json, random, sys
import numpy as np
import torch
from scripts import run_cloudfixer_point2vec as worker
gen = torch.Generator()
torch.cuda.set_rng_state = gen.set_state
unit = {'campaign':'test', 'ids':list(range(worker.registration()['block_size']))}
store = worker.MountStore(sys.argv[1])
data = worker.load_units(store, 'test')[worker.unit_key(unit)]['data']
p = worker.restore_block(data, unit, np.arange(len(unit['ids'])), sys.argv[2])
assert all(v['n_correct'] == len(unit['ids']) for v in worker.counts(p).values())
print(json.dumps([random.random(), np.random.random(), torch.rand(()).item(), torch.rand((), generator=gen).item()]))
"""
    actual = subprocess.check_output([sys.executable, '-c', code, str(tmp_path), before], cwd=worker.ROOT)
    assert json.loads(actual) == expected
    with pytest.raises(ValueError, match='RNG stream'):
        worker.restore_block(encoded_block(data), unit, np.arange(n), 'wrong')


def test_author_entrypoint_checks_original_geometry_rng_and_saved_logits(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class Point2VecClassification(torch.nn.Module):
        def forward(self, x):
            return x.mean() * torch.arange(40) + torch.rand(len(x), 40)

    dataset = [(torch.tensor([[i, 2., 3.]]), i, torch.ones(1, 1), torch.tensor([0])) for i in range(4)]
    adapt = SimpleNamespace(DataLoader=torch.utils.data.DataLoader)
    options = SimpleNamespace(dataset='test')
    gen = torch.Generator().manual_seed(7)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', gen.get_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', gen.set_state)
    original = worker.rng_state()
    def seed():
        worker.restore_rng(original)
        torch.manual_seed(2)

    shift = 0.
    corrupt_head_input = False
    def reconstruct(args, model, x, mask, ind):
        return x + torch.rand_like(x) + shift
    def forward(args, classifier, optimizer, diffusion_model, x, mask, ind):
        xyz = adapt.cloudfixer(args, diffusion_model, x, mask, ind)
        plain = classifier(x if corrupt_head_input else xyz)
        with torch.no_grad():
            return plain, classifier(xyz).softmax(-1).log()
    adapt.cloudfixer, adapt.forward_and_adapt = reconstruct, forward

    def main(args):
        seed()
        head = torch.nn.DataParallel(Point2VecClassification())
        for x, _, mask, ind in adapt.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0):
            head(x)
            adapt.forward_and_adapt(args, head, None, None, x, mask=mask, ind=ind)
    adapt.main = main
    seed()
    head = torch.nn.DataParallel(Point2VecClassification())
    ids = [int(i.item()) for i in torch.utils.data.DataLoader(range(4), batch_size=1, shuffle=True)][:2]
    records, source, plain, vote = [], [], [], []
    with worker.audit_native(adapt, records):
        for i in ids:
            x, _, mask, ind = torch.utils.data.default_collate([dataset[i]])
            source.append(head(x).numpy())
            p, v = adapt.forward_and_adapt(options, head, None, None, x, mask=mask, ind=ind)
            plain.append(p.numpy())
            vote.append(v.numpy()[0])
    payload = {key: np.stack([r[key] for r in records]) for key in records[0]}
    payload.update(ids=np.array(ids), labels=np.array(ids), input_sha256=payload['native_input_sha256'],
                   rng_after=worker.canonical(worker.rng_state()).decode(), source_logits=np.stack(source),
                   cloudfixer_logits=np.stack(plain), vote_log_probs=np.stack(vote))
    shift = -10.  # Independent native reconstruction can change argmax; each head must still replay exactly.
    cache = worker.MountStore(tmp_path/'raw')
    evidence = worker.saved_author_prefix(adapt, options, payload, tmp_path, cache, 'smoke', 'raw-sha')
    assert evidence['native_xyz_plain_vote_replay_exact'] and evidence['plain_cf_argmax_changes'] == 2
    archive = tmp_path/'test_author_reference.npz'
    expected_bytes = archive.read_bytes()
    archive.unlink()
    with monkeypatch.context() as patcher:
        patcher.setattr(worker, 'author_prefix', MagicMock(side_effect=AssertionError('Repeated completed audit')))
        assert worker.saved_author_prefix(adapt, options, payload, tmp_path, cache, 'smoke', 'raw-sha') == evidence
        assert archive.read_bytes() == expected_bytes
        with pytest.raises(ValueError, match='prediction block'):
            worker.saved_author_prefix(adapt, options, payload, tmp_path, cache, 'smoke', 'different-block')
    for key, match in [('ids', 'shuffled IDs'), ('native_rng_sha256', 'native_rng_sha256')]:
        wrong = dict(payload)
        wrong[key] = wrong[key][::-1]
        with pytest.raises(ValueError, match=match):
            worker.author_prefix(adapt, options, wrong, tmp_path)
    with pytest.raises(AssertionError):
        worker.author_prefix(adapt, options, dict(payload, cloudfixer_logits=payload['cloudfixer_logits']+1), tmp_path)
    corrupt_head_input = True
    with pytest.raises(ValueError, match='adapted XYZ'):
        worker.author_prefix(adapt, options, payload, tmp_path)


def test_pointmae_hypothesis_package_distinct_from_historic_campaign():
    from scripts.prepare_cloudfixer_transfer import manifest
    from scripts.run_cloudfixer_hypotheses import registration
    b,old=manifest({},hypotheses=True),manifest({})
    assert b['registration']==registration()
    assert b['registration']!=old['registration']
    assert b['classifier_sha256']==old['classifier_sha256']
    assert 'scripts/analyze_cloudfixer_hypotheses.py' in b['overlay_files']
    assert len(b['prior_manifest_sha256'])==64
    with pytest.raises(ValueError,match='registered for Point-MAE'):
        manifest({},'point2vec',hypotheses=True)
