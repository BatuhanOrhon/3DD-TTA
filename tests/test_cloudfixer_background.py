import json
from pathlib import Path
from zipfile import BadZipFile

import pytest
import torch

from cloudfixer_background import BackgroundGeometry, EXPERIMENT_ARMS, anchor_weights
from cloudfixer_background_checkpoint import block_name, restore_or_score
from cloudfixer_rotation import rotation_policy
from cloudfixer_store import MountStore
from test_cloudfixer_diffusion import ConstantEpsilon

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'cfgs/cloudfixer_background_pilot.json').read_text())


def test_isolation_duplicates_and_rotation():
    x = torch.arange(12.).reshape(1, 12, 1).expand(-1, -1, 3).clone()
    x[0, -1] = 100
    x = torch.cat((x, x[:, :2]), dim=1)
    valid = torch.ones_like(x[..., 0], dtype=torch.bool)
    valid[0, -1] = False
    w = anchor_weights(x, valid, 5)
    torch.testing.assert_close(w, anchor_weights(x[..., [1, 2, 0]], valid, 5))
    assert w[0, -2] == w[0, 0] and w[0, -1] == 0 and 0 < w[0, 11] < .5


def test_scalar_matches_total_force_coefficients_and_view_count():
    x = torch.randn((1, 100, 3), generator=torch.Generator().manual_seed(11))
    x[0, -1] = 100
    valid = torch.ones_like(x[..., 0], dtype=torch.bool)
    fw = torch.arange(100).reshape(1, -1)
    scalar = BackgroundGeometry(x, valid, 'bg_scalar', CONFIG)
    weighted = BackgroundGeometry(x, valid, 'bg_weighted', CONFIG)
    left = scalar.pairs(10, fw, fw, valid, valid)
    right = weighted.pairs(10, fw, fw, valid, valid)
    torch.testing.assert_close(left[2].sum() + left[3].sum(), right[2].sum() + right[3].sum())
    assert not torch.equal(left[2], right[2])
    outputs = []
    for arm in ('bg_view', 'bg_random_view'):
        geo = BackgroundGeometry(x, valid, arm, CONFIG)
        geo.pairs(10, fw, fw, valid, valid)
        outputs.append(geo.view(x, 123))
    assert outputs[0].shape == outputs[1].shape and 64 <= outputs[0].shape[1] < 100


@pytest.mark.parametrize('arm', EXPERIMENT_ARMS)
def test_policy_grid_and_calls(arm):
    x = torch.randn((1, 70, 3), generator=torch.Generator().manual_seed(12))
    valid = torch.ones_like(x[..., 0], dtype=torch.bool)
    model = ConstantEpsilon()
    result = rotation_policy(model, lambda cloud: cloud.mean(1), x, valid, None, arm, background_config=CONFIG)
    assert result.diagnostics['classifier_calls'] == 3
    assert len(model.calls) == 9 and model.calls[-3:] == [20, 10, 0]
    assert torch.isfinite(result.logits).all()
    assert model.calls == result.diagnostics['grid']
    if arm == 'bg_high':
        assert model.calls == CONFIG['high_grid']


def test_interrupted_block_survives_new_worker_and_nonprefix_restore(tmp_path):
    units = [dict(campaign='same', ids=[i]) for i in range(3)]
    counts = [0, 0, 0]

    def value(i):
        counts[i] += 1
        if i == 1 and counts[i] == 1:
            raise InterruptedError('simulated quota loss')
        return bytes([i])

    def run(i):
        store = MountStore(tmp_path)
        def check(payload):
            assert payload == bytes([i])
        return restore_or_score(store, units[i], set(store.keys('unit_')), lambda: value(i), check)

    assert run(2) == b'\x02'
    with pytest.raises(InterruptedError):
        run(1)
    assert not (tmp_path / block_name(units[1])).exists()
    assert [run(i) for i in range(3)] == [bytes([i]) for i in range(3)]
    assert counts == [1, 2, 1]
    (tmp_path / block_name(units[2])).write_bytes(b'corrupted')
    with pytest.raises(BadZipFile):
        run(2)
    assert counts == [1, 2, 1]


def test_host_copy_restores_only_missing_drive_blocks(tmp_path):
    from cloudfixer_background_checkpoint import import_mirror, mirror_group
    campaign = 'a' * 64
    units = [dict(campaign=campaign, ids=[i]) for i in range(2)]
    root = tmp_path/'drive'
    cache = MountStore(root/('background_' + campaign[:16]))
    for unit in units:
        restore_or_score(cache, unit, set(), lambda: b'predictions', lambda _: None)
    mirror = tmp_path/'host.zip'
    mirror_group(cache, units, mirror)
    (cache.folder/block_name(units[0])).unlink()
    import_mirror(root, mirror)
    for unit in units:
        assert restore_or_score(cache, unit, set(cache.keys('unit_')), None, lambda _: None) == b'predictions'


def test_runner_resumes_gaps_on_new_vm_without_rescoring_completed_pairs(tmp_path, monkeypatch):
    import numpy as np
    from contextlib import nullcontext
    from types import SimpleNamespace
    from cloudfixer_store import canonical, sha
    from scripts import run_cloudfixer_rotation as run
    from scripts.analyze_cloudfixer_rotation import read_records, report, background_report

    ids = list(range(6))
    labels = np.arange(6)
    env = dict(model_hashes=['m', 'm'], native='first')
    monkeypatch.setattr(run, 'init_runtime', lambda *_: (None, None, None, None, labels, env))
    monkeypatch.setattr(run, 'model_hash', lambda _: 'm')
    monkeypatch.setattr(run, 'dataset_for', lambda *args: args[2])
    monkeypatch.setattr(run, 'phase_scope', lambda *_: (('background', 'rotation'), ids, [0],
                        {'identity': {}, 'bg_high': {}}))
    monkeypatch.setattr(run, 'item', lambda dataset, ident, seed, label: (ident, None, None, None))
    counts = []
    def score(model, head, x, mask, idx, arm, params, *, rng):
        counts.append(rng)
        if rng == 4 and env['native'] == 'first':
            raise InterruptedError('CUs exhausted mid-block')
        logits = np.zeros((3, 40))
        logits[:, rng] = 2
        pm = np.exp(logits[0]) / np.exp(logits[0]).sum()
        return dict(logits=logits.tolist(), prediction=rng, probabilities=pm.tolist(),
                    seconds=.25, confidence=pm.max(), peak_mib=15., cuda_ms={},
                    diagnostics={'denoiser_calls': 9, 'classifier_calls': 3}), None
    monkeypatch.setattr(run, 'run_arm', score)
    spec = {'classifier': {'class_order': []}, 'registration': {'bootstrap_draws': 100, 'bootstrap_seed': 1}}
    store = MountStore(tmp_path/'drive')
    def attempt(name):
        signature = sha(canonical(env))
        store.put('smoke_' + signature + '.json', canonical(dict(runtime=signature)))
        out = tmp_path / name
        out.mkdir()
        (out/'config.json').write_bytes(canonical(dict(phase='background', spec=spec, run_id=name)))
        run.evaluate(SimpleNamespace(phase='background', output=out), spec, store,
                     SimpleNamespace(operation=lambda _: nullcontext()))
        return out
    with pytest.raises(InterruptedError):
        attempt('one')
    env['native'] = 'fresh-build'
    counts.clear()
    out = attempt('two')
    # ID 2 receives both scored arms only for Rotation: Background was committed on VM one.
    assert counts.count(2) == 2
    cfg, rows = read_records(out)
    assert len(rows) == 12
    assert rows['background', 0, 2]['runtime_sha256'] != rows['background', 0, 4]['runtime_sha256']
    derived = tmp_path/'derived'
    derived.mkdir()
    metrics, _ = report(cfg, rows, raw_output=out, derived_output=derived)
    cfg['phase'] = 'background_smoke'  # Partial fixture has only two conditions.
    background_report(cfg, rows, metrics, derived)
    assert (derived/'pm_vs_terminal.csv').is_file()


def test_noise_pm_interaction_uses_each_terminal_and_clusters_seeds(tmp_path):
    import csv
    import numpy as np
    from scripts.analyze_cloudfixer_rotation import noise_pm_interactions
    arms = ['identity', 'bg_mid', 'bg_high', 'bg_start_only']
    cfg = dict(conditions=['background'], seeds=[0, 1], ids=[0, 1], params=dict.fromkeys(arms),
               spec={'registration': {'bootstrap_draws': 100}})
    rows = {}
    for ident in cfg['ids']:
        for seed in cfg['seeds']:
            entries = {}
            for arm in arms:
                logits = np.tile([2., 0.], (3, 1))
                if arm == 'bg_high' and ident == 1:
                    logits[-1] = [0., 1.]
                entries[arm] = dict(logits=logits.tolist())
            rows['background', seed, ident] = dict(label=0, arms=entries)
    noise_pm_interactions(cfg, rows, tmp_path)
    with (tmp_path/'noise_pm_interaction.csv').open() as f:
        r = next(r for r in csv.DictReader(f) if r['candidate']=='bg_high' and r['reference']=='identity')
    assert float(r['pm_change_pp']) == 0 and float(r['terminal_change_pp']) == -50
    assert float(r['interaction_pp']) == 50 and int(r['n_original_ids']) == 2
