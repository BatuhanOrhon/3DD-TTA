"""One physical LION-frame roundtrip and two separately audited readouts."""
import importlib

import numpy as np
import pytest
import torch

from legacy_trajectory import rng_hash
from shape_latent_refresh import prepare_reference
from test_shape_latent_refresh import RefreshVAE


def feature():
    return importlib.import_module('lion_reencode')


def test_reencoding_refreshes_both_posteriors_with_common_draws_and_new_z_condition():
    vae = RefreshVAE().eval()
    base, initial = prepare_reference(torch.ones(2, 2048, 3), vae)
    decoded = torch.full((2, 2048, 3), .2)
    before = rng_hash()
    selected, proof = feature().reencode(decoded, base, initial, vae)
    assert rng_hash() == before
    qz = vae.encode_global(decoded)
    z = qz.sample_given_rho(initial['eps_z'])
    qh = vae.encode_local(decoded, vae.global2style(z))
    h = qh.sample_given_rho(initial['eps_h'])
    assert torch.equal(selected['shape_latent'].flatten(1), z)
    assert torch.equal(selected['local_latent'].flatten(1), h)
    assert not torch.equal(selected['local_latent'], base['local_latent'])
    assert torch.equal(selected['noise'], base['noise'])
    assert torch.equal(proof['bridge_points'], decoded)
    assert torch.equal(proof['cycle_points'], vae.decoder(None, beta=None, context=h, style=z))
    assert torch.equal(proof['new_h_mu'], qh.mu)


@pytest.mark.parametrize('bad', ['nan', 'frame', 'batch', 'training', 'grad'])
def test_bridge_rejects_invalid_geometry_or_model(bad):
    vae = RefreshVAE().eval()
    base, initial = prepare_reference(torch.ones(2, 2048, 3), vae)
    decoded = torch.full((2, 2048, 3), .2)
    if bad == 'nan': decoded[0, 0, 0] = float('nan')
    if bad == 'frame': decoded = decoded[:, :1024]
    if bad == 'batch': decoded = decoded[:1]
    if bad == 'training': vae.train()
    if bad == 'grad': vae.register_parameter('weight', torch.nn.Parameter(torch.ones(())))
    with pytest.raises(ValueError): feature().reencode(decoded, base, initial, vae)


def test_two_pass_probabilities_have_equal_phase_weights_not_mean_logits():
    m = feature()
    first = np.array([[[5., 0.], [0., 1.], [0., 1.], [0., 1.]]])
    second = np.array([[[0., 2.], [0., 2.], [0., 2.], [0., 2.]]])
    result = m.readouts(first, second)
    softmax = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
    assert np.allclose(result['first_probabilities'], softmax(first).mean(1))
    assert np.allclose(result['combined_probabilities'], .5 * (softmax(first).mean(1) + softmax(second).mean(1)))
    assert np.array_equal(result['combined_prediction'], result['combined_probabilities'].argmax(-1))
    with pytest.raises(ValueError): m.readouts(first[:, :2], second)


def test_observer_keeps_first_and_second_terminal_and_cycle_separate():
    m = feature()
    vae = RefreshVAE().eval()
    classify = lambda points: {'pointmae': points.mean((1, 2))[:, None].repeat(1, 40)}
    obs = m.TwoPassObserver(vae=vae, heads={'pointmae': torch.nn.Identity().eval()},
        classify_decoded=classify, horizon=8, num_classes=40, checkpoint_steps=[2, 4, 6])
    z = torch.zeros(1, 128, 1, 1)
    h = torch.zeros(1, 8192, 1, 1)
    for observer, steps in [(obs.first, [2, 4, 6]), (obs.second, [1, 2, 3])]:
        observer.initial(input_points=torch.ones(1, 2048, 3))
        for step in steps: observer.clean(h, z, step - 1, 0)
        observer.final_state(h, z, 0)
    obs.first.terminal({'pointmae': torch.ones(1, 40)})
    obs.cycle_logits = {'pointmae': np.full((1, 40), 2.)}
    obs.terminal({'pointmae': torch.full((1, 40), 3.)})
    arrays = obs.arrays('pointmae')
    assert arrays['trajectory_logits'].shape == (1, 4, 40)
    assert np.all(arrays['first_trajectory_logits'][:, -1] == 1)
    assert np.all(arrays['cycle_logits'] == 2)
    assert np.all(arrays['trajectory_logits'][:, -1] == 3)
    assert obs.second.horizon == 3 and obs.first.horizon == 8


def test_second_pass_uses_bridge_frame_new_anchor_own_z_and_three_step_schedule(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = importlib.import_module('scripts.eval_lion_reencode')
    vae = RefreshVAE().eval()
    x = torch.ones(1, 2048, 3)
    calls = []
    holder = []
    def observer_factory(center, scale):
        observer = feature().TwoPassObserver(vae=vae, heads={'pointmae': torch.nn.Identity().eval()},
            classify_decoded=lambda points: {'pointmae': torch.ones(len(points), 40)},
            horizon=8, num_classes=40, checkpoint_steps=[2, 4, 6])
        holder.append(observer)
        return observer
    def reconstruct(**parameters):
        calls.append(parameters)
        obs = parameters['trajectory_observer']
        obs.initial(input_points=parameters['x'])
        h = parameters['prepared_state']['local_latent']
        z = parameters['prepared_state']['shape_latent']
        for step in obs.steps: obs.clean(h, z, step - 1, 0)
        obs.final_state(h, z, 0)
        return torch.full_like(x, .2 if len(calls) == 1 else .3)
    def geometry(*args, **kwargs):
        return args[5](x=x, lion=SimpleNamespace(vae=vae), steps_back_local=8,
            prepared_state=None, state_observer=lambda state: None,
            trajectory_observer=observer_factory(None, None), scheduler_observer=lambda scheduler: None,
            gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False), {}
    monkeypatch.setattr(runner.depth, 'reconstruct_geometry', geometry)
    cfg = {'executed_source_sha256': 'source', 'asset_manifest_sha256': 'assets'}
    output, metrics = runner.reconstruct_geometry(x, 'modelnet40_c', 'gaussian',
        SimpleNamespace(vae=vae), None, reconstruct, None,
        reencode_config=cfg, paired_state_path=tmp_path/'paired.npz', observer_factory=observer_factory)
    assert [call['steps_back_local'] for call in calls] == [8, 3]
    assert torch.equal(calls[1]['x'], torch.full_like(x, .2))
    assert calls[1]['trajectory_observer'] is holder[0].second
    assert calls[1]['state_observer'] is None
    assert 'chamfer_reference' not in calls[1]  # Original tta uses the new h1 XYZ.
    assert not torch.equal(calls[0]['prepared_state']['shape_latent'], calls[1]['prepared_state']['shape_latent'])
    assert torch.equal(calls[0]['prepared_state']['noise'], calls[1]['prepared_state']['noise'])
    assert torch.equal(output, torch.full_like(x, .3))
    assert metrics['reencode']['second_reverse_steps'] == 3
    assert (tmp_path/'paired.reencode.npz').is_file()


def test_config_preserves_first_n8_b35_and_reports_second_pm_with_explicit_names(tmp_path):
    from test_eval_shape_latent import args
    runner = importlib.import_module('scripts.eval_lion_reencode')
    config, = runner.configurations(args(tmp_path))
    assert config['normal_reverse_steps'] == 8 and config['background_reverse_steps'] == 35
    assert config['csv_policy'] == 'probability_mean'
    assert config['trajectory_selection']['normal_loop_counts'] == [1, 2, 3, 3]
    assert config['first_trajectory_selection']['normal_loop_counts'] == [2, 4, 6, 8]
    for key, value in [('normal_reverse_steps', 5), ('seed', 1), ('csv_policy', 'terminal')]:
        with pytest.raises(ValueError): runner.validate_config(dict(config, **{key: value}))


def test_worker_preloads_later_policy_source_before_source_seal(tmp_path, monkeypatch):
    import sys
    from test_eval_shape_latent import args
    runner = importlib.import_module('scripts.eval_lion_reencode')
    config, = runner.configurations(args(tmp_path))
    bundle = tmp_path/'bundle'; bundle.mkdir()
    import json
    (bundle/'config.json').write_text(json.dumps(config))
    monkeypatch.delitem(sys.modules, 'scripts.analyze_legacy_trajectory', raising=False)
    def sealed_worker(*args, **kwargs):
        assert 'scripts.analyze_legacy_trajectory' in sys.modules, 'Policy source would first load after the source seal'
    monkeypatch.setattr(runner.shared, 'worker', sealed_worker)
    runner.worker([bundle])


def test_cycle_logits_reject_nonfinite_and_wrong_class_count():
    obs = feature().TwoPassObserver(vae=RefreshVAE().eval(),
        heads={'pointmae': torch.nn.Identity().eval()}, classify_decoded=None,
        horizon=8, num_classes=40, checkpoint_steps=[2, 4, 6])
    for values in (torch.full((1,40), float('nan')), torch.zeros(1,15)):
        with pytest.raises(ValueError): obs.cycle({'pointmae': values})
