"""Fixed shape-pilot CLI/worker boundaries; no GPU or asset access."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import test_chamfer_depth as depth_tests


def feature():
    assert importlib.util.find_spec('scripts.eval_shape_latent') is not None, 'shape runner is missing'
    from scripts import eval_shape_latent
    return eval_shape_latent


def args(root, stage='development', count=64):
    result = depth_tests.DepthConfigTests().args(root)
    (root/'ids.json').write_text(json.dumps(dict(dataset='modelnet40_c', n_examples=2468,
                                               stage=stage, indices=list(range(count)))))
    return result


@pytest.mark.parametrize('arm', ['baseline', 'updated', 'anchor', 'proposal', 'guided3'])
def test_fixed_n8_shape_arms_and_exact_scientific_metadata(tmp_path, arm):
    runner = feature()
    config, = runner.configurations(args(tmp_path)+['--shape-arm', arm])
    assert config['method'] == runner.METHOD
    assert config['normal_reverse_steps'] == 8
    assert config['background_reverse_steps'] == 35
    assert config['guidance_protocol'] == dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
    assert config['trajectory_selection']['normal_loop_counts'] == [2, 4, 6, 8]
    assert config['decoder_style'] == ('original' if arm == 'baseline' else 'updated')
    assert config['shape_latent']['global_forward_calls'] == {'proposal': 1, 'guided3': 3}.get(arm, 0)
    assert config['shape_latent']['global_initial_timestep'] == (20 if arm in ('proposal', 'guided3') else None)
    assert config['shape_latent']['radius_fraction'] == (None if arm in ('baseline', 'updated') else .05)
    runner.validate_config(config)
    for key, value in [('normal_reverse_steps', 10), ('decoder_style', 'wrong'), ('selected_original_ids', [0])]:
        mutated = dict(config, **{key: value})
        with pytest.raises(ValueError): runner.validate_config(mutated)


def test_technical_smoke_is_a_distinct_registered32_id_scope(tmp_path):
    runner = feature()
    config, = runner.configurations(args(tmp_path, 'smoke', 32)+['--shape-arm', 'guided3', '--shape-technical-smoke'])
    assert config['corruptions'] == ['gaussian', 'background']
    assert config['technical_smoke'] and config['evaluation_n_examples'] == 32
    with pytest.raises(ValueError):
        runner.configurations(args(tmp_path, 'development', 32)+['--shape-arm', 'proposal', '--shape-technical-smoke'])
    with pytest.raises(ValueError):
        runner.configurations(args(tmp_path, 'smoke', 32)+['--shape-arm', 'proposal'])


@pytest.mark.parametrize('extra', [['--normal-reverse-steps', '5'], ['--gamma', '.01'],
                                  ['--eta', '.01'], ['--p', '.96'], ['--seed', '1']])
def test_initial_pilot_rejects_altered_registered_settings(tmp_path, extra):
    with pytest.raises(ValueError):
        feature().configurations(args(tmp_path)+['--shape-arm', 'anchor']+extra)


def test_worker_validates_before_any_runtime_or_asset_access(tmp_path):
    runner = feature()
    config, = runner.configurations(args(tmp_path)+['--shape-arm', 'proposal'])
    directory = tmp_path/'bundle'
    directory.mkdir()
    config['shape_latent']['radius_fraction'] = .5
    (directory/'config.json').write_text(json.dumps(config))
    with patch.object(runner.shared, 'worker', side_effect=AssertionError('must not reach GPU')):
        with pytest.raises(ValueError): runner.worker([directory])
