"""Describe existing smoke/pilot variation; never rerun an inference job."""
from pathlib import Path
import json
import numpy as np

op = Path(__file__).parent
result = {}
for arm in ('original', 'mean', 'prior3', 'proxy3', 'oracle3'):
    runs = [next((op / stage / 'raw' / arm).glob('*/*/*/config.json')).parent
            for stage in ('smoke_extracted', 'pilot_extracted')]
    configs = [json.loads((run / 'config.json').read_text()) for run in runs]
    config_keys = ['classifier_state_before_sha256', 'classifier_state_after_sha256',
                   'lion_state_before', 'lion_state_after', 'executed_source_sha256',
                   'source_snapshot_sha256', 'asset_manifest_sha256', 'selected_original_ids']
    config_equal = {key: configs[0][key] == configs[1][key] for key in config_keys}
    assert all(config_equal.values())
    for corruption in ('gaussian', 'background'):
        batches = [run / 'batches' / corruption / 'batch0000' for run in runs]
        diagnostics = [json.loads((batch / 'diagnostics.json').read_text()) for batch in batches]
        equal = {key: diagnostics[0][key] == diagnostics[1][key]
                 for key in ('reconstruction_rng_before_sha256', 'reconstruction_rng_after_sha256')}
        equal['initial_hashes'] = diagnostics[0]['trajectory']['initial_hashes'] == diagnostics[1]['trajectory']['initial_hashes']
        equal['initial_rng'] = diagnostics[0]['trajectory']['initial_rng_sha256'] == diagnostics[1]['trajectory']['initial_rng_sha256']
        assert all(equal.values())
        with np.load(batches[0] / 'predictions.npz', allow_pickle=False) as a, np.load(batches[1] / 'predictions.npz', allow_pickle=False) as b:
            assert np.array_equal(a['index'], b['index']) and np.array_equal(a['label'], b['label'])
            assert a['trajectory_logits'].shape == b['trajectory_logits'].shape == (32, 4, 40)
            result[arm + '/' + corruption] = dict(
                config_equal=config_equal, equal=equal,
                smoke_correct=int((a['probability_mean'] == a['label']).sum()),
                pilot_correct=int((b['probability_mean'] == b['label']).sum()),
                changed_predictions=int((a['probability_mean'] != b['probability_mean']).sum()),
                max_abs_trajectory_logit_difference=float(np.max(np.abs(a['trajectory_logits'] - b['trajectory_logits']))))
assert len(result) == 10
(op / 'technical_pilot_repeatability.json').write_text(json.dumps(result, indent=2) + '\n')
print('EXISTING_SMOKE_PILOT_SOURCE_STATE_RNG_MATCH_VARIATION_CHECK_COMPLETE')
