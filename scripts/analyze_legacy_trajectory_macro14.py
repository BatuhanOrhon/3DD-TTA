"""Explore shared prediction policies on the full-data 10-step macro14 trace.

This CPU-only development analyzer reads an immutable full all-15 bundle,
validates its trace, and scores policies on the 14 non-Background corruptions.
It does not run inference or choose an entropy temperature.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research_artifacts import CORRUPTIONS
from scripts.analyze_legacy_trajectory import (
    Bundle,
    clustered_deltas,
    load_trace,
    policies,
    probabilities,
    write_csv,
)


EXPECTED_NORMAL_CANDIDATES = ['clean_03', 'clean_05', 'clean_08', 'terminal']
MACRO14_CORRUPTIONS = [name for name in CORRUPTIONS if name != 'background']
POLICY_NAMES = (
    'terminal',
    'minimum_entropy',
    'probability_mean',
    'entropy_weighted_mean',
    'latest_clean_terminal_mean',
    'fixed_probe_1',
    'fixed_probe_2',
    'fixed_probe_3',
)


def entropy_weighted_probabilities(probability: np.ndarray,
                                   entropy: np.ndarray) -> tuple:
    """Return p-mean and normalized ``1 - H`` weights, with equal fallback."""
    p = np.asarray(probability, dtype=np.float64)
    h = np.asarray(entropy, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[1] == 0 or p.shape[2] < 2:
        raise ValueError('probabilities must have shape N x candidates x classes')
    if h.shape != p.shape[:2]:
        raise ValueError('entropy shape must match N x candidates')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('probabilities must be finite and nonnegative')
    if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError('probabilities must sum to one')
    if not np.isfinite(h).all() or np.any((h < 0) | (h > 1)):
        raise ValueError('normalized entropy must be finite and between 0 and 1')

    unnormalized = 1.0 - h
    totals = unnormalized.sum(axis=1, keepdims=True)
    weights = np.divide(unnormalized, totals, out=np.zeros_like(unnormalized),
                        where=totals > 0)
    no_weight = totals[:, 0] == 0
    if np.any(no_weight):
        weights[no_weight] = 1.0 / p.shape[1]
    return np.einsum('nk,nkc->nc', weights, p), weights


def trajectory_policy_predictions(logits: np.ndarray,
                                  candidate_names: list) -> dict:
    """Compute common10-step candidates without labels or corruption routing."""
    names = list(candidate_names)
    if names != EXPECTED_NORMAL_CANDIDATES:
        raise ValueError('Expected normal10 candidates in the locked order')
    p, entropy = probabilities(logits)
    if p.shape[1] != len(names):
        raise ValueError('Candidate names do not match logits')
    terminal = names.index('terminal')
    existing = policies(logits, terminal=terminal)
    weighted, weights = entropy_weighted_probabilities(p, entropy)
    latest = names.index('clean_08')
    latest_pair = 0.5 * (p[:, latest] + p[:, terminal])
    return {
        'terminal': existing['terminal'],
        'minimum_entropy': existing['minimum_entropy'],
        'probability_mean': existing['probability_mean'],
        'entropy_weighted_mean': weighted.argmax(axis=1),
        'latest_clean_terminal_mean': latest_pair.argmax(axis=1),
        'fixed_probe_1': p[:, 0].argmax(axis=1),
        'fixed_probe_2': p[:, 1].argmax(axis=1),
        'fixed_probe_3': p[:, 2].argmax(axis=1),
        'entropy_weights': weights,
        'entropy': entropy,
    }


def analyze_macro14_groups(groups: list) -> tuple:
    """Build per-corruption metrics and paired example rows for one seed."""
    by_corruption = {group['corruption']: group for group in groups}
    if set(by_corruption) != set(MACRO14_CORRUPTIONS):
        raise ValueError('Trace must contain exactly the 14 non-Background corruptions')

    per_corruption, paired, diagnostics = [], [], []
    for corruption in MACRO14_CORRUPTIONS:
        group = by_corruption[corruption]
        if group['seed'] != 0:
            raise ValueError('This screen expects the locked seed0 trace')
        ids = np.asarray(group['indices'])
        targets = np.asarray(group['targets'])
        logits = np.asarray(group['logits'])
        if ids.shape != (2468,) or not np.array_equal(ids, np.arange(2468)):
            raise ValueError('Each macro14 corruption must cover original IDs 0..2467')
        if targets.shape != (2468,) or logits.shape != (2468, 4, 40):
            raise ValueError('Unexpected labels/logits shape for a normal10 trace')
        candidates = group['candidates']
        predictions = trajectory_policy_predictions(logits, candidates)
        terminal_correct = predictions['terminal'] == targets
        probability_mean_correct = predictions['probability_mean'] == targets

        for policy in POLICY_NAMES:
            predicted = predictions[policy]
            correct = predicted == targets
            per_corruption.append({
                'corruption': corruption,
                'seed': group['seed'],
                'policy': policy,
                'n': len(targets),
                'n_correct': int(correct.sum()),
                'accuracy': float(correct.mean()),
                'delta_pp_vs_terminal': float(100 * (correct.mean() - terminal_correct.mean())),
                'corrected': int((correct & ~terminal_correct).sum()),
                'broken': int((~correct & terminal_correct).sum()),
                'corrected_vs_probability_mean': int((correct & ~probability_mean_correct).sum()),
                'broken_vs_probability_mean': int((~correct & probability_mean_correct).sum()),
            })
            for i, sample_id in enumerate(ids):
                paired.append({
                    'corruption': corruption,
                    'seed': group['seed'],
                    'index': int(sample_id),
                    'policy': policy,
                    'target': int(targets[i]),
                    'prediction': int(predicted[i]),
                    'correct': int(correct[i]),
                    'terminal_correct': int(terminal_correct[i]),
                    'probability_mean_correct': int(probability_mean_correct[i]),
                    'corrected_vs_probability_mean': int(
                        correct[i] and not probability_mean_correct[i]),
                    'broken_vs_probability_mean': int(
                        not correct[i] and probability_mean_correct[i]),
                })

        diagnostics.append({
            'corruption': corruption,
            'seed': group['seed'],
            'mean_entropy_weight_by_candidate': {
                name: float(predictions['entropy_weights'][:, index].mean())
                for index, name in enumerate(candidates)
            },
        })

    return per_corruption, paired, diagnostics


def validate_full_seed0(config: dict) -> None:
    """Reject pilots, partial traces, and protocols outside the approved host."""
    expected = list(CORRUPTIONS)
    cli = config.get('cli', {})
    trajectory = config.get('trajectory', {})
    if (config.get('execution_status') != 'complete' or config.get('status') != 'complete'
            or config.get('seed') != 0 or config.get('completed_examples') != 37020
            or config.get('expected_classifications') != 37020
            or config.get('corruptions') != expected or config.get('selected_indices')
            or not config.get('model_state_unchanged')
            or config.get('model_state_before') != config.get('model_state_after')):
        raise ValueError('Input must be a complete, frozen-state seed0 all15 full-data run')
    if (cli.get('batch_size') != 70 or cli.get('M_max') != 800
            or cli.get('isolate_policy') != 'legacy' or cli.get('max_batches') != 0
            or cli.get('denoising_step_normal') != 10
            or cli.get('denoising_step_bg') != 30):
        raise ValueError('Input does not match the approved Legacy B70/M800 10/30 host')
    if (trajectory.get('mode') != 'observe'
            or trajectory.get('checkpoints') != {'10': [3, 5, 8], '30': [8, 15, 23]}):
        raise ValueError('Input does not contain the locked clean trajectory checkpoints')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_macro14(bundle_path: Path, output: Path) -> dict:
    """Validate one full seed0 archive and write a fresh macro14 report."""
    bundle = Bundle(bundle_path)
    try:
        config = bundle.config
        validate_full_seed0(config)
        groups = load_trace(bundle)
        macro_groups = [group for group in groups if group['corruption'] != 'background']
        rows, paired, diagnostics = analyze_macro14_groups(macro_groups)
        source_config_sha = hashlib.sha256(bundle.read('config.json')).hexdigest()
        input_sha = _sha256(Path(bundle_path)) if Path(bundle_path).is_file() else None
        intervals = clustered_deltas(paired)
    finally:
        bundle.close()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_corruption_seed.csv', rows)
    write_csv(output / 'paired_examples.csv', paired)
    summary = []
    for policy in POLICY_NAMES:
        policy_rows = [row for row in rows if row['policy'] == policy]
        summary.append({
            'policy': policy,
            'scope': 'macro14',
            'seed': 0,
            'n_corruptions': len(policy_rows),
            'total_examples': sum(row['n'] for row in policy_rows),
            'total_correct': sum(row['n_correct'] for row in policy_rows),
            'macro_accuracy': float(np.mean([row['accuracy'] for row in policy_rows])),
            'delta_pp_vs_terminal': float(np.mean([row['delta_pp_vs_terminal'] for row in policy_rows])),
            'corrected': sum(row['corrected'] for row in policy_rows),
            'broken': sum(row['broken'] for row in policy_rows),
            'corrected_vs_probability_mean': sum(
                row['corrected_vs_probability_mean'] for row in policy_rows),
            'broken_vs_probability_mean': sum(
                row['broken_vs_probability_mean'] for row in policy_rows),
        })
    report = {
        'schema_version': 1,
        'purpose': 'Exploratory offline policy comparison; no tau fitted or selected.',
        'scope': {
            'dataset': 'modelnet40_c',
            'severity': 5,
            'seed': 0,
            'corruptions': MACRO14_CORRUPTIONS,
            'examples_per_corruption': 2468,
            'total_examples': 14 * 2468,
            'horizon': 10,
            'candidates': EXPECTED_NORMAL_CANDIDATES,
        },
        'input': {
            'path': str(bundle_path),
            'sha256': input_sha,
            'config_sha256': source_config_sha,
            'resolved_commit': config.get('resolved_commit'),
            'baseline_archive_sha256': config.get('baseline_archive_sha256'),
        },
        'summary': summary,
        'per_corruption': rows,
        'index_cluster_intervals': intervals,
        'diagnostics': diagnostics,
        'limitations': [
            'This is one-seed development analysis on previously reviewed ModelNet40-C.',
            'The intervals resample original IDs and do not estimate seed or trajectory variability.',
            'Entropy weighting uses normalized entropy with w proportional to 1-H; no tau is fitted.',
            'The inference rule is shared across all 14 corruptions and never reads labels.',
            'Full-trace policies do not early-stop and these results do not measure speedup.',
        ],
    }
    with (output / 'analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='complete full all15 seed0 handoff or run ZIP')
    parser.add_argument('--output', type=Path, required=True,
                        help='new output directory; existing paths are refused')
    args = parser.parse_args()
    report = analyze_macro14(args.bundle, args.output)
    for row in report['summary']:
        print('%-28s %8.4f%% (%+7.4f pp vs terminal)' % (
            row['policy'], 100 * row['macro_accuracy'], row['delta_pp_vs_terminal']))
    print('Validated macro14 analysis written to', args.output)


if __name__ == '__main__':
    main()
