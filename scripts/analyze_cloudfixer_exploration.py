#!/usr/bin/env python3
"""Offline paired-logit diagnostics; refuses incomplete registered coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import canonical, sha, unit_key
from scripts.run_cloudfixer_exploration import registration, units, validate_payload, validate_provenance


def correctness_auc(score: np.ndarray, correct: np.ndarray) -> float | None:
    """P(correct has greater score than wrong), with half-credit for ties."""
    positive, negative = np.asarray(score)[correct], np.sort(np.asarray(score)[~correct])
    if not len(positive) or not len(negative):
        return None
    lower = np.searchsorted(negative, positive, side='left')
    upper = np.searchsorted(negative, positive, side='right')
    return float((lower+(upper-lower)/2).sum()/(len(positive)*len(negative)))


def calibration(confidence: np.ndarray, correct: np.ndarray) -> list:
    result = []
    indices = np.minimum((np.asarray(confidence)*10).astype(int), 9)
    for index in range(10):
        mask = indices == index
        result.append(dict(lower=index/10, upper=(index+1)/10, count=int(mask.sum()),
            mean_max_probability=float(np.mean(confidence[mask])) if mask.any() else None,
            accuracy=float(np.mean(correct[mask])) if mask.any() else None))
    return result


def _mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def _candidate(correct, entropy, normalized_entropy, confidence, terminal) -> dict:
    return dict(count=len(correct), n_correct=int(correct.sum()), accuracy=float(correct.mean()),
        mean_entropy_correct=_mean(entropy, correct), mean_entropy_wrong=_mean(entropy, ~correct),
        mean_max_probability_correct=_mean(confidence, correct), mean_max_probability_wrong=_mean(confidence, ~correct),
        lower_entropy_correctness_auroc=correctness_auc(1-normalized_entropy, correct),
        max_probability_correctness_auroc=correctness_auc(confidence, correct),
        max_probability_reliability_bins=calibration(confidence, correct),
        intermediate_correct_terminal_wrong=int((correct & ~terminal).sum()),
        intermediate_wrong_terminal_correct=int((~correct & terminal).sum()))


def _readout(prediction, labels, corruptions) -> dict:
    correct = prediction == labels
    per_corruption = {name: dict(count=int((corruptions == name).sum()),
        n_correct=int(correct[corruptions == name].sum()), accuracy=float(correct[corruptions == name].mean()))
        for name in sorted(set(corruptions))}
    return dict(count=len(correct), n_correct=int(correct.sum()), accuracy=float(correct.mean()),
        macro_accuracy=float(np.mean([c['accuracy'] for c in per_corruption.values()])),
        per_corruption=per_corruption)


def analyze(records: list, wanted: list) -> dict:
    expected = {unit_key(u): u for u in wanted}
    actual = {unit_key(r['unit']): r for r in records}
    if (len(expected) != len(wanted) or len(actual) != len(records) or set(actual) != set(expected)):
        raise ValueError('Incomplete, duplicate or unexpected registered coverage; no aggregate/ranking')
    grouped = {}
    sources, models = set(), set()
    for key, unit in expected.items():
        p = validate_payload(actual[key]['data'], unit)
        sources.add(str(p['source_sha256']))
        models.add(tuple(p['model_hashes'].tolist()))
        grouped.setdefault(unit['config_id'], []).append((unit, p))
    if len(sources) != 1 or len(models) != 1:
        raise ValueError('Mixed source or frozen-model identities in paired analysis')
    configurations = []
    for config_id, batches in grouped.items():
        config = batches[0][0]['config']
        seen = set()
        for unit, _ in batches:
            identities = {(unit['corruption'], identifier) for identifier in unit['ids']}
            if seen & identities:
                raise ValueError('Duplicate original-object coverage')
            seen.update(identities)
        payloads = [p for _, p in batches]
        logits = np.concatenate([p['logits'] for p in payloads])
        labels = np.concatenate([p['labels'] for p in payloads])
        probabilities = np.concatenate([p['probabilities'] for p in payloads])
        entropy = np.concatenate([p['entropy'] for p in payloads])
        normalized_entropy = np.concatenate([p['normalized_entropy'] for p in payloads])
        confidence = np.concatenate([p['max_probability'] for p in payloads])
        corruptions = np.array([u['corruption'] for u, _ in batches for _ in u['ids']])
        correct = logits.argmax(-1) == labels[:, None]
        terminal = correct[:, -1]
        metadata = json.loads(str(payloads[0]['sampler']))
        candidates = []
        for index in range(logits.shape[1]):
            stats = _candidate(correct[:, index], entropy[:, index], normalized_entropy[:, index], confidence[:, index], terminal)
            stats.update(candidate=index, loop=metadata['candidate_loops'][index],
                diffusion_index=metadata['candidate_indices'][index], diffusion_time=metadata['candidate_indices'][index]/500,
                terminal=index == logits.shape[1]-1,
                interpretation='deterministic t0 clean estimate' if index == logits.shape[1]-1 else 'pre-correction clean x0 estimate')
            stats['per_corruption'] = {}
            for name in sorted(set(corruptions)):
                mask = corruptions == name
                stats['per_corruption'][name] = _candidate(correct[mask, index], entropy[mask, index],
                    normalized_entropy[mask, index], confidence[mask, index], terminal[mask])
            candidates.append(stats)
        slots = [1, 3, 5, 8] if logits.shape[1] == 9 else [0, 1, 2, 3]
        readouts = dict(terminal=_readout(logits[:, -1].argmax(-1), labels, corruptions),
            old4_pm=_readout(probabilities[:, slots].mean(1).argmax(-1), labels, corruptions),
            all_candidates_pm=_readout(probabilities.mean(1).argmax(-1), labels, corruptions))
        configurations.append(dict(config_id=config_id, config=config, candidates=candidates, readouts=readouts,
            actual_indices=metadata['indices'], candidate_indices=metadata['candidate_indices'],
            candidate_times=[index/500 for index in metadata['candidate_indices']],
            candidate_loops=metadata['candidate_loops'], units=len(batches), parity_units=sum(u['config']['steps'] == 8 for u, _ in batches),
            runtime_seconds=sum(float(p['seconds']) for p in payloads),
            corruption_runtime_seconds={name: sum(float(p['seconds']) for u, p in batches if u['corruption'] == name)
                for name in sorted(set(corruptions))},
            peak_gpu_memory_mb=max(float(p['peak_mb']) for p in payloads),
            denoiser_calls=sum(int(p['denoiser_calls']) for p in payloads), classifier_calls_per_batch=logits.shape[1],
            oracle=dict(label_dependent=True, deployable=False, description='Any candidate correct: label-dependent nondeployable diagnostic',
                n_correct=int(correct.any(1).sum()), accuracy=float(correct.any(1).mean()))))
    configurations.sort(key=lambda row: row['config']['steps'])
    ranking = sorted(configurations, key=lambda row: (-row['readouts']['old4_pm']['macro_accuracy'], row['runtime_seconds']))
    return dict(status='complete', completed_units=len(records), expected_units=len(wanted),
        claim='Exploratory reused development subset; no confirmation or automatic winner promotion',
        notes=['Fixed tau per step changes cumulative correction and discretization as N changes.',
            'Floor quarter/half/three-quarter loop positions give different actual times across N; inspect candidate_indices.',
            'All-candidates PM is descriptive; dense N8 changes candidate density and time weighting.',
            'Entropy/correctness association is distinct from max-probability calibration.',
            'Pipeline timing includes all classifier candidates; it is not terminal-only latency.'],
        configurations=configurations,
        exploratory_n_ranking=[dict(steps=row['config']['steps'], old4_pm_macro_accuracy=row['readouts']['old4_pm']['macro_accuracy'],
            runtime_seconds=row['runtime_seconds']) for row in ranking])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Local attempt directory or final attempt ZIP')
    parser.add_argument('--output', required=True, type=Path, help='Analysis JSON; written only after complete validated coverage')
    args = parser.parse_args(argv)
    if args.input.is_dir():
        config = json.loads((args.input/'config.json').read_text())
        files = {p.name: p.read_bytes() for p in args.input.glob('pending_unit_*.npz')}
        environments = {p.name: p.read_bytes() for p in args.input.glob('environment_*.json')}
    else:
        with zipfile.ZipFile(args.input) as archive:
            names = archive.namelist()
            configs = [n for n in names if n.endswith('/config.json')]
            if len(configs) != 1 or len(names) != len(set(names)):
                raise ValueError('Ambiguous attempt archive')
            config = json.loads(archive.read(configs[0]))
            prefix = configs[0].rsplit('/', 1)[0]+'/'
            files = {n[len(prefix):]: archive.read(n) for n in names if n.startswith(prefix+'pending_unit_') and n.endswith('.npz')}
            environments = {n[len(prefix):]: archive.read(n) for n in names if n.startswith(prefix+'environment_') and n.endswith('.json')}
    identity = dict(manifest=config['manifest'], role=config['role'])
    campaign = sha(canonical(identity))
    if config['campaign'] != campaign or config['manifest']['registration'] != registration():
        raise ValueError('Attempt manifest/campaign/registration mismatch')
    wanted = units(campaign, config['manifest']['registration'], config['role'])
    expected = {'pending_'+unit_key(u)+'.npz': u for u in wanted}
    if set(files) != set(expected):
        raise ValueError('Incomplete or unexpected registered archive coverage')
    records = [dict(unit=unit, data=files[name]) for name, unit in expected.items()]
    for record in records:
        payload = validate_payload(record['data'], record['unit'])
        environment_name = 'environment_'+str(payload['environment_sha256'])+'.json'
        if environment_name not in environments:
            raise ValueError('Missing immutable inference environment receipt')
        validate_provenance(payload, config['manifest'], environments[environment_name])
    result = analyze(records, wanted)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
