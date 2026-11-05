#!/usr/bin/env python3
"""Frozen paired original-object bootstrap of completed H5 CPU associations."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_h5_reliability as h5

INPUT_NAMES = ('summary.json', 'features.json', 'rows.jsonl')
DRAWS, SEED = 2000, 0
require = h5.accepted.require


def strict_json(raw):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw, parse_constant=reject)


def load_analysis(directory, pins):
    directory = Path(directory)
    require(set(pins) == set(INPUT_NAMES), 'Three exact analysis SHA pins required')
    identities = {}
    for name in INPUT_NAMES:
        identities[name] = h5.accepted.identity(directory/name)
        require(isinstance(pins[name], str) and re.fullmatch('[0-9a-f]{64}', pins[name])
                and identities[name]['sha256'] == pins[name], 'Analysis SHA mismatch: '+name)
    summary = strict_json((directory/'summary.json').read_text())
    features = strict_json((directory/'features.json').read_text())
    rows = [strict_json(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
    require(summary.get('status') in ('complete', 'complete_with_repeat_floor_failure')
            and summary.get('rows') == 2304 and summary.get('ids') == h5.IDS
            and summary.get('corruptions') == list(h5.CORRUPTIONS) and summary.get('repeats') == 3
            and type(summary.get('seed')) is int and summary['seed'] == 0
            and type(summary.get('native_repeat_gates_all_passed')) is bool, 'Incomplete/wrong frozen H5 analysis')
    require(summary['feature_identity'] == identities['features.json'], 'Frozen feature identity differs')
    require((summary['status'] == 'complete') == summary['native_repeat_gates_all_passed'], 'Native gate/status contradiction')
    expected = set(itertools.product(range(3), h5.IDS, h5.CORRUPTIONS, range(4), range(6)))
    for values in (features, rows):
        require(isinstance(values, list) and len(values) == 2304, 'Frozen H5 row coverage differs')
        found = []
        for row in values:
            for key in ('repeat', 'object_id', 'checkpoint_index', 'band_index'):
                require(type(row.get(key)) is int, 'Invalid H5 integer row key')
            key = tuple(row.get(k) for k in ('repeat', 'object_id', 'corruption', 'checkpoint_index', 'band_index'))
            require(key in expected, 'Frozen H5 row coverage differs')
            found.append(key)
            position = h5.IDS.index(row['object_id'])
            schedule = [9, 18, 27, 35] if row['corruption'] == 'background' else [2, 4, 6, 8]
            require(row['case'] == row['corruption']+'/'+str(position//2)
                    and row['loop_count'] == schedule[row['checkpoint_index']], 'H5 batch/checkpoint identity differs')
            require(set(row['scores']) == set(h5.SCORES)
                    and all(value is None or type(value) in (int, float) and np.isfinite(value)
                            for value in row['scores'].values()), 'Invalid frozen score')
        require(len(set(found)) == 2304 and set(found) == expected, 'Frozen H5 row coverage differs')
    for feature, row in zip(features, rows):
        require(all(row.get(key) == value for key, value in feature.items()), 'Frozen feature values/order changed in oracle rows')
        require(row.get('oracle_benefit') is None or type(row['oracle_benefit']) is bool, 'Oracle label must be resolved bool or None')
    comparison = h5.score_comparison(rows)
    prior = summary['pooled_descriptive_comparison']
    require(all(comparison[key] == prior[key] for key in ('common_eligible_rows', 'common_ordered_row_keys_sha256')),
            'Frozen common mask/key SHA changed')
    require(all(comparison['scores'][score][key] == prior['scores'][score][key]
                for score in h5.SCORES for key in ('n', 'positive', 'negative', 'auroc')), 'Original pooled AUROC/counts changed')
    return rows, dict(summary=summary, identities=identities)


def percentile_ci(values):
    finite = values[np.isfinite(values)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None


def cluster_bootstrap(rows, ids, draws):
    """Reuse the frozen mask/AUC; duplicate all rows of every sampled cluster."""
    require(draws.ndim == 2 and draws.shape[1] == len(ids) and draws.dtype.kind in 'iu'
            and ((draws >= 0) & (draws < len(ids))).all(), 'Invalid whole-cluster draw indices')
    comparison = h5.score_comparison(rows)
    # Delegate eligibility to the original function, so no second mask drifts.
    common = [row for row in rows if h5.score_comparison([row])['common_eligible_rows'] == 1]
    require(h5.score_comparison(common)['common_ordered_row_keys_sha256'] == comparison['common_ordered_row_keys_sha256'], 'Common mask changed')
    clusters = {identity: [row for row in common if row['object_id'] == identity] for identity in ids}
    require(sum(map(len, clusters.values())) == len(common), 'Rows outside original object clusters')
    auc = np.full((len(draws), 3), np.nan)
    differences = np.full((len(draws), 2), np.nan)
    counts = np.zeros((len(draws), 3), np.int64)
    hashes = []
    for i, draw in enumerate(draws):
        sampled = [row for position in draw for row in clusters[ids[int(position)]]]
        result = h5.score_comparison(sampled)
        require(result['common_eligible_rows'] == len(sampled), 'Bootstrap feature mask drifted')
        hashes.append(result['common_ordered_row_keys_sha256'])
        stats = [result['scores'][score] for score in h5.SCORES]
        require(len({(row['n'], row['positive'], row['negative']) for row in stats}) == 1, 'Paired bootstrap rows differ across scores')
        counts[i] = [stats[0]['n'], stats[0]['positive'], stats[0]['negative']]
        auc[i] = [row['auroc'] if row['auroc'] is not None else np.nan for row in stats]
        require(np.isfinite(auc[i]).all() or np.isnan(auc[i]).all(), 'AUROC availability differs across shared rows')
        differences[i] = [auc[i, 2]-auc[i, 0], auc[i, 2]-auc[i, 1]]
    valid = np.isfinite(auc).all(axis=1)
    observed = [comparison['scores'][score]['auroc'] for score in h5.SCORES]
    result = dict(**{key: comparison[key] for key in ('common_eligible_rows', 'common_ordered_row_keys_sha256', 'mask')},
        score_order=list(h5.SCORES), original_clusters=ids, cluster_eligible_row_counts={str(key): len(value) for key, value in clusters.items()},
        observed_auroc=dict(zip(h5.SCORES, observed)), bootstrap_repetitions=len(draws),
        observed_paired_auroc_differences={name: None if observed[2] is None or observed[i] is None else observed[2]-observed[i]
                                         for i, name in enumerate(('combined_minus_frequency', 'combined_minus_residual'))},
        valid_replicates=int(valid.sum()), invalid_replicates=int((~valid).sum()),
        empty_replicates=int((counts[:, 0] == 0).sum()),
        one_class_replicates=int(((counts[:, 0] > 0) & ((counts[:, 1] == 0) | (counts[:, 2] == 0))).sum()),
        combined_auroc_ci95=percentile_ci(auc[:, 2]),
        paired_auroc_difference_ci95={name: percentile_ci(differences[:, i]) for i, name in enumerate(('combined_minus_frequency', 'combined_minus_residual'))},
        interval_semantics='95percentile over defined whole-object draws; undefined/one-class draws remain NaN in raw NPZ and are counted, never assigned zero',
        statistic='Nonlinear pooled AUROC recomputed after cluster duplication; no per-object mean-AUROC substitute')
    raw = dict(draw_cluster_indices=draws, sampled_object_ids=np.asarray(ids, np.int64)[draws],
        auroc=auc, paired_auroc_differences=differences, sampled_common_rows=counts[:, 0],
        positive_rows=counts[:, 1], negative_rows=counts[:, 2],
        draw_common_ordered_row_keys_sha256=np.asarray(hashes), score_names=np.asarray(h5.SCORES),
        difference_names=np.asarray(['combined_minus_frequency', 'combined_minus_residual']))
    return result, raw


def advance_rule(native_passed, result):
    combined = result['combined_auroc_ci95']
    differences = result['paired_auroc_difference_ci95'].values()
    return bool(native_passed is True and combined is not None and combined[0] > .5
                and all(interval is not None and interval[0] > 0 for interval in differences))


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def analyze(directory, pins, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        rows, evidence = load_analysis(directory, pins)
        write_json(output/'inputs.json', dict(directory=str(Path(directory).absolute()), identities=evidence['identities'],
            analyzer=h5.accepted.identity(Path(__file__)), association_helper=h5.accepted.identity(Path(h5.__file__))))
        draws = np.random.default_rng(SEED).integers(0, len(h5.IDS), size=(DRAWS, len(h5.IDS)), dtype=np.int64)
        result, raw = cluster_bootstrap(rows, h5.IDS, draws)
        npz = output/'bootstrap_values.npz'
        with npz.open('xb') as stream:
            np.savez_compressed(stream, **raw)
        native_passed = evidence['summary']['native_repeat_gates_all_passed']
        result.update(status='complete', native_repeat_gates_all_passed=native_passed,
            advanced=advance_rule(native_passed, result), seed=SEED, numpy_version=np.__version__,
            advance_rule='native repeat gates all passed AND combined AUROC lower95CI>.5 AND both combined-minus-baseline AUROC lower95CI>0; otherwise not advanced',
            interpretation='Descriptive development evidence from eight historically exposed original-object clusters, not population or heldout confirmation',
            raw_values=dict(file=npz.name, **h5.accepted.identity(npz)), elapsed_seconds=time.perf_counter()-started)
        result['advance_status'] = 'advanced' if result['advanced'] else 'not_advanced'
        require(all(h5.accepted.identity(Path(directory)/name) == expected for name, expected in evidence['identities'].items()), 'Pinned analysis files changed during bootstrap')
        write_json(output/'summary.json', result)
        return result
    except BaseException:
        write_json(output/'failure.json', dict(status='failed', analysis_directory=str(directory), pins=pins, traceback=traceback.format_exc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', required=True)
    for name in ('summary', 'features', 'rows'):
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    pins = dict(zip(INPUT_NAMES, (args.summary_sha256, args.features_sha256, args.rows_sha256)))
    print(json.dumps(analyze(args.analysis_dir, pins, args.output_dir), allow_nan=False))


if __name__ == '__main__':
    main()
