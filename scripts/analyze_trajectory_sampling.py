"""CPU-only paired analysis of opt-in macro14 sampling-consistency traces."""
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_artifacts import CORRUPTIONS
from scripts.analyze_legacy_trajectory import Bundle, load_trace, write_csv, clustered_deltas
from trajectory_sampling import sampling_protocol, view_policies, view_starts

POLICIES = ('terminal', 'four_a_mean', 'eight_view_mean', 'view_entropy',
            'view_entropy_js', 'terminal_eight_mean')
MACRO14 = [c for c in CORRUPTIONS if c != 'background']


def validate_assets(config: dict) -> None:
    """Match recorded runtime asset identities against the accepted baseline."""
    spec_name = 'cfgs/legacy_trajectory_provenance.json'
    content = (Path(__file__).resolve().parents[1] / spec_name).read_bytes().replace(b'\r\n', b'\n')
    identity = config.get('source_delivery', {}).get('files', {}).get(spec_name, {})
    if (identity.get('sha256') != hashlib.sha256(content).hexdigest()
            or identity.get('bytes') != len(content)):
        raise ValueError('Baseline asset specification differs from source delivery')
    spec = json.loads(content)
    if spec['baseline_archive_sha256'] != '534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968':
        raise ValueError('Unrecognized baseline asset specification')
    normalize = lambda name: posixpath.normpath(str(name).replace('\\', '/'))
    assets = {}
    for name, recorded in config.get('assets', {}).items():
        key = normalize(name)
        if key in assets:
            raise ValueError('Duplicate recorded asset path: ' + key)
        assets[key] = recorded
    cli = config['cli']
    paths = [(cli.get(option, ''), canonical) for option, canonical in (
        ('pointmae_config', 'cfgs/tta_modelnet.yaml'),
        ('pointmae_ckpt', 'pointnet_ckpts/modelnet_jt.pth'),
        ('diff_config', 'lion_ckpts/unconditional_all55_cfg.yml'),
        ('diff_ckpt', 'lion_ckpts/epoch_10999_iters_2100999.pt'),
        ('label_path', 'data/modelnet40_c/label.npy'))]
    paths += [(normalize(cli.get('dataset_root', '')) + '/data_' + c + '_5.npy',
               'data/modelnet40_c/data_' + c + '_5.npy') for c in config['corruptions']]
    for actual_path, canonical in paths:
        recorded = assets.get(normalize(actual_path), {})
        expected = spec['baseline_assets'][canonical]
        if any(recorded.get(k) != expected[k] for k in ('bytes', 'sha256')):
            raise ValueError('Baseline asset identity mismatch: ' + canonical)


def validate_sampling_arrays(arrays: dict, original: np.ndarray, ids: list,
                             seed: int, views: int) -> None:
    """Validate all view identities before labels are used to score policies."""
    n = len(ids)
    logits = arrays['sampling_logits']
    if logits.shape != (n, views, 40) or not np.isfinite(logits).all():
        raise ValueError('Invalid sampling logits')
    if not np.array_equal(logits[:, 0], original):
        raise ValueError('Sampling did not preserve original A logits')
    cloud_size = arrays['sampling_cloud_size']
    if cloud_size.shape != () or int(cloud_size) != 2048:
        raise ValueError('Expected decoded cloud with 2048 points')
    indices, starts = arrays['sampling_indices'], arrays['sampling_starts']
    if (indices.shape != (n, views, 1024) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= int(cloud_size))):
        raise ValueError('Invalid sampling indices')
    expected = np.column_stack((np.zeros(n, dtype=np.int64),
                               view_starts(seed, ids, int(cloud_size), views - 1)))
    if (not np.array_equal(starts, expected)
            or not np.array_equal(indices[:, :, 0], expected)):
        raise ValueError('Sampling start identity mismatch')


def load_sampling(bundle) -> list:
    config = bundle.config
    cli = config['cli']
    delivery = config.get('source_delivery', {})
    source_files = {name.replace('\\', '/'): value for name, value in config.get('source_manifest', {}).items()}
    required = {'eval_gsd_tta_v2.py', 'legacy_gsd_artifacts.py', 'legacy_trajectory.py', 'trajectory_sampling.py'}
    if (not delivery.get('manifest_sha256') or not required.issubset(source_files)
            or delivery.get('base_commit') != 'a351e66ec58f08f3a2104e46cab2cb8b20cdf961'):
        raise ValueError('Missing sampling source delivery provenance')
    for name, identity in source_files.items():
        expected = delivery.get('files', {}).get(name, {})
        if identity.get('sha256') != expected.get('sha256') or identity.get('bytes') != expected.get('bytes'):
            raise ValueError('Runtime source differs from sampling delivery: ' + name)
    if (config['trajectory'].get('sampling') != sampling_protocol()
            or config['trajectory']['checkpoints'] != {'10': [3, 5, 8], '30': [8, 15, 23]}
            or not set(config['corruptions']).issubset(MACRO14)
            or len(set(config['corruptions'])) != len(config['corruptions'])
            or config['seed'] != 0 or cli['batch_size'] != 70 or cli['M_max'] != 800
            or cli['corruption_scope'] != 'macro14' or cli['trajectory_sampling'] != 'paired'
            or cli['isolate_policy'] != 'legacy' or cli['denoising_step_normal'] != 10
            or cli['denoising_step_bg'] != 30 or cli['denoising_step'] is not None
            or cli['use_static_style'] or cli['dynamic_graph'] or cli['use_4d_gft']
            or cli['weight_invariant'] != 0
            or any(cli[k] != v for k, v in dict(beta=2., weight_spectral=1.17,
                weight_chamfer=1., gamma=.01, eta=.01, lambdaa=.95).items())):
        raise ValueError('Sampling trace must match the approved Legacy macro14 seed0 host')
    validate_assets(config)
    groups = load_trace(bundle)  # CRC, manifest SHA, original IDs, labels, full views, frozen state
    for group in groups:
        paired_batches, terminal_batches, overlaps, times = [], [], [], {}
        names = sorted(n for n in bundle.names if n.startswith(
            'trajectory/' + group['corruption'] + '/') and n.endswith('.json'))
        for name in names:
            record = bundle.json(name)
            batch, batch_overlap = [], []
            for j, candidate in enumerate(record['candidates']):
                arrays = bundle.arrays('trajectory/' + group['corruption'] + '/' + candidate['file'])
                views = 8 if j == 3 else 2
                validate_sampling_arrays(arrays, arrays['logits'], record['indices'], group['seed'], views)
                hashes = candidate['hashes']['sampling']
                if (hashes['rng_before'] != hashes['rng_after']
                        or hashes['buffers_before'] != hashes['buffers_after']
                        or hashes['classifier_forwards'] != views - 1):
                    raise ValueError('Sampling RNG/buffer/forward invariant failed')
                if j < 3:
                    h = candidate['hashes']
                    if (h['probe_rng_before'] != h['probe_rng_after']
                            or h['probe_buffers_before'] != h['probe_buffers_after']):
                        raise ValueError('Clean probe invariant failed')
                batch.append(arrays['sampling_logits'][:, :2])
                if j == 3:
                    terminal_batches.append(arrays['sampling_logits'])
                overlap = []
                for a, b in arrays['sampling_indices'][:, :2]:
                    a_set, b_set = set(a.tolist()), set(b.tolist())
                    overlap.append(len(a_set & b_set) / len(a_set | b_set))
                batch_overlap.append(overlap)
            paired_batches.append(np.stack(batch, axis=1))
            overlaps.append(np.asarray(batch_overlap).T)
            for component, seconds in record['times'].items():
                if not np.isfinite(seconds) or seconds < 0:
                    raise ValueError('Invalid component timing')
                times[component] = times.get(component, 0.) + seconds
        group.update(paired_logits=np.concatenate(paired_batches),
                     terminal_votes=np.concatenate(terminal_batches),
                     overlap=np.concatenate(overlaps), acquisition_component_seconds=times)
    return groups


def analyze(paths: list, output: Path) -> dict:
    """Score one completed smoke/pilot, never mix seeds or unmatched sources."""
    if len(paths) != 1:
        raise ValueError('Analyze one complete acquisition bundle at a time')
    if Path(output).exists():
        raise FileExistsError('Preserve existing analysis: ' + str(output))
    bundle = Bundle(paths[0])
    try:
        groups = load_sampling(bundle)
        config = bundle.config
        config_sha = hashlib.sha256(bundle.read('config.json')).hexdigest()
    finally:
        bundle.close()
    rows, samples, diagnostics = [], [], []
    for group in groups:
        result = view_policies(group['paired_logits'], group['terminal_votes'])
        y = group['targets']
        reference = result['terminal'] == y
        mean_correct = result['four_a_mean'] == y
        for policy in POLICIES:
            pred = result[policy]
            correct = pred == y
            rows.append(dict(corruption=group['corruption'], seed=group['seed'], policy=policy,
                n=len(y), n_correct=int(correct.sum()), accuracy=float(correct.mean()),
                delta_pp_vs_terminal=float(100 * (correct.mean() - reference.mean())),
                corrected=int((correct & ~reference).sum()), broken=int((~correct & reference).sum()),
                delta_pp_vs_four_a_mean=float(100 * (correct.mean() - mean_correct.mean())),
                corrected_vs_four_a_mean=int((correct & ~mean_correct).sum()),
                broken_vs_four_a_mean=int((~correct & mean_correct).sum())))
            for i, original_id in enumerate(group['indices']):
                samples.append(dict(corruption=group['corruption'], seed=group['seed'],
                    index=int(original_id), target=int(y[i]), policy=policy, prediction=int(pred[i]),
                    correct=int(correct[i]), terminal_correct=int(reference[i]),
                    four_a_mean_correct=int(mean_correct[i])))
        diagnostics.append(dict(corruption=group['corruption'],
            mean_normalized_js_by_candidate=result['js'].mean(0).tolist(),
            mean_index_set_jaccard_by_candidate=group['overlap'].mean(0).tolist(),
            identical_index_sets_by_candidate=(group['overlap'] == 1).sum(0).tolist(),
            selected_entropy_histogram=np.bincount(result['selected_entropy'], minlength=4).tolist(),
            selected_entropy_js_histogram=np.bincount(result['selected_entropy_js'], minlength=4).tolist(),
            acquisition_component_seconds=group['acquisition_component_seconds']))
    summary = []
    for policy in POLICIES:
        r = [v for v in rows if v['policy'] == policy]
        summary.append(dict(policy=policy, macro_accuracy=float(np.mean([v['accuracy'] for v in r])),
            n=sum(v['n'] for v in r), n_correct=sum(v['n_correct'] for v in r),
            delta_pp_vs_terminal=float(np.mean([v['delta_pp_vs_terminal'] for v in r])),
            delta_pp_vs_four_a_mean=float(np.mean([v['delta_pp_vs_four_a_mean'] for v in r])),
            corrected=sum(v['corrected'] for v in r), broken=sum(v['broken'] for v in r),
            corrected_vs_four_a_mean=sum(v['corrected_vs_four_a_mean'] for v in r),
            broken_vs_four_a_mean=sum(v['broken_vs_four_a_mean'] for v in r)))
    report = dict(schema_version=1, scope=('macro14 development' if config['corruptions'] == MACRO14
                   else 'smoke/subset; not macro14 accuracy'), source=str(paths[0]),
        config_sha256=config_sha, source_delivery=config.get('source_delivery'),
        validated_complete=True, corruptions=config['corruptions'], seed=config['seed'],
        summary=summary, per_corruption=rows, diagnostics=diagnostics,
        index_cluster_intervals=clustered_deltas(samples),
        acquisition_runtime=config.get('trajectory_runtime', {}),
        limitations=['One seed on development data; no independent generalization.',
                     'Acquisition uses 14 forwards; each eight-view policy uses 8. No measured deployment speedup.',
                     'Index-set diversity does not ensure different coordinates or correct predictions.',
                     'Normalized H+JS is a declared experimental score, not an established paper algorithm.'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_corruption.csv', rows)
    write_csv(output / 'paired_examples.csv', samples)
    with (output / 'analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze([args.bundle], args.output)
    print(report['scope'])
    for row in report['summary']:
        print('%-24s %.6f%% (%+.6f pp vs four-A mean)' % (
            row['policy'], 100 * row['macro_accuracy'], row['delta_pp_vs_four_a_mean']))


if __name__ == '__main__':
    main()
