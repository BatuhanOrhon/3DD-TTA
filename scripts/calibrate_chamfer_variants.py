#!/usr/bin/env python3
"""Label-free C-trajectory gradient calibration; run, then summarize two bundles."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chamfer_guidance_variants import reference_density_weights, weighted_selective_chamfer
from chamfer_calibration_report import validate_calibration_report
from gsd_calibration import DENOMINATOR_EPS, gradient_rows, scalar_summary
from research_artifacts import CORRUPTIONS
from scripts import eval_multibackbone_guidance as runner
from scripts.analyze_scan_trajectory import Bundle, require
from semantic_trajectory import SemanticTrajectoryObserver, semantic_loss

CANDIDATES = ('semantic_kl', 'semantic_h', 'weighted')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
PURPOSE = 'label_free_chamfer_calibration'


def update_gradient_rows(chamfer_gradients, candidate_gradients, *, eta, gamma):
    """Map existing gradient math to neutral names and actual sampler rates."""
    raw = gradient_rows(*chamfer_gradients, *candidate_gradients)
    result = []
    for value in raw:
        row = {}
        for component in ('local', 'style'):
            for old, new in (('scd_norm', 'chamfer_grad_norm'), ('spectral_norm', 'candidate_grad_norm'),
                             ('ratio', 'ratio'), ('cosine', 'cosine'),
                             ('scd_near_zero', 'chamfer_near_zero'), ('spectral_near_zero', 'candidate_near_zero')):
                row[component + '_' + new] = value[component + '_' + old]
        c = math.hypot(eta * row['local_chamfer_grad_norm'], gamma * row['style_chamfer_grad_norm'])
        other = math.hypot(eta * row['local_candidate_grad_norm'], gamma * row['style_candidate_grad_norm'])
        row.update(chamfer_update_norm=c, candidate_update_norm=other,
                   combined_update_ratio=other / c if c > DENOMINATOR_EPS else None)
        result.append(row)
    return result


class CalibrationTrajectoryObserver(SemanticTrajectoryObserver):
    """Keep the C path, measuring three alternative gradients at existing probes."""

    def __init__(self, *args, original_ids, corruption, chamfer=None, density_k=10, **kwargs):
        classify = kwargs.pop('classify_semantic_decoded')
        self.pending_losses = {}
        def classify_and_retain(decoded):
            logits = classify(decoded)
            self.pending_losses = {name: semantic_loss(logits['pointmae'], self.reference_logits, kind)
                                   for name, kind in (('semantic_kl', 'kl'), ('semantic_h', 'marginal_entropy'))}
            return logits
        super().__init__(*args, classify_semantic_decoded=classify_and_retain, **kwargs)
        self.original_ids, self.corruption = list(original_ids), corruption
        self.chamfer, self.density_k = chamfer, density_k
        self.reference_weights = None
        self.calibration_rows = []
        self.calibration_probe_calls = 0

    def guidance_components(self, *, ch_loss, semantic_loss, noisy_local, style, pred_latent,
                            h0, loop_count, timestep, eta, gamma, p):
        if loop_count not in self.steps[1:]:
            return
        require(semantic_loss is not None and set(self.pending_losses) == {'semantic_kl', 'semantic_h'},
                'Calibration semantic graphs are missing')
        if self.chamfer is None:
            from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
            self.chamfer = chamfer_3DDist()
        if self.reference_weights is None:
            self.reference_weights = reference_density_weights(h0[:, :, :3], self.density_k)
        first, second, nearest, _ = self.chamfer(pred_latent[:, :, :3], h0[:, :, :3])
        losses = dict(self.pending_losses)
        losses['weighted'] = weighted_selective_chamfer(first, second, nearest, self.reference_weights, p)
        parameters = (noisy_local, style)
        def gradients(loss):
            values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            return tuple(torch.zeros_like(parameter) if value is None else value for value, parameter in zip(values, parameters))
        c_gradients = gradients(ch_loss)
        for candidate, loss in losses.items():
            rows = update_gradient_rows(c_gradients, gradients(loss), eta=eta, gamma=gamma)
            require(len(rows) == len(self.original_ids), 'Calibration IDs do not match gradient batch')
            for original_id, row in zip(self.original_ids, rows):
                row.update(original_id=int(original_id), corruption=self.corruption, loop_count=loop_count,
                           timestep=int(timestep), candidate=candidate, batch_raw_loss=float(loss.detach()),
                           eta_local=float(eta), gamma_style=float(gamma))
                self.calibration_rows.append(row)
        self.calibration_probe_calls += 1
        self.pending_losses = {}

    def diagnostics(self):
        result = super().diagnostics()
        result['calibration'] = dict(purpose=PURPOSE, rows=self.calibration_rows,
                                     probe_calls=self.calibration_probe_calls, density_precompute_calls=int(self.reference_weights is not None),
                                     extra_native_chamfer_calls=self.calibration_probe_calls,
                                     autograd_grad_calls=4 * self.calibration_probe_calls,
                                     updates='C only; semantic weight0; diagnostics never write .grad',
                                     update_rate_mapping='local=eta, style=gamma', labels_used_for_selection=False)
        return result


def validate_run(config):
    require(config['dataset'] in DATASETS and config['seed'] == 0 and config['batch_size'] == 70,
            'Calibration requires dataset scope, seed0 and B70')
    require(config['backbone'] == 'pointmae' and config['backbones'] == ['pointmae'], 'Calibration uses Point-MAE only')
    require(config['corruptions'] == list(CORRUPTIONS) and config['max_batches'] == 0,
            'Calibration requires all15 corruptions without truncation')
    require(config.get('selection_manifest', {}).get('stage') == 'development'
            and len(config.get('selected_original_ids', [])) == 70, 'Calibration requires declared development70 IDs')
    require(config['guidance_protocol']['chamfer_experiment'] == dict(variant='semantic', semantic_weight=0.),
            'Calibration must follow unchanged C trajectory with semantic sham')
    require(config['guidance_protocol']['eta'] == .01 and config['guidance_protocol']['gamma'] == .01
            and config['guidance_protocol']['p'] == .95, 'Calibration locks baseline rates and trimming')
    require(config.get('semantic_guidance', {}).get('loss_reduction') == 'sum',
            'Calibration requires batch-invariant semantic sum reduction')


def coefficient_report(rows_by_dataset, ids_by_dataset):
    """Use gradient records exclusively; no logits, labels or evaluator scores."""
    require(set(rows_by_dataset) == set(DATASETS) and set(ids_by_dataset) == set(DATASETS),
            'Both calibration datasets are required')
    all_rows = []
    for dataset in DATASETS:
        ids, rows = ids_by_dataset[dataset], rows_by_dataset[dataset]
        require(len(ids) == 70 and ids == sorted(set(ids)), 'Expected sorted development70 IDs')
        expected = {(corruption, index, step, candidate) for corruption in CORRUPTIONS for index in ids
                    for step in ((15, 23) if corruption == 'background' else (5, 8)) for candidate in CANDIDATES}
        actual = {(row['corruption'], row['original_id'], row['loop_count'], row['candidate']) for row in rows}
        require(len(rows) == len(expected) and actual == expected, 'Missing or duplicate calibration coverage')
        for row in rows:
            c, other, ratio = row['chamfer_update_norm'], row['candidate_update_norm'], row['combined_update_ratio']
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > DENOMINATOR_EPS for v in (c, other, ratio)),
                    'Nonfinite or zero/near-zero calibration norm/ratio; no coefficient chosen')
            require(math.isclose(ratio, other / c, rel_tol=1e-10), 'Calibration norm ratio mismatch')
        all_rows.extend(rows)
    ratios = {candidate: scalar_summary(row['combined_update_ratio'] for row in all_rows if row['candidate'] == candidate)
              for candidate in CANDIDATES}
    coefficients = {candidate: .1 / ratios[candidate]['median'] for candidate in CANDIDATES[:2]}
    coefficients['weighted_scale'] = 1. / ratios['weighted']['median']
    return dict(schema_version=1, purpose=PURPOSE, coefficients=coefficients, ratios=ratios,
                target_semantic_update_ratio=.1, selection='gradient-only; evaluator labels/accuracy are archived but never read for coefficient selection',
                pooling='equal object/probe/corruption weights within each dataset; equal70 objects in both datasets',
                update_rate_mapping='local=eta, style=gamma',
                loss_reduction='sum for Chamfer, weighted Chamfer, and per-object semantic losses',
                rule='semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)',
                local_cosines={candidate: scalar_summary(row['local_cosine'] for row in all_rows if row['candidate'] == candidate) for candidate in CANDIDATES},
                style_cosines={candidate: scalar_summary(row['style_cosine'] for row in all_rows if row['candidate'] == candidate) for candidate in CANDIDATES})


def summarize_bundles(paths):
    rows, ids, provenance = {}, {}, {}
    for path in paths:
        bundle = Bundle(path)
        try:
            config = bundle.json('config.json')
            validate_run(config)
            dataset = config['dataset']
            require(dataset not in rows and config.get('status') in ('complete', 'partial')
                    and config.get('completed_batches') == {name: 1 for name in CORRUPTIONS},
                    'Require one completed development-subset bundle per dataset')
            require(config.get('chamfer_calibration', {}).get('purpose') == PURPOSE, 'Calibration purpose missing')
            require(config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
                    and config['lion_state_before'] == config['lion_state_after'], 'Calibration model state changed')
            sources = config.get('executed_source_files', {})
            sealed = bundle.json('bundle_manifest.json')
            required_sources = ('scripts/calibrate_chamfer_variants.py', 'tta_gsd_v2.py', 'semantic_trajectory.py')
            require(all('repo/' + name in sources for name in required_sources), 'Calibration source identities missing')
            require(all(sealed.get('source_snapshot/' + name[5:]) == identity for name, identity in sources.items() if name.startswith('repo/')),
                    'Calibration source snapshot mismatch')
            require(config.get('native_extensions') and config.get('asset_manifest', {}).get('files'), 'Calibration native/asset provenance missing')
            require(config.get('runtime', {}).get('cuda') and config['runtime'].get('gpu'), 'Calibration GPU runtime evidence missing')
            selection_raw = bundle.read('selection_manifest.json')
            require(config['selection_manifest_identity'] == dict(bytes=len(selection_raw), sha256=hashlib.sha256(selection_raw).hexdigest())
                    and json.loads(selection_raw) == config['selection_manifest']
                    and config['selection_manifest']['indices'] == config['selected_original_ids'], 'Calibration ID manifest mismatch')
            records = []
            expected_diagnostics = {'batches/' + name + '/batch0000/diagnostics.json' for name in CORRUPTIONS}
            require({name for name in bundle.names if name.startswith('batches/') and name.endswith('/diagnostics.json')} == expected_diagnostics,
                    'Calibration batch coverage mismatch')
            for corruption in CORRUPTIONS:
                diagnostic = bundle.json('batches/' + corruption + '/batch0000/diagnostics.json')
                metrics = diagnostic.get('guidance_metrics', {})
                require(metrics.get('graph_calls') == 0 and metrics.get('spectral_loss_calls') == 0, 'Calibration executed forbidden spectral work')
                calibration = diagnostic['trajectory']['calibration']
                require(diagnostic['trajectory'].get('initial_rng_sha256')
                        and diagnostic['trajectory'].get('probe_rng_unchanged') is True
                        and diagnostic['trajectory'].get('probe_buffers_unchanged') is True,
                        'Calibration initial RNG/buffer provenance missing')
                require(calibration['probe_calls'] == 2 and calibration['autograd_grad_calls'] == 8,
                        'Calibration component call count mismatch')
                require(calibration['labels_used_for_selection'] is False, 'Calibration used label selection')
                records.extend(calibration['rows'])
            rows[dataset], ids[dataset] = records, config['selected_original_ids']
            config_raw = bundle.read('config.json')
            provenance[dataset] = dict(path=str(Path(path).resolve()), run_id=config['run_id'],
                                       config_identity=dict(bytes=len(config_raw), sha256=hashlib.sha256(config_raw).hexdigest()),
                                       sealed_manifest_sha256=hashlib.sha256(bundle.read('bundle_manifest.json')).hexdigest(),
                                       selected_original_ids=ids[dataset], selection_manifest_identity=config['selection_manifest_identity'],
                                       executed_source_files=sources, native_extensions=config['native_extensions'],
                                       asset_manifest_sha256=config['asset_manifest_sha256'], runtime=config['runtime'],
                                       classifier_state_sha256=config['classifier_state_before_sha256'], lion_state=config['lion_state_before'])
        finally:
            bundle.close()
    report = coefficient_report(rows, ids)
    report['calibration_bundles'] = provenance
    report['status'] = 'verified_label_free_gpu'
    return validate_calibration_report(report)


def calibration_worker(directories):
    for directory in directories:
        bundle = runner.source.RunBundle(Path(directory))
        config = json.loads((bundle.path / 'config.json').read_text())
        validate_run(config)
        config['chamfer_calibration'] = dict(purpose=PURPOSE, candidates=list(CANDIDATES), target_semantic_update_ratio=.1,
                                           labels_used_for_selection=False, coefficients_frozen_before_heldout=True,
                                           evaluator_predictions_retained_for_audit=True)
        config['display_name'] = 'Label-free C-trajectory calibration; KL/H/weighted diagnostics'
        bundle.write_config(config)
    def observer_provider(**values):
        return CalibrationTrajectoryObserver(**values)
    runner.worker(directories, observer_provider=observer_provider)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['--worker-bundles']:
        calibration_worker(json.loads(arguments[1]))
        return
    if arguments[:1] == ['summarize']:
        parser = argparse.ArgumentParser(description='Seal fixed coefficients from two validated gradient bundles')
        parser.add_argument('bundles', nargs=2)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        report = summarize_bundles(args.bundles)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        receipt = dict(status='complete', purpose=PURPOSE, report=runner.source.file_identity(args.output),
                       calibration_bundles=report['calibration_bundles'])
        with args.output.with_suffix('.receipt.json').open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(report['coefficients'], sort_keys=True))
        return
    if arguments[:1] != ['run']:
        raise ValueError('Use run <runner options> or summarize <two bundles> --output report.json')
    run_arguments = ['--chamfer-experiment', 'semantic', '--semantic-weight', '0', '--semantic-kind', 'kl',
                     '--backbones', 'pointmae', *arguments[1:]]
    configs = runner.configurations(run_arguments)
    for config in configs:
        validate_run(config)
    if configs[0]['preview']:
        print(json.dumps(dict(purpose=PURPOSE, configs=configs, candidates=list(CANDIDATES),
                              semantic_checkpoints='normal5/8, background15/23',
                              trajectory='unchanged C; semantic weight0; no spectral',
                              diagnostics_per_batch=dict(extra_chamfer_calls=2, autograd_grad_calls=8,
                                                         cached_X_source_forwards=1, density_precomputations=1),
                              selection='gradients only; evaluator predictions retained for audit'), indent=2, sort_keys=True))
        return
    runner.main(run_arguments, worker_script=Path(__file__).resolve())


if __name__ == '__main__':
    main()
