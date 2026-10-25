"""Summarize existing observer dispatch spans; never label them CUDA timings."""
import csv
import json
from pathlib import Path
import sys

from analyze_development_local import identity, require, verify


def main(prep):
    prep = Path(prep).resolve()
    repository = prep.parents[2]
    sys.path.insert(0, str(prep / 'assets_v1/source_snapshot'))
    from scripts.analyze_scan_trajectory import Bundle
    rows = []
    for stage, receipt_name in [('smoke', 'smoke_phase_receipt.json'), ('development', 'development_phase_receipt.json')]:
        receipt = json.loads((prep / receipt_name).read_text())
        require(receipt['state']['status'] == 'complete', 'Incomplete input phase')
        for job in receipt['state']['jobs']:
            batches, heads = {}, set()
            for raw in job['raw_archives']:
                archive = repository / 'result' / job['dataset'] / 'gsd_inspired_latent_spectral_guidance' / Path(raw['local_path']).name
                verify(archive, raw)
                bundle = Bundle(archive)
                try:
                    config = bundle.json('config.json'); heads.add(config['backbone'])
                    for corruption in config['corruptions']:
                        diagnostic = bundle.json('batches/' + corruption + '/batch0000/diagnostics.json')
                        observed = dict(times=diagnostic['trajectory']['times'], metrics=diagnostic['guidance_metrics'])
                        require(all(value >= 0 for value in observed['times'].values()), 'Negative dispatch span')
                        require(corruption not in batches or batches[corruption] == observed, 'Four heads disagree about shared reconstruction spans')
                        batches[corruption] = observed
                finally:
                    bundle.close()
            require(heads == {'dgcnn', 'curvenet', 'pointnext', 'pointmae'} and len(batches) == (2 if stage == 'smoke' else 15), 'Missing shared heads/batches')
            row = dict(stage=stage, dataset=job['dataset'], component_arm=job['arm'], shared_batches=len(batches),
                       timing_scope='unsynchronized observer dispatch wall; terminal_decode also includes terminal head work; not additive CUDA decomposition')
            for tag in ('encode', 'graph_eigh', 'style_noise', 'prior_scheduler', 'guidance_losses_and_diagnostics', 'guidance_backward_update', 'trajectory_probe', 'terminal_decode'):
                row[tag + '_dispatch_seconds'] = sum(batch['times'].get(tag, 0.) for batch in batches.values())
            for name in ('mean_raw_chamfer', 'mean_raw_spectral', 'mean_local_update_norm', 'mean_style_update_norm'):
                row[name + '_equal_batch_mean'] = sum(batch['metrics'][name] for batch in batches.values()) / len(batches)
            rows.append(row)
    require(len(rows) == 16, 'Expected both eight-job phases')
    target = prep / 'local_development_analysis/component_dispatch_costs.csv'
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status='verified_shared_cost_counted_once', rows=len(rows), output=str(target), **identity(target))))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
