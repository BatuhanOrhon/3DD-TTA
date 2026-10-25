"""Eight strict frozen-head loads and no-build native proof; no forward/CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_assets_template import (ROOT, PREPARED, ASSETS, DRIVE, MODELNET, SESSION, MANIFEST_SHA,
                                    SOURCE_SHA, identity, verify, require, utc, seal, preserve_phase_sources)

DATASETS = {'modelnet40_c': (2468, 40, 'official_modelnet40_test'),
            'scanobjectnn_c': (581, 15, 'main_split')}
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')


def validate_rows(rows):
    require(len(rows) == 8 and {(r['dataset'], r['backbone']) for r in rows} ==
            {(dataset, head) for dataset in DATASETS for head in HEADS}, 'Missing/duplicate strict classifier proof')
    for row in rows:
        total, classes, split = DATASETS[row['dataset']]
        load = row['checkpoint_load']
        require((row['n_examples'], row['num_classes'], row['split']) == (total, classes, split), 'Dataset contract mismatch')
        require(load.get('strict') is True and load.get('missing_keys') == [] and load.get('unexpected_keys') == [], 'Strict checkpoint load failed')
        require(row.get('forward_executed') is False and row.get('cpu_frozen_eval') is True, 'CPU/frozen/no-forward proof failed')
        require(len(row['state_sha256']) == 64 and len(load['checkpoint_sha256']) == 64, 'Missing state/checkpoint identity')


def proof(output):
    require(sys.version_info[:2] == (3, 8), 'Use 3dd_tta_env Python3.8')
    require((PREPARED / '.deployed').read_text().strip() == MANIFEST_SHA, 'Deployment marker mismatch')
    deployment = json.loads((ROOT / 'deployment_receipt.json').read_text())
    require(deployment['state']['status'] == 'complete', 'Deployment is not accepted')
    verify(deployment['archive']['local_path'], deployment['archive'])
    verify(deployment['archive']['drive_path'], deployment['archive'])
    sys.path.insert(0, str(ASSETS / 'source_snapshot'))
    from scripts import eval_multibackbone_source as source
    from scripts import eval_multibackbone_guidance as guidance
    require(source.REPO == ASSETS / 'source_snapshot', 'CPU proof imported unsealed source')
    manifest = source.read_manifest(ASSETS, MANIFEST_SHA)
    require(manifest['source_snapshot_sha256'] == SOURCE_SHA, 'Executed source archive mismatch')
    for entry in (manifest['guidance']['lion_checkpoint'], manifest['guidance']['lion_config']):
        verify(entry['path'], entry)
    import torch
    require(torch.__version__ == '2.1.2+cu121' and not torch.cuda.is_initialized(), 'CPU proof environment/CUDA state mismatch')
    hook = source._declared_asset(ASSETS, manifest, manifest['guidance']['native_bootstrap'])
    spec = importlib.util.spec_from_file_location('agent_b_native_cpu_proof', hook)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    native = module.bootstrap(ASSETS)
    require(native['no_build'] and len(native['bindings']) == 4 and not torch.cuda.is_initialized(), 'Four-backend no-build CPU proof failed')
    os.chdir(str(source.REPO))
    rows = []
    for dataset in DATASETS:
        data = MODELNET if dataset == 'modelnet40_c' else ASSETS / 'datasets/scanobjectnn_c'
        filename = 'tta_modelnet.yaml' if dataset == 'modelnet40_c' else 'tta_scanobj.yaml'
        checkpoint = 'modelnet_jt.pth' if dataset == 'modelnet40_c' else 'scanobject_jt.pth'
        configs = guidance.configurations(['--trajectory-observe', '--component-arm', 'combined', '--backbones', *HEADS,
            '--pointmae-config', 'source_snapshot/cfgs/' + filename, '--pointmae-checkpoint', 'checkpoints/' + checkpoint,
            '--dataset', dataset, '--dataset-root', str(data), '--label-path', str(data / 'label.npy'),
            '--asset-root', str(ASSETS), '--asset-manifest-sha256', MANIFEST_SHA,
            '--result-root', str(ROOT / 'not_used_by_cpu_proof')])
        provenance = source.dataset_manifest(manifest, configs[0])
        for name, entry in provenance['files'].items():
            source.load_array(data / name, entry, configs[0], labels=name == 'label.npy')
        for config in configs:
            reference = guidance.matched_source(config, manifest, manifest['guidance'])
            with patch.object(torch.nn.Module, 'cuda', lambda model, *args, **kwargs: model), \
                 patch.object(torch.Tensor, 'cuda', side_effect=RuntimeError('GPU tensor forbidden in CPU proof')), \
                 patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CUDA initialization forbidden in CPU proof')), \
                 patch.object(torch.nn.Module, '_call_impl', side_effect=RuntimeError('Forward forbidden in CPU proof')):
                model = source.load_classifier(config, manifest)
            cpu_frozen_eval = (all(parameter.device.type == 'cpu' and not parameter.requires_grad for parameter in model.parameters())
                               and all(buffer.device.type == 'cpu' for buffer in model.buffers())
                               and not any(layer.training for layer in model.modules()))
            require(cpu_frozen_eval and not torch.cuda.is_initialized(), 'Classifier changed CPU/frozen/CUDA state')
            row = {key: config[key] for key in ('dataset', 'backbone', 'n_examples', 'num_classes', 'split', 'checkpoint_load', 'resolved_model_config')}
            row.update(state_sha256=source.state_hash(model), matched_source=reference, forward_executed=False, cpu_frozen_eval=True)
            rows.append(row)
            (output / 'partial_strict_loads.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
            print('AGENT_B_STRICT_CPU_HEAD', dataset, config['backbone'], flush=True)
            del model
    validate_rows(rows)
    require(not torch.cuda.is_initialized(), 'CPU proof initialized CUDA')
    return dict(status='complete', session=SESSION, manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA,
                models=rows, native=native, cuda_initialized=False, python=sys.version, torch=torch.__version__, research_gpu_launched=False)


def main():
    stamp = 'cpu_preflight_' + utc()
    output = ROOT / stamp; output.mkdir(parents=True, exist_ok=False)
    drive_output = DRIVE / 'phase-evidence' / stamp; drive_output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', session=SESSION, manifest_sha256=MANIFEST_SHA, research_gpu_launched=False)
    started = time.monotonic()
    try:
        require(not (PREPARED / '.complete').exists(), 'CPU acceptance exists; inspect instead of rerunning')
        preserve_phase_sources(output)
        report = proof(output)
        (output / 'cpu_preflight.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        state.update(status='complete', strict_loads=8, cuda_initialized=False)
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        receipt = dict(state=state, archive=seal(output, drive_output / 'cpu_preflight_evidence.zip'),
                       report_identity=identity(output / 'cpu_preflight.json') if (output / 'cpu_preflight.json').is_file() else None)
        (ROOT / 'cpu_preflight_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (drive_output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('AGENT_B_CPU_PROOF_RECEIPT', json.dumps(receipt), flush=True)
    shutil.copyfile(output / 'cpu_preflight.json', PREPARED / 'cpu_preflight.json')
    (PREPARED / 'accepted_cpu_preflight_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (PREPARED / '.complete').write_text(MANIFEST_SHA + '\n')
    print('AGENT_B_COMPONENT_CPU_PROOF_COMPLETE', flush=True)


def self_check():
    rows = [dict(dataset=dataset, backbone=head, n_examples=total, num_classes=classes, split=split,
                 checkpoint_load=dict(strict=True, missing_keys=[], unexpected_keys=[], checkpoint_sha256='a' * 64),
                 state_sha256='b' * 64, forward_executed=False, cpu_frozen_eval=True)
            for dataset, (total, classes, split) in DATASETS.items() for head in HEADS]
    validate_rows(rows)
    for changed in (rows[:-1], [dict(rows[0], num_classes=15), *rows[1:]], [dict(rows[0], forward_executed=True), *rows[1:]]):
        try:
            validate_rows(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid CPU proof accepted')
    print('AGENT_B_CPU_TEMPLATE_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
