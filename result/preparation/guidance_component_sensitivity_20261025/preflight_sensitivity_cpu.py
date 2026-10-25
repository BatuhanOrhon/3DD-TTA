"""Verify both method boundaries for eight frozen heads; prohibit CUDA/forward."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path('/content/studies/guidance-components')
ASSETS = ROOT / 'prepared_sensitivity_v2/assets'
PREPARED = ASSETS.parent
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
MANIFEST_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA = '73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
SESSION = 'tta-guidance-components-a100-20261025-r1'


def main():
    old = ROOT / 'phases'
    assert hashlib.sha256((old / 'deploy_assets_template.py').read_bytes()).hexdigest() == '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
    sys.path.insert(0, str(old))
    from deploy_assets_template import identity, verify, seal, utc
    assert sys.version_info[:2] == (3, 8) and not (PREPARED / '.complete').exists()
    assert (PREPARED / '.deployed').read_text().strip() == MANIFEST_SHA
    deployment = json.loads((ROOT / 'sensitivity_deployment_receipt.json').read_text())
    assert deployment['state']['status'] == 'complete' and deployment['state']['manifest_sha256'] == MANIFEST_SHA
    verify(deployment['archive']['drive_path'], deployment['archive'])
    output = ROOT / ('sensitivity_cpu_preflight_' + utc()); output.mkdir()
    shutil.copyfile(Path(__file__), output / 'preflight_sensitivity_cpu.py')
    state = dict(status='running', session=SESSION, manifest_sha256=MANIFEST_SHA, research_gpu_launched=False)
    started = time.monotonic()
    try:
        sys.path.insert(0, str(ASSETS / 'source_snapshot'))
        from scripts import eval_multibackbone_source as source
        from scripts import eval_multibackbone_guidance as guidance
        assert source.REPO == ASSETS / 'source_snapshot'
        manifest = source.read_manifest(ASSETS, MANIFEST_SHA)
        assert manifest['source_snapshot_sha256'] == SOURCE_SHA
        old_receipt = json.loads((ROOT / 'prepared_v1/accepted_cpu_preflight_receipt.json').read_text())
        old_report = ROOT / 'prepared_v1/cpu_preflight.json'; verify(old_report, old_receipt['report_identity'])
        previous = json.loads(old_report.read_text())
        contracts = {(r['dataset'], r['backbone']): r for r in previous['models']}
        assert len(contracts) == 8 and previous['cuda_initialized'] is False
        import torch
        assert torch.__version__ == '2.1.2+cu121' and not torch.cuda.is_initialized()
        hook = source._declared_asset(ASSETS, manifest, manifest['guidance']['native_bootstrap'])
        spec = importlib.util.spec_from_file_location('sensitivity_native_cpu', hook)
        native_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(native_module)
        native = native_module.bootstrap(ASSETS)
        assert native['no_build'] and native['bindings'] == previous['native']['bindings'] and not torch.cuda.is_initialized()
        os.chdir(str(source.REPO))
        rows = []
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            data = Path('/content/3DD-TTA/data/modelnet40_c') if dataset == 'modelnet40_c' else ASSETS / 'datasets/scanobjectnn_c'
            cfg = 'tta_modelnet.yaml' if dataset == 'modelnet40_c' else 'tta_scanobj.yaml'
            checkpoint = 'modelnet_jt.pth' if dataset == 'modelnet40_c' else 'scanobject_jt.pth'
            for method in ('gsd', 'original3dd'):
                command = ['--dataset', dataset, '--dataset-root', str(data), '--label-path', str(data / 'label.npy'),
                    '--asset-root', str(ASSETS), '--asset-manifest-sha256', MANIFEST_SHA,
                    '--result-root', str(ROOT / 'not_used_by_sensitivity_cpu'), '--method', method,
                    '--backbones', 'dgcnn', 'curvenet', 'pointnext', 'pointmae',
                    '--pointmae-config', 'source_snapshot/cfgs/' + cfg, '--pointmae-checkpoint', 'checkpoints/' + checkpoint]
                if method == 'gsd':
                    command += ['--trajectory-observe', '--component-arm', 'combined']
                configs = guidance.configurations(command)
                assert len(configs) == 4
                for config in configs:
                    reference = guidance.matched_source(config, manifest, manifest['guidance'])
                    with patch.object(torch.nn.Module, 'cuda', lambda model, *args, **kwargs: model), \
                         patch.object(torch.Tensor, 'cuda', side_effect=RuntimeError('CPU proof forbids GPU tensors')), \
                         patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU proof forbids CUDA initialization')), \
                         patch.object(torch.nn.Module, '_call_impl', side_effect=RuntimeError('CPU proof forbids forward')):
                        model = source.load_classifier(config, manifest)
                    old_contract = contracts[(dataset, config['backbone'])]
                    assert config['checkpoint_load']['strict'] and not config['checkpoint_load']['missing_keys'] and not config['checkpoint_load']['unexpected_keys']
                    assert config['checkpoint_load']['checkpoint_sha256'] == old_contract['checkpoint_load']['checkpoint_sha256']
                    state_sha = source.state_hash(model)
                    assert state_sha == old_contract['state_sha256'] and config['resolved_model_config'] == old_contract['resolved_model_config']
                    assert all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters())
                    assert all(b.device.type == 'cpu' for b in model.buffers()) and not any(layer.training for layer in model.modules())
                    assert not torch.cuda.is_initialized()
                    row = {k: config[k] for k in ('dataset', 'backbone', 'method', 'num_classes', 'n_examples', 'split', 'checkpoint_load', 'resolved_model_config')}
                    row.update(state_sha256=state_sha, matched_source=reference, forward_executed=False, cpu_frozen_eval=True)
                    rows.append(row)
                    (output / 'partial_strict_loads.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print('AGENT_B_SENSITIVITY_STRICT_CPU_HEAD', method, dataset, config['backbone'], flush=True)
                    del model
        assert len(rows) == 16 and len({(r['method'], r['dataset'], r['backbone']) for r in rows}) == 16
        report = dict(status='complete', session=SESSION, manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA,
                      models=rows, native=native, cuda_initialized=False, python=sys.version, torch=torch.__version__, research_gpu_launched=False)
        (output / 'cpu_preflight.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        state.update(status='complete', strict_loads=16, cuda_initialized=False)
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        receipt = dict(state=state, archive=seal(output, DRIVE / 'phase-evidence' / output.name / 'cpu_preflight_evidence.zip'),
                       report_identity=identity(output / 'cpu_preflight.json') if (output / 'cpu_preflight.json').exists() else None)
        (ROOT / 'sensitivity_cpu_preflight_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (Path(receipt['archive']['drive_path']).parent / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('AGENT_B_SENSITIVITY_CPU_RECEIPT', json.dumps(receipt['archive']), flush=True)
    shutil.copyfile(output / 'cpu_preflight.json', PREPARED / 'cpu_preflight.json')
    (PREPARED / 'accepted_cpu_preflight_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (PREPARED / '.complete').write_text(MANIFEST_SHA + '\n')
    print('AGENT_B_SENSITIVITY_CPU_PREFLIGHT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
