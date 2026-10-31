"""Owned runtime only: verified fresh overlay deployment and eight-head CPU proof.

Run after all ten canonical setup cells. Never executes a model forward, research
entrypoint, GPU allocation or Colab command. Failure evidence is sealed as failure.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_assets_template as tools

ROOT = Path('/content/studies/chamfer-depth-transfer')
ASSETS = ROOT / 'prepared/assets'
DRIVE = Path('/content/drive/MyDrive/thesis/result/chamfer_depth_transfer_20261031')
BASE = Path('/content/drive/MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T110110Z/r8_assets_v7_full.zip')
BASE_ID = dict(bytes=247002524, sha256='ce9c797ccacb4eeffcc0a2836c65109213c9c7920a40a5145ef598b2adac1333')
BASE_MANIFEST_SHA = '78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9'
PAYLOAD = Path('/content/depth_transfer_assets_v1.zip')
POINTNEXT_SO = dict(bytes=12449056, sha256='a2db745611c01ea03fab6e5d5e1d9e5f08837e847780dd27555eec561bc58aab')
CRITICAL_SOURCES = ('tta.py', 'shared_trajectory.py', 'legacy_trajectory.py', 'research_artifacts.py',
                    'classifier_backbones.py', 'scripts/eval_chamfer_depth.py', 'scripts/eval_multibackbone_guidance.py',
                    'scripts/eval_multibackbone_source.py', 'scripts/analyze_legacy_trajectory.py')


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def deploy(receipt, output, drive_output):
    """All bulk comes from the exact base; source overlay cannot replace bulk files."""
    prepared = ASSETS.parent
    tools.require(not prepared.exists(), 'Fresh prepared destination required; preserve any partial deployment')
    tools.verify(BASE, BASE_ID)
    tools.verify(PAYLOAD, receipt['payload'])
    tools.extract_checked(BASE, prepared)
    tools.require(tools.identity(ASSETS / 'manifest.json')['sha256'] == BASE_MANIFEST_SHA, 'Full base manifest SHA mismatch')
    import zipfile
    with zipfile.ZipFile(PAYLOAD) as archive:
        entries = tools.checked_members(archive)
        tools.require({e.filename for e in entries if not e.is_dir()} == {'assets/manifest.json', 'assets/source_snapshot.zip'},
                      'Overlay payload must contain only manifest and source snapshot')
    tools.extract_checked(PAYLOAD, prepared, overlay=True)
    tools.verify(ASSETS / 'manifest.json', receipt['manifest'])
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    tools.require(manifest.get('manifest_revision') == 'chamfer_depth_transfer_v1' and
                  manifest['guidance']['active_backbones'] == ['pointmae', 'dgcnn', 'curvenet', 'pointnext'], 'Wrong task manifest')
    tools.verify(ASSETS / 'source_snapshot.zip', receipt['source_snapshot'])
    tools.require(receipt['source_snapshot']['sha256'] == manifest['source_snapshot_sha256'], 'Source archive receipt mismatch')
    with zipfile.ZipFile(ASSETS / 'source_snapshot.zip') as archive:
        entries = tools.checked_members(archive)
        tools.require(all(not e.filename.startswith(('checkpoints/', 'datasets/')) for e in entries), 'Bulk overwrite forbidden in source overlay')
    tools.extract_checked(ASSETS / 'source_snapshot.zip', ASSETS, overlay=True, declared=manifest['files'])
    for name, location in (('modelnet_jt.pth', '/content/3DD-TTA/pointnet_ckpts/modelnet_jt.pth'),
                           ('scanobject_jt.pth', '/content/drive/MyDrive/thesis/weights/scanobject_jt.pth')):
        target = ASSETS / 'checkpoints' / name
        expected = manifest['files']['checkpoints/' + name]
        if target.exists():
            tools.verify(target, expected)
        else:
            tools.copy_checked(location, target, expected)
    for name, expected in manifest['files'].items():
        tools.require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe manifest member')
        tools.verify(ASSETS / name, expected)
    for name in ('lion_checkpoint', 'lion_config'):
        entry = manifest['guidance'][name]
        tools.verify(entry['path'], entry)
    tools.verify(ASSETS / 'pointnet2_batch_cuda.cpython-38-x86_64-linux-gnu.so', POINTNEXT_SO)
    tools.ROOT, tools.ASSETS = ROOT, ASSETS
    native = tools.restore_native(manifest, output, drive_output)
    tools.dependency_check(output)
    shutil.copyfile(ASSETS / 'manifest.json', output / 'manifest.json')
    shutil.copyfile(ASSETS / 'source_snapshot.zip', output / 'source_snapshot.zip')
    return manifest, native


def preflight(receipt, manifest, native_restore):
    tools.require(sys.version_info[:2] == (3, 8), 'CPU proof requires Python3.8 ABI')
    setup = json.loads(Path('/content/cli_setup_verification.json').read_text())
    tools.require(setup.get('status') == 'passed', 'All ten canonical setup cells must pass before deployment')
    source_root = ASSETS / 'source_snapshot'
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    from scripts import eval_multibackbone_source as source
    from scripts import eval_chamfer_depth as worker
    tools.require(source.REPO == source_root, 'CPU proof imported a different source checkout')
    source.read_manifest(ASSETS, receipt['manifest']['sha256'])
    import torch
    tools.require(torch.__version__ == '2.1.2+cu121' and not torch.cuda.is_initialized(), 'CPU proof requires fresh pinned Torch process')
    heads = {}
    with patch.object(torch.nn.Module, 'cuda', lambda model, *args, **kwargs: model), \
         patch.object(torch.Tensor, 'cuda', side_effect=RuntimeError('CPU proof forbids GPU tensors')), \
         patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU proof forbids CUDA initialization')), \
         patch.object(torch.nn.Module, '_call_impl', side_effect=RuntimeError('CPU proof forbids model forward')):
        spec = importlib.util.spec_from_file_location('depth_transfer_native_cpu', ASSETS / manifest['guidance']['native_bootstrap'])
        native_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_module)
        native = native_module.bootstrap(ASSETS)
        tools.require(native.get('no_build') is True and len(native.get('bindings', {})) == 4, 'Pinned native no-build bootstrap failed')
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            data = Path('/content/3DD-TTA/data/modelnet40_c') if dataset == 'modelnet40_c' else ASSETS / 'datasets/scanobjectnn_c'
            selection = ASSETS / 'study' / (dataset + '_full_ids.json')
            checkpoint = ASSETS / 'checkpoints' / ('modelnet_jt.pth' if dataset == 'modelnet40_c' else 'scanobject_jt.pth')
            options = ['--transfer-study', '--dataset', dataset, '--dataset-root', str(data), '--label-path', str(data / 'label.npy'),
                       '--asset-root', str(ASSETS), '--asset-manifest-sha256', receipt['manifest']['sha256'],
                       '--result-root', str(ROOT / 'unused_cpu_result'), '--paired-state-root', str(ROOT / 'paired/seed0'),
                       '--index-manifest', str(selection), '--pointmae-config', str(source_root / 'cfgs/tta_modelnet.yaml'),
                       '--pointmae-checkpoint', str(checkpoint)]
            configs = worker.configurations(options)
            tools.require(len(configs) == 4, 'CPU proof requires exactly four heads per dataset')
            dataset_manifest = source.dataset_manifest(manifest, configs[0])
            for name, expected in dataset_manifest['files'].items():
                source.load_array(data / name, expected, configs[0], labels=name == 'label.npy')
            heads[dataset] = {}
            for config in configs:
                model = source.load_classifier(config, manifest)
                load = config['checkpoint_load']
                tools.require(load['strict'] is True and not load['missing_keys'] and not load['unexpected_keys'], 'Strict checkpoint proof failed')
                tools.require(all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters()) and
                              all(b.device.type == 'cpu' for b in model.buffers()) and not any(layer.training for layer in model.modules()),
                              'Classifier must be frozen/eval and entirely CPU')
                heads[dataset][config['backbone']] = dict(checkpoint_load=load, dataset_provenance=dataset_manifest,
                    classifier_state_sha256=source.state_hash(model), config=config)
                del model
    tools.require(not torch.cuda.is_initialized(), 'CPU proof initialized CUDA')
    critical = {'assets/source_snapshot/' + name: tools.identity(source_root / name) for name in CRITICAL_SOURCES}
    for name, actual in critical.items():
        tools.require(actual == manifest['files'][name[len('assets/'):]], 'Critical source identity mismatch: ' + name)
    return dict(status='complete', cuda_initialized=False, forward_executed=False, manifest_sha256=receipt['manifest']['sha256'],
                source_snapshot_sha256=manifest['source_snapshot_sha256'], heads=heads, native=native,
                native_restore=native_restore, required_source_files=critical, setup_receipt=setup,
                python=sys.version, torch=torch.__version__, lion_assets={k: manifest['guidance'][k] for k in ('lion_checkpoint', 'lion_config')},
                scope='CPU strict load/data/source proof only; no LION forward, no GPU research or inference claim')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=Path(__file__).parent / 'preparation_identity.json')
    args = parser.parse_args(argv)
    tools.require(sys.version_info[:2] == (3, 8), 'Run using /usr/local/envs/3dd_tta_env/bin/python')
    tools.require(not (ROOT / 'cpu_preflight_receipt.json').exists(), 'CPU receipt exists; preserve immutable prior phase')
    tools.require(os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir(), 'Actual Drive mount required')
    setup = json.loads(Path('/content/cli_setup_verification.json').read_text())
    tools.require(setup.get('status') == 'passed', 'Canonical ten-cell setup must pass first')
    receipt = json.loads(args.receipt.read_text())
    tools.require(receipt.get('status') == 'sealed_CPU_only', 'Actual builder receipt required')
    output = ROOT / ('deployment_cpu_' + tools.utc())
    output.mkdir(parents=True, exist_ok=False)
    drive_output = DRIVE / 'phase-evidence' / output.name
    drive_output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', cuda_initialized=False, forward_executed=False)
    succeeded = False
    try:
        for name in ('deploy_and_preflight.py', 'deploy_assets_template.py', 'prepare_assets.py'):
            shutil.copyfile(Path(__file__).parent / name, output / name)
        shutil.copyfile(args.receipt, output / 'preparation_identity.json')
        manifest, native_restore = deploy(receipt, output, drive_output)
        report = preflight(receipt, manifest, native_restore)
        succeeded = True
    except BaseException:
        report = dict(status='failed', failure=traceback.format_exc(), research_gpu_executed=False)
        raise
    finally:
        write_new(output / 'cpu_preflight.json', report)
        sealed = tools.seal(output, drive_output / 'deployment_cpu_evidence.zip')
        result = dict(status='complete' if succeeded else 'failed', report=report, archive=sealed)
        write_new(ROOT / 'cpu_preflight_receipt.json', result)
        write_new(drive_output / 'receipt.json', result)
        tools.verify(sealed['drive_path'], sealed)
    write_new(ASSETS.parent / 'completion.json', dict(manifest_sha256=receipt['manifest']['sha256']))
    print('DEPTH_TRANSFER_CPU_PREFLIGHT_COMPLETE', flush=True)
    return result


if __name__ == '__main__':
    main()
