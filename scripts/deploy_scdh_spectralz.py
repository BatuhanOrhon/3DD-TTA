"""Owned task deployment and strict CPU no-forward preflight after canonical setup."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from unittest.mock import patch

BASE=Path('/content/drive/MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T110110Z/r8_assets_v7_full.zip')
BASE_ID=dict(bytes=247002524,sha256='ce9c797ccacb4eeffcc0a2836c65109213c9c7920a40a5145ef598b2adac1333')
BASE_MANIFEST='78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9'
ROOT=Path('/content/studies/scdh-spectralz')
ASSETS=ROOT/'prepared/assets'
DRIVE=Path('/content/drive/MyDrive/thesis/result/scdh_spectralz_search_20261101')


def helper():
    path=Path(__file__).with_name('deploy_assets_template.py')
    if not path.is_file():path=Path(__file__).resolve().parents[1]/'result/preparation/chamfer_depth_transfer_20261031/operator/deploy_assets_template.py'
    spec=importlib.util.spec_from_file_location('_scdh_deploy_tools',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT,module.ASSETS=ROOT,ASSETS
    return module


def cpu_preflight(receipt,output):
    sys.path.insert(0,str(ASSETS/'source_snapshot'));os.chdir(ASSETS/'source_snapshot')
    import torch
    from scripts import eval_scdh_spectralz as worker
    source=worker.shared.source
    tools=helper();manifest=source.read_manifest(ASSETS,receipt['manifest']['sha256'])
    tools.require(sys.version_info[:2]==(3,8) and torch.__version__=='2.1.2+cu121' and not torch.cuda.is_initialized(),'Fresh pinned CPU process required')
    flags=['--stage','technical','--arm','off','--preview','--dataset-root','/content/3DD-TTA/data/modelnet40_c',
        '--label-path','/content/3DD-TTA/data/modelnet40_c/label.npy','--asset-root',str(ASSETS),
        '--asset-manifest-sha256',receipt['manifest']['sha256'],'--result-root',str(ROOT/'unused_cpu_result'),
        '--paired-state-root',str(ROOT/'technical/paired/seed0'),'--index-manifest',str(ASSETS/'study/technical_ids.json'),
        '--pointmae-config',str(ASSETS/'source_snapshot/cfgs/tta_modelnet.yaml'),
        '--pointmae-checkpoint',str(ASSETS/'checkpoints/modelnet_jt.pth')]
    cfg,=worker.configurations(flags)
    with patch.object(torch.nn.Module,'cuda',lambda model,*a,**kw:model), \
         patch.object(torch.Tensor,'cuda',side_effect=RuntimeError('CPU proof forbids GPU tensors')), \
         patch.object(torch.cuda,'_lazy_init',side_effect=RuntimeError('CPU proof forbids CUDA initialization')), \
         patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('CPU proof forbids model forward')):
        spec=importlib.util.spec_from_file_location('_scdh_cpu_native',ASSETS/manifest['guidance']['native_bootstrap'])
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        proof=native.bootstrap(ASSETS)
        tools.require(proof['no_build'] is True and len(proof['bindings'])==4,'Audited no-build native proof missing')
        model=source.load_classifier(cfg,manifest)
        tools.require(all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
            and all(b.device.type=='cpu' for b in model.buffers()) and not any(m.training for m in model.modules()),'Expected frozen/eval CPU classifier')
        provenance=source.dataset_manifest(manifest,cfg)
        for name,expected in provenance['files'].items():
            source.load_array(Path(cfg['dataset_root'])/name,expected,cfg,labels=name=='label.npy')
        state=source.state_hash(model)
        del model
    tools.require(not torch.cuda.is_initialized(),'CPU proof initialized CUDA')
    native_extensions={name:dict(path=str(Path(module.__file__).resolve()),**tools.identity(Path(module.__file__)))
        for name,module in tuple(sys.modules.items()) if getattr(module,'__file__',None)
        and Path(module.__file__).suffix=='.so' and any(fragment in name.lower() for fragment in ('pointnet2','knn','chamfer'))}
    result=dict(status='complete',cuda_initialized=False,forward_executed=False,manifest_sha256=receipt['manifest']['sha256'],
        source_snapshot_sha256=manifest['source_snapshot_sha256'],checkpoint_load=cfg['checkpoint_load'],
        classifier_state_sha256=state,dataset_provenance=provenance,native=proof,native_extensions=native_extensions,
        lion_assets={k:manifest['guidance'][k] for k in ('lion_config','lion_checkpoint')},
        scope='ModelNet40-C/PointMAE CPU strict-load/data/native/source proof; no classifier or LION forward')
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    return result


def deploy(receipt_path,payload):
    tools=helper()
    tools.require(os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir(),'Actual Drive mount required')
    setup=json.loads(Path('/content/cli_setup_verification.json').read_text())
    tools.require(setup.get('status')=='passed','Canonical ten-cell verification must pass first')
    tools.require(not ASSETS.parent.exists(),'Preserve existing/partial deployment; fresh target required')
    receipt=json.loads(receipt_path.read_text())
    tools.require(receipt.get('status')=='sealed_CPU_only','Sealed CPU preparation receipt required')
    output=ROOT/('deployment_'+tools.utc());output.mkdir(parents=True,exist_ok=False)
    drive=DRIVE/'phase-evidence'/output.name;drive.mkdir(parents=True,exist_ok=False)
    result=dict(status='running')
    try:
        tools.verify(BASE,BASE_ID);tools.verify(payload,receipt['payload'])
        tools.extract_checked(BASE,ASSETS.parent)
        tools.require(tools.identity(ASSETS/'manifest.json')['sha256']==BASE_MANIFEST,'Wrong full base bytes')
        import zipfile
        with zipfile.ZipFile(payload) as archive:
            entries=tools.checked_members(archive)
            tools.require({e.filename for e in entries if not e.is_dir()}=={'assets/manifest.json','assets/source_snapshot.zip'},'Unexpected delta members')
        tools.extract_checked(payload,ASSETS.parent,overlay=True)
        tools.verify(ASSETS/'manifest.json',receipt['manifest']);tools.verify(ASSETS/'source_snapshot.zip',receipt['source_snapshot'])
        manifest=json.loads((ASSETS/'manifest.json').read_text())
        tools.require(manifest.get('manifest_revision')=='scdh_spectralz_v1','Wrong task revision')
        tools.extract_checked(ASSETS/'source_snapshot.zip',ASSETS,overlay=True,declared=manifest['files'])
        for name,expected in manifest['files'].items():
            if Path(name).parts[0] not in ('checkpoints','datasets'):tools.verify(ASSETS/name,expected)
        target=ASSETS/'checkpoints/modelnet_jt.pth';expected=manifest['files']['checkpoints/modelnet_jt.pth']
        if not target.exists():tools.copy_checked('/content/3DD-TTA/pointnet_ckpts/modelnet_jt.pth',target,expected)
        tools.verify(target,expected)
        for key in ('lion_config','lion_checkpoint'):
            entry=manifest['guidance'][key];tools.verify(entry['path'],entry)
        restore=tools.restore_native(manifest,output,drive)
        tools.dependency_check(output)
        shutil.copyfile(receipt_path,output/'preparation_identity.json')
        script=ASSETS/'source_snapshot/scripts/deploy_scdh_spectralz.py'
        subprocess.run([sys.executable,'-u',str(script),'--cpu-only','--receipt',str(receipt_path),
                        '--cpu-output',str(output/'cpu_preflight.json')],check=True,timeout=600)
        result=json.loads((output/'cpu_preflight.json').read_text())
        result.update(native_restore=restore,setup_receipt=setup)
    except BaseException:
        result=dict(status='failed',failure=traceback.format_exc(),research_gpu_executed=False)
        raise
    finally:
        with (output/'deployment.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
        archive=tools.seal(output,drive/'deployment_cpu_evidence.zip')
        result['archive']=archive
        with (ROOT/'cpu_preflight_receipt.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
        tools.copy_checked(ROOT/'cpu_preflight_receipt.json',drive/'receipt.json',tools.identity(ROOT/'cpu_preflight_receipt.json'))
    print('SCDH_SPECTRALZ_CPU_DEPLOYMENT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--payload',type=Path)
    parser.add_argument('--cpu-only',action='store_true');parser.add_argument('--cpu-output',type=Path)
    args=parser.parse_args()
    if args.cpu_only:cpu_preflight(json.loads(args.receipt.read_text()),args.cpu_output)
    else:deploy(args.receipt,args.payload)

if __name__=='__main__':main()
