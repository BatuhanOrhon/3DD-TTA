#!/usr/bin/env python3
"""Copy the pinned official checkout and package our small research overlay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ('pointcloud_proximal.py','cloudfixer_diffusion.py','cloudfixer_store.py',
           'research_artifacts.py','scripts/run_cloudfixer_transfer.py','cfgs/cloudfixer_transfer_assets.json')


def colab_phases(output: Path) -> None:
    """Fixed account/deployment adapters; all inference stays in the tested worker."""
    authenticate='''import subprocess
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-c',AUTH_CHILD]
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as child:
    for line in child.stdout:print(line,end='',flush=True)
    assert child.wait()==0
print('CF_TRANSFER_DRIVE_MOUNT_READY',flush=True)
'''
    child='''import sys,time,json,hashlib,zipfile
from pathlib import Path
sys.path.insert(0,'/content/cf_transfer')
from cloudfixer_store import *
started=time.monotonic()
base=Path('/content/cf_transfer_package');root=Path('/content/cf_transfer')
spec=json.loads((root/'manifest.json').read_text())
setup_start=json.loads(Path('/content/cf_transfer_setup_started.json').read_text())
receipt_path=Path('/content/cf_transfer_setup_receipt.json')
if not receipt_path.exists():
    previous=Path('/content/cf_transfer_previous_setups.json')
    attempts=json.loads(previous.read_text()) if previous.exists() else []
    attempts.append(dict(session=setup_start['session'],started_utc=setup_start['started_utc'],
                         seconds=max(0.,time.time()-setup_start['started_epoch'])))
    receipt_path.write_bytes(canonical(dict(attempts=attempts)))
assert Path('/content/drive').is_mount(),'Existing Google Drive mount is required'
parent=Path('/content/drive/MyDrive/thesis/result')/('cloudfixer_transfer_20261114_mount_'+sha(canonical(spec))[:12])
store=MountStore(parent);store.put('task.json',canonical(spec))
record_setup(store,json.loads(receipt_path.read_text()),cap=spec['registration']['budget_seconds'])
budget=Budget(store,cap=spec['registration']['budget_seconds'],started=started)
with budget.operation(600):
    for name in ('official_cloudfixer.tar','research_overlay.zip','manifest.json','package_receipt.json'):
        store.put(name,(base/name).read_bytes())
    evidence=base/'canonical_setup_evidence.zip'
    if not evidence.exists():
        paths=[Path('/content/cli_setup_verification.json'),Path('/content/cf_transfer_setup_started.json'),receipt_path,
               *Path('/content/3DD-TTA/.colab_setup/native').glob('*.complete')]
        with zipfile.ZipFile(evidence,'x',zipfile.ZIP_DEFLATED) as archive:
            for p in paths:
                assert p.is_file() and not p.is_symlink()
                archive.write(p,p.name)
    store.put(evidence.name,evidence.read_bytes())
    task=dict(drive_folder=str(parent),task_sha256=sha(canonical(spec)),session=setup_start['session'],storage='drive_mount')
    Path('/content/cf_transfer_task.json').write_bytes(canonical(task))
    print('CF_TRANSFER_TASK_FOLDER',json.dumps(task),flush=True)
'''
    (output/'prepare_drive_mount.py').write_text(authenticate.replace('AUTH_CHILD',repr(child)))
    archive_child='''import sys,time,json,hashlib,zipfile
from pathlib import Path
started=time.monotonic();sys.path.insert(0,'/content/cf_transfer')
from cloudfixer_store import *
task=json.loads(Path('/content/cf_transfer_task.json').read_text())
store=MountStore(task['drive_folder'])
spec=json.loads(Path('/content/cf_transfer/manifest.json').read_text())
budget=Budget(store,cap=spec['registration']['budget_seconds'],started=started)
output=Path(sys.argv[1]);archive=output.with_suffix('.zip')
with budget.operation(600):
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            assert not p.is_symlink()
            if p.is_file():z.write(p,p.relative_to(output.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    data=archive.read_bytes();store.put(archive.name,data)
    receipt=dict(archive=str(archive),bytes=len(data),sha256=sha(data),drive_folder=task['drive_folder'],
                 mounted_archive=str(store.folder/archive.name),verification='DriveFS fsync/rename/SHA readback; server sync unproven')
    output.with_suffix('.receipt.json').write_bytes(canonical(receipt))
    print('CF_TRANSFER_ARCHIVE_RECEIPT',json.dumps(receipt),flush=True)
'''
    for phase in ('smoke','profile','development'):
        run='''import datetime,json,queue,shutil,subprocess,threading,time
from pathlib import Path
PHASE=PHASE_VALUE
task=json.loads(Path('/content/cf_transfer_task.json').read_text())
assert Path('/content/cf_transfer_deployment.complete').read_text()=='verified\\n'
root=Path('/content/cf_transfer_runs');root.mkdir(exist_ok=True)
run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')+'_'+PHASE
output=root/run_id;log=output.with_suffix('.log')
assert not output.exists() and not log.exists()
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',
    '/content/cf_transfer/scripts/run_cloudfixer_transfer.py','--manifest','/content/cf_transfer/manifest.json',
    '--phase',PHASE,'--drive-folder',task['drive_folder'],'--output',str(output)]
lines=[];stream=log.open('x')
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
    pending=queue.Queue()
    def read():
        for line in child.stdout:pending.put(line)
    reader=threading.Thread(target=read,daemon=True);reader.start();tick=time.monotonic()
    while child.poll() is None or reader.is_alive() or not pending.empty():
        try:
            line=pending.get(timeout=1);lines.append(line.rstrip());stream.write(line);stream.flush();print(line,end='',flush=True)
        except queue.Empty:pass
        if time.monotonic()-tick>=60:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'CF_TRANSFER_WORKER_ALIVE',PHASE,'pid',child.pid,flush=True)
            tick=time.monotonic()
    status=child.wait()
stream.close()
output.mkdir(exist_ok=True)
shutil.copyfile(log,output/'stdout.log')
(output/'exit_status.json').write_text(json.dumps(dict(returncode=status))+'\\n')
archive=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u','-c',ARCHIVE_CHILD,str(output)]
with subprocess.Popen(archive,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as collector:
    for line in collector.stdout:print(line,end='',flush=True)
    assert collector.wait()==0,'Archive pending: preserve local output/runtime'
assert status==0,'Failed phase preserved; inspect before recovery'
assert 'CF_TRANSFER_PHASE_DONE '+PHASE in lines
print('CF_TRANSFER_OUTER_'+PHASE.upper()+'_DONE',flush=True)
'''.replace('PHASE_VALUE',repr(phase)).replace('ARCHIVE_CHILD',repr(archive_child))
        (output/('run_'+phase+'.py')).write_text(run)


def repair_import(source: str) -> str:
    old = '# from pointnet2_ops import pointnet2_utils'
    if source.count(old) != 1:
        raise ValueError('Pinned PointNet++ import repair does not match author source')
    return source.replace(old, 'from pointnet2_ops import pointnet2_utils')


def manifest(official: dict, classifier='pointMAE', *, hypotheses=False) -> dict:
    from scripts.run_cloudfixer_transfer import registration
    assets=json.loads((ROOT/'cfgs/cloudfixer_transfer_assets.json').read_text())
    if hypotheses:
        if classifier != 'pointMAE':
            raise ValueError('Hypothesis confirmation is registered for Point-MAE')
        from scripts.run_cloudfixer_hypotheses import registration as hypothesis_registration
        extra = ('scripts/run_cloudfixer_hypotheses.py', 'scripts/run_cloudfixer_exploration.py',
                 'scripts/analyze_cloudfixer_hypotheses.py', 'scripts/analyze_cloudfixer_exploration.py',
                 'scripts/analyze_cloudfixer_weighted_pm.py', 'scripts/analyze_legacy_trajectory_macro14.py',
                 'scripts/analyze_legacy_trajectory.py')
        base = manifest(official)
        base['registration'] = hypothesis_registration()
        base['prior_manifest_sha256'] = 'b09f8b7730d1b153f9ac4633b20ead8d002e59b71dd4a9ef07e9376436aa7e38'
        base['overlay_files'].update({name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in extra})
        return base
    if classifier == 'point2vec':
        from scripts.run_cloudfixer_point2vec import registration as point2vec_registration
        head = json.loads((ROOT/'cfgs/cloudfixer_point2vec_assets.json').read_text())
        overlay = (*OVERLAY, 'scripts/run_cloudfixer_point2vec.py', 'cfgs/cloudfixer_point2vec_assets.json',
                   'scripts/setup_cloudfixer_point2vec_colab.py')
        return dict(schema=2, classifier=head, classifier_sha256=head['classifier_sha256'],
                    author_commit=assets['author_commit'], official_files=official,
                    overlay_files={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in overlay},
                    dataset_files=assets['dataset_files'], diffusion_sha256=assets['diffusion_sha256'],
                    registration=point2vec_registration(), source_policy='Unmodified author git archive')
    if classifier != 'pointMAE':
        raise ValueError('Unknown classifier: ' + classifier)
    return dict(schema=1, storage='google_drive_mount_fsync_rename_sha256',author_commit=assets['author_commit'],official_files=official,
        overlay_files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in OVERLAY},
        dataset_files=assets['dataset_files'],classifier_sha256=assets['classifier_sha256'],
        diffusion_sha256=assets['diffusion_sha256'],registration=registration(),
        source_policy='Exact git archive plus known utils/misc.py import repair; no model/data/classifier rewrite')


def package(source: Path, output: Path, classifier='pointMAE', *, hypotheses=False) -> None:
    assets=json.loads((ROOT/'cfgs/cloudfixer_transfer_assets.json').read_text())
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit != assets['author_commit']:
        raise ValueError('Official checkout is not the inspected author revision')
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'official_cloudfixer.tar'
    subprocess.run(['git','-C',str(source),'archive','--format=tar','--output',str(archive.resolve()),commit],check=True)
    official={}
    with tarfile.open(archive) as tar:
        for item in tar:
            name=PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or not (item.isfile() or item.isdir()):
                raise ValueError('Unsafe author archive entry')
            if item.isfile():
                data=tar.extractfile(item).read()
                if classifier=='pointMAE' and item.name=='utils/misc.py':
                    data=repair_import(data.decode()).encode()
                official[item.name]=hashlib.sha256(data).hexdigest()
    spec=manifest(official, classifier, hypotheses=hypotheses)
    (output/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    with zipfile.ZipFile(output/'research_overlay.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in spec['overlay_files']:
            z.write(ROOT/name,name)
        z.write(output/'manifest.json','manifest.json')
    receipt={name:dict(bytes=(output/name).stat().st_size,
        sha256=hashlib.sha256((output/name).read_bytes()).hexdigest())
        for name in ('official_cloudfixer.tar','research_overlay.zip','manifest.json')}
    (output/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if classifier == 'point2vec' or hypotheses:
        print(json.dumps(receipt, indent=2))
        return
    # This is an execution adapter, not a reimplementation of any author model.
    deploy = '''import hashlib,json,os,subprocess,tarfile,zipfile
from pathlib import Path,PurePosixPath
BASE=Path('/content/cf_transfer_package')
EXPECTED=PACKAGE_RECEIPT
for name,value in EXPECTED.items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==value['sha256'],name
official=Path('/content/CloudFixer')
overlay=Path('/content/cf_transfer')
assert not official.exists() and not overlay.exists(),'Inspect existing source before recovery'
official.mkdir();overlay.mkdir()
with tarfile.open(BASE/'official_cloudfixer.tar') as tar:
    for item in tar:
        name=PurePosixPath(item.name)
        assert not name.is_absolute() and '..' not in name.parts and (item.isfile() or item.isdir())
    tar.extractall(official)
with zipfile.ZipFile(BASE/'research_overlay.zip') as z:
    for item in z.infolist():
        name=PurePosixPath(item.filename)
        assert not name.is_absolute() and '..' not in name.parts
    z.extractall(overlay)
path=official/'utils/misc.py'
old=path.read_text();needle='# from pointnet2_ops import pointnet2_utils'
assert old.count(needle)==1
path.write_text(old.replace(needle,'from pointnet2_ops import pointnet2_utils'))
spec=json.loads((overlay/'manifest.json').read_text())
for name,digest in spec['official_files'].items():
    assert hashlib.sha256((official/name).read_bytes()).hexdigest()==digest,name
def run(command,cwd=None):
    print('CF_DEPENDENCY_COMMAND',command,flush=True)
    with subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
        for line in child.stdout:print(line,end='',flush=True)
        assert child.wait()==0,command
environment=['conda','run','--no-capture-output','-n','3dd_tta_env']
run(environment+['python','-m','pip','install','fvcore==0.1.5.post20221221','iopath==0.1.10'])
build=r"""
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 FORCE_CUDA=1
python -m pip install --no-build-isolation --no-deps -v git+https://github.com/facebookresearch/pytorch3d.git@2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
python -m pip install --no-build-isolation --no-deps git+https://github.com/otaheri/chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846
"""
run(environment+['bash','-c',build])
weights=official/'checkpoints';weights.mkdir(exist_ok=True)
import shutil
shutil.copyfile('/content/3DD-TTA/pointnet_ckpts/modelnet_jt.pth',weights/'pointMAE_modelnet40.pth')
run(['gdown',DIFFUSION_ID,'-O',str(weights/'diffusion_model_transformer_modelnet40.npy')])
for name,digest in [('pointMAE_modelnet40.pth',spec['classifier_sha256']),('diffusion_model_transformer_modelnet40.npy',spec['diffusion_sha256'])]:
    assert hashlib.sha256((weights/name).read_bytes()).hexdigest()==digest,name
run(environment+['python','-c',"import torch,adapt; from utils.misc import fps; x=torch.rand(2,64,3,device='cuda'); assert fps(x,32).shape==(2,32,3); print('CF_NATIVE_IMPORTS_PASSED')"],cwd=official)
Path('/content/cf_transfer_deployment.complete').write_text('verified\\n')
print('CF_TRANSFER_DEPLOYMENT_DONE',flush=True)
'''.replace('PACKAGE_RECEIPT',repr(receipt)).replace('DIFFUSION_ID',repr(assets['diffusion_drive_id']))
    (output/'deploy_official.py').write_text(deploy)
    colab_phases(output)
    print(json.dumps(dict(output=str(output.resolve()),author_commit=commit,official_files=len(official),
                          receipt=receipt),indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--official-checkout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--classifier', choices=('pointMAE', 'point2vec'), default='pointMAE')
    parser.add_argument('--hypotheses', action='store_true', help='Package the Point-MAE confirmation campaign')
    args=parser.parse_args()
    package(args.official_checkout,args.output,args.classifier,hypotheses=args.hypotheses)


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
