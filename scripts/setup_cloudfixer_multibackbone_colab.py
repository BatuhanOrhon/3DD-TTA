#!/usr/bin/env python3
"""Owned-VM deployment after canonical10 and user-operated Drive consent."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile

DRIVE_MOUNT=Path('/content/drive')
TASK_DRIVE_ROOT=DRIVE_MOUNT/'MyDrive/thesis/result/cloudfixer_multibackbone_20261122'
BASELINE_TASK='cloudfixer_baseline_heads_20261123'
BASELINE_PROFILE='cloudfixer_xyz_proximal_pm_n8_k80_late_v1'


def require_task_drive(folder, *, profile=None, baseline_task=BASELINE_TASK):
    if not os.path.ismount(str(DRIVE_MOUNT)):
        raise ValueError('Real Google Drive must be mounted')
    mount=str(DRIVE_MOUNT.resolve())
    if profile == BASELINE_PROFILE:
        if baseline_task not in (BASELINE_TASK, 'surface_four_models_20261128'):
            raise ValueError('Unknown baseline deployment destination')
        root = DRIVE_MOUNT/'MyDrive/thesis/result'/baseline_task
        if root.is_symlink():
            raise ValueError('The baseline task root cannot be a symlink')
    elif profile is None or profile == 'original_cloudfixer':
        root = TASK_DRIVE_ROOT
    else:
        raise ValueError('Unknown task Drive profile')
    parent=str(root.resolve())
    target=Path(folder).resolve()
    if (os.path.commonpath([parent,mount])!=mount
            or os.path.commonpath([str(target),parent])!=parent):
        raise ValueError('Drive destination must belong to this task')
    return target


def backup_package(store, package, receipt):
    for name in (*receipt,'package_receipt.json'):
        store.put('package_'+name,(package/name).read_bytes())


def identity(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):digest.update(data)
    return dict(bytes=Path(path).stat().st_size,sha256=digest.hexdigest())


def gpu_architecture(accelerator, name, capability):
    expected={'A100':(8,0),'L4':(8,9)}
    if accelerator not in expected or accelerator not in name or tuple(capability)!=expected[accelerator]:
        raise ValueError('GPU model/capability differs from registered hardware')
    return '.'.join(map(str,capability))


def head_native_identity(data):
    filename=data.get('filename')
    if (filename!='pointnet2_batch_cuda.cpython-38-x86_64-linux-gnu.so'
            or data.get('accelerator')!='L4' or type(data.get('bytes')) is not int or data['bytes']<=0
            or not isinstance(data.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',data['sha256'])):
        raise ValueError('Invalid L4 head native receipt')
    return filename,{k:data[k] for k in ('bytes','sha256')}


def preload_head_native(root, accelerator):
    if accelerator=='A100':return
    receipt=Path(root)/'head_native'/'receipt.json'
    if accelerator!='L4' or not receipt.is_file():raise ValueError('Verified L4 PointNeXt native build required')
    data=json.loads(receipt.read_text());filename,expected=head_native_identity(data)
    path=receipt.parent/filename
    if identity(path)!=expected:
        raise ValueError('L4 head native identity mismatch')
    module_spec=importlib.util.spec_from_file_location('pointnet2_batch_cuda',path)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module);sys.modules['pointnet2_batch_cuda']=module
    return module


def publish_checkpoint(source, target, expected):
    if source.is_symlink() or target.is_symlink() or identity(source)!=expected:
        raise ValueError('Checkpoint identity/symlink mismatch')
    if target.exists():
        if identity(target)!=expected:raise ValueError('Existing checkpoint conflicts')
        return
    temporary=target.parent/('.pending_'+uuid.uuid4().hex)
    try:
        with source.open('rb') as incoming, temporary.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,1048576);outgoing.flush();os.fsync(outgoing.fileno())
        if identity(temporary)!=expected:raise ValueError('Checkpoint copy failed')
        os.replace(temporary,target)
        if identity(target)!=expected:raise ValueError('Checkpoint readback failed')
    finally:
        if temporary.exists():temporary.unlink()


def native_snapshot(store, paths):
    from cloudfixer_store import canonical
    records={}
    for name,path in sorted(paths.items()):
        path=Path(path);expected=identity(path)
        key='native_'+expected['sha256']+'.so'
        publish_checkpoint(path,store.folder/key,expected)
        records[name]=dict(path=str(path),object=key,**expected)
    store.put('native_restore.json',canonical(records))


def restore_native(store):
    if not (store.folder/'native_restore.json').exists():return
    for record in json.loads(store.get('native_restore.json')).values():
        path=Path(record['path'])
        allowed=('/content/','/root/.cache/torch_extensions/','/usr/local/envs/3dd_tta_env/')
        if not path.is_absolute() or '..' in path.parts or path.suffix!='.so' or not str(path).startswith(allowed):
            raise ValueError('Unexpected native restore destination')
        expected={k:record[k] for k in ('bytes','sha256')}
        cached=store.folder/record['object']
        if record['object']!='native_'+expected['sha256']+'.so' or identity(cached)!=expected:
            raise ValueError('Native backup identity mismatch')
        if path.exists() and identity(path)==expected:continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.parent/('.pending_'+uuid.uuid4().hex)
        try:
            shutil.copyfile(cached,temporary)
            if identity(temporary)!=expected:raise ValueError('Native restore copy failed')
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()


def run(command, cwd=None):
    print('CF_MB_SETUP_COMMAND',command,flush=True)
    done=threading.Event();start=time.monotonic()
    def heartbeat():
        while not done.wait(30):
            print('CF_MB_NATIVE_SETUP_ACTIVE',round(time.monotonic()-start,1),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        with subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            for line in child.stdout:print(line,end='',flush=True)
            status=child.wait()
        if status:raise subprocess.CalledProcessError(status,command)
    finally:
        done.set();thread.join()


def main(*, baseline_task=BASELINE_TASK):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/content/cf_multibackbone'))
    p.add_argument('--drive-folder',type=Path,required=True)
    a=p.parse_args()
    if json.loads(Path('/content/cli_setup_verification.json').read_text()).get('status')!='passed':
        raise ValueError('Canonical ten cells must pass first')
    receipt=json.loads((a.package/'package_receipt.json').read_text())
    for name,expected in receipt.items():
        if identity(a.package/name)!=expected:raise ValueError('Uploaded package mismatch: '+name)
    spec=json.loads((a.package/'manifest.json').read_text())
    profile = spec.get('profile')
    if (spec.get('task') == BASELINE_TASK) != (profile == BASELINE_PROFILE):
        raise ValueError('Package task and baseline profile disagree')
    a.drive_folder=require_task_drive(a.drive_folder, profile=profile, baseline_task=baseline_task)
    official=a.root/'official';code=a.root/'code';assets=a.root/'assets'
    def write(path,data,expected=None):
        if expected and dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())!=expected:
            raise ValueError('Archive member identity mismatch: '+str(path))
        if path.is_symlink():raise ValueError('Symlink target forbidden')
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Existing deployment conflicts: '+str(path))
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.parent/('.pending_'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
    def safe(name):
        part=PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or '\\' in name:raise ValueError('Unsafe archive path')
    with tarfile.open(a.package/'official_cloudfixer.tar') as tar:
        for member in tar:
            safe(member.name)
            if member.isdir():continue
            if not member.isfile():raise ValueError('Author archive may contain only files/directories')
            data=tar.extractfile(member).read()
            if member.name=='utils/misc.py':
                old=b'# from pointnet2_ops import pointnet2_utils'
                if data.count(old)!=1:raise ValueError('Known author import repair changed')
                data=data.replace(old,b'from pointnet2_ops import pointnet2_utils')
            write(official/member.name,data,spec['official_files'][member.name])
    for name,dest,expected in [('head_assets.zip',assets,spec['files']),('overlay.zip',code,spec['overlay_files'])]:
        with zipfile.ZipFile(a.package/name) as z:
            if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):
                raise ValueError('Bad/duplicate archive entries')
            if set(z.namelist())!=set(expected):raise ValueError('Undeclared/missing package files')
            for item in z.infolist():
                safe(item.filename)
                if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink forbidden')
                write(dest/item.filename,z.read(item),expected[item.filename])
    sys.path.insert(0,str(code))
    from cloudfixer_store import MountStore, canonical, record_setup, sha
    store=MountStore(a.drive_folder)
    store.put('task.json',canonical(spec))
    backup_package(store,a.package,receipt)
    weights=official/'checkpoints/diffusion_model_transformer_modelnet40.npy'
    if not weights.exists():
        cached=store.folder/weights.name
        if cached.is_file() and identity(cached)['sha256']==spec['diffusion_sha256']:
            weights.parent.mkdir(parents=True,exist_ok=True)
            publish_checkpoint(cached,weights,identity(cached))
        else:
            weights.parent.mkdir(parents=True,exist_ok=True)
            temporary=weights.parent/('download_'+uuid.uuid4().hex+'.partial')
            run([sys.executable,'-m','gdown',spec['diffusion_drive_id'],'-O',str(temporary)])
            if identity(temporary)['sha256']!=spec['diffusion_sha256']:
                raise ValueError('Downloaded diffusion checkpoint SHA mismatch')
            publish_checkpoint(temporary,weights,identity(temporary));temporary.unlink()
    if identity(weights)['sha256']!=spec['diffusion_sha256']:
        raise ValueError('Official ModelNet40 XYZ checkpoint SHA mismatch')
    publish_checkpoint(weights,store.folder/weights.name,identity(weights))
    store.put('diffusion_checkpoint_receipt.json',canonical(identity(weights)))
    native=a.root/'native_setup.complete'
    env=['conda','run','--no-capture-output','-n','3dd_tta_env']
    accelerator=spec.get('accelerator','A100')
    gpu=json.loads(subprocess.check_output(env+['python','-c',
        'import json,torch;print(json.dumps([torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0)]))'],text=True))
    architecture=gpu_architecture(accelerator,*gpu)
    if not native.exists():
        run(env+['python','-m','pip','install','fvcore==0.1.5.post20221221','iopath==0.1.10','multimethod==1.9.1'])
        build='''set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="{architecture}" MAX_JOBS=4 FORCE_CUDA=1
python -m pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/pytorch3d.git@2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
python -m pip install --no-build-isolation --no-deps git+https://github.com/otaheri/chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846
'''.format(architecture=architecture)
        run(env+['bash','-c',build],cwd=official)
    if accelerator=='L4':
        head_root=a.root/'head_native';head_root.mkdir(exist_ok=True)
        receipt_key='head_native_receipt.json'
        if (store.folder/receipt_key).is_file():
            head_receipt=json.loads(store.get(receipt_key))
            filename,expected=head_native_identity(head_receipt)
            publish_checkpoint(store.folder/('native_'+expected['sha256']+'.so'),head_root/filename,expected)
        else:
            build='export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4; python setup.py build_ext --build-lib '+str(head_root)
            run(env+['bash','-ec',build],cwd=assets/'openpoints/cpp/pointnet2_batch')
            candidates=list(head_root.glob('pointnet2_batch_cuda*.so'))
            if len(candidates)!=1:raise ValueError('Exactly one L4 head native build required')
            binary=candidates[0];head_receipt=dict(accelerator='L4',filename=binary.name,**identity(binary))
            publish_checkpoint(binary,store.folder/('native_'+head_receipt['sha256']+'.so'),identity(binary))
            store.put(receipt_key,canonical(head_receipt))
        (head_root/'receipt.json').write_bytes(canonical(head_receipt))
    restore_native(store)
    probe='''import os,sys,torch
from pathlib import Path
assert sys.version_info[:2]==(3,8)
assert torch.__version__=='2.1.2+cu121' and torch.cuda.is_available()
assert torch.cuda.device_count()==1 and sys.argv[4] in torch.cuda.get_device_name(0)
sys.path.insert(0,sys.argv[2])
from scripts.setup_cloudfixer_multibackbone_colab import preload_head_native
preload_head_native(Path(sys.argv[3]),sys.argv[4])
sys.path.insert(0,sys.argv[1])
from openpoints.utils import EasyConfig
import pointnet2_batch_cuda
cfg=EasyConfig();cfg.load(os.path.join(sys.argv[1],'pointnext_modelnet_cfg.yaml'),recursive=True)
import adapt,pytorch3d,knn_cuda
from pytorch3d.ops import knn_points
p=torch.rand(1,32,3,device='cuda').requires_grad_(True)
d1,d2,_,_=adapt.chamfer_dist_fn(p,p.detach()+.01)
(d1.mean()+d2.mean()).backward()
assert torch.isfinite(p.grad).all() and torch.isfinite(knn_points(p,p,K=5).dists).all()
print('CF_MB_EXTRA_NATIVE_GPU_PASS',flush=True)
'''
    run(env+['python','-c',probe,str(assets),str(code),str(a.root),accelerator],cwd=official)
    native.write_text('verified\n')
    start=json.loads(Path('/content/cf_transfer_setup_started.json').read_text())
    setup_receipt=a.root/'setup_receipt.json'
    if not setup_receipt.exists():
        previous=Path('/content/cf_mb_previous_setups.json')
        attempts=json.loads(previous.read_text()) if previous.exists() else []
        attempts.append(dict(session=start['session'],started_utc=start['started_utc'],seconds=max(0.,time.time()-start['started_epoch'])))
        setup_receipt.write_bytes(canonical(dict(attempts=attempts)))
    record_setup(store,json.loads(setup_receipt.read_text()),cap=spec['budget_seconds'])
    details=dict(campaign=sha(canonical(spec)),root=str(a.root),code=str(code),assets=str(assets),
        official=str(official),weights=str(weights),manifest=str(a.package/'manifest.json'),drive_folder=str(store.folder))
    (a.root/'deployment.json').write_bytes(canonical(details));store.put('deployment.json',canonical(details))
    print('CF_MULTIBACKBONE_DEPLOYED',json.dumps(details),flush=True)


if __name__=='__main__':
    main()
