#!/usr/bin/env python3
"""Package exact existing head assets plus pinned official CF, never peer edits."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def accepted_reference():
    data=subprocess.check_output(['git','-C',str(ROOT),'show',
        'origin/thesis-integration:knowledge/evidence/fulltest_n5_proximal_n8_20261112/complete_results_20261114.json'])
    if identity(data)!=dict(bytes=94668,sha256='199e52a5d826efaa39e159fcc98c441d95ac463951a6538dada3a450866bcc48'):
        raise ValueError('Accepted LION reference identity changed')
    return data


def package(base, official, output, accelerator='A100', batch_size=32, budget_hours=6):
    if accelerator not in ('A100','L4'):raise ValueError('Unregistered accelerator')
    from scripts.run_cloudfixer_multibackbone import CF_OPTIONS, HEADS, units
    units('registration','profile',batch_size)
    if type(budget_hours) is not int or budget_hours not in (6,8):raise ValueError('Unregistered budget')
    from scripts.prepare_cloudfixer_transfer import repair_import
    accepted = json.loads((ROOT/'knowledge/gsd_multibackbone_provenance_20261018.json').read_text())
    baseline = json.loads((base/'manifest.json').read_text())
    if identity((base/'manifest.json').read_bytes())['sha256']!='599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef':
        raise ValueError('Base manifest differs from accepted sensitivity receipt')
    assets = json.loads((ROOT/'cfgs/cloudfixer_transfer_assets.json').read_text())
    if subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip() != assets['author_commit']:
        raise ValueError('Wrong author source')
    if subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True):
        raise ValueError('Author clone must be clean')
    output.mkdir(parents=True, exist_ok=False)
    names = [n for n in baseline['files'] if n.startswith(('openpoints/','model_probe_sources/'))
             or n in ('pointnext_modelnet_cfg.yaml','pointnet2_batch_cuda.cpython-38-x86_64-linux-gnu.so')]
    checkpoints = {h: ('pointnext_' if h=='pointnext' else 'svwa_'+h+'_')+'modelnet40.pth' for h in HEADS}
    names += ['checkpoints/'+n for n in checkpoints.values()]
    files = {}
    with zipfile.ZipFile(output/'head_assets.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            p=base/name
            if p.is_symlink() or '..' in Path(name).parts or Path(name).is_absolute():
                raise ValueError('Unsafe asset')
            data=p.read_bytes(); actual=identity(data); expected=baseline['files'][name]
            if any(actual[k]!=expected[k] for k in ('bytes','sha256')):
                raise ValueError('Inherited asset identity mismatch: '+name)
            if name.startswith('checkpoints/') and any(actual[k]!=accepted['accepted'][p.name][k] for k in ('bytes','sha256')):
                raise ValueError('Accepted checkpoint mismatch')
            z.writestr(name,data);files[name]=actual
    official_tar=output/'official_cloudfixer.tar'
    subprocess.run(['git','-C',str(official),'archive','--format=tar','--output',str(official_tar),assets['author_commit']],check=True)
    author_files={}
    with tarfile.open(official_tar) as tar:
        for member in tar:
            if member.isfile():
                data=tar.extractfile(member).read()
                if member.name=='utils/misc.py':data=repair_import(data.decode()).encode()
                author_files[member.name]=identity(data)
    overlay=('scripts/run_cloudfixer_multibackbone.py','scripts/setup_cloudfixer_multibackbone_colab.py','scripts/eval_multibackbone_source.py',
        'scripts/run_cloudfixer_point2vec.py','scripts/run_cloudfixer_transfer.py',
        'classifier_backbones.py','research_artifacts.py','cloudfixer_store.py','cloudfixer_diffusion.py','pointcloud_proximal.py')
    overlay_files={n:identity((ROOT/n).read_bytes()) for n in overlay}
    full_bytes=accepted_reference()
    full=json.loads(full_bytes)
    (output/'baseline_reference.json').write_bytes(full_bytes)
    spec=dict(schema=1,task='cloudfixer_multibackbone_20261122',accelerator=accelerator,files=files,official_files=author_files,
        overlay_files=overlay_files,author_commit=assets['author_commit'],
        dataset_files=assets['dataset_files'],diffusion_sha256=assets['diffusion_sha256'],
        diffusion_drive_id=assets['diffusion_drive_id'],
        head_configs={h:dict(backbone=h,dataset='modelnet40_c',num_classes=40,input_layout='bnc',
            checkpoint='checkpoints/'+checkpoints[h],dgcnn_head_dataset_name='scanobject',pointnext_width=64) for h in HEADS},
        head_state_hashes={h:full['classifier_states']['modelnet40_c'][h] for h in HEADS},
        baseline_results=[r for r in full['overall'] if r['dataset']=='modelnet40_c' and r['backbone'] in HEADS],
        baseline_source=full['research_source'],source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        baseline_reference=identity(full_bytes),
        scientific_protocol=dict(dataset='modelnet40_c',objects=2468,severity=5,seed=0,batch_size=batch_size,
            CF_OPTIONS=CF_OPTIONS,heads=list(HEADS),geometry='authorrotateTrue;unitSTD;inverseaxis/unitsphereoutput',
            seed_policy=f'continuousfreshpercorruptionseed0;allRNGsavedper{batch_size};classifierfork;not author full-shuffledstream or common-draw paired with LION'),
        budget_seconds=budget_hours*3600)
    (output/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    setup_name='setup_cloudfixer_multibackbone_colab.py'
    (output/setup_name).write_bytes((ROOT/'scripts'/setup_name).read_bytes())
    with zipfile.ZipFile(output/'overlay.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in overlay:z.write(ROOT/name,name)
    receipt={n:identity((output/n).read_bytes()) for n in ('manifest.json','baseline_reference.json','official_cloudfixer.tar','head_assets.zip','overlay.zip',setup_name)}
    (output/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ('head_assets.zip','overlay.zip'):
        with zipfile.ZipFile(output/name) as z:
            if z.testzip() is not None:raise ValueError('Package CRC failed')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-assets',type=Path,required=True)
    p.add_argument('--official-checkout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accelerator',choices=('A100','L4'),default='A100')
    p.add_argument('--batch-size',type=int,choices=(32,64),default=32)
    p.add_argument('--budget-hours',type=int,choices=(6,8),default=6)
    a=p.parse_args();package(a.base_assets.resolve(),a.official_checkout.resolve(),a.output.resolve(),a.accelerator,a.batch_size,a.budget_hours)
