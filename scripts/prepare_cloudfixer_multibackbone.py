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


def original_cloudfixer_reference():
    from scripts.run_cloudfixer_multibackbone import ORIGINAL_CF_REFERENCE, ORIGINAL_CF_ARCHIVE_INDEX

    directory = ROOT/'knowledge/evidence/cloudfixer_multibackbone_20261122'
    recount = (directory/'a100_full_recount.json').read_bytes()
    archive_index = (directory/'a100_original_archive_index.json').read_bytes()
    if identity(recount) != ORIGINAL_CF_REFERENCE or identity(archive_index) != ORIGINAL_CF_ARCHIVE_INDEX:
        raise ValueError('Accepted original CloudFixer reference identity changed')
    return recount, archive_index


def package(base, official, output, accelerator='A100', batch_size=32, budget_hours=6, *,
            profile='original_cloudfixer'):
    if accelerator not in ('A100','L4'):raise ValueError('Unregistered accelerator')
    from scripts.run_cloudfixer_multibackbone import (
        BASELINE_OVERLAY_FILES, BASELINE_POLICY, BASELINE_PROFILE, BASELINE_TASK, CF_OPTIONS, HEADS,
        ORIGINAL_CF_SOURCE, OVERLAY_FILES, baseline_protocol, manifest_profile, validate_batch_size,
    )
    validate_batch_size(batch_size, profile)
    is_baseline = profile == BASELINE_PROFILE
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
    overlay = BASELINE_OVERLAY_FILES if is_baseline else OVERLAY_FILES
    reference_names = ['baseline_reference.json']
    if is_baseline:
        if subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD','--',*overlay],text=True).strip():
            raise ValueError('Commit baseline overlay source before packaging its identity')
        reference, archive_index = original_cloudfixer_reference()
        reference_names.append('original_archive_index.json')
        (output/'original_archive_index.json').write_bytes(archive_index)
        full = json.loads(reference)
        reference_results = [r for r in full['summary'] if r['method']=='cloudfixer_plain']
        reference_source = ORIGINAL_CF_SOURCE
        head_states = {r['backbone']: r['state_sha256'] for r in accepted['preparation']['cpu_preflight_models']
                       if r['dataset']=='modelnet40_c' and r['backbone'] in HEADS}
    else:
        reference = accepted_reference()
        legacy_reference = json.loads(reference)
        head_states = legacy_reference['classifier_states']['modelnet40_c']
        reference_results = [r for r in legacy_reference['overall'] if r['dataset']=='modelnet40_c' and r['backbone'] in HEADS]
        reference_source = legacy_reference['research_source']
    overlay_files={n:identity((ROOT/n).read_bytes()) for n in overlay}
    (output/'baseline_reference.json').write_bytes(reference)
    spec=dict(schema=1,task='cloudfixer_multibackbone_20261122',accelerator=accelerator,files=files,official_files=author_files,
        overlay_files=overlay_files,author_commit=assets['author_commit'],
        dataset_files=assets['dataset_files'],diffusion_sha256=assets['diffusion_sha256'],
        diffusion_drive_id=assets['diffusion_drive_id'],
        head_configs={h:dict(backbone=h,dataset='modelnet40_c',num_classes=40,input_layout='bnc',
            checkpoint='checkpoints/'+checkpoints[h],dgcnn_head_dataset_name='scanobject',pointnext_width=64) for h in HEADS},
        head_state_hashes={h:head_states[h] for h in HEADS},
        baseline_results=reference_results, baseline_source=reference_source,
        source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        baseline_reference=identity(reference),
        scientific_protocol=dict(dataset='modelnet40_c',objects=2468,severity=5,seed=0,batch_size=batch_size,
            CF_OPTIONS=CF_OPTIONS,heads=list(HEADS),geometry='authorrotateTrue;unitSTD;inverseaxis/unitsphereoutput',
            seed_policy=f'continuousfreshpercorruptionseed0;allRNGsavedper{batch_size};classifierfork;not author full-shuffledstream or common-draw paired with LION'),
        budget_seconds=budget_hours*3600)
    if is_baseline:
        spec.update(task=BASELINE_TASK, profile=BASELINE_PROFILE, policy=BASELINE_POLICY,
                    scientific_protocol=baseline_protocol(batch_size), baseline_archive_index=identity(archive_index))
    manifest_profile(spec)
    (output/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    setup_name='setup_cloudfixer_multibackbone_colab.py'
    (output/setup_name).write_bytes((ROOT/'scripts'/setup_name).read_bytes())
    with zipfile.ZipFile(output/'overlay.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in overlay:z.write(ROOT/name,name)
    receipt={n:identity((output/n).read_bytes()) for n in ('manifest.json',*reference_names,'official_cloudfixer.tar','head_assets.zip','overlay.zip',setup_name)}
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
    from scripts.run_cloudfixer_multibackbone import (
        BASELINE_BATCH_SIZES, BASELINE_PROFILE, ORIGINAL_BATCH_SIZES, ORIGINAL_PROFILE,
    )
    p.add_argument('--batch-size',type=int,choices=sorted(set(ORIGINAL_BATCH_SIZES + BASELINE_BATCH_SIZES)),default=32,
                   help='Registered batch size for the selected profile')
    p.add_argument('--budget-hours',type=int,choices=(6,8),default=6)
    p.add_argument('--profile',choices=(ORIGINAL_PROFILE,BASELINE_PROFILE),default=ORIGINAL_PROFILE)
    a=p.parse_args();package(a.base_assets.resolve(),a.official_checkout.resolve(),a.output.resolve(),a.accelerator,
                             a.batch_size,a.budget_hours,profile=a.profile)
