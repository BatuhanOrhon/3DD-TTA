"""Register label-free IDs and seal this task's overlay over the pinned base."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import prepare_chamfer_variants_assets as assets
from scripts.scdh_spectralz_study import register_ids

REPO=Path(__file__).resolve().parents[1]


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def register(prep,prior_roots):
    exclusions,exposure,full=set(),[],[]
    for root in prior_roots:
        for path in sorted((root/'result/preparation').rglob('*ids.json')):
            data=json.loads(path.read_text())
            if data.get('dataset')!='modelnet40_c':continue
            ids=data.get('indices',[])
            if not isinstance(ids,list) or any(type(i) is not int or not 0<=i<2468 for i in ids):
                raise ValueError('Unrecognized prior ID receipt: '+str(path))
            record=dict(path=str(path),identity=assets.identity(path),count=len(ids))
            if len(set(ids))==2468:full.append(record)
            else:exclusions.update(ids);exposure.append(record)
    selection=register_ids(exclusions)
    selection.update(prior_subset_receipts=exposure,prior_full_receipts=full)
    for stage in ('development','confirmation','calibration','technical','timing'):
        write_new(prep/(stage+'_ids.json'),dict(schema_version=1,dataset='modelnet40_c',n_examples=2468,
            stage='heldout' if stage=='confirmation' else 'development',indices=selection[stage]))
    selection['selected_manifest_identities']={s:assets.identity(prep/(s+'_ids.json')) for s in
        ('development','confirmation','calibration','technical','timing')}
    write_new(prep/'registration.json',selection)
    return selection


def seal(prep,base,base_sha):
    if assets.identity(base/'manifest.json')['sha256']!=base_sha:raise ValueError('Base identity mismatch')
    destination,delta=prep/'assets_v1',prep/'scdh_spectralz_v1_delta.zip'
    if destination.exists() or delta.exists():raise FileExistsError('Seal already exists; never replace it')
    manifest=json.loads((base/'manifest.json').read_text())
    source_names=sorted(set(assets.discover_sources(REPO,manifest))|{
        'result/preparation/chamfer_depth_tuning_20261030/'+n for n in ('operations.py','analyze.py','campaign.py')
    }|{'result/preparation/chamfer_depth_transfer_20261031/operator/deploy_assets_template.py'})
    input_identities={}
    metadata=['registration.json']+[s+'_ids.json' for s in ('development','confirmation','calibration','technical','timing')]
    registration=json.loads((prep/'registration.json').read_text())
    for stage,identity in registration['selected_manifest_identities'].items():
        if assets.identity(prep/(stage+'_ids.json'))!=identity:raise ValueError('Registration ID bytes changed')
    for name in source_names:
        path=REPO/name
        if path.suffix=='.py':ast.parse(path.read_text(),feature_version=(3,8))
        input_identities[name]=assets.identity(path)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain','--',*source_names],cwd=REPO,text=True)
    with tempfile.TemporaryDirectory(prefix='.seal-',dir=prep) as temporary:
        stage=Path(temporary);target=stage/'assets';target.mkdir()
        for name,identity in manifest['files'].items():
            if Path(name).parts[0] in assets.BULK or name=='source_snapshot.zip':continue
            assets._verify(base/name,identity)
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/name,path)
        for name in source_names:
            path=target/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(REPO/name,path)
            manifest['files']['source_snapshot/'+name]=assets._verify(path,input_identities[name])
        for name in metadata:
            path=target/'study'/name;path.parent.mkdir(exist_ok=True);shutil.copyfile(prep/name,path)
            manifest['files']['study/'+name]=assets.identity(path)
        archive=target/'source_snapshot.zip'
        assets._archive(archive,target,[n for n in manifest['files'] if n!='source_snapshot.zip' and Path(n).parts[0] not in assets.BULK])
        manifest['files']['source_snapshot.zip']=assets.identity(archive)
        manifest.update(manifest_revision='scdh_spectralz_v1',supersedes_manifest_sha256=base_sha,
            source_commit=commit,source_dirty=bool(status.strip()),source_snapshot_archive='source_snapshot.zip',
            source_snapshot_sha256=assets.identity(archive)['sha256'],gpu_inference_executed=False,
            revision_reason='Registered SCD-h / offdiag truncated smooth spectral-z finite study; no default promotion')
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        assets._archive(stage/delta.name,stage,['assets/manifest.json','assets/source_snapshot.zip'])
        for name,identity in input_identities.items():assets._verify(REPO/name,identity)
        receipt=dict(status='sealed_CPU_only',base_manifest_sha256=base_sha,source_commit=commit,
            source_dirty=bool(status.strip()),source_git_status=status,source_files=input_identities,
            manifest=assets.identity(target/'manifest.json'),source_snapshot=assets.identity(archive),payload=assets.identity(stage/delta.name))
        target.rename(destination);(stage/delta.name).rename(delta)
        write_new(prep/'preparation_identity.json',receipt)
    return receipt


def make_plan(prep,cpu_receipt):
    receipt=json.loads((prep/'preparation_identity.json').read_text())
    cpu=json.loads(cpu_receipt.read_text())
    if (cpu.get('status')!='complete' or cpu.get('manifest_sha256')!=receipt['manifest']['sha256']
            or cpu.get('source_snapshot_sha256')!=receipt['source_snapshot']['sha256']):
        raise ValueError('Deployment receipt differs from sealed preparation')
    root='/content/studies/scdh-spectralz';asset=root+'/prepared/assets'
    plan=dict(schema_version=1,scope='technical+calibration+A+B+conditional-C+timing',
        root=root+'/campaign',asset_root=asset,source_commit=receipt['source_commit'],
        asset_manifest_sha256=receipt['manifest']['sha256'],source_snapshot_sha256=receipt['source_snapshot']['sha256'],
        python='/usr/local/envs/3dd_tta_env/bin/python',analysis_python='/usr/local/envs/3dd_tta_env/bin/python',
        drive_root='/content/drive/MyDrive/thesis/result/scdh_spectralz_search_20261101',
        dataset_root='/content/3DD-TTA/data/modelnet40_c',label_path='/content/3DD-TTA/data/modelnet40_c/label.npy',
        pointmae_config=asset+'/source_snapshot/cfgs/tta_modelnet.yaml',pointmae_checkpoint=asset+'/checkpoints/modelnet_jt.pth',
        cpu_preflight_receipt=root+'/cpu_preflight_receipt.json',cpu_preflight_receipt_identity=assets.identity(cpu_receipt),
        cumulative_wall_cap_seconds=14400,archive_reserve_seconds=180,minimum_job_reserve_seconds=600,
        job_timeout_seconds=900,max_development_jobs=30,max_confirmation_jobs=12,
        registration_identity=assets.identity(prep/'registration.json'),
        paired_groups='separate technical/calibration/development/confirmation groups; same physical preparation across matched arms within each stage',
        timing='one separate warmup +three fresh32-object Gaussian/Background repeats per arm;384measured+128warmup events')
    write_new(prep/'plan.json',plan)
    from scripts.scdh_spectralz_study import canonical_sha
    write_new(prep/'approval_pending.json',dict(approved=False,scope=plan['scope'],plan_sha256=canonical_sha(plan),user_message='',
        note='Drive consent is not scientific GPU approval; replace only after explicit whole-phase user approval'))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','seal','plan'))
    parser.add_argument('--prep',type=Path,required=True)
    parser.add_argument('--prior-root',type=Path,action='append',default=[])
    parser.add_argument('--base',type=Path)
    parser.add_argument('--base-sha',default=assets.BASE_SHA)
    parser.add_argument('--cpu-receipt',type=Path)
    args=parser.parse_args()
    value=(register(args.prep,args.prior_root) if args.action=='register' else
           seal(args.prep,args.base,args.base_sha) if args.action=='seal' else make_plan(args.prep,args.cpu_receipt))
    print(json.dumps(value,indent=2,sort_keys=True))

if __name__=='__main__':main()
