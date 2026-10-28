"""CPU preservation of180 cache files and failure proof, never final acceptance."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import zipfile

ROOT=Path('/content/studies/guidance-components')
FAILED_HELPER_SHA='a4e712716ffb260c64d8bfdf41b5092445176ae15e21fb4b972d684477d602ff'
DATASETS=('modelnet40_c','scanobjectnn_c')
CORRUPTIONS=('uniform','gaussian','background','impulse','upsampling','distortion_rbf','distortion_rbf_inv',
             'density','density_inc','shear','rotation','cutout','distortion','occlusion','lidar')
SIGNATURE='ValueError: Classifier input contains nonfinite XYZ coordinates'


def require(condition,message):
    if not condition:raise ValueError(message)


def completed_cache_records(state,ids):
    require(state['status']=='failed' and len(state['jobs'])==21 and state['jobs'][20]['status']=='running','Expected original incomplete confirmation state')
    complete=[job for job in state['jobs'] if job['status']=='complete']
    require(len(complete)==20,'Expected20 archived completed workers')
    records={}
    for job in complete:
        rows=job['cache_records']
        if job['trial']['method']=='original3dd':
            require(not rows,'Original3DD unexpectedly used paired cache');continue
        require(job['trial']['method']=='gsd' and len(rows)==30 and len({row['local_path'] for row in rows})==30,'Complete GSD dataset worker requires30 preparations')
        for row in rows:
            path=Path(row['local_path'])
            require(path.parts[-4]=='paired_sensitivity_confirmation_v2_seed%d'%job['seed'] and path.parts[-3]==job['dataset'], 'Cache belongs to a different producer seed/dataset')
            require(row['local_path'] not in records or records[row['local_path']]==row,'Completed producer cache identities disagree')
            records[row['local_path']]=row
    expected=set()
    for dataset in DATASETS:
        selected=ids[dataset]
        require(len(selected)==140 and selected==sorted(set(selected)),'Expected frozen140 IDs/dataset')
        for seed in (0,1,2):
            for corruption in CORRUPTIONS:
                for batch in (0,1):
                    name=hashlib.sha256(struct.pack('<70q',*selected[70*batch:70*(batch+1)])).hexdigest()+'.npz'
                    expected.add(str(ROOT/('paired_sensitivity_confirmation_v2_seed%d'%seed)/dataset/corruption/name))
    require(len(records)==180 and set(records)==expected,'Require all180 unique physical preparation identities from completed workers')
    for job in complete:
        if job['trial']['method']=='gsd':
            for trace in job['traces'].values():
                cached=trace['paired_state'];row=records.get(cached['path'])
                require(row is not None and cached==dict(path=row['local_path'],bytes=row['bytes'],sha256=row['sha256']),'Archived initial-state trace/cache identity differs')
    return [records[key] for key in sorted(records)]


def validate_cache_metadata(path,initial,legacy):
    import numpy as np
    import torch
    require(np.__version__.split('.')[0]=='1' and not torch.cuda.is_initialized(),'CPU cache verification requires NumPy1 and CUDA-uninitialized torch')
    before=legacy.rng_hash()
    with np.load(path,allow_pickle=False) as cached:
        metadata=json.loads(str(cached['metadata'].item()))
        require(metadata['schema_version']==1 and metadata['cuda_devices']==1
                and metadata['input_sha256']==initial['input_points'],'Cache input/schema binding changed')
        require(set(cached.files)=={'metadata','numpy_rng_keys','torch_rng_cpu','torch_rng_cuda_0','shape_latent','local_latent','noise'}
                and set(metadata['tensor_hashes'])=={'shape_latent','local_latent','noise'},'Cache tensor/RNG schema changed')
        shapes={}
        for key in ('shape_latent','local_latent','noise'):
            array=cached[key].copy();shapes[key]=array.shape
            require(array.dtype==np.float32 and array.ndim==4 and array.shape[0]==70 and array.shape[1]>0
                    and array.shape[2:]==(1,1) and np.isfinite(array).all(),'Invalid cached latent/noise tensor')
            require(legacy.tensor_hash(torch.from_numpy(array))==metadata['tensor_hashes'][key]==initial[key],'Cached tensor hash differs from completed producer trace')
        require(shapes['local_latent']==shapes['noise']==(70,8192,1,1),'Local latent/noise dimensions changed')
        py=metadata['python_rng'];nr=metadata['numpy_rng']
        rng=((py[0],tuple(py[1]),py[2]),(nr[0],cached['numpy_rng_keys'].copy(),*nr[1:]),
             torch.from_numpy(cached['torch_rng_cpu'].copy()),[torch.from_numpy(cached['torch_rng_cuda_0'].copy())])
        require(legacy._rng_state_hash(rng)==metadata['rng_after_sha256'],'Exact producer post-preparation RNG hash differs')
    require(not torch.cuda.is_initialized() and legacy.rng_hash()==before,'CPU cache verification changed process RNG or initialized CUDA')
    return dict(path=str(path),status='tensor_input_post_rng_verified_CPU',input_sha256=metadata['input_sha256'],
                tensor_hashes=metadata['tensor_hashes'],rng_before_sha256=metadata['rng_before_sha256'],rng_after_sha256=metadata['rng_after_sha256'])


def main():
    helper=ROOT/'transport_helpers/failed_preservation_phase.py'
    require(hashlib.sha256(helper.read_bytes()).hexdigest()==FAILED_HELPER_SHA,'Pinned failed-preservation anchor changed')
    spec=importlib.util.spec_from_file_location('failed_evidence_anchor',helper)
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    c,plan,receipt,receipt_raw,ledger,ledger_raw=f.load_failed_anchor()
    original=receipt['archive']
    c.verify(c.owned(original['local_path']),original);c.verify(c.owned(original['drive_path'],c.DRIVE),original)
    legacy=ROOT/'phases/deploy_assets_template.py'
    require(c.identity(legacy)['sha256']=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0','Pinned immutable-copy evidence helper changed')
    sys.path.insert(0,str(legacy.parent))
    from deploy_assets_template import copy_checked,seal
    freeze=c.owned(ROOT/'sensitivity_confirmation_freeze.json');development=c.owned(ROOT/'accepted_sensitivity_development.json')
    require(c.identity(development)==dict(bytes=1053,sha256='f94afc837e26a3ec33c8503f04cc9cfb7dafba44fd40cc19cff845eb7bad54ed'),'Original frozen development acceptance changed')
    source_log=c.owned(Path(ledger['attempts'][-1]['output'])/'job_020.log');log_raw=source_log.read_bytes()
    require(SIGNATURE.encode() in log_raw,'Expected recorded numeric-failure signature missing; preserve and inspect without classifying')
    with zipfile.ZipFile(original['drive_path']) as archive:
        require(json.loads(archive.read('state.json'))==receipt['state'] and archive.read('job_020.log')==log_raw
                and archive.read('campaign_plan.json')==(ROOT/'sensitivity_campaign_plan.json').read_bytes()
                and archive.read(freeze.name)==freeze.read_bytes() and archive.read(development.name)==development.read_bytes(),'Original failed archive state/log/freeze/development bytes changed')
    _,_,selections=c.plan_verified(ROOT/'sensitivity_campaign_plan.json',f.PLAN_SHA)
    ids={dataset:selections[('confirmation',dataset)]['indices'] for dataset in DATASETS}
    records=completed_cache_records(receipt['state'],ids)
    complete=receipt['state']['jobs'][:20];c.pairing_verified(complete)
    for job in complete:
        require(job['lion_state']==ledger['lion_state'],'Completed worker LION contract differs from original ledger')
        for head,contract in job['classifier_contracts'].items():
            require(contract==ledger['classifier_contracts'][job['dataset']+'/'+head],'Completed worker frozen classifier contract differs')
    cpu=c.cpu_preflight_verified(plan)
    assets=Path(plan['asset_root']);manifest=json.loads((assets/'manifest.json').read_text())
    require(c.identity(assets/'manifest.json')['sha256']==plan['manifest_sha256'] and c.identity(assets/'source_snapshot.zip')['sha256']==plan['source_snapshot_sha256'],'Pinned source assets changed')
    legacy_path=assets/'source_snapshot/legacy_trajectory.py';c.verify(legacy_path,manifest['files']['source_snapshot/legacy_trajectory.py'])
    import torch,numpy as np
    require(sys.version_info[:2]==(3,8) and torch.__version__=='2.1.2+cu121' and np.__version__.split('.')[0]=='1'
            and not torch.cuda.is_initialized(),'Use the pinned3dd_tta_env CPU process')
    legacy_spec=importlib.util.spec_from_file_location('preserved_legacy_rng',legacy_path)
    legacy_module=importlib.util.module_from_spec(legacy_spec);legacy_spec.loader.exec_module(legacy_module)
    initial_by_path={}
    for job in complete:
        if job['trial']['method']=='gsd':
            for trace in job['traces'].values():
                name=trace['paired_state']['path'];initial=trace['initial_hashes']
                require(name not in initial_by_path or initial_by_path[name]==initial,'Completed producer initial tensors disagree')
                initial_by_path[name]=initial
    require(len(initial_by_path)==180,'All180 producer tensor/RNG bindings required')
    metadata_proofs=[]
    pointer=c.owned(ROOT/'sensitivity_confirmation_failure_preservation_receipt.json')
    require(not pointer.exists(),'Failure-preservation receipt exists; inspect instead of overwriting')
    require(shutil.disk_usage(ROOT).free>=2*sum(row['bytes'] for row in records)+256*1024**2,'Insufficient disk for immutable cache proof/archive')
    output=c.owned(ROOT/('sensitivity_confirmation_failure_proof_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')));output.mkdir(exist_ok=False)
    for number,row in enumerate(records,1):
        path=c.owned(row['local_path']);relative=path.relative_to(ROOT);drive=c.owned(row['drive_path'],c.DRIVE)
        require(drive.relative_to(c.DRIVE)==relative,'Source/Drive cache namespaces differ')
        c.verify(path,row);c.verify(drive,row)
        metadata_proofs.append(validate_cache_metadata(path,initial_by_path[str(path)],legacy_module))
        copy_checked(path,output/relative,row)
        if number%30==0:print('AGENT_B_FAILED_CACHE_PRESERVATION_PROGRESS',number,180,flush=True)
    for name,raw in [('failed_phase_receipt.json',receipt_raw),('failed_campaign_ledger.json',ledger_raw),('job_020.log',log_raw),
                     (freeze.name,freeze.read_bytes()),(development.name,development.read_bytes()),('campaign_plan.json',(ROOT/'sensitivity_campaign_plan.json').read_bytes())]:
        with (output/name).open('xb') as stream:stream.write(raw)
    (output/'cache_records.json').write_text(json.dumps(records,indent=2)+'\n')
    require(len(metadata_proofs)==180 and not torch.cuda.is_initialized(),'Physical cache metadata/RNG verification incomplete')
    (output/'cache_metadata_validation.json').write_text(json.dumps(metadata_proofs,indent=2)+'\n')
    (output/'fresh_cpu_preflight_receipt.json').write_text(json.dumps(cpu,indent=2)+'\n')
    integrity=dict(status='verified_preserved_scope',scope='20 completed jobs and180 physical preparation caches; incomplete job20 final state is not validated',
                   classifier_and_lion_contracts_match_original_ledger=True,paired_preparation_trace_bindings_match=True,
                   current_plan_and_CPU_proof_match=True,physical_source_and_Drive_cache_identities_match=True,cache_input_tensor_post_RNG_hashes_verified=180,
                   CPU_only=True,cuda_initialized=False,numpy_version=np.__version__,torch_version=torch.__version__,incomplete_job20_final_state_validated=False)
    state=dict(status='failed_attempt_cache_bytes_preserved',experimental_result_status='failed',session=f.SESSION,phase_plan_sha256=f.PLAN_SHA,
               manifest_sha256=f.MANIFEST_SHA,source_snapshot_sha256=f.SOURCE_SHA,original_confirmation_archive=original,
               original_receipt_identity=f.RECEIPT_ID,failed_ledger_identity=f.LEDGER_ID,failed_job_index=20,completed_jobs=20,unrun_job_indices=[21,22,23],
               physical_cache_count=180,per_seed_cache_count={seed:60 for seed in (0,1,2)},integrity_verification=integrity)
    (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    require(c.load_ledger(f.PLAN_SHA)==ledger,'Original failed ledger changed during CPU preservation')
    print('AGENT_B_FAILED_CACHE_PROOF_SEALING',180,flush=True)
    archived=seal(output,c.DRIVE/'phase-evidence'/output.name/'failed_confirmation_cache_evidence.zip')
    result=dict(status='verified_failure_preserved',experimental_result_status='failed',session=f.SESSION,phase_plan_sha256=f.PLAN_SHA,
                manifest_sha256=f.MANIFEST_SHA,source_snapshot_sha256=f.SOURCE_SHA,original_confirmation_archive=original,
                original_receipt_identity=f.RECEIPT_ID,failed_ledger_identity=f.LEDGER_ID,failed_job_index=20,numeric_failure_signature=SIGNATURE,
                physical_cache_count=180,local_and_drive_original_archive_verified=True,physical_cache_complete=True,mismatch_detected=False,
                integrity_scope=integrity['scope'],integrity_verification=integrity,archive=archived,
                source_log_identity=dict(f.bytes_identity(log_raw),local_path=str(source_log)))
    with pointer.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    (Path(archived['drive_path']).parent/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AGENT_B_FAILED_CONFIRMATION_PRESERVATION_RECEIPT',json.dumps(result),flush=True)
    print('AGENT_B_CONFIRMATION_FAILURE_PRESERVATION_COMPLETE',flush=True)


if __name__=='__main__':main()
