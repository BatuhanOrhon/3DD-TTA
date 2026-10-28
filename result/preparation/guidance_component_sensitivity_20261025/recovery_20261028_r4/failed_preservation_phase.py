"""CPU transport of the pinned failed confirmation attempt; never accepted results."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath
import re
import shutil

ROOT=Path('/content/studies/guidance-components')
DRIVE=Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
SESSION='tta-guidance-components-a100-20261028-r4'
PLAN_SHA='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
MANIFEST_SHA='599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA='73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
RECEIPT_ID=dict(bytes=3676445,sha256='5ab88805d14edbc9713275816aa11bef1d5e80784ea385b477199f7d6b779754')
LEDGER_ID=dict(bytes=12181,sha256='60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b')
ARCHIVE_ID=dict(bytes=1159246521,sha256='f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e')
MAX_PART=256*1024*1024


def require(condition,message):
    if not condition:raise ValueError(message)


def bytes_identity(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def remote_owned(value,root):
    path=PurePosixPath(value);root=PurePosixPath(root)
    require(isinstance(value,str) and '\\' not in value and path.as_posix()==value and path.is_absolute()
            and '..' not in path.parts and root in path.parents,'Foreign failed-attempt artifact path')
    return path


def validate_failed_state(receipt,ledger):
    state=receipt['state'];archive=receipt['archive']
    require(state['status']=='failed' and state['stage']=='confirmation' and state['session']==SESSION
            and state['phase_plan_sha256']==PLAN_SHA and state['manifest_sha256']==MANIFEST_SHA
            and state['source_snapshot_sha256']==SOURCE_SHA,'This transport only preserves the failed confirmation attempt')
    require(ledger['active'] is None and ledger['session']==SESSION and ledger['phase_plan_sha256']==PLAN_SHA
            and ledger['cap_seconds']==19800 and 155.86660684299568<=ledger['spent_seconds']<=19800,'Failed campaign is not inactive or has changed budget')
    attempts=[row for row in ledger['attempts'] if row['stage']=='confirmation']
    require(len(attempts)==1 and attempts[0]['status']=='failed' and archive['local_path']==attempts[0]['output']+'.zip','Original failed attempt binding changed')
    jobs=state['jobs']
    require(len(jobs)==21 and all(job['status']=='complete' for job in jobs[:20])
            and jobs[20]['status']=='running' and jobs[20]['condition_id']=='tuned-chamfer'
            and jobs[20]['dataset']=='modelnet40_c' and jobs[20]['seed']==2,'Expected20 complete jobs and one recorded incomplete job20')
    local=remote_owned(archive['local_path'],ROOT);drive=remote_owned(archive['drive_path'],DRIVE)
    require(local.parent==PurePosixPath(ROOT) and local.name.startswith('sensitivity_confirmation_') and local.suffix=='.zip'
            and drive==PurePosixPath(DRIVE)/'phase-evidence'/local.stem/'sensitivity_confirmation_evidence.zip'
            and type(archive['bytes']) is int and archive['bytes']>0 and re.fullmatch('[0-9a-f]{64}',archive['sha256']) is not None,'Original failed archive identity/path malformed')
    return archive


def validate_failed_manifest(manifest,receipt_raw,ledger_raw,plan_raw):
    require(bytes_identity(plan_raw)['sha256']==PLAN_SHA,'Pinned current plan changed')
    archive=validate_failed_state(json.loads(receipt_raw),json.loads(ledger_raw))
    require(manifest['status']=='failed_attempt_transport_ready' and manifest['experimental_result_status']=='failed'
            and manifest['schema_version']==1 and manifest['session']==SESSION and manifest['stage']=='confirmation'
            and manifest['phase_plan_sha256']==PLAN_SHA and manifest['original_archive']==archive
            and manifest['original_failed_receipt']==bytes_identity(receipt_raw) and manifest['original_failed_ledger']==bytes_identity(ledger_raw),
            'Failed artifact transport must not masquerade as complete/accepted confirmation')
    transport=remote_owned(manifest['transport_root'],ROOT);bound=manifest['chunk_bytes'];rows=manifest['parts']
    require(transport.parent==PurePosixPath(ROOT) and transport.name.startswith('sensitivity_confirmation_failed_transport_')
            and type(bound) is int and 1<=bound<=MAX_PART and type(manifest['part_count']) is int
            and manifest['part_count']==len(rows)==(archive['bytes']+bound-1)//bound,'Failed transport directory/part bound/count invalid')
    offset=0
    for index,row in enumerate(rows):
        name='part%05d.bin'%index
        require(type(row['index']) is int and row['index']==index and row['name']==name and row['remote_path']==str(transport/name)
                and remote_owned(row['remote_path'],ROOT).parent==transport and type(row['offset']) is int and row['offset']==offset
                and type(row['bytes']) is int and row['bytes']==min(bound,archive['bytes']-offset)
                and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'Failed part order/offset/path/identity mismatch')
        offset+=row['bytes']
    require(offset==archive['bytes'],'Failed transport does not cover original bytes')
    return archive


def load_failed_anchor():
    path=ROOT/'sensitivity_phases/sensitivity_campaign_controller.py'
    require(bytes_identity(path.read_bytes())['sha256']=='0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc','Pinned current controller changed')
    spec=importlib.util.spec_from_file_location('failed_preservation_control',path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    plan,_,_=c.plan_verified(ROOT/'sensitivity_campaign_plan.json',PLAN_SHA)
    receipt_raw=c.owned(ROOT/'sensitivity_confirmation_receipt.json').read_bytes()
    ledger_raw=c.owned(ROOT/'sensitivity_campaign_r4_ledger.json').read_bytes()
    require(bytes_identity(receipt_raw)==RECEIPT_ID and bytes_identity(ledger_raw)==LEDGER_ID,'Original failed receipt/ledger bytes changed; inspect before preservation')
    receipt=json.loads(receipt_raw);ledger=json.loads(ledger_raw)
    require(c.load_ledger(PLAN_SHA)==ledger,'Original inactive failed Drive ledger differs')
    archive=validate_failed_state(receipt,ledger)
    require({key:archive[key] for key in ('bytes','sha256')}==ARCHIVE_ID,'Pinned original failed raw ZIP changed')
    require(c.identity(ROOT/'sensitivity_confirmation_freeze.json')==dict(bytes=798,sha256='ea245f8a75ed729463e34757328875f6d7c789983ef10bac3c0c133b0750c360'), 'Original four-condition freeze changed')
    return c,plan,receipt,receipt_raw,ledger,ledger_raw


def main():
    c,plan,receipt,receipt_raw,ledger,ledger_raw=load_failed_anchor();archive=receipt['archive']
    path=c.owned(ROOT/'transport_helpers/split_stage_archive.py')
    require(bytes_identity(path.read_bytes())['sha256']=='d355092956650f71fd130d51bd2b3d6720f5b6c8200361f240f2939eb9362723','Unchanged bounded transport primitives required')
    spec=importlib.util.spec_from_file_location('bounded_transport_primitives',path)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    source=c.owned(archive['drive_path'],c.DRIVE);c.verify(c.owned(archive['local_path']),archive)
    require(source.stat().st_size==archive['bytes'] and shutil.disk_usage(ROOT).free>=archive['bytes']+MAX_PART,'Failed source size or VM disk bound changed')
    pointer=c.owned(ROOT/'sensitivity_confirmation_failed_transport_receipt.json')
    require(not pointer.exists(),'Failed transport pointer already exists; do not overwrite')
    output=c.owned(ROOT/('sensitivity_confirmation_failed_transport_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')));output.mkdir(exist_ok=False)
    try:
        records=h.split_archive(source,output,archive)
        require(c.load_ledger(PLAN_SHA)==ledger,'Original failed ledger changed during CPU-only preservation')
        manifest=dict(status='failed_attempt_transport_ready',experimental_result_status='failed',schema_version=1,stage='confirmation',session=SESSION,
                      phase_plan_sha256=PLAN_SHA,original_archive=archive,original_failed_receipt=bytes_identity(receipt_raw),original_failed_ledger=bytes_identity(ledger_raw),
                      completed_jobs=20,incomplete_job_index=20,recorded_incomplete_job_status='running',unrun_job_indices=[21,22,23],
                      transport_root=str(output),chunk_bytes=MAX_PART,part_count=len(records),parts=[dict(row,remote_path=str(output/row['name'])) for row in records])
        validate_failed_manifest(manifest,receipt_raw,ledger_raw,(ROOT/'sensitivity_campaign_plan.json').read_bytes())
        for name,raw in [('failed_phase_receipt.json',receipt_raw),('failed_campaign_ledger.json',ledger_raw)]:
            with (output/name).open('xb') as stream:stream.write(raw)
            require(h.identity(output/name)==bytes_identity(raw),'Preserved original failed metadata bytes differ')
        manifest_path=output/'failed_transport_manifest.json'
        with manifest_path.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
        result=dict(status='failed_attempt_transport_ready',experimental_result_status='failed',stage='confirmation',session=SESSION,phase_plan_sha256=PLAN_SHA,
                    original_archive=archive,transport_manifest=dict(h.identity(manifest_path),path=str(manifest_path)),part_count=len(records),
                    failed_receipt_copy=dict(RECEIPT_ID,path=str(output/'failed_phase_receipt.json')),failed_ledger_copy=dict(LEDGER_ID,path=str(output/'failed_campaign_ledger.json')))
        with pointer.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    except BaseException as error:
        with (output/'failure.json').open('x') as stream:json.dump(dict(status='failed_preservation_partial',experimental_result_status='failed',error=repr(error)),stream,indent=2)
        raise
    print('AGENT_B_FAILED_CONFIRMATION_TRANSPORT_RECEIPT',json.dumps(result),flush=True)
    print('AGENT_B_CONFIRMATION_FAILED_ARTIFACT_TRANSPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
