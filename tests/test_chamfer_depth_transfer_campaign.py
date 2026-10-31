"""Controller contracts use synthetic CPU workers; no native/GPU inference."""
import copy
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from scripts import run_chamfer_depth_transfer as campaign
from scripts import eval_chamfer_depth as depth

pytestmark = pytest.mark.skipif(campaign.analysis.np.__version__.split('.')[0] != '1',
                                reason='Physical cache acceptance runs in the separate NumPy1 CPU environment')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    sealed_repo=tmp_path/'assets/source_snapshot'
    for name in campaign.PHASE_SOURCES:
        target=sealed_repo/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(campaign.REPO/name,target)
    monkeypatch.setattr(campaign,'REPO',sealed_repo)
    root=tmp_path/'campaign';root.mkdir()
    drive=tmp_path/'drive';drive.mkdir()
    source='a'*64;assets='b'*64
    cpu=dict(status='complete',report=dict(status='complete',cuda_initialized=False,forward_executed=False,
        source_snapshot_sha256=source,manifest_sha256=assets,heads={}))
    jobs=[]
    for stage,dataset,seed in [('smoke',d,0) for d in campaign.DATASETS]+[
            ('research',d,s) for d in campaign.DATASETS for s in (0,1,2)]:
        slot_id='%s-%s-%d'%(stage,dataset,seed)
        ids=list(range(32)) if stage=='smoke' else [1,3]
        selection=root/(slot_id+'-ids.json')
        selection.write_text(json.dumps(dict(dataset=dataset,n_examples=581 if dataset=='scanobjectnn_c' else 2468,
                                            stage='smoke' if stage=='smoke' else 'heldout',indices=ids)))
        command=[sys.executable,'-u',str(campaign.REPO/'scripts/eval_chamfer_depth.py'),
            '--transfer-study','--dataset',dataset,'--dataset-root',str(tmp_path/'data'),
            '--label-path',str(tmp_path/'data/label.npy'),'--asset-root',str(tmp_path/'assets'),
            '--asset-manifest-sha256',assets,'--result-root',str(root/'slots'/slot_id/'raw'),
            '--index-manifest',str(selection),'--paired-state-root',str(root/('paired-smoke' if stage=='smoke' else 'paired')/('seed%d'%seed)),
            '--pointmae-config','/assets/config.yaml','--pointmae-checkpoint','/assets/model.pth',
            '--drive-root',str(drive),'--seed',str(seed)]
        if stage=='smoke':command+=['--technical-smoke','--corruptions','gaussian','background']
        configs=depth.configurations(command[3:])
        bindings={}
        for cfg in configs:
            checkpoint=dict(strict=True,checkpoint_sha256=str(campaign.HEADS.index(cfg['backbone']))*64,
                            missing_keys=[],unexpected_keys=[])
            provenance=dict(n_examples=cfg['n_examples'],files={'label.npy':{'bytes':1,'sha256':'c'*64}})
            expected={k:cfg[k] for k in ('dataset','backbone','seed','selected_original_ids')}
            expected.update(source_snapshot_sha256=source,asset_manifest_sha256=assets,
                            checkpoint_load=checkpoint,dataset_provenance=provenance)
            required={'repo/'+name:campaign.operations.file_identity(campaign.REPO/name) for name in campaign.analysis.CRITICAL_SOURCES}
            bindings[cfg['backbone']]=dict(expected_config=expected,required_source_files=required)
            cpu['report']['heads'].setdefault(dataset,{})[cfg['backbone']]=dict(checkpoint_load=checkpoint,dataset_provenance=provenance)
        jobs.append(dict(slot_id=slot_id,stage=stage,dataset=dataset,seed=seed,command=command,
                         cwd=str(campaign.REPO),bindings=bindings))
    archive=tmp_path/'cpu.zip';archive.write_bytes(b'CPU fixture archive')
    remote=drive/'cpu.zip';remote.write_bytes(archive.read_bytes())
    cpu['archive']=dict(local_path=str(archive),drive_path=str(remote),**campaign.operations.file_identity(archive))
    receipt=tmp_path/'cpu.json';receipt.write_text(json.dumps(cpu))
    plan=dict(schema_version=1,source_snapshot_sha256=source,asset_manifest_sha256=assets,
              cumulative_wall_cap_seconds=28800,drive_root=str(drive),jobs=jobs,
              final_archive_reserve_seconds=60,per_job_acceptance_archive_reserve_seconds=180,
              phase_sources={name:campaign.operations.file_identity(campaign.REPO/name) for name in campaign.PHASE_SOURCES},
              cpu_preflight_receipt_path=str(receipt),cpu_preflight_receipt_identity=campaign.operations.file_identity(receipt))
    plan_path=tmp_path/'plan.json';approval_path=tmp_path/'approval.json'
    def seal():
        plan_path.write_text(json.dumps(plan))
        approval_path.write_text(json.dumps(dict(approved=True,plan_sha256=campaign.operations.file_identity(plan_path)['sha256'],
             source_snapshot_sha256=source,cumulative_wall_cap_seconds=plan['cumulative_wall_cap_seconds'])))
    seal()
    monkeypatch.setattr(campaign.os.path,'ismount',lambda p:True)
    reports={};attempts=[]
    def worker(command,cwd,console,timeout,slot_id):
        attempts.append(slot_id)
        job=next(j for j in plan['jobs'] if j['slot_id']==slot_id)
        slot=root/'slots'/slot_id
        Path(console).write_text('CPU synthetic worker\n')
        for head,binding in job['bindings'].items():
            raw=slot/'raw'/head;raw.mkdir(parents=True)
            (raw/'bundle_manifest.json').write_text('{}')
            (raw/'config.json').write_text(json.dumps(dict(backbone=head)))
            target=Path(str(raw)+'.zip')
            with zipfile.ZipFile(target,'w') as stream:stream.writestr('fixture.txt','sealed CPU fixture')
            relative=Path('paired-smoke' if job['stage']=='smoke' else 'paired')/('seed%d'%job['seed'])/job['dataset']/'gaussian/cache.npz'
            physical=root/relative;physical.parent.mkdir(parents=True,exist_ok=True);physical.write_bytes(b'physical CPU fixture')
            cache=dict(relative_path=relative.as_posix(),**campaign.operations.file_identity(physical))
            reports[str(target)]=dict(status='accepted',dataset=job['dataset'],backbone=head,seed=job['seed'],
                technical_smoke=job['stage']=='smoke',object_ids=binding['expected_config']['selected_original_ids'],
                labels=[0]*len(binding['expected_config']['selected_original_ids']),corruptions=['gaussian','background'] if job['stage']=='smoke' else list(campaign.analysis.CORRUPTIONS),
                reconstruction_group_id=slot_id,source_snapshot_sha256=source,asset_manifest_sha256=assets,
                batch_guards=[dict(cache=cache,geometry_sha256='d'*64,initial_rng_sha256='e'*64)],cache_proofs=[cache],
                bundle_identity=campaign.operations.file_identity(target),sealed_manifest_identity={'bytes':2,'sha256':'f'*64},
                aggregate=[dict(policy='probability_mean',accuracy_pct=50.)])
        return 0
    def analyze(path,binding,cache_root):return copy.deepcopy(reports[str(path)])
    monkeypatch.setattr(campaign.operations,'run_worker',worker)
    monkeypatch.setattr(campaign.analysis,'analyze_bundle',analyze)
    return dict(root=root,drive=drive,plan=plan,plan_path=plan_path,approval_path=approval_path,seal=seal,
                reports=reports,attempts=attempts,worker=worker,receipt=receipt)


def launch(p):return campaign.run(p['plan_path'],p['approval_path'],p['root'])


def test_all_eight_jobs_accept_and_persist_cache_and_latest_ledger(prepared,capsys):
    p=prepared
    launch(p)
    assert len(p['attempts'])==8
    assert capsys.readouterr().out.count('DEPTH_TRANSFER_CAMPAIGN_COMPLETE')==1
    latest=json.loads((p['root']/'registered_latest_ledger.json').read_text())
    assert latest['status']=='complete' and latest['actual_elapsed_seconds']>=0
    assert len(list((p['drive']/'campaign').rglob('cache.npz')))==8
    assert any('acceptance.json' in path.name for path in (p['drive']/'campaign').rglob('*.json'))
    with pytest.raises((FileExistsError,ValueError)):launch(p)
    assert len(p['attempts'])==8


@pytest.mark.parametrize('field,value',[('approved',False),('plan_sha256','f'*64),('source_snapshot_sha256','f'*64),('cumulative_wall_cap_seconds',1)])
def test_mismatched_approval_stops_before_worker(prepared,field,value):
    p=prepared;approval=json.loads(p['approval_path'].read_text());approval[field]=value
    p['approval_path'].write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='approval|Approval'):launch(p)
    assert not p['attempts']
    assert (p['root']/'failure.json').is_file()


def test_duplicate_registered_job_is_rejected(prepared):
    p=prepared;p['plan']['jobs'][-1]=copy.deepcopy(p['plan']['jobs'][-2]);p['seal']()
    with pytest.raises(ValueError,match='eight|Eight|Duplicate|duplicate'):launch(p)
    assert not p['attempts']


def test_alternate_asset_tree_with_same_manifest_sha_is_rejected(prepared):
    p=prepared
    command=p['plan']['jobs'][0]['command']
    command[command.index('--asset-root')+1]=str(p['root']/'unrelated-assets')
    p['seal']()
    with pytest.raises(ValueError,match='asset|snapshot|checkout'):launch(p)
    assert not p['attempts']


def test_asset_snapshot_symlink_cannot_relabel_an_external_checkout(prepared):
    p=prepared;alternate=p['root']/'linked-assets';alternate.mkdir()
    (alternate/'source_snapshot').symlink_to(campaign.REPO,target_is_directory=True)
    command=p['plan']['jobs'][0]['command'];command[command.index('--asset-root')+1]=str(alternate)
    p['seal']()
    with pytest.raises(ValueError,match='asset|snapshot|checkout'):launch(p)
    assert not p['attempts']


@pytest.mark.parametrize('changes',[{'cumulative_wall_cap_seconds':0},{'cumulative_wall_cap_seconds':28801},
    {'final_archive_reserve_seconds':-1},{'per_job_acceptance_archive_reserve_seconds':0},
    {'final_archive_reserve_seconds':28800},{'final_archive_reserve_seconds':1.5}])
def test_invalid_cap_or_reserve_fails_before_worker(prepared,changes):
    p=prepared;p['plan'].update(changes);p['seal']()
    with pytest.raises(ValueError,match='cap|reserve|budget'):launch(p)
    assert not p['attempts']


def test_lower_approved_cap_and_explicit_reserves_are_supported(prepared,capsys):
    p=prepared;p['plan'].update(cumulative_wall_cap_seconds=2000,
        final_archive_reserve_seconds=300,per_job_acceptance_archive_reserve_seconds=180);p['seal']()
    launch(p)
    latest=json.loads((p['root']/'registered_latest_ledger.json').read_text())
    assert latest['cumulative_wall_cap_seconds']==2000 and latest['status']=='complete'
    assert latest['charged_seconds']==latest['actual_elapsed_seconds']
    assert len(p['attempts'])==8 and 'DEPTH_TRANSFER_CAMPAIGN_COMPLETE' in capsys.readouterr().out


@pytest.mark.parametrize('failure',['exit','missing_head','missing_zip','cross_head_cache'])
def test_failed_second_smoke_blocks_research_and_preserves_partial_raw(prepared,monkeypatch,failure,capsys):
    p=prepared
    def worker(*args):
        result=p['worker'](*args)
        if len(p['attempts'])==2:
            slot=p['root']/'slots'/p['attempts'][-1]
            if failure=='exit':return 9
            if failure=='missing_head':
                shutil.rmtree(slot/'raw'/'pointnext')
                (slot/'raw/pointnext.zip').unlink()
            if failure=='missing_zip':(slot/'raw/pointnext.zip').unlink()
            if failure=='cross_head_cache':
                p['reports'][str(slot/'raw/pointnext.zip')]['batch_guards'][0]['cache']['sha256']='f'*64
        return result
    monkeypatch.setattr(campaign.operations,'run_worker',worker)
    with pytest.raises((ValueError,FileNotFoundError)):launch(p)
    assert len(p['attempts'])==2
    assert 'DEPTH_TRANSFER_CAMPAIGN_COMPLETE' not in capsys.readouterr().out
    assert (p['root']/'failure.json').is_file()
    snapshots=list((p['drive']/'campaign').rglob('failure.json'))
    assert snapshots
    assert list((p['drive']/'campaign').rglob('pointmae.zip'))
    with pytest.raises((FileExistsError,ValueError)):launch(p)
    assert len(p['attempts'])==2


def test_invalid_cpu_receipt_identity_never_launches_worker(prepared):
    p=prepared;p['receipt'].write_text('{}')
    with pytest.raises(ValueError,match='CPU|receipt'):launch(p)
    assert not p['attempts']


@pytest.mark.parametrize('mutation',['phase_source','cuda_forward','cpu_checkpoint','fixed_setting','missing_binding'])
def test_sealed_but_invalid_preflight_or_job_is_rejected(prepared,mutation):
    p=prepared
    if mutation=='phase_source':p['plan']['phase_sources']['tta.py']['sha256']='f'*64
    elif mutation in ('cuda_forward','cpu_checkpoint'):
        cpu=json.loads(p['receipt'].read_text())
        if mutation=='cuda_forward':cpu['report']['forward_executed']=True
        else:cpu['report']['heads']['modelnet40_c']['pointmae']['checkpoint_load']['checkpoint_sha256']='f'*64
        p['receipt'].write_text(json.dumps(cpu))
        p['plan']['cpu_preflight_receipt_identity']=campaign.operations.file_identity(p['receipt'])
    elif mutation=='fixed_setting':p['plan']['jobs'][0]['command']+=['--gamma','.01']
    else:del p['plan']['jobs'][0]['bindings']['pointnext']
    p['seal']()
    with pytest.raises(ValueError):launch(p)
    assert not p['attempts']


@pytest.mark.parametrize('mutation',['ids','cpu_receipt','local_plan'])
def test_changed_frozen_inputs_stop_before_next_worker(prepared,monkeypatch,mutation):
    p=prepared
    def worker(*args):
        result=p['worker'](*args)
        if len(p['attempts'])==1:
            if mutation=='ids':
                command=p['plan']['jobs'][1]['command'];path=Path(command[command.index('--index-manifest')+1])
                content=json.loads(path.read_text());content['indices']=list(range(1,33));path.write_text(json.dumps(content))
            elif mutation=='cpu_receipt':p['receipt'].write_text('{}')
            else:(p['root']/'plan.json').write_text('{}')
        return result
    monkeypatch.setattr(campaign.operations,'run_worker',worker)
    with pytest.raises(ValueError):launch(p)
    assert len(p['attempts'])==1
    assert (p['root']/'failure.json').is_file()


def test_transport_failure_preserves_failure_and_stops_new_jobs(prepared,monkeypatch):
    p=prepared;persist=campaign.operations.persist_checkpoint
    def publish(*args,**kwargs):
        if len(p['attempts'])==1:raise OSError('synthetic transport failure')
        return persist(*args,**kwargs)
    monkeypatch.setattr(campaign.operations,'persist_checkpoint',publish)
    with pytest.raises(OSError,match='transport'):launch(p)
    assert len(p['attempts'])==1
    latest=json.loads((p['root']/'registered_latest_ledger.json').read_text())
    assert latest['status']=='failed'
    assert (p['root']/'failure.json').is_file()


def test_approval_changed_during_preflight_is_not_rebound_to_new_bytes(prepared,monkeypatch):
    p=prepared;copy_verified=campaign.operations.copy_verified
    def publish(source,target):
        result=copy_verified(source,target)
        if Path(target)==p['root']/'plan.json':
            approval=json.loads(p['approval_path'].read_text());approval['approved']=False
            p['approval_path'].write_text(json.dumps(approval))
        return result
    monkeypatch.setattr(campaign.operations,'copy_verified',publish)
    with pytest.raises(ValueError,match='approval|Approval'):launch(p)
    assert not p['attempts']


def test_final_archive_overrun_prevents_completion_marker(prepared,monkeypatch,capsys):
    p=prepared;clock=[0.];persist=campaign.operations.persist_checkpoint
    monkeypatch.setattr(campaign.time,'monotonic',lambda:clock[0])
    def publish(*args,**kwargs):
        result=persist(*args,**kwargs)
        if len(p['attempts'])==8 and args[4] is None:clock[0]=28801.
        return result
    monkeypatch.setattr(campaign.operations,'persist_checkpoint',publish)
    with pytest.raises(ValueError,match='cap'):launch(p)
    assert len(p['attempts'])==8
    assert 'DEPTH_TRANSFER_CAMPAIGN_COMPLETE' not in capsys.readouterr().out
    assert json.loads((p['root']/'registered_latest_ledger.json').read_text())['status']=='failed'
    assert list((p['drive']/'campaign').rglob('final_failure.json'))


def test_exhausted_budget_preserves_failure_without_starting_another_job(prepared,monkeypatch):
    p=prepared
    clock=[0.]
    monkeypatch.setattr(campaign.time,'monotonic',lambda:clock[0])
    def worker(*args):
        result=p['worker'](*args);clock[0]=28801.;return result
    monkeypatch.setattr(campaign.operations,'run_worker',worker)
    with pytest.raises(ValueError,match='cap|budget|Budget'):launch(p)
    assert len(p['attempts'])==1
    latest=json.loads((p['root']/'registered_latest_ledger.json').read_text())
    assert latest['status']=='failed' and latest['actual_elapsed_seconds']==28801.
    assert (p['root']/'failure.json').is_file()
