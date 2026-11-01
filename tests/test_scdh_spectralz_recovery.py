"""CPU-only persistence, unavailable-candidate and frozen-lineage regressions."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_scdh_spectralz as operator
from scripts.prepare_scdh_spectralz import write_new
from scripts.prepare_chamfer_variants_assets import identity
from scripts.scdh_spectralz_study import canonical_sha


def simulated_score(root,j,report):
    slot=root/'slots'/j['slot'];slot.mkdir(parents=True,exist_ok=True)
    for name in ('accepted.json','score.json'):write_new(slot/name,report)
    write_new(slot/'receipt.json',dict(status='accepted',job=j,report_identity=identity(slot/'accepted.json'),score_identity=identity(slot/'score.json')))
    archives=root/'archives';archives.mkdir(exist_ok=True)
    archive=operator.archive_slot(slot,archives/(j['slot']+'.zip'))
    write_new(slot/'archive_receipt.json',dict(archive_identity=identity(archive)))


def context(root):
    plan=dict(root=str(root),drive_root=str(root/'drive'),cumulative_wall_cap_seconds=14400,
              source_snapshot_sha256='a'*64,asset_manifest_sha256='b'*64,archive_reserve_seconds=180,
              dataset_root='/data',label_path='/data/label.npy',asset_root='/assets',
              pointmae_config='/assets/config.yaml',pointmae_checkpoint='/assets/model.pth',python='python')
    approval=dict(approved=True,scope=operator.SCOPE,plan_sha256=canonical_sha(plan),user_message='scientific phase test fixture')
    seen=[]
    def execute(plan,j,deadline,manifest):
        seen.append(j)
        n=128 if j['stage']=='confirmation' else 64 if j['stage']=='development' else 32
        outcomes=[[False]*n for _ in range(15)];correct=0
        if j['arm']=='spectral':outcomes[0][0]=True;correct=1
        report=dict(j,status='accepted',correct=correct,object_ids=list(range(n)),labels=[0]*n,
                    corruptions=list(range(15)),calibration_binding={},batch_guards=[],outcomes={'probability_mean':outcomes})
        if j['arm']=='calibration':report['calibration_observations']=[dict(M_max=m,ratios=[1.]*32) for m in operator.study.MODES for _ in range(12)]
        simulated_score(root,j,report)
        return report
    return plan,approval,seen,execute


class RecoveryTests(unittest.TestCase):
    def test_malformed_predecessor_stops_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,approval,_,execute=context(root)
            (root/'technical_accepted.json').write_text('{')
            with patch.object(operator,'validate_plan',return_value=(root,{})),patch.object(operator,'execute_job') as run:
                with self.assertRaises(ValueError):operator.run(plan,approval,'calibration')
                run.assert_not_called()

    def test_failed_phase_publication_resumes_copies_without_repeating_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,approval,seen,execute=context(root)
            original=operator.operations.copy_verified;failed=[]
            def copy(source,target):
                if Path(target).name=='technical_accepted.json' and not failed:
                    failed.append(True);raise OSError('Drive unavailable')
                return original(source,target)
            with patch.object(operator,'validate_plan',return_value=(root,{})),patch.object(operator,'execute_job',side_effect=execute),patch.object(operator.operations,'copy_verified',side_effect=copy):
                with self.assertRaises(OSError):operator.run(plan,approval,'technical')
                count=len(seen);operator.run(plan,approval,'technical')
                self.assertEqual(len(seen),count)
            target=root/'drive/campaign'/canonical_sha(plan)/'technical_accepted.json'
            self.assertEqual(identity(target),identity(root/'technical_accepted.json'))
            for name in ('plan.json','approval.json','started.json'):
                self.assertEqual(identity(target.parent/name),identity(root/name))

    def test_frozen_calibration_and_selection_cannot_be_rehashed_into_new_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,approval,_,execute=context(root)
            with patch.object(operator,'validate_plan',return_value=(root,{})),patch.object(operator,'execute_job',side_effect=execute):
                for phase in ('technical','calibration'):operator.run(plan,approval,phase)
                path=root/'frozen_calibration.json';original=path.read_bytes();report=json.loads(original)
                report['coefficients']['200']['b_M']*=2;path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):operator.command(plan,operator.study.phase_a()[2])
                path.write_bytes(original)
                operator.run(plan,approval,'A')
                path=root/'frozen_top2.json';report=json.loads(path.read_text());report['top2'][0]['modes']=400;path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):operator.phase_jobs('B',root)

    def test_numerical_candidate_is_persisted_unavailable_and_not_reexecuted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,_,_,_=context(root)
            j=operator.study.job('spectral',modes=200,rho=.1)
            cfg=dict(status='failed',scdh_failure_category='numerical_candidate_unavailable',
                source_snapshot_sha256=plan['source_snapshot_sha256'],asset_manifest_sha256=plan['asset_manifest_sha256'],
                executed_source_sha256='d'*64,executed_source_files={},classifier_state_before_sha256='c'*64,
                classifier_state_after_sha256='c'*64,lion_state_before={'vae':'e'*64,'priors':'f'*64},checkpoint_load={'strict':True})
            manifest={'files':{}}
            for name in operator.analysis.CRITICAL:
                expected={'bytes':1,'sha256':'d'*64};manifest['files']['source_snapshot/'+name]=expected;cfg['executed_source_files']['assets/source_snapshot/'+name]=expected
            cfg['executed_source_sha256']=canonical_sha(cfg['executed_source_files'])
            def failure(*args):
                raw=root/'slots'/j['slot']/'raw/run';raw.mkdir(parents=True);write_new(raw/'config.json',cfg)
                return 1
            with patch.object(operator,'command',return_value=['python','-u','entry']),patch.object(operator.worker,'configurations',return_value=[cfg]),patch.object(operator.operations,'run_worker',side_effect=failure) as run:
                first=operator.execute_job(plan,j,time.time()+3000,manifest)
                second=operator.execute_job(plan,j,time.time()+3000,manifest)
                self.assertEqual(first,second);self.assertEqual(first['status'],'unavailable');self.assertNotIn('correct',first)
                self.assertEqual(run.call_count,1)

class PublicationTests(unittest.TestCase):
    def test_owned_incomplete_copy_is_preserved_then_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'complete evidence');target=root/'drive/file'
            def partial(src,dst):Path(dst).write_bytes(b'partial');raise OSError('transfer interrupted')
            with patch.object(operator.operations,'copy_verified',side_effect=partial):
                with self.assertRaises(OSError):operator.persist(source,target)
            operator.persist(source,target)
            self.assertEqual(identity(source),identity(target))
            self.assertEqual(next(target.parent.glob('file.incomplete-*')).read_bytes(),b'partial')

    def test_operator_protocol_failure_gets_an_immutable_local_and_Drive_archive(self):
        import argparse
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,approval,_,_=context(root)
            write_new(root/'plan.json',plan);write_new(root/'approval.json',approval)
            args=argparse.Namespace(plan=root/'plan.json',approval=root/'approval.json',phase='A')
            with patch('argparse.ArgumentParser.parse_args',return_value=args),patch.object(operator,'run',side_effect=ValueError('protocol failure')):
                with self.assertRaises(ValueError):operator.main()
            archive=next((root/'operator_failures').glob('*.zip'))
            self.assertEqual(identity(archive),identity(operator.publication_root(plan)/'operator_failures'/archive.name))

class DurationAmendmentTests(unittest.TestCase):
    def test_explicit_duration_amendment_keeps_original_clock_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,approval,_,_=context(root)
            operator.budget(root,plan,approval,now=1000)
            original=(root/'started.json').read_bytes()
            amendment=dict(approved=True,scope=operator.SCOPE,plan_sha256=canonical_sha(plan),
                original_approval_sha256=canonical_sha(approval),original_started_identity=identity(root/'started.json'),
                original_start_unix_seconds=1000,cumulative_wall_cap_seconds=46800,
                operator_identity=identity(Path(operator.__file__)),user_message='Continue without further duration questions')
            self.assertEqual(operator.budget(root,plan,approval,now=16000,amendment=amendment),47800)
            self.assertEqual((root/'started.json').read_bytes(),original)
            self.assertEqual(operator.budget(root,plan,approval,now=17000,amendment=amendment),47800)
            with self.assertRaises(ValueError):operator.budget(root,plan,approval,now=17000)
            for key,value in [('original_start_unix_seconds',2000),('approved',False),('operator_identity',{}),('original_started_identity',{})]:
                with self.assertRaises(ValueError):operator.budget(root,plan,approval,now=17000,amendment=dict(amendment,**{key:value}))
            with self.assertRaises(ValueError):operator.budget(root,plan,approval,now=47801,amendment=amendment)

class ConfirmationTimeoutRecoveryTests(unittest.TestCase):
    def test_only_failed_confirmation_slot_and_its_watchdog_change(self):
        from types import SimpleNamespace
        from scripts.recover_scdh_confirmation import install_recovery
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);receipt=root/'receipt.json';receipt.write_text('{}');source=root/'source.py';source.write_text('# fixed')
            failed=dict(slot='confirmation-spectral-first',stage='confirmation',arm='spectral',modes=400,rho=1.,alpha_h=.03,seed=0)
            control=dict(slot='confirmation-scd-first',stage='confirmation',arm='scd')
            baseline=dict(slot='development-scd',stage='development',arm='scd')
            jobs=lambda phase,root:[dict(failed),dict(control)] if phase=='C' else [dict(baseline)]
            calls=[]
            controller=SimpleNamespace(phase_jobs=jobs,operations=SimpleNamespace(run_worker=lambda *a:calls.append(a)),evidence=lambda root,values:{'existing':'fixed'})
            record=dict(per_job_timeout_seconds=1800,failed_job=failed,retry_slot=failed['slot']+'-retry1')
            install_recovery(controller,record,receipt,source)
            actual=controller.phase_jobs('C',root)
            self.assertEqual(actual[0],dict(failed,slot=record['retry_slot']));self.assertEqual(actual[1],control)
            self.assertEqual(controller.phase_jobs('A',root),[baseline]);self.assertEqual(controller.evidence(root,[baseline]),{'existing':'fixed'})
            self.assertEqual(set(controller.evidence(root,actual)),{'existing','receipt.json','source.py'})
            for timeout,slot in [(900,'confirmation-test'),(300,'confirmation-test'),(900,'development-test')]:controller.operations.run_worker([],root,root/'log',timeout,slot)
            self.assertEqual([c[3] for c in calls],[1800,300,900])
            record['failed_job']=dict(failed,modes=800)
            with self.assertRaises(ValueError):controller.phase_jobs('C',root)

class NativeRuntimeTransportTests(unittest.TestCase):
    def test_frozen_coefficients_keep_their_origin_and_only_declared_native_ids_change(self):
        from copy import deepcopy
        from scripts.recover_scdh_confirmation import transport_calibration,NATIVE_REBUILDS
        old_native={name:dict(path='/native/'+name,bytes=row['bytes'],sha256=row['old']) for name,row in NATIVE_REBUILDS.items()}
        old_native['knn']=dict(path='/native/knn',bytes=12,sha256='a'*64)
        current=deepcopy(old_native)
        for name,row in NATIVE_REBUILDS.items():current[name]['sha256']=row['new']
        original=dict(schema_version=1,status='frozen',coefficients={'400':{'b_M':19.769744597096295,'status':'available','n_valid':384}},binding={'native_extensions':old_native,'source_snapshot_sha256':'b'*64},observations_sha256='c'*64)
        before=deepcopy(original)
        report=transport_calibration(original,{'sha256':'d'*64,'bytes':42},current)
        self.assertEqual(original,before);self.assertEqual(report['coefficients'],before['coefficients'])
        self.assertEqual(report['binding']['native_extensions'],current)
        self.assertEqual(report['runtime_transport']['original_binding'],before['binding'])
        self.assertFalse(report['runtime_transport']['native_bitwise_equality_claimed'])
        self.assertFalse(report['runtime_transport']['coefficients_recomputed'])
        for name in current:
            corrupt=deepcopy(current);corrupt[name]['sha256']='f'*64
            with self.assertRaises(ValueError):transport_calibration(original,{},corrupt)

    def test_transport_uses_current_native_proof_only_for_new_conditions(self):
        from types import SimpleNamespace
        from scripts.recover_scdh_confirmation import install_recovery,identity as rid
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('receipt.json','source.py','current_cpu.json','transport.json'):(root/name).write_text('{}')
            plan=dict(cpu_preflight_receipt='historical.json',root=str(root));(root/'plan.json').write_text(json.dumps(plan))
            failed=dict(slot='confirmation-spectral',stage='confirmation',arm='spectral')
            control=dict(slot='confirmation-scd',stage='confirmation',arm='scd')
            baseline=dict(slot='development-scd',stage='development',arm='scd')
            c=SimpleNamespace(phase_jobs=lambda phase,r:[dict(failed),dict(control)] if phase=='C' else [dict(baseline)],operations=SimpleNamespace(run_worker=lambda *a:None),evidence=lambda r,j:{'original':'fixed'},publication_root=lambda p:'publication-'+p['cpu_preflight_receipt'])
            c.command=lambda p,j:['python','--calibration-report','original.json','--calibration-report-sha256','old']
            c.execute_job=lambda p,j,d,m:dict(cpu=p['cpu_preflight_receipt'],command=c.command(p,j),publication=c.publication_root(p))
            transport=dict(cpu_file='current_cpu.json',cpu_identity=rid(root/'current_cpu.json'),calibration_file='transport.json',calibration_identity=rid(root/'transport.json'))
            record=dict(per_job_timeout_seconds=1800,failed_job=failed,retry_slot='confirmation-spectral-retry1',runtime_transport=transport)
            install_recovery(c,record,root/'receipt.json',root/'source.py')
            current=c.execute_job(plan,failed,99,{})
            self.assertEqual(current['cpu'],str(root/'current_cpu.json'))
            self.assertEqual(current['publication'],'publication-historical.json')
            self.assertEqual(current['command'][2],str(root/'transport.json'))
            self.assertEqual(c.execute_job(plan,baseline,99,{})['cpu'],'historical.json')
            self.assertEqual(c.evidence(root,[baseline]),{'original':'fixed'})
            self.assertIn('transport.json',c.evidence(root,[failed]))
