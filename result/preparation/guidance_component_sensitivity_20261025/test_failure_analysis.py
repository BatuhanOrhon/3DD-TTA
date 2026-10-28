"""Synthetic failure-state fixtures; no live confirmation predictions."""
import copy
import importlib.util
import io
import types
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,'/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
import analyze_failed_confirmation_local as analysis


def expected_jobs():
    return [dict(condition_id=condition,role=condition,trial=dict(id=condition),dataset=dataset,seed=seed)
            for seed in (0,1,2) for condition in ('spectral-candidate','matched-chamfer','tuned-chamfer','original3dd')
            for dataset in analysis.complete.DATASETS]


def fixture(second_failure=False):
    expected=expected_jobs()
    old=[dict(job,status='complete') for job in expected[:20]]+[dict(expected[20],status='running')]
    remaining=[dict(expected[i],registered_job_index=i,status='failed_numerical' if i==21 and second_failure else 'complete') for i in (21,22,23)]
    failures={20:dict(failure_type=analysis.FAILURE_SIGNATURE,partial_raw_archives=[{'synthetic':True}]*4)}
    if second_failure:failures[21]=copy.deepcopy(failures[20])
    return expected,old,remaining,failures


class FailureAnalysisTests(unittest.TestCase):
    def test_composite23complete_plusonefailed_preserves_all24(self):
        slots=analysis.compose_slots(*fixture())
        self.assertEqual(len(slots),24)
        self.assertEqual(sum(s['status']=='complete' for s in slots),23)
        self.assertEqual(slots[20]['status'],'failed_numerical')
        self.assertEqual(analysis.endpoint_conditions(slots,['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd']),
                         ['spectral-candidate','matched-chamfer','original3dd'])
        self.assertEqual(sum(s['status']=='complete' and s['condition_id']!='tuned-chamfer' for s in slots),18)

    def test_second_numerical_failure_does_not_change_primary_grid(self):
        slots=analysis.compose_slots(*fixture(True))
        self.assertEqual([s['registered_job_index'] for s in slots if s['status']=='failed_numerical'],[20,21])
        self.assertEqual(sum(s['status']=='complete' for s in slots),22)
        self.assertNotIn('tuned-chamfer',analysis.endpoint_conditions(slots,['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd']))

    def test_missing_or_duplicate_remaining_slots_are_not_silently_dropped(self):
        expected,old,remaining,failures=fixture()
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,remaining[:-1],failures)
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,remaining+[remaining[0]],failures)
        bad=copy.deepcopy(remaining);bad[0]['condition_id']='matched-chamfer'
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,bad,failures)

    def test_unknown_failure_and_failed_primary_method_stop(self):
        expected,old,remaining,failures=fixture()
        bad=copy.deepcopy(failures);bad[20]['failure_type']='CUDA out of memory'
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,remaining,bad)
        bad=copy.deepcopy(remaining);bad[-1]['status']='failed_numerical';f=copy.deepcopy(failures);f[23]=f[20]
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,bad,f)
        with self.assertRaises(ValueError):analysis.compose_slots(expected,old,remaining,{})

    def test_complete_only_phase_acceptance_is_not_weakened(self):
        plan=dict(session='r4',manifest_sha256='a',source_snapshot_sha256='b')
        state=dict(status='failed',stage='confirmation',session='r4',phase_plan_sha256='c',manifest_sha256='a',source_snapshot_sha256='b',full_benchmark=False)
        analysis.phase_header(state,plan,'c','confirmation','failed')
        with self.assertRaises(ValueError):analysis.phase_header(state,plan,'c','confirmation','complete')
        wrong=copy.deepcopy(state);wrong['source_snapshot_sha256']='foreign'
        with self.assertRaises(ValueError):analysis.phase_header(wrong,plan,'c','confirmation','failed')

    def test_main_bootstrap_reuses280objects_not840_and_never_tuned_fallback(self):
        slots=analysis.compose_slots(*fixture())
        names=analysis.endpoint_conditions(slots,['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd'])
        outcomes={(d,h,c,p,s):np.zeros((15,140),dtype=bool)
                  for d in analysis.complete.DATASETS for h in analysis.complete.HEADS for c in names
                  for p in (('terminal',) if c=='original3dd' else analysis.complete.POLICIES) for s in analysis.complete.SEEDS}
        for key,value in outcomes.items():
            if key[2]=='spectral-candidate':value[:,:70]=True
        _,rows=analysis.complete.paired_tables(outcomes,['background']+['c%d'%i for i in range(14)],['matched-chamfer','original3dd'])
        self.assertTrue(all(r['n_original_object_clusters']==280 for r in rows))
        self.assertFalse(any(r['comparator']=='tuned-chamfer' for r in rows))
        mean=next(r for r in rows if r['scope']=='all15' and r['seed'] is None and r['comparator']=='matched-chamfer')
        seed=next(r for r in rows if r['scope']=='all15' and r['seed']==0 and r['comparator']=='matched-chamfer')
        self.assertEqual(mean['conditional_object_bootstrap_low_pp'],seed['conditional_object_bootstrap_low_pp'])
        self.assertEqual(mean['conditional_object_bootstrap_high_pp'],seed['conditional_object_bootstrap_high_pp'])
        json.dumps(rows,allow_nan=False)

    def test_known_failure_rejects_additional_error_or_missing_source_frame(self):
        log='Traceback (most recent call last):\n  File "classifier_backbones.py", line 48, in forward\n    raise ValueError("Classifier input contains nonfinite XYZ coordinates")\nValueError: '+analysis.FAILURE_SIGNATURE+'\n'
        self.assertTrue(analysis.known_failure(log))
        self.assertFalse(analysis.known_failure(log+'RuntimeError: CUDA failure\n'))
        self.assertFalse(analysis.known_failure(log.replace('line 48','line 49')))
        self.assertFalse(analysis.known_failure(log+log))

    def test_physical_evidence_tampering_fails_and_complete_reader_rejects_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            plan=dict(session='r4',manifest_sha256='a',source_snapshot_sha256='b')
            state=dict(status='failed',stage='confirmation',session='r4',phase_plan_sha256=analysis.shared.PLAN_SHA,
                       manifest_sha256='a',source_snapshot_sha256='b',full_benchmark=False,synthetic_fixture=True)
            files={'state.json':json.dumps(state).encode(),'command.txt':b'synthetic fixture'}
            files['evidence_manifest.json']=json.dumps({name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()}).encode()
            archive=root/'synthetic_failed.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name,raw in files.items():z.writestr(name,raw)
            reference=dict(analysis.shared.identity(archive),local_path='/content/studies/guidance-components/synthetic_failed.zip',drive_path=analysis.shared.DRIVE+'synthetic/failed.zip')
            observed=analysis.extract_evidence(archive,reference,root/'good')
            self.assertEqual(observed,state)
            receipt=root/'receipt.json';receipt.write_text(json.dumps(dict(state=state,archive=reference)))
            with self.assertRaisesRegex(ValueError,'Phase status/source/ownership'):
                analysis.shared.unpack_phase(archive,receipt,'confirmation',plan,root/'must_not_accept')
            files['command.txt']=b'changed fixture'
            with zipfile.ZipFile(archive,'w') as z:
                for name,raw in files.items():z.writestr(name,raw)
            reference.update(analysis.shared.identity(archive))
            with self.assertRaisesRegex(ValueError,'Evidence member hash/size'):
                analysis.extract_evidence(archive,reference,root/'bad')

    def test_prefixed_failed_bundle_reads_only_ids_labels_and_metadata(self):
        prep=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
        Bundle=analysis.shared.load_context(prep)[7]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);ids=np.arange(140,dtype=np.int64);labels=ids%40
            selected=dict(indices=ids.tolist());selection=json.dumps(selected).encode()
            log='Traceback (most recent call last):\n  File "classifier_backbones.py", line 48, in forward\n    raise ValueError("Classifier input contains nonfinite XYZ coordinates")\nValueError: '+analysis.FAILURE_SIGNATURE+'\n'
            config=dict(status='failed',failure=log,run_id='synthetic-failed',backbone='dgcnn',
                selection_manifest_identity=dict(bytes=len(selection),sha256=hashlib.sha256(selection).hexdigest()),
                asset_manifest_sha256='synthetic',checkpoint_load=dict(checkpoint_sha256='synthetic'),native_bootstrap=dict(bindings={}),
                classifier_state_before_sha256='synthetic',lion_state_before={})
            required=['scripts/eval_multibackbone_guidance.py','scripts/eval_multibackbone_source.py','classifier_backbones.py','spectral_guidance_losses.py','tta_gsd_v2.py','shared_trajectory.py']
            source_files={'source_snapshot/'+name:dict(bytes=1,sha256='synthetic') for name in required}
            config['executed_source_files']={'assets/'+name:pin for name,pin in source_files.items()}
            files={'config.json':json.dumps(config).encode(),'selection_manifest.json':selection,'summary.csv':b'status\nfailed\n'}
            for corruption,batch in [('uniform',0),('uniform',1),('gaussian',0),('gaussian',1),('background',0)]:
                prefix='batches/'+corruption+'/batch%04d/'%batch
                raw=io.BytesIO();np.savez(raw,index=ids[batch*70:(batch+1)*70],label=labels[batch*70:(batch+1)*70])
                files[prefix+'predictions.npz']=raw.getvalue()
                files[prefix+'diagnostics.json']=json.dumps(dict(seed=2,rng_unchanged=True,geometry_shape=[70,2048,3],
                    trajectory=dict(initial_hashes={'input_points':'synthetic'}),paired_state=dict(path='synthetic'))).encode()
            files['bundle_manifest.json']=json.dumps({n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for n,raw in files.items()}).encode()
            path=root/'synthetic-failed.zip'
            with zipfile.ZipFile(path,'w') as archive:
                for name,raw in files.items():archive.writestr('synthetic-failed/'+name,raw)
            guarded=[]
            controller=types.SimpleNamespace(verify_common_config=lambda cfg,*args:guarded.append(cfg['status']))
            context=(dict(source_snapshot_sha256='synthetic'),controller,None,root,dict(files=source_files),None,{('confirmation','modelnet40_c'):selected},Bundle,None,None,
                     ('uniform','gaussian','background')+tuple('c%d'%i for i in range(12)))
            row=analysis.partial_failure(path,dict(dataset='modelnet40_c',seed=2),context,{},log,{'modelnet40_c':dict(label=labels)})
            self.assertEqual(guarded,['partial']);self.assertFalse(row['accuracy_used']);self.assertEqual(len(row['completed_prefix']),5)
            bundle=Bundle(path)
            try:self.assertEqual(bundle.json('config.json')['status'],'failed')
            finally:bundle.close()

    def test_prefixed_terminal_reader_matches_flat_without_estimator_change(self):
        prep=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
        Bundle=analysis.shared.load_context(prep)[7]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=io.BytesIO();np.savez(raw,label=np.array([0,1]),prediction=np.array([0,0]))
            files={'config.json':b'{}','batches/gaussian/batch0000/predictions.npz':raw.getvalue()}
            files['bundle_manifest.json']=json.dumps({n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}).encode()
            values=[]
            for prefix in ('','synthetic-run/'):
                path=root/('prefixed.zip' if prefix else 'flat.zip')
                with zipfile.ZipFile(path,'w') as archive:
                    for name,data in files.items():archive.writestr(prefix+name,data)
                values.append(analysis.raw_outcomes(path,True,['gaussian'],None,Bundle)['terminal'])
            self.assertTrue(np.array_equal(values[0],values[1]));self.assertEqual(values[0].tolist(),[[True,False]])

    def test_failed_provenance_recovers_only_verified_missing_shared_inventory(self):
        prep=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
        context=analysis.shared.load_context(prep)
        plan,manifest=context[0],context[4]
        required=['scripts/eval_multibackbone_guidance.py','scripts/eval_multibackbone_source.py','classifier_backbones.py','spectral_guidance_losses.py','tta_gsd_v2.py']
        root=plan['asset_root']+'/source_snapshot/'
        log='Traceback (most recent call last):\n  File "'+root+'scripts/eval_multibackbone_guidance.py", line 525, in worker\n  File "'+root+'shared_trajectory.py", line 73, in clean\n  File "classifier_backbones.py", line 48, in forward\n    raise ValueError("Classifier input contains nonfinite XYZ coordinates")\nValueError: '+analysis.FAILURE_SIGNATURE+'\n'
        config=dict(status='failed',failure=log,executed_source_files={'assets/source_snapshot/'+name:manifest['files']['source_snapshot/'+name] for name in required})
        raw=(prep/'assets_v2/source_snapshot.zip').read_bytes()
        bundle=types.SimpleNamespace(read=lambda name:raw)
        guarded,proof=analysis.failed_source_contract(config,bundle,context)
        key='assets/source_snapshot/shared_trajectory.py'
        self.assertEqual(config['status'],'failed');self.assertNotIn(key,config['executed_source_files'])
        self.assertEqual(guarded['executed_source_files'][key],manifest['files']['source_snapshot/shared_trajectory.py'])
        self.assertTrue(proof['source_inventory_recovered_from_snapshot']);self.assertEqual(proof['missing_original_inventory_keys'],[key])
        with self.assertRaisesRegex(ValueError,'snapshot changed'):
            analysis.failed_source_contract(config,types.SimpleNamespace(read=lambda name:b'bad'+raw),context)
        bad_context=list(context);bad_manifest=copy.deepcopy(manifest);bad_manifest['files']['source_snapshot/shared_trajectory.py']['sha256']='0'*64;bad_context[4]=bad_manifest
        with self.assertRaisesRegex(ValueError,'member identity'):
            analysis.failed_source_contract(config,bundle,tuple(bad_context))
        bad=copy.deepcopy(config);bad['executed_source_files'].pop('assets/source_snapshot/tta_gsd_v2.py')
        with self.assertRaisesRegex(ValueError,'sole known'):
            analysis.failed_source_contract(bad,bundle,context)
        bad=copy.deepcopy(config);bad['executed_source_files']['assets/source_snapshot/tta_gsd_v2.py']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'recorded source identity'):
            analysis.failed_source_contract(bad,bundle,context)
        bad=copy.deepcopy(config);bad['failure']=log.replace('line 525','line 524')
        with self.assertRaisesRegex(ValueError,'pinned worker'):
            analysis.failed_source_contract(bad,bundle,context)

    def test_tuned_endpoint_is_explicitly_null_notzero_or_two_seed_mean(self):
        slots=analysis.compose_slots(*fixture())
        endpoint=analysis.unavailable_endpoints(slots,['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd'])
        self.assertEqual(len(endpoint),1)
        self.assertEqual(endpoint[0]['condition_id'],'tuned-chamfer')
        self.assertIsNone(endpoint[0]['three_seed_balanced_accuracy_pct'])
        self.assertIsNone(endpoint[0]['conditional_object_bootstrap_interval_pp'])
        self.assertEqual(endpoint[0]['failed_registered_jobs'],[20])
        self.assertEqual(endpoint[0]['completed_jobs'],5)


if __name__=='__main__':unittest.main()
