"""CPU checks for the prospective140-ID confirmation analysis."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

import analyze_sensitivity_confirmation_local as analysis
import analyze_sensitivity_local as shared

PREP = None
MIGRATION_PLAN = None


def outcome_fixture():
    return {(dataset, head, condition, policy, seed): np.zeros((15, 140), dtype=bool)
            for dataset in analysis.DATASETS for head in analysis.HEADS for condition in ('spectral-candidate', 'matched-chamfer', 'original3dd')
            for policy in (('terminal',) if condition == 'original3dd' else analysis.POLICIES) for seed in analysis.SEEDS}


class ConfirmationTests(unittest.TestCase):
    def test_identical_seeds_keep_object_interval_and280clusters(self):
        outcomes = outcome_fixture()
        for key, values in outcomes.items():
            if key[2] == 'spectral-candidate':
                values[:, :70] = True
        _, rows = analysis.paired_tables(outcomes, ['background'] + ['c%d' % i for i in range(14)], ['matched-chamfer', 'original3dd'], 700)
        rows = [r for r in rows if r['scope'] == 'all15' and r['comparator'] == 'matched-chamfer']
        mean = next(r for r in rows if r['seed'] is None)
        single = next(r for r in rows if r['seed'] == 0)
        self.assertEqual(mean['n_original_object_clusters'], 280)
        self.assertEqual(mean['conditional_object_bootstrap_low_pp'], single['conditional_object_bootstrap_low_pp'])
        self.assertEqual(mean['conditional_object_bootstrap_high_pp'], single['conditional_object_bootstrap_high_pp'])
        self.assertGreater(mean['conditional_object_bootstrap_high_pp'], mean['conditional_object_bootstrap_low_pp'])
        self.assertEqual(mean['delta_pp'], 50.)
        json.dumps(rows, allow_nan=False)

    def test_opposite_same_object_seed_effects_cancel_exactly(self):
        outcomes = outcome_fixture()
        for dataset in analysis.DATASETS:
            for head in analysis.HEADS:
                outcomes[(dataset,head,'spectral-candidate','probability_mean',0)][:,:70] = True
                outcomes[(dataset,head,'matched-chamfer','probability_mean',1)][:,:70] = True
        _, rows = analysis.paired_tables(outcomes, ['background'] + ['c%d' % i for i in range(14)], ['matched-chamfer'], 700)
        row = next(r for r in rows if r['scope'] == 'all15' and r['seed'] is None)
        self.assertEqual((row['delta_pp'],row['conditional_object_bootstrap_low_pp'],row['conditional_object_bootstrap_high_pp']), (0.,0.,0.))

    def test_balanced_datasets_and_macro14_are_preserved(self):
        outcomes = outcome_fixture()
        for head in analysis.HEADS:
            for seed in analysis.SEEDS:
                outcomes[(analysis.DATASETS[0],head,'spectral-candidate','probability_mean',seed)][:] = True
        _, rows = analysis.paired_tables(outcomes, ['background'] + ['c%d' % i for i in range(14)], ['matched-chamfer'], 100)
        self.assertTrue(all(row['delta_pp'] == 50. for row in rows))
        outcomes = outcome_fixture()
        for key,values in outcomes.items():
            if key[2] == 'spectral-candidate':values[0] = True
        _, rows = analysis.paired_tables(outcomes, ['background'] + ['c%d' % i for i in range(14)], ['matched-chamfer'], 100)
        self.assertTrue(all(row['delta_pp'] == 0. for row in rows if row['scope'] == 'macro14'))
        self.assertTrue(all(abs(row['delta_pp']-100./15) < 1e-10 for row in rows if row['scope'] == 'all15'))

    def test_original_uses_terminal_and140ids_five_batches_for_seed2(self):
        spec=importlib.util.spec_from_file_location('old_fixtures',PREP/'test_analyze_sensitivity_local.py')
        fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
        context=shared.load_context(PREP)
        with tempfile.TemporaryDirectory() as directory:
            path,job,context=fixtures.make_original_fixture(Path(directory),context)
            with zipfile.ZipFile(path) as archive:files={name:archive.read(name) for name in archive.namelist()}
            for name in list(files):
                if name=='config.json' or name.endswith('/diagnostics.json'):
                    value=json.loads(files[name]);value['seed']=2;files[name]=json.dumps(value).encode()
            files['bundle_manifest.json']=json.dumps({name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items() if name!='bundle_manifest.json'}).encode()
            with zipfile.ZipFile(path,'w') as archive:
                for name,data in files.items():archive.writestr(name,data)
            job['seed']=2
            config,_,_,traces,_=shared.inspect_raw(path,job,'development',context,{},False,{},{},{},[])
            self.assertEqual(config['seed'],2);self.assertEqual(len(traces),10)
            outcomes=analysis.raw_outcomes(path,True,context[-1],context[-2])
            self.assertEqual(set(outcomes),{'terminal'});self.assertEqual(outcomes['terminal'].shape,(2,140));self.assertTrue(outcomes['terminal'].all())
            job['seed']=1
            with self.assertRaisesRegex(ValueError,'head/seed'):
                shared.inspect_raw(path,job,'development',context,{},False,{},{},{},[])

    def test_frozen_conditions_match_registered_construction_and_dedup(self):
        context=shared.load_context(PREP);controller,registry=context[1],context[5]
        calibration=dict(coefficients=dict(legacy_l2=10.,huber=20.,power=.3))
        scores=[dict(trial_id=name,family=trial['family'],accuracy_pct=99. if name in ('cs-huber-r010','c-base') else 1.) for name,trial in registry.items()]
        accepted=dict(status='accepted',phase_plan_sha256=shared.PLAN_SHA,physical_cache_complete=True,selection_policy='probability_mean',selected_spectral_trial_id='cs-huber-r010',selected_chamfer_trial_id='c-base')
        expected=shared.freeze_selection(scores,registry,dict(bytes=1,sha256='a'*64),controller,calibration)
        self.assertEqual(len(analysis.frozen_conditions(expected,accepted,scores,registry,controller,calibration,dict(bytes=1,sha256='a'*64),shared.PLAN_SHA)),3)
        bad=copy.deepcopy(expected);bad['conditions'].insert(2,dict(id='tuned-chamfer',role='tuned_chamfer',trial_id='c-base'))
        with self.assertRaisesRegex(ValueError,'Frozen'):
            analysis.frozen_conditions(bad,accepted,scores,registry,controller,calibration,dict(bytes=1,sha256='a'*64),shared.PLAN_SHA)

    def test_cost_counts_shared_once_and_sums_classifier_only(self):
        rows=[dict(dataset='modelnet40_c',backbone=head,condition_id='spectral-candidate',seed=1,batch='gaussian/batch0000',shared_reconstruction_seconds=10.,classifier_forward_seconds=.5,whole_process_peak_gpu_memory_mb=100.,preparation_mode='cached_replay') for head in analysis.HEADS]
        cost,=analysis.cost_table(rows)
        self.assertEqual(cost['shared_reconstruction_seconds'],10.)
        self.assertEqual(cost['classifier_forward_seconds'],2.)
        self.assertEqual(cost['peak_gpu_memory_mb'],100.)
        with self.assertRaisesRegex(ValueError,'four'):
            analysis.cost_table(rows[:-1])


    def test_missing180physical_caches_remain_pending(self):
        records,paired=[],{}
        for seed in analysis.SEEDS:
            for dataset in analysis.DATASETS:
                for i in range(15):
                    for batch in range(2):
                        path=str(shared.REMOTE/('paired_sensitivity_confirmation_v2_seed%d'%seed)/dataset/('c%d'%i)/('b%d.npz'%batch))
                        records.append(dict(local_path=path,drive_path=shared.DRIVE+'cache.npz',bytes=1,sha256='a'*64))
                        paired[(seed,dataset,'c%d'%i,batch)]=dict(paired_state=dict(path=path,bytes=1,sha256='a'*64))
        rows=analysis.validate_physical_caches(records,paired,[])
        self.assertEqual(len(rows),180)
        self.assertFalse(any(r['local_verified'] for r in rows))
        records[-1]['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'physical-cache receipt'):
            analysis.validate_physical_caches(records,paired,[])

    def test_new140_labels_bind_full_sealed_files_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=dict(datasets={});selections={}
            modelnet=root/'modelnet_labels.npy'
            for dataset,count,classes,dtype in [('modelnet40_c',2468,40,np.uint8),('scanobjectnn_c',581,15,np.int64)]:
                path=modelnet if dataset=='modelnet40_c' else root/'datasets'/dataset/'label.npy'
                path.parent.mkdir(parents=True,exist_ok=True)
                values=(np.arange(count)%classes).astype(dtype)
                if dataset=='modelnet40_c':values=values[:,None]
                np.save(path,values)
                expected=dict(shared.identity(path),shape=list(values.shape),dtype=str(values.dtype))
                manifest['datasets'][dataset]=dict(files={'label.npy':expected})
                selections[('confirmation',dataset)]=dict(indices=list(range(140)))
            context=[None,None,None,root,manifest,None,selections]
            labels,proof=analysis.label_references(context,modelnet)
            self.assertEqual(len(proof),2)
            self.assertEqual(labels['modelnet40_c']['label'].tolist(),(np.arange(140)%40).tolist())
            modelnet.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                analysis.label_references(context,modelnet)

    def test_actual_migrated_context_reuses_exact_original_calibration(self):
        if MIGRATION_PLAN is None:
            self.skipTest('Pass --migration-plan for actual recovery integration')
        sha=shared.identity(MIGRATION_PLAN)['sha256']
        context=analysis.load_context(PREP,MIGRATION_PLAN,sha)
        self.assertEqual(len(context[5]),19)
        saved=shared.PLAN_SHA
        try:
            shared.PLAN_SHA=sha
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                receipt=shared.unpack_calibration(PREP,PREP/'calibrate_evidence_drive_roundtrip.zip',PREP/'calibrate_receipt.json',context[0],root/'calibration')
                report,caches=shared.calibration_report(root/'calibration',receipt,context)
                self.assertEqual(shared.PLAN_SHA,sha)
                self.assertEqual(report['session'],'tta-guidance-components-a100-20261025-r1')
                self.assertEqual(len(caches),30)
                bad=copy.deepcopy(receipt);bad['state']['session']=context[0]['session']
                path=root/'bad_receipt.json';path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'Phase status/source/ownership'):
                    shared.unpack_calibration(PREP,PREP/'calibrate_evidence_drive_roundtrip.zip',path,context[0],root/'bad_calibration')
                self.assertEqual(shared.PLAN_SHA,sha)
        finally:
            shared.PLAN_SHA=saved


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prep',type=Path,required=True);parser.add_argument('--migration-plan',type=Path)
    args,remaining=parser.parse_known_args();PREP=args.prep.resolve();MIGRATION_PLAN=args.migration_plan
    unittest.main(argv=[__file__]+remaining)
