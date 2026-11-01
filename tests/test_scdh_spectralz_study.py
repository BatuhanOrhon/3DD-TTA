"""Finite registration, label-free scaling and selection witnesses."""
import unittest

class StudyTests(unittest.TestCase):
    def test_registration_is_label_free_disjoint_and_excludes_prior_subset(self):
        from scripts.scdh_spectralz_study import register_ids
        first=register_ids([1,2,3], salt='fixed')
        self.assertEqual(first,register_ids([1,2,3],salt='fixed'))
        dev,heldout=first['development'],first['confirmation']
        self.assertEqual((len(dev),len(heldout)),(64,128))
        self.assertFalse(set(dev)&set(heldout));self.assertFalse((set(dev)|set(heldout))&{1,2,3})
        self.assertEqual(first['calibration'],sorted(dev)[:32])
        with self.assertRaises(ValueError):register_ids(range(2400),salt='fixed')

    def test_calibration_pools_only_valid_ratios_and_never_inflates_zero(self):
        from scripts.scdh_spectralz_study import freeze_calibration
        records=[dict(M_max=m,ratios=[1.,3.,None,0.]) for m in (200,400,800,1200)]
        records.append(dict(M_max=1800,ratios=[None]))
        report=freeze_calibration(records,{'source_snapshot_sha256':'a'*64})
        self.assertEqual(report['coefficients']['200']['b_M'],1.)
        self.assertEqual(report['coefficients']['200']['n_valid'],3)
        self.assertEqual(report['coefficients']['200']['n_excluded'],1)
        self.assertEqual(report['coefficients']['1800']['status'],'unavailable')
        self.assertNotIn('labels',str(report))

    def test_grid_bounds_ties_reuse_and_confirmation_dedup(self):
        from scripts.scdh_spectralz_study import phase_a,phase_b,confirmation,rank_candidates
        jobs=phase_a()
        self.assertEqual(len(jobs),22)
        scores=[dict(j,correct=100) for j in jobs if j['arm']=='spectral']
        top=rank_candidates(scores)[:2]
        self.assertEqual([(r['modes'],r['rho']) for r in top],[(200,.1),(200,.3)])
        later=phase_b(top)
        self.assertEqual(len(later),8)
        self.assertFalse({j['slot'] for j in jobs}&{j['slot'] for j in later})
        self.assertTrue(all(j['alpha_h']!=.03 for j in later))
        self.assertEqual(len(confirmation(top[0])),9)
        top[0]['alpha_h']=.01
        self.assertEqual(len(confirmation(top[0])),12)
        self.assertEqual(len({j['slot'] for j in confirmation(top[0])}),12)

    def test_cluster_bootstrap_keeps_all_corruptions_and_seeds_of_object_together(self):
        import numpy as np
        from scripts.analyze_scdh_spectralz import paired_metrics
        baseline=np.zeros((3,15,4),dtype=bool)
        winner=baseline.copy();winner[:,:,0]=True
        report=paired_metrics(winner,baseline)
        self.assertEqual(report['delta_pp'],25.)
        self.assertEqual(report['corrected'],45)
        self.assertEqual(report['harmed'],0)
        self.assertEqual(report['bootstrap_repetitions'],10000)
        self.assertEqual(report['seed_delta_pp'],[25.]*3)
        self.assertEqual(report['object_cluster_ci95_pp'],[0.,75.])
        with self.assertRaises(ValueError):paired_metrics(winner,baseline[:2])


class OperatorTests(unittest.TestCase):
    def test_cumulative_approval_cannot_restart_or_accept_Drive_consent(self):
        from scripts.run_scdh_spectralz import budget
        from scripts.scdh_spectralz_study import canonical_sha
        from pathlib import Path
        import tempfile
        plan={'cumulative_wall_cap_seconds':14400,'study':'fixed'}
        approval={'approved':True,'scope':'technical+calibration+A+B+conditional-C+timing',
                  'plan_sha256':canonical_sha(plan),'user_message':'Explicit scientific GPU phase approval'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(budget(root,plan,approval,now=1000),15400)
            self.assertEqual(budget(root,plan,approval,now=2000),15400)
            with self.assertRaises(ValueError):budget(root,plan,approval,now=15401)
            approval['scope']='Drive consent'
            with self.assertRaises(ValueError):budget(root,plan,approval,now=2000)

    def test_failure_archive_excludes_runtime_basis_and_rejects_symlink(self):
        from scripts.run_scdh_spectralz import archive_slot
        from pathlib import Path
        import tempfile,zipfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);slot=root/'slot';slot.mkdir()
            (slot/'failure.json').write_text('{}')
            (slot/'runtime.spectral_basis.npz').write_bytes(b'large runtime-only basis')
            path=archive_slot(slot,root/'failed.zip')
            with zipfile.ZipFile(path) as archive:self.assertEqual(archive.namelist(),['failure.json'])
            (slot/'bad').symlink_to('/tmp')
            with self.assertRaises(ValueError):archive_slot(slot,root/'bad.zip')

    def test_cpu_controller_freezes_top2_reuses_A_and_deduplicates_C(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import run_scdh_spectralz as operator
        from scripts.scdh_spectralz_study import canonical_sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=dict(root=str(root),cumulative_wall_cap_seconds=14400,drive_root=str(root/'drive'))
            approval=dict(approved=True,scope=operator.SCOPE,plan_sha256=canonical_sha(plan),user_message='GPU scientific phase test fixture')
            seen=[]
            def execute(plan,j,deadline,manifest):
                seen.append(j)
                count=128 if j['stage']=='confirmation' else 64 if j['stage']=='development' else 32
                outcomes=[[False]*count for _ in range(15)]
                correct=0
                if j['arm']=='spectral':outcomes[0][0]=True;correct=1
                report=dict(j,status='accepted',correct=correct,object_ids=list(range(count)),labels=[0]*count,
                    corruptions=list(range(15)),calibration_binding={},batch_guards=[],outcomes={'probability_mean':outcomes})
                if j['arm']=='calibration':
                    report['calibration_observations']=[dict(M_max=m,ratios=[1.]*32) for m in operator.study.MODES for _ in range(12)]
                    path=root/'slots'/j['slot']/'accepted.json';path.parent.mkdir(parents=True);path.write_text('{}')
                return report
            with patch.object(operator,'validate_plan',return_value=(root,{})),patch.object(operator,'execute_job',side_effect=execute),patch.object(operator.operations,'copy_verified'):
                operator.run(plan,approval,'all')
                self.assertEqual(sum(j['stage']=='development' for j in seen),30)
                self.assertEqual(sum(j['stage']=='confirmation' for j in seen),9)
                self.assertEqual(sum(j['stage']=='timing' for j in seen),8)
                self.assertEqual(len(seen),len({j['slot'] for j in seen}))
                self.assertEqual(json.loads((root/'frozen_winner.json').read_text())['winner']['modes'],200)
                count=len(seen);operator.run(plan,approval,'all');self.assertEqual(len(seen),count)

class PreparationTests(unittest.TestCase):
    def test_task_seal_inherits_bulk_and_preserves_recoverable_metadata(self):
        import json,zipfile
        from unittest.mock import patch
        from tests.test_prepare_chamfer_variants_assets import AssetBuilderTests
        from scripts import prepare_scdh_spectralz as preparation
        fixture=AssetBuilderTests();fixture.setUp()
        try:
            for name in ('operations.py','analyze.py','campaign.py'):
                fixture.write(fixture.repository/'result/preparation/chamfer_depth_tuning_20261030'/name,'pass\n')
            fixture.write(fixture.repository/'result/preparation/chamfer_depth_transfer_20261031/operator/deploy_assets_template.py','pass\n')
            preparation.register(fixture.prep,[])
            with patch.object(preparation,'REPO',fixture.repository),patch.object(preparation.assets,'discover_sources',return_value=['tta_gsd_v2.py']),patch.object(preparation.subprocess,'check_output',return_value='fake'):
                receipt=preparation.seal(fixture.prep,fixture.base,fixture.base_sha)
                with self.assertRaises(FileExistsError):preparation.seal(fixture.prep,fixture.base,fixture.base_sha)
            manifest=json.loads((fixture.prep/'assets_v1/manifest.json').read_text())
            self.assertEqual(manifest['files']['checkpoints/huge.pth'],fixture.base_manifest['files']['checkpoints/huge.pth'])
            self.assertEqual(receipt['status'],'sealed_CPU_only')
            with zipfile.ZipFile(fixture.prep/'assets_v1/source_snapshot.zip') as archive:
                self.assertIn('study/registration.json',archive.namelist())
                self.assertIn('source_snapshot/result/preparation/chamfer_depth_tuning_20261030/analyze.py',archive.namelist())
        finally:fixture.doCleanups()

if __name__=='__main__':unittest.main()
