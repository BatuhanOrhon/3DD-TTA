"""Narrow Scan/CurveNet/H proof, independent of the unchanged old V1 schema."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chamfer_calibration_report import validate_calibration_report
from legacy_trajectory import rng_hash
from research_artifacts import CORRUPTIONS
from tests import test_semantic_trajectory as fixtures
from tests.test_chamfer_calibration_report import valid_report
from tests import test_chamfer_calibration as archive_fixtures
from tests.scan_curvenet_fixtures import valid_curve_report


class CurveCalibrationContracts(unittest.TestCase):
    def modules(self):
        self.assertIsNotNone(importlib.util.find_spec('scan_curvenet_entropy_report'))
        self.assertIsNotNone(importlib.util.find_spec('scripts.calibrate_scan_curvenet_entropy'))
        import scan_curvenet_entropy_report as report
        from scripts import calibrate_scan_curvenet_entropy as collector
        return report,collector

    def rows(self):
        return [dict(original_id=index,corruption=corruption,loop_count=count,combined_update_ratio=2.,
                     chamfer_update_norm=1.,candidate_update_norm=2.,local_cosine=.4,style_cosine=.3,
                     local_candidate_grad_norm=1.,style_candidate_grad_norm=1.,teacher_decoded_grad_norm=1.)
                for corruption in CORRUPTIONS for index in range(70)
                for count in ((15,23) if corruption=='background' else (5,8))]

    def test_scoped_h_coefficient_uses_exact2100coverage_and_old_v1_stays_strict(self):
        report,collector=self.modules()
        value=collector.coefficient_report(self.rows(),list(range(70)))
        self.assertEqual(value['coefficients'],{'semantic_h_curvenet':.05})
        self.assertEqual(value['ratios']['semantic_h_curvenet']['count'],2100)
        with self.assertRaises(ValueError):validate_calibration_report(value)
        validate_calibration_report(valid_report())
        with self.assertRaises(ValueError):report.validate_report(valid_report())
        for bad in (self.rows()[:-1],self.rows()+self.rows()[:1]):
            with self.assertRaisesRegex(ValueError,'coverage'):collector.coefficient_report(bad,list(range(70)))
        bad=self.rows();bad[0]['teacher_decoded_grad_norm']=0
        with self.assertRaisesRegex(ValueError,'nonzero'):collector.coefficient_report(bad,list(range(70)))

    def test_native_proof_validator_rejects_zero_or_missing_positive_path_and_mutation(self):
        report,_=self.modules()
        proof=dict(corruption='gaussian',loop_count=5,objects=70,rng_unchanged=True,leaf_grad_fields_unchanged=True,
                   zero_local_nonzero=0,zero_style_nonzero=0,zero_local_finite=True,zero_style_finite=True,
                   zero_decoded_nonzero=0,zero_decoded_finite=True,
                   h_local_min_norm=1.,h_style_min_norm=1.,h_decoded_min_norm=1.)
        report.validate_native_probe(proof)
        for key,value in [('zero_local_nonzero',1),('zero_decoded_nonzero',1),('h_style_min_norm',0.),('h_decoded_min_norm',float('nan')),
                          ('rng_unchanged',False),('leaf_grad_fields_unchanged',False)]:
            changed=dict(proof,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):report.validate_native_probe(changed)

    def test_h_only_observer_checks_actual_zero_and_nonzero_paths_without_grad_or_rng_writes(self):
        _,collector=self.modules()
        vae,head,_,plain,_,local,style=fixtures.SemanticContracts().fixture(15)
        def classify(decoded):return {'curvenet':plain(decoded)['pointmae']}
        observer=collector.CurveEntropyCalibrationObserver(vae,{'curvenet':head},classify,10,15,
            classify_semantic_decoded=classify,kind='marginal_entropy',guidance_head='curvenet',
            reference_logits=torch.zeros(2,15),reference_record={'forward_calls':1},original_ids=[3,7],corruption='gaussian')
        before=rng_hash();observer.clean(local,style,2,970)
        predicted=2*local+.2*style.mean(1,keepdim=True)
        loss=observer.clean(predicted,style,4,950)
        chamfer=predicted.square().sum()
        with patch('chamfer_guidance_variants.reference_density_weights',side_effect=AssertionError('weighted calculation')):
            observer.guidance_components(ch_loss=chamfer,semantic_loss=loss,noisy_local=local,style=style,
                pred_latent=predicted,h0=predicted.detach(),loop_count=5,timestep=950,eta=.01,gamma=.01,p=.95)
        self.assertEqual(rng_hash(),before);self.assertIsNone(local.grad);self.assertIsNone(style.grad)
        self.assertEqual(len(observer.calibration_rows),2)
        self.assertEqual(len(observer.native_probes),1)
        self.assertTrue(all(row['teacher_decoded_grad_norm']>0 for row in observer.calibration_rows))
        self.assertEqual(observer.native_probes[0]['zero_local_nonzero'],0)
        self.assertEqual(observer.native_probes[0].get('zero_decoded_nonzero'),0)
        (chamfer+0*loss).backward()
        self.assertGreater(float(local.grad.abs().sum()),0.)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_sealed_collector_summary_requires_native_proofs_without_reading_labels(self):
        report,collector=self.modules()
        with tempfile.TemporaryDirectory() as directory:
            path=archive_fixtures.CalibrationContracts().bundle(Path(directory),'scanobjectnn_c',self.rows())
            config=json.loads((path/'config.json').read_text());teacher=valid_curve_report()['teacher']
            config.update(backbone='curvenet',backbones=['curvenet'],checkpoint=teacher['checkpoint'],graph=None,
                semantic_guidance=dict(head='curvenet',kind='marginal_entropy',loss_reduction='sum',teacher_checkpoint=teacher['checkpoint'],
                                       teacher_preprocessing=teacher['preprocessing'],teacher_input_layout='bcn'),
                curvenet_entropy_calibration={'purpose':report.PURPOSE})
            config['guidance_protocol'].update(loss_weights=dict(chamfer=1.,spectral=0.,invariant=0.),skip_inactive_components=True)
            config['asset_manifest']['files'][teacher['checkpoint']]=teacher['checkpoint_identity']
            for name in ('scripts/calibrate_scan_curvenet_entropy.py','scan_curvenet_entropy_report.py'):
                source=path/'source_snapshot'/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(b'x')
                config['executed_source_files']['repo/'+name]=dict(bytes=1,sha256=hashlib.sha256(b'x').hexdigest())
            (path/'config.json').write_text(json.dumps(config))
            probes=valid_curve_report()['native_probes']
            for corruption in CORRUPTIONS:
                target=path/'batches'/corruption/'batch0000/diagnostics.json'
                value=json.loads(target.read_text());value['guidance_metrics']['semantic_loss_calls']=2
                value['trajectory']['curvenet_entropy_calibration']=dict(
                    rows=[row for row in self.rows() if row['corruption']==corruption],
                    native_probes=[row for row in probes if row['corruption']==corruption],gradient_calls=8,
                    labels_used_for_selection=False,weighted_calls=0,kl_calls=0)
                target.write_text(json.dumps(value))
            def reseal():
                manifest={str(file.relative_to(path)):dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                          for file in path.rglob('*') if file.is_file() and file.name!='bundle_manifest.json'}
                (path/'bundle_manifest.json').write_text(json.dumps(manifest))
            reseal();value=collector.summarize_bundle(path)
            self.assertEqual(value['coefficients'],{'semantic_h_curvenet':.05})
            self.assertEqual(len(value['native_probes']),30)
            target=path/'batches/gaussian/batch0000/diagnostics.json';bad=json.loads(target.read_text())
            bad['trajectory']['curvenet_entropy_calibration']['native_probes'][0]['zero_local_nonzero']=1
            target.write_text(json.dumps(bad));reseal()
            with self.assertRaisesRegex(ValueError,'zero-weight'):collector.summarize_bundle(path)


if __name__=='__main__':unittest.main()
