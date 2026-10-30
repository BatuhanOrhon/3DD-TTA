"""Opt-in Scan CurveNet teacher; legacy semantic defaults remain locked."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from classifier_backbones import FrozenClassifier
from legacy_trajectory import rng_hash
from scripts import eval_multibackbone_guidance as runner
from semantic_trajectory import SemanticTrajectoryObserver, semantic_loss
from tests import test_semantic_trajectory as fixtures
from tests.scan_curvenet_fixtures import valid_curve_report


class CurveTeacherContracts(unittest.TestCase):
    def args(self, *extra):
        return ['--preview','--dataset','scanobjectnn_c','--dataset-root','/data','--label-path','/data/labels.npy',
                '--asset-root','/assets','--result-root','/results',*extra]

    def test_curvenet_is_explicit_scan_h_only_and_preserves_probability_mean(self):
        args=self.args('--chamfer-experiment','semantic','--semantic-kind','marginal_entropy',
                       '--semantic-guidance-head','curvenet')
        config=runner.configurations(args)[0]
        self.assertEqual(config['semantic_guidance']['head'],'curvenet')
        self.assertEqual(config['semantic_guidance']['teacher_checkpoint'],'checkpoints/3deformrs_curvenet_scanobjectnn_better.t7')
        self.assertEqual(config['semantic_guidance']['teacher_input_layout'],'bcn')
        self.assertEqual(config['trajectory_selection']['normal_loop_counts'],[3,5,8,10])
        self.assertIsNone(config['graph'])
        for extra in (['--dataset','modelnet40_c'],['--semantic-kind','kl'],['--backbones','pointnext']):
            with self.subTest(extra=extra),self.assertRaises(ValueError):runner.configurations([*args,*extra])
        with self.assertRaises(ValueError):runner.configurations(self.args('--semantic-guidance-head','curvenet'))

    def test_default_pointmae_preview_matches_frozen_pre_followup_config(self):
        frozen=Path(runner.__file__).parents[1]/'result/preparation/chamfer_guidance_variants_20261030/revision_3/assets_v1/source_snapshot/scripts/eval_multibackbone_guidance.py'
        if not frozen.exists():self.skipTest('Historical sealed source exists only in the research workspace')
        args=self.args('--chamfer-experiment','semantic','--semantic-kind','marginal_entropy','--backbones','pointmae',
                       '--pointmae-config','/pointmae.yaml','--pointmae-checkpoint','/pointmae.pth')
        before=subprocess.run([sys.executable,'-S',str(frozen),*args],capture_output=True,text=True,check=True)
        self.assertEqual(runner.configurations(args),json.loads(before.stdout))

    def test_curvenet_normalized_fps_path_has_input_and_latent_style_gradients(self):
        class Head(torch.nn.Module):
            def forward(self,x):
                values=x.square().mean(-1)
                return torch.cat([values,2*values,3*values,4*values,5*values],1)
        head=FrozenClassifier(Head(),'curvenet',15,input_layout='bcn')
        geometry=torch.linspace(-2,3,2*2048*3).reshape(2,2048,3).requires_grad_()
        # Break affine symmetry so normalization does not erase every direction.
        geometry=geometry+torch.arange(2048).remainder(17).reshape(1,-1,1)*.001
        config={'curvenet':{'backbone':'curvenet','dataset':'scanobjectnn_c'}}
        before=rng_hash()
        logits=runner.classify_semantic_geometry(geometry,{'curvenet':head},config,lambda points,count:points[:,:count],guidance_head='curvenet')
        grad,=torch.autograd.grad(semantic_loss(logits['curvenet'],torch.zeros(2,15),'marginal_entropy'),(geometry,))
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        self.assertEqual(before,rng_hash());self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
        vae,base,_,plain,callback,local,style=fixtures.SemanticContracts().fixture(15)
        def classify(decoded):return {'curvenet':plain(decoded)['pointmae']}
        observer=SemanticTrajectoryObserver(vae,{'curvenet':base},classify,10,15,
            classify_semantic_decoded=classify,kind='marginal_entropy',reference_logits=torch.zeros(2,15),
            reference_record={'forward_calls':1},guidance_head='curvenet')
        observer.clean(local,style,2,970)
        loss=observer.clean(local,style,4,950)
        for value in torch.autograd.grad(loss,(local,style)):
            self.assertTrue(torch.isfinite(value).all());self.assertGreater(float(value.abs().sum()),0)

    def test_curvenet_requires_its_own_proof_and_rejects_pointmae_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';path.write_text(json.dumps(valid_curve_report()))
            args=self.args('--chamfer-experiment','semantic','--semantic-kind','marginal_entropy',
                           '--semantic-guidance-head','curvenet','--semantic-weight','.05',
                           '--calibration-report',str(path),'--calibration-report-sha256',runner.source.file_identity(path)['sha256'])
            config=runner.configurations(args)[0]
            self.assertEqual(config['calibration_reference']['coefficient_key'],'semantic_h_curvenet')
            self.assertEqual(config['calibration_reference']['teacher'],valid_curve_report()['teacher'])
            with self.assertRaises(ValueError):runner.configurations([*args,'--semantic-weight','.1'])
            path.write_text(json.dumps(fixtures.SemanticContracts().calibration_report()))
            with self.assertRaises(ValueError):runner.configurations([*args,'--calibration-report-sha256',runner.source.file_identity(path)['sha256']])


if __name__=='__main__':unittest.main()
