"""Native-free contracts for the separately archived posthoc readout operator."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from classifier_backbones import FrozenClassifier
import legacy_trajectory as legacy
from scripts import eval_multibackbone_guidance as shared
from scripts import classify_shared_probe_readouts as readout

OPERATOR = Path(readout.__file__)


class ReadoutTests(unittest.TestCase):
    def fixture(self, root):
        root = root.resolve()
        capture, mechanism = root/'capture', root/'mechanism'
        capture.mkdir()
        mechanism.mkdir()
        report = dict(cases=[], artifacts={})
        def arrays(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            return readout.write_arrays(path, payload)
        def json_file(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            readout.write_json(path, payload)
        for corruption in readout.CORRUPTIONS:
            for batch in range(2):
                ids = readout.IDS[batch*2:batch*2+2]
                prefix = '%s/batch%04d' % (corruption, batch)
                raw = capture/'batches'/prefix
                loops = [9, 27] if corruption == 'background' else [2, 6]
                schedule = list(range(340, -1, -10)) if corruption == 'background' else list(range(70, -1, -10))
                frame = dict(index=np.asarray(ids), normalization_center=np.zeros((2, 1, 3), dtype=np.float32),
                             normalization_scale=np.ones((2, 1, 1), dtype=np.float32))
                frame_name = 'frame.npz'
                frame_id = arrays(raw/frame_name, frame)
                snapshots = [dict(loop_count=0, stage='frame', file=frame_name, identity=frame_id,
                    tensor_sha256={k: legacy.tensor_hash(torch.from_numpy(v)) for k, v in frame.items() if k != 'index'})]
                for loop in loops:
                    for stage in ('pre_prior', 'pre_decoder', 'decoded'):
                        name = '%02d_%s.npz' % (loop, stage)
                        snapshots.append(dict(loop_count=loop, stage=stage, timestep=schedule[loop-1], file=name,
                            identity=arrays(raw/name, dict(index=np.asarray(ids)))))
                shared_record = dict(schema_version=1, complete=True, original_ids=ids, checkpoint_steps=loops,
                                     completed_checkpoints=loops, snapshots=snapshots)
                json_file(raw/'shared_states.json', shared_record)
                json_file(raw/'diagnostics.json', dict(shared_states=readout.identity(raw/'shared_states.json'),
                                                     scheduler=dict(active_timesteps=schedule)))
                json_file(mechanism/prefix/'input.json', dict(original_ids=ids,
                    identities={name: readout.identity(raw/name) for name in ('shared_states.json', 'diagnostics.json')}))
                loop = loops[0]
                stem = readout.branch_key(corruption, batch, loop, 'zero', .5)
                arrays(mechanism/(stem+'_prediction.npz'), dict(decoded_xyz=np.zeros((2, 2048, 3), dtype=np.float32)))
                arrays(mechanism/(stem+'_state.npz'), dict(noisy_local_next=np.zeros((2, 8), dtype=np.float32)))
                json_file(mechanism/prefix/('loop%02d/directions.json' % loop), dict(status='complete_with_unavailable',
                    next_timestep=schedule[loop], branches=[dict(name='zero', fraction=.5, status='complete')],
                    unavailable=dict(S_phys='Failed native FD; retained')))
                report['cases'].append(dict(corruption=corruption, batch=batch, status='complete_with_unavailable'))
        report['artifacts'] = {str(p.relative_to(mechanism)): readout.identity(p) for p in mechanism.rglob('*') if p.is_file()}
        return capture, mechanism, report

    def test_fixed_keys_frame_coverage_and_tampered_raw_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture, mechanism, report = self.fixture(Path(temporary))
            jobs, coverage = readout.collect_jobs(capture, mechanism, report, tensor_hash=legacy.tensor_hash)
            self.assertEqual(len(jobs), 8)
            self.assertEqual(len(coverage), 224)
            self.assertEqual(sum(row['status'] == 'available' for row in coverage), 8)
            self.assertEqual(jobs[0]['metadata']['original_ids'], [459, 545])
            self.assertEqual(jobs[1]['metadata']['original_ids'], [1229, 1435])
            self.assertEqual(jobs[0]['frame']['normalization_scale'].shape, (2, 1, 1))
            self.assertTrue(any(row.get('reason') == 'Failed native FD; retained' for row in coverage))
            raw = capture/'batches/gaussian/batch0000/frame.npz'
            raw.write_bytes(raw.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                readout.collect_jobs(capture, mechanism, report, tensor_hash=legacy.tensor_hash)
        for args in (('gaussian', 0, 4, 'C', .5), ('gaussian', 0, 2, 'C', .25), ('original', 0, 2, 'C', .5)):
            with self.assertRaises(ValueError):
                readout.branch_key(*args)

    def test_tampered_cloud_and_immutable_unlabelled_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture, mechanism, report = self.fixture(Path(temporary))
            cloud = mechanism/'gaussian/batch0000/loop02/branch_zero_0p5_prediction.npz'
            cloud.write_bytes(cloud.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                readout.collect_jobs(capture, mechanism, report, tensor_hash=legacy.tensor_hash)
            output = Path(temporary)/'readout.npz'
            logits = torch.arange(80, dtype=torch.float32).reshape(2, 40)
            readout.write_arrays(output, dict(index=np.asarray([459, 545]), logits=logits, prediction=logits.argmax(-1)))
            logits.zero_()
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {'index', 'logits', 'prediction'})
                self.assertEqual(archive['logits'][1, -1], 79)
            with self.assertRaises(FileExistsError):
                readout.write_arrays(output, {})
            with self.assertRaises(ValueError):
                readout.safe_file(Path(temporary), '../escape')
            with self.assertRaises(ValueError):
                readout.validate_config({}, 'a'*64, Path('/assets'))
        ast.parse(OPERATOR.read_text(), feature_version=(3, 8))
        ast.parse(Path(__file__).read_text(), feature_version=(3, 8))

    def test_existing_classifier_helper_flags_frame_rng_and_input_preservation(self):
        class Model(torch.nn.Module):
            def classification_only(self, points, *, only_unmasked):
                self.flags = (torch.is_grad_enabled(), points.requires_grad, only_unmasked)
                return points.mean((1, 2))[:, None].expand(-1, 40)
        classifier = FrozenClassifier(Model(), 'pointmae', 40, input_layout='bnc')
        cloud = torch.arange(2*6*3, dtype=torch.float32).reshape(2, 6, 3)
        frame = dict(normalization_center=torch.ones(2, 1, 3), normalization_scale=torch.ones(2, 1, 1)*2)
        config = dict(dataset='modelnet40_c', backbone='pointmae')
        forwarded, sampled = [], []
        def geometry(value, dataset, scale, center, ops):
            forwarded.append((dataset, scale is frame['normalization_scale'], center is frame['normalization_center'], ops))
            return value.clone()*.1
        def fps(points, number):
            sampled.append(number)
            return points[:, :3].contiguous()
        helpers = SimpleNamespace(seed_inference=shared.seed_inference, classify_geometry=shared.classify_geometry,
                                  decoded_to_geometry=geometry)
        before_rng, before_cloud = legacy.rng_hash(), legacy.tensor_hash(cloud)
        result, points, guards = readout.classify_one(cloud, frame, classifier, config,
            shared=helpers, ops='existing_ops', fps=fps, legacy=legacy)
        self.assertEqual(result.shape, (2, 40))
        self.assertEqual(forwarded, [('modelnet40_c', True, True, 'existing_ops')])
        self.assertEqual(sampled, [1024])
        self.assertEqual(classifier.model.flags, (False, False, False))
        self.assertEqual(guards['classifier_forward_flags'], dict(global_grad_enabled=False, input_requires_grad=False,
                                                                 all_eval=True, all_parameters_frozen=True))
        self.assertEqual(legacy.rng_hash(), before_rng)
        self.assertEqual(legacy.tensor_hash(cloud), before_cloud)
        self.assertEqual(len(classifier._forward_pre_hooks), 0)
        classifier.model.training = True
        with self.assertRaises(ValueError):
            readout.classify_one(cloud, frame, classifier, config, shared=helpers, ops=None, fps=fps, legacy=legacy)


if __name__ == '__main__':
    unittest.main()
