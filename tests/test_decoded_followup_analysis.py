"""Checks for the predeclared repeat/transfer analysis gates."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'knowledge/evidence/decoded_physical_followup_20261104/analyze_followup.py'
SPEC = importlib.util.spec_from_file_location('followup_analysis', FILE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def npz(**values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


class FakeBundle:
    def __init__(self, offset=0.):
        self.diagnostic = dict(paired_state=dict(sha256='cache'), prepared_x_sha256='X',
                               input_rng_before_sha256='before', reconstruction_rng_after_sha256='after',
                               scheduler={}, trajectory=dict(initial_hashes={}, initial_rng_sha256='initial'))
        self.records = [dict(loop_count=i, full_repeat_decoder_max_absolute_difference=0.,
                             full_repeat_decoder_rms_difference=0., detached_decoder_max_absolute_difference=0.,
                             detached_decoder_rms_difference=0., full_repeat_decoder_output_requires_grad=True)
                        for i in (2, 4, 6)]
        self.files = {'predictions.npz': npz(trajectory_logits=np.full((2, 4, 40), offset),
                                            index=np.array([44, 91]), candidate_loop_counts=np.array([2, 4, 6, 8])),
                      'reference.npz': npz(prepared_x=np.zeros((2, 2048, 3)), initial_h0_xyz=np.zeros((2, 2048, 3))),
                      'decoded_first_checkpoint.npz': npz(decoded_xyz=np.zeros((2, 2048, 3)))}

    def json(self, name):
        return self.records if name.endswith('probe_records.json') else self.diagnostic

    def read(self, name):
        return self.files[Path(name).name]

    def close(self):
        pass


class FollowupAnalysisTests(unittest.TestCase):
    def test_all_pairs_detect_third_repeat_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(i) for i in range(6)]
            for path in paths:
                path.write_text(path.name)
            def opened(path):
                i = int(Path(path).name)
                config = dict(selected_original_ids=[44, 91], corruptions=['gaussian'],
                              executed_source_files={}, native_extensions={}, native_bootstrap={}, guidance_protocol={}, lion_state_before={},
                              classifier_state_before_sha256='head', runtime={},
                              backend_policy='legacy' if i < 3 else 'controlled')
                return FakeBundle(.1 if i == 2 else 0.), config
            with patch.object(analysis, 'open_bundle', side_effect=opened):
                result = analysis.summarize_p0(paths)
            legacy, controlled = result['policies']['legacy'], result['policies']['controlled']
            self.assertEqual(len(legacy['pairs']), 3)
            self.assertEqual(sum(p['passed'] for p in legacy['pairs']), 1)
            self.assertFalse(legacy['cross_process_passed'])
            self.assertTrue(controlled['cross_process_passed'])
            self.assertEqual(len(legacy['pairs'][0]['logits']), 8)
            json.dumps(result, allow_nan=False)

    def test_mismatched_controls_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(i) for i in range(6)]
            for path in paths:
                path.write_text(path.name)
            def opened(path):
                i = int(Path(path).name)
                config = dict(selected_original_ids=[44, 91], corruptions=['gaussian'],
                              executed_source_files={}, native_extensions={}, native_bootstrap={}, guidance_protocol={}, lion_state_before={},
                              classifier_state_before_sha256='head' if i == 0 else 'changed', runtime={},
                              backend_policy='legacy' if i < 3 else 'controlled')
                return FakeBundle(), config
            with patch.object(analysis, 'open_bundle', side_effect=opened):
                with self.assertRaisesRegex(AssertionError, 'identity changed'):
                    analysis.summarize_p0(paths)


if __name__ == '__main__':
    unittest.main()
