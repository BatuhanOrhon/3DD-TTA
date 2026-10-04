"""Strict fixture validation and paired composition summaries."""
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from gsd_composition_protocol import _SPLIT, write_arm_bundle
from scripts.analyze_gsd_composition import analyze_root


def _write_arm(root, arm_id, predictions, *, corruption="gaussian",
               input_hash="a" * 64, indices=None,
               logits=None, labels=None, locked_identity="c" * 64,
               phase="diagnose", block_id="gaussian-seed0-attempt01"):
    labels = np.tile(np.asarray([1, 2, 3, 4] if labels is None else labels, dtype=np.int64), 16)
    predictions = np.tile(np.asarray(predictions, dtype=np.int64), 16)
    indices = list(indices) if indices is not None else _SPLIT["pilot_indices"][:64]
    if logits is None:
        logits = np.zeros((len(labels), 40), dtype=np.float32)
        logits[np.arange(len(labels)), predictions] = 3.0
    config = {"local_mode": "scd", "style_mode": "scd" if arm_id == "C_SCD" else "spectral",
              "local_scd_weight": 1.0, "local_spectral_weight": 1.0,
              "style_scd_weight": 1.0, "style_spectral_weight": 1.0,
              "norm_floor": 1e-12, "schema_version": 1}
    manifest = dict(
        method="gsd_guidance_composition_v1", arm_id=arm_id,
        corruption=corruption, seed=0, status="complete", original_indices=indices,
        input_sha256=input_hash, block_id=block_id, phase=phase,
        runtime_fingerprint="b" * 64, experiment_fingerprint="same-experiment",
        config=config, total_runtime_seconds=2.5, peak_gpu_memory_mb=400.0,
    )
    if locked_identity is not None:
        manifest["locked_config_fingerprint"] = locked_identity
    path = root / phase / corruption / arm_id
    write_arm_bundle(path, manifest, labels, predictions, logits,
                     [{"step_index": 0, "timestep": 999, "style_gradient_cosine": -0.2}])
    return path


class AnalyzeCompositionTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("analyze-composition-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_reports_per_corruption_accuracy_transitions_oracle_and_runtime(self):
        root = self.root
        _write_arm(root, "C_SCD", [1, 2, 0, 0])
        _write_arm(root, "R_SG", [1, 0, 3, 0])

        report = analyze_root(root)

        self.assertEqual(report["validation"]["complete_arms"], 2)
        self.assertAlmostEqual(report["accuracy"]["diagnose"]["R_SG"]["by_corruption"]["gaussian"]["accuracy"], 0.5)
        transition = report["paired_transitions"]["diagnose"]["C_SCD__vs__R_SG"]["gaussian"]
        self.assertEqual(transition["both_correct"], 16)
        self.assertEqual(transition["left_only_correct"], 16)
        self.assertEqual(transition["right_only_correct"], 16)
        self.assertEqual(transition["both_wrong"], 16)
        self.assertEqual(transition["oracle_accuracy"], 0.75)
        uncertainty = report["uncertainty"]["diagnose"]["C_SCD__vs__R_SG"]["gaussian"]
        self.assertEqual(uncertainty["paired_accuracy_delta"]["resamples"], 2000)
        self.assertEqual(uncertainty["paired_accuracy_delta"]["n_object_ids"], 64)
        self.assertIn("runtime", report)
        self.assertIn("gradient_geometry", report)

    def test_gradient_geometry_keeps_each_corruption_separate(self):
        _write_arm(self.root, "C_SCD", [1, 2, 3, 4], corruption="gaussian")
        _write_arm(self.root, "C_SCD", [1, 2, 3, 4], corruption="background",
                   block_id="background-seed0-attempt01")

        report = analyze_root(self.root)

        geometry = report["gradient_geometry"]["diagnose"]["C_SCD"]
        self.assertEqual(set(geometry), {"gaussian", "background"})
        self.assertEqual(geometry["gaussian"]["metrics"]["style_gradient_cosine"]["count"], 1)
        self.assertEqual(geometry["background"]["metrics"]["style_gradient_cosine"]["count"], 1)

    def test_rejects_duplicate_original_indices_and_nonfinite_logits(self):
        duplicate_root = self.root / "duplicate"
        duplicate_root.mkdir()
        duplicate_indices = list(_SPLIT["pilot_indices"][:64])
        duplicate_indices[1] = duplicate_indices[0]
        duplicate = _write_arm(duplicate_root, "C_SCD", [1, 2, 3, 4], indices=duplicate_indices)
        with self.assertRaises(ValueError):
            analyze_root(duplicate)
        nonfinite_root = self.root / "nonfinite"
        nonfinite_root.mkdir()
        logits = np.zeros((64, 40), dtype=np.float32)
        logits[0, 1] = float("nan")
        _write_arm(nonfinite_root, "C_SCD", [1, 2, 3, 4], logits=logits)
        with self.assertRaises(ValueError):
            analyze_root(nonfinite_root)

    def test_rejects_counts_predictions_and_mixed_paired_inputs(self):
        root = self.root
        _write_arm(root, "C_SCD", [1, 2, 3, 4])
        _write_arm(root, "R_SG", [1, 2, 3, 4], input_hash="d" * 64)
        with self.assertRaises(ValueError):
            analyze_root(root)

    def test_missing_identity_never_pairs_and_mismatched_labels_are_rejected(self):
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        _write_arm(incomplete, "C_SCD", [1, 2, 3, 4], locked_identity=None)
        _write_arm(incomplete, "R_SG", [1, 2, 3, 4], locked_identity=None)
        report = analyze_root(incomplete)
        self.assertEqual(report["validation"]["paired_blocks"], 0)
        self.assertEqual(report["validation"]["complete_arms_without_pair_identity"], 2)
        self.assertEqual(report["paired_transitions"], {})

        mismatch = self.root / "label-mismatch"
        mismatch.mkdir()
        _write_arm(mismatch, "C_SCD", [1, 2, 3, 4])
        _write_arm(mismatch, "R_SG", [1, 2, 3, 4], labels=[0, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "mismatched labels"):
            analyze_root(mismatch)

    def test_exact_reference_reuse_pairs_across_phases_without_pooling_accuracy(self):
        root = self.root
        shared = dict(block_id="same-prepared-block", input_hash="a" * 64,
                      indices=_SPLIT["pilot_indices"][:64], locked_identity="c" * 64)
        _write_arm(root, "C_SCD", [1, 2, 3, 4], phase="diagnose", **shared)
        _write_arm(root, "R_SG", [1, 0, 3, 0], phase="routing", **shared)

        report = analyze_root(root)

        self.assertIn("diagnose", report["accuracy"])
        self.assertIn("routing", report["accuracy"])
        self.assertIn("cross_phase:diagnose__routing", report["paired_transitions"])
        transition = report["paired_transitions"]["cross_phase:diagnose__routing"]["C_SCD__vs__R_SG"]["gaussian"]
        self.assertEqual(transition["n_examples"], 64)


if __name__ == "__main__":
    unittest.main()
