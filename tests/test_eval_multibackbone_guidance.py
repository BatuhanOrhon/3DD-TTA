"""CPU contracts for shared reconstruction and isolated frozen heads."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts import eval_multibackbone_guidance as runner
from legacy_trajectory import isolated_probe, rng_hash


class GuidanceContracts(unittest.TestCase):
    def args(self, *extra):
        return ["--dataset", "scanobjectnn_c", "--dataset-root", "/data", "--label-path", "/data/label.npy",
                "--asset-root", "/assets", "--result-root", "/results", *extra]

    def test_preview_without_native_imports_preserves_separate_method_and_locked_protocol(self):
        result = subprocess.run([sys.executable, "-S", str(Path(runner.__file__)), *self.args("--preview")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        configs = json.loads(result.stdout)
        self.assertEqual({c["backbone"] for c in configs}, {"curvenet", "pointnext"})
        self.assertTrue(all(c["method"] == runner.METHOD and c["method"] != "source_only" for c in configs))
        self.assertEqual(configs[0]["batch_size"], 70)
        self.assertEqual(configs[0]["guidance_protocol"]["M_max"], 800)
        self.assertFalse(configs[0]["guidance_protocol"]["use_static_style"])
        self.assertEqual(configs[0]["n_examples"], 581)

    def operations(self):
        source = Path(runner.source.REPO / "utilities_3dd_tta.py").read_text()
        names = {"normalize", "unnormalize_data", "rotate_pointcloud", "rotateback_pointcloud"}
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"torch": torch, "np": np}
        exec(compile(tree, "audited_utilities", "exec"), namespace)
        return SimpleNamespace(**namespace)

    def test_actual_normalization_rotation_order_and_scan_inverse_keep_full_geometry(self):
        ops = self.operations()
        raw = torch.tensor([[[1., 2., 3.], [5., 8., 11.], [3., 4., 7.]]])
        expected, center, scale = ops.normalize(raw)
        calls = []

        def upsample(values, count):
            calls.append((values.copy(), count))
            return np.resize(values, (1, 2048, 3))

        ops.upsample_all = upsample
        seen = []

        def reconstruct(**kwargs):
            seen.append(kwargs)
            return kwargs["x"].clone(), {"mean_raw_spectral": 1., "mean_raw_chamfer": 2.}

        geometry, metrics = runner.reconstruct_geometry(raw, "scanobjectnn_c", "background", None, None, reconstruct, ops, device="cpu")
        np.testing.assert_array_equal(calls[0][0], expected.numpy())
        self.assertEqual(calls[0][1], 2048)
        torch.testing.assert_close(geometry, torch.from_numpy(np.resize(raw.numpy(), (1, 2048, 3))))
        self.assertEqual(geometry.shape, (1, 2048, 3))
        self.assertEqual(seen[0]["steps_back_local"], 30)
        self.assertEqual(seen[0]["M_max"], 800)
        self.assertNotIn("trajectory_observer", seen[0])
        self.assertFalse(seen[0]["use_static_style"])
        expected_input = ops.rotate_pointcloud(torch.from_numpy(np.resize(expected.numpy(), (1, 2048, 3))) * 3.3885)
        torch.testing.assert_close(seen[0]["x"], expected_input)
        modelnet, _ = runner.reconstruct_geometry(raw, "modelnet40_c", "gaussian", None, None, reconstruct, ops, device="cpu")
        torch.testing.assert_close(modelnet, ops.normalize(torch.from_numpy(np.resize(expected.numpy(), (1, 2048, 3))))[0])
        self.assertEqual(seen[-1]["steps_back_local"], 10)

    def test_each_head_preprocess_is_rng_isolated_and_does_not_share_fps(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))
            def forward(self, points, *, features=None):
                self.counter.add_(1)
                return torch.rand(len(points), 2)
        heads = {name: Head().eval() for name in runner.BACKBONES}
        configs = {name: {"backbone": name, "dataset": "modelnet40_c"} for name in heads}
        calls = []
        def fps(points, count):
            calls.append(count)
            random.random(); np.random.random()
            return points[:, torch.randperm(points.shape[1])[:4]]
        geometry = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3)
        before = rng_hash()
        logits, diagnostics = runner.classify_geometry(geometry, heads, configs, fps)
        self.assertEqual(calls, [1024, 1024])
        self.assertEqual(before, rng_hash())
        self.assertTrue(all(head.counter.item() == 0 for head in heads.values()))
        self.assertTrue(all(value["rng_unchanged"] for value in diagnostics.values()))
        self.assertTrue(torch.equal(logits["curvenet"], logits["pointnext"]))

    def test_head_initialization_and_seed_boundary_remove_initialization_draws(self):
        configs = {name: {"backbone": name} for name in runner.BACKBONES}
        def loader(config, manifest):
            random.random(); np.random.random(); torch.rand(7)
            return torch.nn.Linear(3, 2).eval()
        before = rng_hash()
        heads, guard = runner.initialize_heads(configs, {}, loader)
        self.assertEqual(before, rng_hash())
        self.assertTrue(guard["rng_unchanged"])
        runner.seed_inference(3)
        expected = (random.random(), np.random.random(), torch.rand(1))
        loader({}, {})
        runner.seed_inference(3)
        observed = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(expected[:2], observed[:2])
        self.assertTrue(torch.equal(expected[2], observed[2]))

    def test_failed_head_restores_rng_and_buffers_and_retains_partial_counts(self):
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.register_buffer("counter", torch.zeros(1))
            def forward(self, points, *, features=None):
                self.counter.add_(1); torch.rand(1); np.random.random(); random.random()
                raise RuntimeError("classifier failed")
        head = Broken().eval()
        before = rng_hash()
        with self.assertRaisesRegex(RuntimeError, "classifier failed"):
            runner.classify_geometry(torch.ones(1, 4, 3), {"curvenet": head},
                                     {"curvenet": {"backbone": "curvenet", "dataset": "modelnet40_c"}}, lambda points, count: points)
        self.assertEqual(rng_hash(), before)
        self.assertEqual(head.counter.item(), 0)
        rows = []
        config = {"run_id": "test", "seed": 0, "dataset": "modelnet40_c"}
        runner.update_row(rows, config, "gaussian", 70, 40, 3., 100., "partial")
        runner.update_row(rows, config, "gaussian", 70, 40, 3., 100., "failed")
        self.assertEqual((len(rows), rows[0]["n_examples"], rows[0]["n_correct"], rows[0]["status"]), (1, 70, 40, "failed"))

    def test_matched_source_binds_checkpoint_preprocessing_counts_and_dataset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = runner.configurations(self.args("--preview", "--asset-root", str(root), "--corruptions", "gaussian"))[0]
            files = {"label.npy": {"sha256": "a" * 64, "bytes": 10, "shape": [581]},
                     "data_gaussian_5.npy": {"sha256": "b" * 64, "bytes": 20, "shape": [581, 1024, 3]}}
            dataset = {"n_examples": 581, "split": "main_split", "files": files}
            baseline = root / "source_manifest.json"
            baseline.write_text(json.dumps({"datasets": {"scanobjectnn_c": dataset}}))
            row = dict(dataset="scanobjectnn_c", backbone="curvenet", checkpoint_sha256="c" * 64,
                       preprocessing=config["source_classifier_preprocessing"], archive_sha256="d" * 64,
                       accuracy_percent=32., total_examples=581 * 15)
            validation = dict(status="complete", macro=[row], source_manifest_sha256=runner.source.file_identity(baseline)["sha256"])
            validation_path = root / "validation.json"
            validation_path.write_text(json.dumps(validation))
            manifest = {"files": {"source_manifest.json": runner.source.file_identity(baseline), "validation.json": runner.source.file_identity(validation_path),
                                  config["checkpoint"]: {"sha256": "c" * 64}}, "datasets": {"scanobjectnn_c": dataset}}
            guidance = dict(matched_source_results="validation.json", matched_source_manifest="source_manifest.json")
            self.assertEqual(runner.matched_source(config, manifest, guidance), row)
            for key, invalid in (("checkpoint_sha256", "e" * 64), ("preprocessing", "different"), ("total_examples", 70)):
                modified = dict(validation, macro=[dict(row, **{key: invalid})])
                validation_path.write_text(json.dumps(modified))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.matched_source(config, manifest, guidance)
            validation_path.write_text(json.dumps(validation))
            dataset["files"]["data_gaussian_5.npy"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "data differs"):
                runner.matched_source(config, manifest, guidance)

    def test_batch_shards_are_recoverable_and_drive_receipt_requires_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = runner.source.RunBundle.create(root, "test", {"dataset": "modelnet40_c", "method": runner.METHOD, "seed": 0}, "test")
            arrays = {"index": np.array([0, 1]), "label": np.array([1, 0]), "prediction": np.array([1, 1]), "logits": np.array([[0., 2.], [0., 2.]])}
            files = runner.write_batch_shard(bundle, "gaussian", 0, arrays, {"geometry_sha256": "a" * 64, "rng_unchanged": True})
            receipt = runner.copy_batch_shard(bundle, root / "drive", "gaussian", 0, files)
            receipt_path = Path(receipt["receipt_path"])
            self.assertEqual(json.loads(receipt_path.read_text())["status"], "complete")
            for name, identity in receipt["files"].items():
                self.assertEqual(runner.source.file_identity(receipt_path.parent / name), identity)
            with np.load(receipt_path.parent / "predictions.npz") as values:
                np.testing.assert_array_equal(values["index"], [0, 1])
            with self.assertRaises(FileExistsError):
                runner.copy_batch_shard(bundle, root / "drive", "gaussian", 0, files)
            with patch.object(runner.source, "verify_identity", side_effect=ValueError("SHA-256 mismatch")):
                with self.assertRaises(ValueError):
                    runner.copy_batch_shard(bundle, root / "drive", "gaussian", 1, files)
            self.assertFalse((receipt_path.parent.parent / "batch0001/receipt.json").exists())

    def test_bad_manifest_seals_both_failed_bundles_and_drive_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            assets = root / "assets"; assets.mkdir()
            (assets / "manifest.json").write_text("{}")
            arguments = self.args("--asset-root", str(assets), "--result-root", str(root / "result"), "--drive-root", str(root / "drive"), "--asset-manifest-sha256", "0" * 64)
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                runner.main(arguments)
            bundles = [p for p in (root / "result").glob("*/*/*") if p.is_dir()]
            self.assertEqual(len(bundles), 2)
            for bundle in bundles:
                self.assertEqual(json.loads((bundle / "config.json").read_text())["status"], "failed")
                with zipfile.ZipFile(str(bundle) + ".zip") as archive:
                    self.assertIsNone(archive.testzip())
                receipt = root / "drive/scanobjectnn_c" / runner.METHOD / bundle.name / "final.receipt.json"
                self.assertEqual(json.loads(receipt.read_text())["result_status"], "failed")


if __name__ == "__main__":
    unittest.main()
