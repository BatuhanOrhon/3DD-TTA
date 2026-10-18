"""CPU checks for the source-only entry point; no native/GPU imports."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import zipfile

from scripts import eval_multibackbone_source as runner


class SourceRunnerContracts(unittest.TestCase):
    def arguments(self, *extra):
        return runner.parse_args([
            "--backbone", "dgcnn", "--dataset", "scanobjectnn_c",
            "--dataset-root", "/data", "--label-path", "/data/label.npy",
            "--asset-root", "/assets", "--result-root", "/results", *extra])

    def test_preview_is_read_only_and_does_not_require_assets(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            runner.main([
                "--preview", "--backbone", "dgcnn", "--dataset", "scanobjectnn_c",
                "--dataset-root", "/absent", "--label-path", "/absent/label.npy",
                "--asset-root", "/absent/assets", "--result-root", "/absent/results"])
        config = json.loads(stream.getvalue())
        self.assertEqual(config["n_examples"], 581)
        self.assertEqual(config["split"], "main_split")
        self.assertEqual(config["dgcnn_head_dataset_name"], "modelnet")
        self.assertEqual(len(config["corruptions"]), 15)

    def test_manifest_hash_required_before_run(self):
        with self.assertRaisesRegex(ValueError, "manifest-sha256"):
            runner.resolve_config(self.arguments())

    def test_corruptions_and_limits_are_validated(self):
        for extra in (("--corruptions", "gaussian", "gaussian"),
                      ("--batch-size", "0"), ("--max-batches", "-1")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.resolve_config(self.arguments("--preview", *extra))

    def test_manifest_checks_hash_size_and_path_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "weights").write_bytes(b"abc")
            entry = {"sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}
            manifest = {"files": {"weights": entry}, "datasets": {}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            digest = runner.file_identity(path)["sha256"]
            self.assertEqual(runner.read_manifest(root, digest), manifest)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.read_manifest(root, "0" * 64)
            (root / "weights").write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.read_manifest(root, digest)
            with self.assertRaisesRegex(ValueError, "outside"):
                runner.asset_path(root, "../escape")
            (root / "linked").symlink_to(root / "weights")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                runner.asset_path(root, "linked")

    def test_dataset_split_cannot_be_relabelled_nobg(self):
        config = runner.resolve_config(self.arguments("--preview"))
        for split, count in (("main_split_nobg", 581), ("main_split", 580)):
            manifest = {"datasets": {"scanobjectnn_c": {
                "split": split, "n_examples": count, "files": {}}}}
            with self.subTest(split=split, count=count), self.assertRaises(ValueError):
                runner.dataset_manifest(manifest, config)

    def test_checkpoint_protocol_is_declared_before_evaluation(self):
        scan = runner.resolve_config(self.arguments("--preview", "--backbone", "curvenet"))
        self.assertIn("L-infinity", scan["preprocessing"])
        self.assertEqual(scan["input_layout"], "bcn")
        self.assertFalse(scan["paper_checkpoint_identity_verified"])
        pointnext = runner.resolve_config(self.arguments("--preview", "--backbone", "pointnext"))
        self.assertIn("height", pointnext["preprocessing"])
        self.assertEqual(pointnext["pointnext_width"], 32)

    def test_source_archive_is_copied_verified_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "source_snapshot.zip"
            with zipfile.ZipFile(archive, "x") as file:
                file.writestr("source.py", "print('pinned snapshot')\n")
            identity = runner.file_identity(archive)
            manifest = {"files": {"source_snapshot.zip": identity},
                        "source_snapshot_archive": "source_snapshot.zip",
                        "source_snapshot_sha256": identity["sha256"]}
            output = root / "bundle"
            output.mkdir()
            bundle = runner.RunBundle(output)
            config = {"asset_root": str(root)}
            runner.snapshot_sources(bundle, config, manifest)
            destination = output / "source_snapshot.zip"
            self.assertEqual(destination.read_bytes(), archive.read_bytes())
            initial_mtime = destination.stat().st_mtime_ns
            runner.snapshot_sources(bundle, config, manifest)
            self.assertEqual(destination.stat().st_mtime_ns, initial_mtime)
            self.assertEqual(runner.file_identity(destination), identity)
            manifest["source_snapshot_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "archive SHA-256"):
                runner.snapshot_sources(bundle, config, manifest)
            manifest["source_snapshot_sha256"] = identity["sha256"]
            malformed = dict(manifest, files={"source_snapshot.zip": dict(identity, sha256="0" * 64)})
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.snapshot_sources(bundle, config, malformed)
            destination.write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.snapshot_sources(bundle, config, manifest)

    def test_synthetic_relative_module_is_skipped_but_missing_asset_stays_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "bundle"
            output.mkdir()
            config = {"asset_root": str(root)}
            bundle = runner.RunBundle(output)
            manifest = {"files": {}}
            synthetic = ModuleType("synthetic_torch_namespace")
            synthetic.__file__ = "__nonexistent_synthetic_torch_classes__.py"
            with patch.dict(runner.sys.modules, {synthetic.__name__: synthetic}):
                runner.snapshot_sources(bundle, config, manifest)
            missing = ModuleType("missing_pinned_asset")
            missing.__file__ = str(root / "missing.py")
            with patch.dict(runner.sys.modules, {missing.__name__: missing}):
                with self.assertRaisesRegex(ValueError, "absent from manifest"):
                    runner.snapshot_sources(bundle, config, manifest)
                manifest["files"]["missing.py"] = {"sha256": "0" * 64, "bytes": 0}
                with self.assertRaises(FileNotFoundError):
                    runner.snapshot_sources(bundle, config, manifest)


class SourceTensorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
        except ImportError:
            raise unittest.SkipTest("Optional CPU tensor checks require numpy and torch")
        cls.np, cls.torch = np, torch

    def test_height_is_computed_before_sphere_normalization(self):
        points = self.torch.tensor([[[0., 2., 0.], [2., 4., 0.], [4., 6., 0.]]])
        observed = []

        def fps(values, count):
            observed.append((values.clone(), count))
            return values

        xyz, features = runner.preprocess(points, {"backbone": "pointnext", "dataset": "scanobjectnn_c"}, fps)
        self.assertEqual(observed[0][1], 1024)
        self.assertTrue(self.torch.equal(observed[0][0], points))
        self.assertTrue(self.torch.equal(features[:, 3, :], self.torch.tensor([[0., 2., 4.]])))
        self.assertTrue(self.torch.equal(features[:, :3, :], xyz.transpose(1, 2)))
        self.assertTrue(self.torch.allclose(xyz.mean(1), self.torch.zeros(1, 3)))
        self.assertTrue(self.torch.allclose(xyz.norm(dim=-1).amax(1), self.torch.ones(1)))

    def test_curve_scan_normalizes_before_fps_and_other_paths_do_not(self):
        points = self.torch.tensor([[[2., 0., 0.], [4., 2., 0.]]])
        observed = []

        def fps(values, count):
            observed.append(values.clone())
            return values

        normalized, _ = runner.preprocess(points, {"backbone": "curvenet", "dataset": "scanobjectnn_c"}, fps)
        self.assertTrue(self.torch.equal(observed[-1], self.torch.tensor([[[-1., -1., 0.], [1., 1., 0.]]])))
        for backbone, dataset in (("dgcnn", "scanobjectnn_c"), ("curvenet", "modelnet40_c"), ("pointmae", "modelnet40_c")):
            xyz, features = runner.preprocess(points, {"backbone": backbone, "dataset": dataset}, fps)
            self.assertIs(xyz, points)
            self.assertIsNone(features)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            runner.preprocess(self.torch.zeros(1, 4, 3), {"backbone": "pointnext", "dataset": "scanobjectnn_c"}, fps)

    def test_dataset_schema_finite_coordinates_and_labels_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "data.npy"
            config = {"n_examples": 2, "num_classes": 15}
            for values, labels, valid in (
                (self.np.zeros((2, 3, 3), dtype=self.np.float32), False, True),
                (self.np.full((2, 3, 3), self.np.nan), False, False),
                (self.np.zeros((2, 3, 4)), False, False),
                (self.np.array([[0], [14]]), True, True),
                (self.np.array([0, 15]), True, False),
                (self.np.array([0., 1.]), True, False),
            ):
                self.np.save(path, values)
                identity = dict(runner.file_identity(path), shape=list(values.shape))
                with self.subTest(shape=values.shape, labels=labels, valid=valid):
                    if valid:
                        self.assertEqual(len(runner.load_array(path, identity, config, labels=labels)), 2)
                    else:
                        with self.assertRaises(ValueError):
                            runner.load_array(path, identity, config, labels=labels)

    def test_frozen_state_hash_includes_buffers(self):
        model = self.torch.nn.BatchNorm1d(3)
        initial = runner.state_hash(model)
        model.running_mean[0] = 1
        self.assertNotEqual(runner.state_hash(model), initial)

    def test_failed_invocation_is_sealed_with_seven_standard_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            assets = root / "assets"
            assets.mkdir()
            (assets / "manifest.json").write_text("{}")
            arguments = ["--backbone", "dgcnn", "--dataset", "scanobjectnn_c",
                         "--asset-root", str(assets), "--dataset-root", str(root),
                         "--label-path", str(root / "labels.npy"), "--result-root", str(root / "results"),
                         "--asset-manifest-sha256", "0" * 64]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit):
                runner.main(arguments)
            self.assertIn("Asset manifest SHA-256 mismatch", stream.getvalue())
            bundle, = (path for path in (root / "results").glob("*/*/*") if path.is_dir())
            for name in ("config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
                         "per_corruption.csv", "summary.csv"):
                self.assertTrue((bundle / name).is_file(), name)
            self.assertEqual(json.loads((bundle / "config.json").read_text())["status"], "failed")
            self.assertIn("Asset manifest SHA-256 mismatch", (bundle / "stdout.log").read_text())
            seal = json.loads((bundle / "bundle_manifest.json").read_text())
            for relative, expected in seal.items():
                self.assertEqual(runner.file_identity(bundle / relative), expected)
            with zipfile.ZipFile(str(bundle) + ".zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()),
                                 {bundle.name + "/" + str(path.relative_to(bundle))
                                  for path in bundle.rglob("*") if path.is_file()})
                for relative, expected in seal.items():
                    content = archive.read(bundle.name + "/" + relative)
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected["sha256"])


class DriveCheckpointContracts(unittest.TestCase):
    def bundle(self, root):
        bundle = runner.RunBundle.create(root / "results", "test-seed0",
                                         {"dataset": "scanobjectnn_c", "method": "source_only", "seed": 0}, "test")
        (bundle.path / "source_snapshot.zip").write_bytes(b"recoverable source bytes")
        return bundle

    def test_partial_then_final_archives_are_verified_and_do_not_grow_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = self.bundle(root)
            row = runner.corruption_row(bundle.path.name, 0, "gaussian", 2, 1, 1., 0., "partial", dataset="scanobjectnn_c", method="source_only")
            bundle.write_results([row], "partial")
            (bundle.path / "gaussian_predictions.npz").write_bytes(b"indexed predictions")
            receipt = runner.save_drive_checkpoint(bundle, root / "drive", "corruption-gaussian")
            archive = Path(receipt["archive"]["path"])
            self.assertEqual(runner.file_identity(archive), {key: receipt["archive"][key] for key in ("sha256", "bytes")})
            self.assertTrue(receipt["stdout_is_prefix"])
            with zipfile.ZipFile(archive) as file:
                manifest = json.loads(file.read(bundle.path.name + "/checkpoint_manifest.json"))
                self.assertIn("source_snapshot.zip", manifest)
                self.assertIn("gaussian_predictions.npz", manifest)
                self.assertIn("2,1,0.5", file.read(bundle.path.name + "/per_corruption.csv").decode())
                for relative, expected in manifest.items():
                    content = file.read(bundle.path.name + "/" + relative)
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected["sha256"])
            self.assertFalse((bundle.path / "checkpoint_manifest.json").exists())
            self.assertFalse(any("corruption-gaussian" in path.name for path in bundle.path.iterdir()))
            config = json.loads((bundle.path / "config.json").read_text())
            config["status"] = "partial"
            bundle.write_config(config)
            runner.seal_bundle(bundle)
            final = runner.save_drive_checkpoint(bundle, root / "drive", "final", final=True)
            self.assertEqual(final["result_status"], "partial")
            self.assertFalse(final["stdout_is_prefix"])
            self.assertEqual(Path(final["archive"]["path"]).read_bytes(), Path(str(bundle.path) + ".zip").read_bytes())
            persisted = json.loads((Path(final["archive"]["path"]).parent / "final.receipt.json").read_text())
            self.assertEqual(persisted, final)

    def test_collisions_and_copy_failure_publish_no_completed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = self.bundle(root)
            runner.seal_bundle(bundle)
            receipt = runner.save_drive_checkpoint(bundle, root / "drive", "final", final=True)
            original = Path(receipt["archive"]["path"]).read_bytes()
            with self.assertRaises(FileExistsError):
                runner.save_drive_checkpoint(bundle, root / "drive", "final", final=True)
            self.assertEqual(Path(receipt["archive"]["path"]).read_bytes(), original)
            with patch.object(runner.shutil, "copyfileobj", side_effect=OSError("Drive interrupted")):
                with self.assertRaisesRegex(OSError, "Drive interrupted"):
                    runner.save_drive_checkpoint(bundle, root / "drive", "interrupted", final=True)
            target = Path(receipt["archive"]["path"]).parent
            self.assertTrue((target / "interrupted.zip").exists())
            self.assertFalse((target / "interrupted.receipt.json").exists())
            with self.assertRaises(FileExistsError):
                runner.save_drive_checkpoint(bundle, root / "drive", "interrupted", final=True)
            with patch.object(runner, "verify_identity", side_effect=ValueError("SHA-256 mismatch")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    runner.save_drive_checkpoint(bundle, root / "drive", "corrupt", final=True)
            self.assertFalse((target / "corrupt.receipt.json").exists())
            self.assertIsNone(runner.save_drive_checkpoint(bundle, None, "disabled"))


if __name__ == "__main__":
    unittest.main()
