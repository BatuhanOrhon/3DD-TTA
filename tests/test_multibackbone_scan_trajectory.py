"""CPU contracts for the opt-in four-head Scan trajectory runner."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import rng_hash
from scripts import eval_multibackbone_guidance as runner
from scripts import verify_multibackbone_pointmae_parity as parity


class ScanTrajectoryRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return ["--preview", "--dataset", "scanobjectnn_c", "--dataset-root", "/data",
                "--label-path", "/data/label.npy", "--asset-root", "/assets", "--result-root", "/result", *extra]

    def operations(self):
        tree = ast.parse((runner.source.REPO / "utilities_3dd_tta.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"normalize", "unnormalize_data", "rotate_pointcloud", "rotateback_pointcloud"}]
        namespace = {"torch": torch, "np": np}
        exec(compile(tree, "audited_scan_geometry", "exec"), namespace)
        namespace["upsample_all"] = lambda values, count: values[:, np.arange(count) % values.shape[1], :]
        return SimpleNamespace(**namespace)

    def test_scope_and_pointmae_paths_are_explicit_without_changing_defaults(self):
        default = runner.configurations(self.args())
        self.assertEqual([c["backbone"] for c in default], ["curvenet", "pointnext"])
        self.assertTrue(all(not c.get("trajectory_observation") for c in default))
        arguments = ("--trajectory-observe", "--backbones", "curvenet", "pointnext", "dgcnn", "pointmae",
                     "--pointmae-config", "source_snapshot/cfgs/tta_scanobj.yaml", "--pointmae-checkpoint", "checkpoints/scanobject_jt.pth")
        configs = runner.configurations(self.args(*arguments))
        self.assertEqual(len(configs), 4)
        for config in configs:
            self.assertTrue(config["trajectory_observation"])
            self.assertEqual((config["n_examples"], config["num_classes"], config["batch_size"]), (581, 15, 70))
            self.assertEqual(config["split"], "main_split")
            self.assertFalse(config["trajectory_selection"]["labels_used"])
        self.assertEqual(configs[-1]["checkpoint"], "checkpoints/scanobject_jt.pth")
        self.assertEqual(runner.matched_source(configs[-1], {}, {})["status"], "unavailable")
        for invalid in (("--trajectory-observe", "--dataset", "modelnet40_c"),
                        ("--trajectory-observe", "--method", "original3dd"),
                        ("--backbones", "pointmae"), ("--trajectory-observe", "--backbones", "pointmae")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runner.configurations(self.args(*invalid))

    def test_probe_and_terminal_share_inverse_bbox_with_nonzero_center_and_scale(self):
        ops = self.operations()
        raw = torch.tensor([[[1., 2., 3.], [4., 6., 9.], [2., 5., 8.]]]).repeat(21, 1, 1)
        captured, seen_kwargs = [], []
        def reconstruct(**kwargs):
            seen_kwargs.append(set(kwargs))
            decoded = kwargs["x"].clone() + .1
            if "trajectory_observer" in kwargs:
                kwargs["trajectory_observer"](decoded)
            return decoded, {"spectral": .5}
        def factory(center, scale):
            self.assertTrue(torch.all(center != 0))
            self.assertTrue(torch.all(scale != 1))
            def probe(decoded):
                before = decoded.clone()
                captured.append(runner.decoded_to_geometry(decoded, "scanobjectnn_c", scale, center, ops))
                self.assertTrue(torch.equal(before, decoded))
            return probe
        before = rng_hash()
        ordinary, _ = runner.reconstruct_geometry(raw, "scanobjectnn_c", "gaussian", None, None, reconstruct, ops, device="cpu")
        observed, _ = runner.reconstruct_geometry(raw, "scanobjectnn_c", "gaussian", None, None, reconstruct, ops,
                                                   device="cpu", observer_factory=factory)
        self.assertNotIn("trajectory_observer", seen_kwargs[0])
        self.assertIn("trajectory_observer", seen_kwargs[1])
        self.assertTrue(torch.equal(ordinary, observed))
        self.assertTrue(torch.equal(captured[0], observed))
        normalized, center, scale = ops.normalize(raw)
        decoded = ops.rotate_pointcloud(torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float() * 3.3885) + .1
        expected = ops.unnormalize_data(ops.rotateback_pointcloud(decoded) / 3.3885, scale, center)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(rng_hash(), before)

    def test_candidate_arrays_are_inside_hash_verified_drive_batch_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bundle"; path.mkdir()
            config = dict(dataset="scanobjectnn_c", method=runner.METHOD, run_id="trajectory-test")
            (path / "config.json").write_text(json.dumps(config))
            bundle = SimpleNamespace(path=path)
            arrays = dict(index=np.arange(560, 581), label=np.zeros(21, dtype=np.int64),
                          prediction=np.zeros(21, dtype=np.int64), logits=np.zeros((21, 15), dtype=np.float32),
                          trajectory_logits=np.arange(21 * 4 * 15, dtype=np.float32).reshape(21, 4, 15),
                          candidate_names=np.array(["clean_03", "clean_05", "clean_08", "terminal"]),
                          candidate_loop_counts=np.array([3, 5, 8, 10]))
            files = runner.write_batch_shard(bundle, "gaussian", 8, arrays, {"trajectory": {"complete": True}})
            receipt = runner.copy_batch_shard(bundle, root / "drive", "gaussian", 8, files)
            destination = Path(receipt["receipt_path"]).parent
            with np.load(destination / "predictions.npz", allow_pickle=False) as restored:
                self.assertEqual(set(restored.files), set(arrays))
                for key, value in arrays.items():
                    np.testing.assert_array_equal(restored[key], value)
            for filename, identity in receipt["files"].items():
                self.assertEqual(runner.source.file_identity(destination / filename), identity)
            with self.assertRaises(FileExistsError):
                runner.copy_batch_shard(bundle, root / "drive", "gaussian", 8, files)

    def test_pointmae_loader_rejects_wrong_bytes_before_unpickling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "scanobject_jt.pth"
            checkpoint.write_bytes(b"untrusted legacy pickle")
            config = dict(asset_root=str(root), backbone="pointmae", dataset="scanobjectnn_c", pointmae_checkpoint=checkpoint.name)
            manifest = dict(files={checkpoint.name: runner.source.file_identity(checkpoint)})
            with patch("torch.load") as load, patch.object(parity, "load_pinned_legacy_checkpoint", wraps=parity.load_pinned_legacy_checkpoint) as legacy:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    runner.source.load_classifier(config, manifest)
                self.assertEqual(legacy.call_args.args[1], "53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
