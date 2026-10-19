"""CPU checks for locked original-3DD dispatch and explicit DGCNN selection."""
import ast
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import _rng_state, _restore_rng, rng_hash
from scripts import eval_multibackbone_guidance as runner


class Scheduler:
    def __init__(self, **options):
        self.config = dict(options, set_alpha_to_one=True, steps_offset=0)
        self.final_alpha_cumprod = torch.tensor(1.)
        self.alphas_cumprod = torch.linspace(.99, .1, 1000)

    def set_timesteps(self, total, device=None):
        self.timesteps = torch.arange(990, -1, -10)

    def step(self, noise, t, latent):
        return SimpleNamespace(pred_original_sample=latent - .1 * noise, prev_sample=latent - .05 * noise)


class Original3DDExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def args(self, *extra):
        return ["--dataset", "modelnet40_c", "--dataset-root", "/data", "--label-path", "/data/label.npy",
                "--asset-root", "/assets", "--result-root", "/result", *extra]

    def test_all_requested_modes_preview_without_heavy_imports_and_enforce_locked_batches(self):
        for dataset in ("modelnet40_c", "scanobjectnn_c"):
            for method, heads in (("gsd", ["curvenet", "pointnext"]), ("original3dd", ["curvenet", "pointnext"]),
                                  ("gsd", ["dgcnn"]), ("original3dd", ["dgcnn"])):
                arguments = self.args("--preview", "--dataset", dataset, "--method", method, "--backbones", *heads)
                result = subprocess.run([sys.executable, "-S", runner.__file__, *arguments], text=True, capture_output=True)
                with self.subTest(dataset=dataset, method=method, heads=heads):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    configs = json.loads(result.stdout)
                    self.assertEqual([c["backbone"] for c in configs], heads)
                    for config in configs:
                        original = method == "original3dd"
                        self.assertEqual(config["method"], "3dd_original" if original else runner.METHOD)
                        self.assertEqual(config["batch_size"], 32 if original else 70)
                        self.assertEqual(config["scheduler_final_alpha_to_one"], original)
                        self.assertEqual(config["decoder_style"], "original" if original else "updated")
                        self.assertIn("/" + str(len(heads)), config["timing_policy"])
                        self.assertEqual(config["num_classes"], 15 if dataset == "scanobjectnn_c" else 40)
                        if "dgcnn" in heads:
                            self.assertIn("weaker", config["backbone_caveat"])
        for arguments in (("--method", "original3dd", "--batch-size", "70"), ("--method", "gsd", "--batch-size", "32"),
                          ("--backbones", "curvenet", "curvenet")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                runner.configurations(self.args("--preview", *arguments))

    def test_scheduler_audit_asserts_actual_config_full_and_active_steps(self):
        scheduler = Scheduler(beta_end=.02, beta_start=.0001, beta_schedule="linear", clip_sample=False,
                              num_train_timesteps=1000, prediction_type="epsilon")
        scheduler.set_timesteps(100)
        record = runner.original_scheduler_record(scheduler, 35)
        self.assertEqual(record["timesteps"], list(range(990, -1, -10)))
        self.assertEqual(record["active_timesteps"], list(range(340, -1, -10)))
        self.assertTrue(record["config"]["set_alpha_to_one"])
        self.assertEqual(record["final_alpha_cumprod"], 1.)
        for key, value in (("set_alpha_to_one", False), ("beta_end", .03), ("steps_offset", 1)):
            with self.subTest(key=key), patch.dict(scheduler.config, {key: value}), self.assertRaises(ValueError):
                runner.original_scheduler_record(scheduler, 5)
        scheduler.timesteps[0] = 999
        with self.assertRaisesRegex(ValueError, "timesteps"):
            runner.original_scheduler_record(scheduler, 5)

    def operations(self):
        tree = ast.parse((runner.source.REPO / "utilities_3dd_tta.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"normalize", "unnormalize_data", "rotate_pointcloud", "rotateback_pointcloud"}]
        namespace = {"torch": torch, "np": np}
        exec(compile(tree, "audited_utilities", "exec"), namespace)
        namespace["upsample_all"] = lambda values, count: values[:, np.arange(count) % values.shape[1], :]
        return SimpleNamespace(**namespace)

    def original_module(self):
        chamfer = ModuleType("third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D")
        def distances(left, right):
            values = (left - right).square().sum(-1)
            return values, values, None, None
        chamfer.chamfer_3DDist = lambda: distances
        diffusers = ModuleType("diffusers"); diffusers.DDIMScheduler = Scheduler
        utilities = ModuleType("utilities_3dd_tta"); utilities.grad_freeze = lambda model: model.requires_grad_(False)
        spec = importlib.util.spec_from_file_location("_study_original_tta_cpu", runner.source.REPO / "tta.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module, "diffusers": diffusers,
                                     "utilities_3dd_tta": utilities, chamfer.__name__: chamfer}):
            spec.loader.exec_module(module)
        return module

    def test_original_dispatch_matches_unchanged_tta_for_partial_batch18_and_both_datasets(self):
        class VAE(torch.nn.Module):
            def encode(self, points):
                local = torch.cat((points, torch.ones_like(points[:, :, :1])), -1).reshape(len(points), -1)
                return None, None, [[torch.ones(len(points), 8)], [local]]
            def global2style(self, shape):
                return shape * 2
            def decoder(self, unused, *, beta, context, style):
                self.style = style.detach().clone()
                return context.reshape(-1, 2048, 4)[:, :, :3] + .01 * style.mean(1)[:, None, None]
        class Prior(torch.nn.Module):
            def forward(self, *, x, t, condition_input, clip_feat):
                return .1 * x + .05 * condition_input.mean(1, keepdim=True)
        module, ops = self.original_module(), self.operations()
        raw = torch.tensor([[[1., 2., 3.], [4., 6., 9.], [2., 5., 8.]]]).repeat(18, 1, 1)
        original_ones = torch.ones
        def cpu_ones(*args, **kwargs):
            kwargs.pop("device", None)
            return original_ones(*args, **kwargs)
        for dataset, corruption, steps in (("modelnet40_c", "gaussian", 5), ("scanobjectnn_c", "background", 35)):
            lion = SimpleNamespace(vae=VAE(), priors=[None, Prior()])
            state = _rng_state()
            normalized, center, scale = ops.normalize(raw)
            prepared = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float() * 3.3885
            prepared = ops.rotate_pointcloud(prepared)
            with patch.object(torch, "ones", side_effect=cpu_ones):
                direct = module.tta_reconstruct(prepared, lion, steps, .01, .01, .95, 100, scd_normalize=False)
            direct_rng = rng_hash()
            direct = ops.rotateback_pointcloud(direct)
            expected = ops.unnormalize_data(direct / 3.3885, scale, center) if dataset == "scanobjectnn_c" else ops.normalize(direct)[0]
            _restore_rng(state)
            observed = []
            with patch.object(torch, "ones", side_effect=cpu_ones):
                geometry, metrics = runner.reconstruct_geometry(raw, dataset, corruption, lion, None, module.tta_reconstruct, ops,
                                                               device="cpu", method="3dd_original", scheduler_observer=observed.append)
            with self.subTest(dataset=dataset):
                self.assertTrue(torch.equal(geometry, expected.detach()))
                self.assertEqual(rng_hash(), direct_rng)
                self.assertEqual(geometry.shape, (18, 2048, 3))
                self.assertEqual(len(observed), 1)
                self.assertEqual(len(observed[0]["active_timesteps"]), steps)
                self.assertEqual(metrics, {})
                self.assertTrue(torch.equal(lion.vae.style, original_ones(18, 8)))

    def test_original_method_counts_and_failed_dgcnn_delivery_use_separate_schema_path(self):
        config = runner.configurations(self.args("--preview", "--method", "original3dd", "--backbones", "dgcnn"))[0]
        config["run_id"] = "test"
        rows = []
        runner.update_row(rows, config, "gaussian", 18, 9, 2., 100., "partial")
        self.assertEqual(rows[0]["method"], "3dd_original")
        self.assertEqual(rows[0]["accuracy"], .5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); assets = root / "assets"; assets.mkdir()
            (assets / "manifest.json").write_text("{}")
            arguments = self.args("--method", "original3dd", "--backbones", "dgcnn", "--asset-root", str(assets),
                                  "--result-root", str(root / "result"), "--drive-root", str(root / "drive"), "--asset-manifest-sha256", "0" * 64)
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                runner.main(arguments)
            bundle, = (root / "result/modelnet40_c/3dd_original").glob("*/")
            failed = json.loads((bundle / "config.json").read_text())
            self.assertEqual((failed["status"], failed["backbone"]), ("failed", "dgcnn"))
            receipt = root / "drive/modelnet40_c/3dd_original" / bundle.name / "final.receipt.json"
            self.assertEqual(json.loads(receipt.read_text())["result_status"], "failed")


if __name__ == "__main__":
    unittest.main()
