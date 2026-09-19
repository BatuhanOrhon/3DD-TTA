from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest

import numpy as np
import torch

import run_baseline


class PreprocessingIdentityTests(unittest.TestCase):
    def test_cli_accepts_locked_preprocessing_identity_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "preprocessing_identity",
            "--batch_size", "32",
            "--seed", "0",
            "--severity", "5",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.method, "preprocessing_identity")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.severity, 5)
        self.assertEqual(args.corruptions, list(run_baseline.CORRUPTIONS))

    def test_cli_accepts_preprocessing_identity_seed_stability_seed_one_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "preprocessing_identity_seed_stability",
            "--batch_size", "32",
            "--seed", "1",
            "--severity", "5",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.method, "preprocessing_identity_seed_stability")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.severity, 5)

    def test_preprocessing_identity_seed_stability_config_records_fixed_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "preprocessing_identity_seed_stability",
            "--batch_size", "32",
            "--seed", "2",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        config = run_baseline.build_config(args)

        self.assertEqual(config["method"], "preprocessing_identity_seed_stability")
        self.assertEqual(config["stage"], "preprocessing_identity_seed_stability")
        self.assertFalse(config["lion_loaded"])
        self.assertEqual(config["seed_stability_reference"],
                         "preprocessing_identity seed0 archive")
        self.assertEqual(config["lion_mode_policy"], "bypassed")
        self.assertEqual(config["final_decode_style"], "identity; no decode")

    def test_preprocessing_identity_seed_stability_rejects_seed_zero(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run_baseline.parse_arguments([
                    "--method", "preprocessing_identity_seed_stability",
                    "--batch_size", "32",
                    "--seed", "0",
                    "--max-batches", "0",
                    "--corruptions", *run_baseline.CORRUPTIONS,
                ])

    def test_seed_stability_uses_preprocessing_identity_route(self):
        self.assertTrue(run_baseline.is_preprocessing_identity_method("preprocessing_identity"))
        self.assertTrue(run_baseline.is_preprocessing_identity_method(
            "preprocessing_identity_seed_stability"))
        self.assertFalse(run_baseline.is_preprocessing_identity_method("pure_vae_seed_stability"))

    def test_cli_accepts_locked_pure_vae_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "pure_vae_encode_decode",
            "--batch_size", "32",
            "--seed", "0",
            "--severity", "5",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.method, "pure_vae_encode_decode")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.severity, 5)

    def test_cli_accepts_pure_vae_seed_stability_seed_one_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "pure_vae_seed_stability",
            "--batch_size", "32",
            "--seed", "1",
            "--severity", "5",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.method, "pure_vae_seed_stability")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.severity, 5)

    def test_pure_vae_seed_stability_config_records_fixed_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "pure_vae_seed_stability",
            "--batch_size", "32",
            "--seed", "2",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        config = run_baseline.build_config(args)

        self.assertEqual(config["method"], "pure_vae_seed_stability")
        self.assertEqual(config["stage"], "pure_vae_seed_stability")
        self.assertTrue(config["lion_loaded"])
        self.assertEqual(config["seed_stability_reference"],
                         "pure_vae_encode_decode seed0 archive")
        self.assertEqual(config["lion_mode_policy"], "raw VAE eval; priors bypassed")
        self.assertFalse(config["prior_used"])
        self.assertEqual(config["vae_contract"], "encode -> decompose_eps -> sample")

    def test_pure_vae_seed_stability_rejects_seed_zero(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run_baseline.parse_arguments([
                    "--method", "pure_vae_seed_stability",
                    "--batch_size", "32",
                    "--seed", "0",
                    "--max-batches", "0",
                    "--corruptions", *run_baseline.CORRUPTIONS,
                ])

    def test_identity_preprocessing_runs_without_a_lion_call(self):
        calls = []

        def normalize(data):
            calls.append("normalize")
            return data, None, None

        def upsample_all(data, number):
            calls.append(("upsample_all", number))
            self.assertIsInstance(data, np.ndarray)
            return np.broadcast_to(data[:, :1], (data.shape[0], number, data.shape[2])).copy()

        def rotate(data):
            calls.append("rotate")
            return data + 1

        def rotateback(data):
            calls.append("rotateback")
            return data - 1

        def fps(data, number):
            calls.append(("fps", number))
            return data[:, :number]

        baseline = SimpleNamespace(
            normalize=normalize,
            upsample_all=upsample_all,
            rotate_pointcloud=rotate,
            rotateback_pointcloud=rotateback,
            misc=SimpleNamespace(fps=fps),
        )
        args = SimpleNamespace(device="cpu", dataset_name="modelnet-c")
        data = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]])

        points = run_baseline.preprocessing_identity_points(data, baseline, args, torch)

        self.assertEqual(points.shape, (1, 1024, 3))
        expected = torch.tensor([2.0, 3.0, 4.0]).view(1, 1, 3) * 3.3885
        self.assertTrue(torch.allclose(points, expected.expand_as(points)))
        self.assertEqual(calls, ["normalize", ("upsample_all", 2048), "rotate", "rotateback",
                                 "normalize", ("fps", 1024)])

    def test_identity_config_records_opt_in_control_contract(self):
        args = run_baseline.parse_arguments([
            "--method", "preprocessing_identity",
            "--batch_size", "32",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        config = run_baseline.build_config(args)

        self.assertEqual(config["method"], "preprocessing_identity")
        self.assertEqual(config["stage"], "preprocessing_identity")
        self.assertEqual(config["severity"], 5)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["scale_factor"], 3.3885)
        self.assertEqual(config["num_input_points"], 2048)
        self.assertEqual(config["num_classifier_points"], 1024)
        self.assertEqual(config["lion_mode_policy"], "bypassed")
        self.assertEqual(config["final_decode_style"], "identity; no decode")
        self.assertEqual(config["preprocessing"], run_baseline.IDENTITY_PREPROCESSING)
        self.assertFalse(config["lion_loaded"])
        self.assertEqual(config["spectral"], {})
        self.assertEqual(config["projection"], {})
        self.assertEqual(config["scheduler_config"], {})
        self.assertEqual(config["cli_args"]["method"], "preprocessing_identity")

    def test_pure_vae_config_records_vae_only_contract(self):
        args = run_baseline.parse_arguments([
            "--method", "pure_vae_encode_decode",
            "--batch_size", "32",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        config = run_baseline.build_config(args)

        self.assertEqual(config["method"], "pure_vae_encode_decode")
        self.assertEqual(config["stage"], "pure_vae_encode_decode")
        self.assertTrue(config["lion_loaded"])
        self.assertEqual(config["lion_mode_policy"], "raw VAE eval; priors bypassed")
        self.assertFalse(config["prior_used"])
        self.assertEqual(config["vae_contract"], "encode -> decompose_eps -> sample")

    def test_identity_rejects_scope_changes(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run_baseline.parse_arguments([
                    "--method", "preprocessing_identity",
                    "--batch_size", "16",
                    "--max-batches", "0",
                    "--corruptions", *run_baseline.CORRUPTIONS,
                ])

    def test_pure_vae_uses_encoded_latents_without_prior_or_guidance(self):
        calls = []

        def normalize(data):
            calls.append("normalize")
            return data, None, None

        def upsample_all(data, number):
            calls.append(("upsample_all", number))
            return np.broadcast_to(data[:, :1], (data.shape[0], number, data.shape[2])).copy()

        def rotate(data):
            calls.append("rotate")
            return data

        def rotateback(data):
            calls.append("rotateback")
            return data

        def fps(data, number):
            calls.append(("fps", number))
            return data[:, :number]

        class FakeVAE:
            def encode(self, data):
                calls.append(("encode", tuple(data.shape)))
                return ("encoded",)

            def decompose_eps(self, encoded):
                calls.append(("decompose_eps", encoded))
                return ("global", "local")

            def sample(self, num_samples, decomposed_eps):
                calls.append(("sample", num_samples, decomposed_eps))
                return torch.full((num_samples, 2048, 3), 7.0)

        baseline = SimpleNamespace(
            normalize=normalize,
            upsample_all=upsample_all,
            rotate_pointcloud=rotate,
            rotateback_pointcloud=rotateback,
            misc=SimpleNamespace(fps=fps),
        )
        lion = SimpleNamespace(vae=FakeVAE())
        args = SimpleNamespace(device="cpu", dataset_name="modelnet-c")
        data = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]])

        points = run_baseline.pure_vae_encode_decode_points(data, baseline, lion, args, torch)

        self.assertEqual(points.shape, (1, 1024, 3))
        self.assertTrue(torch.allclose(points, torch.full_like(points, 7.0)))
        self.assertEqual(calls, [
            "normalize", ("upsample_all", 2048), "rotate",
            ("encode", (1, 2048, 3)), ("decompose_eps", "encoded"),
            ("sample", 1, ("global", "local")), "rotateback",
            "normalize", ("fps", 1024),
        ])
