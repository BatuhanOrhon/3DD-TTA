import unittest
from contextlib import redirect_stderr
from io import StringIO

import gsd_protocol
import run_baseline
from eval_gsd_smooth import build_commands


class GSDSmoothLauncherTests(unittest.TestCase):
    def test_parameters_round_trip_without_six_digit_rounding(self):
        beta, hard, smooth = 1.23456789123, 20.123456789, .000987654321
        commands = build_commands("smoke", beta=beta, hard_weight=hard, smooth_weight=smooth)
        args = [run_baseline.parse_arguments(command[3:]) for command in commands]
        self.assertEqual(args[1].gsd_weight, hard)
        self.assertEqual(args[2].gsd_weight, smooth)
        self.assertEqual(args[2].gsd_beta, beta)

    def test_pilot_builds_v1_hard_v2_and_smooth_v2_arms(self):
        commands = build_commands(
            "pilot", beta=1.5, hard_weight=2.0, smooth_weight=3.0)
        self.assertEqual(len(commands), 9)
        methods = [command[command.index("--method") + 1] for command in commands]
        self.assertEqual(methods.count(gsd_protocol.METHOD), 3)
        self.assertEqual(methods.count(gsd_protocol.SMOOTH_METHOD), 6)
        for command in commands:
            args = run_baseline.parse_arguments(command[3:])
            if args.method == gsd_protocol.METHOD:
                self.assertNotIn("--gsd-profile", command)
                self.assertEqual(args.gsd_weight, 1.0)
            elif args.gsd_profile == "hard":
                self.assertEqual(args.gsd_weight, 2.0)
                self.assertIsNone(args.gsd_beta)
            else:
                self.assertEqual(args.gsd_profile, "smooth")
                self.assertEqual(args.gsd_weight, 3.0)
                self.assertEqual(args.gsd_beta, 1.5)

    def test_smoke_defaults_to_three_matched_methods(self):
        commands = build_commands("smoke", beta=1.0, hard_weight=1.0,
                                  smooth_weight=1.0)
        self.assertEqual(len(commands), 3)
        self.assertEqual({command[command.index("--seed") + 1]
                          for command in commands}, {"0"})

    def test_smooth_v2_requires_explicit_weight(self):
        common = ["--method", gsd_protocol.SMOOTH_METHOD, "--batch_size", "32",
                  "--max-batches", "0", "--corruptions", "gaussian", "impulse",
                  "--gsd-profile", "smooth", "--gsd-beta", "1.5"]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_baseline.parse_arguments(common)
        args = run_baseline.parse_arguments(common + ["--gsd-weight", "3"])
        self.assertEqual(args.gsd_weight, 3)

    def test_v2_protocol_rejects_missing_profile_beta_and_benchmark(self):
        common = ["--method", gsd_protocol.SMOOTH_METHOD, "--batch_size", "32",
                  "--max-batches", "0", "--corruptions", "gaussian", "impulse",
                  "--gsd-weight", "2"]
        invalid = ([], ["--gsd-profile", "smooth"],
                   ["--gsd-profile", "hard", "--gsd-beta", "1"],
                   ["--gsd-profile", "smooth", "--gsd-beta", "1", "--gsd-stage", "benchmark",
                    "--corruptions", *run_baseline.CORRUPTIONS])
        for extra in invalid:
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                run_baseline.parse_arguments(common + list(extra))


if __name__ == "__main__":
    unittest.main()
