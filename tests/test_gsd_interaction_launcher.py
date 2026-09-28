import unittest

from scripts.run_gsd_interaction_from_scratch import build_rebuild_commands


class InteractionLauncherTests(unittest.TestCase):
    def test_rebuilds_only_the_four_required_prerequisite_conditions(self):
        calibration = {
            "run_id": "20260928-120000_gsd-cal-diagnose-reference-seed0-n64",
            "cli_args": {"gsd_split_seed": 20260927},
        }
        report = {
            "candidates": {
                "0.5": {"weights": {"0.001": 2.5, "0.01": 25.0}},
                "2.0": {"weights": {"0.001": 4.0, "0.01": 8.0}},
            }
        }

        commands = build_rebuild_commands(calibration, report, "/runs/calibration/config.json")

        self.assertEqual(len(commands), 4)
        args_by_name = {command[command.index("--run-name") + 1]: command for command in commands}
        self.assertEqual(len(args_by_name), 4)
        cells = set()
        baseline_count = 0
        for name, command in args_by_name.items():
            if "--gsd-beta" in command:
                beta = command[command.index("--gsd-beta") + 1]
                rho = command[command.index("--gsd-target-rho") + 1]
                cells.add((beta, rho))
                self.assertEqual(command[command.index("--gsd-profile") + 1], "smooth")
            else:
                baseline_count += 1
                self.assertEqual(command[command.index("--gsd-weight") + 1], "0.0")
        self.assertEqual(baseline_count, 1)
        self.assertCountEqual(cells, {("0.5", "0.001"), ("2.0", "0.001"), ("2.0", "0.01")})
        self.assertTrue(all("--gsd-calibration-reference" in command for command in commands))

    def test_rebuild_commands_use_weights_from_the_supplied_reference(self):
        calibration = {
            "run_id": "cal-reference",
            "cli_args": {"gsd_split_seed": 20260927},
        }
        report = {
            "candidates": {
                "0.5": {"weights": {"0.001": 2.5, "0.01": 25.0}},
                "2.0": {"weights": {"0.001": 4.0, "0.01": 8.0}},
            }
        }

        commands = build_rebuild_commands(calibration, report, "/runs/calibration/config.json")

        weights = {
            (command[command.index("--gsd-beta") + 1],
             command[command.index("--gsd-target-rho") + 1]):
                command[command.index("--gsd-weight") + 1]
            for command in commands if "--gsd-beta" in command
        }
        self.assertEqual(weights, {("0.5", "0.001"): "2.5",
                                   ("2.0", "0.001"): "4.0",
                                   ("2.0", "0.01"): "8.0"})


if __name__ == "__main__":
    unittest.main()
