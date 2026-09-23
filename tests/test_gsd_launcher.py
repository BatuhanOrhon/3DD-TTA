from pathlib import Path
import unittest

import run_baseline


class GSDLauncherTests(unittest.TestCase):
    def test_presets_resolve_to_locked_validated_methods(self):
        import eval_gsd_tta
        for stage, count in (("smoke", 3), ("pilot", 9), ("all15", 9)):
            commands = eval_gsd_tta.build_commands(stage)
            self.assertEqual(len(commands), count)
            for command in commands:
                self.assertEqual(Path(command[2]).name, "run_baseline.py")
                args = run_baseline.parse_arguments(command[3:])
                self.assertEqual(args.batch_size, 32)
                self.assertTrue(args.lion_eval_mode)
                self.assertFalse(args.lion_ema_mode)
                if stage == "all15":
                    self.assertEqual(args.corruptions, list(run_baseline.CORRUPTIONS))
                    self.assertEqual(args.max_batches, 0)
                if stage == "pilot":
                    self.assertEqual(args.corruptions, ["gaussian", "impulse"])
                    self.assertEqual(args.max_batches, 0)
            self.assertEqual(sum("3dd_original" in c for c in commands), count // 3)


if __name__ == "__main__":
    unittest.main()
