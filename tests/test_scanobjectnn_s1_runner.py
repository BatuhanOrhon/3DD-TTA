"""Regression tests for the ScanObjectNN severity-1 diagnostic runner."""

import ast
import unittest
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "result/preparation/cli_scanobjectnn_s1_20261018"
    / "run_scanobjectnn_severity1_profile.py"
)


class ScanObjectNNSeverity1RunnerTests(unittest.TestCase):
    def test_table3_values_are_mapped_by_corruption_name(self):
        module = ast.parse(RUNNER.read_text())
        paper_assignment = next(
            node for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "paper"
                    for target in node.targets)
        )
        self.assertIsInstance(
            paper_assignment.value,
            ast.Dict,
            "Table 3 values must be an explicit name-to-value mapping",
        )
        paper = ast.literal_eval(paper_assignment.value)

        self.assertEqual(paper, {
            "uniform": 59.0,
            "gaussian": 59.0,
            "background": 57.3,
            "impulse": 66.6,
            "upsampling": 64.4,
            "distortion_rbf": 73.6,
            "distortion_rbf_inv": 72.9,
            "density": 81.2,
            "density_inc": 82.4,
            "shear": 71.9,
            "rotation": 59.7,
            "cutout": 78.3,
            "distortion": 70.7,
        })


if __name__ == "__main__":
    unittest.main()
