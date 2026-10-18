"""CLI defaults for the corruption dataset generator."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]


class CorruptedDatasetCliTests(unittest.TestCase):
    def test_scanobjectnn_defaults_to_released_evaluation_severity(self):
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.random = types.ModuleType("numpy.random")
        fake_numpy.array = lambda value: value
        fake_numpy.ones = lambda shape: shape
        fake_pygem = types.ModuleType("pygem")
        fake_pygem.FFD = object
        fake_pygem.RBF = object
        dependencies = {
            "h5py": types.ModuleType("h5py"),
            "numpy": fake_numpy,
            "open3d": types.ModuleType("open3d"),
            "pygem": fake_pygem,
        }
        spec = importlib.util.spec_from_file_location(
            "create_corrupted_dataset",
            REPO / "datasets_mate/create_corrupted_dataset.py",
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, dependencies), mock.patch.object(
            sys, "argv", ["create_corrupted_dataset.py", "--dataset", "scanobjectnn"]
        ):
            spec.loader.exec_module(module)
            args = module.get_args()

        self.assertEqual(args.severity, 5)


if __name__ == "__main__":
    unittest.main()
