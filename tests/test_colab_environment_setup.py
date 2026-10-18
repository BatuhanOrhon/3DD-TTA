"""Contracts for the restartable Colab environment setup notebook."""

import json
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SETUP_NOTEBOOK = REPO / "scripts/colab_environment_setup.ipynb"
BUILD_SCRIPT = REPO / "scripts/build_cuda_extensions_colab.sh"


def setup_code_cells():
    notebook = json.loads(SETUP_NOTEBOOK.read_text())
    return notebook, [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


class ColabEnvironmentSetupTests(unittest.TestCase):
    def test_setup_notebook_splits_native_build_into_restartable_phases(self):
        notebook, cells = setup_code_cells()
        self.assertEqual(len(cells), 10)
        self.assertEqual(len(notebook["cells"]), 11)
        self.assertTrue(all(cell["cell_type"] == "code" for cell in notebook["cells"][1:]))
        phases = ("prepare", "emd", "chamfer", "pointnet2", "pvcnn-smoke")
        for cell, phase in zip(cells[3:8], phases):
            source = "".join(cell["source"])
            self.assertIn("build_cuda_extensions_colab.sh", source)
            self.assertIn('"' + phase + '"', source)
        self.assertIn("modelnet40_c.zip", "".join(cells[8]["source"]))
        self.assertIn("SETUP_ENVIRONMENT_VERIFIED", "".join(cells[9]["source"]))

    def test_dependency_phase_installs_pinned_packages_in_order(self):
        expected_commands = [
            "!conda run -n 3dd_tta_env pip install "
            "torch==2.1.2+cu121 torchvision==0.16.2+cu121 "
            "torchaudio==2.1.2+cu121 "
            "--extra-index-url https://download.pytorch.org/whl/cu121",
            "!conda run -n 3dd_tta_env pip install "
            "/content/KNN_CUDA-0.2-py3-none-any.whl",
            "!conda run -n 3dd_tta_env pip install -r requirements.txt",
            "!conda run -n 3dd_tta_env pip install "
            "git+https://github.com/openai/CLIP.git",
        ]

        _, cells = setup_code_cells()
        source = "".join(cells[3]["source"])
        commands = [
            line for line in source.splitlines()
            if line.startswith("!conda run -n 3dd_tta_env pip install")
        ]
        self.assertEqual(commands[:4], expected_commands)

    def test_build_script_rejects_an_unknown_phase_before_environment_access(self):
        result = subprocess.run(
            ["bash", str(BUILD_SCRIPT), "not-a-phase"],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Unknown build phase: not-a-phase", result.stdout)


if __name__ == "__main__":
    unittest.main()
