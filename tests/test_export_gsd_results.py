import shutil
import unittest
from pathlib import Path

from scripts.export_gsd_results import export_archives


class ExportGsdResultsTests(unittest.TestCase):
    root = Path.cwd() / "tmp_export_gsd_tests"

    def setUp(self):
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_exports_all15_zip_into_matching_run_directory(self):
        root = self.root
        archive = root / "result" / "modelnet40_c" / "gsd_latent_spectral_v1" / (
            "20260923-120000_gsd-v1-all15-on-seed0.zip"
        )
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"immutable zip placeholder")
        drive = root / "drive" / "thesis" / "result"

        exported = export_archives(root / "result", drive, stage="all15")

        target = drive / "modelnet40_c" / "gsd_latent_spectral_v1" / archive.stem / archive.name
        self.assertEqual(exported, [target])
        self.assertEqual(target.read_bytes(), archive.read_bytes())

    def test_refuses_different_existing_archive(self):
        root = self.root
        archive = root / "result" / "modelnet40_c" / "gsd_latent_spectral_v1" / (
            "20260923-120000_gsd-v1-all15-on-seed0.zip"
        )
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"new")
        drive = root / "drive" / "thesis" / "result"
        target = drive / "modelnet40_c" / "gsd_latent_spectral_v1" / archive.stem / archive.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old")

        with self.assertRaises(FileExistsError):
            export_archives(root / "result", drive, stage="all15")

    def test_name_filter_exports_only_requested_band(self):
        root = self.root
        method = root / "result" / "modelnet40_c" / "gsd_latent_spectral_v1"
        method.mkdir(parents=True)
        for name in ("20260923_gsd-v1-pilot-on-seed0-m240.zip",
                     "20260923_gsd-v1-pilot-on-seed0-m400.zip",
                     "20260923_gsd-v1-pilot-on-seed0.zip"):
            (method / name).write_bytes(name.encode())
        exported = export_archives(root / "result", root / "drive", stage="pilot",
                                   name_contains=("m240", "m400"))
        self.assertEqual(len(exported), 2)
        self.assertTrue(all("m240" in p.name or "m400" in p.name for p in exported))


if __name__ == "__main__":
    unittest.main()
