"""CPU checks for dataset identity and Drive round trips; no classifier/GPU."""
import importlib.util
import json
import shutil
import uuid
import unittest
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scanobjectnn_s5", REPO / "scripts/scanobjectnn_severity5_colab.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class ScanObjectNNDeliveryTests(unittest.TestCase):
    def setUp(self):
        test_root = REPO / "tmp/scanobjectnn_s5_tests"
        self.root = test_root / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        def cleanup():
            if self.root.resolve().parent != test_root.resolve():
                raise ValueError("Refusing cleanup outside the test workspace")
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        self.session = self.root / "session"
        self.session.mkdir()
        self.data = np.linspace(-0.7, 0.7, 581 * 2048 * 3,
                                dtype=np.float32).reshape(581, 2048, 3)
        self.labels = np.arange(581, dtype=np.int32) % 15

    def generate(self):
        return workflow.generate_dataset(
            self.data, self.labels, REPO / "result/preparation/scanobjectnn_author_generator_107305f.py",
            self.session, {"sha256": "0" * 64, "bytes": 1, "path": "synthetic.h5"})

    def test_generation_matches_sigma003_and_verified_drive_roundtrip(self):
        self.generate()
        dataset = self.session / "dataset"
        rng = np.random.RandomState(0)
        rng.choice(2048, 2048, replace=False)
        for cloud in self.data:
            rng.uniform(-0.05, 0.05, cloud.shape)
        indices = rng.choice(2048, 2048, replace=False)
        expected = np.stack([
            np.clip((cloud[indices] + rng.normal(size=(2048, 3)) * 0.03)
                    .astype("float32"), -1, 1) for cloud in self.data])
        np.testing.assert_array_equal(np.load(dataset / "data_gaussian_5.npy"), expected)
        np.testing.assert_array_equal(np.load(dataset / "data_original.npy"), self.data)
        drive = self.root / "drive"
        receipt = workflow.publish(self.session, drive)
        restored = workflow.restore(drive, self.root / "restored")
        np.testing.assert_array_equal(np.load(restored / "data_gaussian_5.npy"), expected)
        self.assertEqual(workflow.restore(drive, self.root / "restored"), restored)
        self.assertTrue(receipt["complete"])

    def test_wrong_severity_and_tampered_labels_are_rejected(self):
        self.generate()
        dataset = self.session / "dataset"
        path = dataset / "manifest.json"
        original = json.loads(path.read_text())
        wrong = dict(original, severity=8)
        path.write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):
            workflow.verify_dataset(dataset)
        path.write_text(json.dumps(original))
        np.save(dataset / "label.npy", self.labels[::-1])
        with self.assertRaises(ValueError):
            workflow.publish(self.session, self.root / "drive")
        self.assertFalse((self.root / "drive/latest_gaussian.json").exists())

    def test_failed_attempt_is_backed_up_without_promoting_latest(self):
        audit = self.session / "audit"
        audit.mkdir()
        (audit / "status.json").write_text('{"status":"failed"}')
        receipt = workflow.publish(self.session, self.root / "drive")
        self.assertFalse(receipt["complete"])
        self.assertFalse((self.root / "drive/latest_gaussian.json").exists())

    def test_unsafe_archive_path_is_rejected(self):
        path = self.root / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../outside.npy", b"bad")
        with zipfile.ZipFile(path) as archive:
            with self.assertRaises(ValueError):
                workflow.checked_members(archive)

    def test_modified_drive_zip_is_rejected_before_loading(self):
        self.generate()
        drive = self.root / "drive"
        receipt = workflow.publish(self.session, drive)
        archive = drive / receipt["archive_relative_path"]
        with archive.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            workflow.restore(drive, self.root / "restored")
        self.assertFalse((self.root / "restored").exists())

    def test_original_h5_selection_rejects_variant_and_ambiguous_paths(self):
        wrong = self.root / "main_split_nobg/test_objectdataset.h5"
        with self.assertRaises(ValueError):
            workflow.locate_h5(str(wrong), self.root, self.session)
        for name in ("first", "second"):
            path = self.root / name / "main_split/test_objectdataset.h5"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"placeholder")
        with self.assertRaises(ValueError):
            workflow.locate_h5("", self.root, self.session)


if __name__ == "__main__":
    unittest.main()
