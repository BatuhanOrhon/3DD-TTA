import io
import zipfile

import numpy as np
import pytest

from scripts import run_cloudfixer_surface_four as runner


class Runtime:
    labels = np.arange(10)
    identity = {"test": "fixed-scientific-identity", "frozen_states": {"test": "unchanged"}}

    def __init__(self):
        self.calls = []
        self.calibrations = []

    def calibrate(self, condition):
        self.calibrations.append(condition)
        return dict(selected_batch_size=4, trials=[dict(size=1, equivalent=True), dict(size=4, equivalent=True)],
                    independent_b1={"0": {"heads": {m: {"exact": True} for m in runner.MODELS}}})

    def predict(self, condition, ids):
        self.calls.append(tuple(ids))
        n = len(ids)
        logits = np.zeros((n, 4, 3, 40), dtype=np.float32)
        logits[:, :, :2, 1] = 3
        logits[:, :, 2, 2] = 2
        return dict(logits=logits, pm=np.ones((n, 4), dtype=int), terminal=np.full((n, 4), 2),
                    pose=np.tile(np.eye(3), (n, 1, 1)),
                    reason=np.repeat("unstable", n), angle_degrees=np.zeros(n),
                    denoise_seconds=.4, head_seconds=np.ones(4) * .1,
                    wall_seconds=1., peak_mib=123.,
                    input_sha256=["a"*64] * n, trajectory_sha256=["b"*64] * n)

    def verify_frozen(self):
        return {"test": "unchanged"}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "N_OBJECTS", 10)
    monkeypatch.setattr(runner, "CORRUPTIONS", ("rotation",))
    monkeypatch.setattr(runner, "BLOCK_SIZE", 8)
    runtime = Runtime()
    runner.run(runtime, tmp_path)
    return runtime, tmp_path


def test_recovery_reuses_verified_blocks_and_recomputes_only_bad_block(campaign):
    runtime, root = campaign
    assert runtime.calls == [(0, 1, 2, 3), (4, 5, 6, 7), (8,), (9,)]
    runner.run(runtime, root)
    assert len(runtime.calls) == 4
    first = root / "blocks/rotation/0000-0008.npz"
    last = root / "blocks/rotation/0008-0010.npz"
    original = first.read_bytes()
    last.write_bytes(b"truncated")
    runner.run(runtime, root)
    assert runtime.calls[4:] == [(8,), (9,)]
    assert runtime.calibrations == ["rotation"]
    assert first.read_bytes() == original
    assert (root / "quarantine").is_dir()


def test_independent_archive_recount_and_reject_semantic_damage(campaign):
    _, root = campaign
    path = runner.finalize(root)
    counts = runner.verify_archive(path)
    assert counts["model_object_records"] == 40
    assert counts["cells"]["dgcnn"]["rotation"]["pm_correct"] == 1
    assert counts["cells"]["pointmae"]["rotation"]["terminal_correct"] == 1
    first = root / "blocks/rotation/0000-0008.npz"
    with zipfile.ZipFile(path) as archive:
        assert archive.read(first.relative_to(root).as_posix()) == first.read_bytes()
    with np.load(first, allow_pickle=False) as saved:
        arrays = dict(saved)
    arrays["dgcnn_pm"][:] = 2
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    first.write_bytes(stream.getvalue())
    runner.sidecar(first).write_text(runner.sha(stream.getvalue())+"\n")
    with pytest.raises(ValueError, match="PM"):
        runner.finalize(root)


def test_duplicate_or_foreign_ranges_and_changed_campaign_are_rejected(campaign):
    runtime, root = campaign
    bad = root / "blocks/rotation/0001-0008.npz"
    bad.write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Unexpected"):
        runner.finalize(root)
    bad.unlink()
    runtime.identity = {"test": "changed"}
    with pytest.raises(ValueError, match="campaign"):
        runner.run(runtime, root)


def test_incomplete_archive_and_wrong_sha_are_rejected(campaign):
    _, root = campaign
    path = runner.finalize(root)
    runner.sidecar(path).write_text("0"*64+"\n")
    with pytest.raises(ValueError, match="SHA"):
        runner.verify_archive(path)
    path = root / "incomplete.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(root / "campaign.json", "campaign.json")
    runner.sidecar(path).write_text(runner.file_sha(path)+"\n")
    with pytest.raises(ValueError):
        runner.verify_archive(path)


def test_registration_sha_recovers_without_destroying_valid_blocks(campaign):
    runtime, root = campaign
    path = root / "campaign.json"
    expected = path.read_bytes()
    path.write_bytes(b"truncated")
    runner.run(runtime, root)
    assert len(runtime.calls) == 4
    assert path.read_bytes() == expected
    runner.sidecar(path).write_text("garbled\n")
    runner.run(runtime, root)
    assert len(runtime.calls) == 4
    assert runner.sidecar(path).read_text().strip() == runner.sha(expected)


def test_repack_produces_one_stable_content_addressed_archive(campaign, monkeypatch):
    _, root = campaign
    first = runner.finalize(root)
    original = zipfile.time.localtime
    monkeypatch.setattr(zipfile.time, "localtime", lambda *args: original(1600000000))
    assert runner.finalize(root) == first
    renamed = root / ("raw-"+"a"*64+".zip")
    first.rename(renamed)
    runner.sidecar(first).rename(runner.sidecar(renamed))
    with pytest.raises(ValueError, match="SHA"):
        runner.verify_archive(renamed)



def test_bad_live_gpu_decision_is_never_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "N_OBJECTS", 10)
    monkeypatch.setattr(runner, "CORRUPTIONS", ("rotation",))
    runtime = Runtime()
    original = runtime.predict

    def wrong_class(condition, ids):
        result = original(condition, ids)
        result["pm"][:, 0] = 7
        return result

    runtime.predict = wrong_class
    with pytest.raises(ValueError, match="PM"):
        runner.run(runtime, tmp_path)
    assert not list(tmp_path.glob("blocks/*/*.npz"))


def test_corrupt_profile_restores_measured_profile_and_reuses_valid_blocks(campaign):
    runtime, root = campaign
    path = root / "profiles/rotation.json"
    original = path.read_bytes()
    path.write_bytes(b"truncated")
    runner.run(runtime, root)
    assert runtime.calibrations == ["rotation"]
    assert len(runtime.calls) == 4
    assert path.read_bytes() == original



def test_unverified_calibration_choice_or_changed_frozen_states_cannot_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "N_OBJECTS", 10)
    monkeypatch.setattr(runner, "CORRUPTIONS", ("rotation",))
    runtime = Runtime()
    original = runtime.calibrate
    runtime.calibrate = lambda c: dict(original(c), selected_batch_size=8)
    with pytest.raises(ValueError, match="calibration"):
        runner.run(runtime, tmp_path)
    assert not runtime.calls
    runtime.calibrate = original
    runtime.verify_frozen = lambda: {"test": "changed"}
    with pytest.raises(ValueError, match="rozen"):
        runner.run(runtime, tmp_path)
    assert not (tmp_path / "frozen.json").exists()
