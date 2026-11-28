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


def mixed_inputs(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runner, "N_OBJECTS", 10)
    monkeypatch.setattr(runner, "CORRUPTIONS", ("rotation",))
    monkeypatch.setattr(runner, "STRESS", ("rotation",))
    monkeypatch.setattr(runner, "BLOCK_SIZE", 8)
    old, new = tmp_path/"old", tmp_path/"new"
    def machine(name):
        r = Runtime()
        r.identity = dict(r.identity, hardware={"gpu": name, "total_bytes": 80 if name=="80GB" else 40, "torch": "pinned"})
        original = r.calibrate
        def calibrate(c):
            p = original(c)
            head = dict(exact=True, logits_sha256="c"*64, clouds_sha256="a"*64, pm=1, reason="unstable",
                        angle_degrees=0, probe_predictions=[1,1,2])
            item = dict(input_sha256="a"*64, heads={m:head for m in runner.MODELS})
            return dict(p, ids=[321,1146], independent_b1={"321":item,"1146":item})
        r.calibrate = calibrate
        return r
    a, b = machine("80GB"), machine("40GB")
    runner.run(a, old)
    runner.run(b, new, calibrate_only=True)
    for path in (old/"frozen.json", old/"blocks/rotation/0008-0010.npz"):
        path.unlink()
        runner.sidecar(path).unlink()
    ticket = dict(schema=1, status="passed", source_campaign_sha256=runner.file_sha(old/"campaign.json"),
                  target_campaign_sha256=runner.file_sha(new/"campaign.json"),
                  profiles={"rotation": {k+"_sha256":runner.file_sha(r/"profiles/rotation.json")
                            for k,r in (("source",old),("target",new))}})
    receipt = tmp_path/"parity.json"
    receipt.write_text(json.dumps(ticket))
    runner.sidecar(receipt).write_text(runner.file_sha(receipt)+"\n")
    return old, new, b, receipt, ticket


def test_mixed_raw_blocks_keep_original_provenance_and_interrupted_state(tmp_path,monkeypatch):
    old,new,b,receipt,ticket = mixed_inputs(tmp_path,monkeypatch)
    first = "blocks/rotation/0000-0008.npz"
    old_bytes = (old/first).read_bytes()
    runner.run(b,new,resume_from=old,parity_receipt=receipt)
    assert b.calls == [(8,), (9,)]
    assert (new/first).read_bytes() == old_bytes
    (new/first).write_bytes(b"truncated")
    runner.run(b,new,resume_from=old,parity_receipt=receipt)
    assert b.calls == [(8,), (9,)]
    with zipfile.ZipFile(runner.finalize(new)) as archive:
        assert archive.read(first) == old_bytes
        assert archive.read(first+".sha256") == runner.sidecar(old/first).read_bytes()
        prior = "segments/"+ticket["source_campaign_sha256"]+"/"
        assert archive.read(prior+"campaign.json") == (old/"campaign.json").read_bytes()
        assert prior+"frozen.json" not in archive.namelist()
    seal = runner.verify_archive(runner.finalize(new))
    old_seg,new_seg=(seal["hardware_segments"][ticket[k+"_campaign_sha256"]] for k in ("source","target"))
    assert old_seg["blocks"] == [first] and not old_seg["final_frozen_receipt"]
    assert new_seg["blocks"] == ["blocks/rotation/0008-0010.npz"] and new_seg["final_frozen_receipt"]
    assert old_seg["hardware"] != new_seg["hardware"] and seal["model_object_records"] == 40


def test_cross_device_decisions_must_match_even_with_fresh_valid_sidecars(tmp_path,monkeypatch):
    import json
    old,new,b,receipt,ticket = mixed_inputs(tmp_path,monkeypatch)
    p = new/"profiles/rotation.json"
    row = json.loads(p.read_bytes())
    row["independent_b1"]["321"]["heads"]["dgcnn"]["logits_sha256"] = "5"*64
    for dest in (p,new/"profile_backups/rotation.json"):
        dest.write_bytes(runner.canonical(row))
        runner.sidecar(dest).write_text(runner.file_sha(dest)+"\n")
    ticket["profiles"]["rotation"]["target_sha256"] = runner.file_sha(p)
    receipt.write_bytes(runner.canonical(ticket))
    runner.sidecar(receipt).write_text(runner.file_sha(receipt)+"\n")
    with pytest.raises(ValueError,match="parity"):
        runner.run(b,new,resume_from=old,parity_receipt=receipt)
    assert not b.calls and not list(new.glob("blocks/*/*.npz"))


def test_interrupted_import_can_resume_without_replaying_or_replacing_a_sealed_block(tmp_path,monkeypatch):
    old,new,b,receipt,ticket = mixed_inputs(tmp_path,monkeypatch)
    first = "blocks/rotation/0000-0008.npz"
    original = runner.restore_imported

    def interrupted(*args):
        raise InterruptedError("interrupted before first copy")

    monkeypatch.setattr(runner, "restore_imported", interrupted)
    with pytest.raises(InterruptedError):
        runner.run(b,new,resume_from=old,parity_receipt=receipt)
    assert not b.calls and (new/runner.TRANSFER).is_file()
    monkeypatch.setattr(runner, "restore_imported", original)
    runner.run(b,new,resume_from=old,parity_receipt=receipt)
    # Once membership is sealed, losing the old payload cannot relabel or replay the good imported copy.
    (old/first).unlink()
    runner.run(b,new,resume_from=old,parity_receipt=receipt)
    runner.run(b,new)
    assert b.calls == [(8,), (9,)]
    with np.load(new/"blocks/rotation/0008-0010.npz", allow_pickle=False) as block:
        assert block["writer_sha256"] == runner.file_sha(runner.Path(runner.__file__))
    mixed = runner.finalize(new)
    with zipfile.ZipFile(mixed, "a") as archive:
        archive.writestr("segments/"+ticket["source_campaign_sha256"]+"/frozen.json", b"{}")
    tampered = new/"tampered.zip"
    mixed.rename(tampered)
    runner.sidecar(tampered).write_text(runner.file_sha(tampered)+"\n")
    with pytest.raises(ValueError,match="segment"):
        runner.verify_archive(tampered)


def test_transfer_rejects_numeric_or_data_changes(tmp_path,monkeypatch):
    import json
    old,new,b,receipt,ticket = mixed_inputs(tmp_path,monkeypatch)
    old_meta = json.loads((old/"campaign.json").read_bytes())
    new_meta = json.loads((new/"campaign.json").read_bytes())
    for changed in (dict(new_meta, labels=[39]*10),
                    dict(new_meta, runtime=dict(new_meta["runtime"], hardware=dict(b.identity["hardware"], torch="other")))):
        with pytest.raises(ValueError,match="numeric runtime"):
            runner.check_campaign_pair(old_meta,changed,runner.file_sha(runner.Path(runner.__file__)))
