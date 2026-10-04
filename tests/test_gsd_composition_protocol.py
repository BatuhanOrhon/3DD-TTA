"""Pure protocol, split, random-key, and new-bundle contracts."""
import json
import shutil
import unittest
import uuid
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from gsd_composition_protocol import (
    EXPECTED_ARM_FILES, PILOT_CORRUPTIONS, ArmSpec, build_phase_plan, build_split_manifest,
    canonical_sha256, derive_draw_seeds, experiment_fingerprint,
    make_selection_manifest, validate_arm_bundle, validate_resume_block,
    validate_all15_reference, validate_selection_manifest, write_arm_bundle, write_arm_zip,
)


class CompositionProtocolTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("composition-protocol-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_phase_plans_lock_expected_arm_counts_modes_and_scopes(self):
        smoke = build_phase_plan("smoke")
        diagnose = build_phase_plan("diagnose")
        scale = build_phase_plan("scale")
        routing = build_phase_plan("routing")
        projection = build_phase_plan("projection")
        self.assertEqual([arm.arm_id for arm in smoke.arms], ["C_SCD", "C_SPEC", "C_SUM"])
        self.assertEqual(smoke.corruptions, ("gaussian", "background"))
        self.assertEqual(len(smoke.indices), 32)
        self.assertEqual(len(diagnose.indices), 64)
        self.assertEqual(len(scale.arms), 4)
        self.assertEqual(len(routing.arms), 4)
        self.assertEqual(len(projection.arms), 3)
        self.assertEqual(routing.arms[1].config.local_mode, "scd")
        self.assertEqual(routing.arms[1].config.style_mode, "spectral")
        self.assertEqual(projection.arms[1].config.style_mode, "pcgrad")
        selection = make_selection_manifest("routing", "R_SG", rationale="fixture",
                                            relevant_comparators=["C_SCD", "R_S0"])
        self.assertEqual(len(build_phase_plan("replicate", selection).indices), 256)
        self.assertEqual(build_phase_plan("replicate", selection).seeds, (0, 1, 2))
        scale_selection = make_selection_manifest(
            "scale", "SCALE_100", rationale="scale pilot winner",
            relevant_comparators=["SCALE_0", "SCALE_1"],
        )
        scale_replicate = build_phase_plan("replicate", scale_selection)
        self.assertEqual([arm.arm_id for arm in scale_replicate.arms],
                         ["SCALE_0", "SCALE_1", "SCALE_100"])
        self.assertEqual(len(scale_replicate.indices), 256)
        self.assertEqual(scale_replicate.seeds, (0, 1, 2))
        replicated_selection = make_selection_manifest(
            "replicate", "R_SG", rationale="replication follow-up",
            relevant_comparators=["C_SCD"],
        )
        self.assertEqual(len(build_phase_plan("all15", replicated_selection).corruptions), 15)

    def test_manifest_split_is_seeded_unique_and_serialized_with_hash(self):
        left = build_split_manifest()
        right = build_split_manifest()
        self.assertEqual(left, right)
        indices = left["indices"]
        self.assertEqual(len(indices), 2468)
        self.assertEqual(len(set(indices)), 2468)
        self.assertEqual(left["pilot_indices"], indices[:64])
        self.assertEqual(left["replication_indices"], indices[64:320])
        self.assertEqual(left["indices_sha256"], canonical_sha256(indices))

    def test_shared_draw_seeds_are_arm_phase_order_independent(self):
        indices = [12, 3, 99]
        first = derive_draw_seeds("modelnet40_c", "gaussian", 2, indices)
        second = derive_draw_seeds("modelnet40_c", "gaussian", 2, indices)
        self.assertEqual(first, second)
        self.assertNotEqual(first["preparation_seed"], first["classification_seed"])
        self.assertNotEqual(first, derive_draw_seeds("modelnet40_c", "gaussian", 1, indices))
        self.assertNotEqual(first, derive_draw_seeds("modelnet40_c", "gaussian", 2, list(reversed(indices))))

    def test_selection_manifest_freezes_candidate_and_rejects_invalid_local_projection(self):
        selection = make_selection_manifest(
            "routing", "R_SG", rationale="positive pilot contrast",
            relevant_comparators=["C_SCD", "R_S0"],
        )
        self.assertEqual(validate_selection_manifest(selection)["candidate_id"], "R_SG")
        bad = dict(selection)
        bad["candidate"] = dict(selection["candidate"], local_mode="pcgrad")
        with self.assertRaises(ValueError):
            validate_selection_manifest(bad)
        with self.assertRaises(ValueError):
            validate_selection_manifest(None)
        with self.assertRaises(ValueError):
            make_selection_manifest("projection", "P_PC", rationale="missing norm control",
                                    relevant_comparators=["C_SCD", "P_SUM"])
        with self.assertRaises(ValueError):
            make_selection_manifest("routing", "P_PC", rationale="wrong phase",
                                    relevant_comparators=["C_SCD", "P_SUM", "P_NORM"])

    def test_selection_rejects_resealed_noncanonical_candidate_and_wrong_phase(self):
        selection = make_selection_manifest("routing", "R_SG", rationale="fixture",
                                             relevant_comparators=["C_SCD", "R_S0"])
        tampered = dict(selection)
        tampered["candidate"] = dict(selection["candidate"], style_mode="sum")
        tampered["selection_sha256"] = canonical_sha256(
            {key: value for key, value in tampered.items() if key != "selection_sha256"})
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_selection_manifest(tampered)

        all15_tampered = dict(selection, source_phase="routing")
        all15_tampered["selection_sha256"] = canonical_sha256(
            {key: value for key, value in all15_tampered.items() if key != "selection_sha256"})
        with self.assertRaisesRegex(ValueError, "replication"):
            build_phase_plan("all15", all15_tampered)
        promoted = make_selection_manifest("replicate", "R_SG", rationale="replicated winner",
                                           relevant_comparators=["C_SCD"])
        self.assertEqual(len(build_phase_plan("all15", promoted).indices), 2468)

    def test_all15_requires_matching_candidate_and_complete_replication_blocks(self):
        selected = make_selection_manifest("replicate", "R_SG", rationale="fixture",
                                           relevant_comparators=["C_SCD"])
        replicated = make_selection_manifest("routing", "R_SG", rationale="fixture",
                                              relevant_comparators=["C_SCD", "R_S0"])
        def blocks_for(arm_ids):
            blocks = []
            for corruption in PILOT_CORRUPTIONS:
                for seed in (0, 1, 2):
                    indices = build_split_manifest()["replication_indices"]
                    identity = dict(runtime_fingerprint="a" * 64,
                                    input_sha256=canonical_sha256([corruption, seed]),
                                    locked_config_fingerprint="c" * 64,
                                    seed=seed, indices=indices)
                    block_id = "{}-{}".format(corruption, seed)
                    blocks.append(dict(
                        block_id=block_id, corruption=corruption, seed=seed,
                        indices=indices, identity=identity,
                        arms=[dict(arm_id=arm_id, status="complete", identity=identity,
                                   bundle_path="runs/{}/{}".format(block_id, arm_id))
                              for arm_id in arm_ids]))
            return blocks
        reference = dict(phase="replicate", status="complete", experiment_fingerprint="fp",
                         selection_manifest=replicated,
                         paired_blocks=blocks_for(["R_SG", "C_SCD", "R_S0"]))
        self.assertEqual(validate_all15_reference(selected, reference)["candidate_id"], "R_SG")
        changed_candidate = make_selection_manifest("routing", "R_GS", rationale="wrong candidate",
                                                    relevant_comparators=["C_SPEC", "R_G0"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_all15_reference(selected, dict(reference, selection_manifest=changed_candidate))
        shortened = dict(reference, paired_blocks=reference["paired_blocks"][:-1])
        with self.assertRaisesRegex(ValueError, "12"):
            validate_all15_reference(selected, shortened)
        missing_control = dict(reference, paired_blocks=[
            dict(block, arms=[dict(arm_id="R_SG", status="complete")])
            for block in reference["paired_blocks"]])
        with self.assertRaisesRegex(ValueError, "arm inventory"):
            validate_all15_reference(selected, missing_control)
        mismatched_identity = dict(reference, paired_blocks=blocks_for(["R_SG", "C_SCD", "R_S0"]))
        bad_block = mismatched_identity["paired_blocks"][0]
        bad_block["arms"][1]["identity"] = dict(bad_block["identity"], input_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_all15_reference(selected, mismatched_identity)
        missing_bundle = dict(reference, paired_blocks=blocks_for(["R_SG", "C_SCD", "R_S0"]))
        missing_bundle["paired_blocks"][0]["arms"][0]["bundle_path"] = ""
        with self.assertRaisesRegex(ValueError, "bundle_path"):
            validate_all15_reference(selected, missing_bundle)

        all15_scale = make_selection_manifest("replicate", "SCALE_100", rationale="scale winner",
                                              relevant_comparators=["C_SCD"])
        scale_replicated = make_selection_manifest("scale", "SCALE_100", rationale="scale winner",
                                                    relevant_comparators=["SCALE_0", "SCALE_1"])
        scale_reference = dict(reference, selection_manifest=scale_replicated,
                               paired_blocks=blocks_for(["SCALE_0", "SCALE_1", "SCALE_100"]))
        self.assertEqual(validate_all15_reference(all15_scale, scale_reference)["candidate_id"],
                         "SCALE_100")

    def test_fingerprint_is_canonical_and_changes_when_locked_setting_changes(self):
        left = experiment_fingerprint({"seed": 0, "config": {"weight": 1.0}})
        right = experiment_fingerprint({"config": {"weight": 1.0}, "seed": 0})
        self.assertEqual(left, right)
        self.assertNotEqual(left, experiment_fingerprint({"seed": 0, "config": {"weight": 2.0}}))

    def test_resume_requires_complete_arm_inventory_and_exact_identity(self):
        identity = {"fingerprint": "abc", "seed": 0, "indices": [1, 2], "input_sha256": "noise"}
        records = [dict(arm_id=arm, status="complete", identity=identity) for arm in ("C_SCD", "C_SPEC")]
        self.assertTrue(validate_resume_block(records, ("C_SCD", "C_SPEC"), identity))
        for changed in (
            records[:1],
            [records[0], dict(records[1], status="partial")],
        ):
            with self.assertRaises(ValueError):
                validate_resume_block(changed, ("C_SCD", "C_SPEC"), identity)
        for key, value in (("seed", 1), ("binary_sha256", "wrong-binary"),
                           ("loss_reduction", "mean"), ("weights", [1, 100]),
                           ("indices", [1, 3])):
            altered_identity = dict(identity, **{key: value})
            with self.assertRaises(ValueError):
                validate_resume_block(
                    [records[0], dict(records[1], identity=altered_identity)],
                    ("C_SCD", "C_SPEC"), identity,
                )

    def test_new_arm_bundle_has_exact_inventory_and_validates_predictions(self):
        path = self.root / "arm"
        from gsd_composition_protocol import _SPLIT
        labels = np.asarray([1, 2] * 16, dtype=np.int64)
        logits = np.zeros((32, 40), dtype=np.float32)
        logits[0, 1] = 5
        logits[1, 3] = 5
        predictions = labels.copy()
        predictions[1] = 3
        logits[2::2, 1] = 5
        logits[3::2, 2] = 5
        original_indices = _SPLIT["pilot_indices"][:32]
        manifest = dict(method="gsd_guidance_composition_v1", arm_id="C_SCD",
                        corruption="gaussian", seed=0, status="complete", phase="smoke",
                        original_indices=original_indices, input_sha256="input-a",
                        block_id="gaussian-seed0-attempt01", runtime_fingerprint="runtime-a",
                        config=dict(local_mode="scd", style_mode="scd",
                                    local_scd_weight=1.0, local_spectral_weight=1.0,
                                    style_scd_weight=1.0, style_spectral_weight=1.0,
                                    norm_floor=1e-12, schema_version=1))
        write_arm_bundle(path, manifest, labels, predictions, logits, [])
        self.assertEqual({item.name for item in path.iterdir()}, EXPECTED_ARM_FILES)
        checked = validate_arm_bundle(path)
        self.assertEqual(checked["n_examples"], 32)
        self.assertEqual(checked["n_correct"], 31)
        self.assertEqual(checked["manifest"]["method"], "gsd_guidance_composition_v1")
        raw_hashes = {item.name: item.read_bytes() for item in path.iterdir()}
        archive = write_arm_zip(path)
        zipped = validate_arm_bundle(archive)
        self.assertEqual(zipped["n_correct"], 31)
        self.assertEqual({item.name: item.read_bytes() for item in path.iterdir()}, raw_hashes)
        archived_bytes = archive.read_bytes()
        with self.assertRaises(FileExistsError):
            write_arm_zip(path, archive)
        self.assertEqual(archive.read_bytes(), archived_bytes)

    def test_complete_bundle_must_match_phase_scope_but_partial_may_be_short(self):
        from gsd_composition_protocol import _SPLIT
        indices = _SPLIT["pilot_indices"][:2]
        labels = np.asarray([1, 2], dtype=np.int64)
        predictions = labels.copy()
        logits = np.zeros((2, 40), dtype=np.float32)
        logits[0, 1] = logits[1, 2] = 5
        manifest = dict(method="gsd_guidance_composition_v1", arm_id="C_SCD",
                        corruption="gaussian", seed=0, status="complete", phase="smoke",
                        original_indices=indices, input_sha256="input-a",
                        block_id="gaussian-seed0-attempt01", runtime_fingerprint="runtime-a",
                        config=dict(local_mode="scd", style_mode="scd",
                                    local_scd_weight=1.0, local_spectral_weight=1.0,
                                    style_scd_weight=1.0, style_spectral_weight=1.0,
                                    norm_floor=1e-12, schema_version=1))
        complete_path = self.root / "short-complete"
        write_arm_bundle(complete_path, manifest, labels, predictions, logits, [])
        with self.assertRaisesRegex(ValueError, "phase scope"):
            validate_arm_bundle(complete_path)
        partial_path = self.root / "short-partial"
        write_arm_bundle(partial_path, dict(manifest, status="partial"), labels, predictions, logits, [])
        self.assertEqual(validate_arm_bundle(partial_path)["n_examples"], 2)

    def test_complete_bundle_scope_is_locked_for_every_phase(self):
        from gsd_composition_protocol import CORRUPTIONS, PILOT_CORRUPTIONS, SMOKE_CORRUPTIONS, _SPLIT
        phase_scope = {
            "smoke": (SMOKE_CORRUPTIONS, _SPLIT["pilot_indices"][:32]),
            "diagnose": (PILOT_CORRUPTIONS, _SPLIT["pilot_indices"]),
            "scale": (PILOT_CORRUPTIONS, _SPLIT["pilot_indices"]),
            "routing": (PILOT_CORRUPTIONS, _SPLIT["pilot_indices"]),
            "projection": (PILOT_CORRUPTIONS, _SPLIT["pilot_indices"]),
            "replicate": (PILOT_CORRUPTIONS, _SPLIT["replication_indices"]),
            "all15": (CORRUPTIONS, list(range(2468))),
        }
        route = dict(local_mode="scd", style_mode="scd", local_scd_weight=1.0,
                     local_spectral_weight=1.0, style_scd_weight=1.0,
                     style_spectral_weight=1.0, norm_floor=1e-12, schema_version=1)
        for phase, (corruptions, expected_indices) in phase_scope.items():
            count = len(expected_indices)
            labels = np.zeros(count, dtype=np.int64)
            logits = np.zeros((count, 40), dtype=np.float32)
            manifest = dict(method="gsd_guidance_composition_v1", arm_id="C_SCD",
                            corruption=corruptions[0], seed=0, status="complete", phase=phase,
                            original_indices=expected_indices, input_sha256="input-a",
                            block_id="{}-seed0-block".format(phase), runtime_fingerprint="runtime-a",
                            config=route)
            path = self.root / ("phase-" + phase)
            write_arm_bundle(path, manifest, labels, labels, logits, [])
            self.assertEqual(validate_arm_bundle(path)["n_examples"], count)

    def test_rejects_unsafe_archive_paths(self):
        archive = self.root / "unsafe.zip"
        with ZipFile(archive, "w") as file:
            file.writestr("../escape.txt", "bad")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            validate_arm_bundle(archive)


if __name__ == "__main__":
    unittest.main()
