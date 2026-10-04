"""Dry-run and safe orchestration contracts; no model or CUDA work."""
import io
import json
import random
import shutil
import subprocess
import sys
import unittest
import uuid
from dataclasses import asdict
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_gsd_composition as runner
from gsd_composition_protocol import (
    build_phase_plan, build_split_manifest, make_selection_manifest,
    validate_arm_bundle, write_arm_bundle, write_manifest,
)


class CompositionRunnerTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("composition-runner-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _write_all15_reference_fixture(self):
        indices = build_split_manifest()["replication_indices"]
        source_selection = make_selection_manifest(
            "routing", "R_SG", rationale="completed replication source",
            relevant_comparators=["C_SCD", "R_S0"],
        )
        phase_plan = build_phase_plan("replicate", source_selection)
        arm_configs = {arm.arm_id: asdict(arm.config) for arm in phase_plan.arms}
        runtime_fingerprint = "a" * 64
        input_sha256 = "b" * 64
        pairing_fingerprint = "c" * 64
        phase_fingerprint = "d" * 64
        logits = np.zeros((len(indices), 40), dtype=np.float32)
        logits[:, 0] = 1.0
        labels = np.zeros(len(indices), dtype=np.int64)
        reference = {
            "phase": "replicate", "status": "complete",
            "experiment_fingerprint": "ref-fingerprint",
            "config_sha256": phase_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "pairing_config_sha256": pairing_fingerprint,
            "selection_manifest": source_selection,
            "arms": [dict(arm_id=arm_id, config=config)
                     for arm_id, config in arm_configs.items()],
            "paired_blocks": [],
        }
        for corruption in ("gaussian", "impulse", "background", "shear"):
            for seed in (0, 1, 2):
                block_id = "{}-seed{}".format(corruption, seed)
                identity = dict(
                    fingerprint=runtime_fingerprint, runtime_fingerprint=runtime_fingerprint,
                    input_sha256=input_sha256, locked_config_fingerprint=pairing_fingerprint,
                    seed=seed, indices=list(indices),
                    preparation_keys=["e" * 64], classification_keys=["f" * 64],
                )
                records = []
                for arm_id, config in arm_configs.items():
                    bundle_path = self.root / "bundles" / corruption / str(seed) / arm_id
                    bundle_path.parent.mkdir(parents=True, exist_ok=True)
                    arm_manifest = dict(
                        run_id=arm_id, phase="replicate", arm_id=arm_id,
                        corruption=corruption, seed=seed, status="complete",
                        original_indices=list(indices), expected_indices=list(indices),
                        identity=identity, config=config,
                        experiment_fingerprint=phase_fingerprint,
                        runtime_fingerprint=runtime_fingerprint,
                        locked_config_fingerprint=pairing_fingerprint,
                        input_sha256=input_sha256, block_id=block_id,
                        scope="CPU validation fixture",
                    )
                    write_arm_bundle(bundle_path, arm_manifest, labels, labels, logits, [])
                    records.append(dict(arm_id=arm_id, status="complete",
                                        identity=identity, bundle_path=str(bundle_path)))
                reference["paired_blocks"].append(dict(
                    block_id=block_id, corruption=corruption, seed=seed,
                    indices=list(indices), identity=identity, arms=records,
                ))
        return reference

    def test_dry_run_never_enters_model_execution(self):
        with patch.object(runner, "_execute_plan", side_effect=AssertionError("execution called")):
            result = runner.main([
                "--phase", "smoke", "--result-root", str(self.root / "results")
            ])
        self.assertEqual(result, 0)

    def test_cli_scripts_start_from_repository_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
                "scripts/run_gsd_composition.py",
                "scripts/analyze_gsd_composition.py"):
            with self.subTest(script=relative_path):
                completed = subprocess.run(
                    [sys.executable, str(repo_root / relative_path), "--help"],
                    cwd=str(repo_root), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_phase_dry_runs_print_a_valid_scope_and_do_not_require_cuda(self):
        reference = self.root / "reference.json"
        for phase in ("smoke", "diagnose", "scale", "routing", "projection", "replicate", "all15"):
            args = ["--phase", phase, "--result-root", str(self.root / "results")]
            selection_data = None
            if phase != "smoke":
                reference_data = {"phase": runner.required_reference_phase(phase),
                                  "status": "complete",
                                  "experiment_fingerprint": "ref-fingerprint"}
            if phase in ("replicate", "all15"):
                selection = self.root / "selection.json"
                write_manifest(selection, make_selection_manifest(
                    "replicate" if phase == "all15" else "routing", "R_SG",
                    rationale="frozen CPU fixture",
                    relevant_comparators=["C_SCD"] if phase == "all15" else ["C_SCD", "R_S0"],
                ))
                selection_data = json.loads(selection.read_text(encoding="utf-8"))
                args.extend(["--selection-manifest", str(selection)])
                if phase == "all15":
                    reference_data = self._write_all15_reference_fixture()
            if phase != "smoke":
                write_manifest(reference, reference_data)
                args.extend(["--reference-manifest", str(reference)])
            with patch.object(runner, "_execute_plan", side_effect=AssertionError("execution called")):
                self.assertEqual(runner.main(args), 0)

    def test_replication_and_all15_reject_missing_selection_before_side_effects(self):
        reference = self.root / "reference.json"
        for phase in ("replicate", "all15"):
            write_manifest(reference, {"phase": runner.required_reference_phase(phase),
                                      "status": "complete",
                                      "experiment_fingerprint": "ref-fingerprint"})
            with self.assertRaises(ValueError):
                runner.build_plan([
                    "--phase", phase, "--result-root", str(self.root / "out"),
                    "--reference-manifest", str(reference),
                ])

    def test_reference_manifest_must_be_completed_and_phase_compatible(self):
        reference = self.root / "reference.json"
        write_manifest(reference, {"phase": "smoke", "status": "partial",
                                  "experiment_fingerprint": "ref-fingerprint"})
        with self.assertRaises(ValueError):
            runner.build_plan([
                "--phase", "diagnose", "--result-root", str(self.root / "out"),
                "--reference-manifest", str(reference),
            ])

    def test_all15_rejects_json_only_completed_reference_without_valid_bundles(self):
        reference = self._write_all15_reference_fixture()
        reference_path = self.root / "reference.json"
        reference["paired_blocks"][0]["arms"][0]["bundle_path"] = str(
            self.root / "missing-bundle")
        write_manifest(reference_path, reference)
        selection = make_selection_manifest(
            "replicate", "R_SG", rationale="frozen CPU fixture",
            relevant_comparators=["C_SCD", "R_S0"],
        )
        with self.assertRaisesRegex(ValueError, "invalid all15 reference arm bundle"):
            runner._check_reference("all15", selection, reference, reference_path)

    def test_plan_contains_locked_method_and_phase_independent_draw_seeds(self):
        plan = runner.build_plan([
            "--phase", "smoke", "--result-root", str(self.root / "out")
        ])
        self.assertEqual(plan["method"], "gsd_guidance_composition_v1")
        self.assertEqual(plan["phase"], "smoke")
        self.assertTrue(all(block["preparation_key"] != block["classification_key"]
                            for block in plan["blocks"]))
        json.dumps(plan, allow_nan=False)

    def test_pairing_config_fingerprint_is_stable_across_phases(self):
        reference = self.root / "reference.json"
        write_manifest(reference, {"phase": "smoke", "status": "complete",
                                  "experiment_fingerprint": "ref-fingerprint"})
        smoke = runner.build_plan([
            "--phase", "smoke", "--result-root", str(self.root / "smoke")
        ])
        diagnose = runner.build_plan([
            "--phase", "diagnose", "--result-root", str(self.root / "diagnose"),
            "--reference-manifest", str(reference),
        ])
        self.assertEqual(smoke["pairing_config_sha256"], diagnose["pairing_config_sha256"])
        self.assertNotEqual(smoke["config_sha256"], diagnose["config_sha256"])

    def test_injected_worker_runs_only_when_execute_is_requested(self):
        calls = []
        worker = lambda plan: calls.append(plan) or {"status": "complete"}
        args = ["--phase", "smoke", "--result-root", str(self.root / "results")]
        self.assertEqual(runner.main(args, worker=worker), 0)
        self.assertEqual(calls, [])
        self.assertEqual(runner.main(args + ["--execute"], worker=worker), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["phase"], "smoke")

    def test_completed_runner_manifest_is_accepted_as_next_phase_reference(self):
        result_root = self.root / "results"
        self.assertEqual(runner.main([
            "--phase", "smoke", "--result-root", str(result_root), "--execute"
        ], worker=lambda plan: {"status": "complete"}), 0)
        reference = next((result_root / "smoke").glob("attempt-*/phase_manifest.json"))
        self.assertEqual(runner.main([
            "--phase", "diagnose", "--result-root", str(result_root),
            "--reference-manifest", str(reference),
        ], worker=lambda plan: self.fail("dry-run executed worker")), 0)

    def test_dry_run_writes_unique_attempt_manifests_without_overwriting(self):
        args = ["--phase", "smoke", "--result-root", str(self.root / "results")]
        self.assertEqual(runner.main(args), 0)
        self.assertEqual(runner.main(args), 0)
        manifests = list((self.root / "results" / "smoke").glob("attempt-*/phase_manifest.json"))
        self.assertEqual(len(manifests), 2)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
        self.assertEqual({payload["attempt_id"] for payload in payloads},
                         {path.parent.name for path in manifests})

    def test_classification_seed_is_reset_before_postprocessing_and_classifier(self):
        calls = []
        points = object()

        def seed(value):
            calls.append(("seed", value))

        def postprocess(value):
            calls.append(("postprocess", value))
            return "processed"

        def classify(value):
            calls.append(("classify", value))
            return "logits"

        result = runner._postprocess_and_classify(
            points, 17, seed, postprocess, classify
        )
        self.assertEqual(result, "logits")
        self.assertEqual([call[0] for call in calls],
                         ["seed", "postprocess", "classify"])

    def test_preparation_rng_context_restores_python_numpy_torch_and_cuda_states(self):
        class FakeCuda:
            def __init__(self):
                self.states = [31, 32]

            @staticmethod
            def is_available():
                return True

            def get_rng_state_all(self):
                return list(self.states)

            def set_rng_state_all(self, states):
                self.states = list(states)

        class FakeTorch:
            def __init__(self):
                self.cpu_state = 21
                self.cuda = FakeCuda()

            def get_rng_state(self):
                return self.cpu_state

            def set_rng_state(self, state):
                self.cpu_state = state

        torch_mock = FakeTorch()
        random.seed(901)
        np.random.seed(902)
        expected_python = random.random()
        expected_numpy = np.random.random()
        random.seed(901)
        np.random.seed(902)
        with runner._preserve_rng_state(random, np, torch_mock):
            random.random()
            np.random.random()
            torch_mock.cpu_state = 99
            torch_mock.cuda.states = [98, 97]
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertEqual(torch_mock.cpu_state, 21)
        self.assertEqual(torch_mock.cuda.states, [31, 32])

    def test_graph_time_aggregates_each_graph_observer_event_once(self):
        events = [
            {"kind": "graph", "graph_runtime_seconds": 0.25},
            {"kind": "graph", "graph_runtime_seconds": None},
            {"kind": "graph", "graph_runtime_seconds": float("nan")},
            {"kind": "graph", "graph_runtime_seconds": float("inf")},
            {"kind": "step", "graph_runtime_seconds": 99.0},
            {"kind": "graph", "graph_runtime_seconds": 0.5},
        ]
        self.assertEqual(runner._graph_runtime_seconds(events), 0.75)

    def test_arm_draw_provenance_exposes_both_shared_draw_keys(self):
        block = {"preparation_key": "prep", "classification_key": "class"}
        self.assertEqual(runner._draw_key_fields(block), {
            "preparation_key": "prep", "classification_key": "class",
        })

    def test_failed_phase_manifest_records_partial_arm_artifacts(self):
        result_root = self.root / "results"
        partial = [{"arm_id": "C_SCD", "status": "partial",
                    "bundle_path": str(self.root / "partial-run")}]

        def worker(plan):
            plan["partial_arms"] = partial
            raise RuntimeError("injected batch failure")

        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "injected batch failure"):
                runner.main([
                    "--phase", "smoke", "--result-root", str(result_root), "--execute"
                ], worker=worker)
        manifest_path = next((result_root / "smoke").glob("attempt-*/phase_manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["partial_arms"], partial)

    def test_partial_arm_bundle_preserves_completed_subset_and_expected_indices(self):
        from dataclasses import asdict
        from gsd_composition import CompositionConfig

        scope_indices = build_split_manifest()["pilot_indices"][:32]
        manifest = {
            "run_id": "partial-run", "phase": "smoke", "arm_id": "C_SCD",
            "corruption": "gaussian", "seed": 0, "config": asdict(CompositionConfig()),
            "original_indices": scope_indices[:2], "expected_indices": scope_indices,
            "identity": {"fingerprint": "a" * 64}, "block_id": "b" * 64,
        }
        path = runner._persist_partial_arm_on_failure(
            self.root / "partial-run", manifest,
            labels=np.array([1, 2]), predictions=np.array([1, 3]),
            logits=np.eye(40, dtype=np.float32)[[1, 3]], diagnostics=[],
        )
        bundle = validate_arm_bundle(path)
        self.assertEqual(bundle["manifest"]["status"], "partial")
        self.assertEqual(bundle["manifest"]["original_indices"], scope_indices[:2])
        self.assertEqual(bundle["manifest"]["expected_indices"], scope_indices)
        self.assertEqual(bundle["indices"].tolist(), scope_indices[:2])
        self.assertIsNone(runner._persist_partial_arm_on_failure(
            self.root / "empty-partial", manifest, labels=np.array([]),
            predictions=np.array([]), logits=np.empty((0, 40), dtype=np.float32),
            diagnostics=[],
        ))


if __name__ == "__main__":
    unittest.main()
