"""Behavioral contract for preserving expensive Colab runtimes."""

import unittest
from pathlib import Path


SKILL = (
    Path(__file__).resolve().parents[1]
    / ".codex/skills/colab-research-operator/SKILL.md"
)


class ColabResearchOperatorSkillTests(unittest.TestCase):
    def test_runtime_teardown_requires_task_completion_and_close_permission(self):
        text = SKILL.read_text()

        required = (
            "## Runtime lifecycle and teardown",
            "both that the current task is complete and that the runtime may be closed",
            "This applies to agent-created and shared runtimes alike.",
            "Successful setup, completed experiments, artifact retrieval, an idle runtime, "
            "cost concerns, or the end of the current turn are not teardown authorization.",
            "A request to pause work is not permission to close the runtime.",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_idle_runtime_prompts_user_without_automatic_teardown(self):
        text = SKILL.read_text()

        required = (
            "## Idle-runtime reminder",
            "roughly 10 minutes",
            "remind the user that the named runtime is idle and may continue to incur cost",
            "ask whether they want it closed",
            "Ask for both task-completion and close permission in the same prompt",
            "Never close it merely because the reminder was sent or went unanswered.",
            "Send at most one reminder per continuous idle period",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_connection_resilience_preserves_real_work_without_fake_keepalive(self):
        text = SKILL.read_text()

        required = (
            "## Connection resilience and progress preservation",
            "Colab-managed maximum lifetime, quota, or compute-unit termination cannot be prevented",
            "Do not use fake clicks, JavaScript keep-alive loops, empty busy loops, or similar anti-idle workarounds.",
            "A local CLI or WebSocket disconnect is not evidence that the remote kernel or process stopped.",
            "reconnect to the same named session",
            "Persist checkpoints, manifests, logs, and intermediate results to Drive",
            "A mobile device or browser tab does not need to remain open after interactive authorization",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_long_runs_use_meaningful_heartbeats_not_print_cells(self):
        text = SKILL.read_text()

        required = (
            "Do not submit a second notebook cell merely to print while the kernel is busy",
            "emit periodic meaningful progress heartbeats from the running phase itself",
            "UTC timestamp, current phase, and last confirmed unit of progress",
            "A heartbeat improves observability; it does not guarantee runtime retention.",
            "forward child stdout and stderr into the notebook's IOPub stream",
            "Bare `subprocess.run` output may bypass the CLI stream",
            "query the server session inventory through a separate CLI connection",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_monitoring_does_not_treat_local_status_as_remote_truth(self):
        text = SKILL.read_text()

        required = (
            "`colab status` reports CLI-local bookkeeping",
            "Do not infer remote execution or completion from one `status` snapshot",
            "silence in the streamed output",
            "same primary `colab exec` process",
            "exact wrapper completion marker",
            "phase's on-disk `.complete` marker",
            "Use `colab sessions` only to confirm that the server assignment still exists",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_browser_url_requires_independent_runtime_verification(self):
        text = SKILL.read_text()

        required = (
            "`colab url -s SESSION`",
            "requests that the browser attach to an existing CLI-created runtime",
            "`empty.ipynb`",
            "does not display prior CLI cell output",
            "is not proof that the browser attached to the requested endpoint",
            "unexpected CPU assignment",
            "Do not run browser cells concurrently with a CLI-managed cell",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_fresh_runtime_uses_split_ten_cell_setup(self):
        text = SKILL.read_text()

        required = (
            "use the ten preparation code cells",
            "continue with cells2–10 in order",
            "Cells5–8 run the EMD, Chamfer, PointNet++ and PVCNN/smoke phases separately",
            "require each phase's exact completion marker before advancing",
            "Cell9 uses only",
            "Cell10 runs the",
            "embedded `scripts/verify_colab_environment.py` checks",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
