"""Count-based research records. Standard library only."""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

CORRUPTIONS = (
    "uniform", "gaussian", "background", "impulse", "upsampling",
    "distortion_rbf", "distortion_rbf_inv", "density", "density_inc",
    "shear", "rotation", "cutout", "distortion", "occlusion", "lidar",
)
CLEAN_CONTROL = "original"
SUPPORTED_INPUTS = CORRUPTIONS + (CLEAN_CONTROL,)
PER_COLUMNS = (
    "run_id", "dataset", "severity", "method", "seed", "corruption",
    "n_examples", "n_correct", "accuracy", "runtime_seconds",
    "peak_gpu_memory_mb", "status",
)
SUMMARY_COLUMNS = (
    "run_id", "dataset", "severity", "method", "seed", "n_corruptions",
    "macro_accuracy", "total_examples", "total_correct", "micro_accuracy",
    "total_runtime_seconds", "status",
)
DECODER_PER_COLUMNS = (
    "original_style_n_correct", "original_style_accuracy",
    "updated_style_n_correct", "updated_style_accuracy", "paired_delta_pp",
    "prediction_disagreement", "decoder_output_difference", "style_displacement",
)
DECODER_SUMMARY_COLUMNS = (
    "original_style_macro_accuracy", "original_style_micro_accuracy",
    "updated_style_macro_accuracy", "updated_style_micro_accuracy",
    "paired_delta_pp_macro", "paired_delta_pp_micro", "prediction_disagreement",
    "decoder_output_difference", "style_displacement",
)


def validate_selection(names: list[str]) -> list[str]:
    if not names or len(set(names)) != len(names) or set(names) - set(SUPPORTED_INPUTS):
        raise ValueError("Select unique supported inputs.")
    return list(names)


def corruption_row(run_id: str, seed: int, corruption: str, n_examples: int,
                   n_correct: int, runtime: float, memory: float, status: str,
                   *, method: str = "3dd_original", severity: int = 5, dataset: str = "modelnet40_c") -> dict:
    validate_selection([corruption])
    if type(n_examples) is not int or type(n_correct) is not int:
        raise ValueError("Counts must be integers.")
    if not 0 <= n_correct <= n_examples or status not in {"complete", "partial", "failed"}:
        raise ValueError("Invalid counts or status.")
    if status == "complete" and n_examples == 0:
        raise ValueError("An empty run cannot be complete.")
    if any(not math.isfinite(value) or value < 0 for value in (runtime, memory)):
        raise ValueError("Runtime/memory must be finite and non-negative.")
    if type(severity) is not int or severity < 0:
        raise ValueError("Severity must be a non-negative integer.")
    return dict(run_id=run_id, dataset=dataset, severity=severity,
                method=method, seed=seed, corruption=corruption,
                n_examples=n_examples, n_correct=n_correct,
                accuracy=n_correct / n_examples if n_examples else "",
                runtime_seconds=runtime, peak_gpu_memory_mb=memory, status=status)


def summarize(rows: list[dict], status: str | None = None) -> dict:
    if not rows:
        raise ValueError("Summary needs at least one row.")
    identity = ("run_id", "dataset", "severity", "method", "seed")
    if len({tuple(row[key] for key in identity) for row in rows}) != 1:
        raise ValueError("Cannot combine different runs/configurations.")
    if len({row["corruption"] for row in rows}) != len(rows):
        raise ValueError("Duplicate corruption rows.")
    checked = [corruption_row(row["run_id"], row["seed"], row["corruption"],
                              row["n_examples"], row["n_correct"],
                              row["runtime_seconds"], row["peak_gpu_memory_mb"],
                              row["status"], method=row["method"],
                              severity=row["severity"], dataset=row["dataset"]) for row in rows]
    n = sum(row["n_examples"] for row in checked)
    correct = sum(row["n_correct"] for row in checked)
    observed = [row["accuracy"] for row in checked if row["n_examples"]]
    resolved = ("failed" if any(row["status"] == "failed" for row in checked)
                else "partial" if any(row["status"] == "partial" for row in checked)
                else "complete")
    if status is not None:
        if status not in {"running", "complete", "partial", "failed"}:
            raise ValueError("Unknown summary status.")
        if status == "complete" and resolved != "complete":
            raise ValueError("Incomplete rows cannot form a complete run.")
        resolved = status
    summary = dict(**{key: rows[0][key] for key in identity},
                n_corruptions=len(rows),
                macro_accuracy=sum(observed) / len(observed) if observed else "",
                total_examples=n, total_correct=correct,
                micro_accuracy=correct / n if n else "",
                total_runtime_seconds=sum(row["runtime_seconds"] for row in checked),
                status=resolved)
    if "original_style_accuracy" in rows[0]:
        summary.update(decoder_control_summary(rows))
    return summary


def decoder_control_row(n_examples: int, original_correct: int, updated_correct: int,
                        disagreement_n: int, decoder_output_difference: float,
                        style_displacement: float) -> dict:
    """Return validated per-corruption metrics for the shared-style control."""
    if type(n_examples) is not int or type(original_correct) is not int or type(updated_correct) is not int:
        raise ValueError("Decoder-control counts must be integers.")
    if type(disagreement_n) is not int or not 0 <= disagreement_n <= n_examples:
        raise ValueError("Invalid decoder-control disagreement count.")
    if not (0 <= original_correct <= n_examples and 0 <= updated_correct <= n_examples):
        raise ValueError("Invalid decoder-control correct count.")
    if any(not math.isfinite(value) or value < 0 for value in
           (decoder_output_difference, style_displacement)):
        raise ValueError("Decoder-control distances must be finite and non-negative.")
    original_accuracy = original_correct / n_examples if n_examples else ""
    updated_accuracy = updated_correct / n_examples if n_examples else ""
    return dict(
        original_style_n_correct=original_correct,
        original_style_accuracy=original_accuracy,
        updated_style_n_correct=updated_correct,
        updated_style_accuracy=updated_accuracy,
        paired_delta_pp=(updated_accuracy - original_accuracy) * 100 if n_examples else "",
        prediction_disagreement=disagreement_n / n_examples if n_examples else "",
        decoder_output_difference=decoder_output_difference,
        style_displacement=style_displacement,
    )


def decoder_control_summary(rows: list[dict]) -> dict:
    """Aggregate shared-style metrics with explicit macro/micro semantics."""
    if not rows or any("original_style_accuracy" not in row for row in rows):
        raise ValueError("Decoder-control summary needs decoder-control rows.")
    total = sum(row["n_examples"] for row in rows)
    original_correct = sum(row["original_style_n_correct"] for row in rows)
    updated_correct = sum(row["updated_style_n_correct"] for row in rows)
    macro_original = sum(row["original_style_accuracy"] for row in rows) / len(rows)
    macro_updated = sum(row["updated_style_accuracy"] for row in rows) / len(rows)
    return dict(
        original_style_macro_accuracy=macro_original,
        original_style_micro_accuracy=original_correct / total if total else "",
        updated_style_macro_accuracy=macro_updated,
        updated_style_micro_accuracy=updated_correct / total if total else "",
        paired_delta_pp_macro=(macro_updated - macro_original) * 100,
        paired_delta_pp_micro=((updated_correct - original_correct) / total * 100) if total else "",
        prediction_disagreement=sum(row["prediction_disagreement"] * row["n_examples"] for row in rows) / total,
        decoder_output_difference=sum(row["decoder_output_difference"] * row["n_examples"] for row in rows) / total,
        style_displacement=sum(row["style_displacement"] * row["n_examples"] for row in rows) / total,
    )


class RunBundle:
    """Own one fresh directory; never resume or merge existing raw artifacts."""
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def create(cls, root: Path, name: str, config: dict, command: str,
               timestamp: str | None = None) -> RunBundle:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError("Run name must be lowercase ASCII, without paths/spaces.")
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if not re.fullmatch(r"\d{8}-\d{6}", stamp):
            raise ValueError("Invalid timestamp.")
        path = root.resolve() / config.get("dataset", "modelnet40_c") / config.get("method", "3dd_original") / (stamp + "_" + name)
        path.mkdir(parents=True, exist_ok=False)
        bundle = cls(path)
        bundle.write_config(dict(config, run_id=path.name, status="running", timestamp_timezone="UTC"))
        for filename, content in {
            "command.txt": command + "\n",
            "environment.txt": "Environment collection pending.\n",
            "stdout.log": "",
            "notes.md": "# Run notes\n\nInitialized; not yet completed.\n",
        }.items():
            (path / filename).write_text(content, encoding="utf-8")
        bundle._write_csv("per_corruption.csv", PER_COLUMNS, [])
        bundle.write_results([], "running")
        return bundle

    def write_config(self, config: dict) -> None:
        (self.path / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")

    def write_results(self, rows: list[dict], status: str | None = None) -> None:
        decoder_control = bool(rows and "original_style_accuracy" in rows[0])
        if rows:
            summary = summarize(rows, status)
        else:
            if status not in {"running", "failed"}:
                raise ValueError("Empty results must be running or failed.")
            config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
            summary = dict(
                run_id=config["run_id"], dataset=config.get("dataset", "modelnet40_c"), severity=config.get("severity", 5),
                method=config.get("method", "3dd_original"), seed=config["seed"], n_corruptions=0,
                macro_accuracy="", total_examples=0, total_correct=0, micro_accuracy="",
                total_runtime_seconds=0, status=status)
        per_columns = PER_COLUMNS + DECODER_PER_COLUMNS if decoder_control else PER_COLUMNS
        summary_columns = SUMMARY_COLUMNS + DECODER_SUMMARY_COLUMNS if decoder_control else SUMMARY_COLUMNS
        self._write_csv("per_corruption.csv", per_columns, rows)
        self._write_csv("summary.csv", summary_columns, [summary])

    def mark_failed(self) -> None:
        """Finalize this invocation after a worker crash, preserving recorded counts."""
        with (self.path / "summary.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row["status"] = "failed"
        columns = tuple(rows[0].keys()) if rows else SUMMARY_COLUMNS
        self._write_csv("summary.csv", columns, rows)

    def _write_csv(self, filename: str, columns: tuple[str, ...], rows: list[dict]) -> None:
        with (self.path / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
