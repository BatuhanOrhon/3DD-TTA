"""Run missing GSD v1 spectral-only Background and derive the all-15 macro."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from datetime import date
from zipfile import BadZipFile, ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research_artifacts import CORRUPTIONS, PER_COLUMNS, SUMMARY_COLUMNS

METHOD = "gsd_latent_spectral_v1"
OLD14 = REPO / "result/modelnet40_c/gsd_latent_spectral_v1/20260926-141029_gsd-v1-ablation14-on-seed0-spectral-only.zip"
OLD14_SHA256 = "a083cf795dc7e2b951fa80f47ea32e939cbeb8f9e41c027449b02431ee6b804a"
MUTABLE_OR_ADDITIVE_SOURCES = {"run_baseline.py", "gsd_protocol.py", "tta_gsd.py"}
FILES = {"config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
         "per_corruption.csv", "summary.csv"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_bundle(path: Path) -> dict:
    path = Path(path)
    try:
        with ZipFile(path) as archive:
            files = [i for i in archive.infolist() if not i.is_dir()]
            names = [i.filename for i in files]
            parsed = [PurePosixPath(name) for name in names]
            if (len(names) != 7 or len(set(names)) != 7 or archive.testzip() is not None or
                    any(p.is_absolute() or ".." in p.parts or "\\" in name for p, name in zip(parsed, names)) or
                    any(len(p.parts) != 2 for p in parsed) or {p.name for p in parsed} != FILES or
                    len({p.parts[0] for p in parsed}) != 1):
                raise ValueError("Expected a valid seven-file run ZIP: " + str(path))
            root = parsed[0].parts[0]
            contents = {p.name: archive.read(name) for p, name in zip(parsed, names)}
    except (BadZipFile, OSError) as error:
        raise ValueError("Cannot read run ZIP: " + str(path)) from error
    config = json.loads(contents["config.json"])
    per_rows = list(csv.DictReader(contents["per_corruption.csv"].decode("utf-8").splitlines()))
    summary_rows = list(csv.DictReader(contents["summary.csv"].decode("utf-8").splitlines()))
    if not per_rows or tuple(per_rows[0].keys()) != PER_COLUMNS:
        raise ValueError("per_corruption.csv header mismatch: " + str(path))
    if not summary_rows or tuple(summary_rows[0].keys()) != SUMMARY_COLUMNS:
        raise ValueError("summary.csv header mismatch: " + str(path))
    if config.get("run_id") != root:
        raise ValueError("ZIP root and config run_id disagree: " + str(path))
    rows = {row["corruption"]: row for row in per_rows}
    if len(rows) != len(per_rows):
        raise ValueError("Duplicate corruption rows: " + str(path))
    return {"path": str(path.resolve()), "sha256": _sha256(path), "config": config,
            "rows": rows, "summary": summary_rows[0], "contents": contents}


def _validate_rows(run: dict, expected: list[str]) -> None:
    config, rows = run["config"], run["rows"]
    if list(rows) != expected:
        raise ValueError("Unexpected corruption scope/order in " + run["path"])
    if (config.get("dataset") != "modelnet40_c" or config.get("severity") != 5 or
            config.get("method") != METHOD or config.get("seed") != 0 or
            config.get("status") != "complete" or config.get("execution_status") != "complete"):
        raise ValueError("Run identity/completion mismatch: " + run["path"])
    cli = config.get("cli_args", {})
    if cli.get("corruptions") != expected:
        raise ValueError("CLI corruption list disagrees with archive rows: " + run["path"])
    expected_stage = "background_completion" if expected == ["background"] else "ablation_no_background"
    if cli.get("gsd_stage") != expected_stage:
        raise ValueError("CLI stage disagrees with archive scope: " + run["path"])
    locked = {"method": METHOD, "dataset_name": "modelnet-c", "severity": 5, "seed": 0,
              "batch_size": 32, "max_batches": 0, "lambdaa": .95, "gamma": .01, "eta": .01,
              "lion_eval_mode": True, "lion_ema_mode": False,
              "gsd_weight": 1., "gsd_scd_weight": 0., "gsd_k": 10,
              "gsd_delta": .1, "gsd_graph_gamma": .6, "gsd_modes": 100}
    for key, value in locked.items():
        if cli.get(key) != value:
            raise ValueError("Locked protocol mismatch for " + key + ": " + run["path"])
    inventory = config.get("dataset_inventory", {})
    observed = config.get("observed_batch_sizes", {})
    for name in expected:
        row = rows[name]
        n, correct = int(row["n_examples"]), int(row["n_correct"])
        accuracy = float(row["accuracy"])
        if (n != 2468 or not 0 <= correct <= n or row["status"] != "complete" or
                row["run_id"] != config["run_id"] or row["dataset"] != "modelnet40_c" or
                row["method"] != METHOD or int(row["severity"]) != 5 or int(row["seed"]) != 0 or
                not math.isclose(accuracy, correct / n, rel_tol=0, abs_tol=1e-12) or
                inventory.get(name, {}).get("total_examples") != n or
                sum(observed.get(name, [])) != n):
            raise ValueError("Invalid complete-file counts/accuracy for " + name + ": " + run["path"])
    macro = sum(float(rows[name]["accuracy"]) for name in expected) / len(expected)
    summary = run["summary"]
    total_n = sum(int(rows[name]["n_examples"]) for name in expected)
    total_correct = sum(int(rows[name]["n_correct"]) for name in expected)
    if (summary.get("status") != "complete" or int(summary.get("n_corruptions", -1)) != len(expected) or
            int(summary.get("total_examples", -1)) != total_n or
            int(summary.get("total_correct", -1)) != total_correct or
            not math.isclose(float(summary["macro_accuracy"]), macro, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("summary.csv macro disagrees with per-corruption rows: " + run["path"])


def _validate_old_and_background(old: dict, background: dict) -> list[str]:
    expected14 = [name for name in CORRUPTIONS if name != "background"]
    _validate_rows(old, expected14)
    old_config = old["config"]
    if (old_config.get("stage") != "ablation_no_background" or
            old_config.get("completed_corruptions") != expected14):
        raise ValueError("Existing ZIP is not the expected ablation_no_background run")
    old_cli = old_config["cli_args"]
    if old_cli.get("gsd_stage") != "ablation_no_background":
        raise ValueError("Existing ZIP CLI stage mismatch")

    _validate_rows(background, ["background"])
    new_config = background["config"]
    if (new_config.get("stage") != "background_completion" or
            new_config["cli_args"].get("gsd_stage") != "background_completion"):
        raise ValueError("New ZIP is not the locked background_completion run")

    for key in ("asset_manifest",):
        if old_config.get(key) != new_config.get(key):
            raise ValueError("Old and new run differ in " + key)
    old_hashes = old_config.get("dataset_hash_manifest", {})
    new_hashes = new_config.get("dataset_hash_manifest", {})
    if set(old_hashes) != set(expected14) or set(new_hashes) != {"background"}:
        raise ValueError("Corruption hash manifests do not match the 14+1 scopes")
    for name, identity in {**old_hashes, **new_hashes}.items():
        data_file = REPO / "data/modelnet40_c" / Path(identity.get("path", "")).name
        if (not data_file.is_file() or data_file.stat().st_size != identity.get("bytes") or
                _sha256(data_file) != identity.get("sha256")):
            raise ValueError("Test-file identity no longer matches run archive: " + name)
    if not Path(new_hashes["background"].get("path", "")).name == "data_background_5.npy":
        raise ValueError("Background hash manifest points to an unexpected file")

    old_sources, new_sources = (old_config.get("runtime_source_manifest", {}),
                                new_config.get("runtime_source_manifest", {}))
    source_mismatches = sorted(
        name for name in set(old_sources) | set(new_sources)
        if name not in MUTABLE_OR_ADDITIVE_SOURCES and old_sources.get(name) != new_sources.get(name))
    if source_mismatches:
        raise ValueError("Unapproved inference-source changes: " + ", ".join(source_mismatches))
    if old_config.get("dataset_inventory", {}).keys() != set(expected14):
        raise ValueError("Existing archive inventory is incomplete")
    if set(new_config.get("dataset_inventory", {})) != {"background"}:
        raise ValueError("New archive must inventory only Background")
    return sorted(name for name in MUTABLE_OR_ADDITIVE_SOURCES
                  if old_sources.get(name) != new_sources.get(name))


def combine(old: dict, background: dict) -> dict:
    allowed_source_differences = _validate_old_and_background(old, background)
    rows = {**old["rows"], **background["rows"]}
    if set(rows) != set(CORRUPTIONS):
        raise ValueError("Combined evidence does not cover all 15 corruptions")
    macro = sum(float(rows[name]["accuracy"]) for name in CORRUPTIONS) / 15
    total_n = sum(int(rows[name]["n_examples"]) for name in CORRUPTIONS)
    total_correct = sum(int(rows[name]["n_correct"]) for name in CORRUPTIONS)
    env_old, env_new = old["config"].get("randomness", {}), background["config"].get("randomness", {})
    env_keys = ("torch_version", "cuda_version", "cudnn_version", "gpu")
    environment_matches = {key: env_old.get(key) == env_new.get(key) for key in env_keys}
    extension_matches = (old["config"].get("extension_inventory") ==
                         background["config"].get("extension_inventory"))
    return {
        "evidence_status": "derived 15-corruption composite: archived 14-corruption run plus a separate full Background run",
        "method": METHOD, "condition": "spectral-only v1, weight=1, SCD=0, M=100",
        "dataset": "modelnet40_c", "severity": 5, "seed": 0,
        "corruptions": list(CORRUPTIONS), "examples_per_corruption": 2468,
        "macro_accuracy": macro, "macro_accuracy_percent": 100 * macro,
        "micro_accuracy": total_correct / total_n, "total_correct": total_correct,
        "total_examples": total_n,
        "per_corruption": {name: {"n_correct": int(rows[name]["n_correct"]),
                                  "n_examples": int(rows[name]["n_examples"]),
                                  "accuracy": float(rows[name]["accuracy"])} for name in CORRUPTIONS},
        "source_archives": {"existing_14": {"path": old["path"], "sha256": old["sha256"]},
                            "background": {"path": background["path"], "sha256": background["sha256"]}},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "comparison_identity": {"shared_assets_equal": True, "archived_dataset_files_match_current": True,
                                "runtime_source_files_with_declared_changes": allowed_source_differences,
                                "environment_fields_equal": environment_matches,
                                "extension_inventory_equal": extension_matches,
                                "interpretation": "A reconstructed descriptive macro, not one contemporaneous 15-corruption run."}}


def _run_command(result_root: str) -> tuple[list[str], str]:
    run_name = "gsd-v1-background-completion-seed0-spectral-only-m100"
    command = [sys.executable, "-u", str(REPO / "run_baseline.py"),
               "--method", METHOD, "--batch_size", "32", "--seed", "0", "--severity", "5",
               "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01", "--max-batches", "0",
               "--lion-eval-mode", "--result-root", result_root, "--run-name", run_name,
               "--corruptions", "background", "--gsd-stage", "background_completion",
               "--gsd-weight", "1", "--gsd-scd-weight", "0", "--gsd-k", "10",
               "--gsd-delta", ".1", "--gsd-graph-gamma", ".6", "--gsd-modes", "100"]
    return command, run_name


def _find_background_archives(method_root: Path, base: str) -> list[dict]:
    matches = []
    for path in sorted(method_root.glob("*_" + base + "*.zip")):
        logical = path.stem.split("_", 1)[-1]
        if logical != base and not logical.startswith(base + "-attempt"):
            continue
        try:
            run = _read_bundle(path)
            _validate_rows(run, ["background"])
            if run["config"].get("stage") != "background_completion":
                continue
            matches.append(run)
        except (OSError, ValueError, KeyError) as error:
            print("Retaining invalid/incomplete prior attempt:", path, error, flush=True)
    if len(matches) > 1:
        raise ValueError("Multiple valid Background completions found; resolve duplicate archives first")
    return matches


def run_and_combine(existing14: Path, result_root: Path) -> Path:
    if _sha256(existing14) != OLD14_SHA256:
        raise ValueError("Existing 14-corruption archive hash does not match the audited seed-0 run")
    old = _read_bundle(existing14)
    # Validate the archived 14-corruption evidence before spending GPU time.
    _validate_rows(old, [name for name in CORRUPTIONS if name != "background"])
    method_root = result_root / "modelnet40_c" / METHOD
    existing = _find_background_archives(method_root, "gsd-v1-background-completion-seed0-spectral-only-m100")
    if existing:
        background = existing[0]
        print("Using validated Background completion:", background["path"], flush=True)
    else:
        command, base = _run_command(str(result_root))
        suffix = 1
        while list(method_root.glob("*_{0}*.zip".format(base))) or list(method_root.glob("*_{0}".format(base))):
            base = "gsd-v1-background-completion-seed0-spectral-only-m100-attempt{:02d}".format(suffix)
            command[command.index("--run-name") + 1] = base
            suffix += 1
        print("$ " + shlex.join(command), flush=True)
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        zip_text = None
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("ZIP to provide:"):
                zip_text = line.split(":", 1)[1].strip()
        code = process.wait()
        if code != 0 or not zip_text:
            raise RuntimeError("Background run failed; its artifact was retained for diagnosis")
        background = _read_bundle(Path(zip_text))
        _validate_rows(background, ["background"])
        if background["config"]["cli_args"].get("run_name") != base:
            raise ValueError("New ZIP run name disagrees with request")

    summary = combine(old, background)
    output = method_root / "spectral_only_m100_all15_summary_seed0.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise ValueError("Refusing to overwrite a different derived summary: " + str(output))
    output.write_text(encoded, encoding="utf-8")
    print("Wrote derived 15-corruption summary:", output, flush=True)
    _append_knowledge(summary, output)
    return output


def _append_knowledge(summary: dict, summary_path: Path) -> None:
    path = REPO / "knowledge/findings_log.md"
    marker = "<!-- gsd-v1-spectral-only-m100-all15-seed0 -->"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = [marker]
    lines.extend([
        "", "## {} GSD v1 spectral-only M100: reconstructed all-15 score".format(date.today().isoformat()),
        "", "[Run] Seed0, ModelNet40-C severity5, 2,468 examples for each of 15 corruptions. "
        "The score combines the archived 14-corruption spectral-only run with the separately "
        "completed full Background file; it is a derived composite, not one contemporaneous run.",
        "", "- Equal-corruption macro: {:.4f}% ({:.6f})".format(
            summary["macro_accuracy_percent"], summary["macro_accuracy"]),
        "- Micro accuracy: {:.4f}% ({}/{})".format(
            100 * summary["micro_accuracy"], summary["total_correct"], summary["total_examples"]),
        "- Background: {}/{} ({:.4f}%)".format(
            summary["per_corruption"]["background"]["n_correct"],
            summary["per_corruption"]["background"]["n_examples"],
            100 * summary["per_corruption"]["background"]["accuracy"]),
        "- Derived summary: `{}`".format(summary_path.relative_to(REPO).as_posix()),
        "- Code commit: `{}`".format(summary["git_commit"]),
        "- Source ZIPs and comparison identity are recorded in the JSON summary.", ""])
    block = "\n".join(lines)
    if marker in text:
        if block not in text:
            raise ValueError("Knowledge file already has a different completion record; review manually")
        print("Knowledge record already present; left unchanged.", flush=True)
        return
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n" + block)
    print("Appended validated result to:", path, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-14", type=Path, default=OLD14)
    parser.add_argument("--result-root", type=Path, default=REPO / "result")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from /content/3DD-TTA in the existing 3dd_tta_env Colab checkout")
    run_and_combine(args.existing_14.resolve(), args.result_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
