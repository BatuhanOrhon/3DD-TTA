"""Run/resume GSD v1 + SCD on all ModelNet40-C severity-5 corruptions in Colab."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import statistics
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research_artifacts import CORRUPTIONS, PER_COLUMNS, SUMMARY_COLUMNS

METHOD = "gsd_latent_spectral_v1"
RUN_PREFIX = "gsd-v1-scd-all15-m100"
FILES = {"config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
         "per_corruption.csv", "summary.csv"}
EXPECTED_EXAMPLES = 2468


def build_commands(result_root: str) -> list[list[str]]:
    """Build the fixed three-seed v1+SCD benchmark command matrix."""
    commands = []
    for seed in range(3):
        run_name = f"{RUN_PREFIX}-seed{seed}"
        commands.append([
            sys.executable, "-u", str(REPO / "run_baseline.py"),
            "--method", METHOD, "--batch_size", "32", "--seed", str(seed),
            "--severity", "5", "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01",
            "--max-batches", "0", "--lion-eval-mode", "--result-root", result_root,
            "--run-name", run_name, "--corruptions", *CORRUPTIONS,
            "--gsd-stage", "benchmark", "--gsd-weight", "1",
            "--gsd-scd-weight", "1", "--gsd-k", "10", "--gsd-delta", ".1",
            "--gsd-graph-gamma", ".6", "--gsd-modes", "100",
        ])
    return commands


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run(path: Path) -> dict:
    """Read a seven-file result ZIP and reject partial or malformed evidence."""
    path = Path(path)
    try:
        with ZipFile(path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in infos]
            parsed = [PurePosixPath(name) for name in names]
            if (len(names) != 7 or len(set(names)) != 7 or archive.testzip() is not None or
                    any(item.is_absolute() or ".." in item.parts or "\\" in name
                        for item, name in zip(parsed, names)) or
                    any(len(item.parts) != 2 for item in parsed) or
                    {item.name for item in parsed} != FILES or
                    len({item.parts[0] for item in parsed}) != 1):
                raise ValueError("Expected a safe, valid seven-file run ZIP: " + str(path))
            run_id = parsed[0].parts[0]
            contents = {item.name: archive.read(name) for item, name in zip(parsed, names)}
    except (BadZipFile, OSError) as error:
        raise ValueError("Cannot read run ZIP: " + str(path)) from error

    try:
        config = json.loads(contents["config.json"])
        per_rows = list(csv.DictReader(contents["per_corruption.csv"].decode("utf-8").splitlines()))
        summary_rows = list(csv.DictReader(contents["summary.csv"].decode("utf-8").splitlines()))
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("Run ZIP contains invalid JSON/CSV text: " + str(path)) from error
    if not per_rows or tuple(per_rows[0]) != PER_COLUMNS:
        raise ValueError("per_corruption.csv header mismatch: " + str(path))
    if not summary_rows or tuple(summary_rows[0]) != SUMMARY_COLUMNS or len(summary_rows) != 1:
        raise ValueError("summary.csv header/row mismatch: " + str(path))
    rows = {row["corruption"]: row for row in per_rows}
    if len(rows) != len(per_rows) or list(rows) != list(CORRUPTIONS):
        raise ValueError("Archive must contain the canonical all-15 corruption rows: " + str(path))
    if config.get("run_id") != run_id:
        raise ValueError("ZIP root and config run_id disagree: " + str(path))
    return {"path": str(path.resolve()), "sha256": _sha256(path), "config": config,
            "rows": rows, "summary": summary_rows[0]}


def _validate_run(run: dict, *, seed: int, run_name: str) -> None:
    config, rows, summary = run["config"], run["rows"], run["summary"]
    if (config.get("dataset") != "modelnet40_c" or config.get("severity") != 5 or
            config.get("method") != METHOD or config.get("seed") != seed or
            config.get("status") != "complete" or config.get("execution_status") != "complete" or
            config.get("stage") != "benchmark" or
            config.get("corruptions") != list(CORRUPTIONS) or
            config.get("completed_corruptions") != list(CORRUPTIONS)):
        raise ValueError("Run identity, stage, or completion mismatch: " + run["path"])
    cli = config.get("cli_args", {})
    locked = {"method": METHOD, "dataset_name": "modelnet-c", "severity": 5,
              "seed": seed, "batch_size": 32, "max_batches": 0,
              "lambdaa": .95, "gamma": .01, "eta": .01, "lion_eval_mode": True,
              "lion_ema_mode": False, "gsd_stage": "benchmark", "gsd_weight": 1.,
              "gsd_scd_weight": 1., "gsd_k": 10, "gsd_delta": .1,
              "gsd_graph_gamma": .6, "gsd_modes": 100,
              "corruptions": list(CORRUPTIONS), "run_name": run_name}
    for key, expected in locked.items():
        if cli.get(key) != expected:
            raise ValueError(f"Protocol mismatch for {key}: {run['path']}")
    if config.get("spectral", {}).get("weight") != 1. or config.get("spectral", {}).get("scd_weight") != 1.:
        raise ValueError("Serialized spectral contract is not v1 + SCD: " + run["path"])

    inventory = config.get("dataset_inventory", {})
    batches = config.get("observed_batch_sizes", {})
    for name in CORRUPTIONS:
        row = rows[name]
        n, correct = int(row["n_examples"]), int(row["n_correct"])
        accuracy = float(row["accuracy"])
        if (n != EXPECTED_EXAMPLES or not 0 <= correct <= n or
                row["status"] != "complete" or row["run_id"] != config["run_id"] or
                row["dataset"] != "modelnet40_c" or row["method"] != METHOD or
                int(row["severity"]) != 5 or int(row["seed"]) != seed or
                not math.isclose(accuracy, correct / n, rel_tol=0, abs_tol=1e-12) or
                inventory.get(name, {}).get("total_examples") != n or
                sum(batches.get(name, [])) != n):
            raise ValueError("Invalid full-file count/accuracy for " + name + ": " + run["path"])
    total_n = sum(int(rows[name]["n_examples"]) for name in CORRUPTIONS)
    total_correct = sum(int(rows[name]["n_correct"]) for name in CORRUPTIONS)
    macro = statistics.mean(float(rows[name]["accuracy"]) for name in CORRUPTIONS)
    if (summary.get("status") != "complete" or int(summary.get("n_corruptions", -1)) != 15 or
            int(summary.get("total_examples", -1)) != total_n or
            int(summary.get("total_correct", -1)) != total_correct or
            not math.isclose(float(summary["macro_accuracy"]), macro, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("summary.csv disagrees with complete per-corruption rows: " + run["path"])


def _method_root(result_root: str | Path) -> Path:
    root = Path(result_root)
    if not root.is_absolute():
        root = REPO / root
    return root.resolve() / "modelnet40_c" / METHOD


def _matching_runs(method_root: Path, base: str, seed: int) -> list[dict]:
    matches = []
    for path in sorted(method_root.glob("*_" + base + "*.zip")):
        logical = path.stem.split("_", 1)[-1]
        if logical != base and not logical.startswith(base + "-attempt"):
            continue
        try:
            run = _read_run(path)
            _validate_run(run, seed=seed, run_name=logical)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Incomplete/invalid archive retained: {path}: {error}", flush=True)
            continue
        matches.append(run)
    if len(matches) > 1:
        raise ValueError("Multiple valid archives exist for seed {}; resolve duplicates: {}".format(seed, base))
    return matches


def _check_sources(run: dict) -> None:
    from run_baseline import file_identity
    for name, identity in run["config"].get("runtime_source_manifest", {}).items():
        if file_identity(REPO / name) != identity:
            raise ValueError("Completed archive uses different source; refusing to mix: " + name)


def _free_run_name(base: str, method_root: Path) -> str:
    suffix = 1
    while True:
        name = base if suffix == 1 else f"{base}-attempt{suffix:02d}"
        if not list(method_root.glob("*_" + name + "*.zip")) and not list(method_root.glob("*_" + name)):
            return name
        suffix += 1


def _run_streamed(command: list[str]) -> tuple[int, str | None]:
    print("$ " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    zip_path = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("ZIP to provide:"):
            zip_path = line.split(":", 1)[1].strip()
    return process.wait(), zip_path


def _summarize(runs: list[dict]) -> dict:
    by_seed = {run["config"]["seed"]: run for run in runs}
    if sorted(by_seed) != [0, 1, 2]:
        raise ValueError("Summary requires exactly complete seed 0/1/2 archives")
    first = by_seed[0]["config"]
    for seed in (1, 2):
        config = by_seed[seed]["config"]
        for key in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest",
                    "extension_inventory", "dataset_inventory"):
            if config.get(key) != first.get(key):
                raise ValueError("Seed archives differ in " + key)

    def stats(values: list[float]) -> dict:
        return {"mean_percent": 100 * statistics.mean(values),
                "sample_sd_pp": 100 * statistics.stdev(values),
                "seed_percent": {str(seed): 100 * values[seed] for seed in range(3)}}

    per_corruption = {
        name: stats([float(by_seed[seed]["rows"][name]["accuracy"]) for seed in range(3)])
        for name in CORRUPTIONS
    }
    seed_macros = [float(by_seed[seed]["summary"]["macro_accuracy"]) for seed in range(3)]
    no_background = [name for name in CORRUPTIONS if name != "background"]
    seed_macros_no_background = [
        statistics.mean(float(by_seed[seed]["rows"][name]["accuracy"]) for name in no_background)
        for seed in range(3)
    ]
    return {
        "evidence_status": "three complete all-15 GSD v1 + SCD benchmark runs",
        "method": METHOD, "condition": "GSD v1 M=100, spectral weight=1, SCD weight=1",
        "dataset": "modelnet40_c", "severity": 5, "examples_per_corruption": EXPECTED_EXAMPLES,
        "corruptions": list(CORRUPTIONS), "seeds": [0, 1, 2],
        "macro_accuracy": stats(seed_macros),
        "macro_accuracy_excluding_background": stats(seed_macros_no_background),
        "per_corruption": per_corruption,
        "runs": [{"path": by_seed[seed]["path"], "sha256": by_seed[seed]["sha256"],
                  "run_id": by_seed[seed]["config"]["run_id"], "seed": seed,
                  "macro_accuracy": float(by_seed[seed]["summary"]["macro_accuracy"])}
                 for seed in range(3)],
        "git_commit": first.get("git_commit"),
    }


def _write_outputs(summary: dict, method_root: Path) -> tuple[Path, Path]:
    json_path = method_root / "gsd_v1_scd_all15_summary.json"
    csv_path = method_root / "gsd_v1_scd_all15_table.csv"
    json_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    table_rows = []
    for name in summary["corruptions"]:
        values = summary["per_corruption"][name]
        table_rows.append({"corruption": name, "seed0_percent": values["seed_percent"]["0"],
                           "seed1_percent": values["seed_percent"]["1"],
                           "seed2_percent": values["seed_percent"]["2"],
                           "mean_percent": values["mean_percent"],
                           "sample_sd_pp": values["sample_sd_pp"]})
    for label, macro in (("ALL15_MACRO", summary["macro_accuracy"]),
                         ("MACRO_EXCLUDING_BACKGROUND", summary["macro_accuracy_excluding_background"])):
        table_rows.append({"corruption": label, "seed0_percent": macro["seed_percent"]["0"],
                           "seed1_percent": macro["seed_percent"]["1"],
                           "seed2_percent": macro["seed_percent"]["2"],
                           "mean_percent": macro["mean_percent"],
                           "sample_sd_pp": macro["sample_sd_pp"]})
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("corruption", "seed0_percent", "seed1_percent",
                                                "seed2_percent", "mean_percent", "sample_sd_pp"))
    writer.writeheader()
    writer.writerows(table_rows)
    csv_text = buffer.getvalue()
    method_root.mkdir(parents=True, exist_ok=True)
    for path, encoded in ((json_path, json_text), (csv_path, csv_text)):
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Refusing to overwrite a different summary; review manually: " + str(path))
        if not path.exists():
            path.write_text(encoded, encoding="utf-8", newline="\n")
    return json_path, csv_path


def run_all(result_root: str) -> tuple[Path, Path]:
    commands = build_commands(result_root)
    method_root = _method_root(result_root)
    runs = []
    for command in commands:
        base = command[command.index("--run-name") + 1]
        seed = int(command[command.index("--seed") + 1])
        existing = _matching_runs(method_root, base, seed)
        if existing:
            _check_sources(existing[0])
            print("SKIP validated completion:", existing[0]["path"], flush=True)
            runs.extend(existing)
            continue
        name = _free_run_name(base, method_root)
        command[command.index("--run-name") + 1] = name
        code, zip_text = _run_streamed(command)
        if code or not zip_text:
            raise RuntimeError(f"Run stopped; exit={code}; rerun this launcher to resume safely")
        run = _read_run(Path(zip_text))
        _validate_run(run, seed=seed, run_name=name)
        _check_sources(run)
        runs.append(run)
    summary = _summarize(runs)
    outputs = _write_outputs(summary, method_root)
    print("Summary:", outputs[0], flush=True)
    print("Table:", outputs[1], flush=True)
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--dry-run", action="store_true", help="Print the three commands; do not run inference")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from /content/3DD-TTA in the existing 3dd_tta_env Colab checkout")
    if args.dry_run:
        for command in build_commands(args.result_root):
            print(shlex.join(command))
    else:
        run_all(args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
