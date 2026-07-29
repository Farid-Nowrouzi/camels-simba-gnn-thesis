#!/usr/bin/env python3
"""Safe sequential runner for a configured experiment family."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiment_pipeline.common import (
    PipelineError,
    build_training_command,
    filtered_runs,
    inspect_run,
    load_family_spec,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or sequentially execute a configured experiment family."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Family JSON path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and print commands without training (the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize sequential training.",
    )
    parser.add_argument("--only-group", default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def check_training_module(module_name: str, repo_root: Path) -> None:
    """Import the configured training module without invoking its CLI."""

    previous = Path.cwd()
    root_text = str(repo_root)
    inserted = root_text not in sys.path
    try:
        os.chdir(repo_root)
        if inserted:
            sys.path.insert(0, root_text)
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preflight must surface all import failures.
        raise PipelineError(f"training module import failed: {module_name}: {exc}") from exc
    finally:
        if inserted:
            sys.path.remove(root_text)
        os.chdir(previous)


def preflight(
    repo_root: Path,
    spec: dict[str, Any],
    runs: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], list[str]]], list[str]]:
    jobs: list[tuple[dict[str, Any], list[str]]] = []
    messages: list[str] = []
    failures: list[str] = []
    for run in runs:
        dataset = resolve_repo_path(repo_root, run["dataset_path"])
        if not dataset.is_file():
            failures.append(
                f"{run['experiment_name']}: dataset file is missing: {run['dataset_path']}"
            )
        inspection = inspect_run(repo_root, spec, run)
        label = (
            f"{spec['grouping_field']}={run['group_value']} seed={run['seed']} "
            f"{run['experiment_name']}"
        )
        if run["action"] == "exclude":
            messages.append(f"EXCLUDE {label}: {run.get('exclusion_reason', '')}")
        elif inspection.status == "complete":
            if inspection.compatibility != "compatible":
                failures.append(
                    f"{label}: completed folder is not compatible: "
                    f"{'; '.join(inspection.compatibility_errors)}"
                )
            else:
                messages.append(f"SKIP complete {label}")
        elif inspection.status == "partial":
            failures.append(
                f"{label}: partial folder exists; refusing to overwrite: "
                f"{run['experiment_path']} missing={list(inspection.missing_artifacts)}"
            )
        elif run["action"] == "reuse":
            failures.append(f"{label}: reusable experiment is missing and cannot be run")
        elif run["action"] == "run_if_missing":
            command = build_training_command(spec, run)
            jobs.append((run, command))
            messages.append(f"RUN missing {label}")
        else:
            failures.append(f"{label}: unsupported action/status combination")
    if failures:
        raise PipelineError("preflight failed:\n- " + "\n- ".join(failures))
    return jobs, messages


def execute_job(
    repo_root: Path,
    spec: dict[str, Any],
    run: dict[str, Any],
    command: list[str],
) -> None:
    log_root = resolve_repo_path(repo_root, spec["runner"]["log_directory"])
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run['experiment_name']}.log"
    time_prefix = ["/usr/bin/time", "-v"] if os.access("/usr/bin/time", os.X_OK) else []
    actual_command = [*time_prefix, *command]
    if time_prefix:
        timing_message = "GNU time: /usr/bin/time -v"
    else:
        timing_message = "GNU time unavailable; continuing without resource statistics"

    start = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        header = (
            f"Experiment: {run['experiment_name']}\n"
            f"Start UTC: {start}\n"
            f"{timing_message}\n"
            f"Command: {shlex.join(actual_command)}\n"
        )
        print(header, end="")
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            actual_command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        footer = (
            f"Finish UTC: {utc_now()}\n"
            f"Exit status: {return_code}\n"
        )
        print(footer, end="")
        log.write(footer)
    if return_code != 0:
        raise PipelineError(
            f"training failed for {run['experiment_name']} with exit status {return_code}"
        )
    inspection = inspect_run(repo_root, spec, run)
    if inspection.status != "complete":
        raise PipelineError(
            f"{run['experiment_name']} exited successfully but is "
            f"{inspection.status}; missing={list(inspection.missing_artifacts)}"
        )


def main() -> int:
    args = parse_args()
    try:
        repo_root = args.repo_root.expanduser().resolve()
        spec_path = resolve_repo_path(repo_root, args.spec)
        spec = load_family_spec(spec_path)
        runs = filtered_runs(spec, args.only_group, args.only_seed)
        check_training_module(spec["training_module"], repo_root)
        jobs, messages = preflight(repo_root, spec, runs)

        print(f"Family runner preflight: {spec['family_id']}")
        for message in messages:
            print(message)
        print(f"\nJobs selected: {len(jobs)}")
        for index, (run, command) in enumerate(jobs, start=1):
            print(
                f"{index}. {spec['grouping_field']}={run['group_value']} "
                f"seed={run['seed']}"
            )
            print(f"   {shlex.join(command)}")

        if not args.execute:
            print("\nDRY RUN: no training was executed.")
            return 0
        if not (os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")):
            raise PipelineError(
                "--execute requires an active virtualenv or conda environment"
            )
        if not jobs:
            print("\nNothing to execute; all selected runs are already complete or excluded.")
            return 0
        if shutil.which("python3") is None:
            raise PipelineError("python3 is unavailable")
        for run, command in jobs:
            execute_job(repo_root, spec, run, command)
        print("\nAll selected missing runs completed sequentially.")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
