#!/usr/bin/env python3
"""Inspect live experiment-family status without training or dataset loading."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from experiment_pipeline.common import (
    PipelineError,
    format_markdown_table,
    format_terminal_table,
    inspect_family,
    load_family_spec,
    resolve_repo_path,
    write_csv,
    write_text,
)


STATUS_COLUMNS = (
    "grouping_value",
    "seed",
    "experiment_path",
    "origin",
    "action",
    "status",
    "artifacts",
    "compatibility",
    "runnable",
    "missing_artifacts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only live-status inspection for a configured experiment family."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Family JSON path.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_root = args.repo_root.expanduser().resolve()
        spec_path = resolve_repo_path(repo_root, args.spec)
        spec = load_family_spec(spec_path)
        inspections = inspect_family(repo_root, spec)
        inspection_errors = [
            error
            for item in inspections
            if item.compatibility == "inspection_error"
            for error in item.compatibility_errors
        ]
        if inspection_errors:
            raise PipelineError(
                "live-status inspection failed:\n- " + "\n- ".join(inspection_errors)
            )
        rows = []
        for item in inspections:
            rows.append(
                {
                    "grouping_value": item.grouping_value,
                    "seed": item.seed,
                    "experiment_path": item.experiment_path,
                    "origin": item.origin,
                    "action": item.action,
                    "status": item.status,
                    "artifacts": f"{item.artifact_count}/{item.artifact_total}",
                    "compatibility": item.compatibility,
                    "runnable": "yes" if item.runnable else "no",
                    "missing_artifacts": ";".join(item.missing_artifacts),
                }
            )
        counts = Counter(row["status"] for row in rows)
        runnable = sum(row["runnable"] == "yes" for row in rows)

        print(f"Family: {spec['family_id']}")
        print(f"Grouping field: {spec['grouping_field']}")
        print(format_terminal_table(rows, STATUS_COLUMNS))
        print()
        print(
            "Summary: "
            f"total_required={len(rows)} "
            f"complete={counts['complete']} "
            f"missing={counts['missing']} "
            f"partial={counts['partial']} "
            f"excluded={counts['excluded']} "
            f"runnable={runnable}"
        )

        if args.output_csv is not None:
            output = resolve_repo_path(repo_root, args.output_csv)
            write_csv(output, rows, STATUS_COLUMNS)
            print(f"Wrote: {output}")
        if args.output_md is not None:
            output = resolve_repo_path(repo_root, args.output_md)
            markdown = "\n".join(
                [
                    f"# Live Status: {spec['family_title']}",
                    "",
                    format_markdown_table(rows, STATUS_COLUMNS),
                    "",
                    (
                        f"Total required: {len(rows)}; complete: {counts['complete']}; "
                        f"missing: {counts['missing']}; partial: {counts['partial']}; "
                        f"excluded: {counts['excluded']}; runnable: {runnable}."
                    ),
                ]
            )
            write_text(output, markdown)
            print(f"Wrote: {output}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
