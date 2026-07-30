#!/usr/bin/env python3
"""Independent verifier for a configured controlled experiment family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiment_pipeline.common import (
    PipelineError,
    format_markdown_table,
    format_terminal_table,
    load_family_spec,
    resolve_repo_path,
    verify_family,
    write_csv,
    write_text,
)


RESULT_COLUMNS = (
    "family_id",
    "grouping_field",
    "grouping_value",
    "seed",
    "experiment_name",
    "experiment_path",
    "origin",
    "train_count",
    "val_count",
    "test_count",
    "best_epoch",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
    "pearson_status",
    "pearson_std_tolerance",
    "target_mean",
    "target_std",
    "prediction_mean",
    "prediction_std",
    "prediction_std_ratio",
    "unique_prediction_count",
    "exact_repeated_prediction_fraction",
    "approximate_unique_prediction_count",
    "approximate_repeated_prediction_fraction",
    "approximate_repeat_tolerance",
    "prediction_min",
    "prediction_max",
    "prediction_range",
    "residual_mean",
    "residual_std",
    "split_signature",
    "source_commit_if_available",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify artifacts, protocol, splits, and predictions for a family."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Family JSON path.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing/partial rows, while still failing incompatible completed rows.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_root = args.repo_root.expanduser().resolve()
        spec_path = resolve_repo_path(repo_root, args.spec)
        spec = load_family_spec(spec_path)
        result = verify_family(
            repo_root,
            spec,
            allow_incomplete=args.allow_incomplete,
        )
        rows = list(result.rows)

        print(f"Family verification: {spec['family_id']}")
        print(
            format_terminal_table(
                rows,
                (
                    "grouping_value",
                    "seed",
                    "experiment_name",
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "test_pearson",
                    "pearson_status",
                ),
            )
        )
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"- {warning}")
        if result.incomplete_messages:
            print("\nIncomplete rows:")
            for message in result.incomplete_messages:
                print(f"- {message}")
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"- {error}")

        total_required = sum(
            run.get("action") != "exclude" for run in spec["runs"]
        )
        if result.valid and result.complete:
            verdict = "PASS"
            print(f"\nPASS: all {total_required} required rows are verified.")
        elif result.valid and args.allow_incomplete:
            verdict = "INCOMPLETE"
            print(
                f"\nINCOMPLETE: {len(rows)}/{total_required} required rows verified; "
                "no final-family validity claim was made."
            )
        else:
            verdict = "FAIL"
            print("\nFAIL: family verification did not pass.")

        if args.output_csv is not None:
            output = resolve_repo_path(repo_root, args.output_csv)
            write_csv(output, rows, RESULT_COLUMNS)
            print(f"Wrote: {output}")
        if args.output_md is not None:
            output = resolve_repo_path(repo_root, args.output_md)
            parts = [
                f"# Verification: {spec['family_title']}",
                "",
                f"Verdict: **{verdict}**",
                "",
                format_markdown_table(rows, RESULT_COLUMNS),
            ]
            if result.incomplete_messages:
                parts.extend(
                    [
                        "",
                        "## Incomplete rows",
                        "",
                        *[f"- {message}" for message in result.incomplete_messages],
                    ]
                )
            if result.errors:
                parts.extend(
                    ["", "## Errors", "", *[f"- {error}" for error in result.errors]]
                )
            if result.warnings:
                parts.extend(
                    [
                        "",
                        "## Warnings",
                        "",
                        *[f"- {warning}" for warning in result.warnings],
                    ]
                )
            write_text(output, "\n".join(parts))
            print(f"Wrote: {output}")
        return 0 if result.valid else 1
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
