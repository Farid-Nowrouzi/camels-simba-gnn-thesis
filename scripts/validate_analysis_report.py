#!/usr/bin/env python3
"""Validate a configuration-driven analysis without writing report outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis_reporting.common import validate_analysis
from experiment_pipeline.common import PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate complete experiment families, metrics, protocols, and paired "
            "split IDs without loading datasets or checkpoints."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Analysis JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = args.repo_root.expanduser().resolve()
        validation = validate_analysis(root, args.spec)
        family_count = len(validation.family_specs)
        input_count = sum(len(rows) for _, rows in validation.family_rows)
        print(f"Analysis: {validation.spec['analysis_name']}")
        print(f"Families: {family_count}")
        print(f"Verified input experiments: {input_count}")
        print(
            "Expected coverage: "
            f"{family_count} families × "
            f"{len(validation.spec['grouping_values'])} groups × "
            f"{len(validation.spec['required_seeds'])} seeds"
        )
        if validation.intentional_differences:
            print("Intentional protocol differences:")
            for item in validation.intentional_differences:
                print(f"- {item['field']}: {item['reason']}")
        if validation.errors:
            print("Unexpected differences or validation errors:", file=sys.stderr)
            for error in validation.errors:
                print(f"- {error}", file=sys.stderr)
            print("FAIL: report generation is blocked.", file=sys.stderr)
            return 1
        print("Split compatibility: PASS")
        print("Metric/artifact compatibility: PASS")
        print("PASS: analysis inputs are complete and scientifically compatible.")
        return 0
    except (PipelineError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
