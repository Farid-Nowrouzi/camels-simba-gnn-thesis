#!/usr/bin/env python3
"""Build a validated, configuration-driven scientific analysis package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis_reporting.common import build_analysis_package, validate_analysis
from experiment_pipeline.common import PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw experiment artifacts, then build tables, figures, "
            "provenance, compatibility evidence, and a scientific summary."
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
        if not validation.valid:
            print("FAIL: validation failed; no report outputs were generated.", file=sys.stderr)
            for error in validation.errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Validation: PASS")
        result = build_analysis_package(root, validation)
        print(f"Wrote analysis package: {result['output_directory']}")
        print(f"Seed-level rows: {result['seed_rows']}")
        print(f"Aggregated rows: {result['aggregate_rows']}")
        print(f"Paired rows: {result['paired_rows']}")
        print(f"Figures: {len(result['figures'])} × configured formats")
        return 0
    except (PipelineError, KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
