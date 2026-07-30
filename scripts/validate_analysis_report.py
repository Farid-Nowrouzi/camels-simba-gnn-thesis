#!/usr/bin/env python3
"""Validate a configuration-driven analysis without writing report outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from analysis_reporting.common import validate_analysis
from experiment_pipeline.common import PipelineError


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_generated_outputs(
    root: Path,
    spec: dict,
) -> tuple[list[str], dict[str, int]]:
    """Optionally validate a completed package when the spec declares outputs."""

    expected_counts = spec.get("expected_row_counts")
    required_files = spec.get("required_output_files")
    required_figures = spec.get("required_figures")
    if not any((expected_counts, required_files, required_figures)):
        return [], {}
    output_dir = root / str(spec["output_directory"])
    errors: list[str] = []
    observed_counts: dict[str, int] = {}
    count_files = {
        "seed_level_results": "seed_level_results.csv",
        "aggregated_results": "aggregated_results.csv",
        "paired_normalization_differences": "paired_normalization_differences.csv",
        "paired_model_differences": "paired_model_differences.csv",
        "paired_pooling_differences": "paired_pooling_differences.csv",
        "paired_head_differences": "paired_head_differences.csv",
        "prediction_diagnostics": "prediction_diagnostics.csv",
        "representative_runs": "representative_runs.csv",
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for key, expected in (expected_counts or {}).items():
        filename = count_files.get(key)
        if filename is None:
            errors.append(f"unsupported expected row-count key: {key}")
            continue
        path = output_dir / filename
        if not path.is_file():
            errors.append(f"missing counted output: {path.relative_to(root)}")
            continue
        rows = read_csv_rows(path)
        tables[key] = rows
        observed_counts[key] = len(rows)
        if len(rows) != int(expected):
            errors.append(f"{filename}: rows={len(rows)}, expected={expected}")

    for relative in required_files or ():
        path = output_dir / str(relative)
        if not path.is_file():
            errors.append(f"missing required output: {path.relative_to(root)}")
    for name in required_figures or ():
        for relative in (
            f"figures/{name}.png",
            f"figures/{name}.pdf",
            f"plot_data/{name}.csv",
        ):
            path = output_dir / relative
            if not path.is_file():
                errors.append(f"missing required figure artifact: {path.relative_to(root)}")

    seed_rows = tables.get("seed_level_results", [])
    if seed_rows:
        if len({row["experiment_path"] for row in seed_rows}) != len(seed_rows):
            errors.append("seed-level experiment mappings are not unique")
        if len({row["prediction_path"] for row in seed_rows}) != len(seed_rows):
            errors.append("seed-level prediction mappings are not unique")
        primary_fields = ("test_mae", "test_rmse", "test_mse", "test_r2")
        for row in seed_rows:
            if not all(math.isfinite(float(row[field])) for field in primary_fields):
                errors.append(f"non-finite primary metric: {row['experiment_name']}")
            if max(
                float(row["saved_mae_absolute_delta"]),
                float(row["saved_rmse_absolute_delta"]),
                float(row["saved_mse_absolute_delta"]),
            ) > float(spec.get("metric_tolerance", 1e-6)):
                errors.append(f"saved metric disagreement: {row['experiment_name']}")
            pearson = float(row["test_pearson"])
            status = row["pearson_status"]
            if status == "defined" and not math.isfinite(pearson):
                errors.append(f"unexplained undefined Pearson: {row['experiment_name']}")
            if status != "defined" and math.isfinite(pearson):
                errors.append(f"undefined Pearson represented as finite: {row['experiment_name']}")
        if spec.get("require_unique_prediction_hashes"):
            if len({row["prediction_sha256"] for row in seed_rows}) != len(seed_rows):
                errors.append("canonical prediction hashes are not unique")
        if spec.get("require_exact_paired_splits") or spec.get(
            "grouping_field"
        ) == "normalization":
            signatures: dict[int, set[str]] = {}
            for row in seed_rows:
                signatures.setdefault(int(row["seed"]), set()).add(
                    row["split_signature"]
                )
            if any(len(values) != 1 for values in signatures.values()):
                errors.append("seed-level split signatures do not match across all pairs")
            if spec.get("require_negative_r2_retention", True) and not any(
                float(row["test_r2"]) < 0 for row in seed_rows
            ):
                errors.append("negative-R² rows were not retained")
            if spec.get("grouping_field") == "normalization" and not any(
                float(row["exact_repeated_prediction_fraction"]) > 0
                for row in seed_rows
            ):
                errors.append("repeated-prediction rows were not retained")
        grouping_values = list(spec.get("grouping_values", ()))
        grouping_field = spec.get("grouping_field")
        if grouping_values and grouping_field:
            observed_order = list(dict.fromkeys(row[grouping_field] for row in seed_rows))
            if observed_order != grouping_values:
                errors.append(
                    f"{grouping_field} order={observed_order}, expected={grouping_values}"
                )

    manifest_path = output_dir / "analysis_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = set(manifest.get("generated_outputs", ()))
        actual = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        if listed != actual:
            errors.append(
                "analysis_manifest generated_outputs does not exactly list package files"
            )
        if seed_rows and spec.get("representative_seed_policy", {}).get(
            "policy"
        ) == "median_test_mae":
            specific_validation = manifest.get(
                "normalization_specific_validation"
            ) or manifest.get("graph_pooling_specific_validation") or manifest.get(
                "regression_head_specific_validation", {}
            )
            selected = {
                (
                    row["model"],
                    row[spec["grouping_field"]],
                ): int(row["seed"])
                for row in specific_validation.get("representative_runs", ())
            }
            expected_selected: dict[tuple[str, str], int] = {}
            grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in seed_rows:
                grouped.setdefault(
                    (row["model"], row[spec["grouping_field"]]), []
                ).append(row)
            for key, rows in grouped.items():
                ordered = sorted(
                    rows, key=lambda row: (float(row["test_mae"]), int(row["seed"]))
                )
                expected_selected[key] = int(ordered[len(ordered) // 2]["seed"])
            if selected != expected_selected:
                errors.append("representative runs do not follow median-test-MAE policy")
    elif expected_counts:
        errors.append(f"missing analysis manifest: {manifest_path.relative_to(root)}")
    return errors, observed_counts


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
        output_errors, output_counts = validate_generated_outputs(root, validation.spec)
        if output_errors:
            print("Generated-output validation errors:", file=sys.stderr)
            for error in output_errors:
                print(f"- {error}", file=sys.stderr)
            print("FAIL: generated analysis package is invalid.", file=sys.stderr)
            return 1
        print("Split compatibility: PASS")
        print("Metric/artifact compatibility: PASS")
        if output_counts:
            for name, count in output_counts.items():
                print(f"{name}: {count} rows")
            print("Generated tables/figures/manifest: PASS")
        print("PASS: analysis inputs are complete and scientifically compatible.")
        return 0
    except (PipelineError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
