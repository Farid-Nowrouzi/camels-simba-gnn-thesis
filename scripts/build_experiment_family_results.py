#!/usr/bin/env python3
"""Build verified seed-level, aggregate, report, and figure family results."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment_pipeline.common import (
    PipelineError,
    format_markdown_table,
    load_family_spec,
    resolve_repo_path,
    sample_standard_deviation,
    verify_family,
    write_csv,
    write_text,
)


SEED_COLUMNS = (
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
    "target_mean",
    "target_std",
    "prediction_mean",
    "prediction_std",
    "prediction_std_ratio",
    "split_signature",
    "source_commit_if_available",
    "notes",
)

AGGREGATE_COLUMNS = (
    "grouping_field",
    "grouping_value",
    "seeds",
    "n_seeds",
    "mean_test_mae",
    "std_test_mae",
    "mean_test_rmse",
    "std_test_rmse",
    "mean_test_r2",
    "std_test_r2",
    "mean_test_pearson",
    "std_test_pearson",
    "mean_prediction_std_ratio",
    "std_prediction_std_ratio",
    "best_seed_by_mae",
    "best_test_mae",
    "worst_seed_by_mae",
    "worst_test_mae",
    "mean_best_epoch",
    "test_samples_per_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final results only after complete family verification."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Family JSON path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_rows(
    seed_rows: list[dict[str, Any]],
    grouping_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_values: dict[str, Any] = {}
    for row in seed_rows:
        key = str(row["grouping_value"])
        grouped[key].append(row)
        group_values[key] = row["grouping_value"]
    result: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["seed"]))
        metric_values = {
            metric: [float(row[metric]) for row in rows]
            for metric in (
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_pearson",
                "prediction_std_ratio",
            )
        }
        best = min(rows, key=lambda row: float(row["test_mae"]))
        worst = max(rows, key=lambda row: float(row["test_mae"]))
        result.append(
            {
                "grouping_field": grouping_field,
                "grouping_value": group_values[key],
                "seeds": ";".join(str(row["seed"]) for row in rows),
                "n_seeds": len(rows),
                "mean_test_mae": mean(metric_values["test_mae"]),
                "std_test_mae": sample_standard_deviation(metric_values["test_mae"]),
                "mean_test_rmse": mean(metric_values["test_rmse"]),
                "std_test_rmse": sample_standard_deviation(metric_values["test_rmse"]),
                "mean_test_r2": mean(metric_values["test_r2"]),
                "std_test_r2": sample_standard_deviation(metric_values["test_r2"]),
                "mean_test_pearson": mean(metric_values["test_pearson"]),
                "std_test_pearson": sample_standard_deviation(
                    metric_values["test_pearson"]
                ),
                "mean_prediction_std_ratio": mean(
                    metric_values["prediction_std_ratio"]
                ),
                "std_prediction_std_ratio": sample_standard_deviation(
                    metric_values["prediction_std_ratio"]
                ),
                "best_seed_by_mae": best["seed"],
                "best_test_mae": best["test_mae"],
                "worst_seed_by_mae": worst["seed"],
                "worst_test_mae": worst["test_mae"],
                "mean_best_epoch": mean([float(row["best_epoch"]) for row in rows]),
                "test_samples_per_seed": ";".join(
                    str(row["test_count"]) for row in rows
                ),
            }
        )
    return result


def create_figure(
    path: Path,
    aggregated: list[dict[str, Any]],
    figure: dict[str, Any],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib unavailable; figure was not generated.")
        return False
    x = [row["grouping_value"] for row in aggregated]
    y = [float(row[figure["y_metric"]]) for row in aggregated]
    error_key = figure.get("y_error_metric")
    yerr = [float(row[error_key]) for row in aggregated] if error_key else None
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    if figure.get("use_error_bars", True):
        axis.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    else:
        axis.plot(x, y, marker="o")
    axis.set_xlabel(figure["x_axis_label"])
    axis.set_ylabel(figure["y_axis_label"])
    axis.set_title(figure["title"])
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    try:
        repo_root = args.repo_root.expanduser().resolve()
        spec_path = resolve_repo_path(repo_root, args.spec)
        output_dir = resolve_repo_path(repo_root, args.output_dir)
        spec = load_family_spec(spec_path)
        verification = verify_family(repo_root, spec, allow_incomplete=False)
        if not verification.valid or not verification.complete:
            print("FAIL: complete verification is required; no result files were written.")
            for error in verification.errors:
                print(f"- {error}")
            return 1

        seed_rows = list(verification.rows)
        aggregated = aggregate_rows(seed_rows, spec["grouping_field"])
        group_order = {
            str(value): index for index, value in enumerate(spec["grouping_values"])
        }
        aggregated.sort(key=lambda row: group_order[str(row["grouping_value"])])
        result_names = spec["results"]
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_path = output_dir / result_names["seed_results_filename"]
        aggregate_path = output_dir / result_names["aggregated_results_filename"]
        plotting_path = output_dir / result_names["plotting_data_filename"]
        report_path = output_dir / result_names["markdown_report_filename"]
        figure_path = output_dir / result_names["figure_filename"]

        write_csv(seed_path, seed_rows, SEED_COLUMNS)
        write_csv(aggregate_path, aggregated, AGGREGATE_COLUMNS)
        plotting_columns = (
            "grouping_value",
            "mean_test_mae",
            "std_test_mae",
            "n_seeds",
        )
        write_csv(plotting_path, aggregated, plotting_columns)
        figure_created = create_figure(figure_path, aggregated, spec["figure"])
        report = "\n".join(
            [
                f"# {spec['family_title']}",
                "",
                spec["scientific_question"],
                "",
                "All rows passed independent artifact, configuration, split, and "
                "prediction verification before aggregation.",
                "",
                format_markdown_table(aggregated, AGGREGATE_COLUMNS),
                "",
                spec["figure"]["caption"],
                "",
                "Uncertainty is the sample standard deviation across independent "
                "seed-level metrics (denominator n-1); test samples are not pooled.",
            ]
        )
        write_text(report_path, report)
        for path in (seed_path, aggregate_path, plotting_path, report_path):
            print(f"Wrote: {path}")
        if figure_created:
            print(f"Wrote: {figure_path}")
        return 0
    except (PipelineError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
