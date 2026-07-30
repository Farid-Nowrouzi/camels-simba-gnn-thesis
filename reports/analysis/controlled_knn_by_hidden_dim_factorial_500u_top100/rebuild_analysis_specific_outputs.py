#!/usr/bin/env python3
"""Rebuild and validate the kNN by hidden-dimension factorial outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_FROM_SCRIPT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_FROM_SCRIPT / "scripts"))

from analysis_reporting.common import validate_analysis  # noqa: E402
from experiment_pipeline.common import (  # noqa: E402
    APPROXIMATE_REPEAT_TOLERANCE,
    PEARSON_STD_TOLERANCE,
    PipelineError,
    format_markdown_table,
    read_json,
    read_prediction_rows,
    resolve_repo_path,
    sample_standard_deviation,
    verify_family,
    write_csv,
    write_text,
)

MODELS = ("EvolveGCN-H", "Static GCN")
WIDTHS = (32, 64)
K_VALUES = (4, 6, 8, 12)
SEEDS = (42, 123, 2025)
PRIMARY_METRICS = ("test_mae", "test_rmse", "test_r2")
TOLERANCE = 1e-6
EXPECTED_SPLIT_SIGNATURES = {
    42: "0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748",
    123: "853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266",
    2025: "3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate and validate all factorial-specific tables and figures."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPOSITORY_FROM_SCRIPT,
        help="Repository root.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(
            "configs/analysis_reports/"
            "controlled_knn_by_hidden_dim_factorial_500u_top100.json"
        ),
        help="Factorial analysis specification.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def sd(values: Iterable[float]) -> float:
    return sample_standard_deviation(list(values))


def finite_summary(values: Iterable[float]) -> tuple[float | None, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return (
        statistics.mean(finite) if finite else None,
        sample_standard_deviation(finite) if finite else None,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def markdown_and_latex(
    base: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    def latex_value(value: Any) -> str:
        return (
            str(value)
            .replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("±", "$\\pm$")
        )

    write_csv(base.with_suffix(".csv"), rows, columns)
    write_text(base.with_suffix(".md"), format_markdown_table(rows, columns))
    latex = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(latex_value(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        latex.append(
            " & ".join(latex_value(row.get(column, "")) for column in columns)
            + " \\\\"
        )
    latex.extend(("\\hline", "\\end{tabular}"))
    write_text(base.with_suffix(".tex"), "\n".join(latex))


def parse_cell(value: str) -> tuple[int, int]:
    try:
        width_text, k_text = value.split("_")
        return int(width_text.removeprefix("h")), int(k_text.removeprefix("k"))
    except (ValueError, AttributeError) as exc:
        raise PipelineError(f"invalid factorial cell: {value!r}") from exc


def load_rows(
    root: Path, validation: Any
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int, int], tuple[list[float], list[float]]]]:
    seed_rows: list[dict[str, Any]] = []
    prediction_pairs: dict[
        tuple[str, int, int, int], tuple[list[float], list[float]]
    ] = {}
    family_by_label = {
        label: family for label, _, family in validation.family_specs
    }
    for label, _, family in validation.family_specs:
        result = verify_family(root, family, allow_incomplete=False)
        if not result.valid or not result.complete:
            raise PipelineError(f"{label} family verification failed: {result.errors}")
        run_by_name = {run["experiment_name"]: run for run in family["runs"]}
        for verified in result.rows:
            width, k_value = parse_cell(str(verified["grouping_value"]))
            if (
                int(verified["train_count"]),
                int(verified["val_count"]),
                int(verified["test_count"]),
            ) != (350, 75, 75):
                raise PipelineError(
                    f"unexpected factorial split counts: {verified['experiment_name']}"
                )
            run = run_by_name[verified["experiment_name"]]
            experiment_dir = resolve_repo_path(root, run["experiment_path"])
            prediction_path = experiment_dir / family["prediction_file"]
            ids, targets, predictions = read_prediction_rows(
                prediction_path,
                family["target_column_aliases"],
                family["prediction_column_aliases"],
                family.get("id_column_aliases", ("universe_id",)),
            )
            config = read_json(experiment_dir / "config.json")
            if ids != config["test_ids"]:
                raise PipelineError(f"ordered IDs changed after verification: {prediction_path}")
            for artifact in family["expected_artifacts"]:
                artifact_path = experiment_dir / artifact
                if not artifact_path.is_file():
                    raise PipelineError(f"missing required artifact: {artifact_path}")
                try:
                    if artifact.endswith(".pt"):
                        with artifact_path.open("rb") as handle:
                            handle.read(1)
                    else:
                        with artifact_path.open("r", encoding="utf-8") as handle:
                            handle.read(1)
                except OSError as exc:
                    raise PipelineError(f"unreadable artifact: {artifact_path}: {exc}") from exc
            row = {
                "model": label,
                "hidden_dim": width,
                "k": k_value,
                "seed": int(verified["seed"]),
                "factorial_cell": verified["grouping_value"],
                "experiment_name": verified["experiment_name"],
                "experiment_path": verified["experiment_path"],
                "prediction_path": prediction_path.relative_to(root).as_posix(),
                "split_signature": verified["split_signature"],
                "test_count": int(verified["test_count"]),
                "test_mae": verified["test_mae"],
                "test_rmse": verified["test_rmse"],
                "test_mse": verified["test_mse"],
                "test_r2": verified["test_r2"],
                "test_pearson": (
                    verified["test_pearson"]
                    if math.isfinite(float(verified["test_pearson"]))
                    else ""
                ),
                "pearson_status": verified["pearson_status"],
                "target_mean": verified["target_mean"],
                "prediction_mean": verified["prediction_mean"],
                "target_std": verified["target_std"],
                "prediction_std": verified["prediction_std"],
                "prediction_std_ratio": verified["prediction_std_ratio"],
                "unique_prediction_count": verified["unique_prediction_count"],
                "exact_repeated_prediction_fraction": verified[
                    "exact_repeated_prediction_fraction"
                ],
                "approximate_unique_prediction_count": verified[
                    "approximate_unique_prediction_count"
                ],
                "approximate_repeated_prediction_fraction": verified[
                    "approximate_repeated_prediction_fraction"
                ],
                "prediction_min": verified["prediction_min"],
                "prediction_max": verified["prediction_max"],
                "prediction_range": verified["prediction_range"],
                "residual_mean": verified["residual_mean"],
                "residual_std": verified["residual_std"],
                "best_epoch": verified["best_epoch"],
            }
            expected_signature = EXPECTED_SPLIT_SIGNATURES[int(verified["seed"])]
            if verified["split_signature"] != expected_signature:
                raise PipelineError(
                    f"seed {verified['seed']} split signature mismatch: "
                    f"{verified['split_signature']} != {expected_signature}"
                )
            seed_rows.append(row)
            prediction_pairs[(label, width, k_value, int(verified["seed"]))] = (
                targets,
                predictions,
            )
    seed_rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            row["hidden_dim"],
            row["k"],
            SEEDS.index(row["seed"]),
        )
    )
    expected = {
        (model, width, k_value, seed)
        for model in MODELS
        for width in WIDTHS
        for k_value in K_VALUES
        for seed in SEEDS
    }
    observed = {
        (row["model"], row["hidden_dim"], row["k"], row["seed"])
        for row in seed_rows
    }
    if len(seed_rows) != 48 or observed != expected:
        raise PipelineError(
            f"factorial membership mismatch: rows={len(seed_rows)}, "
            f"missing={sorted(expected - observed)}"
        )
    if len({row["experiment_path"] for row in seed_rows}) != 48:
        raise PipelineError("experiment mappings are not unique")
    if len({row["prediction_path"] for row in seed_rows}) != 48:
        raise PipelineError("prediction mappings are not unique")
    return seed_rows, prediction_pairs


def aggregate_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in MODELS:
        for width in WIDTHS:
            for k_value in K_VALUES:
                rows = [
                    row
                    for row in seed_rows
                    if row["model"] == model
                    and row["hidden_dim"] == width
                    and row["k"] == k_value
                ]
                record: dict[str, Any] = {
                    "model": model,
                    "hidden_dim": width,
                    "k": k_value,
                    "seed_count": len(rows),
                    "seed_list": ";".join(str(row["seed"]) for row in rows),
                }
                for metric in PRIMARY_METRICS:
                    values = [float(row[metric]) for row in rows]
                    record[f"mean_{metric}"] = mean(values)
                    record[f"std_{metric}"] = sd(values)
                for metric in (
                    "prediction_std_ratio",
                    "exact_repeated_prediction_fraction",
                ):
                    values = [float(row[metric]) for row in rows]
                    record[f"mean_{metric}"] = mean(values)
                    record[f"std_{metric}"] = sd(values)
                pearsons = [
                    float(row["test_pearson"])
                    for row in rows
                    if row["test_pearson"] != ""
                ]
                pearson_mean, pearson_sd = finite_summary(pearsons)
                record["mean_test_pearson"] = (
                    pearson_mean if pearson_mean is not None else ""
                )
                record["std_test_pearson"] = (
                    pearson_sd if pearson_sd is not None else ""
                )
                record["undefined_pearson_count"] = sum(
                    row["pearson_status"] != "defined" for row in rows
                )
                result.append(record)
    if len(result) != 16 or any(row["seed_count"] != 3 for row in result):
        raise PipelineError("aggregate coverage is not 16 cells × 3 seeds")
    return result


def paired_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["model"], row["hidden_dim"], row["k"], row["seed"]): row
        for row in seed_rows
    }
    width_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for k_value in K_VALUES:
            for seed in SEEDS:
                h32 = lookup[(model, 32, k_value, seed)]
                h64 = lookup[(model, 64, k_value, seed)]
                if h32["split_signature"] != h64["split_signature"]:
                    raise PipelineError("width pair has mismatched splits")
                width_rows.append(
                    {
                        "model": model,
                        "k": k_value,
                        "seed": seed,
                        "split_signature": h32["split_signature"],
                        "h32_experiment_name": h32["experiment_name"],
                        "h64_experiment_name": h64["experiment_name"],
                        "mae_h64_minus_h32": float(h64["test_mae"])
                        - float(h32["test_mae"]),
                        "rmse_h64_minus_h32": float(h64["test_rmse"])
                        - float(h32["test_rmse"]),
                        "r2_h64_minus_h32": float(h64["test_r2"])
                        - float(h32["test_r2"]),
                    }
                )
    model_rows: list[dict[str, Any]] = []
    for width in WIDTHS:
        for k_value in K_VALUES:
            for seed in SEEDS:
                evolve = lookup[("EvolveGCN-H", width, k_value, seed)]
                static = lookup[("Static GCN", width, k_value, seed)]
                if evolve["split_signature"] != static["split_signature"]:
                    raise PipelineError("model pair has mismatched splits")
                model_rows.append(
                    {
                        "hidden_dim": width,
                        "k": k_value,
                        "seed": seed,
                        "split_signature": evolve["split_signature"],
                        "evolve_experiment_name": evolve["experiment_name"],
                        "static_experiment_name": static["experiment_name"],
                        "mae_evolve_minus_static": float(evolve["test_mae"])
                        - float(static["test_mae"]),
                        "rmse_evolve_minus_static": float(evolve["test_rmse"])
                        - float(static["test_rmse"]),
                        "r2_evolve_minus_static": float(evolve["test_r2"])
                        - float(static["test_r2"]),
                    }
                )
    if len(width_rows) != 24 or len(model_rows) != 24:
        raise PipelineError("paired row count mismatch")
    return width_rows, model_rows


def interaction_rows(width_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in MODELS:
        baseline = mean(
            float(row["mae_h64_minus_h32"])
            for row in width_rows
            if row["model"] == model and row["k"] == 4
        )
        for k_value in K_VALUES:
            values = [
                float(row["mae_h64_minus_h32"])
                for row in width_rows
                if row["model"] == model and row["k"] == k_value
            ]
            result.append(
                {
                    "model": model,
                    "k": k_value,
                    "seed_count": len(values),
                    "mean_mae_h64_minus_h32": mean(values),
                    "std_mae_h64_minus_h32": sd(values),
                    "negative_count_favors_h64": sum(value < 0 for value in values),
                    "positive_count_favors_h32": sum(value > 0 for value in values),
                    "change_from_k4_mean_width_effect": mean(values) - baseline,
                }
            )
    return result


def representatives(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for model in MODELS:
        for width in WIDTHS:
            for k_value in K_VALUES:
                rows = sorted(
                    (
                        row
                        for row in seed_rows
                        if row["model"] == model
                        and row["hidden_dim"] == width
                        and row["k"] == k_value
                    ),
                    key=lambda row: (float(row["test_mae"]), int(row["seed"])),
                )
                selected.append(dict(rows[1]))
    return selected


def consistency_check(
    root: Path,
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    model_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    old_dir = (
        root
        / "reports/analysis/"
        "controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64"
    )
    old_seed = read_csv(old_dir / "seed_level_results.csv")
    old_aggregate = read_csv(old_dir / "aggregated_results.csv")
    old_pairs = read_csv(old_dir / "paired_model_differences.csv")
    new_seed = {
        (row["model"], int(row["k"]), int(row["seed"])): row
        for row in seed_rows
        if row["hidden_dim"] == 64
    }
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    for old in old_seed:
        key = (old["model"], int(old["k"]), int(old["seed"]))
        new = new_seed.get(key)
        if new is None:
            errors.append(f"missing new h64 seed row: {key}")
            continue
        for field in (
            "experiment_name",
            "experiment_path",
            "split_signature",
        ):
            if str(new[field]) != old[field]:
                errors.append(f"h64 {key} {field} mismatch")
        for field in PRIMARY_METRICS:
            if not math.isclose(
                float(new[field]), float(old[field]), rel_tol=0.0, abs_tol=TOLERANCE
            ):
                errors.append(f"h64 {key} {field} mismatch")
    new_aggregate = {
        (row["model"], int(row["k"])): row
        for row in aggregates
        if row["hidden_dim"] == 64
    }
    for old in old_aggregate:
        key = (old["model"], int(old["k"]))
        new = new_aggregate.get(key)
        if new is None:
            errors.append(f"missing new h64 aggregate: {key}")
            continue
        for metric in PRIMARY_METRICS:
            for prefix in ("mean", "std"):
                field = f"{prefix}_{metric}"
                if not math.isclose(
                    float(new[field]),
                    float(old[field]),
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                ):
                    errors.append(f"h64 aggregate {key} {field} mismatch")
        checks.append(
            {
                "model": key[0],
                "k": key[1],
                "old_mean_test_mae": old["mean_test_mae"],
                "new_mean_test_mae": new["mean_test_mae"],
                "absolute_difference": abs(
                    float(old["mean_test_mae"]) - float(new["mean_test_mae"])
                ),
            }
        )
    new_pairs = {
        (int(row["k"]), int(row["seed"])): row
        for row in model_pairs
        if row["hidden_dim"] == 64
    }
    for old in old_pairs:
        key = (int(old["k"]), int(old["seed"]))
        new = new_pairs.get(key)
        if new is None or not math.isclose(
            float(new["mae_evolve_minus_static"]),
            -float(old["static_minus_evolve_mae"]),
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            errors.append(f"h64 paired model MAE mismatch: {key}")
    parts = [
        "# h64 Slice Consistency Report",
        "",
        f"Verdict: **{'PASS' if not errors else 'FAIL'}**",
        "",
        "The new h64 slice was compared with the established h64 package for exact "
        "experiment membership and split signatures, seed-level MAE/RMSE/R², "
        "aggregate means/SDs, and paired model differences.",
        "",
        f"Tolerance: absolute `{TOLERANCE}` with zero relative tolerance.",
        "",
        *([f"- {error}" for error in errors] or ["No disagreements found."]),
    ]
    write_text(output_dir / "h64_slice_consistency_report.md", "\n".join(parts))
    if errors:
        raise PipelineError("h64 consistency failed: " + "; ".join(errors))
    return checks


def format_number(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, str)):
        return str(value)
    return f"{float(value):.6f}"


def build_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    width_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    main_rows = [
        {
            "model": row["model"],
            "hidden_dim": row["hidden_dim"],
            "k": row["k"],
            "n": row["seed_count"],
            "test_mae_mean_sd": f"{row['mean_test_mae']:.6f} ± {row['std_test_mae']:.6f}",
            "test_rmse_mean_sd": f"{row['mean_test_rmse']:.6f} ± {row['std_test_rmse']:.6f}",
            "test_r2_mean_sd": f"{row['mean_test_r2']:.6f} ± {row['std_test_r2']:.6f}",
            "prediction_sd_ratio_mean_sd": (
                f"{row['mean_prediction_std_ratio']:.6f} ± "
                f"{row['std_prediction_std_ratio']:.6f}"
            ),
            "repeat_fraction_mean_sd": (
                f"{row['mean_exact_repeated_prediction_fraction']:.6f} ± "
                f"{row['std_exact_repeated_prediction_fraction']:.6f}"
            ),
            "undefined_pearson_count": row["undefined_pearson_count"],
        }
        for row in aggregates
    ]
    main_columns = tuple(main_rows[0])
    markdown_and_latex(table_dir / "main_results_table", main_rows, main_columns)
    for width, name in (
        (64, "h64_knn_model_comparison"),
        (32, "h32_knn_model_comparison_descriptive"),
    ):
        rows = [row for row in main_rows if row["hidden_dim"] == width]
        markdown_and_latex(table_dir / name, rows, main_columns)
    width_summaries: list[dict[str, Any]] = []
    for model in MODELS:
        for k_value in K_VALUES:
            values = [
                float(row["mae_h64_minus_h32"])
                for row in width_pairs
                if row["model"] == model and row["k"] == k_value
            ]
            width_summaries.append(
                {
                    "model": model,
                    "k": k_value,
                    "seed_count": len(values),
                    "mean_mae_h64_minus_h32": mean(values),
                    "std_mae_h64_minus_h32": sd(values),
                    "negative_count_favors_h64": sum(value < 0 for value in values),
                    "positive_count_favors_h32": sum(value > 0 for value in values),
                }
            )
    width_columns = tuple(width_summaries[0])
    for model, name in (
        ("EvolveGCN-H", "width_effect_within_evolvegcn"),
        ("Static GCN", "width_effect_within_static_gcn"),
    ):
        rows = [row for row in width_summaries if row["model"] == model]
        markdown_and_latex(table_dir / name, rows, width_columns)
    best_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for width in WIDTHS:
            candidates = [
                row
                for row in aggregates
                if row["model"] == model and row["hidden_dim"] == width
            ]
            best = min(candidates, key=lambda row: row["mean_test_mae"])
            best_rows.append(
                {
                    "model": model,
                    "hidden_dim": width,
                    "descriptive_best_k": best["k"],
                    "mean_test_mae": best["mean_test_mae"],
                    "std_test_mae": best["std_test_mae"],
                    "interpretation": "Descriptively lowest mean MAE under the tested protocol.",
                }
            )
    best_columns = tuple(best_rows[0])
    write_csv(table_dir / "best_k_descriptive_summary.csv", best_rows, best_columns)
    write_text(
        table_dir / "best_k_descriptive_summary.md",
        format_markdown_table(best_rows, best_columns),
    )
    return best_rows


def setup_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise PipelineError("matplotlib and numpy are required") from exc
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    return plt, np


def save_figure(
    plt: Any,
    output_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fig: Any,
) -> None:
    figure_dir = output_dir / "figures"
    plot_dir = output_dir / "plot_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    columns = tuple(rows[0]) if rows else ("status",)
    write_csv(plot_dir / f"{name}.csv", rows, columns)


def create_figures(
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    width_pairs: Sequence[Mapping[str, Any]],
    model_pairs: Sequence[Mapping[str, Any]],
    representatives_rows: Sequence[Mapping[str, Any]],
    prediction_pairs: Mapping[
        tuple[str, int, int, int], tuple[list[float], list[float]]
    ],
    consistency_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt, np = setup_matplotlib()
    colors = {32: "#1f77b4", 64: "#d62728"}
    for metric, name, ylabel in (
        ("test_mae", "test_mae_vs_k_by_hidden_dim", "Test MAE"),
        ("test_rmse", "test_rmse_vs_k_by_hidden_dim", "Test RMSE"),
        ("test_r2", "test_r2_vs_k_by_hidden_dim", "Test R²"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        plot_rows: list[dict[str, Any]] = []
        for axis, model in zip(axes, MODELS):
            for width in WIDTHS:
                cells = [
                    row
                    for row in aggregates
                    if row["model"] == model and row["hidden_dim"] == width
                ]
                axis.errorbar(
                    K_VALUES,
                    [row[f"mean_{metric}"] for row in cells],
                    yerr=[row[f"std_{metric}"] for row in cells],
                    color=colors[width],
                    marker="o",
                    capsize=3,
                    label=f"h{width}",
                )
                for row in seed_rows:
                    if row["model"] == model and row["hidden_dim"] == width:
                        axis.scatter(
                            row["k"],
                            row[metric],
                            color=colors[width],
                            alpha=0.35,
                            s=18,
                        )
                        plot_rows.append(
                            {
                                "model": model,
                                "hidden_dim": width,
                                "k": row["k"],
                                "seed": row["seed"],
                                metric: row[metric],
                                "point_type": "seed",
                            }
                        )
                for row in cells:
                    plot_rows.append(
                        {
                            "model": model,
                            "hidden_dim": width,
                            "k": row["k"],
                            "seed": "",
                            metric: row[f"mean_{metric}"],
                            "sample_standard_deviation": row[f"std_{metric}"],
                            "point_type": "mean",
                        }
                    )
            axis.set_title(model)
            axis.set_xlabel("k")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            axis.legend()
        save_figure(plt, output_dir, name, plot_rows, fig)

    for rows, field, name, ylabel, series_field in (
        (
            width_pairs,
            "model",
            "paired_width_mae_difference_vs_k",
            "MAE(h64) − MAE(h32)",
            "mae_h64_minus_h32",
        ),
        (
            model_pairs,
            "hidden_dim",
            "paired_model_mae_difference_vs_k",
            "MAE(Evolve) − MAE(Static)",
            "mae_evolve_minus_static",
        ),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        plot_rows = []
        series_values = MODELS if field == "model" else WIDTHS
        for index, series in enumerate(series_values):
            selected = [row for row in rows if row[field] == series]
            grouped = {
                k_value: [
                    float(row[series_field])
                    for row in selected
                    if row["k"] == k_value
                ]
                for k_value in K_VALUES
            }
            color = ("#2ca02c", "#9467bd")[index]
            for row in selected:
                axis.scatter(row["k"], row[series_field], color=color, alpha=0.4)
                plot_rows.append({**row, "point_type": "seed"})
            means = [mean(grouped[k_value]) for k_value in K_VALUES]
            stds = [sd(grouped[k_value]) for k_value in K_VALUES]
            axis.errorbar(
                K_VALUES,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                color=color,
                label=str(series) if field == "model" else f"h{series}",
            )
            for k_value, value, deviation in zip(K_VALUES, means, stds):
                plot_rows.append(
                    {
                        field: series,
                        "k": k_value,
                        "seed": "",
                        series_field: value,
                        "sample_standard_deviation": deviation,
                        "point_type": "mean",
                    }
                )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("k")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
        save_figure(plt, output_dir, name, plot_rows, fig)

    for metric, name, ylabel in (
        ("std_test_mae", "seed_variability_vs_k_and_width", "Between-seed SD (MAE)"),
        (
            "mean_prediction_std_ratio",
            "prediction_std_ratio_vs_k_and_width",
            "Mean prediction SD / target SD",
        ),
        (
            "mean_exact_repeated_prediction_fraction",
            "repeated_prediction_fraction_vs_k_and_width",
            "Mean exact repeated-prediction fraction",
        ),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.7))
        plot_rows = []
        for model_index, model in enumerate(MODELS):
            for width in WIDTHS:
                rows = [
                    row
                    for row in aggregates
                    if row["model"] == model and row["hidden_dim"] == width
                ]
                style = "-" if model_index == 0 else "--"
                axis.plot(
                    K_VALUES,
                    [row[metric] for row in rows],
                    marker="o",
                    linestyle=style,
                    color=colors[width],
                    label=f"{model}, h{width}",
                )
                plot_rows.extend(
                    {
                        "model": model,
                        "hidden_dim": width,
                        "k": row["k"],
                        metric: row[metric],
                    }
                    for row in rows
                )
        axis.set_xlabel("k")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2)
        save_figure(plt, output_dir, name, plot_rows, fig)

    for model, name in (
        ("EvolveGCN-H", "mae_heatmap_k_by_hidden_dim_evolvegcn"),
        ("Static GCN", "mae_heatmap_k_by_hidden_dim_static"),
    ):
        matrix = np.array(
            [
                [
                    next(
                        row["mean_test_mae"]
                        for row in aggregates
                        if row["model"] == model
                        and row["hidden_dim"] == width
                        and row["k"] == k_value
                    )
                    for k_value in K_VALUES
                ]
                for width in WIDTHS
            ]
        )
        fig, axis = plt.subplots(figsize=(6.4, 3.2))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(K_VALUES)), K_VALUES)
        axis.set_yticks(range(len(WIDTHS)), [f"h{width}" for width in WIDTHS])
        axis.set_xlabel("k")
        axis.set_ylabel("Hidden dimension")
        axis.set_title(f"{model}: mean test MAE")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, label="Mean test MAE")
        plot_rows = [
            {
                "model": model,
                "hidden_dim": width,
                "k": k_value,
                "mean_test_mae": matrix[i, j],
            }
            for i, width in enumerate(WIDTHS)
            for j, k_value in enumerate(K_VALUES)
        ]
        save_figure(plt, output_dir, name, plot_rows, fig)

    for residual_mode, name in (
        (False, "representative_true_vs_predicted"),
        (True, "representative_residuals_vs_true"),
    ):
        fig, axes = plt.subplots(4, 4, figsize=(13, 11))
        plot_rows = []
        for row_index, (model, width) in enumerate(
            ((model, width) for model in MODELS for width in WIDTHS)
        ):
            for column_index, k_value in enumerate(K_VALUES):
                axis = axes[row_index][column_index]
                representative = next(
                    row
                    for row in representatives_rows
                    if row["model"] == model
                    and row["hidden_dim"] == width
                    and row["k"] == k_value
                )
                targets, predictions = prediction_pairs[
                    (model, width, k_value, representative["seed"])
                ]
                plotted = (
                    [prediction - target for target, prediction in zip(targets, predictions)]
                    if residual_mode
                    else predictions
                )
                axis.scatter(targets, plotted, s=13, alpha=0.65)
                if residual_mode:
                    axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
                else:
                    low = min(targets + predictions)
                    high = max(targets + predictions)
                    axis.plot([low, high], [low, high], "k--", linewidth=0.8)
                axis.set_title(f"{model}, h{width}, k{k_value}, s{representative['seed']}")
                axis.set_xlabel("True")
                axis.set_ylabel("Residual" if residual_mode else "Predicted")
                plot_rows.extend(
                    {
                        "model": model,
                        "hidden_dim": width,
                        "k": k_value,
                        "seed": representative["seed"],
                        "target": target,
                        "prediction": prediction,
                        "residual": prediction - target,
                    }
                    for target, prediction in zip(targets, predictions)
                )
        save_figure(plt, output_dir, name, plot_rows, fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    distribution_rows = []
    values = []
    labels = []
    for row in representatives_rows:
        _, predictions = prediction_pairs[
            (row["model"], row["hidden_dim"], row["k"], row["seed"])
        ]
        values.append(predictions)
        labels.append(
            f"{'E' if row['model'] == 'EvolveGCN-H' else 'S'}"
            f"-h{row['hidden_dim']}-k{row['k']}"
        )
        distribution_rows.extend(
            {
                "model": row["model"],
                "hidden_dim": row["hidden_dim"],
                "k": row["k"],
                "seed": row["seed"],
                "prediction": prediction,
            }
            for prediction in predictions
        )
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Predicted Omega_m")
    axis.set_title("Representative prediction distributions (median-MAE seeds)")
    save_figure(
        plt,
        output_dir,
        "representative_prediction_distributions",
        distribution_rows,
        fig,
    )

    fig, axis = plt.subplots(figsize=(5.5, 5))
    old_values = [float(row["old_mean_test_mae"]) for row in consistency_rows]
    new_values = [float(row["new_mean_test_mae"]) for row in consistency_rows]
    axis.scatter(old_values, new_values)
    low, high = min(old_values + new_values), max(old_values + new_values)
    axis.plot([low, high], [low, high], "k--")
    axis.set_xlabel("Established h64 mean MAE")
    axis.set_ylabel("Factorial h64 mean MAE")
    axis.set_title("h64 consistency")
    save_figure(
        plt,
        output_dir,
        "h64_consistency_with_existing_analysis",
        consistency_rows,
        fig,
    )


def write_summaries(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    width_pairs: Sequence[Mapping[str, Any]],
    model_pairs: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
) -> None:
    best_text = "; ".join(
        f"{row['model']} h{row['hidden_dim']}: k={row['descriptive_best_k']}"
        for row in best_rows
    )
    collapsed = [
        row
        for row in seed_rows
        if float(row["prediction_std_ratio"]) < 0.05
        or float(row["exact_repeated_prediction_fraction"]) > 0
    ]
    width_lines = []
    for model in MODELS:
        values = [
            float(row["mae_h64_minus_h32"])
            for row in width_pairs
            if row["model"] == model
        ]
        width_lines.append(
            f"- {model}: h64 had lower MAE in {sum(value < 0 for value in values)}/"
            f"{len(values)} matched rows; mean h64-minus-h32 MAE was {mean(values):.6f}."
        )
    model_values = [float(row["mae_evolve_minus_static"]) for row in model_pairs]
    parts = [
        "# Scientific Summary",
        "",
        "All 48 factorial rows are artifact-complete and verifier-complete. "
        "All primary metrics are finite and saved-versus-recomputed checks pass.",
        "",
        "## k and hidden-dimension findings",
        "",
        f"Descriptive best k under the tested protocol: {best_text}.",
        "",
        *width_lines,
        "",
        "The h64-minus-h32 differences change with k and sometimes change sign, "
        "which is descriptive evidence of a k × hidden-dimension interaction. "
        "With three seeds, this is not an inferential claim.",
        "",
        "Differences between k-cell mean MAEs are generally small relative to "
        "between-seed variability; changing k from 4 to 12 does not produce a "
        "consistent improvement at either width.",
        "",
        "## Descriptive cross-model comparison",
        "",
        f"EvolveGCN-H had lower MAE in {sum(value < 0 for value in model_values)}/"
        f"{len(model_values)} matched rows and Static GCN in "
        f"{sum(value > 0 for value in model_values)}/{len(model_values)}. "
        "This compares complete protocols and is not a pure causal test of temporal input.",
        "",
        "## Prediction compression and repetition",
        "",
        f"{len(collapsed)} rows met the displayed compression/repetition diagnostic "
        "(prediction-SD ratio < 0.05 or any exact repeat). The Static GCN h32/k4/"
        "seed42 row is exactly constant, so Pearson is undefined due to zero "
        "prediction variance; it remains in every applicable table and plot. "
        "Denser connectivity does not consistently resolve prediction compression.",
        "",
        "Negative R² values and collapsed rows are retained. No p-values are reported.",
    ]
    write_text(output_dir / "scientific_summary.md", "\n".join(parts))
    presentation = [
        "# Presentation-Ready Summary",
        "",
        "## Recommended main items",
        "",
        "1. `protocol_table.md`",
        "2. `tables/main_results_table.md`",
        "3. `figures/test_mae_vs_k_by_hidden_dim.png`",
        "4. `figures/paired_width_mae_difference_vs_k.png`",
        "5. `figures/prediction_std_ratio_vs_k_and_width.png`",
        "6. `figures/repeated_prediction_fraction_vs_k_and_width.png`",
        "7. `figures/representative_true_vs_predicted.png`",
        "",
        "## Central conclusion",
        "",
        "Training and artifact completion are 48/48, and both family verifiers pass. "
        "Under the tested protocol, changing k or increasing width does not yield a "
        "consistent performance improvement relative to between-seed variability. "
        "Prediction compression is widespread and one Static h32/k4/seed42 run is "
        "exactly constant, making Pearson undefined rather than zero. Results use "
        "three seeds, and Static-versus-Evolve differences are descriptive protocol "
        "comparisons rather than causal temporal ablations.",
    ]
    write_text(output_dir / "presentation_ready_summary.md", "\n".join(presentation))
    readme = [
        "# Controlled kNN × Hidden-Dimension Factorial",
        "",
        "This package is regenerated by:",
        "",
        "```bash",
        "env MPLCONFIGDIR=/tmp/codex-matplotlib-cache \\",
        "  envs/camels-gnn/bin/python \\",
        "  reports/analysis/controlled_knn_by_hidden_dim_factorial_500u_top100/"
        "rebuild_analysis_specific_outputs.py \\",
        "  --repo-root /home/ml/thesis-camels \\",
        "  --spec configs/analysis_reports/"
        "controlled_knn_by_hidden_dim_factorial_500u_top100.json",
        "```",
        "",
        "The script reads only JSON/CSV/text artifacts and one byte from each checkpoint "
        "to verify readability; it never deserializes checkpoints or datasets.",
        "",
        f"Pearson SD tolerance: `{PEARSON_STD_TOLERANCE}`. Approximate-repeat "
        f"absolute tolerance: `{APPROXIMATE_REPEAT_TOLERANCE}`.",
    ]
    write_text(output_dir / "README.md", "\n".join(readme))


def update_manifest(
    root: Path,
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    width_pairs: Sequence[Mapping[str, Any]],
    model_pairs: Sequence[Mapping[str, Any]],
) -> None:
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    outputs = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    hashes: dict[str, str] = {}
    for relative in outputs:
        digest = hashlib.sha256()
        with (output_dir / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[relative] = digest.hexdigest()
    manifest.update(
        {
            "factorial_specific_rebuild_timestamp_utc": dt.datetime.now(
                dt.timezone.utc
            ).isoformat(),
            "generated_row_counts": {
                "seed_level_results": len(seed_rows),
                "aggregated_results": len(aggregates),
                "paired_width_differences": len(width_pairs),
                "paired_model_differences": len(model_pairs),
            },
            "undefined_pearson_policy": {
                "pearson_std_tolerance": PEARSON_STD_TOLERANCE,
                "representation": "blank in CSV; explicit pearson_status",
                "zero_is_not_imputed": True,
            },
            "approximate_repeat_tolerance": APPROXIMATE_REPEAT_TOLERANCE,
            "h64_consistency": "PASS",
            "generated_outputs": ["analysis_manifest.json", *outputs],
            "generated_output_hashes_sha256": hashes,
        }
    )
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    validation = validate_analysis(root, args.spec)
    if not validation.valid:
        raise PipelineError("analysis validation failed: " + "; ".join(validation.errors))
    output_dir = resolve_repo_path(root, validation.spec["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, prediction_pairs = load_rows(root, validation)
    aggregates = aggregate_rows(seed_rows)
    width_pairs, model_pairs = paired_rows(seed_rows)
    interactions = interaction_rows(width_pairs)
    representative_rows = representatives(seed_rows)

    seed_columns = tuple(seed_rows[0])
    aggregate_columns = tuple(aggregates[0])
    width_columns = tuple(width_pairs[0])
    model_columns = tuple(model_pairs[0])
    interaction_columns = tuple(interactions[0])
    write_csv(output_dir / "seed_level_results.csv", seed_rows, seed_columns)
    write_csv(output_dir / "aggregated_results.csv", aggregates, aggregate_columns)
    write_csv(output_dir / "paired_width_differences.csv", width_pairs, width_columns)
    write_csv(output_dir / "paired_model_differences.csv", model_pairs, model_columns)
    write_csv(
        output_dir / "factorial_interaction_summary.csv",
        interactions,
        interaction_columns,
    )
    diagnostic_columns = (
        "model",
        "hidden_dim",
        "k",
        "seed",
        "experiment_name",
        "prediction_path",
        "test_count",
        "pearson_status",
        "target_std",
        "prediction_std",
        "prediction_std_ratio",
        "unique_prediction_count",
        "exact_repeated_prediction_fraction",
        "approximate_unique_prediction_count",
        "approximate_repeated_prediction_fraction",
        "prediction_min",
        "prediction_max",
        "prediction_range",
        "residual_mean",
        "residual_std",
    )
    write_csv(
        output_dir / "prediction_diagnostics.csv",
        seed_rows,
        diagnostic_columns,
    )
    consistency_rows = consistency_check(
        root, output_dir, seed_rows, aggregates, model_pairs
    )
    best_rows = build_tables(output_dir, aggregates, width_pairs)
    write_summaries(
        output_dir,
        aggregates,
        width_pairs,
        model_pairs,
        best_rows,
        seed_rows,
    )
    create_figures(
        output_dir,
        seed_rows,
        aggregates,
        width_pairs,
        model_pairs,
        representative_rows,
        prediction_pairs,
        consistency_rows,
    )
    update_manifest(
        root, output_dir, seed_rows, aggregates, width_pairs, model_pairs
    )
    print(f"PASS: rebuilt {output_dir}")
    print("Seed-level rows: 48")
    print("Aggregate rows: 16")
    print("Paired-width rows: 24")
    print("Paired-model rows: 24")
    print("h64 consistency: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
