#!/usr/bin/env python3
"""Rebuild the controlled EvolveGCN-H graph-pooling analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[3]
POOLING_ORDER = ("mean", "mean_max")
SEEDS = (42, 123, 777, 999, 2025)
TOLERANCE = 1e-6
FIGURES = (
    "test_mae_by_graph_pooling",
    "test_rmse_by_graph_pooling",
    "test_r2_by_graph_pooling",
    "paired_mae_difference_by_seed",
    "paired_rmse_difference_by_seed",
    "paired_r2_difference_by_seed",
    "per_seed_mae_trajectories",
    "prediction_std_ratio_by_graph_pooling",
    "repeated_prediction_fraction_by_graph_pooling",
    "representative_true_vs_predicted",
    "representative_residuals_vs_true",
    "representative_prediction_distributions",
    "seed_variability_by_graph_pooling",
    "pooling_effect_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate verified graph-pooling tables and figures."
    )
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(
            "configs/analysis_reports/"
            "controlled_evolvegcn_graph_pooling_ablation_750u_top1000.json"
        ),
    )
    return parser.parse_args()


def configure_imports(root: Path) -> None:
    sys.path.insert(0, str(root / "scripts"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write headerless empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("±", "$\\pm$")
    )


def save_table_set(
    table_dir: Path,
    name: str,
    caption: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    latex: bool = True,
) -> None:
    fields = list(rows[0])
    write_csv(table_dir / f"{name}.csv", rows)
    (table_dir / f"{name}.md").write_text(
        f"# {caption}\n\n{markdown_table(rows, fields)}", encoding="utf-8"
    )
    if latex:
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{" + latex_escape(caption) + "}",
            "\\begin{tabular}{" + "l" * len(fields) + "}",
            "\\hline",
            " & ".join(latex_escape(field) for field in fields) + " \\\\",
            "\\hline",
        ]
        lines.extend(
            " & ".join(latex_escape(row.get(field, "")) for field in fields)
            + " \\\\"
            for row in rows
        )
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
        (table_dir / f"{name}.tex").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def read_inputs(root: Path, validation: Any) -> tuple[list[dict[str, Any]], dict]:
    from experiment_pipeline.common import read_prediction_rows, recompute_metrics

    if len(validation.family_rows) != 1:
        raise RuntimeError("expected exactly one verified family")
    label, verified_rows = validation.family_rows[0]
    family_path = root / validation.spec["families"][0]["family_spec_path"]
    family = json.loads(family_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    for verified in verified_rows:
        pooling = str(verified["grouping_value"])
        seed = int(verified["seed"])
        experiment_dir = root / str(verified["experiment_path"])
        prediction_path = experiment_dir / family["prediction_file"]
        ids, targets, estimates = read_prediction_rows(
            prediction_path,
            family["target_column_aliases"],
            family["prediction_column_aliases"],
            family["id_column_aliases"],
        )
        metrics = recompute_metrics(targets, estimates)
        residuals = [
            prediction - target
            for target, prediction in zip(targets, estimates)
        ]
        config = json.loads(
            (experiment_dir / "config.json").read_text(encoding="utf-8")
        )
        saved_metrics = json.loads(
            (experiment_dir / "metrics.json").read_text(encoding="utf-8")
        )
        saved = saved_metrics["test"]
        deltas = {
            "saved_mae_absolute_delta": abs(
                float(saved["mae"]) - float(metrics["test_mae"])
            ),
            "saved_rmse_absolute_delta": abs(
                float(saved["rmse"]) - float(metrics["test_rmse"])
            ),
            "saved_mse_absolute_delta": abs(
                float(saved["mse"]) - float(metrics["test_mse"])
            ),
        }
        if max(deltas.values()) > TOLERANCE:
            raise RuntimeError(
                f"saved metric mismatch: {verified['experiment_name']}"
            )
        train_ids = list(config["train_ids"])
        val_ids = list(config["val_ids"])
        test_ids = list(config["test_ids"])
        all_ids = train_ids + val_ids + test_ids
        if (
            (len(train_ids), len(val_ids), len(test_ids)) != (450, 99, 201)
            or set(train_ids) & set(val_ids)
            or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids)
            or set(all_ids) != {f"LH_{index}" for index in range(750)}
            or len(set(all_ids)) != 750
            or ids != test_ids
            or len(ids) != len(set(ids)) != 201
        ):
            raise RuntimeError(
                f"split/prediction ID verification failed: {verified['experiment_name']}"
            )
        record = {
            "model": label,
            "graph_pooling": pooling,
            "seed": seed,
            "experiment_name": verified["experiment_name"],
            "experiment_path": verified["experiment_path"],
            "prediction_path": prediction_path.relative_to(root).as_posix(),
            "split_signature": verified["split_signature"],
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": int(metrics["test_count"]),
            "test_mae": metrics["test_mae"],
            "test_rmse": metrics["test_rmse"],
            "test_mse": metrics["test_mse"],
            "test_r2": metrics["test_r2"],
            "test_pearson": metrics["test_pearson"],
            "pearson_status": metrics["pearson_status"],
            "target_mean": metrics["target_mean"],
            "prediction_mean": metrics["prediction_mean"],
            "target_sd": metrics["target_std"],
            "prediction_sd": metrics["prediction_std"],
            "prediction_sd_ratio": metrics["prediction_std_ratio"],
            "unique_prediction_count": metrics["unique_prediction_count"],
            "exact_repeated_prediction_fraction": metrics[
                "exact_repeated_prediction_fraction"
            ],
            "approximate_repeated_prediction_fraction": metrics[
                "approximate_repeated_prediction_fraction"
            ],
            "approximate_repeat_tolerance": metrics[
                "approximate_repeat_tolerance"
            ],
            "prediction_min": metrics["prediction_min"],
            "prediction_max": metrics["prediction_max"],
            "prediction_range": metrics["prediction_range"],
            "residual_mean": metrics["residual_mean"],
            "residual_sd": metrics["residual_std"],
            "maximum_absolute_residual": max(abs(value) for value in residuals),
            "best_epoch": saved_metrics.get("best_epoch", ""),
            **deltas,
            "prediction_sha256": sha256(prediction_path),
        }
        records.append(record)
        predictions[(pooling, seed)] = {
            "ids": ids,
            "targets": targets,
            "predictions": estimates,
        }
    pool_index = {value: index for index, value in enumerate(POOLING_ORDER)}
    seed_index = {value: index for index, value in enumerate(SEEDS)}
    records.sort(
        key=lambda row: (
            pool_index[row["graph_pooling"]],
            seed_index[row["seed"]],
        )
    )
    if len(records) != 10:
        raise RuntimeError(f"seed rows={len(records)}, expected=10")
    if len({row["experiment_path"] for row in records}) != 10:
        raise RuntimeError("canonical experiment mappings are not unique")
    if len({row["prediction_path"] for row in records}) != 10:
        raise RuntimeError("canonical prediction mappings are not unique")
    if len({row["prediction_sha256"] for row in records}) != 10:
        raise RuntimeError("canonical prediction files contain a duplicate")
    return records, predictions


def aggregate_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for pooling in POOLING_ORDER:
        rows = [row for row in seed_rows if row["graph_pooling"] == pooling]
        maes = [float(row["test_mae"]) for row in rows]
        rmses = [float(row["test_rmse"]) for row in rows]
        r2s = [float(row["test_r2"]) for row in rows]
        ratios = [float(row["prediction_sd_ratio"]) for row in rows]
        repeats = [
            float(row["exact_repeated_prediction_fraction"]) for row in rows
        ]
        output.append(
            {
                "graph_pooling": pooling,
                "seed_count": len(rows),
                "seeds": "|".join(str(row["seed"]) for row in rows),
                "mean_test_mae": statistics.mean(maes),
                "sd_test_mae": sample_sd(maes),
                "median_test_mae": statistics.median(maes),
                "mean_test_rmse": statistics.mean(rmses),
                "sd_test_rmse": sample_sd(rmses),
                "mean_test_r2": statistics.mean(r2s),
                "sd_test_r2": sample_sd(r2s),
                "mean_prediction_sd_ratio": statistics.mean(ratios),
                "sd_prediction_sd_ratio": sample_sd(ratios),
                "mean_repeated_prediction_fraction": statistics.mean(repeats),
                "undefined_pearson_count": sum(
                    row["pearson_status"] != "defined" for row in rows
                ),
            }
        )
    return output


def paired_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["graph_pooling"]), int(row["seed"])): row for row in seed_rows
    }
    output = []
    for seed in SEEDS:
        mean = lookup[("mean", seed)]
        mean_max = lookup[("mean_max", seed)]
        if mean["split_signature"] != mean_max["split_signature"]:
            raise RuntimeError(f"paired split mismatch for seed {seed}")
        output.append(
            {
                "seed": seed,
                "split_signature": mean["split_signature"],
                "mean_experiment_name": mean["experiment_name"],
                "mean_max_experiment_name": mean_max["experiment_name"],
                "mae_mean_max_minus_mean": float(mean_max["test_mae"])
                - float(mean["test_mae"]),
                "rmse_mean_max_minus_mean": float(mean_max["test_rmse"])
                - float(mean["test_rmse"]),
                "r2_mean_max_minus_mean": float(mean_max["test_r2"])
                - float(mean["test_r2"]),
                "prediction_sd_ratio_mean_max_minus_mean": float(
                    mean_max["prediction_sd_ratio"]
                )
                - float(mean["prediction_sd_ratio"]),
                "repeat_fraction_mean_max_minus_mean": float(
                    mean_max["exact_repeated_prediction_fraction"]
                )
                - float(mean["exact_repeated_prediction_fraction"]),
            }
        )
    metric_fields = (
        "mae_mean_max_minus_mean",
        "rmse_mean_max_minus_mean",
        "r2_mean_max_minus_mean",
        "prediction_sd_ratio_mean_max_minus_mean",
        "repeat_fraction_mean_max_minus_mean",
    )
    for field in metric_fields:
        values = [float(row[field]) for row in output]
        prefix = field.removesuffix("_mean_max_minus_mean")
        summary = {
            f"mean_paired_{prefix}_difference": statistics.mean(values),
            f"sd_paired_{prefix}_difference": sample_sd(values),
            f"median_paired_{prefix}_difference": statistics.median(values),
            f"minimum_paired_{prefix}_difference": min(values),
            f"maximum_paired_{prefix}_difference": max(values),
            f"{prefix}_negative_count": sum(value < 0 for value in values),
            f"{prefix}_zero_count": sum(value == 0 for value in values),
            f"{prefix}_positive_count": sum(value > 0 for value in values),
        }
        for row in output:
            row.update(summary)
    for row in output:
        row["mae_seeds_favouring_mean_max"] = row["mae_negative_count"]
        row["mae_seeds_favouring_mean"] = row["mae_positive_count"]
    return output


def representative_rows(
    seed_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for pooling in POOLING_ORDER:
        rows = sorted(
            (row for row in seed_rows if row["graph_pooling"] == pooling),
            key=lambda row: (float(row["test_mae"]), int(row["seed"])),
        )
        selected = dict(rows[len(rows) // 2])
        output.append(
            {
                "graph_pooling": pooling,
                "selection_policy": "median_test_mae",
                "seed": selected["seed"],
                "experiment_name": selected["experiment_name"],
                "experiment_path": selected["experiment_path"],
                "prediction_path": selected["prediction_path"],
                "test_mae": selected["test_mae"],
                "test_rmse": selected["test_rmse"],
                "test_r2": selected["test_r2"],
                "prediction_sd_ratio": selected["prediction_sd_ratio"],
            }
        )
    return output


def rounded_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "graph pooling": row["graph_pooling"],
            "seeds": row["seed_count"],
            "mean test MAE": f"{float(row['mean_test_mae']):.6f}",
            "SD test MAE": f"{float(row['sd_test_mae']):.6f}",
            "median test MAE": f"{float(row['median_test_mae']):.6f}",
            "mean test RMSE": f"{float(row['mean_test_rmse']):.6f}",
            "SD test RMSE": f"{float(row['sd_test_rmse']):.6f}",
            "mean test R²": f"{float(row['mean_test_r2']):.6f}",
            "SD test R²": f"{float(row['sd_test_r2']):.6f}",
            "mean prediction-SD ratio": f"{float(row['mean_prediction_sd_ratio']):.6f}",
            "SD prediction-SD ratio": f"{float(row['sd_prediction_sd_ratio']):.6f}",
            "mean repeated fraction": f"{float(row['mean_repeated_prediction_fraction']):.6f}",
            "undefined Pearson": row["undefined_pearson_count"],
        }
        for row in rows
    ]


def write_main_table(
    table_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    caption = (
        "EvolveGCN-H U750 Top1000 graph-pooling results: no normalization, "
        "periodic kNN k=8, hidden dimension 32, two layers, temporal mean "
        "pooling, linear head, and five matched seeds."
    )
    fields = list(rows[0])
    write_csv(table_dir / "main_results_table.csv", rows)
    markdown_rows = []
    for row in rows:
        item = dict(row)
        if row["graph pooling"] == "mean":
            for field in (
                "mean test MAE",
                "mean test RMSE",
                "mean test R²",
            ):
                item[field] = f"**{item[field]}**"
        markdown_rows.append(item)
    (table_dir / "main_results_table.md").write_text(
        f"# Main graph-pooling results\n\n{caption}\n\n"
        + markdown_table(markdown_rows, fields),
        encoding="utf-8",
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + latex_escape(caption) + "}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(fields) + "}",
        "\\hline",
        " & ".join(latex_escape(field) for field in fields) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = latex_escape(row[field])
            if row["graph pooling"] == "mean" and field in {
                "mean test MAE",
                "mean test RMSE",
                "mean test R²",
            }:
                value = "\\textbf{" + value + "}"
            values.append(value)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(
        ["\\hline", "\\end{tabular}}", "\\end{table}", ""]
    )
    (table_dir / "main_results_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(
    plt: Any,
    figure: Any,
    output_dir: Path,
    name: str,
    plot_rows: Sequence[Mapping[str, Any]],
) -> None:
    figure_dir = output_dir / "figures"
    plot_dir = output_dir / "plot_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    write_csv(plot_dir / f"{name}.csv", plot_rows)
    plt.close(figure)


def create_figures(
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    plt = configure_matplotlib()
    colors = {"mean": "#386cb0", "mean_max": "#f0027f"}
    x = list(range(len(POOLING_ORDER)))

    for name, metric, mean_field, sd_field, ylabel in (
        ("test_mae_by_graph_pooling", "test_mae", "mean_test_mae", "sd_test_mae", "Test MAE"),
        ("test_rmse_by_graph_pooling", "test_rmse", "mean_test_rmse", "sd_test_rmse", "Test RMSE"),
        ("test_r2_by_graph_pooling", "test_r2", "mean_test_r2", "sd_test_r2", "Test R²"),
        ("prediction_std_ratio_by_graph_pooling", "prediction_sd_ratio", "mean_prediction_sd_ratio", "sd_prediction_sd_ratio", "Prediction SD / target SD"),
        ("repeated_prediction_fraction_by_graph_pooling", "exact_repeated_prediction_fraction", "mean_repeated_prediction_fraction", None, "Exact repeated-prediction fraction"),
    ):
        figure, axis = plt.subplots(figsize=(6.3, 4.4))
        means = [float(row[mean_field]) for row in aggregates]
        errors = [
            float(row[sd_field]) if sd_field else 0.0 for row in aggregates
        ]
        axis.errorbar(x, means, yerr=errors, fmt="o", color="black", capsize=5)
        plot_rows = []
        for pool_index, pooling in enumerate(POOLING_ORDER):
            rows = [row for row in seed_rows if row["graph_pooling"] == pooling]
            for point_index, row in enumerate(rows):
                jitter = (point_index - 2) * 0.025
                value = float(row[metric])
                axis.scatter(
                    pool_index + jitter,
                    value,
                    color=colors[pooling],
                    alpha=0.85,
                    zorder=3,
                )
                plot_rows.append(
                    {
                        "graph_pooling": pooling,
                        "seed": row["seed"],
                        metric: value,
                        "aggregate_mean": means[pool_index],
                        "aggregate_sd": errors[pool_index],
                    }
                )
        axis.set_xticks(x, POOLING_ORDER)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        save_figure(plt, figure, output_dir, name, plot_rows)

    for name, field, ylabel in (
        ("paired_mae_difference_by_seed", "mae_mean_max_minus_mean", "MAE(mean_max) − MAE(mean)"),
        ("paired_rmse_difference_by_seed", "rmse_mean_max_minus_mean", "RMSE(mean_max) − RMSE(mean)"),
        ("paired_r2_difference_by_seed", "r2_mean_max_minus_mean", "R²(mean_max) − R²(mean)"),
    ):
        figure, axis = plt.subplots(figsize=(7.0, 4.3))
        values = [float(row[field]) for row in pairs]
        axis.bar(
            [str(row["seed"]) for row in pairs],
            values,
            color=["#d95f02" if value > 0 else "#1b9e77" for value in values],
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xlabel("Seed")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        save_figure(plt, figure, output_dir, name, pairs)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    trajectory_rows = []
    for seed in SEEDS:
        values = [
            float(
                next(
                    row[ "test_mae"]
                    for row in seed_rows
                    if row["seed"] == seed and row["graph_pooling"] == pooling
                )
            )
            for pooling in POOLING_ORDER
        ]
        axis.plot(x, values, marker="o", alpha=0.75, label=str(seed))
        for pooling, value in zip(POOLING_ORDER, values):
            trajectory_rows.append(
                {"seed": seed, "graph_pooling": pooling, "test_mae": value}
            )
    axis.set_xticks(x, POOLING_ORDER)
    axis.set_ylabel("Test MAE")
    axis.legend(title="Seed", ncol=3)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(
        plt, figure, output_dir, "per_seed_mae_trajectories", trajectory_rows
    )

    figure, axis = plt.subplots(figsize=(6.3, 4.3))
    variability_rows = []
    for index, row in enumerate(aggregates):
        axis.errorbar(
            index,
            float(row["mean_test_mae"]),
            yerr=float(row["sd_test_mae"]),
            fmt="o",
            capsize=7,
            color=colors[row["graph_pooling"]],
        )
        variability_rows.append(
            {
                "graph_pooling": row["graph_pooling"],
                "mean_test_mae": row["mean_test_mae"],
                "sd_test_mae": row["sd_test_mae"],
            }
        )
    axis.set_xticks(x, POOLING_ORDER)
    axis.set_ylabel("Mean test MAE ± seed SD")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(
        plt,
        figure,
        output_dir,
        "seed_variability_by_graph_pooling",
        variability_rows,
    )

    representative_lookup = {
        row["graph_pooling"]: int(row["seed"]) for row in representatives
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    true_pred_rows = []
    for axis, pooling in zip(axes, POOLING_ORDER):
        seed = representative_lookup[pooling]
        data = predictions[(pooling, seed)]
        source = next(
            row
            for row in seed_rows
            if row["graph_pooling"] == pooling and row["seed"] == seed
        )
        axis.scatter(data["targets"], data["predictions"], s=20, alpha=0.7, color=colors[pooling])
        low = min(data["targets"] + data["predictions"])
        high = max(data["targets"] + data["predictions"])
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        axis.set_title(f"{pooling}, median-MAE seed {seed}")
        axis.set_xlabel("True Omega_m")
        axis.text(
            0.03,
            0.97,
            f"MAE={float(source['test_mae']):.3f}\n"
            f"RMSE={float(source['test_rmse']):.3f}\n"
            f"R²={float(source['test_r2']):.3f}\n"
            f"SD ratio={float(source['prediction_sd_ratio']):.3f}",
            transform=axis.transAxes,
            va="top",
        )
        for identifier, target, prediction in zip(
            data["ids"], data["targets"], data["predictions"]
        ):
            true_pred_rows.append(
                {
                    "graph_pooling": pooling,
                    "seed": seed,
                    "universe_id": identifier,
                    "target": target,
                    "prediction": prediction,
                }
            )
    axes[0].set_ylabel("Predicted Omega_m")
    figure.tight_layout()
    save_figure(
        plt,
        figure,
        output_dir,
        "representative_true_vs_predicted",
        true_pred_rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    residual_rows = []
    for axis, pooling in zip(axes, POOLING_ORDER):
        seed = representative_lookup[pooling]
        data = predictions[(pooling, seed)]
        residuals = [
            prediction - target
            for target, prediction in zip(data["targets"], data["predictions"])
        ]
        axis.scatter(data["targets"], residuals, s=20, alpha=0.7, color=colors[pooling])
        axis.axhline(0, linestyle="--", color="black", linewidth=1)
        axis.set_title(f"{pooling}, median-MAE seed {seed}")
        axis.set_xlabel("True Omega_m")
        for identifier, target, residual in zip(data["ids"], data["targets"], residuals):
            residual_rows.append(
                {
                    "graph_pooling": pooling,
                    "seed": seed,
                    "universe_id": identifier,
                    "target": target,
                    "residual": residual,
                }
            )
    axes[0].set_ylabel("Prediction − truth")
    figure.tight_layout()
    save_figure(
        plt,
        figure,
        output_dir,
        "representative_residuals_vs_true",
        residual_rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    distribution_rows = []
    for axis, pooling in zip(axes, POOLING_ORDER):
        seed = representative_lookup[pooling]
        data = predictions[(pooling, seed)]
        axis.hist(data["targets"], bins=18, alpha=0.45, label="truth", color="#4daf4a")
        axis.hist(data["predictions"], bins=18, alpha=0.55, label="prediction", color=colors[pooling])
        axis.set_title(f"{pooling}, median-MAE seed {seed}")
        axis.set_xlabel("Omega_m")
        axis.legend()
        for kind, values in (
            ("target", data["targets"]),
            ("prediction", data["predictions"]),
        ):
            for value in values:
                distribution_rows.append(
                    {
                        "graph_pooling": pooling,
                        "seed": seed,
                        "distribution": kind,
                        "value": value,
                    }
                )
    axes[0].set_ylabel("Count")
    figure.tight_layout()
    save_figure(
        plt,
        figure,
        output_dir,
        "representative_prediction_distributions",
        distribution_rows,
    )

    effect_fields = (
        ("MAE", "mae_mean_max_minus_mean"),
        ("RMSE", "rmse_mean_max_minus_mean"),
        ("R²", "r2_mean_max_minus_mean"),
        ("SD ratio", "prediction_sd_ratio_mean_max_minus_mean"),
    )
    effect_rows = [
        {
            "metric": label,
            "mean_difference": statistics.mean(float(row[field]) for row in pairs),
            "sd_difference": sample_sd(float(row[field]) for row in pairs),
            "convention": "mean_max_minus_mean",
        }
        for label, field in effect_fields
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for axis, effect in zip(axes.flat, effect_rows):
        value = float(effect["mean_difference"])
        axis.bar([effect["metric"]], [value], color="#d95f02" if value > 0 else "#1b9e77")
        axis.errorbar(
            [0],
            [value],
            yerr=float(effect["sd_difference"]),
            fmt="none",
            color="black",
            capsize=5,
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title("mean_max − mean")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(
        plt, figure, output_dir, "pooling_effect_summary", effect_rows
    )


def implementation_text() -> str:
    return """# Graph-pooling implementation

EvolveGCN-H receives node embeddings with shape `[B, T, N, H]` and a
`[B, T, N, 1]` real-node mask.

- `mean` computes the masked node sum divided by the masked node count and
  returns `[B, T, H]`.
- `mean_max` concatenates that mean with a masked maximum and returns
  `[B, T, 2H]`.

With `H=32`, the graph representation is 32-dimensional for mean and
64-dimensional for mean_max. Temporal pooling is mean in both methods and
preserves that final feature dimension. The fixed linear head therefore gains
exactly 32 input weights for mean_max. This is the necessary mathematical
consequence of concatenation, not a different head design.

Both reductions are permutation invariant and operate independently for every
graph and snapshot. Masks exclude padded nodes. Dataset validation disallows
zero-real-node snapshots; if that guarantee were bypassed, the masked maximum
would expose a latent all-masked edge case.

No canonical attention, sum, max-only, TopK, SAGPool, or Set2Set experiment was
trained. Attention pooling is unimplemented and untested.
"""


def scientific_text(
    aggregates: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> str:
    aggregate = {row["graph_pooling"]: row for row in aggregates}
    first = pairs[0]
    paired_mean = float(first["mean_paired_mae_difference"])
    paired_sd = float(first["sd_paired_mae_difference"])
    return f"""# Scientific summary

## Objective

This analysis tests whether concatenating maximum node embeddings with the
mean-pooled representation improves EvolveGCN-H Omega_m regression.

## Experimental design

Ten completed runs form five exact seed-matched pairs. All use the same U750
Top1000 unnormalized temporal dataset, periodic kNN with k=8, hidden dimension
32, two Evolve layers, batch size 4, temporal mean pooling, and a linear head.
Only graph pooling and its required 32-versus-64-dimensional linear-head input
differ.

## Verification

Every config, metric file, training log, prediction file, and checkpoint path
was present. Checkpoints and graph datasets were not loaded. All prediction
files contain 201 finite, uniquely identified rows in exact declared test
order. Ordered splits match within every seed. Saved MAE, RMSE, and MSE agree
with independent recomputation within 1e-6.

## Quantitative results

Mean pooling achieved MAE {float(aggregate['mean']['mean_test_mae']):.6f} ±
{float(aggregate['mean']['sd_test_mae']):.6f}, RMSE
{float(aggregate['mean']['mean_test_rmse']):.6f} ±
{float(aggregate['mean']['sd_test_rmse']):.6f}, and mean R²
{float(aggregate['mean']['mean_test_r2']):.6f}. Mean_max achieved MAE
{float(aggregate['mean_max']['mean_test_mae']):.6f} ±
{float(aggregate['mean_max']['sd_test_mae']):.6f}, RMSE
{float(aggregate['mean_max']['mean_test_rmse']):.6f} ±
{float(aggregate['mean_max']['sd_test_rmse']):.6f}, and mean R²
{float(aggregate['mean_max']['mean_test_r2']):.6f}.

The mean paired MAE difference, defined as mean_max minus mean, was
{paired_mean:+.6f} ± {paired_sd:.6f}. All five MAE and RMSE differences were
positive, and all five R² differences were negative. The paired MAE effect
exceeded the ordinary seed SD within either pooling cell.

## Prediction-dispersion diagnostics

Mean prediction-SD ratios were
{float(aggregate['mean']['mean_prediction_sd_ratio']):.6f} for mean and
{float(aggregate['mean_max']['mean_prediction_sd_ratio']):.6f} for mean_max.
The average values were nearly unchanged, while seed-level changes were
inconsistent. Neither method produced exact or approximate repeated
predictions. The error degradation is therefore not primarily explained by
stronger prediction collapse.

## Interpretation

Directly appending maximum node embeddings did not recover a more informative
readout under this protocol. The maxima may add noisy or unstable extreme
responses that the linear head does not use effectively.

## Limitations

This is a controlled comparison of only mean and concatenated mean-and-maximum
pooling for EvolveGCN-H at U750 Top1000. It does not establish that mean is
universally optimal, that maximum-related information is irrelevant, or that
learned attention would fail. Attention pooling remains unimplemented and
untested.

## Conclusion

Across five matched seeds, mean_max increased MAE and RMSE and reduced R² in
every pair without materially changing aggregate prediction dispersion.
Simple maximum concatenation therefore does not improve the tested graph
readout; adaptive pooling would require a separate controlled experiment.
"""


def presentation_text() -> str:
    return """# Presentation-ready summary

## Recommended material

1. Protocol table.
2. Two-row main results table.
3. Test MAE by graph pooling.
4. Paired MAE difference by seed.
5. Prediction-SD ratio comparison.
6. Representative true-versus-predicted figure.
7. Representative residual figure.

## Supervisor-meeting conclusion

In a five-seed controlled EvolveGCN-H comparison, mean_max graph pooling
worsened MAE, RMSE, and R² in every matched seed. Prediction dispersion
remained similar and neither method repeated predictions, so the degradation
was not mainly an additional-collapse effect. Simple maximum concatenation
does not improve this U750 Top1000 readout; adaptive pooling remains separate
future work.
"""


def protocol_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "graph_pooling": pooling,
            "seeds": "42|123|777|999|2025",
            "universes": 750,
            "top_n": 1000,
            "snapshots": 5,
            "normalization": "none",
            "periodic_knn": True,
            "k": 8,
            "box_size": 25,
            "hidden_dim": 32,
            "num_layers": 2,
            "batch_size": 4,
            "epochs": 300,
            "patience": 40,
            "temporal_pooling": "mean",
            "head_type": "linear",
            "graph_embedding_dim": 32 if pooling == "mean" else 64,
            "train_val_test": "450|99|201",
        }
        for pooling in POOLING_ORDER
    ]


def update_manifest(
    root: Path,
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> None:
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = manifest.setdefault("source_file_hashes_sha256", {})
    for source in (
        output_dir / "rebuild_graph_pooling_analysis.py",
        root / "scripts/validate_analysis_report.py",
        root
        / "reports/experiment_registry/graph_pooling_ablation_final_verification.md",
    ):
        if source.is_file():
            source_hashes[source.relative_to(root).as_posix()] = sha256(source)
    manifest["generated_row_counts"] = {
        "seed_level_results": len(seed_rows),
        "aggregated_results": len(aggregates),
        "paired_pooling_differences": len(pairs),
        "prediction_diagnostics": len(seed_rows),
        "representative_runs": len(representatives),
    }
    manifest["graph_pooling_specific_validation"] = {
        "unique_experiment_mappings": len(
            {row["experiment_path"] for row in seed_rows}
        ),
        "unique_prediction_mappings": len(
            {row["prediction_path"] for row in seed_rows}
        ),
        "unique_prediction_hashes": len(
            {row["prediction_sha256"] for row in seed_rows}
        ),
        "pooling_order": list(POOLING_ORDER),
        "representative_policy": "median_test_mae",
        "representative_runs": [
            {
                "model": "EvolveGCN-H",
                "graph_pooling": row["graph_pooling"],
                "seed": row["seed"],
                "experiment_name": row["experiment_name"],
            }
            for row in representatives
        ],
        "all_primary_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in seed_rows
            for field in ("test_mae", "test_rmse", "test_mse", "test_r2")
        ),
        "all_saved_metrics_match_1e-6": all(
            max(
                float(row["saved_mae_absolute_delta"]),
                float(row["saved_rmse_absolute_delta"]),
                float(row["saved_mse_absolute_delta"]),
            )
            <= TOLERANCE
            for row in seed_rows
        ),
        "undefined_pearson_count": sum(
            row["pearson_status"] != "defined" for row in seed_rows
        ),
    }
    generated = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    manifest["generated_outputs"] = generated
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    configure_imports(root)
    from analysis_reporting.common import validate_analysis

    validation = validate_analysis(root, args.spec)
    if not validation.valid:
        for error in validation.errors:
            print(error, file=sys.stderr)
        return 1
    output_dir = root / validation.spec["output_directory"]
    seed_rows, predictions = read_inputs(root, validation)
    aggregates = aggregate_rows(seed_rows)
    pairs = paired_rows(seed_rows)
    representatives = representative_rows(seed_rows)

    write_csv(output_dir / "seed_level_results.csv", seed_rows)
    write_csv(output_dir / "aggregated_results.csv", aggregates)
    write_csv(output_dir / "paired_pooling_differences.csv", pairs)
    write_csv(output_dir / "prediction_diagnostics.csv", seed_rows)
    write_csv(output_dir / "representative_runs.csv", representatives)
    protocols = protocol_rows(seed_rows)
    write_csv(output_dir / "protocol_table.csv", protocols)
    (output_dir / "protocol_table.md").write_text(
        "# Controlled protocol\n\n"
        + markdown_table(protocols, list(protocols[0])),
        encoding="utf-8",
    )
    (output_dir / "pooling_implementation.md").write_text(
        implementation_text(), encoding="utf-8"
    )
    (output_dir / "scientific_summary.md").write_text(
        scientific_text(aggregates, pairs), encoding="utf-8"
    )
    (output_dir / "presentation_ready_summary.md").write_text(
        presentation_text(), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Controlled EvolveGCN-H graph-pooling ablation\n\n"
        "This deterministic package verifies and compares ten existing U750 "
        "Top1000 mean and mean_max runs. It reads JSON and CSV artifacts only; "
        "it never trains, loads checkpoints, or deserializes graph datasets.\n\n"
        "Run the generic builder first, then this focused rebuild script, and "
        "finally `scripts/validate_analysis_report.py`.\n",
        encoding="utf-8",
    )

    table_dir = output_dir / "tables"
    aggregate_table = rounded_aggregate(aggregates)
    write_main_table(table_dir, aggregate_table)
    save_table_set(
        table_dir,
        "seed_level_pooling_results",
        "Seed-level graph-pooling results",
        seed_rows,
    )
    save_table_set(
        table_dir,
        "aggregate_pooling_results",
        "Aggregate graph-pooling results",
        aggregate_table,
    )
    save_table_set(
        table_dir,
        "paired_pooling_differences",
        "Paired differences (mean_max minus mean)",
        pairs,
    )
    diagnostic_fields = (
        "graph_pooling",
        "seed",
        "experiment_name",
        "test_mae",
        "test_r2",
        "pearson_status",
        "target_sd",
        "prediction_sd",
        "prediction_sd_ratio",
        "unique_prediction_count",
        "exact_repeated_prediction_fraction",
        "approximate_repeated_prediction_fraction",
        "prediction_range",
        "maximum_absolute_residual",
    )
    diagnostic_table = [
        {field: row[field] for field in diagnostic_fields} for row in seed_rows
    ]
    save_table_set(
        table_dir,
        "prediction_collapse_diagnostics",
        "Prediction-collapse diagnostics",
        diagnostic_table,
    )
    descriptive = [
        {
            "descriptively_better_pooling": "mean",
            "mean_wins_by_mae": pairs[0]["mae_positive_count"],
            "mean_max_wins_by_mae": pairs[0]["mae_negative_count"],
            "mean_paired_mae_difference": f"{float(pairs[0]['mean_paired_mae_difference']):+.6f}",
            "sd_paired_mae_difference": f"{float(pairs[0]['sd_paired_mae_difference']):.6f}",
            "repeated_predictions_either_method": False,
            "interpretation": "mean_max worsened error without materially changing aggregate dispersion",
        }
    ]
    save_table_set(
        table_dir,
        "descriptive_pooling_summary",
        "Descriptive graph-pooling summary",
        descriptive,
        latex=False,
    )
    create_figures(
        output_dir,
        seed_rows,
        aggregates,
        pairs,
        representatives,
        predictions,
    )
    update_manifest(
        root,
        output_dir,
        seed_rows,
        aggregates,
        pairs,
        representatives,
    )
    print("Graph-pooling focused rebuild: PASS")
    print(f"Seed-level rows: {len(seed_rows)}")
    print(f"Aggregate rows: {len(aggregates)}")
    print(f"Paired-pooling rows: {len(pairs)}")
    print(f"Prediction-diagnostic rows: {len(seed_rows)}")
    print(f"Representative rows: {len(representatives)}")
    print(f"Figures: {len(FIGURES)} × PNG/PDF/CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
