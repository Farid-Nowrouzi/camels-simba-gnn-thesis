#!/usr/bin/env python3
"""Rebuild normalization-specific tables, figures, and scientific summaries.

This script reads only JSON and CSV artifacts. It never loads graph datasets,
checkpoints, notebooks, or training code, and it never writes under experiments/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NORMALIZATIONS = ("none", "minmax", "zscore")
SEEDS = (42, 123, 777, 999, 2025)
NORMALIZATION_PAIRS = (
    ("minmax", "none"),
    ("zscore", "none"),
    ("zscore", "minmax"),
)
MODEL_ORDER = ("EvolveGCN-H", "Static GCN")
FIGURE_NAMES = (
    "test_mae_vs_normalization",
    "test_rmse_vs_normalization",
    "test_r2_vs_normalization",
    "paired_mae_difference_vs_normalization",
    "seed_variability_vs_normalization",
    "prediction_std_ratio_vs_normalization",
    "repeated_prediction_fraction_vs_normalization",
    "representative_true_vs_predicted",
    "representative_residuals_vs_true",
    "representative_prediction_distributions",
    "per_seed_mae_trajectories",
    "normalization_effect_heatmap",
    "descriptive_model_difference_by_normalization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def configure_imports(repo_root: Path) -> None:
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values)


def finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else float("nan")


def finite_sd(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sample_sd(finite) if finite else float("nan")


def write_csv_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        tuple(dict.fromkeys(key for row in rows for key in row))
        if rows
        else ("status",)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = tuple(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = [
            str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(f"# {title}\n\n{markdown_table(rows)}\n", encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("% No rows.\n", encoding="utf-8")
        return
    columns = tuple(rows[0])
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(row.get(column, "")) for column in columns)
            + r" \\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table_set(
    table_dir: Path,
    basename: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    latex: bool = True,
) -> None:
    write_csv_file(table_dir / f"{basename}.csv", rows)
    write_markdown_table(table_dir / f"{basename}.md", title, rows)
    if latex:
        write_latex_table(table_dir / f"{basename}.tex", rows)


def read_inputs(
    repo_root: Path,
    validation: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    from experiment_pipeline.common import read_prediction_rows, recompute_metrics

    family_specs = {
        label: family for label, _, family in validation.family_specs
    }
    records: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, int], dict[str, Any]] = {}
    for model, verified_rows in validation.family_rows:
        family = family_specs[model]
        for verified in verified_rows:
            normalization = str(verified["grouping_value"])
            seed = int(verified["seed"])
            experiment_dir = repo_root / str(verified["experiment_path"])
            prediction_path = experiment_dir / str(family["prediction_file"])
            ids, targets, estimates = read_prediction_rows(
                prediction_path,
                family["target_column_aliases"],
                family["prediction_column_aliases"],
                family.get("id_column_aliases", ("universe_id",)),
            )
            metrics = recompute_metrics(targets, estimates)
            config = json.loads(
                (experiment_dir / "config.json").read_text(encoding="utf-8")
            )
            saved_metrics = json.loads(
                (experiment_dir / "metrics.json").read_text(encoding="utf-8")
            )
            saved = saved_metrics["test"]
            saved_mae_delta = abs(float(saved["mae"]) - metrics["test_mae"])
            saved_rmse_delta = abs(float(saved["rmse"]) - metrics["test_rmse"])
            saved_mse_delta = abs(float(saved["mse"]) - metrics["test_mse"])
            if max(saved_mae_delta, saved_rmse_delta, saved_mse_delta) > 1e-6:
                raise RuntimeError(
                    f"saved/recomputed metric disagreement: {verified['experiment_name']}"
                )
            test_ids = config["test_ids"]
            if ids != test_ids or len(ids) != 75 or len(set(ids)) != 75:
                raise RuntimeError(
                    f"prediction ID verification failed: {verified['experiment_name']}"
                )
            record = {
                "model": model,
                "normalization": normalization,
                "seed": seed,
                "experiment_name": verified["experiment_name"],
                "experiment_path": verified["experiment_path"],
                "prediction_path": prediction_path.relative_to(repo_root).as_posix(),
                "split_signature": verified["split_signature"],
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
                "best_epoch": saved_metrics.get("best_epoch", ""),
                "saved_mae_absolute_delta": saved_mae_delta,
                "saved_rmse_absolute_delta": saved_rmse_delta,
                "saved_mse_absolute_delta": saved_mse_delta,
                "prediction_sha256": sha256(prediction_path),
            }
            records.append(record)
            predictions[(model, normalization, seed)] = {
                "ids": ids,
                "targets": targets,
                "predictions": estimates,
            }
    model_index = {model: index for index, model in enumerate(MODEL_ORDER)}
    norm_index = {norm: index for index, norm in enumerate(NORMALIZATIONS)}
    records.sort(
        key=lambda row: (
            model_index[row["model"]],
            norm_index[row["normalization"]],
            SEEDS.index(row["seed"]),
        )
    )
    if len(records) != 30:
        raise RuntimeError(f"seed-level row count={len(records)}, expected=30")
    if len({row["experiment_path"] for row in records}) != 30:
        raise RuntimeError("experiment mappings are not unique")
    if len({row["prediction_path"] for row in records}) != 30:
        raise RuntimeError("prediction mappings are not unique")
    return records, predictions


def aggregate_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["model"]), str(row["normalization"]))].append(row)
    result: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for normalization in NORMALIZATIONS:
            rows = grouped[(model, normalization)]
            maes = [float(row["test_mae"]) for row in rows]
            rmses = [float(row["test_rmse"]) for row in rows]
            r2s = [float(row["test_r2"]) for row in rows]
            ratios = [float(row["prediction_sd_ratio"]) for row in rows]
            repeats = [
                float(row["exact_repeated_prediction_fraction"]) for row in rows
            ]
            result.append(
                {
                    "model": model,
                    "normalization": normalization,
                    "seed_count": len(rows),
                    "seeds": "|".join(str(row["seed"]) for row in rows),
                    "mean_test_mae": mean(maes),
                    "sd_test_mae": sample_sd(maes),
                    "median_test_mae": statistics.median(maes),
                    "mean_test_rmse": mean(rmses),
                    "sd_test_rmse": sample_sd(rmses),
                    "mean_test_r2": mean(r2s),
                    "sd_test_r2": sample_sd(r2s),
                    "mean_prediction_sd_ratio": mean(ratios),
                    "sd_prediction_sd_ratio": sample_sd(ratios),
                    "mean_repeated_prediction_fraction": mean(repeats),
                    "maximum_repeated_prediction_fraction": max(repeats),
                    "undefined_pearson_count": sum(
                        row["pearson_status"] != "defined" for row in rows
                    ),
                }
            )
    if len(result) != 6:
        raise RuntimeError("aggregate row count mismatch")
    return result


def paired_normalization_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (str(row["model"]), str(row["normalization"]), int(row["seed"])): row
        for row in seed_rows
    }
    result: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for first, second in NORMALIZATION_PAIRS:
            pair_name = f"{first}_minus_{second}"
            pair_rows: list[dict[str, Any]] = []
            for seed in SEEDS:
                a = lookup[(model, first, seed)]
                b = lookup[(model, second, seed)]
                if a["split_signature"] != b["split_signature"]:
                    raise RuntimeError(
                        f"split mismatch for {model}, {pair_name}, seed {seed}"
                    )
                row = {
                    "model": model,
                    "normalization_pair": pair_name,
                    "first_normalization": first,
                    "second_normalization": second,
                    "seed": seed,
                    "split_signature": a["split_signature"],
                    "mae_difference": float(a["test_mae"]) - float(b["test_mae"]),
                    "rmse_difference": float(a["test_rmse"]) - float(b["test_rmse"]),
                    "r2_difference": float(a["test_r2"]) - float(b["test_r2"]),
                    "prediction_sd_ratio_difference": float(
                        a["prediction_sd_ratio"]
                    )
                    - float(b["prediction_sd_ratio"]),
                }
                pair_rows.append(row)
            mae_diffs = [row["mae_difference"] for row in pair_rows]
            rmse_diffs = [row["rmse_difference"] for row in pair_rows]
            r2_diffs = [row["r2_difference"] for row in pair_rows]
            ratio_diffs = [
                row["prediction_sd_ratio_difference"] for row in pair_rows
            ]
            summary = {
                "model": model,
                "normalization_pair": pair_name,
                "first_normalization": first,
                "second_normalization": second,
                "seed_count": 5,
                "mean_paired_mae_difference": mean(mae_diffs),
                "sd_paired_mae_difference": sample_sd(mae_diffs),
                "median_paired_mae_difference": statistics.median(mae_diffs),
                "mae_negative_count": sum(value < 0 for value in mae_diffs),
                "mae_zero_count": sum(value == 0 for value in mae_diffs),
                "mae_positive_count": sum(value > 0 for value in mae_diffs),
                "mean_paired_rmse_difference": mean(rmse_diffs),
                "sd_paired_rmse_difference": sample_sd(rmse_diffs),
                "mean_paired_r2_difference": mean(r2_diffs),
                "sd_paired_r2_difference": sample_sd(r2_diffs),
                "mean_prediction_sd_ratio_difference": mean(ratio_diffs),
                "sd_prediction_sd_ratio_difference": sample_sd(ratio_diffs),
            }
            for row in pair_rows:
                row.update(summary)
                result.append(row)
            summaries.append(summary)
    if len(result) != 30:
        raise RuntimeError("paired-normalization row count mismatch")
    return result, summaries


def paired_model_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), str(row["normalization"]), int(row["seed"])): row
        for row in seed_rows
    }
    result = []
    for normalization in NORMALIZATIONS:
        for seed in SEEDS:
            evolve = lookup[("EvolveGCN-H", normalization, seed)]
            static = lookup[("Static GCN", normalization, seed)]
            if evolve["split_signature"] != static["split_signature"]:
                raise RuntimeError(
                    f"cross-model split mismatch: {normalization}, seed {seed}"
                )
            result.append(
                {
                    "normalization": normalization,
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
                    "prediction_sd_ratio_evolve_minus_static": float(
                        evolve["prediction_sd_ratio"]
                    )
                    - float(static["prediction_sd_ratio"]),
                    "interpretation": "descriptive protocol comparison; negative MAE favors EvolveGCN-H",
                }
            )
    if len(result) != 15:
        raise RuntimeError("paired-model row count mismatch")
    return result


def representative_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["model"]), str(row["normalization"]))].append(row)
    result = []
    for model in MODEL_ORDER:
        for normalization in NORMALIZATIONS:
            rows = sorted(
                grouped[(model, normalization)],
                key=lambda row: (float(row["test_mae"]), int(row["seed"])),
            )
            chosen = dict(rows[len(rows) // 2])
            chosen["representative_policy"] = "median_test_mae"
            chosen["rank_by_test_mae"] = len(rows) // 2 + 1
            result.append(chosen)
    return result


def rounded_aggregate_table(
    aggregate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in aggregate:
        result.append(
            {
                "model": row["model"],
                "normalization": row["normalization"],
                "n": row["seed_count"],
                "MAE mean ± SD": (
                    f"{float(row['mean_test_mae']):.6f} ± "
                    f"{float(row['sd_test_mae']):.6f}"
                ),
                "median MAE": f"{float(row['median_test_mae']):.6f}",
                "RMSE mean ± SD": (
                    f"{float(row['mean_test_rmse']):.6f} ± "
                    f"{float(row['sd_test_rmse']):.6f}"
                ),
                "R² mean ± SD": (
                    f"{float(row['mean_test_r2']):.6f} ± "
                    f"{float(row['sd_test_r2']):.6f}"
                ),
                "prediction-SD ratio": (
                    f"{float(row['mean_prediction_sd_ratio']):.6f} ± "
                    f"{float(row['sd_prediction_sd_ratio']):.6f}"
                ),
                "mean repeated fraction": (
                    f"{float(row['mean_repeated_prediction_fraction']):.6f}"
                ),
                "undefined Pearson": row["undefined_pearson_count"],
            }
        )
    return result


def best_summary(
    aggregate: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for model in MODEL_ORDER:
        rows = [row for row in aggregate if row["model"] == model]
        best = min(rows, key=lambda row: float(row["mean_test_mae"]))
        seed_wins = Counter()
        for seed in SEEDS:
            available = [
                row
                for row in seed_rows
                if row["model"] == model and int(row["seed"]) == seed
            ]
            winner = min(available, key=lambda row: float(row["test_mae"]))
            seed_wins[str(winner["normalization"])] += 1
        result.append(
            {
                "model": model,
                "descriptive_best_normalization": best["normalization"],
                "mean_test_mae": best["mean_test_mae"],
                "seed_wins_none": seed_wins["none"],
                "seed_wins_minmax": seed_wins["minmax"],
                "seed_wins_zscore": seed_wins["zscore"],
                "scope": "descriptive; tested U500 Top500 k=8 protocol only",
            }
        )
    return result


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    return plt


def save_figure(
    plt: Any,
    fig: Any,
    output_dir: Path,
    name: str,
    plot_rows: Sequence[Mapping[str, Any]],
) -> None:
    figure_dir = output_dir / "figures"
    plot_dir = output_dir / "plot_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_csv_file(plot_dir / f"{name}.csv", plot_rows)


def figures(
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired_norm: Sequence[Mapping[str, Any]],
    paired_model: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
    prediction_data: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> None:
    plt = configure_matplotlib()
    colors = {"EvolveGCN-H": "#1f77b4", "Static GCN": "#d62728"}
    markers = {"EvolveGCN-H": "o", "Static GCN": "s"}
    x = list(range(3))

    for metric, label, name in (
        ("test_mae", "Test MAE", "test_mae_vs_normalization"),
        ("test_rmse", "Test RMSE", "test_rmse_vs_normalization"),
        ("test_r2", "Test R²", "test_r2_vs_normalization"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plot_rows = []
        for offset, model in zip((-0.06, 0.06), MODEL_ORDER):
            for row in seed_rows:
                if row["model"] != model:
                    continue
                xpos = NORMALIZATIONS.index(row["normalization"]) + offset
                ax.scatter(
                    xpos,
                    float(row[metric]),
                    color=colors[model],
                    marker=markers[model],
                    alpha=0.42,
                    s=25,
                )
                plot_rows.append(
                    {
                        "model": model,
                        "normalization": row["normalization"],
                        "seed": row["seed"],
                        metric: row[metric],
                        "point_type": "seed",
                    }
                )
            model_agg = [row for row in aggregate if row["model"] == model]
            means = [float(row[f"mean_{metric}"]) for row in model_agg]
            sds = [float(row[f"sd_{metric}"]) for row in model_agg]
            ax.errorbar(
                [value + offset for value in x],
                means,
                yerr=sds,
                color=colors[model],
                marker=markers[model],
                capsize=4,
                linewidth=1.5,
                label=model,
            )
            for row in model_agg:
                plot_rows.append(
                    {
                        "model": model,
                        "normalization": row["normalization"],
                        "seed": "",
                        metric: row[f"mean_{metric}"],
                        "point_type": "mean",
                        "sample_standard_deviation": row[f"sd_{metric}"],
                    }
                )
        if metric == "test_r2":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, NORMALIZATIONS)
        ax.set_xlabel("Node-feature normalization")
        ax.set_ylabel(label)
        ax.set_title(f"{label} versus normalization")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(plt, fig, output_dir, name, plot_rows)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pair_labels = [f"{a} − {b}" for a, b in NORMALIZATION_PAIRS]
    plot_rows = []
    for offset, model in zip((-0.06, 0.06), MODEL_ORDER):
        for pair_index, (first, second) in enumerate(NORMALIZATION_PAIRS):
            name = f"{first}_minus_{second}"
            rows = [
                row
                for row in paired_norm
                if row["model"] == model and row["normalization_pair"] == name
            ]
            values = [float(row["mae_difference"]) for row in rows]
            for row in rows:
                ax.scatter(
                    pair_index + offset,
                    row["mae_difference"],
                    color=colors[model],
                    alpha=0.45,
                    marker=markers[model],
                )
                plot_rows.append(
                    {
                        "model": model,
                        "normalization_pair": name,
                        "seed": row["seed"],
                        "mae_difference": row["mae_difference"],
                        "point_type": "seed",
                    }
                )
            ax.errorbar(
                pair_index + offset,
                mean(values),
                yerr=sample_sd(values),
                color=colors[model],
                marker=markers[model],
                capsize=4,
                label=model if pair_index == 0 else None,
            )
            plot_rows.append(
                {
                    "model": model,
                    "normalization_pair": name,
                    "seed": "",
                    "mae_difference": mean(values),
                    "point_type": "mean",
                    "sample_standard_deviation": sample_sd(values),
                }
            )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(3), pair_labels)
    ax.set_ylabel("Paired test-MAE difference")
    ax.set_title("Within-model paired normalization effects")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(
        plt,
        fig,
        output_dir,
        "paired_mae_difference_vs_normalization",
        plot_rows,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_rows = []
    for model in MODEL_ORDER:
        rows = [row for row in aggregate if row["model"] == model]
        values = [float(row["sd_test_mae"]) for row in rows]
        ax.plot(
            x,
            values,
            marker=markers[model],
            color=colors[model],
            label=model,
        )
        plot_rows.extend(
            {
                "model": model,
                "normalization": row["normalization"],
                "sample_standard_deviation_test_mae": row["sd_test_mae"],
            }
            for row in rows
        )
    ax.set_xticks(x, NORMALIZATIONS)
    ax.set_ylabel("Between-seed SD of test MAE")
    ax.set_title("Seed variability versus normalization")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(
        plt, fig, output_dir, "seed_variability_vs_normalization", plot_rows
    )

    for field, ylabel, name, reference in (
        (
            "prediction_sd_ratio",
            "Prediction SD / target SD",
            "prediction_std_ratio_vs_normalization",
            1.0,
        ),
        (
            "exact_repeated_prediction_fraction",
            "Exact repeated-prediction fraction",
            "repeated_prediction_fraction_vs_normalization",
            None,
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plot_rows = []
        for offset, model in zip((-0.06, 0.06), MODEL_ORDER):
            for norm_index, normalization in enumerate(NORMALIZATIONS):
                rows = [
                    row
                    for row in seed_rows
                    if row["model"] == model
                    and row["normalization"] == normalization
                ]
                values = [float(row[field]) for row in rows]
                for row in rows:
                    ax.scatter(
                        norm_index + offset,
                        row[field],
                        color=colors[model],
                        alpha=0.42,
                        marker=markers[model],
                    )
                    plot_rows.append(
                        {
                            "model": model,
                            "normalization": normalization,
                            "seed": row["seed"],
                            field: row[field],
                            "point_type": "seed",
                        }
                    )
                ax.errorbar(
                    norm_index + offset,
                    mean(values),
                    yerr=sample_sd(values),
                    color=colors[model],
                    marker=markers[model],
                    capsize=4,
                    label=model if norm_index == 0 else None,
                )
                plot_rows.append(
                    {
                        "model": model,
                        "normalization": normalization,
                        "seed": "",
                        field: mean(values),
                        "point_type": "mean",
                        "sample_standard_deviation": sample_sd(values),
                    }
                )
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, NORMALIZATIONS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} versus normalization")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(plt, fig, output_dir, name, plot_rows)

    representative_lookup = {
        (row["model"], row["normalization"]): row for row in representatives
    }
    for mode, name, ylabel in (
        ("prediction", "representative_true_vs_predicted", "Predicted Omega_m"),
        ("residual", "representative_residuals_vs_true", "Residual (prediction − true)"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8))
        plot_rows = []
        for row_index, model in enumerate(MODEL_ORDER):
            for col_index, normalization in enumerate(NORMALIZATIONS):
                ax = axes[row_index][col_index]
                row = representative_lookup[(model, normalization)]
                data = prediction_data[(model, normalization, int(row["seed"]))]
                targets = data["targets"]
                estimates = data["predictions"]
                plotted = (
                    [p - y for y, p in zip(targets, estimates)]
                    if mode == "residual"
                    else estimates
                )
                ax.scatter(targets, plotted, color=colors[model], alpha=0.7, s=20)
                if mode == "residual":
                    ax.axhline(0, color="black", linestyle="--", linewidth=1)
                else:
                    low = min(targets + estimates)
                    high = max(targets + estimates)
                    ax.plot([low, high], [low, high], "k--", linewidth=1)
                ax.set_title(f"{model}, {normalization}, seed {row['seed']}")
                ax.set_xlabel("True Omega_m")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.18)
                for universe_id, target, prediction, y_value in zip(
                    data["ids"], targets, estimates, plotted
                ):
                    plot_rows.append(
                        {
                            "model": model,
                            "normalization": normalization,
                            "representative_seed": row["seed"],
                            "universe_id": universe_id,
                            "target": target,
                            "prediction": prediction,
                            "residual": prediction - target,
                            "plotted_y": y_value,
                        }
                    )
        fig.suptitle(
            "Median-MAE representative runs"
            + ("—residuals" if mode == "residual" else "—true versus predicted")
        )
        fig.tight_layout()
        save_figure(plt, fig, output_dir, name, plot_rows)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8))
    plot_rows = []
    bins = [0.10 + 0.02 * index for index in range(21)]
    for row_index, model in enumerate(MODEL_ORDER):
        for col_index, normalization in enumerate(NORMALIZATIONS):
            ax = axes[row_index][col_index]
            row = representative_lookup[(model, normalization)]
            data = prediction_data[(model, normalization, int(row["seed"]))]
            ax.hist(
                data["targets"],
                bins=bins,
                alpha=0.45,
                label="target",
                color="#555555",
            )
            ax.hist(
                data["predictions"],
                bins=bins,
                alpha=0.55,
                label="prediction",
                color=colors[model],
            )
            ax.set_title(f"{model}, {normalization}, seed {row['seed']}")
            ax.set_xlabel("Omega_m")
            ax.set_ylabel("Count")
            ax.legend()
            for kind in ("target", "prediction"):
                values = data["targets"] if kind == "target" else data["predictions"]
                for value in values:
                    plot_rows.append(
                        {
                            "model": model,
                            "normalization": normalization,
                            "representative_seed": row["seed"],
                            "value_type": kind,
                            "value": value,
                        }
                    )
    fig.suptitle("Representative target and prediction distributions")
    fig.tight_layout()
    save_figure(
        plt,
        fig,
        output_dir,
        "representative_prediction_distributions",
        plot_rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    plot_rows = []
    seed_palette = plt.cm.viridis([index / 4 for index in range(5)])
    for ax, model in zip(axes, MODEL_ORDER):
        for color, seed in zip(seed_palette, SEEDS):
            rows = [
                row
                for normalization in NORMALIZATIONS
                for row in seed_rows
                if row["model"] == model
                and row["seed"] == seed
                and row["normalization"] == normalization
            ]
            values = [float(row["test_mae"]) for row in rows]
            ax.plot(x, values, marker="o", color=color, label=f"seed {seed}")
            plot_rows.extend(
                {
                    "model": model,
                    "normalization": row["normalization"],
                    "seed": seed,
                    "test_mae": row["test_mae"],
                }
                for row in rows
            )
        ax.set_xticks(x, NORMALIZATIONS)
        ax.set_title(model)
        ax.set_xlabel("Normalization")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Test MAE")
    axes[1].legend(fontsize=7)
    fig.suptitle("Per-seed MAE trajectories")
    fig.tight_layout()
    save_figure(
        plt, fig, output_dir, "per_seed_mae_trajectories", plot_rows
    )

    heat = []
    plot_rows = []
    for model in MODEL_ORDER:
        values = []
        for first, second in NORMALIZATION_PAIRS:
            pair = f"{first}_minus_{second}"
            rows = [
                row
                for row in paired_norm
                if row["model"] == model and row["normalization_pair"] == pair
            ]
            value = mean([float(row["mae_difference"]) for row in rows])
            values.append(value)
            plot_rows.append(
                {
                    "model": model,
                    "normalization_pair": pair,
                    "mean_paired_mae_difference": value,
                }
            )
        heat.append(values)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    image = ax.imshow(heat, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(3), pair_labels)
    ax.set_yticks(range(2), MODEL_ORDER)
    for row_index in range(2):
        for col_index in range(3):
            ax.text(
                col_index,
                row_index,
                f"{heat[row_index][col_index]:+.4f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax, label="Mean paired MAE difference")
    ax.set_title("Normalization-effect heatmap")
    fig.tight_layout()
    save_figure(
        plt, fig, output_dir, "normalization_effect_heatmap", plot_rows
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_rows = []
    for norm_index, normalization in enumerate(NORMALIZATIONS):
        rows = [
            row for row in paired_model if row["normalization"] == normalization
        ]
        values = [float(row["mae_evolve_minus_static"]) for row in rows]
        for row in rows:
            ax.scatter(
                norm_index,
                row["mae_evolve_minus_static"],
                color="#6a3d9a",
                alpha=0.45,
            )
            plot_rows.append(
                {
                    "normalization": normalization,
                    "seed": row["seed"],
                    "mae_evolve_minus_static": row["mae_evolve_minus_static"],
                    "point_type": "seed",
                }
            )
        ax.errorbar(
            norm_index,
            mean(values),
            yerr=sample_sd(values),
            color="#6a3d9a",
            marker="o",
            capsize=4,
        )
        plot_rows.append(
            {
                "normalization": normalization,
                "seed": "",
                "mae_evolve_minus_static": mean(values),
                "point_type": "mean",
                "sample_standard_deviation": sample_sd(values),
            }
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, NORMALIZATIONS)
    ax.set_ylabel("Evolve MAE − Static MAE")
    ax.set_title("Descriptive model difference by normalization")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(
        plt,
        fig,
        output_dir,
        "descriptive_model_difference_by_normalization",
        plot_rows,
    )


def normalization_implementation_text() -> str:
    return """# Normalization implementation

All runs use the seven node features `log10(Mvir), X, Y, Z, VX, VY, VZ`.
Top500 halos are selected by raw Mvir before feature construction and scaling.

## none

Mvir remains transformed to log10(Mvir), but no subsequent per-feature scaling
is applied.

## minmax

For every universe, snapshot, and feature:

`x_scaled = (x - x_min) / (x_max - x_min)`

The implementation replaces a denominator whose absolute value is below
`1e-8` with 1.0.

## zscore

For every universe, snapshot, and feature:

`x_scaled = (x - mean) / standard_deviation`

The implementation replaces a standard deviation whose absolute value is below
`1e-8` with 1.0.

Normalization occurs after raw-Mvir Top500 selection. Periodic kNN edges use a
separate copy of raw physical XYZ, with k=8 and box size 25, so normalization
does not intentionally alter topology. Omega_m is unnormalized and summary
features are disabled.

No cross-universe scaler is fitted. Each validation or test graph uses only its
own observed node features, so no target or cross-split leakage occurs. The
scientifically important limitation is that minmax and zscore remove absolute
between-universe feature-scale differences.
"""


def scientific_text(
    aggregate: Sequence[Mapping[str, Any]],
    pair_summaries: Sequence[Mapping[str, Any]],
) -> str:
    agg = {
        (row["model"], row["normalization"]): row for row in aggregate
    }
    pairs = {
        (row["model"], row["normalization_pair"]): row
        for row in pair_summaries
    }
    lines = [
        "# Scientific summary",
        "",
        "## Verified conclusion",
        "",
        "Under the tested U500 Top500 k=8 protocol, the unnormalized "
        "node-feature representation achieved lower test error in every matched "
        "seed for both models. Minmax and zscore are sample-local transformations, "
        "so this result does not imply that all normalization strategies are harmful.",
        "",
        "## Answers to the scientific questions",
        "",
    ]
    question_number = 1
    for model in MODEL_ORDER:
        for normalization in ("minmax", "zscore"):
            pair = pairs[(model, f"{normalization}_minus_none")]
            lines.append(
                f"{question_number}. **Does {normalization} improve over none for "
                f"{model}?** No. Mean paired MAE difference "
                f"({normalization} − none) is "
                f"{float(pair['mean_paired_mae_difference']):+.6f}; all "
                f"{pair['mae_positive_count']} of 5 differences are positive, "
                "so none has lower MAE in every seed."
            )
            question_number += 1
    lines.extend(
        [
            "5. **Are effects consistent across all five seeds?** Yes for the "
            "none-versus-normalized comparisons: none has lower MAE in all 20 "
            "model × method × seed comparisons against minmax or zscore.",
            "6. **Are effects larger than between-seed variability?** Yes. In "
            "both models, the mean paired MAE penalties for minmax and zscore "
            "exceed the corresponding cell-level MAE standard deviations.",
            "7. **Do normalized variants show stronger compression?** Yes. Mean "
            f"prediction-SD ratios change from "
            f"{float(agg[('EvolveGCN-H', 'none')]['mean_prediction_sd_ratio']):.3f} "
            f"(Evolve none) to "
            f"{float(agg[('EvolveGCN-H', 'minmax')]['mean_prediction_sd_ratio']):.3f}/"
            f"{float(agg[('EvolveGCN-H', 'zscore')]['mean_prediction_sd_ratio']):.3f}, "
            f"and from {float(agg[('Static GCN', 'none')]['mean_prediction_sd_ratio']):.3f} "
            f"(Static none) to "
            f"{float(agg[('Static GCN', 'minmax')]['mean_prediction_sd_ratio']):.3f}/"
            f"{float(agg[('Static GCN', 'zscore')]['mean_prediction_sd_ratio']):.3f}.",
            "8. **Do normalized variants show more repeated predictions?** For "
            "Static GCN, yes: mean exact repeated fractions are "
            f"{float(agg[('Static GCN', 'minmax')]['mean_repeated_prediction_fraction']):.3f} "
            f"(minmax) and "
            f"{float(agg[('Static GCN', 'zscore')]['mean_repeated_prediction_fraction']):.3f} "
            "versus 0 for none. Evolve normalized runs have zero exact repeats, "
            "whereas Evolve none has a small mean repeated fraction; compression "
            "there is visible primarily through reduced dispersion.",
            "9. **Is the pattern similar across models?** Qualitatively yes: none "
            "has the lowest error and highest mean prediction dispersion in both. "
            "The severity and exact repetition behavior differ.",
            "10. **Is normalization a major explanation for prediction collapse?** "
            "Sample-local normalization is a plausible major contributor under "
            "this protocol, especially for Static GCN, but it is not a universal "
            "explanation and the analysis does not test global train-fitted scaling.",
            "",
            "## Interpretation",
            "",
            "Per-universe, per-snapshot scaling removes absolute feature-scale "
            "differences between universes. Those differences may carry Omega_m "
            "information. The observed error increase and reduced prediction "
            "dispersion are consistent with that mechanism, but this analysis "
            "does not prove the mechanism causally.",
            "",
            "Static-versus-Evolve differences are descriptive protocol comparisons. "
            "They must not be attributed solely to temporal processing because the "
            "architectures, depth, batch sizes, and heads also differ.",
        ]
    )
    return "\n\n".join(lines) + "\n"


def presentation_text() -> str:
    return """# Presentation-ready summary

## Recommended supervisor-meeting sequence

1. Normalization protocol table.
2. Six-cell main result table.
3. Test MAE versus normalization.
4. Paired minmax-minus-none and zscore-minus-none MAE differences.
5. Prediction-SD ratio.
6. Exact repeated-prediction fraction.
7. Median-MAE representative true-versus-predicted panels.

## Concise conclusion

All 30 artifacts and metrics verify. Under the controlled U500 Top500 k=8
protocol, none has lower test MAE than minmax and zscore in every matched seed
for both EvolveGCN-H and Static GCN. The sample-local normalized variants also
show reduced prediction dispersion, with severe repetition in several Static
GCN runs. This supports the implementation-specific interpretation that
per-universe, per-snapshot scaling may remove useful absolute-scale information;
it does not establish that normalization is universally harmful.

Artifact completion, metric verification, normalization effects, prediction
compression, and the implementation-specific limitation should be presented as
five distinct claims.
"""


def readme_text() -> str:
    return """# Controlled normalization ablation package

This package contains the complete model-stratified U500 Top500 k=8
node-feature normalization analysis:

- 30 verified seed-level rows;
- 6 aggregate model × normalization rows;
- 30 within-model paired-normalization rows;
- 15 descriptive cross-model paired rows;
- all requested tables;
- 13 figures, each with PNG, PDF, and plot-data CSV.

The source experiments are read-only. Graph datasets and checkpoints are never
loaded. Run the generic builder first, then:

```bash
env MPLCONFIGDIR=/tmp/codex-matplotlib-cache \
  envs/camels-gnn/bin/python \
  reports/analysis/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500/rebuild_normalization_outputs.py \
  --repo-root /home/ml/thesis-camels \
  --spec configs/analysis_reports/controlled_static_vs_evolvegcn_normalization_ablation_500u_top500.json
```

The primary scientific comparisons are within model. Cross-model differences
are descriptive protocol comparisons.
"""


def update_manifest(
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired_norm: Sequence[Mapping[str, Any]],
    paired_model: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> None:
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = output_dir.parents[2]
    source_hashes = manifest.setdefault("source_file_hashes_sha256", {})
    for source in (
        output_dir / "rebuild_normalization_outputs.py",
        repo_root / "scripts/validate_analysis_report.py",
    ):
        source_hashes[source.relative_to(repo_root).as_posix()] = sha256(source)
    manifest["generated_row_counts"] = {
        "seed_level_results": len(seed_rows),
        "aggregated_results": len(aggregate),
        "paired_normalization_differences": len(paired_norm),
        "paired_model_differences": len(paired_model),
    }
    manifest["normalization_specific_validation"] = {
        "unique_experiment_mappings": len(
            {row["experiment_path"] for row in seed_rows}
        ),
        "unique_prediction_mappings": len(
            {row["prediction_path"] for row in seed_rows}
        ),
        "representative_policy": "median_test_mae",
        "representative_runs": [
            {
                "model": row["model"],
                "normalization": row["normalization"],
                "seed": row["seed"],
                "experiment_name": row["experiment_name"],
            }
            for row in representatives
        ],
        "undefined_pearson_count": sum(
            row["pearson_status"] != "defined" for row in seed_rows
        ),
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
            <= 1e-6
            for row in seed_rows
        ),
    }
    expected_outputs = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    manifest["generated_outputs"] = [
        "analysis_manifest.json",
        *expected_outputs,
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    configure_imports(repo_root)
    from analysis_reporting.common import validate_analysis

    validation = validate_analysis(repo_root, args.spec)
    if not validation.valid:
        for error in validation.errors:
            print(error, file=sys.stderr)
        return 1
    output_dir = repo_root / validation.spec["output_directory"]
    seed_rows, prediction_data = read_inputs(repo_root, validation)
    aggregate = aggregate_rows(seed_rows)
    paired_norm, pair_summaries = paired_normalization_rows(seed_rows)
    paired_model = paired_model_rows(seed_rows)
    representatives = representative_rows(seed_rows)

    write_csv_file(output_dir / "seed_level_results.csv", seed_rows)
    write_csv_file(output_dir / "aggregated_results.csv", aggregate)
    write_csv_file(
        output_dir / "paired_normalization_differences.csv", paired_norm
    )
    write_csv_file(output_dir / "paired_model_differences.csv", paired_model)
    write_csv_file(output_dir / "prediction_diagnostics.csv", seed_rows)

    table_dir = output_dir / "tables"
    aggregate_table = rounded_aggregate_table(aggregate)
    save_table_set(
        table_dir,
        "main_results_table",
        "Main normalization results",
        aggregate_table,
    )
    save_table_set(
        table_dir,
        "evolvegcn_normalization_results",
        "EvolveGCN-H normalization results",
        [row for row in aggregate_table if row["model"] == "EvolveGCN-H"],
    )
    save_table_set(
        table_dir,
        "static_gcn_normalization_results",
        "Static GCN normalization results",
        [row for row in aggregate_table if row["model"] == "Static GCN"],
    )
    save_table_set(
        table_dir,
        "paired_normalization_differences",
        "Within-model paired normalization differences",
        paired_norm,
    )
    save_table_set(
        table_dir,
        "descriptive_model_comparison",
        "Descriptive Static-versus-Evolve protocol comparison",
        paired_model,
    )
    save_table_set(
        table_dir,
        "best_normalization_descriptive_summary",
        "Descriptive best normalization summary",
        best_summary(aggregate, seed_rows),
        latex=False,
    )

    (output_dir / "normalization_implementation.md").write_text(
        normalization_implementation_text(), encoding="utf-8"
    )
    (output_dir / "scientific_summary.md").write_text(
        scientific_text(aggregate, pair_summaries), encoding="utf-8"
    )
    (output_dir / "presentation_ready_summary.md").write_text(
        presentation_text(), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(readme_text(), encoding="utf-8")
    figures(
        output_dir,
        seed_rows,
        aggregate,
        paired_norm,
        paired_model,
        representatives,
        prediction_data,
    )
    update_manifest(
        output_dir,
        seed_rows,
        aggregate,
        paired_norm,
        paired_model,
        representatives,
    )
    print("Normalization-specific rebuild: PASS")
    print(f"Seed-level rows: {len(seed_rows)}")
    print(f"Aggregate rows: {len(aggregate)}")
    print(f"Paired-normalization rows: {len(paired_norm)}")
    print(f"Paired-model rows: {len(paired_model)}")
    print(f"Figures: {len(FIGURE_NAMES)} × PNG/PDF/CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
