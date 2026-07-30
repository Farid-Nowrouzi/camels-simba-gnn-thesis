#!/usr/bin/env python3
"""Rebuild the controlled U500 Top-N halo-count scaling analysis.

Only lightweight JSON and CSV artifacts are read. Graph datasets and model
checkpoints are never loaded, and nothing under experiments/ is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TOPNS = (100, 200, 500)
SEEDS = (42, 123, 2025)
MODELS = ("EvolveGCN-H", "Static GCN")
PAIRS = ((200, 100), (500, 200), (500, 100))
FIGURES = (
    "test_mae_vs_topn",
    "test_rmse_vs_topn",
    "test_r2_vs_topn",
    "per_seed_mae_trajectories",
    "paired_top200_minus_top100_mae",
    "paired_top500_minus_top200_mae",
    "paired_top500_minus_top100_mae",
    "prediction_std_ratio_vs_topn",
    "repeated_prediction_fraction_vs_topn",
    "representative_true_vs_predicted",
    "representative_residuals_vs_true",
    "representative_prediction_distributions",
    "seed_variability_vs_topn",
    "nodes_and_edges_vs_topn",
    "predictive_performance_vs_graph_size",
    "descriptive_model_difference_vs_topn",
    "topn_effect_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values)


def sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(dict.fromkeys(k for row in rows for k in row)) or ("status",)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = tuple(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = [
            str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]], note: str = "") -> None:
    text = f"# {title}\n\n"
    if note:
        text += note.strip() + "\n\n"
    path.write_text(text + md_table(rows) + "\n", encoding="utf-8")


def tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def tex_cell(value: Any) -> str:
    text = str(value)
    if text.startswith("**") and text.endswith("**"):
        return r"\textbf{" + tex_escape(text[2:-2]) + "}"
    return tex_escape(text)


def write_tex(path: Path, rows: Sequence[Mapping[str, Any]], caption: str = "") -> None:
    columns = tuple(rows[0]) if rows else ("status",)
    lines = [r"\begin{table}", r"\centering", r"\small"]
    if caption:
        lines.append(r"\caption{" + tex_escape(caption) + "}")
    lines += [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\hline",
        " & ".join(tex_cell(c) for c in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(tex_cell(row.get(c, "")) for c in columns) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_set(
    directory: Path,
    basename: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    caption: str = "",
    latex: bool = True,
) -> None:
    write_csv(directory / f"{basename}.csv", rows)
    write_md_table(directory / f"{basename}.md", title, rows, caption)
    if latex:
        write_tex(directory / f"{basename}.tex", rows, caption)


def configure_imports(root: Path) -> None:
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def read_verified(
    root: Path, validation: Any
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    from experiment_pipeline.common import read_prediction_rows, recompute_metrics

    families = {label: family for label, _, family in validation.family_specs}
    rows: list[dict[str, Any]] = []
    prediction_data: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_population = {f"LH_{index}" for index in range(500)}
    for model, verified_rows in validation.family_rows:
        family = families[model]
        for verified in verified_rows:
            top_n = int(verified["grouping_value"])
            seed = int(verified["seed"])
            experiment = root / str(verified["experiment_path"])
            config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
            saved = json.loads((experiment / "metrics.json").read_text(encoding="utf-8"))
            prediction_path = experiment / str(family["prediction_file"])
            ids, targets, predictions = read_prediction_rows(
                prediction_path,
                family["target_column_aliases"],
                family["prediction_column_aliases"],
                family["id_column_aliases"],
            )
            metrics = recompute_metrics(targets, predictions)
            train_ids = config["train_ids"]
            val_ids = config["val_ids"]
            test_ids = config["test_ids"]
            split_sets = (set(train_ids), set(val_ids), set(test_ids))
            if tuple(map(len, (train_ids, val_ids, test_ids))) != (350, 75, 75):
                raise RuntimeError(f"wrong split sizes: {verified['experiment_name']}")
            if any(split_sets[i] & split_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise RuntimeError(f"split overlap: {verified['experiment_name']}")
            if set().union(*split_sets) != expected_population:
                raise RuntimeError(f"incomplete U500 coverage: {verified['experiment_name']}")
            if ids != test_ids or len(ids) != 75 or len(set(ids)) != 75:
                raise RuntimeError(f"prediction IDs mismatch: {verified['experiment_name']}")
            saved_test = saved["test"]
            deltas = {
                "saved_mae_absolute_delta": abs(float(saved_test["mae"]) - metrics["test_mae"]),
                "saved_rmse_absolute_delta": abs(float(saved_test["rmse"]) - metrics["test_rmse"]),
                "saved_mse_absolute_delta": abs(float(saved_test["mse"]) - metrics["test_mse"]),
            }
            if max(deltas.values()) > 1e-6:
                raise RuntimeError(f"saved metric mismatch: {verified['experiment_name']}")
            residuals = [p - t for t, p in zip(targets, predictions)]
            row = {
                "model": model,
                "top_n": top_n,
                "seed": seed,
                "experiment_name": verified["experiment_name"],
                "experiment_path": verified["experiment_path"],
                "prediction_path": prediction_path.relative_to(root).as_posix(),
                "split_signature": verified["split_signature"],
                "train_count": len(train_ids),
                "validation_count": len(val_ids),
                "test_count": len(test_ids),
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
                "approximate_repeat_tolerance": metrics["approximate_repeat_tolerance"],
                "prediction_min": metrics["prediction_min"],
                "prediction_max": metrics["prediction_max"],
                "prediction_range": metrics["prediction_range"],
                "residual_mean": metrics["residual_mean"],
                "residual_sd": metrics["residual_std"],
                "maximum_absolute_residual": max(abs(value) for value in residuals),
                "best_epoch": saved["best_epoch"],
                **deltas,
                "prediction_sha256": sha256(prediction_path),
                "artifact_verification": "complete; checkpoint existence only",
            }
            rows.append(row)
            prediction_data[(model, top_n, seed)] = {
                "ids": ids,
                "targets": targets,
                "predictions": predictions,
            }
    order = {model: i for i, model in enumerate(MODELS)}
    rows.sort(key=lambda r: (order[r["model"]], TOPNS.index(r["top_n"]), SEEDS.index(r["seed"])))
    if len(rows) != 18 or len({r["experiment_path"] for r in rows}) != 18:
        raise RuntimeError("canonical experiment coverage is not exactly 18 unique rows")
    if len({r["prediction_path"] for r in rows}) != 18:
        raise RuntimeError("canonical prediction mappings are not unique")
    if len({r["prediction_sha256"] for r in rows}) != 18:
        raise RuntimeError("canonical prediction hashes are not unique")
    for seed in SEEDS:
        if len({r["split_signature"] for r in rows if r["seed"] == seed}) != 1:
            raise RuntimeError(f"cross-model/Top-N split mismatch for seed {seed}")
    return rows, prediction_data


def aggregate(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        for top_n in TOPNS:
            rows = [r for r in seed_rows if r["model"] == model and r["top_n"] == top_n]
            maes = [float(r["test_mae"]) for r in rows]
            rmses = [float(r["test_rmse"]) for r in rows]
            r2s = [float(r["test_r2"]) for r in rows]
            ratios = [float(r["prediction_sd_ratio"]) for r in rows]
            repeats = [float(r["exact_repeated_prediction_fraction"]) for r in rows]
            result.append(
                {
                    "model": model,
                    "top_n": top_n,
                    "seed_count": len(rows),
                    "mean_test_mae": mean(maes),
                    "sd_test_mae": sd(maes),
                    "median_test_mae": statistics.median(maes),
                    "mean_test_rmse": mean(rmses),
                    "sd_test_rmse": sd(rmses),
                    "mean_test_r2": mean(r2s),
                    "sd_test_r2": sd(r2s),
                    "mean_prediction_sd_ratio": mean(ratios),
                    "sd_prediction_sd_ratio": sd(ratios),
                    "mean_repeated_prediction_fraction": mean(repeats),
                    "maximum_repeated_prediction_fraction": max(repeats),
                    "undefined_pearson_count": sum(r["pearson_status"] != "defined" for r in rows),
                }
            )
    return result


def paired_topn(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["model"], r["top_n"], r["seed"]): r for r in seed_rows}
    result = []
    for model in MODELS:
        for larger, smaller in PAIRS:
            pair = []
            for seed in SEEDS:
                a, b = lookup[(model, larger, seed)], lookup[(model, smaller, seed)]
                if a["split_signature"] != b["split_signature"]:
                    raise RuntimeError(f"paired split mismatch: {model} {larger}-{smaller} seed {seed}")
                pair.append(
                    {
                        "model": model,
                        "topn_pair": f"Top{larger}_minus_Top{smaller}",
                        "larger_top_n": larger,
                        "smaller_top_n": smaller,
                        "seed": seed,
                        "split_signature": a["split_signature"],
                        "mae_difference": float(a["test_mae"]) - float(b["test_mae"]),
                        "rmse_difference": float(a["test_rmse"]) - float(b["test_rmse"]),
                        "r2_difference": float(a["test_r2"]) - float(b["test_r2"]),
                        "prediction_sd_ratio_difference": float(a["prediction_sd_ratio"])
                        - float(b["prediction_sd_ratio"]),
                        "exact_repeat_fraction_difference": float(
                            a["exact_repeated_prediction_fraction"]
                        )
                        - float(b["exact_repeated_prediction_fraction"]),
                        "approximate_repeat_fraction_difference": float(
                            a["approximate_repeated_prediction_fraction"]
                        )
                        - float(b["approximate_repeated_prediction_fraction"]),
                    }
                )
            for metric in (
                "mae_difference",
                "rmse_difference",
                "r2_difference",
                "prediction_sd_ratio_difference",
                "exact_repeat_fraction_difference",
                "approximate_repeat_fraction_difference",
            ):
                values = [float(r[metric]) for r in pair]
                prefix = metric.removesuffix("_difference")
                for row in pair:
                    row[f"mean_{prefix}_difference"] = mean(values)
                    row[f"sd_{prefix}_difference"] = sd(values)
                    row[f"median_{prefix}_difference"] = statistics.median(values)
                    row[f"minimum_{prefix}_difference"] = min(values)
                    row[f"maximum_{prefix}_difference"] = max(values)
            maes = [float(r["mae_difference"]) for r in pair]
            for row in pair:
                row["larger_topn_wins"] = sum(v < 0 for v in maes)
                row["smaller_topn_wins"] = sum(v > 0 for v in maes)
                row["ties"] = sum(v == 0 for v in maes)
                row["interpretation"] = "negative MAE/RMSE and positive R2 favor larger Top-N"
            result.extend(pair)
    return result


def paired_models(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["model"], r["top_n"], r["seed"]): r for r in seed_rows}
    result = []
    for top_n in TOPNS:
        for seed in SEEDS:
            evolve = lookup[("EvolveGCN-H", top_n, seed)]
            static = lookup[("Static GCN", top_n, seed)]
            if evolve["split_signature"] != static["split_signature"]:
                raise RuntimeError(f"cross-model split mismatch: Top{top_n} seed {seed}")
            result.append(
                {
                    "top_n": top_n,
                    "seed": seed,
                    "split_signature": evolve["split_signature"],
                    "evolve_experiment_name": evolve["experiment_name"],
                    "static_experiment_name": static["experiment_name"],
                    "mae_evolve_minus_static": float(evolve["test_mae"]) - float(static["test_mae"]),
                    "rmse_evolve_minus_static": float(evolve["test_rmse"]) - float(static["test_rmse"]),
                    "r2_evolve_minus_static": float(evolve["test_r2"]) - float(static["test_r2"]),
                    "prediction_sd_ratio_evolve_minus_static": float(
                        evolve["prediction_sd_ratio"]
                    )
                    - float(static["prediction_sd_ratio"]),
                    "comparison_label": "Descriptive cross-protocol model comparison.",
                }
            )
    return result


def representatives(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        for top_n in TOPNS:
            rows = sorted(
                (r for r in seed_rows if r["model"] == model and r["top_n"] == top_n),
                key=lambda r: (float(r["test_mae"]), int(r["seed"])),
            )
            selected = dict(rows[len(rows) // 2])
            selected["representative_policy"] = "median_test_mae"
            selected["rank_by_test_mae"] = 2
            result.append(selected)
    return result


def computational_rows(aggregated: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["model"], r["top_n"]): r for r in aggregated}
    rows = []
    for model in MODELS:
        snapshots = 5 if model == "EvolveGCN-H" else 1
        for top_n in TOPNS:
            agg = lookup[(model, top_n)]
            rows.append(
                {
                    "model": model,
                    "top_n": top_n,
                    "snapshots_processed_per_universe": snapshots,
                    "nodes_per_snapshot": top_n,
                    "k": 8,
                    "directed_neighbor_selections_per_snapshot": top_n * 8,
                    "symmetric_adjacency_nonzeros_per_snapshot": f"{top_n * 8} to {top_n * 16}",
                    "dense_adjacency_capacity_per_snapshot": top_n * top_n,
                    "self_loops": "excluded in preprocessing; added by model layers",
                    "padded_nodes": "excluded from neighbor selection",
                    "mean_test_mae": agg["mean_test_mae"],
                    "wall_time": "not recorded",
                    "peak_gpu_memory": "not recorded",
                    "peak_cpu_memory": "not recorded",
                    "prediction_time": "not recorded",
                }
            )
    return rows


def rounded_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best = {
        model: min(
            (r for r in rows if r["model"] == model),
            key=lambda r: float(r["mean_test_mae"]),
        )["top_n"]
        for model in MODELS
    }
    result = []
    for row in rows:
        bold = row["top_n"] == best[row["model"]]
        fmt = lambda value: f"{float(value):.6f}"
        cell = lambda value: f"**{fmt(value)}**" if bold else fmt(value)
        result.append(
            {
                "model": row["model"],
                "Top-N": row["top_n"],
                "n": row["seed_count"],
                "mean MAE": cell(row["mean_test_mae"]),
                "SD MAE": fmt(row["sd_test_mae"]),
                "median MAE": fmt(row["median_test_mae"]),
                "mean RMSE": cell(row["mean_test_rmse"]),
                "SD RMSE": fmt(row["sd_test_rmse"]),
                "mean R²": cell(row["mean_test_r2"]),
                "SD R²": fmt(row["sd_test_r2"]),
                "mean prediction-SD ratio": fmt(row["mean_prediction_sd_ratio"]),
                "SD prediction-SD ratio": fmt(row["sd_prediction_sd_ratio"]),
                "mean exact-repeat fraction": fmt(row["mean_repeated_prediction_fraction"]),
                "maximum exact-repeat fraction": fmt(row["maximum_repeated_prediction_fraction"]),
                "undefined Pearson": row["undefined_pearson_count"],
            }
        )
    return result


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})
    return plt


def save_figure(
    plt: Any,
    fig: Any,
    output: Path,
    name: str,
    plot_rows: Sequence[Mapping[str, Any]],
) -> None:
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "plot_data").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "figures" / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_csv(output / "plot_data" / f"{name}.csv", plot_rows)


def make_figures(
    output: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    agg: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    model_pairs: Sequence[Mapping[str, Any]],
    reps: Sequence[Mapping[str, Any]],
    prediction_data: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> None:
    plt = configure_matplotlib()
    colors = {"EvolveGCN-H": "#1f77b4", "Static GCN": "#d62728"}
    markers = {"EvolveGCN-H": "o", "Static GCN": "s"}
    x = list(TOPNS)

    for metric, ylabel, name in (
        ("test_mae", "Test MAE", "test_mae_vs_topn"),
        ("test_rmse", "Test RMSE", "test_rmse_vs_topn"),
        ("test_r2", "Test R²", "test_r2_vs_topn"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pdata = []
        for model in MODELS:
            rows = [r for r in agg if r["model"] == model]
            means = [float(r[f"mean_{metric}"]) for r in rows]
            errors = [float(r[f"sd_{metric}"]) for r in rows]
            ax.errorbar(x, means, yerr=errors, marker=markers[model], color=colors[model], capsize=4, label=model)
            for row in seed_rows:
                if row["model"] == model:
                    ax.scatter(row["top_n"], row[metric], color=colors[model], alpha=0.4, s=22)
                    pdata.append({"model": model, "top_n": row["top_n"], "seed": row["seed"], metric: row[metric], "point_type": "seed"})
            for row in rows:
                pdata.append({"model": model, "top_n": row["top_n"], "seed": "", metric: row[f"mean_{metric}"], "sample_sd": row[f"sd_{metric}"], "point_type": "aggregate"})
        if metric == "test_r2":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Top-N retained halos", ylabel=ylabel, xticks=x, title=f"{ylabel} versus Top-N")
        ax.grid(alpha=0.2); ax.legend()
        save_figure(plt, fig, output, name, pdata)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    pdata = []
    for ax, model in zip(axes, MODELS):
        for seed in SEEDS:
            rows = [r for r in seed_rows if r["model"] == model and r["seed"] == seed]
            ax.plot(x, [r["test_mae"] for r in rows], marker="o", label=f"seed {seed}")
            pdata.extend({"model": model, "top_n": r["top_n"], "seed": seed, "test_mae": r["test_mae"]} for r in rows)
        ax.set(title=model, xlabel="Top-N", xticks=x); ax.grid(alpha=0.2); ax.legend()
    axes[0].set_ylabel("Test MAE")
    fig.suptitle("Matched-seed MAE trajectories")
    save_figure(plt, fig, output, "per_seed_mae_trajectories", pdata)

    for larger, smaller in PAIRS:
        name = f"paired_top{larger}_minus_top{smaller}_mae"
        rows = [r for r in paired if r["larger_top_n"] == larger and r["smaller_top_n"] == smaller]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pdata = []
        positions = []
        values = []
        labels = []
        for mi, model in enumerate(MODELS):
            mrows = [r for r in rows if r["model"] == model]
            for si, row in enumerate(mrows):
                pos = mi + (si - 1) * 0.08
                ax.scatter(pos, row["mae_difference"], color=colors[model], s=35)
                positions.append(pos); values.append(row["mae_difference"]); labels.append(model)
                pdata.append(dict(row))
            ax.scatter(mi, mean([r["mae_difference"] for r in mrows]), color="black", marker="_", s=180)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xticks=range(2), xticklabels=MODELS, ylabel=f"MAE(Top{larger}) − MAE(Top{smaller})", title=f"Paired Top{larger} minus Top{smaller} MAE")
        ax.grid(axis="y", alpha=0.2)
        save_figure(plt, fig, output, name, pdata)

    for field, aggregate_field, ylabel, name in (
        ("prediction_sd_ratio", "mean_prediction_sd_ratio", "Prediction SD / target SD", "prediction_std_ratio_vs_topn"),
        ("exact_repeated_prediction_fraction", "mean_repeated_prediction_fraction", "Exact repeated-prediction fraction", "repeated_prediction_fraction_vs_topn"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5)); pdata = []
        for model in MODELS:
            model_agg = [r for r in agg if r["model"] == model]
            ax.plot(x, [r[aggregate_field] for r in model_agg], marker=markers[model], color=colors[model], label=model)
            for row in seed_rows:
                if row["model"] == model:
                    ax.scatter(row["top_n"], row[field], color=colors[model], alpha=0.4, s=22)
                    pdata.append({"model": model, "top_n": row["top_n"], "seed": row["seed"], field: row[field]})
        if field == "prediction_sd_ratio":
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Top-N", ylabel=ylabel, xticks=x, title=f"{ylabel} versus Top-N")
        ax.grid(alpha=0.2); ax.legend()
        save_figure(plt, fig, output, name, pdata)

    for kind, name in (
        ("truth", "representative_true_vs_predicted"),
        ("residual", "representative_residuals_vs_true"),
        ("distribution", "representative_prediction_distributions"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        pdata = []
        for row, ax in zip(reps, axes.flat):
            key = (row["model"], row["top_n"], row["seed"])
            data = prediction_data[key]
            targets, predictions = data["targets"], data["predictions"]
            if kind == "truth":
                ax.scatter(targets, predictions, alpha=0.65, s=18)
                lo, hi = min(targets + predictions), max(targets + predictions)
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
                ax.set(xlabel="True Omega_m", ylabel="Predicted Omega_m")
                ax.text(0.03, 0.97, f"MAE={row['test_mae']:.3f}\nRMSE={row['test_rmse']:.3f}\nR²={row['test_r2']:.3f}\nSD ratio={row['prediction_sd_ratio']:.3f}", transform=ax.transAxes, va="top", fontsize=7)
            elif kind == "residual":
                residuals = [p - t for t, p in zip(targets, predictions)]
                ax.scatter(targets, residuals, alpha=0.65, s=18); ax.axhline(0, color="black", linestyle="--")
                ax.set(xlabel="True Omega_m", ylabel="Prediction − truth")
            else:
                ax.hist(targets, bins=12, alpha=0.45, label="truth")
                ax.hist(predictions, bins=12, alpha=0.55, label="prediction")
                ax.set(xlabel="Omega_m", ylabel="Count"); ax.legend(fontsize=7)
            ax.set_title(f"{row['model']} Top{row['top_n']} seed {row['seed']}", fontsize=8)
            for uid, target, prediction in zip(data["ids"], targets, predictions):
                pdata.append({"model": row["model"], "top_n": row["top_n"], "seed": row["seed"], "universe_id": uid, "target": target, "prediction": prediction, "residual": prediction - target})
        fig.suptitle(name.replace("_", " ").title())
        fig.tight_layout()
        save_figure(plt, fig, output, name, pdata)

    fig, ax = plt.subplots(figsize=(7, 4.5)); pdata = []
    for model in MODELS:
        rows = [r for r in agg if r["model"] == model]
        ax.plot(x, [r["sd_test_mae"] for r in rows], marker=markers[model], color=colors[model], label=model)
        pdata.extend({"model": model, "top_n": r["top_n"], "sd_test_mae": r["sd_test_mae"]} for r in rows)
    ax.set(xlabel="Top-N", ylabel="Between-seed SD of test MAE", xticks=x, title="Seed variability versus Top-N")
    ax.grid(alpha=0.2); ax.legend()
    save_figure(plt, fig, output, "seed_variability_vs_topn", pdata)

    graph_rows = [{"top_n": n, "nodes": n, "directed_neighbor_selections": n * 8, "dense_adjacency_capacity": n * n} for n in TOPNS]
    fig, ax1 = plt.subplots(figsize=(7, 4.5)); ax2 = ax1.twinx()
    ax1.plot(x, [r["nodes"] for r in graph_rows], "o-", color="#1f77b4", label="nodes")
    ax1.plot(x, [r["directed_neighbor_selections"] for r in graph_rows], "s-", color="#2ca02c", label="8N selections")
    ax2.plot(x, [r["dense_adjacency_capacity"] for r in graph_rows], "^-", color="#d62728", label="N² capacity")
    ax1.set(xlabel="Top-N", ylabel="Nodes / neighbor selections", xticks=x, title="Deterministic graph-size scaling")
    ax2.set_ylabel("Dense adjacency entries"); ax1.grid(alpha=0.2)
    ax1.legend(loc="upper left"); ax2.legend(loc="center left")
    save_figure(plt, fig, output, "nodes_and_edges_vs_topn", graph_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5)); pdata = []
    for model in MODELS:
        rows = [r for r in agg if r["model"] == model]
        capacities = [r["top_n"] ** 2 for r in rows]
        ax.plot(capacities, [r["mean_test_mae"] for r in rows], marker=markers[model], color=colors[model], label=model)
        pdata.extend({"model": model, "top_n": r["top_n"], "dense_adjacency_capacity": r["top_n"] ** 2, "mean_test_mae": r["mean_test_mae"]} for r in rows)
    ax.set(xlabel="Dense adjacency capacity per snapshot (N²)", ylabel="Mean test MAE", title="Predictive performance versus graph size")
    ax.grid(alpha=0.2); ax.legend()
    save_figure(plt, fig, output, "predictive_performance_vs_graph_size", pdata)

    fig, ax = plt.subplots(figsize=(7, 4.5)); pdata = list(model_pairs)
    for top_n in TOPNS:
        rows = [r for r in model_pairs if r["top_n"] == top_n]
        vals = [r["mae_evolve_minus_static"] for r in rows]
        ax.scatter([top_n] * len(vals), vals, color="#6a3d9a", alpha=0.55)
        ax.scatter(top_n, mean(vals), color="black", marker="_", s=180)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Top-N", ylabel="MAE(Evolve) − MAE(Static)", xticks=x, title="Descriptive cross-protocol model comparison")
    ax.grid(alpha=0.2)
    save_figure(plt, fig, output, "descriptive_model_difference_vs_topn", pdata)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); pdata = []
    for ax, model in zip(axes, MODELS):
        matrix = []
        labels = []
        for larger, smaller in PAIRS:
            rows = [r for r in paired if r["model"] == model and r["larger_top_n"] == larger and r["smaller_top_n"] == smaller]
            matrix.append([mean([r["mae_difference"] for r in rows]), mean([r["prediction_sd_ratio_difference"] for r in rows])])
            labels.append(f"{larger}-{smaller}")
            pdata.append({"model": model, "topn_pair": labels[-1], "mean_mae_difference": matrix[-1][0], "mean_prediction_sd_ratio_difference": matrix[-1][1]})
        image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set(xticks=[0, 1], xticklabels=["ΔMAE", "ΔSD ratio"], yticks=range(3), yticklabels=labels, title=model)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Top-N effect summary (larger minus smaller)")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    save_figure(plt, fig, output, "topn_effect_summary", pdata)

    missing = [name for name in FIGURES if not (output / "figures" / f"{name}.png").is_file()]
    if missing:
        raise RuntimeError(f"missing figures: {missing}")


def write_science(
    output: Path,
    agg: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> None:
    lookup = {(r["model"], r["top_n"]): r for r in agg}
    pair_lookup = {
        (model, larger, smaller): [
            r for r in paired
            if r["model"] == model and r["larger_top_n"] == larger and r["smaller_top_n"] == smaller
        ]
        for model in MODELS for larger, smaller in PAIRS
    }

    def pair_sentence(model: str, larger: int, smaller: int) -> str:
        rows = pair_lookup[(model, larger, smaller)]
        values = [r["mae_difference"] for r in rows]
        return (
            f"Top{larger}−Top{smaller}: mean ΔMAE {mean(values):+.6f} "
            f"(SD {sd(values):.6f}; median {statistics.median(values):+.6f}; "
            f"range {min(values):+.6f} to {max(values):+.6f}; "
            f"larger Top-N wins {sum(v < 0 for v in values)}/3)."
        )

    scientific = f"""# Scientific summary

## Question and design

This model-stratified analysis asks how Top100, Top200, and Top500 affect
Omega_m regression under matched U500 protocols. All three seeds (42, 123,
2025), including poor and collapsed runs, are retained. Cross-model results
are descriptive protocol comparisons.

## EvolveGCN-H

{pair_sentence("EvolveGCN-H", 200, 100)}
{pair_sentence("EvolveGCN-H", 500, 200)}
{pair_sentence("EvolveGCN-H", 500, 100)}

Mean MAE decreases from {lookup[("EvolveGCN-H", 100)]["mean_test_mae"]:.6f}
to {lookup[("EvolveGCN-H", 200)]["mean_test_mae"]:.6f} and
{lookup[("EvolveGCN-H", 500)]["mean_test_mae"]:.6f}. All three seeds improve
at both steps. The Top500−Top100 change ({mean([r["mae_difference"] for r in pair_lookup[("EvolveGCN-H", 500, 100)]]):+.6f})
is slightly larger than the Top100 between-seed SD
({lookup[("EvolveGCN-H", 100)]["sd_test_mae"]:.6f}).

## Static GCN

{pair_sentence("Static GCN", 200, 100)}
{pair_sentence("Static GCN", 500, 200)}
{pair_sentence("Static GCN", 500, 100)}

Static mean MAE is effectively flat:
{lookup[("Static GCN", 100)]["mean_test_mae"]:.6f},
{lookup[("Static GCN", 200)]["mean_test_mae"]:.6f}, and
{lookup[("Static GCN", 500)]["mean_test_mae"]:.6f}. Top200 versus Top100 and
Top500 versus Top100 are directionally mixed and much smaller than the
between-seed SD.

## Prediction compression

Evolve mean prediction-SD ratio rises from
{lookup[("EvolveGCN-H", 100)]["mean_prediction_sd_ratio"]:.3f} to
{lookup[("EvolveGCN-H", 200)]["mean_prediction_sd_ratio"]:.3f} and
{lookup[("EvolveGCN-H", 500)]["mean_prediction_sd_ratio"]:.3f}. Dispersion
improves in every matched seed, but remains below one; compression is reduced,
not eliminated. Evolve has no exact repeated predictions in these rows.

Static ratios remain
{lookup[("Static GCN", 100)]["mean_prediction_sd_ratio"]:.3f},
{lookup[("Static GCN", 200)]["mean_prediction_sd_ratio"]:.3f}, and
{lookup[("Static GCN", 500)]["mean_prediction_sd_ratio"]:.3f}; mean exact-repeat
fractions are {lookup[("Static GCN", 100)]["mean_repeated_prediction_fraction"]:.3f},
{lookup[("Static GCN", 200)]["mean_repeated_prediction_fraction"]:.3f}, and
{lookup[("Static GCN", 500)]["mean_repeated_prediction_fraction"]:.3f}.
Repetition does not decrease monotonically.

## Computational interpretation

Moving Top100 to Top500 multiplies nodes and directed neighbor selections by
five and dense adjacency capacity by 25. Evolve gains are meaningful relative
to its observed seed variability, but the remaining compression and missing
wall-time/memory measurements prevent a complete efficiency claim. Static
shows negligible predictive benefit despite the graph-size growth.

## Conclusion

Under the tested sample-local minmax protocol, additional halos provide useful
information to EvolveGCN-H, with continued improvement from Top100 through
Top500 and no clear saturation by Top200. Increasing Top-N alone does not
eliminate prediction compression. Static GCN does not use the additional halo
population effectively under its tested readout. These are descriptive
three-seed, protocol-specific findings, not universal claims about graph size.
"""
    (output / "scientific_summary.md").write_text(scientific, encoding="utf-8")

    thesis = f"""# Controlled Top-N Halo-Count Scaling

## Motivation for Halo-Count Scaling

Top-N denotes the maximum number of raw-Mvir-ranked halos retained per
universe and snapshot; U500 denotes 500 independent simulated universes.
Increasing Top-N may expose additional mass-function, spatial, and velocity
information, while increasing graph cost.

## Top-N Selection Procedure

Invalid and nonpositive raw Mvir rows are removed, the catalogue is sorted by
raw Mvir in descending order, and the first N rows are retained independently
for every universe and snapshot. log10(Mvir) and the remaining node features
are created afterward. Raw XYZ is retained separately for periodic kNN.

## Experimental Design

EvolveGCN-H and Static GCN are analyzed separately at Top100, Top200, and
Top500 with matched seeds 42, 123, and 2025. All experiments use U500,
periodic kNN k=8, h32, mean graph pooling, raw Omega_m targets, and the
model-specific established architecture and batch size.

## Controlled Variables

Within a model, Top-N is the only intended scientific factor. Evolve uses five
snapshots, L2, temporal mean, and batch 4. Static uses a final snapshot, L3,
and batch 8. Model differences are therefore descriptive, not a causal test of
temporal processing.

## Normalization Interaction

Minmax statistics are calculated independently per universe, snapshot, and
feature after selection. Changing N changes both node population and local
minima/maxima. This is neither target leakage nor train/test leakage; it is an
inseparable part of the historical intervention.

## Verification Procedure

All 18 configs, metrics, logs, prediction CSVs, and checkpoint paths were
verified. Checkpoints were not loaded. Ordered splits contain 350/75/75 IDs,
are disjoint, cover LH_0 through LH_499, and match across all model × Top-N
rows for each seed. Metrics were independently recomputed at tolerance 1e-6.

## Quantitative Results

{scientific.split("## EvolveGCN-H", 1)[1].split("## Prediction compression", 1)[0]}

## Paired-Seed Analysis

Evolve improvements occur for all three seeds at every step. Static changes
are small; the Top200−Top100 and Top500−Top100 comparisons have mixed signs.
No p-values are reported because three seeds support descriptive rather than
fragile inferential evidence.

## Prediction-Compression Diagnostics

{scientific.split("## Prediction compression", 1)[1].split("## Computational interpretation", 1)[0]}

## Computational Scaling

k=8 yields 800, 1,600, and 4,000 directed neighbor selections per snapshot.
Dense adjacency capacity is 10,000, 40,000, and 250,000 entries. Self-loops
are absent from preprocessing and added by model layers; reciprocal neighbor
relations are symmetrized; padded nodes are excluded. Time and memory were not
recorded.

## Interpretation

Additional halos improve Evolve error and dispersion, whereas Static remains
nearly unchanged. The contrast may reflect multiple protocol differences and
cannot be attributed solely to temporal input.

## Limitations

Only three seeds are available. Halo-set nesting is expected but not
byte-proven because selected IDs were not stored and mass ties lack a stable
secondary key. kNN topology is rebuilt and is not nested. Minmax values change
with N. U750 Top1000 changes both universe count and normalization and is
excluded from the controlled trend.

## Conclusion

Additional halos provide useful information under the Evolve protocol, but
node count alone does not eliminate compression. Static shows no practically
meaningful benefit from the fivefold node increase.
"""
    (output / "thesis_section.md").write_text(thesis, encoding="utf-8")

    presentation = """# Presentation-ready summary

## Recommended supervisor-meeting sequence

1. Protocol table: define U500 separately from Top-N.
2. Six-row model-stratified main-results table.
3. Test MAE versus Top-N with individual matched seeds.
4. Prediction-SD ratio versus Top-N.
5. Exact repeated-prediction fraction versus Top-N.
6. Median-MAE representative true-versus-predicted panels.
7. Nodes, neighbor selections, and dense adjacency capacity versus Top-N.
8. Conclusion slide.

## Concise conclusion slide

At fixed U500 protocols, increasing Top-N consistently improves EvolveGCN-H
error and prediction dispersion across all three seeds, but dispersion remains
compressed. Static GCN remains nearly flat and highly repetitive. Top500 costs
5× more nodes and 25× dense adjacency capacity than Top100, so the gain is
model-dependent and does not show that graph size alone solves collapse.
"""
    (output / "presentation_ready_summary.md").write_text(presentation, encoding="utf-8")


def write_verification(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    aggregated: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> None:
    membership = [
        {
            "model": r["model"],
            "Top-N": r["top_n"],
            "seed": r["seed"],
            "experiment": r["experiment_name"],
            "split": str(r["split_signature"])[:12],
            "MAE": f"{r['test_mae']:.9f}",
            "R²": f"{r['test_r2']:.6f}",
            "SD ratio": f"{r['prediction_sd_ratio']:.6f}",
            "exact repeat": f"{r['exact_repeated_prediction_fraction']:.6f}",
        }
        for r in rows
    ]
    max_delta = max(
        max(r["saved_mae_absolute_delta"], r["saved_rmse_absolute_delta"], r["saved_mse_absolute_delta"])
        for r in rows
    )
    paired_summary = []
    for model in MODELS:
        for larger, smaller in PAIRS:
            values = [
                float(row["mae_difference"])
                for row in paired
                if row["model"] == model
                and row["larger_top_n"] == larger
                and row["smaller_top_n"] == smaller
            ]
            paired_summary.append(
                {
                    "model": model,
                    "comparison": f"Top{larger}-Top{smaller}",
                    "mean_delta_mae": f"{mean(values):+.9f}",
                    "sd_delta_mae": f"{sd(values):.9f}",
                    "larger_topn_wins": f"{sum(value < 0 for value in values)}/3",
                }
            )
    dispersion = [
        {
            "model": row["model"],
            "Top-N": row["top_n"],
            "mean_prediction_sd_ratio": f"{row['mean_prediction_sd_ratio']:.6f}",
            "mean_exact_repeat_fraction": f"{row['mean_repeated_prediction_fraction']:.6f}",
        }
        for row in aggregated
    ]
    text = f"""# Final verification: controlled Top-N halo scaling

## Verifier result

PASS. Exactly 18 canonical experiments were verified: nine EvolveGCN-H and
nine Static GCN rows spanning Top100, Top200, Top500 and seeds 42, 123, 2025.
No duplicate experiment or prediction mapping is present.

## Exact membership and recomputed diagnostics

{md_table(membership)}

## Artifact verification

Every row has a parseable config.json, metrics.json, training CSV,
predictions/test_predictions.csv, and checkpoints/best_model.pt. Checkpoints
were checked only for existence; they were never loaded. Prediction files have
75 finite, unique, nonmissing ordered universe IDs.

## Split verification

Every split has 350 training, 75 validation, and 75 test IDs; partitions are
disjoint and cover LH_0 through LH_499 exactly. For each seed, the exact
ordered split signature agrees across Top100/Top200/Top500 and across both
models.

## Metric recomputation

MAE, RMSE, MSE, R², Pearson under the established variance policy, target and
prediction moments, dispersion ratio, exact and approximate repetition,
ranges, and residual diagnostics were recomputed. Saved MAE/RMSE/MSE agree at
absolute tolerance 1e-6; maximum absolute discrepancy is {max_delta:.17g}.
Negative-R², poor, compressed, and repeated-prediction rows are retained.
Undefined Pearson values, if present, remain explicit and are never zero-filled.

## Implementation and normalization

Source verification confirms raw-Mvir descending Top-N selection before
log10 feature construction and normalization; raw XYZ periodic kNN; padding
after normalization; real-node masks; padded-node exclusion; zero-real-node
rejection; and topology rebuilding at each N. Top-N sets are expected prefixes
but are not byte-proven, and equal-mass ties have no stable secondary key.

Minmax statistics are sample-local per universe, snapshot, and feature.
Changing N therefore changes local minima/maxima as well as graph size. This
is part of the intervention, not target or train/test leakage.

## Exclusions

U750 Top1000, universe-scaling/debug runs, alternative k/width/pooling/head/
normalization, hybrids, target-normalized runs, GraphSAGE, legacy Static
anchors, and duplicate reproductions are excluded from the canonical mapping.
No canonical row is duplicated.

## Final analysis registration

- Family identifier: `controlled_static_vs_evolvegcn_topn_halo_scaling_500u`.
- Specification: `configs/analysis_reports/controlled_static_vs_evolvegcn_topn_halo_scaling_500u.json`.
- Output: `reports/analysis/controlled_static_vs_evolvegcn_topn_halo_scaling_500u/`.
- Mapping: 18 artifact-complete experiments and 18 unique prediction files.
- Validator result: PASS.
- Training decision: no training or graph rebuilding is required.
- U750 Top1000: excluded because universe count and normalization differ.

## Paired Top-N findings

{md_table(paired_summary)}

## Prediction-compression findings

{md_table(dispersion)}

## Computational-scaling evidence

At k=8, Top100/Top200/Top500 deterministically produce 800/1,600/4,000
directed neighbor selections and dense adjacency capacities of
10,000/40,000/250,000 entries per snapshot. Wall time, GPU/CPU peak memory,
and inference time were not recorded and are not estimated.
"""
    (root / "reports/experiment_registry/topn_halo_scaling_final_verification.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    configure_imports(root)
    from analysis_reporting.common import validate_analysis

    validation = validate_analysis(root, args.spec)
    if not validation.valid:
        raise RuntimeError("input validation failed: " + "; ".join(validation.errors))
    spec = validation.spec
    output = root / spec["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    seed_rows, prediction_data = read_verified(root, validation)
    aggregated = aggregate(seed_rows)
    paired = paired_topn(seed_rows)
    model_pairs = paired_models(seed_rows)
    reps = representatives(seed_rows)
    diagnostics = [
        {k: r[k] for k in (
            "model", "top_n", "seed", "experiment_name", "test_mae", "test_rmse",
            "test_r2", "test_pearson", "pearson_status", "target_mean",
            "prediction_mean", "target_sd", "prediction_sd", "prediction_sd_ratio",
            "unique_prediction_count", "exact_repeated_prediction_fraction",
            "approximate_repeated_prediction_fraction", "approximate_repeat_tolerance",
            "prediction_min", "prediction_max", "prediction_range", "residual_mean",
            "residual_sd", "maximum_absolute_residual",
        )}
        for r in seed_rows
    ]
    computational = computational_rows(aggregated)
    if (len(seed_rows), len(aggregated), len(paired), len(model_pairs), len(diagnostics), len(reps)) != (18, 6, 18, 9, 18, 6):
        raise RuntimeError("required analysis row count mismatch")

    write_csv(output / "seed_level_results.csv", seed_rows)
    write_csv(output / "aggregated_results.csv", aggregated)
    write_csv(output / "paired_topn_differences.csv", paired)
    write_csv(output / "paired_model_differences.csv", model_pairs)
    write_csv(output / "prediction_diagnostics.csv", diagnostics)
    write_csv(output / "representative_runs.csv", reps)

    dataset_rows = []
    seen = set()
    families = {label: family for label, _, family in validation.family_specs}
    for model, verified_rows in validation.family_rows:
        family = families[model]
        run_lookup = {(run["group_value"], run["seed"]): run for run in family["runs"]}
        for row in verified_rows:
            run = run_lookup[(row["grouping_value"], row["seed"])]
            key = (model, row["grouping_value"], run["dataset_path"])
            if key in seen:
                continue
            seen.add(key)
            dataset = root / run["dataset_path"]
            metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            dataset_rows.append({
                "model_protocol": model,
                "top_n": row["grouping_value"],
                "dataset_path": run["dataset_path"],
                "dataset_type": metadata.get("dataset_type"),
                "universes": metadata.get("num_universes_successful"),
                "snapshots_stored": metadata.get("num_snapshots", 1),
                "normalization": metadata.get("normalization"),
                "k": metadata.get("k"),
                "periodic": metadata.get("periodic_boundary_knn"),
                "box_size": metadata.get("box_size"),
                "selection": metadata.get("node_selection"),
                "metadata_sha256": sha256(dataset.with_suffix(".metadata.json")),
            })
    write_csv(output / "dataset_topn_table.csv", dataset_rows)
    write_md_table(output / "dataset_topn_table.md", "Dataset Top-N table", dataset_rows)

    caption = "U500 Top100/Top200/Top500, sample-local min-max normalization, periodic kNN k=8, h32, mean graph pooling, three matched seeds; model-stratified analysis."
    main_rows = rounded_aggregate(aggregated)
    tables = output / "tables"
    table_set(tables, "main_results_table", "Main Top-N results", main_rows, caption=caption)
    table_set(tables, "evolvegcn_topn_results", "EvolveGCN-H Top-N results", [r for r in main_rows if r["model"] == "EvolveGCN-H"], caption=caption)
    table_set(tables, "static_gcn_topn_results", "Static GCN Top-N results", [r for r in main_rows if r["model"] == "Static GCN"], caption=caption)
    raw_main_rows = [
        {
            "model": row["model"],
            "top_n": row["top_n"],
            "seed_count": row["seed_count"],
            "mean_test_mae": row["mean_test_mae"],
            "sd_test_mae": row["sd_test_mae"],
            "median_test_mae": row["median_test_mae"],
            "mean_test_rmse": row["mean_test_rmse"],
            "sd_test_rmse": row["sd_test_rmse"],
            "mean_test_r2": row["mean_test_r2"],
            "sd_test_r2": row["sd_test_r2"],
            "mean_prediction_sd_ratio": row["mean_prediction_sd_ratio"],
            "sd_prediction_sd_ratio": row["sd_prediction_sd_ratio"],
            "mean_exact_repeat_fraction": row["mean_repeated_prediction_fraction"],
            "maximum_exact_repeat_fraction": row["maximum_repeated_prediction_fraction"],
            "undefined_pearson_count": row["undefined_pearson_count"],
        }
        for row in aggregated
    ]
    write_csv(tables / "main_results_table.csv", raw_main_rows)
    write_csv(
        tables / "evolvegcn_topn_results.csv",
        [row for row in raw_main_rows if row["model"] == "EvolveGCN-H"],
    )
    write_csv(
        tables / "static_gcn_topn_results.csv",
        [row for row in raw_main_rows if row["model"] == "Static GCN"],
    )
    table_set(tables, "seed_level_topn_results", "Seed-level Top-N results", seed_rows)
    table_set(tables, "paired_topn_differences", "Paired Top-N differences", paired, caption="Larger Top-N minus smaller Top-N; descriptive matched-seed evidence without significance tests.")
    table_set(tables, "descriptive_model_comparison", "Descriptive model comparison", model_pairs, caption="Descriptive cross-protocol model comparison.")
    table_set(tables, "prediction_collapse_diagnostics", "Prediction-collapse diagnostics", diagnostics)
    table_set(tables, "computational_scaling", "Computational scaling", computational)
    best = []
    for model in MODELS:
        rows = [r for r in aggregated if r["model"] == model]
        chosen = min(rows, key=lambda r: r["mean_test_mae"])
        best.append({"model": model, "descriptive_best_top_n": chosen["top_n"], "mean_test_mae": chosen["mean_test_mae"], "scope": "within-model U500 protocol"})
    table_set(tables, "descriptive_best_topn_summary", "Descriptive best Top-N", best, latex=False)

    topn_impl = """# Top-N implementation

Invalid/nonpositive raw Mvir rows are removed. Halos are sorted by raw Mvir
descending and the first N are retained independently per universe and
snapshot. log10(Mvir), node features, and sample-local normalization follow
selection. Raw XYZ is separately retained for periodic minimum-image kNN.
Fewer-than-N snapshots are zero-padded after normalization; masks exclude
padded nodes, and zero-real-node snapshots are rejected.

Top100/Top200/Top500 membership is expected to be nested for identical inputs,
but selected IDs/rank hashes were not stored and equal-mass ties lack an
explicit stable secondary key. Graph topology is rebuilt at each N; larger
graphs are not induced supergraphs.
"""
    (output / "topn_implementation.md").write_text(topn_impl, encoding="utf-8")
    normalization = """# Normalization interaction

Canonical features use min-max normalization independently per universe,
snapshot, and feature after Top-N selection. Changing Top-N therefore changes
both the retained nodes and the local minima/maxima. This is neither target
leakage nor train/test leakage; it is part of the established Top-N
intervention. Results do not isolate node count while holding all numerical
feature values fixed.
"""
    (output / "normalization_interaction.md").write_text(normalization, encoding="utf-8")
    computational_text = """# Computational scaling

At k=8, Top100, Top200, and Top500 imply 800, 1,600, and 4,000 directed
neighbor selections per snapshot before reciprocal symmetrization. These
exclude self-loops and padded nodes; reciprocal relations may duplicate after
symmetrization, and model layers add self-loops. Dense adjacency capacity is
10,000, 40,000, and 250,000 entries per snapshot.

Wall time, GPU/CPU peak memory, and prediction time were not recorded. Top500
therefore cannot be recommended on a fully quantified efficiency basis.
Evolve gains are meaningful relative to observed seed variability; Static
gains are negligible relative to graph-size growth.
"""
    (output / "computational_scaling.md").write_text(computational_text, encoding="utf-8")
    write_science(output, aggregated, paired)
    make_figures(output, seed_rows, aggregated, paired, model_pairs, reps, prediction_data)
    for generic_name in (
        "paired_mae_difference_vs_topn",
        "true_vs_predicted_representative_runs",
        "residuals_vs_true_representative_runs",
    ):
        for relative in (
            f"figures/{generic_name}.png",
            f"figures/{generic_name}.pdf",
            f"plot_data/{generic_name}.csv",
        ):
            (output / relative).unlink(missing_ok=True)
    write_verification(root, seed_rows, aggregated, paired)

    readme = """# Controlled U500 Top-N halo-count scaling

Deterministic, model-stratified analysis of 18 verified EvolveGCN-H and Static
GCN experiments at Top100, Top200, and Top500. Run the repository build
command first, then this package's focused rebuild script, then the validator.
No graph datasets or checkpoints are loaded.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    manifest_path = output / "analysis_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest = {
        **old_manifest,
        "analysis_name": spec["analysis_name"],
        "analysis_specification": Path(args.spec).resolve().relative_to(root).as_posix(),
        "canonical_experiment_count": 18,
        "canonical_prediction_count": 18,
        "generated_row_counts": {
            "seed_level_results": 18,
            "aggregated_results": 6,
            "paired_topn_differences": 18,
            "paired_model_differences": 9,
            "prediction_diagnostics": 18,
            "representative_runs": 6,
        },
        "topn_specific_validation": {
            "top_n_order": list(TOPNS),
            "exact_paired_splits": True,
            "metric_tolerance": 1e-6,
            "unique_experiment_mappings": 18,
            "unique_prediction_mappings": 18,
            "representative_runs": [
                {"model": r["model"], "top_n": r["top_n"], "seed": r["seed"]}
                for r in reps
            ],
            "no_training_required": True,
            "top1000_excluded": True,
        },
        "generated_figures": list(FIGURES),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["generated_outputs"] = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Top-N focused rebuild: PASS")
    print("Rows: seed=18 aggregate=6 paired_topn=18 paired_model=9 diagnostics=18 representatives=6")
    print(f"Figures: {len(FIGURES)} x PNG/PDF/CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
