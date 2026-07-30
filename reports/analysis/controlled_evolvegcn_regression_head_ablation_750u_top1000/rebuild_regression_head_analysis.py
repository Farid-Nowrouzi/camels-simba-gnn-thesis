#!/usr/bin/env python3
"""Rebuild the controlled EvolveGCN-H regression-head analysis package."""

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

ROOT_DEFAULT = Path(__file__).resolve().parents[3]
HEADS = ("linear", "mlp")
SEEDS = (42, 123, 777, 999, 2025)
TOLERANCE = 1e-6
PARAMETERS = {"linear": 33, "mlp": 1089}
FIGURES = (
    "test_mae_by_regression_head",
    "test_rmse_by_regression_head",
    "test_r2_by_regression_head",
    "paired_mae_difference_by_seed",
    "paired_rmse_difference_by_seed",
    "paired_r2_difference_by_seed",
    "per_seed_mae_trajectories",
    "prediction_std_ratio_by_head",
    "repeated_prediction_fraction_by_head",
    "representative_true_vs_predicted",
    "representative_residuals_vs_true",
    "representative_prediction_distributions",
    "seed_variability_by_head",
    "head_parameter_count_vs_mae",
    "controlled_progression",
    "head_effect_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate verified regression-head tables and figures."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(
            "configs/analysis_reports/"
            "controlled_evolvegcn_regression_head_ablation_750u_top1000.json"
        ),
    )
    return parser.parse_args()


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
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
    ):
        text = text.replace(old, new)
    return text


def table_set(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    title: str,
    *,
    latex: bool = True,
) -> None:
    fields = list(rows[0])
    write_csv(directory / f"{name}.csv", rows)
    (directory / f"{name}.md").write_text(
        f"# {title}\n\n" + md_table(rows, fields), encoding="utf-8"
    )
    if latex:
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{" + latex_escape(title) + "}",
            "\\resizebox{\\textwidth}{!}{%",
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
        lines.extend(["\\hline", "\\end{tabular}}", "\\end{table}", ""])
        (directory / f"{name}.tex").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def read_inputs(root: Path, validation: Any) -> tuple[list[dict], dict]:
    from experiment_pipeline.common import read_prediction_rows, recompute_metrics

    if len(validation.family_rows) != 1:
        raise RuntimeError("expected one verified family")
    label, verified = validation.family_rows[0]
    family = json.loads(
        (root / validation.spec["families"][0]["family_spec_path"]).read_text()
    )
    records: list[dict] = []
    predictions: dict[tuple[str, int], dict] = {}
    for item in verified:
        head = str(item["grouping_value"])
        seed = int(item["seed"])
        experiment = root / str(item["experiment_path"])
        prediction_path = experiment / family["prediction_file"]
        ids, targets, estimates = read_prediction_rows(
            prediction_path,
            family["target_column_aliases"],
            family["prediction_column_aliases"],
            family["id_column_aliases"],
        )
        metrics = recompute_metrics(targets, estimates)
        config = json.loads((experiment / "config.json").read_text())
        saved = json.loads((experiment / "metrics.json").read_text())
        primary = saved["test"]
        deltas = {
            "saved_mae_absolute_delta": abs(
                float(primary["mae"]) - metrics["test_mae"]
            ),
            "saved_rmse_absolute_delta": abs(
                float(primary["rmse"]) - metrics["test_rmse"]
            ),
            "saved_mse_absolute_delta": abs(
                float(primary["mse"]) - metrics["test_mse"]
            ),
        }
        if max(deltas.values()) > TOLERANCE:
            raise RuntimeError(f"saved metric mismatch: {experiment.name}")
        train_ids, val_ids, test_ids = (
            list(config["train_ids"]),
            list(config["val_ids"]),
            list(config["test_ids"]),
        )
        all_ids = train_ids + val_ids + test_ids
        if (
            (len(train_ids), len(val_ids), len(test_ids)) != (450, 99, 201)
            or set(train_ids) & set(val_ids)
            or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids)
            or len(set(all_ids)) != 750
            or set(all_ids) != {f"LH_{index}" for index in range(750)}
            or ids != test_ids
            or len(ids) != 201
            or len(ids) != len(set(ids))
        ):
            raise RuntimeError(f"split verification failed: {experiment.name}")
        residuals = [prediction - target for target, prediction in zip(targets, estimates)]
        record = {
            "model": label,
            "head_type": head,
            "head_parameter_count": PARAMETERS[head],
            "seed": seed,
            "experiment_name": item["experiment_name"],
            "experiment_path": item["experiment_path"],
            "prediction_path": prediction_path.relative_to(root).as_posix(),
            "split_signature": item["split_signature"],
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
            "train_mae": saved["train"]["mae"],
            "validation_mae": saved["val"]["mae"],
            "train_mse": saved["train"]["mse"],
            "validation_mse": saved["val"]["mse"],
            "best_epoch": saved.get("best_epoch", ""),
            **deltas,
            "prediction_sha256": sha256(prediction_path),
        }
        records.append(record)
        predictions[(head, seed)] = {
            "ids": ids, "targets": targets, "predictions": estimates
        }
    order = {value: index for index, value in enumerate(HEADS)}
    records.sort(key=lambda row: (order[row["head_type"]], SEEDS.index(row["seed"])))
    if len(records) != 10:
        raise RuntimeError("expected ten seed rows")
    for field in ("experiment_path", "prediction_path", "prediction_sha256"):
        if len({row[field] for row in records}) != 10:
            raise RuntimeError(f"canonical {field} values are not unique")
    return records, predictions


def aggregate(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows = []
    for head in HEADS:
        selected = [row for row in seed_rows if row["head_type"] == head]
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in selected]
        rows.append({
            "head_type": head,
            "seed_count": len(selected),
            "head_parameter_count": PARAMETERS[head],
            "mean_test_mae": statistics.mean(values("test_mae")),
            "sd_test_mae": sample_sd(values("test_mae")),
            "median_test_mae": statistics.median(values("test_mae")),
            "mean_test_rmse": statistics.mean(values("test_rmse")),
            "sd_test_rmse": sample_sd(values("test_rmse")),
            "mean_test_r2": statistics.mean(values("test_r2")),
            "sd_test_r2": sample_sd(values("test_r2")),
            "mean_prediction_sd_ratio": statistics.mean(values("prediction_sd_ratio")),
            "sd_prediction_sd_ratio": sample_sd(values("prediction_sd_ratio")),
            "mean_repeated_prediction_fraction": statistics.mean(
                values("exact_repeated_prediction_fraction")
            ),
            "maximum_repeated_prediction_fraction": max(
                values("exact_repeated_prediction_fraction")
            ),
            "undefined_pearson_count": sum(
                row["pearson_status"] != "defined" for row in selected
            ),
            "mean_train_mae": statistics.mean(values("train_mae")),
            "mean_validation_mae": statistics.mean(values("validation_mae")),
            "mean_validation_minus_train_mae": statistics.mean(
                float(row["validation_mae"]) - float(row["train_mae"])
                for row in selected
            ),
        })
    return rows


def pair(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    lookup = {(row["head_type"], int(row["seed"])): row for row in seed_rows}
    rows = []
    for seed in SEEDS:
        linear, mlp = lookup[("linear", seed)], lookup[("mlp", seed)]
        if linear["split_signature"] != mlp["split_signature"]:
            raise RuntimeError(f"paired split mismatch: {seed}")
        rows.append({
            "seed": seed,
            "split_signature": linear["split_signature"],
            "linear_experiment_name": linear["experiment_name"],
            "mlp_experiment_name": mlp["experiment_name"],
            "mae_linear_minus_mlp": linear["test_mae"] - mlp["test_mae"],
            "rmse_linear_minus_mlp": linear["test_rmse"] - mlp["test_rmse"],
            "r2_linear_minus_mlp": linear["test_r2"] - mlp["test_r2"],
            "prediction_sd_ratio_linear_minus_mlp":
                linear["prediction_sd_ratio"] - mlp["prediction_sd_ratio"],
            "repeat_fraction_linear_minus_mlp":
                linear["exact_repeated_prediction_fraction"]
                - mlp["exact_repeated_prediction_fraction"],
        })
    for field in (
        "mae_linear_minus_mlp", "rmse_linear_minus_mlp", "r2_linear_minus_mlp",
        "prediction_sd_ratio_linear_minus_mlp",
        "repeat_fraction_linear_minus_mlp",
    ):
        values = [float(row[field]) for row in rows]
        prefix = field.removesuffix("_linear_minus_mlp")
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
        for row in rows:
            row.update(summary)
    for row in rows:
        row["mae_seeds_favouring_linear"] = row["mae_negative_count"]
        row["mae_seeds_favouring_mlp"] = row["mae_positive_count"]
        row["mae_ties"] = row["mae_zero_count"]
    return rows


def representatives(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    output = []
    for head in HEADS:
        rows = sorted(
            (row for row in seed_rows if row["head_type"] == head),
            key=lambda row: (float(row["test_mae"]), int(row["seed"])),
        )
        row = rows[len(rows) // 2]
        output.append({
            "model": row["model"],
            "head_type": head,
            "selection_policy": "median_test_mae",
            "seed": row["seed"],
            "experiment_name": row["experiment_name"],
            "experiment_path": row["experiment_path"],
            "prediction_path": row["prediction_path"],
            "test_mae": row["test_mae"],
            "test_rmse": row["test_rmse"],
            "test_r2": row["test_r2"],
            "prediction_sd_ratio": row["prediction_sd_ratio"],
        })
    return output


def save_plot(plt: Any, out: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(out / "plot_data" / f"{name}.csv", rows)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(out / "figures" / f"{name}.{suffix}", dpi=180)
    plt.close()


def plots(
    out: Path, seed_rows: list[dict], aggregate_rows: list[dict],
    paired_rows: list[dict], representative_rows: list[dict],
    predictions: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"linear": "#2b6cb0", "mlp": "#dd6b20"}
    for name, field, label in (
        ("test_mae_by_regression_head", "test_mae", "Test MAE"),
        ("test_rmse_by_regression_head", "test_rmse", "Test RMSE"),
        ("test_r2_by_regression_head", "test_r2", "Test R²"),
        ("prediction_std_ratio_by_head", "prediction_sd_ratio", "Prediction SD / target SD"),
        ("repeated_prediction_fraction_by_head", "exact_repeated_prediction_fraction", "Repeated-prediction fraction"),
    ):
        data = [{"head_type": r["head_type"], "seed": r["seed"], field: r[field]} for r in seed_rows]
        plt.figure(figsize=(6.4, 4.4))
        for index, head in enumerate(HEADS):
            values = [float(r[field]) for r in seed_rows if r["head_type"] == head]
            jitter = [-.08, -.04, 0, .04, .08]
            plt.scatter([index + j for j in jitter], values, color=colors[head], label=head)
            plt.errorbar(index, statistics.mean(values), yerr=sample_sd(values), fmt="D", color="black", capsize=5)
        plt.xticks(range(2), HEADS)
        plt.ylabel(label)
        plt.title(label + " by regression head")
        save_plot(plt, out, name, data)

    for name, field, label in (
        ("paired_mae_difference_by_seed", "mae_linear_minus_mlp", "MAE(linear) − MAE(MLP)"),
        ("paired_rmse_difference_by_seed", "rmse_linear_minus_mlp", "RMSE(linear) − RMSE(MLP)"),
        ("paired_r2_difference_by_seed", "r2_linear_minus_mlp", "R²(linear) − R²(MLP)"),
    ):
        data = [{"seed": r["seed"], field: r[field]} for r in paired_rows]
        plt.figure(figsize=(7, 4.4))
        values = [float(r[field]) for r in paired_rows]
        plt.bar([str(r["seed"]) for r in paired_rows], values, color=["#2b6cb0" if v < 0 else "#dd6b20" for v in values])
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Seed")
        plt.ylabel(label)
        plt.title(label + " by matched seed")
        save_plot(plt, out, name, data)

    trajectory_data = []
    plt.figure(figsize=(7, 4.8))
    lookup = {(r["head_type"], r["seed"]): r for r in seed_rows}
    for seed in SEEDS:
        values = [lookup[(head, seed)]["test_mae"] for head in HEADS]
        plt.plot(HEADS, values, marker="o", alpha=.75, label=str(seed))
        trajectory_data.extend(
            {"seed": seed, "head_type": head, "test_mae": value}
            for head, value in zip(HEADS, values)
        )
    plt.ylabel("Test MAE")
    plt.title("Per-seed MAE trajectories")
    plt.legend(title="Seed", ncol=3)
    save_plot(plt, out, "per_seed_mae_trajectories", trajectory_data)

    variability_data = [{
        "head_type": r["head_type"], "mean_test_mae": r["mean_test_mae"],
        "sd_test_mae": r["sd_test_mae"]
    } for r in aggregate_rows]
    plt.figure(figsize=(6.4, 4.4))
    plt.bar(
        HEADS, [r["sd_test_mae"] for r in aggregate_rows],
        color=[colors[h] for h in HEADS]
    )
    plt.ylabel("Between-seed SD of test MAE")
    plt.title("Seed variability by head")
    save_plot(plt, out, "seed_variability_by_head", variability_data)

    parameter_data = [{
        "head_type": r["head_type"], "head_parameter_count": r["head_parameter_count"],
        "mean_test_mae": r["mean_test_mae"], "sd_test_mae": r["sd_test_mae"]
    } for r in aggregate_rows]
    plt.figure(figsize=(6.4, 4.4))
    for row in parameter_data:
        plt.errorbar(row["head_parameter_count"], row["mean_test_mae"], yerr=row["sd_test_mae"], fmt="o", markersize=8, label=row["head_type"])
    plt.xscale("log")
    plt.xlabel("Head parameter count (log scale)")
    plt.ylabel("Mean test MAE")
    plt.title("Head capacity versus MAE")
    plt.legend()
    save_plot(plt, out, "head_parameter_count_vs_mae", parameter_data)

    progression = [
        {"stage":"Baseline","head_type":"mlp","mean_test_mae":aggregate_rows[1]["mean_test_mae"],"delta_from_mlp":0.0},
        {"stage":"Head ablation","head_type":"linear","mean_test_mae":aggregate_rows[0]["mean_test_mae"],"delta_from_mlp":aggregate_rows[0]["mean_test_mae"]-aggregate_rows[1]["mean_test_mae"]},
    ]
    plt.figure(figsize=(6.4, 4.4))
    plt.bar([r["stage"] for r in progression], [r["mean_test_mae"] for r in progression], color=[colors["mlp"], colors["linear"]])
    plt.ylabel("Mean test MAE")
    plt.title("Controlled head progression")
    save_plot(plt, out, "controlled_progression", progression)

    effect_data = []
    summary_fields = [
        ("MAE", "mae_linear_minus_mlp"),
        ("RMSE", "rmse_linear_minus_mlp"),
        ("R²", "r2_linear_minus_mlp"),
        ("SD ratio", "prediction_sd_ratio_linear_minus_mlp"),
    ]
    for metric, field in summary_fields:
        values = [float(row[field]) for row in paired_rows]
        effect_data.append({"metric":metric,"mean_difference":statistics.mean(values),"sd_difference":sample_sd(values)})
    plt.figure(figsize=(7, 4.5))
    plt.bar([r["metric"] for r in effect_data], [r["mean_difference"] for r in effect_data], yerr=[r["sd_difference"] for r in effect_data], capsize=4, color="#2b6cb0")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Linear − MLP")
    plt.title("Regression-head effect summary")
    save_plot(plt, out, "head_effect_summary", effect_data)

    rep_lookup = {r["head_type"]: r for r in representative_rows}
    true_pred_data, residual_data, distribution_data = [], [], []
    plt.figure(figsize=(6.4, 5.2))
    low, high = math.inf, -math.inf
    for head in HEADS:
        rep = rep_lookup[head]
        data = predictions[(head, int(rep["seed"]))]
        low = min(low, min(data["targets"]), min(data["predictions"]))
        high = max(high, max(data["targets"]), max(data["predictions"]))
        plt.scatter(data["targets"], data["predictions"], s=14, alpha=.55, color=colors[head], label=f"{head}, seed {rep['seed']}")
        true_pred_data.extend(
            {"head_type":head,"seed":rep["seed"],"universe_id":uid,"target":target,"prediction":prediction}
            for uid,target,prediction in zip(data["ids"],data["targets"],data["predictions"])
        )
    plt.plot([low, high], [low, high], "--", color="black", label="identity")
    plt.xlabel("True Ωm"); plt.ylabel("Predicted Ωm")
    plt.title("Median-MAE representative runs")
    plt.legend()
    save_plot(plt, out, "representative_true_vs_predicted", true_pred_data)

    plt.figure(figsize=(6.4, 5.2))
    for head in HEADS:
        rep = rep_lookup[head]
        data = predictions[(head, int(rep["seed"]))]
        residuals = [p-t for t,p in zip(data["targets"],data["predictions"])]
        plt.scatter(data["targets"], residuals, s=14, alpha=.55, color=colors[head], label=f"{head}, seed {rep['seed']}")
        residual_data.extend(
            {"head_type":head,"seed":rep["seed"],"universe_id":uid,"target":target,"residual":residual}
            for uid,target,residual in zip(data["ids"],data["targets"],residuals)
        )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("True Ωm"); plt.ylabel("Prediction − truth")
    plt.title("Representative residuals")
    plt.legend()
    save_plot(plt, out, "representative_residuals_vs_true", residual_data)

    plt.figure(figsize=(6.4, 4.8))
    for head in HEADS:
        rep = rep_lookup[head]
        data = predictions[(head, int(rep["seed"]))]
        plt.hist(data["predictions"], bins=20, alpha=.5, color=colors[head], label=head)
        distribution_data.extend(
            {"head_type":head,"seed":rep["seed"],"universe_id":uid,"prediction":prediction}
            for uid,prediction in zip(data["ids"],data["predictions"])
        )
    plt.xlabel("Predicted Ωm"); plt.ylabel("Count")
    plt.title("Representative prediction distributions")
    plt.legend()
    save_plot(plt, out, "representative_prediction_distributions", distribution_data)


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root / "scripts"))
    from analysis_reporting.common import validate_analysis

    spec_path = args.spec if args.spec.is_absolute() else root / args.spec
    validation = validate_analysis(root, spec_path)
    if validation.errors:
        raise RuntimeError("; ".join(validation.errors))
    spec = validation.spec
    out = root / spec["output_directory"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "plot_data").mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    seed_rows, prediction_cache = read_inputs(root, validation)
    aggregate_rows = aggregate(seed_rows)
    paired_rows = pair(seed_rows)
    representative_rows = representatives(seed_rows)
    diagnostics = [{
        key: row[key] for key in (
            "model","head_type","seed","experiment_name","prediction_path",
            "test_pearson","pearson_status","target_mean","prediction_mean",
            "target_sd","prediction_sd","prediction_sd_ratio",
            "unique_prediction_count","exact_repeated_prediction_fraction",
            "approximate_repeated_prediction_fraction","prediction_range",
            "residual_mean","residual_sd","maximum_absolute_residual",
        )
    } for row in seed_rows]

    write_csv(out / "seed_level_results.csv", seed_rows)
    write_csv(out / "aggregated_results.csv", aggregate_rows)
    write_csv(out / "paired_head_differences.csv", paired_rows)
    write_csv(out / "prediction_diagnostics.csv", diagnostics)
    write_csv(out / "representative_runs.csv", representative_rows)

    protocol = [
        {"field":"model","value":"EvolveGCN-H"},
        {"field":"universes","value":750},
        {"field":"halos","value":"Top1000"},
        {"field":"normalization","value":"none"},
        {"field":"periodic kNN","value":"k=8, box size 25"},
        {"field":"hidden dimension / layers","value":"32 / 2"},
        {"field":"graph / temporal pooling","value":"mean / mean"},
        {"field":"regression-head factor","value":"linear (33 parameters) vs MLP (1089 parameters)"},
        {"field":"seeds","value":"42, 123, 777, 999, 2025"},
        {"field":"splits","value":"450 train / 99 validation / 201 test; exact matched IDs"},
    ]
    write_csv(out / "protocol_table.csv", protocol)
    (out / "protocol_table.md").write_text(
        "# Controlled regression-head protocol\n\n" + md_table(protocol, ["field","value"]),
        encoding="utf-8",
    )

    rounded = [{
        "regression head": row["head_type"],
        "seeds": row["seed_count"],
        "head parameters": row["head_parameter_count"],
        "mean test MAE": f"{row['mean_test_mae']:.6f}",
        "SD test MAE": f"{row['sd_test_mae']:.6f}",
        "median test MAE": f"{row['median_test_mae']:.6f}",
        "mean test RMSE": f"{row['mean_test_rmse']:.6f}",
        "SD test RMSE": f"{row['sd_test_rmse']:.6f}",
        "mean test R²": f"{row['mean_test_r2']:.6f}",
        "SD test R²": f"{row['sd_test_r2']:.6f}",
        "mean prediction-SD ratio": f"{row['mean_prediction_sd_ratio']:.6f}",
        "SD prediction-SD ratio": f"{row['sd_prediction_sd_ratio']:.6f}",
        "mean repeated fraction": f"{row['mean_repeated_prediction_fraction']:.6f}",
        "undefined Pearson": row["undefined_pearson_count"],
    } for row in aggregate_rows]
    main_caption = (
        "EvolveGCN-H U750 Top1000 regression-head results: no normalization, "
        "periodic kNN k=8, hidden dimension 32, two layers, graph mean and "
        "temporal mean pooling, linear versus MLP heads, and five matched seeds."
    )
    table_set(tables, "main_results_table", rounded, main_caption)
    main_fields = list(rounded[0])
    marked_rows = []
    preferred_fields = {
        "mean test MAE", "mean test RMSE", "mean test R²",
        "mean prediction-SD ratio", "mean repeated fraction",
    }
    for row in rounded:
        marked = dict(row)
        if row["regression head"] == "linear":
            for field in preferred_fields:
                marked[field] = f"**{marked[field]}**"
        marked_rows.append(marked)
    (tables / "main_results_table.md").write_text(
        "# Main regression-head results\n\n" + main_caption + "\n\n"
        + md_table(marked_rows, main_fields), encoding="utf-8"
    )
    tex = (tables / "main_results_table.tex").read_text(encoding="utf-8")
    for field in preferred_fields:
        value = latex_escape(rounded[0][field])
        tex = tex.replace(value, "\\textbf{" + value + "}", 1)
    (tables / "main_results_table.tex").write_text(tex, encoding="utf-8")
    table_set(tables, "seed_level_head_results", seed_rows, "Seed-level regression-head results")
    table_set(tables, "aggregate_head_results", aggregate_rows, "Aggregate regression-head results")
    table_set(tables, "paired_head_differences", paired_rows, "Paired linear-minus-MLP differences")
    table_set(tables, "prediction_collapse_diagnostics", diagnostics, "Prediction-compression diagnostics")
    parameter_rows = [
        {"head_type":"linear","architecture":"Linear(32,1)","input_dimension":32,"hidden_dimensions":"","activation":"none","dropout":0.0,"output_activation":"identity","parameter_count":33},
        {"head_type":"mlp","architecture":"Linear(32,32)-ReLU-Dropout(0.2)-Linear(32,1)","input_dimension":32,"hidden_dimensions":"32","activation":"ReLU","dropout":0.2,"output_activation":"identity","parameter_count":1089},
    ]
    table_set(tables, "head_parameter_comparison", parameter_rows, "Regression-head parameter comparison")
    descriptive = [{
        "head_type": r["head_type"], "mean_test_mae": f"{r['mean_test_mae']:.6f}",
        "mean_test_rmse": f"{r['mean_test_rmse']:.6f}",
        "mean_test_r2": f"{r['mean_test_r2']:.6f}",
        "interpretation": "descriptively better" if r["head_type"]=="linear" else "tested nonlinear reference"
    } for r in aggregate_rows]
    table_set(tables, "descriptive_head_summary", descriptive, "Descriptive head summary", latex=False)

    progression = [
        {"stage":"Baseline","fixed_protocol":"U750 Top1000; graph mean; temporal mean; all encoder/training settings fixed","change":"MLP head","mean_mae":f"{aggregate_rows[1]['mean_test_mae']:.6f}","paired_delta":"—"},
        {"stage":"Head ablation","fixed_protocol":"identical to baseline","change":"linear head","mean_mae":f"{aggregate_rows[0]['mean_test_mae']:.6f}","paired_delta":f"{paired_rows[0]['mean_paired_mae_difference']:+.6f}"},
    ]
    write_csv(out / "controlled_progression_table.csv", progression)
    (out / "controlled_progression_table.md").write_text(
        "# Controlled progression\n\n" + md_table(progression, list(progression[0])),
        encoding="utf-8",
    )
    table_set(tables, "controlled_progression_table", progression, "Controlled head-only progression")
    history = [
        {"stage":"MLP temporal-mean baseline","mean_mae":"0.061404","protocol_changes":"reference U750 Top1000 protocol"},
        {"stage":"Controlled linear head","mean_mae":"0.055843","protocol_changes":"head only: MLP → linear"},
        {"stage":"Linear temporal-last context","mean_mae":"0.055351","protocol_changes":"temporal pooling mean → last; descriptive only"},
        {"stage":"Linear mean_max context","mean_mae":"0.080973","protocol_changes":"graph pooling mean → mean_max and required input width; descriptive only"},
    ]
    write_csv(out / "historical_context_table.csv", history)
    (out / "historical_context_table.md").write_text(
        "# Descriptive project progression — protocols differ\n\n"
        + md_table(history, list(history[0])), encoding="utf-8"
    )

    implementation = """# Regression-head implementation

Both heads receive the same 32-dimensional representation after masked mean
graph pooling and temporal mean pooling.

- `linear`: `Linear(32,1)`, 33 trainable parameters.
- `mlp`: `Linear(32,32) → ReLU → Dropout(0.2) → Linear(32,1)`, 1,089
  trainable parameters.

All Linear layers use bias and identity output activation. The Evolve head
modules use PyTorch default initialization. AdamW receives `model.parameters`,
so head parameters are optimized. Target Omega_m is unnormalized. Head type
changes checkpoint structure; checkpoints were checked only for existence.
"""
    (out / "head_implementation.md").write_text(implementation, encoding="utf-8")

    d = paired_rows[0]
    summary = f"""# Scientific summary

## Objective

Test whether a direct linear head or the repository's shallow MLP better maps
the fixed EvolveGCN-H pooled representation to Omega_m.

## Experimental design

Ten U750 Top1000 runs form five exact matched-seed pairs. Dataset, encoder,
graph mean pooling, temporal mean pooling, target scale, optimization, and
training settings are fixed; only the regression head differs.

## Verification

All artifacts and 201-row prediction files verify. Ordered train/validation/
test splits match exactly within every pair. Saved and recomputed primary
metrics agree within 1e-6.

## Quantitative results

Linear achieves MAE {aggregate_rows[0]['mean_test_mae']:.6f} ±
{aggregate_rows[0]['sd_test_mae']:.6f}, RMSE
{aggregate_rows[0]['mean_test_rmse']:.6f}, and R²
{aggregate_rows[0]['mean_test_r2']:.6f}. MLP achieves MAE
{aggregate_rows[1]['mean_test_mae']:.6f} ±
{aggregate_rows[1]['sd_test_mae']:.6f}, RMSE
{aggregate_rows[1]['mean_test_rmse']:.6f}, and R²
{aggregate_rows[1]['mean_test_r2']:.6f}. Mean paired MAE(linear−MLP) is
{d['mean_paired_mae_difference']:+.6f} ±
{d['sd_paired_mae_difference']:.6f}; linear wins four of five seeds.

## Prediction-compression diagnostics

Mean prediction-SD ratios are {aggregate_rows[0]['mean_prediction_sd_ratio']:.6f}
for linear and {aggregate_rows[1]['mean_prediction_sd_ratio']:.6f} for MLP.
Linear has no repeated predictions. MLP has repeated predictions in two seeds,
with mean repeated fraction
{aggregate_rows[1]['mean_repeated_prediction_fraction']:.6f}.

## Interpretation

The tested linear head offers a modest and mostly consistent improvement. This
is compatible with useful Omega_m information already being linearly
accessible in the pooled representation, but does not prove that explanation.
The head affects compression in some runs but is not a complete explanation
for remaining prediction compression.

## Limitations

The effect is comparable to between-seed variability and reverses at seed 777.
Only one shallow MLP was tested. Similar train-to-validation MAE gaps
({aggregate_rows[0]['mean_validation_minus_train_mae']:.6f} linear;
{aggregate_rows[1]['mean_validation_minus_train_mae']:.6f} MLP) do not support
a specific MLP-overfitting claim. Deeper, residual, probabilistic, and
uncertainty-aware heads remain untested.

## Conclusion

Under the tested U750 Top1000 protocol, the simpler linear head is
descriptively preferable to the shallow MLP, while the magnitude and one
reversed seed require a protocol-specific, non-universal conclusion.
"""
    (out / "scientific_summary.md").write_text(summary, encoding="utf-8")
    thesis = summary.replace("# Scientific summary", "# Regression-Head Ablation").replace(
        "## Objective", "## Regression-Head Motivation"
    ).replace("## Experimental design", "## Experimental Design\n\n### Controlled Variables").replace(
        "## Verification", "## Verification Procedure"
    ).replace("## Quantitative results", "## Quantitative Results\n\n### Paired-Seed Analysis").replace(
        "## Prediction-compression diagnostics", "## Prediction-Compression Diagnostics"
    )
    (out / "thesis_section.md").write_text(thesis, encoding="utf-8")
    presentation = f"""# Presentation-ready summary

## Recommended material

1. Protocol table.
2. Two-row main head table.
3. Test MAE by regression head.
4. Paired MAE difference by seed.
5. Prediction-SD-ratio comparison.
6. Representative true-versus-predicted figure.
7. Controlled progression table.

## Supervisor-meeting narrative

**Question:** Does a nonlinear MLP improve the final Omega_m mapping?

**Method:** Five matched seeds with the encoder, pooling, data, splits, and
training protocol fixed.

**Result:** Linear reduced paired MAE by {abs(d['mean_paired_mae_difference']):.6f}
on average and won four of five seeds.

**Diagnostic:** Linear predictions were unrepeated and somewhat better
dispersed on average, but the dispersion effect varied by seed.

**Conclusion:** The tested MLP adds parameters without improving aggregate
generalization; more expressive head families require separate controls.
"""
    (out / "presentation_ready_summary.md").write_text(presentation, encoding="utf-8")
    (out / "README.md").write_text(
        "# Controlled EvolveGCN-H regression-head ablation\n\n"
        "Deterministic, verified analysis of ten existing U750 Top1000 runs. "
        "Regenerate with `rebuild_regression_head_analysis.py`; no dataset or "
        "checkpoint loading is required.\n", encoding="utf-8"
    )

    plots(out, seed_rows, aggregate_rows, paired_rows, representative_rows, prediction_cache)

    manifest_path = out / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "analysis_name": spec["analysis_name"],
        "specification": spec_path.relative_to(root).as_posix(),
        "focused_rebuild_script": Path(__file__).relative_to(root).as_posix(),
        "metric_tolerance": TOLERANCE,
        "regression_head_specific_validation": {
            "status": "PASS",
            "seed_level_rows": len(seed_rows),
            "aggregate_rows": len(aggregate_rows),
            "paired_head_rows": len(paired_rows),
            "prediction_diagnostic_rows": len(diagnostics),
            "representative_rows": len(representative_rows),
            "unique_experiment_mappings": len({r["experiment_path"] for r in seed_rows}),
            "unique_prediction_mappings": len({r["prediction_path"] for r in seed_rows}),
            "unique_prediction_hashes": len({r["prediction_sha256"] for r in seed_rows}),
            "exact_paired_splits": True,
            "representative_policy": "median_test_mae",
            "representative_runs": representative_rows,
            "head_order": list(HEADS),
            "checkpoint_policy": "existence_only_never_loaded",
        },
    })
    package_files = sorted(
        path.relative_to(out).as_posix()
        for path in out.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    if "analysis_manifest.json" not in package_files:
        package_files.append("analysis_manifest.json")
        package_files.sort()
    manifest["generated_outputs"] = package_files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Focused regression-head build: {out}")
    print(f"Rows: seed={len(seed_rows)}, aggregate={len(aggregate_rows)}, paired={len(paired_rows)}, diagnostics={len(diagnostics)}, representatives={len(representative_rows)}")
    print(f"Requested figure triples: {len(FIGURES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
