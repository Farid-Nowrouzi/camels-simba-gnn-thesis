#!/usr/bin/env python3
"""Validate and report the completed U1000 Top1000 training-scaling matrix.

This script reads only registry, manifest, metric, prediction, and target-table
artifacts. It never imports or invokes training code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = ROOT / "reports/experiment_registry"
ALL_RUNS_PATH = REGISTRY_DIR / "u1000_top1000_training_scaling_all_runs.csv"
MASTER_PATH = REGISTRY_DIR / "master_experiment_registry.csv"
MATRIX_PATH = ROOT / "configs/experiment_registry/u1000_top1000_training_scaling_matrix.json"
FIGURE_DIR = ROOT / "presentation_assets/u1000_top1000_training_scaling"
SORTED_DIR = REGISTRY_DIR / "u1000_top1000_sorted_predictions"
TRAINING_COUNTS = (20, 50, 100, 200, 450, 700)
SEEDS = (42, 123, 2025)
MODEL_LABELS = {
    "EvolveGCNHRegressor": "EvolveGCN-H",
    "StaticGCNRegressor": "Static GCN",
}
MODEL_SLUGS = {
    "EvolveGCNHRegressor": "evolvegcn_h",
    "StaticGCNRegressor": "static_gcn",
}
METRIC_TOLERANCE = 1e-9
TARGET_TOLERANCE = 1e-7


class ValidationError(RuntimeError):
    """Raised when an authoritative artifact violates the expected matrix."""


@dataclass(frozen=True)
class ValidatedRun:
    model: str
    training_count: int
    seed: int
    registry: dict[str, str]
    master: dict[str, str]
    metrics: dict[str, Any]
    universe_ids: tuple[str, ...]
    targets: tuple[float, ...]
    predictions: tuple[float, ...]
    derived: dict[str, float]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValidationError(f"missing CSV: {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"CSV has no header: {path.relative_to(ROOT)}")
        return list(reader.fieldnames), list(reader)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing JSON: {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def repo_path(value: str, field: str) -> Path:
    if not value:
        raise ValidationError(f"registry field {field!r} is empty")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValidationError(f"registry field {field!r} escapes repository: {value}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fields(actual: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValidationError(f"{label} missing fields: {', '.join(missing)}")


def finite_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{label} is not finite: {value!r}")
    return number


def close(actual: float, expected: float, label: str, tolerance: float = METRIC_TOLERANCE) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValidationError(f"{label} mismatch: recomputed={actual:.17g}, reported={expected:.17g}")


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def sample_sd(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def derive_metrics(targets: tuple[float, ...], predictions: tuple[float, ...]) -> dict[str, float]:
    residuals = [pred - target for target, pred in zip(targets, predictions)]
    absolute = [abs(value) for value in residuals]
    squared = [value * value for value in residuals]
    target_mean = mean(targets)
    ss_res = sum(squared)
    ss_tot = sum((target - target_mean) ** 2 for target in targets)
    if ss_tot == 0:
        raise ValidationError("R2 is undefined because all test targets are identical")
    order = sorted(range(len(targets)), key=lambda index: (targets[index], index))
    quintile_size = math.ceil(len(targets) * 0.2)
    groups = {
        "lowest_quintile": order[:quintile_size],
        "middle_60_percent": order[quintile_size:-quintile_size],
        "highest_quintile": order[-quintile_size:],
    }
    result = {
        "mae": mean(absolute),
        "mse": mean(squared),
        "rmse": math.sqrt(mean(squared)),
        "r2": 1.0 - ss_res / ss_tot,
        "prediction_mean": mean(predictions),
        "prediction_standard_deviation": sample_sd(predictions),
        "true_target_standard_deviation": sample_sd(targets),
        "residual_mean": mean(residuals),
        "residual_standard_deviation": sample_sd(residuals),
    }
    result["prediction_sd_ratio"] = (
        result["prediction_standard_deviation"] / result["true_target_standard_deviation"]
    )
    for name, indices in groups.items():
        result[f"{name}_mae"] = mean(absolute[index] for index in indices)
        result[f"{name}_bias"] = mean(residuals[index] for index in indices)
    return result


def validate() -> list[ValidatedRun]:
    all_fields, all_rows = read_csv(ALL_RUNS_PATH)
    master_fields, master_rows = read_csv(MASTER_PATH)
    require_fields(
        all_fields,
        ("model", "seed", "training_universe_count", "canonical_experiment_id",
         "split_manifest_path", "configuration_path", "experiment_directory", "status",
         "validation_result", "test_count", "best_epoch", "runtime_seconds", "test_mae",
         "test_mse", "test_rmse", "test_r2"),
        ALL_RUNS_PATH.name,
    )
    require_fields(
        master_fields,
        ("canonical_experiment_id", "metrics_path", "predictions_path", "split_manifest_path",
         "configuration_path", "experiment_directory", "status", "validation_result"),
        MASTER_PATH.name,
    )
    if len(all_rows) != 36:
        raise ValidationError(f"expected 36 all-runs rows, found {len(all_rows)}")
    if any(row["status"] != "completed" or row["validation_result"] != "PASS" for row in all_rows):
        raise ValidationError("all-runs table contains a row that is not completed/PASS")
    if any("failed_invalid_config" in " ".join(row.values()) for row in all_rows):
        raise ValidationError("all-runs table includes a failed_invalid_config experiment")

    ids = [row["canonical_experiment_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValidationError("all-runs table contains duplicate canonical experiment IDs")
    master_by_id: dict[str, dict[str, str]] = {}
    for row in master_rows:
        run_id = row["canonical_experiment_id"]
        if run_id in ids:
            if run_id in master_by_id:
                raise ValidationError(f"duplicate master-registry row for {run_id}")
            master_by_id[run_id] = row
    if set(master_by_id) != set(ids):
        missing = sorted(set(ids) - set(master_by_id))
        raise ValidationError(f"master registry is missing matrix IDs: {missing}")

    matrix = read_json(MATRIX_PATH)
    entries = matrix.get("entries")
    if not isinstance(entries, list):
        raise ValidationError("matrix JSON entries must be a list")
    matrix_valid = [
        entry for entry in entries
        if isinstance(entry, dict)
        and entry.get("status") == "completed"
        and entry.get("validation_result") == "PASS"
        and entry.get("matrix_included") is True
    ]
    if len(matrix_valid) != 36:
        raise ValidationError(f"expected 36 completed/PASS matrix entries, found {len(matrix_valid)}")
    if {entry.get("canonical_experiment_id") for entry in matrix_valid} != set(ids):
        raise ValidationError("matrix JSON and all-runs CSV canonical IDs differ")

    combo_counts = Counter(
        (row["model"], int(row["training_universe_count"]), int(row["seed"])) for row in all_rows
    )
    expected = {
        (model, training_count, seed)
        for model in MODEL_LABELS for training_count in TRAINING_COUNTS for seed in SEEDS
    }
    if set(combo_counts) != expected or any(count != 1 for count in combo_counts.values()):
        raise ValidationError("matrix does not contain exactly one row for every model/count/seed combination")
    model_counts = Counter(row["model"] for row in all_rows)
    if model_counts != Counter({model: 18 for model in MODEL_LABELS}):
        raise ValidationError(f"expected 18 rows per model, found {dict(model_counts)}")

    target_cache: dict[Path, tuple[str, dict[str, float]]] = {}
    validated: list[ValidatedRun] = []
    for row in sorted(
        all_rows,
        key=lambda item: (list(MODEL_LABELS).index(item["model"]), int(item["training_universe_count"]), int(item["seed"])),
    ):
        run_id = row["canonical_experiment_id"]
        master = master_by_id[run_id]
        if master["status"] != "completed" or master["validation_result"] != "PASS":
            raise ValidationError(f"master row is not completed/PASS: {run_id}")
        for field in ("split_manifest_path", "configuration_path", "experiment_directory"):
            if row[field] != master[field]:
                raise ValidationError(f"registry path disagreement for {run_id}: {field}")

        manifest_path = repo_path(row["split_manifest_path"], "split_manifest_path")
        predictions_path = repo_path(master["predictions_path"], "predictions_path")
        metrics_path = repo_path(master["metrics_path"], "metrics_path")
        config_path = repo_path(row["configuration_path"], "configuration_path")
        experiment_path = repo_path(row["experiment_directory"], "experiment_directory")
        for path, label in ((manifest_path, "manifest"), (predictions_path, "predictions"),
                            (metrics_path, "metrics"), (config_path, "configuration")):
            if not path.is_file():
                raise ValidationError(f"{run_id} missing {label}: {path.relative_to(ROOT)}")
        if not experiment_path.is_dir():
            raise ValidationError(f"{run_id} missing experiment directory: {experiment_path.relative_to(ROOT)}")

        manifest = read_json(manifest_path)
        test_ids = manifest.get("test_ids")
        if not isinstance(test_ids, list) or not all(isinstance(value, str) for value in test_ids):
            raise ValidationError(f"{run_id} manifest test_ids must be a string list")
        if len(test_ids) != 201 or int(row["test_count"]) != 201:
            raise ValidationError(f"{run_id} does not have exactly 201 test IDs")
        if len(test_ids) != len(set(test_ids)):
            raise ValidationError(f"{run_id} manifest contains duplicate test IDs")
        dataset_binding = manifest.get("dataset_binding")
        if not isinstance(dataset_binding, dict):
            raise ValidationError(f"{run_id} manifest lacks dataset_binding")
        target_path = repo_path(str(dataset_binding.get("target_table_path", "")), "target_table_path")
        target_hash = str(dataset_binding.get("target_table_sha256", ""))
        if target_path not in target_cache:
            target_fields, target_rows = read_csv(target_path)
            require_fields(target_fields, ("universe_id", "omega_m"), target_path.name)
            target_map: dict[str, float] = {}
            for target_row in target_rows:
                universe_id = target_row["universe_id"]
                if universe_id in target_map:
                    raise ValidationError(f"authoritative target table has duplicate ID: {universe_id}")
                target_map[universe_id] = finite_float(target_row["omega_m"], f"target {universe_id}")
            target_cache[target_path] = (sha256(target_path), target_map)
        actual_target_hash, target_map = target_cache[target_path]
        if actual_target_hash != target_hash or (master.get("target_table_sha256") and actual_target_hash != master["target_table_sha256"]):
            raise ValidationError(f"{run_id} authoritative target-table SHA-256 mismatch")
        missing_targets = [universe_id for universe_id in test_ids if universe_id not in target_map]
        if missing_targets:
            raise ValidationError(f"{run_id} target table is missing test IDs: {missing_targets[:3]}")

        prediction_fields, prediction_rows = read_csv(predictions_path)
        require_fields(
            prediction_fields,
            ("universe_id", "true_omega_m", "pred_omega_m", "absolute_error", "squared_error"),
            predictions_path.name,
        )
        if len(prediction_rows) != 201:
            raise ValidationError(f"{run_id} prediction row count is {len(prediction_rows)}, not 201")
        prediction_ids = [item["universe_id"] for item in prediction_rows]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValidationError(f"{run_id} predictions contain duplicate IDs")
        if prediction_ids != test_ids:
            raise ValidationError(f"{run_id} prediction IDs do not exactly match ordered manifest test IDs")
        targets: list[float] = []
        predictions: list[float] = []
        for prediction_row in prediction_rows:
            universe_id = prediction_row["universe_id"]
            target = finite_float(prediction_row["true_omega_m"], f"{run_id} target {universe_id}")
            prediction = finite_float(prediction_row["pred_omega_m"], f"{run_id} prediction {universe_id}")
            if not math.isclose(target, target_map[universe_id], rel_tol=0.0, abs_tol=TARGET_TOLERANCE):
                raise ValidationError(f"{run_id} target mismatch for {universe_id}")
            absolute_error = finite_float(prediction_row["absolute_error"], f"{run_id} absolute error")
            squared_error = finite_float(prediction_row["squared_error"], f"{run_id} squared error")
            close(absolute_error, abs(prediction - target), f"{run_id} saved absolute error", 1e-7)
            close(squared_error, (prediction - target) ** 2, f"{run_id} saved squared error", 1e-7)
            targets.append(target)
            predictions.append(prediction)

        metrics = read_json(metrics_path)
        reported_test = metrics.get("test")
        if not isinstance(reported_test, dict):
            raise ValidationError(f"{run_id} metrics JSON lacks test object")
        derived = derive_metrics(tuple(targets), tuple(predictions))
        for metric in ("mae", "mse", "rmse", "r2"):
            close(derived[metric], finite_float(str(reported_test.get(metric)), f"{run_id} metrics.test.{metric}"),
                  f"{run_id} metrics JSON {metric}")
            close(derived[metric], finite_float(row[f"test_{metric}"], f"{run_id} all-runs {metric}"),
                  f"{run_id} all-runs {metric}")
            if master.get(f"test_{metric}"):
                close(derived[metric], finite_float(master[f"test_{metric}"], f"{run_id} master {metric}"),
                      f"{run_id} master-registry {metric}")
        validated.append(
            ValidatedRun(row["model"], int(row["training_universe_count"]), int(row["seed"]),
                         row, master, metrics, tuple(test_ids), tuple(targets), tuple(predictions), derived)
        )
    return validated


def number(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.12g}"


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: number(row[field]) if isinstance(row.get(field), (float, int)) else row.get(field, "") for field in fields})


PER_RUN_DERIVED = [
    "mae", "mse", "rmse", "r2", "prediction_mean", "prediction_standard_deviation",
    "true_target_standard_deviation", "prediction_sd_ratio", "residual_mean",
    "residual_standard_deviation", "lowest_quintile_mae", "lowest_quintile_bias",
    "middle_60_percent_mae", "middle_60_percent_bias", "highest_quintile_mae",
    "highest_quintile_bias",
]


def per_run_rows(runs: list[ValidatedRun]) -> list[dict[str, Any]]:
    return [
        {
            "model": MODEL_LABELS[run.model], "training_count": run.training_count, "seed": run.seed,
            **run.derived, "best_epoch": int(run.registry["best_epoch"]),
            "runtime_seconds": finite_float(run.registry["runtime_seconds"], "runtime_seconds"),
        }
        for run in runs
    ]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = PER_RUN_DERIVED + ["best_epoch", "runtime_seconds"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["training_count"])].append(row)
    result = []
    for model in MODEL_LABELS.values():
        for training_count in TRAINING_COUNTS:
            group = grouped[(model, training_count)]
            item: dict[str, Any] = {"model": model, "training_count": training_count, "seed_count": len(group)}
            for field in numeric:
                values = [float(row[field]) for row in group]
                item[f"{field}_mean"] = mean(values)
                item[f"{field}_sample_std"] = sample_sd(values)
            result.append(item)
    return result


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["training_count"], row["seed"]): row for row in rows}
    result = []
    for training_count in TRAINING_COUNTS:
        for seed in SEEDS:
            static = lookup[("Static GCN", training_count, seed)]
            evolve = lookup[("EvolveGCN-H", training_count, seed)]
            item: dict[str, Any] = {"training_count": training_count, "seed": seed}
            for metric in ("mae", "mse", "rmse", "r2", "prediction_sd_ratio"):
                item[f"static_{metric}"] = static[metric]
                item[f"evolve_{metric}"] = evolve[metric]
                item[f"static_minus_evolve_{metric}"] = static[metric] - evolve[metric]
            result.append(item)
    return result


def train450_vs700_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["training_count"], row["seed"]): row for row in rows}
    result = []
    for model in MODEL_LABELS.values():
        for seed in SEEDS:
            train450 = lookup[(model, 450, seed)]
            train700 = lookup[(model, 700, seed)]
            item: dict[str, Any] = {"model": model, "seed": seed}
            for metric in ("mae", "mse", "rmse", "r2", "prediction_sd_ratio"):
                item[f"train450_{metric}"] = train450[metric]
                item[f"train700_{metric}"] = train700[metric]
                item[f"train700_minus_train450_{metric}"] = train700[metric] - train450[metric]
            result.append(item)
    return result


def generate_sorted_csvs(runs: list[ValidatedRun]) -> None:
    fields = ["rank", "universe_id", "true_omega_m", "predicted_omega_m", "residual",
              "absolute_error", "model", "seed"]
    for run in runs:
        if run.training_count != 700:
            continue
        ordered = sorted(zip(run.universe_ids, run.targets, run.predictions), key=lambda value: (value[1], value[0]))
        rows = []
        for rank, (universe_id, target, prediction) in enumerate(ordered, start=1):
            residual = prediction - target
            rows.append({"rank": rank, "universe_id": universe_id, "true_omega_m": target,
                         "predicted_omega_m": prediction, "residual": residual,
                         "absolute_error": abs(residual), "model": MODEL_LABELS[run.model], "seed": run.seed})
        filename = f"{MODEL_SLUGS[run.model]}_train700_seed{run.seed}_sorted.csv"
        write_csv(SORTED_DIR / filename, fields, rows)


def plot_outputs(runs: list[ValidatedRun], rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"EvolveGCN-H": "#d55e00", "Static GCN": "#0072b2"}
    markers = {"EvolveGCN-H": "o", "Static GCN": "s"}
    metric_plots = [
        ("mae", "MAE", "01_mae_learning_curve.png"),
        ("mse", "MSE", "02_mse_learning_curve.png"),
        ("rmse", "RMSE", "03_rmse_learning_curve.png"),
        ("r2", "R²", "04_r2_learning_curve.png"),
        ("prediction_sd_ratio", "Prediction SD / target SD", "05_prediction_sd_ratio_learning_curve.png"),
    ]
    for metric, ylabel, filename in metric_plots:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for model in MODEL_LABELS.values():
            selected = [item for item in aggregates if item["model"] == model]
            ax.errorbar(TRAINING_COUNTS, [item[f"{metric}_mean"] for item in selected],
                        yerr=[item[f"{metric}_sample_std"] for item in selected], label=model,
                        color=colors[model], marker=markers[model], capsize=3, linewidth=1.8)
        if metric == "prediction_sd_ratio":
            ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1, label="Equal spread")
        elif metric == "r2":
            ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        ax.set(xlabel="Training universes", ylabel=ylabel, title=f"U1000 Top1000: {ylabel} learning curve")
        ax.set_xscale("log")
        ax.set_xticks(TRAINING_COUNTS, labels=[str(value) for value in TRAINING_COUNTS])
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=220)
        plt.close(fig)

    train700 = [run for run in runs if run.training_count == 700]
    global_min = min(min(run.targets + run.predictions) for run in train700)
    global_max = max(max(run.targets + run.predictions) for run in train700)
    margin = 0.04 * (global_max - global_min)
    limits = (global_min - margin, global_max + margin)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.1), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, train700):
        ax.scatter(run.targets, run.predictions, s=12, alpha=0.65, color=colors[MODEL_LABELS[run.model]], edgecolors="none")
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        d = run.derived
        ax.text(0.03, 0.97, f"MAE={d['mae']:.4f}\nMSE={d['mse']:.4f}\nRMSE={d['rmse']:.4f}\nR²={d['r2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"})
        ax.set_title(f"{MODEL_LABELS[run.model]}, seed {run.seed}")
        ax.set_xlim(limits); ax.set_ylim(limits); ax.grid(alpha=0.2)
    fig.supxlabel("True Ωm"); fig.supylabel("Predicted Ωm")
    fig.suptitle("Train700 true versus predicted Ωm", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_train700_true_vs_predicted.png", dpi=220)
    plt.close(fig)

    true_min = min(min(run.targets) for run in train700)
    pred_max = max(max(run.predictions) for run in train700)
    pred_min = min(min(run.predictions) for run in train700)
    true_max = max(max(run.targets) for run in train700)
    sorted_limits = (min(true_min, pred_min) - margin, max(true_max, pred_max) + margin)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.1), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, train700):
        ordered = sorted(zip(run.targets, run.predictions, run.universe_ids), key=lambda value: (value[0], value[2]))
        ranks = range(1, len(ordered) + 1)
        ax.plot(ranks, [value[0] for value in ordered], color="black", linewidth=1.7, label="True")
        ax.plot(ranks, [value[1] for value in ordered], color=colors[MODEL_LABELS[run.model]], linewidth=1.15, label="Predicted")
        ax.text(0.02, 0.97, f"{MODEL_LABELS[run.model]}, seed {run.seed}", transform=ax.transAxes,
                va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_ylim(sorted_limits); ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.supxlabel("Rank within each run's test set (low to high true Ωm)")
    fig.supylabel("Ωm")
    fig.suptitle("Train700 sorted true and predicted Ωm (test sets kept separate)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_train700_sorted_low_to_high_omega_m.png", dpi=220)
    plt.close(fig)

    residual_limit = max(abs(pred - target) for run in train700 for target, pred in zip(run.targets, run.predictions)) * 1.05
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.1), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, train700):
        residuals = [pred - target for target, pred in zip(run.targets, run.predictions)]
        ax.scatter(run.targets, residuals, s=12, alpha=0.65, color=colors[MODEL_LABELS[run.model]], edgecolors="none")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.text(0.02, 0.97, f"{MODEL_LABELS[run.model]}, seed {run.seed}", transform=ax.transAxes,
                va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_ylim(-residual_limit, residual_limit); ax.grid(alpha=0.2)
    fig.supxlabel("True Ωm"); fig.supylabel("Residual (prediction − true)")
    fig.suptitle("Train700 residuals versus true Ωm", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08_train700_residual_vs_true.png", dpi=220)
    plt.close(fig)

    paired = paired_rows(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for seed in SEEDS:
        selected = [item for item in paired if item["seed"] == seed]
        ax.plot(TRAINING_COUNTS, [item["static_minus_evolve_mae"] for item in selected],
                marker="o", linewidth=1, alpha=0.6, label=f"seed {seed}")
    values_by_count = [[item["static_minus_evolve_mae"] for item in paired if item["training_count"] == count] for count in TRAINING_COUNTS]
    ax.plot(TRAINING_COUNTS, [mean(values) for values in values_by_count], color="black", marker="s", linewidth=2.2, label="paired mean")
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(xlabel="Training universes", ylabel="Static MAE − Evolve MAE", title="Paired Static minus Evolve MAE")
    ax.set_xscale("log"); ax.set_xticks(TRAINING_COUNTS, labels=[str(value) for value in TRAINING_COUNTS])
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIGURE_DIR / "09_static_minus_evolve_mae.png", dpi=220)
    plt.close(fig)

    comparison = train450_vs700_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, metric, ylabel in zip(axes, ("mae", "r2"), ("MAE", "R²")):
        for model in MODEL_LABELS.values():
            selected = [item for item in comparison if item["model"] == model]
            for item in selected:
                ax.plot((450, 700), (item[f"train450_{metric}"], item[f"train700_{metric}"]),
                        color=colors[model], alpha=0.35, linewidth=1)
            ax.plot((450, 700),
                    (mean(item[f"train450_{metric}"] for item in selected), mean(item[f"train700_{metric}"] for item in selected)),
                    color=colors[model], marker=markers[model], linewidth=2.4, label=model)
        ax.set(xticks=(450, 700), xlabel="Training universes", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(); fig.suptitle("Train450 versus Train700 (thin lines: paired seeds)")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "10_train450_vs_train700.png", dpi=220)
    plt.close(fig)


def fmt_mean_sd(values: Iterable[float], digits: int = 4) -> str:
    vals = list(values)
    return f"{mean(vals):.{digits}f} ± {sample_sd(vals):.{digits}f}"


def generate_reports(rows: list[dict[str, Any]], paired: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> None:
    lookup = defaultdict(list)
    for row in rows:
        lookup[(row["model"], row["training_count"])].append(row)
    static_better = sum(item["static_minus_evolve_mae"] < 0 for item in paired)
    train20_evolve = lookup[("EvolveGCN-H", 20)]
    train700_static = lookup[("Static GCN", 700)]
    train700_evolve = lookup[("EvolveGCN-H", 700)]
    low_bias = {model: fmt_mean_sd([item["lowest_quintile_bias"] for item in lookup[(model, 700)]]) for model in MODEL_LABELS.values()}
    mid_bias = {model: fmt_mean_sd([item["middle_60_percent_bias"] for item in lookup[(model, 700)]]) for model in MODEL_LABELS.values()}
    high_bias = {model: fmt_mean_sd([item["highest_quintile_bias"] for item in lookup[(model, 700)]]) for model in MODEL_LABELS.values()}
    change_text = {}
    for model in MODEL_LABELS.values():
        selected = [item for item in comparison if item["model"] == model]
        change_text[model] = fmt_mean_sd([item["train700_minus_train450_mae"] for item in selected])
    interpretation = f"""# U1000 Top1000 Training Scaling Scientific Interpretation

## Validated evidence base

This interpretation uses all 36 completed, validated runs: two models, six training-set sizes, and three seeds per cell. Every run has 201 ordered test predictions whose IDs and targets were checked against its stored split manifest and the authoritative target table. MAE, MSE, RMSE, and R² were independently recomputed from those predictions and matched the reported metrics.

## Learning curves

Static GCN improves substantially as the training set grows: its mean MAE changes from {fmt_mean_sd([item['mae'] for item in lookup[('Static GCN', 20)]])} at Train20 to {fmt_mean_sd([item['mae'] for item in train700_static])} at Train700. EvolveGCN-H is much less monotonic and remains worse at Train700, where its mean MAE is {fmt_mean_sd([item['mae'] for item in train700_evolve])}. The MSE, RMSE, and R² curves give the same broad ranking while exposing particularly large errors in unstable EvolveGCN-H runs.

## Static versus Evolve performance and seed stability

Static GCN has lower paired MAE in {static_better} of {len(paired)} model/count/seed comparisons. At Train700, Static GCN MAE is {fmt_mean_sd([item['mae'] for item in train700_static])}, compared with {fmt_mean_sd([item['mae'] for item in train700_evolve])} for EvolveGCN-H. Static GCN is also more stable across seeds at the larger training sizes. The tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol.

## Small-training-set Evolve instability

At Train20, EvolveGCN-H has MAE {fmt_mean_sd([item['mae'] for item in train20_evolve])}, R² {fmt_mean_sd([item['r2'] for item in train20_evolve], 3)}, and prediction-SD ratio {fmt_mean_sd([item['prediction_sd_ratio'] for item in train20_evolve], 3)}. The large across-seed dispersion and poor R² indicate unstable generalization in this low-data regime; they do not establish a universal property of evolving graph models.

## Prediction spread and residual structure

At Train700, the prediction-SD ratios are {fmt_mean_sd([item['prediction_sd_ratio'] for item in train700_static], 3)} for Static GCN and {fmt_mean_sd([item['prediction_sd_ratio'] for item in train700_evolve], 3)} for EvolveGCN-H. Ratios below one indicate compressed prediction spread; ratios above one indicate excessive spread relative to the true targets. The residual definition is prediction minus true target. For the lowest / middle 60% / highest target ranges, the Train700 mean biases are respectively {low_bias['Static GCN']}, {mid_bias['Static GCN']}, and {high_bias['Static GCN']} for Static GCN, and {low_bias['EvolveGCN-H']}, {mid_bias['EvolveGCN-H']}, and {high_bias['EvolveGCN-H']} for EvolveGCN-H. Thus the models tend to overpredict low targets when low-quintile bias is positive and underpredict high targets when high-quintile bias is negative, a regression-to-the-mean pattern visible in the sorted-target and residual plots.

## Train450 versus Train700

The paired Train700-minus-Train450 MAE change is {change_text['Static GCN']} for Static GCN and {change_text['EvolveGCN-H']} for EvolveGCN-H (negative favors Train700). Because each comparison preserves model and seed, this is the cleanest assessment of the final increase in training size, but only three seed pairs support each estimate.

## Scope of the temporal conclusion

The comparison does not show that temporal information is inherently unhelpful. The architectures differ in how they process the representation: the tested EvolveGCN-H uses all five temporal snapshots, whereas the tested Static GCN uses the final snapshot, alongside architecture-specific parameterization and optimization behavior. The results therefore support a statement about these tested implementations under this controlled protocol, not a general causal claim about temporal information. Better temporal architectures, tuning, regularization, or alternative temporal summaries could change the outcome.

## Thesis-ready conclusion

Across the completed U1000 Top1000 scaling matrix, Static GCN delivers lower error, stronger R², and generally greater seed stability than EvolveGCN-H, especially once moderate-to-large training sets are available. EvolveGCN-H is notably unstable in the smallest-data regime and its prediction spread and target-dependent residuals reveal calibration limitations. Increasing training data from 450 to 700 universes should be interpreted from the paired estimates above rather than assumed to help uniformly. Overall, the tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol, while the experiment does not establish that temporal information itself lacks predictive value.
"""
    (REGISTRY_DIR / "u1000_top1000_training_scaling_scientific_interpretation.md").write_text(interpretation, encoding="utf-8")

    summary = f"""# U1000 Top1000 Training Scaling Thesis Summary

The final matrix contains 36 completed and validated runs: EvolveGCN-H and Static GCN at 20, 50, 100, 200, 450, and 700 training universes, with seeds 42, 123, and 2025 and 201 test universes per run. Prediction IDs, ordering, targets, finiteness, and reported metrics were independently verified from the stored artifacts.

At Train700, Static GCN reaches MAE {fmt_mean_sd([item['mae'] for item in train700_static])} and R² {fmt_mean_sd([item['r2'] for item in train700_static], 3)}, whereas EvolveGCN-H reaches MAE {fmt_mean_sd([item['mae'] for item in train700_evolve])} and R² {fmt_mean_sd([item['r2'] for item in train700_evolve], 3)}. Static GCN has lower MAE in {static_better}/{len(paired)} seed-matched comparisons across the full learning curve. The EvolveGCN-H results are especially variable at Train20, and both models show target-dependent residual structure consistent with some regression toward the mean.

The thesis-level conclusion is deliberately architecture-specific: **the tested Static GCN architecture uses the available representation more effectively than the tested EvolveGCN-H architecture under the controlled protocol.** This result does not imply that temporal information is inherently unhelpful, because the comparison also reflects model architecture, capacity, optimization, and the particular temporal aggregation strategy.

Supporting evidence is provided in the per-run and mean±sample-SD tables, paired model and Train450/Train700 comparisons, six ID-preserving sorted prediction tables, and ten figures in `presentation_assets/u1000_top1000_training_scaling/`.
"""
    (REGISTRY_DIR / "u1000_top1000_training_scaling_thesis_summary.md").write_text(summary, encoding="utf-8")


def generate(runs: list[ValidatedRun]) -> None:
    rows = per_run_rows(runs)
    per_fields = ["model", "training_count", "seed", *PER_RUN_DERIVED, "best_epoch", "runtime_seconds"]
    write_csv(REGISTRY_DIR / "u1000_top1000_final_per_run_metrics.csv", per_fields, rows)
    aggregates = aggregate_rows(rows)
    aggregate_fields = ["model", "training_count", "seed_count"]
    for field in PER_RUN_DERIVED + ["best_epoch", "runtime_seconds"]:
        aggregate_fields.extend((f"{field}_mean", f"{field}_sample_std"))
    write_csv(REGISTRY_DIR / "u1000_top1000_final_mean_std_metrics.csv", aggregate_fields, aggregates)
    paired = paired_rows(rows)
    paired_fields = ["training_count", "seed"]
    for metric in ("mae", "mse", "rmse", "r2", "prediction_sd_ratio"):
        paired_fields.extend((f"static_{metric}", f"evolve_{metric}", f"static_minus_evolve_{metric}"))
    write_csv(REGISTRY_DIR / "u1000_top1000_static_vs_evolve_paired_comparison.csv", paired_fields, paired)
    comparison = train450_vs700_rows(rows)
    comparison_fields = ["model", "seed"]
    for metric in ("mae", "mse", "rmse", "r2", "prediction_sd_ratio"):
        comparison_fields.extend((f"train450_{metric}", f"train700_{metric}", f"train700_minus_train450_{metric}"))
    write_csv(REGISTRY_DIR / "u1000_top1000_train450_vs_train700_comparison.csv", comparison_fields, comparison)
    generate_sorted_csvs(runs)
    plot_outputs(runs, rows, aggregates)
    generate_reports(rows, paired, comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="validate completed artifacts without writing outputs")
    mode.add_argument("--generate", action="store_true", help="validate first, then generate all tables, figures, and reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        runs = validate()
        print(f"PASS: validated {len(runs)} completed runs, 201 predictions each, with all metrics matching.")
        if args.generate:
            generate(runs)
            print("PASS: generated four tables, six sorted CSVs, ten figures, and two Markdown reports.")
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
