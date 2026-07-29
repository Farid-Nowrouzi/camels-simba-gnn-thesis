from __future__ import annotations

"""
diagnose_prediction_collapse_summary.py

Compare prediction collapse across saved experiment prediction CSVs.

This script is diagnostic only. It does not modify training, preprocessing,
datasets, splits, model architecture, checkpoints, or experiment predictions.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


TARGET_CANDIDATES = [
    "true_omega_m",
    "target",
    "y_true",
    "true",
    "omega_m",
]
PREDICTION_CANDIDATES = [
    "pred_omega_m",
    "prediction",
    "y_pred",
    "pred",
]
UNIVERSE_ID_CANDIDATES = [
    "universe_id",
    "id",
    "uid",
    "universe",
]


def find_column(fieldnames: List[str], candidates: List[str]) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find any of {candidates} in columns: {fieldnames}"
    )


def find_prediction_csv(experiment_dir: Path) -> Path:
    predictions_dir = experiment_dir / "predictions"

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    preferred = predictions_dir / "test_predictions.csv"
    if preferred.exists():
        return preferred

    csv_paths = sorted(predictions_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {predictions_dir}")

    return csv_paths[0]


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")

    return float(value)


def load_prediction_rows(csv_path: Path) -> tuple[List[Dict[str, str]], str, str, str | None]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        target_col = find_column(fieldnames, TARGET_CANDIDATES)
        prediction_col = find_column(fieldnames, PREDICTION_CANDIDATES)

        universe_id_col = None
        try:
            universe_id_col = find_column(fieldnames, UNIVERSE_ID_CANDIDATES)
        except ValueError:
            universe_id_col = None

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no prediction rows: {csv_path}")

    return rows, target_col, prediction_col, universe_id_col


def pearson_corr(targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(targets) < 2:
        return float("nan")

    if float(np.std(targets)) == 0.0 or float(np.std(predictions)) == 0.0:
        return float("nan")

    return float(np.corrcoef(targets, predictions)[0, 1])


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    ss_res = float(np.sum((targets - predictions) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))

    if ss_tot == 0.0:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def collapse_diagnosis(std_ratio: float) -> str:
    if math.isnan(std_ratio):
        return "unknown"

    if std_ratio < 0.25:
        return "Strong mean collapse"

    if std_ratio < 0.50:
        return "Partial mean collapse"

    return "Not heavily collapsed"


def analyze_experiment(experiment_dir: Path) -> Dict[str, Any]:
    csv_path = find_prediction_csv(experiment_dir)
    rows, target_col, prediction_col, universe_id_col = load_prediction_rows(csv_path)

    targets = np.asarray(
        [safe_float(row[target_col]) for row in rows],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [safe_float(row[prediction_col]) for row in rows],
        dtype=np.float64,
    )

    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    target_std = float(np.std(targets))
    prediction_std = float(np.std(predictions))
    std_ratio = (
        prediction_std / target_std
        if target_std != 0.0
        else float("nan")
    )

    preview_rows = []
    for index, row in enumerate(rows[:20]):
        universe_id = (
            row[universe_id_col]
            if universe_id_col is not None
            else str(index)
        )
        target = float(targets[index])
        prediction = float(predictions[index])
        preview_rows.append(
            {
                "universe_id": universe_id,
                "target": target,
                "prediction": prediction,
                "error": prediction - target,
            }
        )

    return {
        "experiment_dir": str(experiment_dir),
        "prediction_csv": str(csv_path),
        "target_column": target_col,
        "prediction_column": prediction_col,
        "universe_id_column": universe_id_col,
        "num_samples": int(len(rows)),
        "target_mean": float(np.mean(targets)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": prediction_std,
        "prediction_std_over_target_std": float(std_ratio),
        "target_range": [
            float(np.min(targets)),
            float(np.max(targets)),
        ],
        "prediction_range": [
            float(np.min(predictions)),
            float(np.max(predictions)),
        ],
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "pearson_correlation": pearson_corr(targets, predictions),
        "r2": r2_score(targets, predictions),
        "diagnosis": collapse_diagnosis(float(std_ratio)),
        "first_20_rows": preview_rows,
    }


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    print("=" * 150)
    print("PREDICTION COLLAPSE COMPARISON")
    print("=" * 150)
    print(
        f"{'Experiment':<48}"
        f"{'N':>5}"
        f"{'Target Std':>12}"
        f"{'Pred Std':>12}"
        f"{'Std Ratio':>11}"
        f"{'MAE':>11}"
        f"{'RMSE':>11}"
        f"{'Pearson':>11}"
        f"{'R2':>11}"
        f"{'Diagnosis':>24}"
    )
    print("-" * 150)

    for result in results:
        name = Path(result["experiment_dir"]).name
        print(
            f"{name:<48.48}"
            f"{result['num_samples']:>5}"
            f"{result['target_std']:>12.6f}"
            f"{result['prediction_std']:>12.6f}"
            f"{result['prediction_std_over_target_std']:>11.4f}"
            f"{result['mae']:>11.6f}"
            f"{result['rmse']:>11.6f}"
            f"{result['pearson_correlation']:>11.4f}"
            f"{result['r2']:>11.4f}"
            f"{result['diagnosis']:>24}"
        )

    print("-" * 150)


def print_preview_rows(result: Dict[str, Any]) -> None:
    print()
    print(Path(result["experiment_dir"]).name)
    print("-" * 90)
    print(
        f"{'universe_id':<18}"
        f"{'target':>14}"
        f"{'prediction':>14}"
        f"{'error':>14}"
    )
    print("-" * 90)

    for row in result["first_20_rows"]:
        print(
            f"{str(row['universe_id']):<18}"
            f"{row['target']:>14.6f}"
            f"{row['prediction']:>14.6f}"
            f"{row['error']:>14.6f}"
        )


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mean-prediction collapse across experiments."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Experiment directories containing predictions/*.csv.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="experiments/prediction_collapse_comparison.json",
    )
    parser.add_argument(
        "--show_rows",
        action="store_true",
        help="Print first 20 prediction rows for each experiment.",
    )

    args = parser.parse_args()

    experiment_dirs = [Path(path) for path in args.experiment_dirs]
    results = [analyze_experiment(path) for path in experiment_dirs]

    output = {
        "num_experiments": len(results),
        "results": results,
    }

    print_comparison_table(results)

    if args.show_rows:
        for result in results:
            print_preview_rows(result)

    output_path = Path(args.output_path)
    save_json(output, output_path)

    print()
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
