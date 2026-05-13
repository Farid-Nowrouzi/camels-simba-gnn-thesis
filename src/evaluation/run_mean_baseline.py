from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible train/validation/test splitting.
    """
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Target loading
# ============================================================

def load_targets(target_csv: str | Path) -> pd.DataFrame:
    """
    Load Omega_m targets from the target inspection CSV.

    Accepted universe ID columns:
        universe_id
        universe

    Accepted target columns:
        omega_m
        Omega_m
        target
        omega_m_value
    """
    target_csv = Path(target_csv)

    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    df = pd.read_csv(target_csv)

    lower_columns = {col.lower(): col for col in df.columns}

    # ------------------------------------------------------------
    # Detect universe column
    # ------------------------------------------------------------
    universe_column = None

    for candidate in ["universe_id", "universe"]:
        if candidate in lower_columns:
            universe_column = lower_columns[candidate]
            break

    if universe_column is None:
        raise ValueError(
            f"Could not find a universe column in {target_csv}. "
            f"Expected one of ['universe_id', 'universe']. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------
    # Detect target column
    # ------------------------------------------------------------
    target_column = None

    for candidate in ["omega_m", "Omega_m", "target", "omega_m_value"]:
        if candidate.lower() in lower_columns:
            target_column = lower_columns[candidate.lower()]
            break

    if target_column is None:
        raise ValueError(
            f"Could not find an Omega_m target column in {target_csv}. "
            f"Expected one of ['omega_m', 'Omega_m', 'target', 'omega_m_value']. "
            f"Available columns: {list(df.columns)}"
        )

    clean_df = df[[universe_column, target_column]].copy()
    clean_df.columns = ["universe_id", "omega_m"]

    clean_df["universe_id"] = clean_df["universe_id"].astype(str)
    clean_df["omega_m"] = clean_df["omega_m"].astype(float)

    clean_df = clean_df.sort_values(
        by="universe_id",
        key=lambda col: col.str.replace("LH_", "", regex=False).astype(int),
    ).reset_index(drop=True)

    if clean_df["omega_m"].isna().any():
        raise ValueError("NaN values found in Omega_m targets.")

    return clean_df

# ============================================================
# Splitting
# ============================================================

def split_universes(
    universe_ids: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split universe IDs into train, validation, and test sets.
    Same logic style as the EvolveGCN training script.
    """
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1. Got {total}"
        )

    universe_ids = list(universe_ids)

    rng = random.Random(seed)
    rng.shuffle(universe_ids)

    n = len(universe_ids)

    if n < 5:
        raise ValueError("Need at least 5 universes for train/val/test splitting.")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = n - n_train - n_val

    if n_test < 1:
        n_train -= 1
        n_test = 1

    train_ids = universe_ids[:n_train]
    val_ids = universe_ids[n_train:n_train + n_val]
    test_ids = universe_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# ============================================================
# Metrics
# ============================================================

def compute_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, Any]:
    """
    Compute standard regression metrics.
    """
    errors = pred_values - true_values
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "num_samples": int(len(true_values)),
    }


def build_prediction_rows(
    df: pd.DataFrame,
    universe_ids: List[str],
    prediction_value: float,
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Create prediction rows for one split.
    """
    split_df = df[df["universe_id"].isin(universe_ids)].copy()

    rows = []

    for _, row in split_df.iterrows():
        true_value = float(row["omega_m"])
        pred_value = float(prediction_value)

        rows.append(
            {
                "split": split_name,
                "universe_id": row["universe_id"],
                "true_omega_m": true_value,
                "pred_omega_m": pred_value,
                "absolute_error": abs(pred_value - true_value),
                "squared_error": (pred_value - true_value) ** 2,
            }
        )

    return rows


def save_predictions_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """
    Save prediction rows to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "universe_id",
        "true_omega_m",
        "pred_omega_m",
        "absolute_error",
        "squared_error",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Baseline runner
# ============================================================

def run_mean_baseline(
    target_csv: str | Path,
    experiment_name: str,
    output_root: str | Path = "experiments",
    seed: int = 123,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Run mean baseline.

    The model predicts the mean Omega_m value from the training set
    for every train/validation/test sample.
    """
    set_seed(seed)

    target_csv = Path(target_csv)
    output_root = Path(output_root)
    experiment_dir = output_root / experiment_name
    predictions_dir = experiment_dir / "predictions"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("CAMELS-SIMBA Mean Baseline")
    print("=" * 90)
    print(f"Target CSV:       {target_csv}")
    print(f"Experiment name:  {experiment_name}")
    print(f"Experiment dir:   {experiment_dir}")
    print(f"Seed:             {seed}")
    print(f"Train ratio:      {train_ratio}")
    print(f"Val ratio:        {val_ratio}")
    print(f"Test ratio:       {test_ratio}")

    df = load_targets(target_csv)

    universe_ids = df["universe_id"].tolist()

    train_ids, val_ids, test_ids = split_universes(
        universe_ids=universe_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_df = df[df["universe_id"].isin(train_ids)].copy()
    val_df = df[df["universe_id"].isin(val_ids)].copy()
    test_df = df[df["universe_id"].isin(test_ids)].copy()

    train_mean = float(train_df["omega_m"].mean())

    print()
    print("Split summary")
    print("-" * 90)
    print(f"Total universes:  {len(df)}")
    print(f"Train universes:  {len(train_ids)}")
    print(f"Val universes:    {len(val_ids)}")
    print(f"Test universes:   {len(test_ids)}")
    print(f"Train mean Omega_m prediction: {train_mean:.8f}")

    print()
    print("Split IDs")
    print("-" * 90)
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs:   {val_ids}")
    print(f"Test IDs:  {test_ids}")

    train_pred = np.full(len(train_df), train_mean)
    val_pred = np.full(len(val_df), train_mean)
    test_pred = np.full(len(test_df), train_mean)

    train_metrics = compute_metrics(train_df["omega_m"].to_numpy(), train_pred)
    val_metrics = compute_metrics(val_df["omega_m"].to_numpy(), val_pred)
    test_metrics = compute_metrics(test_df["omega_m"].to_numpy(), test_pred)

    train_rows = build_prediction_rows(train_df, train_ids, train_mean, "train")
    val_rows = build_prediction_rows(val_df, val_ids, train_mean, "val")
    test_rows = build_prediction_rows(test_df, test_ids, train_mean, "test")

    all_rows = train_rows + val_rows + test_rows

    save_predictions_csv(all_rows, predictions_dir / "mean_baseline_predictions.csv")

    config = {
        "target_csv": str(target_csv),
        "experiment_name": experiment_name,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_total_universes": len(df),
        "num_train_universes": len(train_ids),
        "num_val_universes": len(val_ids),
        "num_test_universes": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "prediction_rule": "predict_training_set_mean_omega_m",
        "train_mean_omega_m": train_mean,
    }

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "train_mean_omega_m": train_mean,
    }

    save_json(config, experiment_dir / "config.json")
    save_json(metrics, experiment_dir / "metrics.json")

    print()
    print("Metrics")
    print("-" * 90)
    print(json.dumps(metrics, indent=2))

    print()
    print("=" * 90)
    print("MEAN BASELINE COMPLETE")
    print("=" * 90)
    print(f"Config:       {experiment_dir / 'config.json'}")
    print(f"Metrics:      {experiment_dir / 'metrics.json'}")
    print(f"Predictions:  {predictions_dir / 'mean_baseline_predictions.csv'}")

    return metrics


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mean Omega_m baseline for CAMELS-SIMBA targets."
    )

    parser.add_argument("--target_csv", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="experiments")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    args = parser.parse_args()

    run_mean_baseline(
        target_csv=args.target_csv,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()