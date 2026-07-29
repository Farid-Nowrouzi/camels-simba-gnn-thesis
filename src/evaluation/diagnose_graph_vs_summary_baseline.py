from __future__ import annotations

"""
diagnose_graph_vs_summary_baseline.py

Pure diagnostic comparing a saved EvolveGCN-H experiment against summary-feature
baselines on the exact same dataset and train/validation/test split.

This script does not train the GNN, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads an existing
experiment config.json, reuses dataset_path and split IDs, computes temporal
summary features in-memory, fits classical baselines on TRAIN only, and
evaluates on train/validation/test in original Omega_m scale.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - depends on local sklearn install
    MLPRegressor = None

from src.evaluation.diagnose_evolvegcn_h_representations import (
    load_json,
    save_json,
)
from src.evaluation.run_summary_feature_baseline import (
    build_arrays_for_ids,
    extract_temporal_summary,
    get_target,
    validate_split_ids,
)


TARGET_COLUMNS = ["true_omega_m", "target", "y_true", "true", "omega_m"]
PREDICTION_COLUMNS = ["pred_omega_m", "prediction", "y_pred", "pred"]


def load_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "pearson": pearson_corr(predictions, targets),
    }


def build_feature_target_arrays(
    data: Dict[str, Dict[str, Any]],
    universe_ids: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    features_by_id = {}
    targets_by_id = {}

    for universe_id in universe_ids:
        sample = data[universe_id]
        features_by_id[universe_id] = extract_temporal_summary(sample)
        targets_by_id[universe_id] = get_target(sample)

    return features_by_id, targets_by_id


def find_column(fieldnames: List[str], candidates: List[str]) -> str | None:
    lower_to_original = {field.lower(): field for field in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    return None


def load_prediction_metrics(prediction_path: Path) -> Dict[str, float] | None:
    if not prediction_path.exists():
        return None

    with prediction_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None

        target_column = find_column(reader.fieldnames, TARGET_COLUMNS)
        prediction_column = find_column(reader.fieldnames, PREDICTION_COLUMNS)

        if target_column is None or prediction_column is None:
            return None

        targets = []
        predictions = []
        for row in reader:
            targets.append(float(row[target_column]))
            predictions.append(float(row[prediction_column]))

    return regression_metrics(
        predictions=np.asarray(predictions, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
    )


def derive_saved_metric_r2(
    split_metrics: Dict[str, Any],
    targets: np.ndarray,
) -> float:
    if "r2" in split_metrics:
        return float(split_metrics["r2"])

    if "mse" in split_metrics:
        mse = float(split_metrics["mse"])
    elif "rmse" in split_metrics:
        mse = float(split_metrics["rmse"]) ** 2
    else:
        return float("nan")

    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    if target_variance == 0.0:
        return float("nan")

    return 1.0 - mse / target_variance


def load_saved_gnn_metrics(
    experiment_dir: Path,
    targets_by_split: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    metrics_path = experiment_dir / "metrics.json"
    predictions_dir = experiment_dir / "predictions"

    if not metrics_path.exists():
        return {
            "metrics_path": str(metrics_path),
            "available": False,
            "splits": {},
        }

    with metrics_path.open("r", encoding="utf-8") as f:
        raw_metrics = json.load(f)

    splits = {}
    for split in ["train", "val", "test"]:
        prediction_metrics = load_prediction_metrics(
            predictions_dir / f"{split}_predictions.csv"
        )

        if prediction_metrics is not None:
            splits[split] = {
                **prediction_metrics,
                "source": str(predictions_dir / f"{split}_predictions.csv"),
            }
            continue

        saved_split_metrics = raw_metrics.get(split, {})
        if not isinstance(saved_split_metrics, dict):
            saved_split_metrics = {}

        splits[split] = {
            "r2": derive_saved_metric_r2(
                split_metrics=saved_split_metrics,
                targets=targets_by_split[split],
            ),
            "mae": (
                float(saved_split_metrics["mae"])
                if "mae" in saved_split_metrics
                else float("nan")
            ),
            "rmse": (
                float(saved_split_metrics["rmse"])
                if "rmse" in saved_split_metrics
                else float("nan")
            ),
            "pearson": (
                float(saved_split_metrics["pearson"])
                if "pearson" in saved_split_metrics
                else float("nan")
            ),
            "source": str(metrics_path),
        }

    return {
        "metrics_path": str(metrics_path),
        "available": True,
        "raw_metrics": raw_metrics,
        "splits": splits,
    }


def build_models(seed: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
        ),
    }

    if MLPRegressor is not None:
        models["MLPRegressor"] = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed,
            ),
        )

    return models


def evaluate_baseline(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    model.fit(X_train, y_train)

    results = {}
    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        predictions = model.predict(X)
        results[split] = regression_metrics(
            predictions=np.asarray(predictions, dtype=np.float64),
            targets=np.asarray(y, dtype=np.float64),
        )

    return results


def format_metric(value: float) -> str:
    if value != value:
        return "nan"

    return f"{value:.6f}"


def table_row(
    model_name: str,
    results: Dict[str, Dict[str, float]],
) -> str:
    return (
        f"{model_name:<24}"
        f"{format_metric(results['train']['r2']):>12}"
        f"{format_metric(results['val']['r2']):>12}"
        f"{format_metric(results['test']['r2']):>12}"
        f"{format_metric(results['test']['mae']):>14}"
        f"{format_metric(results['test']['rmse']):>14}"
        f"{format_metric(results['test']['pearson']):>16}"
    )


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("GRAPH MODEL VS SUMMARY-FEATURE BASELINES")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Dataset:        {output['dataset_path']}")
    print(f"Feature dim:    {output['num_summary_features']}")
    print()
    print(
        f"{'Model':<24}"
        f"{'Train R2':>12}"
        f"{'Val R2':>12}"
        f"{'Test R2':>12}"
        f"{'Test MAE':>14}"
        f"{'Test RMSE':>14}"
        f"{'Test Pearson':>16}"
    )
    print("-" * 112)

    if output["saved_gnn_metrics"]["available"]:
        print(
            table_row(
                "EvolveGCN-H saved",
                output["saved_gnn_metrics"]["splits"],
            )
        )

    for model_name, results in output["summary_baselines"].items():
        print(table_row(model_name, results))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved EvolveGCN-H metrics with summary-feature baselines."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_path = (
        Path(args.config_path)
        if args.config_path is not None
        else experiment_dir / "config.json"
    )
    config = load_json(config_path)
    dataset_path = Path(
        args.dataset_path if args.dataset_path is not None else config["dataset_path"]
    )
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))

    data = load_dataset(dataset_path)
    universe_ids = sorted(data.keys())

    required_split_keys = ["train_ids", "val_ids", "test_ids"]
    for key in required_split_keys:
        if key not in config:
            raise KeyError(f"Config does not contain split key: {key}")

    train_ids = list(config["train_ids"])
    val_ids = list(config["val_ids"])
    test_ids = list(config["test_ids"])

    validate_split_ids(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        dataset_ids=universe_ids,
        split_config=config,
    )

    features_by_id, targets_by_id = build_feature_target_arrays(
        data=data,
        universe_ids=universe_ids,
    )

    X_train, y_train = build_arrays_for_ids(train_ids, features_by_id, targets_by_id)
    X_val, y_val = build_arrays_for_ids(val_ids, features_by_id, targets_by_id)
    X_test, y_test = build_arrays_for_ids(test_ids, features_by_id, targets_by_id)

    targets_by_split = {
        "train": y_train.astype(np.float64),
        "val": y_val.astype(np.float64),
        "test": y_test.astype(np.float64),
    }

    saved_gnn_metrics = load_saved_gnn_metrics(
        experiment_dir=experiment_dir,
        targets_by_split=targets_by_split,
    )

    summary_baselines = {}
    skipped_models = {}

    for model_name, model in build_models(seed=seed).items():
        try:
            summary_baselines[model_name] = evaluate_baseline(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
            )
        except Exception as exc:  # pragma: no cover - diagnostic robustness
            skipped_models[model_name] = repr(exc)

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "seed": seed,
        "split_source": str(config_path),
        "num_universes": len(universe_ids),
        "num_summary_features": int(X_train.shape[1]),
        "split_sizes": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "target_summary": {
            split: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for split, values in targets_by_split.items()
        },
        "saved_gnn_metrics": saved_gnn_metrics,
        "summary_baselines": summary_baselines,
        "skipped_models": skipped_models,
        "interpretation": {
            "summary_beats_gnn": (
                "If one or more summary baselines clearly exceed the saved GNN "
                "test metrics on this same split, the graph model/representation "
                "is failing to use signal available in the current temporal dataset."
            ),
            "summary_also_fails": (
                "If summary baselines also perform poorly on this same split, the "
                "top500 temporal representation may lack stable Omega_m signal for "
                "this split or feature set."
            ),
        },
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "graph_vs_summary_baseline.json"
    )
    save_json(output, output_path)

    print_summary(output)
    if skipped_models:
        print()
        print("Skipped models")
        print("-" * 112)
        for model_name, reason in skipped_models.items():
            print(f"{model_name}: {reason}")
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
