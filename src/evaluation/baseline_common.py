"""Shared contracts for modern non-graph Omega_m baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PREDICTION_COLUMNS = [
    "universe_id",
    "true_omega_m",
    "pred_omega_m",
    "absolute_error",
    "squared_error",
]
SPLIT_FILENAMES = {
    "train": "train_predictions.csv",
    "val": "val_predictions.csv",
    "test": "test_predictions.csv",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prediction_rows(
    universe_ids: Sequence[str], true_values: Sequence[float], predictions: Sequence[float]
) -> list[dict[str, float | str]]:
    if not (len(universe_ids) == len(true_values) == len(predictions)):
        raise ValueError("Prediction IDs, targets, and predictions must have equal lengths.")
    rows = []
    for universe_id, true_value, prediction in zip(universe_ids, true_values, predictions):
        true_float = float(true_value)
        pred_float = float(prediction)
        rows.append({
            "universe_id": str(universe_id),
            "true_omega_m": true_float,
            "pred_omega_m": pred_float,
            "absolute_error": abs(pred_float - true_float),
            "squared_error": (pred_float - true_float) ** 2,
        })
    return rows


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Match the modern graph trainers' metrics, calculated from prediction rows."""
    if not rows:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "num_samples": 0}
    squared = np.asarray([float(row["squared_error"]) for row in rows], dtype=np.float64)
    absolute = np.asarray([float(row["absolute_error"]) for row in rows], dtype=np.float64)
    targets = np.asarray([float(row["true_omega_m"]) for row in rows], dtype=np.float64)
    mse = float(np.mean(squared))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2, dtype=np.float64))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(absolute)),
        "r2": float(1.0 - float(np.sum(squared)) / ss_tot) if ss_tot > 0.0 else float("nan"),
        "num_samples": len(rows),
    }


def read_prediction_csv(path: str | Path) -> list[dict[str, float | str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"Prediction schema mismatch in {path}: {reader.fieldnames}")
        return [{
            "universe_id": row["universe_id"],
            "true_omega_m": float(row["true_omega_m"]),
            "pred_omega_m": float(row["pred_omega_m"]),
            "absolute_error": float(row["absolute_error"]),
            "squared_error": float(row["squared_error"]),
        } for row in reader]


def write_prediction_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def git_provenance(repo: str | Path = ".") -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    status = git("status", "--porcelain")
    return {
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_working_tree": status not in {"", "unknown"},
    }


def required_artifacts(model_family: str) -> list[str]:
    files = ["config.json", "metrics.json"]
    if model_family != "mean":
        files.append("model.joblib")
    files.extend(f"predictions/{name}" for name in SPLIT_FILENAMES.values())
    return files


def completion_state(experiment_dir: str | Path, expected_config: Mapping[str, Any]) -> str:
    """Return missing/complete, or raise on partial/incompatible destinations."""
    path = Path(experiment_dir)
    if not path.exists():
        return "missing"
    if not path.is_dir():
        raise FileExistsError(f"Experiment destination is not a directory: {path}")
    config_path = path / "config.json"
    family = str(expected_config["model_family"])
    required = required_artifacts(family)
    present = [relative for relative in required if (path / relative).is_file()]
    if len(present) != len(required):
        raise RuntimeError(f"Partial experiment destination must not be overwritten: {path}")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    identity_fields = (
        "experiment_name", "model_family", "representation", "snapshot_protocol",
        "summary_definition_version", "top_n", "seed", "dataset_sha256",
        "split_manifest_sha256", "target", "target_normalization", "model_hyperparameters",
    )
    mismatches = [key for key in identity_fields if saved.get(key) != expected_config.get(key)]
    if mismatches:
        raise RuntimeError(f"Completed destination has incompatible config fields {mismatches}: {path}")
    return "complete"


def assert_finite_metrics(metrics: Mapping[str, Any]) -> None:
    for split in ("train", "val", "test"):
        for key in ("mse", "rmse", "mae", "r2"):
            if not np.isfinite(float(metrics[split][key])):
                raise ValueError(f"Non-finite metric {split}.{key}")
