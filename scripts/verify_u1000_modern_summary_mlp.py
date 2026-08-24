#!/usr/bin/env python3
"""Verify one or all modern U1000 final-snapshot Summary MLP artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baseline_common import (  # noqa: E402
    SPLIT_FILENAMES,
    compute_metrics,
    read_prediction_csv,
    sha256_file,
)
from src.evaluation.run_modern_summary_baselines import load_bound_data, load_summary_data  # noqa: E402
from src.evaluation.run_modern_summary_mlp import (  # noqa: E402
    ARCHITECTURE_VERSION,
    INPUT_DIM,
    build_run_config,
    expand_family_jobs,
    fit_feature_scaler,
    load_checkpoint,
    load_family,
    model_from_checkpoint,
    predict,
    required_artifacts,
    transform_features,
)
from src.evaluation.summary_features import SUMMARY_FEATURE_NAMES, arrays_for_ids  # noqa: E402


METRIC_ATOL = 1e-12
PREDICTION_ATOL = 1e-7
PREDICTION_RTOL = 1e-6


def _close(left: float, right: float, tolerance: float = METRIC_ATOL) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance, equal_nan=False))


def verify_run(
    job: Mapping[str, Any],
    experiment_dir: str | Path,
    bound_data: tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]] | None = None,
    hashes_verified: bool = False,
) -> dict[str, Any]:
    path = Path(experiment_dir)
    bad_artifacts = [
        relative for relative in required_artifacts()
        if not (path / relative).is_file() or (path / relative).stat().st_size == 0
    ]
    if bad_artifacts:
        raise ValueError(f"Missing or empty artifacts in {path}: {bad_artifacts}")
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected = build_run_config(job, REPO_ROOT, str(saved.get("device")))
    exact_fields = (
        "experiment_name", "model_family", "representation", "summary_definition_version",
        "feature_names", "feature_count", "snapshot_protocol", "final_scale_factor",
        "top_n", "seed", "dataset_path", "dataset_sha256", "split_manifest_path",
        "split_manifest_sha256", "split_manifest_canonical_sha256", "split_counts",
        "target", "target_normalization", "feature_scaling", "architecture", "training",
    )
    mismatches = [key for key in exact_fields if saved.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(f"Scientific config mismatch in {path}: {mismatches}")
    provenance = ("device", "python_version", "torch_version", "numpy_version", "git_branch", "git_head", "git_dirty_working_tree")
    missing_provenance = [key for key in provenance if key not in saved]
    if missing_provenance:
        raise ValueError(f"Missing provenance fields in {path}: {missing_provenance}")
    if not hashes_verified:
        if sha256_file(job["dataset_path"]) != job["dataset_sha256"]:
            raise ValueError(f"Dataset hash mismatch for {path}")
        if sha256_file(job["split_manifest_path"]) != job["split_manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch for {path}")

    ids, features, targets, manifest = bound_data or load_bound_data(job)
    if features.shape != (1000, INPUT_DIM) or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"Invalid shared summary matrix for {path}: {features.shape}")
    if not np.array_equal(features[:, 0], np.full(1000, int(job["top_n"]), dtype=np.float64)):
        raise ValueError(f"Final-snapshot halo counts do not match Top-N for {path}")
    train_x, _ = arrays_for_ids(manifest["train_ids"], ids, features, targets)
    expected_mean, expected_scale = fit_feature_scaler(train_x)
    checkpoint = load_checkpoint(path / "checkpoints" / "best_model.pt")
    checkpoint_fields = (
        "model_state_dict", "scaler_mean", "scaler_scale", "best_epoch",
        "best_validation_mse", "architecture_version", "feature_names",
        "dataset_sha256", "split_manifest_sha256", "seed", "top_n",
    )
    missing_checkpoint = [key for key in checkpoint_fields if key not in checkpoint]
    if missing_checkpoint:
        raise ValueError(f"Checkpoint fields missing in {path}: {missing_checkpoint}")
    identity = {
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_names": SUMMARY_FEATURE_NAMES,
        "dataset_sha256": job["dataset_sha256"],
        "split_manifest_sha256": job["split_manifest_sha256"],
        "seed": job["seed"],
        "top_n": job["top_n"],
    }
    identity_mismatches = [key for key, value in identity.items() if checkpoint.get(key) != value]
    if identity_mismatches:
        raise ValueError(f"Checkpoint identity mismatch in {path}: {identity_mismatches}")
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    if not np.array_equal(scaler_mean, expected_mean) or not np.array_equal(scaler_scale, expected_scale):
        raise ValueError(f"Checkpoint scaler does not match the training rows: {path}")
    model = model_from_checkpoint(checkpoint)

    saved_metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if int(saved_metrics.get("best_epoch", -1)) != int(checkpoint["best_epoch"]):
        raise ValueError(f"Best epoch mismatch in {path}")
    if not _close(float(saved_metrics.get("best_validation_mse", np.nan)), float(checkpoint["best_validation_mse"])):
        raise ValueError(f"Best validation MSE mismatch in {path}")
    for split, manifest_key, metric_key in (
        ("train", "train_ids", "train"),
        ("val", "val_ids", "validation"),
        ("test", "test_ids", "test"),
    ):
        rows = read_prediction_csv(path / "predictions" / SPLIT_FILENAMES[split])
        expected_ids = list(manifest[manifest_key])
        if [row["universe_id"] for row in rows] != expected_ids:
            raise ValueError(f"Prediction ID order mismatch: {path} {split}")
        expected_count = {"train": 700, "val": 99, "test": 201}[split]
        if len(rows) != expected_count or len(expected_ids) != expected_count:
            raise ValueError(f"Prediction count mismatch: {path} {split}")
        numerical = np.asarray([
            [row["true_omega_m"], row["pred_omega_m"], row["absolute_error"], row["squared_error"]]
            for row in rows
        ], dtype=np.float64)
        if not np.isfinite(numerical).all():
            raise ValueError(f"Non-finite prediction values: {path} {split}")
        errors = numerical[:, 1] - numerical[:, 0]
        if not np.allclose(numerical[:, 2], np.abs(errors), rtol=0.0, atol=METRIC_ATOL):
            raise ValueError(f"Absolute errors are inconsistent: {path} {split}")
        if not np.allclose(numerical[:, 3], errors ** 2, rtol=0.0, atol=METRIC_ATOL):
            raise ValueError(f"Squared errors are inconsistent: {path} {split}")
        split_x, split_y = arrays_for_ids(expected_ids, ids, features, targets)
        if not np.allclose(numerical[:, 0], split_y, rtol=0.0, atol=METRIC_ATOL):
            raise ValueError(f"Saved targets do not match the dataset: {path} {split}")
        scaled = transform_features(split_x, scaler_mean, scaler_scale)
        reloaded = predict(model, scaled, torch.device("cpu"))
        if not np.allclose(numerical[:, 1], reloaded, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL):
            raise ValueError(f"Reloaded checkpoint predictions do not reproduce: {path} {split}")
        recalculated = compute_metrics(rows)
        for metric in ("mse", "rmse", "mae", "r2"):
            if not _close(float(saved_metrics[metric_key][metric]), float(recalculated[metric])):
                raise ValueError(f"Metric mismatch: {path} {metric_key}.{metric}")
        if int(saved_metrics[metric_key]["num_samples"]) != expected_count:
            raise ValueError(f"Metric sample count mismatch: {path} {metric_key}")

    with (path / "train_log.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = [
            "epoch", "train_mse", "validation_mse", "validation_rmse",
            "validation_mae", "learning_rate", "improved", "patience_counter",
        ]
        if reader.fieldnames != required_columns or not list(reader):
            raise ValueError(f"Invalid or empty training log: {path}")
    return {"experiment_name": job["experiment_name"], "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", default="configs/experiment_families/u1000_modern_summary_mlp.json")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--experiment-name", help="Verify one exact family member; default verifies all 12.")
    args = parser.parse_args()
    family = load_family(args.family_spec)
    jobs = expand_family_jobs(family, REPO_ROOT)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
        if not jobs:
            raise ValueError(f"Experiment is not an exact family member: {args.experiment_name}")
    summary_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for job in jobs:
        key = str(job["dataset_path"])
        if key not in summary_cache:
            summary_cache[key] = load_summary_data(job)
        bound = load_bound_data(job, summary_data=summary_cache[key])
        result = verify_run(job, Path(args.experiments_root) / job["experiment_name"], bound, hashes_verified=True)
        print(f"VERIFIED {result['experiment_name']}")
    print(f"Verified {len(jobs)} exact modern Summary MLP artifacts.")


if __name__ == "__main__":
    main()
