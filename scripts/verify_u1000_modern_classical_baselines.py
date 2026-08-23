#!/usr/bin/env python3
"""Verify one or all modern U1000 classical baseline artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baseline_common import (  # noqa: E402
    SPLIT_FILENAMES,
    compute_metrics,
    read_prediction_csv,
    required_artifacts,
    sha256_file,
)
from src.evaluation.run_modern_summary_baselines import (  # noqa: E402
    build_run_config,
    expand_family_jobs,
    load_bound_data,
    load_family,
    load_summary_data,
)
from src.evaluation.summary_features import arrays_for_ids  # noqa: E402


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance, equal_nan=True))


def verify_run(
    job: Mapping[str, Any], experiment_dir: str | Path,
    bound_data: tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]] | None = None,
    hashes_verified: bool = False,
) -> dict[str, Any]:
    path = Path(experiment_dir)
    expected = build_run_config(job, REPO_ROOT)
    missing = [relative for relative in required_artifacts(str(job["model_family"])) if not (path / relative).is_file()]
    if missing:
        raise ValueError(f"Missing artifacts in {path}: {missing}")
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    exact_fields = (
        "experiment_name", "model_family", "representation", "snapshot_protocol",
        "final_scale_factor", "summary_definition_version", "top_n", "seed",
        "dataset_path", "dataset_sha256", "split_manifest_path", "split_manifest_sha256",
        "feature_names", "feature_count", "target", "target_normalization", "model_hyperparameters",
    )
    mismatches = [key for key in exact_fields if saved.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(f"Scientific config mismatch in {path}: {mismatches}")
    provenance_fields = ("python_version", "sklearn_version", "git_branch", "git_head", "git_dirty_working_tree")
    missing_provenance = [key for key in provenance_fields if key not in saved]
    if missing_provenance:
        raise ValueError(f"Missing provenance fields in {path}: {missing_provenance}")
    if not hashes_verified:
        if sha256_file(job["dataset_path"]) != job["dataset_sha256"]:
            raise ValueError(f"Dataset hash mismatch for {path}")
        if sha256_file(job["split_manifest_path"]) != job["split_manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch for {path}")
    ids, features, targets, manifest = bound_data or load_bound_data(job)
    saved_metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    model = None if job["model_family"] == "mean" else joblib.load(path / "model.joblib")
    for split, manifest_key in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
        rows = read_prediction_csv(path / "predictions" / SPLIT_FILENAMES[split])
        expected_ids = list(manifest[manifest_key])
        if [row["universe_id"] for row in rows] != expected_ids:
            raise ValueError(f"Prediction ID order mismatch: {path} {split}")
        if len(rows) != {"train": 700, "val": 99, "test": 201}[split]:
            raise ValueError(f"Prediction count mismatch: {path} {split}")
        for row in rows:
            error = float(row["pred_omega_m"]) - float(row["true_omega_m"])
            if not _close(float(row["absolute_error"]), abs(error)) or not _close(float(row["squared_error"]), error ** 2):
                raise ValueError(f"Stored prediction errors are inconsistent: {path} {split}")
        recalculated = compute_metrics(rows)
        for metric in ("mse", "rmse", "mae", "r2"):
            if not np.isfinite(float(saved_metrics[split][metric])):
                raise ValueError(f"Non-finite saved metric: {path} {split}.{metric}")
            if not _close(float(saved_metrics[split][metric]), float(recalculated[metric])):
                raise ValueError(f"Metric mismatch: {path} {split}.{metric}")
        if int(saved_metrics[split]["num_samples"]) != len(rows):
            raise ValueError(f"Metric count mismatch: {path} {split}")
        X, y = arrays_for_ids(expected_ids, ids, features, targets)
        stored_true = np.asarray([row["true_omega_m"] for row in rows], dtype=np.float64)
        if not np.allclose(stored_true, y, rtol=0.0, atol=1e-12):
            raise ValueError(f"Stored targets do not reproduce dataset targets: {path} {split}")
        expected_predictions = (
            np.full(len(y), float(saved["training_target_mean"]), dtype=np.float64)
            if model is None else np.asarray(model.predict(X), dtype=np.float64)
        )
        stored_predictions = np.asarray([row["pred_omega_m"] for row in rows], dtype=np.float64)
        if not np.allclose(stored_predictions, expected_predictions, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Reloaded-model predictions do not reproduce: {path} {split}")
    return {"experiment_name": job["experiment_name"], "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", default="configs/experiment_families/u1000_modern_classical_baselines.json")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--experiment-name", help="Verify one family run; default verifies all 39.")
    args = parser.parse_args()
    family = load_family(args.family_spec)
    jobs = expand_family_jobs(family)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
        if not jobs:
            raise ValueError(f"Experiment is not an exact family member: {args.experiment_name}")
    summary_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for job in jobs:
        key = str(job["dataset_path"])
        if key not in summary_cache:
            summary_cache[key] = load_summary_data(job)
        bound_data = load_bound_data(job, summary_data=summary_cache[key])
        result = verify_run(
            job, Path(args.experiments_root) / job["experiment_name"], bound_data,
            hashes_verified=True,
        )
        print(f"VERIFIED {result['experiment_name']}")
    print(f"Verified {len(jobs)} exact modern classical baseline artifacts.")


if __name__ == "__main__":
    main()
