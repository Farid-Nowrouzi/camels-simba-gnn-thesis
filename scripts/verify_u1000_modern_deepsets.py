#!/usr/bin/env python3
"""Verify completed modern U1000 raw-halo DeepSets artifacts."""

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

from src.evaluation.baseline_common import SPLIT_FILENAMES, compute_metrics, read_prediction_csv, sha256_file  # noqa: E402
from src.evaluation.run_modern_deepsets import (  # noqa: E402
    ARCHITECTURE_VERSION, FEATURE_NAMES, RawHaloSetDataset, build_run_config,
    expand_family_jobs, load_bound_data, load_checkpoint, load_family,
    make_loader, model_from_checkpoint, collect_predictions, required_artifacts,
)


METRIC_ATOL = 1e-12
PREDICTION_ATOL = 1e-7
PREDICTION_RTOL = 1e-6


def verify_run(job: Mapping[str, Any], experiment_dir: str | Path, bound_data=None,
               hashes_verified: bool = False) -> dict[str, Any]:
    path = Path(experiment_dir)
    bad = [name for name in required_artifacts() if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if bad:
        raise ValueError(f"Missing or empty artifacts in {path}: {bad}")
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected = build_run_config(job, REPO_ROOT, str(saved.get("device")))
    exact = ("experiment_name", "model_family", "model", "representation", "uses_graph_edges",
             "snapshot_protocol", "final_scale_factor", "top_n", "seed", "dataset_path",
             "dataset_sha256", "split_manifest_path", "split_manifest_sha256",
             "split_manifest_canonical_sha256", "split_counts", "feature_names", "feature_count",
             "feature_normalization", "target", "target_normalization", "pooling", "architecture",
             "trainable_parameters", "training")
    mismatches = [key for key in exact if saved.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(f"Scientific config mismatch in {path}: {mismatches}")
    if saved["representation"] != "raw_halo_set" or saved["uses_graph_edges"] is not False:
        raise ValueError(f"Graph-free representation contract violated in {path}")
    provenance = ("device", "python_version", "torch_version", "numpy_version", "git_branch", "git_head", "git_dirty_working_tree")
    if any(key not in saved for key in provenance):
        raise ValueError(f"Missing provenance fields in {path}")
    if not hashes_verified:
        if sha256_file(job["dataset_path"]) != job["dataset_sha256"]:
            raise ValueError(f"Dataset hash mismatch for {path}")
        if sha256_file(job["split_manifest_path"]) != job["split_manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch for {path}")
    data, manifest = bound_data or load_bound_data(job)
    checkpoint = load_checkpoint(path / "checkpoints/best_model.pt")
    identity = {"architecture_version": ARCHITECTURE_VERSION, "feature_names": FEATURE_NAMES,
                "dataset_sha256": job["dataset_sha256"], "split_manifest_sha256": job["split_manifest_sha256"],
                "seed": job["seed"], "top_n": job["top_n"]}
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Checkpoint identity mismatch in {path}")
    model = model_from_checkpoint(checkpoint)
    saved_metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if saved_metrics.get("best_epoch") != checkpoint.get("best_epoch") or not np.isclose(
        float(saved_metrics.get("best_validation_mse", np.nan)), float(checkpoint.get("best_validation_mse", np.nan)),
        rtol=0.0, atol=METRIC_ATOL,
    ):
        raise ValueError(f"Best-checkpoint metadata mismatch in {path}")
    for split, key, metric_key, count in (("train", "train_ids", "train", 700),
                                           ("val", "val_ids", "validation", 99),
                                           ("test", "test_ids", "test", 201)):
        rows = read_prediction_csv(path / "predictions" / SPLIT_FILENAMES[split])
        if len(rows) != count or [row["universe_id"] for row in rows] != list(manifest[key]):
            raise ValueError(f"Prediction ID/count mismatch: {path} {split}")
        numerical = np.asarray([[row[column] for column in ("true_omega_m", "pred_omega_m", "absolute_error", "squared_error")]
                                for row in rows], dtype=np.float64)
        if not np.isfinite(numerical).all():
            raise ValueError(f"Non-finite predictions: {path} {split}")
        error = numerical[:, 1] - numerical[:, 0]
        if not np.allclose(numerical[:, 2], np.abs(error), rtol=0.0, atol=METRIC_ATOL) or not np.allclose(
            numerical[:, 3], error ** 2, rtol=0.0, atol=METRIC_ATOL,
        ):
            raise ValueError(f"Prediction error columns inconsistent: {path} {split}")
        dataset_targets = np.asarray([float(torch.as_tensor(data[item]["target"]).reshape(-1)[0]) for item in manifest[key]])
        if not np.allclose(numerical[:, 0], dataset_targets, rtol=0.0, atol=METRIC_ATOL):
            raise ValueError(f"True targets mismatch dataset: {path} {split}")
        reproduced = collect_predictions(model, make_loader(data, manifest[key], False), torch.device("cpu"))
        reproduced_values = np.asarray([row["pred_omega_m"] for row in reproduced])
        if not np.allclose(numerical[:, 1], reproduced_values, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL):
            raise ValueError(f"Checkpoint predictions do not reproduce: {path} {split}")
        recalculated = compute_metrics(rows)
        for metric in ("mse", "rmse", "mae", "r2"):
            if not np.isclose(float(saved_metrics[metric_key][metric]), float(recalculated[metric]), rtol=0.0, atol=METRIC_ATOL):
                raise ValueError(f"Metric mismatch: {path} {metric_key}.{metric}")
    with (path / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["epoch", "train_mse", "validation_mse", "validation_rmse", "validation_mae",
                                "learning_rate", "improved", "patience_counter"] or not list(reader):
            raise ValueError(f"Invalid training log: {path}")
    return {"status": "verified", "experiment": job["experiment_name"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", default="configs/experiment_families/u1000_modern_deepsets.json")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    jobs = expand_family_jobs(load_family(args.family_spec), REPO_ROOT)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
    if len(jobs) not in (1, 12):
        raise ValueError("Verifier must select one exact job or the complete 12-job family.")
    cache = {}
    for job in jobs:
        key = str(job["dataset_path"])
        bound = load_bound_data(job, cache.get(key))
        cache[key] = bound[0]
        result = verify_run(job, Path(args.experiments_root) / job["experiment_name"], bound, True)
        print(f"VERIFIED: {result['experiment']}")
    print(f"VERIFIED {len(jobs)}/{len(jobs)} modern DeepSets experiments.")


if __name__ == "__main__":
    main()
