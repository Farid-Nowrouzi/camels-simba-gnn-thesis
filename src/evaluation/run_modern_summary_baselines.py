"""Run the manifest-bound modern classical final-snapshot baseline family."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.baseline_common import (
    SPLIT_FILENAMES,
    build_prediction_rows,
    completion_state,
    compute_metrics,
    git_provenance,
    sha256_file,
    write_json,
    write_prediction_csv,
)
from src.evaluation.summary_features import (
    SUMMARY_DEFINITION_VERSION,
    SUMMARY_FEATURE_NAMES,
    arrays_for_ids,
    extract_dataset_summaries,
    load_processed_dataset,
)
from src.training.split_manifest import load_split_manifest


MODEL_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "mean": {"statistic": "training_target_mean"},
    "ridge": {"scaler": "StandardScaler", "alpha": 1.0},
    "random_forest": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "n_jobs": -1,
    },
    "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3},
}


def build_model(model_family: str, seed: int) -> Any:
    if model_family == "ridge":
        return Pipeline([("standard_scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        )
    if model_family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=3, random_state=seed,
        )
    if model_family == "mean":
        return None
    raise ValueError(f"Unsupported model family: {model_family}")


def experiment_name(model_family: str, seed: int, top_n: int | None = None) -> str:
    if model_family == "mean":
        return f"mean_baseline_u1000_train700_seed{seed}_targetmean"
    if top_n is None:
        raise ValueError("top_n is required for fitted models.")
    names = {
        "ridge": f"ridge_u1000_top{top_n}_summary20_final_train700_seed{seed}_trainzscore_alpha1",
        "random_forest": f"random_forest_u1000_top{top_n}_summary20_final_train700_seed{seed}_none_t300_leaf2",
        "gradient_boosting": f"gradient_boosting_u1000_top{top_n}_summary20_final_train700_seed{seed}_none_t300_lr0p03_d3",
    }
    try:
        return names[model_family]
    except KeyError:
        raise ValueError(f"Unsupported model family: {model_family}") from None


def load_family(path: str | Path) -> dict[str, Any]:
    family = json.loads(Path(path).read_text(encoding="utf-8"))
    if family.get("schema_version") != "1.0":
        raise ValueError("Unsupported modern baseline family schema.")
    if family.get("snapshot_protocol") != "final":
        raise ValueError("This runner only supports the final-snapshot protocol.")
    if family.get("summary_definition_version") != SUMMARY_DEFINITION_VERSION:
        raise ValueError("Family summary definition does not match the implementation.")
    if family.get("seeds") != [42, 123, 2025] or family.get("top_n") != [500, 1000, 1500, 2000]:
        raise ValueError("Family must declare the controlled U1000 Top-N/seed matrix.")
    if family.get("models") != ["ridge", "random_forest", "gradient_boosting"]:
        raise ValueError("Family must contain exactly Ridge, RF, and Gradient Boosting.")
    return family


def expand_family_jobs(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    canonical = str(family["mean_baseline"]["canonical_dataset_top_n"])
    for seed in family["seeds"]:
        source = family["datasets"][canonical]
        jobs.append({
            "model_family": "mean", "top_n": None, "seed": seed,
            "experiment_name": experiment_name("mean", seed),
            "dataset_path": source["path"], "dataset_sha256": source["sha256"],
            "split_manifest_path": source["split_manifests"][str(seed)]["path"],
            "split_manifest_sha256": source["split_manifests"][str(seed)]["sha256"],
        })
    for top_n in family["top_n"]:
        source = family["datasets"][str(top_n)]
        for seed in family["seeds"]:
            for model_family in family["models"]:
                jobs.append({
                    "model_family": model_family, "top_n": top_n, "seed": seed,
                    "experiment_name": experiment_name(model_family, seed, top_n),
                    "dataset_path": source["path"], "dataset_sha256": source["sha256"],
                    "split_manifest_path": source["split_manifests"][str(seed)]["path"],
                    "split_manifest_sha256": source["split_manifests"][str(seed)]["sha256"],
                })
    if len(jobs) != 39:
        raise AssertionError(f"Expected 39 family jobs, got {len(jobs)}")
    return jobs


def build_run_config(job: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    hyperparameters = dict(MODEL_HYPERPARAMETERS[str(job["model_family"])])
    if job["model_family"] in {"random_forest", "gradient_boosting"}:
        hyperparameters["random_state"] = int(job["seed"])
    return {
        "schema_version": "1.0",
        "experiment_name": job["experiment_name"],
        "model_family": job["model_family"],
        "representation": "engineered_summary",
        "snapshot_protocol": "final",
        "final_scale_factor": 1.0,
        "summary_definition_version": SUMMARY_DEFINITION_VERSION,
        "top_n": job["top_n"],
        "seed": int(job["seed"]),
        "dataset_path": str(job["dataset_path"]),
        "dataset_sha256": job["dataset_sha256"],
        "split_manifest_path": str(job["split_manifest_path"]),
        "split_manifest_sha256": job["split_manifest_sha256"],
        "feature_names": SUMMARY_FEATURE_NAMES,
        "feature_count": 20,
        "target": "Omega_m",
        "target_normalization": "none",
        "model_hyperparameters": hyperparameters,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        **git_provenance(repo_root),
    }


def load_summary_data(job: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    dataset_path = Path(job["dataset_path"])
    if sha256_file(dataset_path) != job["dataset_sha256"]:
        raise ValueError(f"Dataset SHA-256 mismatch: {dataset_path}")
    data = load_processed_dataset(dataset_path)
    ids, features, targets = extract_dataset_summaries(data, snapshot_protocol="final")
    del data
    return ids, features, targets


def load_bound_data(
    job: Mapping[str, Any],
    summary_data: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = Path(job["split_manifest_path"])
    if sha256_file(manifest_path) != job["split_manifest_sha256"]:
        raise ValueError(f"Split manifest SHA-256 mismatch: {manifest_path}")
    ids, features, targets = summary_data or load_summary_data(job)
    manifest = load_split_manifest(
        manifest_path, ids, dataset_identity=str(job["dataset_sha256"]), expected_seed=int(job["seed"])
    )
    expected_counts = {"train": 700, "val": 99, "test": 201, "unused": 0}
    if any(int(manifest["counts"].get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError(f"Manifest does not have exact 700/99/201/0 counts: {manifest_path}")
    return ids, features, targets, manifest


def run_job(
    job: Mapping[str, Any], output_root: str | Path = "experiments", repo_root: str | Path = ".",
    bound_data: tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> str:
    repo = Path(repo_root).resolve()
    config = build_run_config(job, repo)
    destination = Path(output_root) / str(job["experiment_name"])
    state = completion_state(destination, config)
    if state == "complete":
        return "skipped_complete"
    ids, features, targets, manifest = bound_data or load_bound_data(job)
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_ids: dict[str, list[str]] = {}
    for split, manifest_key in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
        split_ids[split] = list(manifest[manifest_key])
        split_arrays[split] = arrays_for_ids(split_ids[split], ids, features, targets)
    model = build_model(str(job["model_family"]), int(job["seed"]))
    if model is None:
        prediction_value = float(np.mean(split_arrays["train"][1], dtype=np.float64))
        predictions = {
            split: np.full(len(values[1]), prediction_value, dtype=np.float64)
            for split, values in split_arrays.items()
        }
        config["training_target_mean"] = prediction_value
    else:
        model.fit(*split_arrays["train"])
        predictions = {split: model.predict(values[0]) for split, values in split_arrays.items()}
    rows = {
        split: build_prediction_rows(split_ids[split], split_arrays[split][1], predictions[split])
        for split in ("train", "val", "test")
    }
    metrics = {split: compute_metrics(rows[split]) for split in ("train", "val", "test")}
    Path(output_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{job['experiment_name']}.tmp-", dir=output_root))
    try:
        write_json(config, temporary / "config.json")
        write_json(metrics, temporary / "metrics.json")
        for split, filename in SPLIT_FILENAMES.items():
            write_prediction_csv(rows[split], temporary / "predictions" / filename)
        if model is not None:
            joblib.dump(model, temporary / "model.joblib")
        os.rename(temporary, destination)
    except Exception:
        # The isolated temp directory intentionally remains for diagnosis; destination stays absent.
        raise
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--execute", action="store_true", help="Execute jobs; default is selection-only dry run.")
    parser.add_argument("--experiment-name", help="Restrict to one deterministic experiment name.")
    args = parser.parse_args()
    family = load_family(args.family_spec)
    jobs = expand_family_jobs(family)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
        if not jobs:
            raise ValueError(f"Unknown family experiment: {args.experiment_name}")
    print(f"Selected {len(jobs)} jobs ({sum(j['model_family'] != 'mean' for j in jobs)} fitted sklearn models).")
    for job in jobs:
        print(f"{job['model_family']:18s} seed={job['seed']:4d} top_n={str(job['top_n']):4s} {job['experiment_name']}")
    if not args.execute:
        print("DRY RUN ONLY: no datasets loaded and no experiment artifacts created.")
        return
    summary_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for job in jobs:
        key = str(job["dataset_path"])
        if key not in summary_cache:
            summary_cache[key] = load_summary_data(job)
        bound_data = load_bound_data(job, summary_data=summary_cache[key])
        result = run_job(job, args.output_root, bound_data=bound_data)
        print(f"{result}: {job['experiment_name']}")


if __name__ == "__main__":
    main()
