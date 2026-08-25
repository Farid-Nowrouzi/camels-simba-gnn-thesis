#!/usr/bin/env python3
"""Verify the frozen T2/T5 mean-vs-last EvolveGCN-H diagnostic.

Default mode is a production-safe preflight: all six reused mean cells must be
complete and compatible, while the six last cells must be absent and runnable.
Use ``--require-complete`` only after production to require all twelve cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_pipeline.common import load_family_spec  # noqa: E402
from src.models.evolvegcn_h import EvolveGCNHRegressor, count_parameters  # noqa: E402
from src.training.train_evolvegcn_h import TEMPORAL_PROTOCOL_NESTED_SUFFIX, snapshot_value  # noqa: E402


EXPECTED_DATASET_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
EXPECTED_VALUES = [0.2, 0.25, 0.51209, 0.75065, 1.0]
EXPECTED_SUBSETS = {2: [3, 4], 5: [0, 1, 2, 3, 4]}
EXPECTED_SPLIT_SHAS = {
    42: "f5556ec5c193e7cb80f2231705edbdae32d4de206889dec308a819bdde427ab7",
    123: "18a295106ec844848053f3040d2be3cdf73a443010be31e6e2cd962362982471",
    2025: "c233c0631b1a24d963ffccc7c6389054fddf135a7aabc4dc4ab7bf5976fab3a9",
}
EXPECTED_COUNTS = {"train": 700, "val": 99, "test": 201}
EXPECTED_PARAMS = 3_408_097
ARTIFACTS = (
    "config.json", "metrics.json", "train_log.csv",
    "predictions/train_predictions.csv", "predictions/val_predictions.csv",
    "predictions/test_predictions.csv", "checkpoints/best_model.pt",
)
SCIENCE_KEYS = (
    "model", "dataset_path", "dataset_identity", "seed", "batch_size", "epochs",
    "patience", "learning_rate", "weight_decay", "hidden_dim", "num_layers",
    "dropout", "activation", "graph_pooling", "head_type", "add_self_loops",
    "train_ratio", "val_ratio", "test_ratio", "grad_clip_norm", "optimizer", "loss",
    "scheduler", "checkpoint_criterion", "normalize_target", "use_summary_features",
    "num_total_universes", "num_train_universes", "num_val_universes",
    "num_test_universes", "num_nodes", "node_features", "trainable_parameters",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def state_schema(state: Mapping[str, torch.Tensor]) -> list[tuple[str, tuple[int, ...]]]:
    return [(key, tuple(value.shape)) for key, value in state.items()]


def new_model(pooling: str) -> EvolveGCNHRegressor:
    return EvolveGCNHRegressor(
        node_features=7, hidden_dim=32, num_layers=2, dropout=0.2,
        activation="relu", temporal_pooling=pooling, graph_pooling="mean",
        head_type="linear", add_self_loops=True,
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reproduce(rows: list[dict[str, str]]) -> dict[str, float]:
    targets = [float(row["true_omega_m"]) for row in rows]
    predictions = [float(row["pred_omega_m"]) for row in rows]
    if not rows or not all(math.isfinite(value) for value in targets + predictions):
        raise ValueError("Empty or non-finite prediction file")
    residuals = [prediction - target for prediction, target in zip(predictions, targets)]
    mse = sum(value * value for value in residuals) / len(residuals)
    target_mean = sum(targets) / len(targets)
    total = sum((value - target_mean) ** 2 for value in targets)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": sum(abs(value) for value in residuals) / len(residuals),
        "r2": 1.0 - sum(value * value for value in residuals) / total,
    }


def normalized_temporal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy full-T5 configs that predate explicit subset provenance."""
    T = int(config["num_snapshots"])
    indices = config.get("selected_snapshot_indices")
    if indices is None and T == 5:
        indices = EXPECTED_SUBSETS[5]
    values = config.get("selected_snapshot_values")
    if values is None and T == 5:
        values = EXPECTED_VALUES
    return {
        **{key: config.get(key) for key in SCIENCE_KEYS},
        "num_snapshots": T,
        "selected_snapshot_indices": indices,
        "selected_snapshot_values": values,
        "temporal_protocol": config.get("temporal_protocol") or TEMPORAL_PROTOCOL_NESTED_SUFFIX,
        "train_ids": config.get("train_ids"),
        "val_ids": config.get("val_ids"),
        "test_ids": config.get("test_ids"),
    }


def verify_family_matrix(spec: Mapping[str, Any]) -> None:
    if spec.get("conceptual_cell_count") != 12 or len(spec.get("runs", [])) != 12:
        raise ValueError("Family must encode exactly 12 conceptual cells")
    if spec.get("grouping_values") != ["T2_mean", "T2_last", "T5_mean", "T5_last"]:
        raise ValueError("Family must contain T2/T5 mean/last only")
    expected = {(T, pooling, seed) for T in (2, 5) for pooling in ("mean", "last") for seed in (42, 123, 2025)}
    observed: set[tuple[int, str, int]] = set()
    for run in spec["runs"]:
        key = (int(run["T"]), str(run["pooling"]), int(run["seed"]))
        observed.add(key)
        T, pooling, _seed = key
        indices = EXPECTED_SUBSETS[T]
        values = [EXPECTED_VALUES[index] for index in indices]
        if run["group_value"] != f"T{T}_{pooling}":
            raise ValueError(f"Incorrect group label: {run['group_value']}")
        if run["selected_snapshot_indices"] != indices or run["selected_snapshot_values"] != values:
            raise ValueError(f"Incorrect temporal subset: {key}")
        if run["temporal_protocol"] != TEMPORAL_PROTOCOL_NESTED_SUFFIX:
            raise ValueError(f"Incorrect temporal protocol: {key}")
        if run["dataset_sha256"] != EXPECTED_DATASET_SHA:
            raise ValueError(f"Dataset substitution in family: {key}")
        expected_action = "reuse" if pooling == "mean" else "run_if_missing"
        if run["action"] != expected_action:
            raise ValueError(f"Incorrect action for {key}: {run['action']}")
        if pooling == "last" and "tlast" not in run["experiment_name"]:
            raise ValueError(f"Last-pooling name is ambiguous: {run['experiment_name']}")
    if observed != expected:
        raise ValueError(f"Matrix mismatch: missing={expected-observed}, extra={observed-expected}")


def verify_manifests(spec: Mapping[str, Any], dataset_ids: set[str]) -> dict[int, dict[str, Any]]:
    manifests: dict[int, dict[str, Any]] = {}
    for seed in (42, 123, 2025):
        run = next(item for item in spec["runs"] if int(item["seed"]) == seed)
        path = ROOT / run["split_manifest_path"]
        if sha256(path) != EXPECTED_SPLIT_SHAS[seed] or run["split_manifest_sha256"] != EXPECTED_SPLIT_SHAS[seed]:
            raise ValueError(f"Split SHA mismatch for seed {seed}")
        manifest = read_json(path)
        if manifest.get("seed") != seed or manifest.get("dataset_identity") != EXPECTED_DATASET_SHA:
            raise ValueError(f"Split identity mismatch for seed {seed}")
        ids: list[str] = []
        for split, count in EXPECTED_COUNTS.items():
            split_ids = manifest[f"{split}_ids"]
            if len(split_ids) != count:
                raise ValueError(f"Split count mismatch: seed={seed}, split={split}")
            ids.extend(split_ids)
        if len(ids) != len(set(ids)) or set(ids) != dataset_ids:
            raise ValueError(f"Split is not an exact ordered partition for seed {seed}")
        manifests[seed] = manifest
    return manifests


def verify_artifacts(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_schema: list,
    fixed_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    path = ROOT / run["experiment_path"]
    config = read_json(path / "config.json")
    metrics = read_json(path / "metrics.json")
    normalized = normalized_temporal_config(config)
    T = int(run["T"])
    for key, expected in fixed_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Frozen config mismatch: {path}, {key}={config.get(key)!r}, expected={expected!r}"
            )
    if config.get("temporal_pooling") != run["pooling"] or config.get("seed") != run["seed"]:
        raise ValueError(f"Model/pooling/seed substitution: {path}")
    if normalized["num_snapshots"] != T:
        raise ValueError(f"Snapshot depth mismatch: {path}")
    if normalized["selected_snapshot_indices"] != EXPECTED_SUBSETS[T]:
        raise ValueError(f"Snapshot indices mismatch: {path}")
    if normalized["selected_snapshot_values"] != [EXPECTED_VALUES[index] for index in EXPECTED_SUBSETS[T]]:
        raise ValueError(f"Snapshot values mismatch: {path}")
    if config.get("dataset_identity") != EXPECTED_DATASET_SHA:
        raise ValueError(f"Dataset identity mismatch: {path}")
    if config.get("split_manifest_sha256") != EXPECTED_SPLIT_SHAS[int(run["seed"])]:
        raise ValueError(f"Split provenance mismatch: {path}")
    if int(config.get("trainable_parameters", -1)) != EXPECTED_PARAMS:
        raise ValueError(f"Parameter count mismatch: {path}")

    split_rows: dict[str, list[dict[str, str]]] = {}
    for split, count in EXPECTED_COUNTS.items():
        expected_ids = manifest[f"{split}_ids"]
        if config.get(f"{split}_ids") != expected_ids:
            raise ValueError(f"Ordered config IDs mismatch: {path}, {split}")
        rows = csv_rows(path / "predictions" / f"{split}_predictions.csv")
        split_rows[split] = rows
        if len(rows) != count or [row["universe_id"] for row in rows] != expected_ids:
            raise ValueError(f"Ordered prediction IDs/count mismatch: {path}, {split}")
        reproduced = reproduce(rows)
        for metric, value in reproduced.items():
            if not math.isclose(float(metrics[split][metric]), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Metric does not reproduce: {path}, {split}.{metric}")

    checkpoint_path = path / "checkpoints/best_model.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if state_schema(checkpoint["model_state_dict"]) != expected_schema:
        raise ValueError(f"Checkpoint state schema mismatch: {path}")
    if checkpoint.get("config") != config:
        raise ValueError(f"Checkpoint/config identity mismatch: {path}")
    log = csv_rows(path / "train_log.csv")
    min_val = min(float(row["val_mse"]) for row in log)
    if not math.isclose(float(checkpoint["best_val_mse"]), min_val, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Checkpoint is not minimum-validation-MSE: {path}")
    if int(checkpoint["epoch"]) != int(metrics["best_epoch"]):
        raise ValueError(f"Checkpoint epoch mismatch: {path}")
    return config, split_rows


def verify_pair_controls(completed: Mapping[tuple[int, str, int], tuple[dict, dict]]) -> None:
    for T in (2, 5):
        for seed in (42, 123, 2025):
            mean = completed.get((T, "mean", seed))
            last = completed.get((T, "last", seed))
            if mean is None or last is None:
                continue
            mean_config, mean_rows = mean
            last_config, last_rows = last
            normalized_mean = normalized_temporal_config(mean_config)
            normalized_last = normalized_temporal_config(last_config)
            if normalized_mean != normalized_last:
                differing = [key for key in normalized_mean if normalized_mean[key] != normalized_last[key]]
                raise ValueError(f"Mean/last scientific configs differ beyond pooling: T={T}, seed={seed}, fields={differing}")
            for split in EXPECTED_COUNTS:
                mean_truth = [(row["universe_id"], row["true_omega_m"]) for row in mean_rows[split]]
                last_truth = [(row["universe_id"], row["true_omega_m"]) for row in last_rows[split]]
                if mean_truth != last_truth:
                    raise ValueError(f"Mean/last ordered IDs or truth differ: T={T}, seed={seed}, split={split}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", default="configs/experiment_families/u1000_top1500_temporal_pooling_diagnostic_evolvegcn_h.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        spec = load_family_spec(ROOT / args.family_spec)
        verify_family_matrix(spec)
        dataset_path = ROOT / spec["master_dataset"]["path"]
        if sha256(dataset_path) != EXPECTED_DATASET_SHA:
            raise ValueError("Frozen Top1500 dataset SHA-256 mismatch")
        try:
            data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(dataset_path, map_location="cpu")
        if len(data) != 1000:
            raise ValueError(f"Dataset universe count is {len(data)}, expected 1000")
        first = data[next(iter(data))]
        if [snapshot_value(item) for item in first["snapshots"]] != EXPECTED_VALUES:
            raise ValueError("Dataset master snapshot values changed")
        manifests = verify_manifests(spec, set(data))

        models = {pooling: new_model(pooling) for pooling in ("mean", "last")}
        schemas = {pooling: state_schema(candidate.state_dict()) for pooling, candidate in models.items()}
        if any(count_parameters(candidate) != EXPECTED_PARAMS for candidate in models.values()):
            raise ValueError("Mean/last parameter count is not 3,408,097")
        if schemas["mean"] != schemas["last"]:
            raise ValueError("Mean/last state-dict schemas differ")

        completed: dict[tuple[int, str, int], tuple[dict, dict]] = {}
        missing_last = []
        collisions = []
        for run in spec["runs"]:
            path = ROOT / run["experiment_path"]
            present = [(path / artifact).is_file() and (path / artifact).stat().st_size > 0 for artifact in ARTIFACTS]
            if not path.exists():
                if run["pooling"] == "mean":
                    raise ValueError(f"Required reused mean cell is missing: {path}")
                missing_last.append(run)
                continue
            if not all(present):
                collisions.append(run["experiment_name"])
                continue
            key = (int(run["T"]), str(run["pooling"]), int(run["seed"]))
            completed[key] = verify_artifacts(
                run,
                manifests[int(run["seed"])],
                schemas["mean"],
                spec["fixed_scientific_settings"]["config"],
            )
        if collisions:
            raise ValueError(f"Partial/unexpected experiment collisions: {collisions}")
        if args.require_complete and missing_last:
            raise ValueError(f"Full verification requested but {len(missing_last)} last cells are missing")
        if sum(key[1] == "mean" for key in completed) != 6:
            raise ValueError("Exactly six reused mean cells were not verified")
        verify_pair_controls(completed)

        print("VERIFIED frozen T2/T5 mean-vs-last temporal-pooling diagnostic")
        print("Conceptual = 12")
        print("Reused mean = 6")
        print(f"New last = {len(missing_last)} missing/runnable; {6-len(missing_last)} complete")
        print("Unexpected collisions = 0")
        print(f"Completed cells verified = {len(completed)}")
        print("Parameter count = 3408097; mean/last state schema identical")
        print("PREFLIGHT PASS" if not args.require_complete else "FULL 12-CELL VERIFICATION PASS")
        return 0
    except Exception as exc:  # noqa: BLE001 - verifier must surface integrity failures.
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
