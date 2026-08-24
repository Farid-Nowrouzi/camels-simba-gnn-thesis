#!/usr/bin/env python3
"""Verify the frozen U1000 Top1500 temporal-context ablation family.

The default mode is production-safe preflight: missing T1--T4 cells are reported
as runnable, while the three historical T5 cells must exist and pass provenance
checks. ``--require-complete`` upgrades missing/partial cells to errors and checks
the completed 15-cell artifact matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.evolvegcn_h import EvolveGCNHRegressor, count_parameters  # noqa: E402
from src.training.sparse_batch import collate_sparse_temporal  # noqa: E402
from src.training.train_evolvegcn_h import (  # noqa: E402
    TEMPORAL_PROTOCOL_NESTED_SUFFIX,
    select_temporal_snapshots,
    snapshot_value,
)


EXPECTED_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
EXPECTED_VALUES = [0.2, 0.25, 0.51209, 0.75065, 1.0]
EXPECTED_SUBSETS = {T: list(range(5 - T, 5)) for T in range(1, 6)}
EXPECTED_COUNTS = {"train": 700, "val": 99, "test": 201}
EXPECTED_PARAMS = 3_408_097
FEATURES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]
CONFIG_FIXED = {
    "model": "EvolveGCNHRegressor",
    "dataset_identity": EXPECTED_SHA,
    "batch_size": 4,
    "epochs": 300,
    "patience": 40,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.2,
    "activation": "relu",
    "temporal_pooling": "mean",
    "graph_pooling": "mean",
    "head_type": "linear",
    "add_self_loops": True,
    "train_ratio": 0.7,
    "val_ratio": 0.099,
    "test_ratio": 0.201,
    "grad_clip_norm": 1.0,
    "normalize_target": False,
    "use_summary_features": False,
    "optimizer": "AdamW",
    "loss": "MSELoss",
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 10,
        "min_lr": 0.000001,
    },
    "checkpoint_criterion": "minimum_validation_mse",
    "num_total_universes": 1000,
    "num_train_universes": 700,
    "num_val_universes": 99,
    "num_test_universes": 201,
    "num_nodes": 1500,
    "node_features": 7,
    "trainable_parameters": EXPECTED_PARAMS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def model() -> EvolveGCNHRegressor:
    return EvolveGCNHRegressor(
        node_features=7,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        activation="relu",
        temporal_pooling="mean",
        graph_pooling="mean",
        head_type="linear",
        add_self_loops=True,
    )


def state_schema(state: Mapping[str, torch.Tensor]) -> list[tuple[str, tuple[int, ...]]]:
    return [(key, tuple(tensor.shape)) for key, tensor in state.items()]


def verify_family_structure(spec: Mapping[str, Any]) -> None:
    if spec.get("temporal_protocol") != TEMPORAL_PROTOCOL_NESTED_SUFFIX:
        raise ValueError("Family temporal protocol is not nested_suffix_context_v1.")
    if spec.get("conceptual_cell_count") != 15 or len(spec.get("runs", [])) != 15:
        raise ValueError("Family must contain exactly 15 conceptual cells.")
    if spec.get("grouping_values") != [1, 2, 3, 4, 5]:
        raise ValueError("Family grouping values must be exactly T1--T5.")
    seen: set[tuple[int, int]] = set()
    for run in spec["runs"]:
        T, seed = int(run["T"]), int(run["seed"])
        if (T, seed) in seen:
            raise ValueError(f"Duplicate family cell T={T}, seed={seed}.")
        seen.add((T, seed))
        indices = EXPECTED_SUBSETS[T]
        values = [EXPECTED_VALUES[index] for index in indices]
        if run.get("group_value") != T:
            raise ValueError(f"T={T}, seed={seed}: group_value mismatch.")
        if run.get("selected_snapshot_indices") != indices:
            raise ValueError(f"T={T}, seed={seed}: incorrect snapshot indices.")
        if run.get("selected_snapshot_values") != values:
            raise ValueError(f"T={T}, seed={seed}: incorrect snapshot values.")
        if run.get("temporal_protocol") != TEMPORAL_PROTOCOL_NESTED_SUFFIX:
            raise ValueError(f"T={T}, seed={seed}: incorrect temporal protocol.")
        if values[-1] != 1.0 or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError(f"T={T}, seed={seed}: non-chronological snapshot values.")
        reused = T == 5
        if reused != (run.get("action") == "reuse"):
            raise ValueError(f"T={T}, seed={seed}: incorrect reuse action.")
    expected = {(T, seed) for T in range(1, 6) for seed in (42, 123, 2025)}
    if seen != expected:
        raise ValueError(f"Family matrix mismatch: missing={expected-seen}, extra={seen-expected}.")


def verify_manifests(spec: Mapping[str, Any], dataset_ids: Iterable[str] | None) -> dict[int, dict]:
    manifests: dict[int, dict] = {}
    universe = set(dataset_ids) if dataset_ids is not None else None
    for run in spec["runs"]:
        seed = int(run["seed"])
        if seed in manifests:
            continue
        path = REPO_ROOT / run["split_manifest_path"]
        if sha256_file(path) != run["split_manifest_sha256"]:
            raise ValueError(f"Split manifest SHA mismatch: {path}")
        manifest = read_json(path)
        if manifest.get("seed") != seed or manifest.get("dataset_identity") != EXPECTED_SHA:
            raise ValueError(f"Split manifest identity mismatch: {path}")
        ids = [*manifest["train_ids"], *manifest["val_ids"], *manifest["test_ids"]]
        if [len(manifest[f"{split}_ids"]) for split in ("train", "val", "test")] != [700, 99, 201]:
            raise ValueError(f"Split manifest counts mismatch: {path}")
        if len(ids) != len(set(ids)) or (universe is not None and set(ids) != universe):
            raise ValueError(f"Split manifest is not an exact disjoint dataset partition: {path}")
        manifests[seed] = manifest
    return manifests


def audit_real_dataset(data: Mapping[str, dict]) -> dict[str, Any]:
    if len(data) != 1000:
        raise ValueError(f"Dataset contains {len(data)} universes, expected 1000.")
    padded_snapshots = 0
    subset_samples = 0
    minimum_real_nodes = 1500
    for universe_id, sample in data.items():
        if sample.get("feature_names") != FEATURES or sample.get("normalization") != "none":
            raise ValueError(f"{universe_id}: feature/normalization identity mismatch.")
        target_before = torch.as_tensor(sample["target"]).clone()
        for T, indices in EXPECTED_SUBSETS.items():
            values = [EXPECTED_VALUES[index] for index in indices]
            selected = select_temporal_snapshots(sample, indices, values, sample_label=universe_id)
            subset_samples += 1
            if any(len(selected[field]) != T for field in ("Nodes_list", "mask_list", "edge_index_list", "snapshots")):
                raise ValueError(f"{universe_id}: selected temporal field lengths mismatch T={T}.")
            if [snapshot_value(item) for item in selected["snapshots"]] != values:
                raise ValueError(f"{universe_id}: selected values mismatch T={T}.")
            if not torch.equal(torch.as_tensor(selected["target"]), target_before):
                raise ValueError(f"{universe_id}: target changed under T={T} view.")
            for nodes, mask, edges in zip(selected["Nodes_list"], selected["mask_list"], selected["edge_index_list"]):
                if nodes.ndim != 2 or nodes.shape[1] != 7 or mask.shape[0] != nodes.shape[0]:
                    raise ValueError(f"{universe_id}: invalid node/mask shape.")
                valid = mask.reshape(-1) > 0
                real_count = int(valid.sum())
                minimum_real_nodes = min(minimum_real_nodes, real_count)
                if real_count < nodes.shape[0]:
                    padded_snapshots += 1
                if edges.ndim != 2 or edges.shape[0] != 2:
                    raise ValueError(f"{universe_id}: invalid edge shape.")
                if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes.shape[0]):
                    raise ValueError(f"{universe_id}: out-of-bounds edge.")
                if edges.numel() and not bool(valid[edges].all()):
                    raise ValueError(f"{universe_id}: edge touches serialized padding.")
    return {
        "universes": len(data),
        "subset_views": subset_samples,
        "temporal_graphs_checked": sum(1000 * T for T in range(1, 6)),
        "padded_temporal_graph_occurrences": padded_snapshots,
        "minimum_real_nodes": minimum_real_nodes,
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reproduced_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    targets = [float(row["true_omega_m"]) for row in rows]
    predictions = [float(row["pred_omega_m"]) for row in rows]
    if not all(math.isfinite(value) for value in targets + predictions):
        raise ValueError("Prediction CSV contains non-finite values.")
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    mse = sum(error * error for error in errors) / len(errors)
    mean = sum(targets) / len(targets)
    total = sum((target - mean) ** 2 for target in targets)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "r2": 1.0 - sum(error * error for error in errors) / total,
    }


def verify_artifacts(run: Mapping[str, Any], manifest: Mapping[str, Any], expected_schema: list) -> dict[str, Any]:
    path = REPO_ROOT / run["experiment_path"]
    config = read_json(path / "config.json")
    metrics = read_json(path / "metrics.json")
    for key, expected in CONFIG_FIXED.items():
        if config.get(key) != expected:
            raise ValueError(f"{path}: config {key}={config.get(key)!r}, expected {expected!r}.")
    if config.get("dataset_path") != run["dataset_path"] or config.get("seed") != run["seed"]:
        raise ValueError(f"{path}: dataset/seed identity mismatch.")
    T = int(run["T"])
    if config.get("num_snapshots") != T:
        raise ValueError(f"{path}: num_snapshots mismatch.")
    if T < 5:
        required = {
            "experiment_type": "temporal_context_depth_ablation",
            "temporal_protocol": TEMPORAL_PROTOCOL_NESTED_SUFFIX,
            "num_selected_snapshots": T,
            "selected_snapshot_indices": run["selected_snapshot_indices"],
            "selected_snapshot_values": run["selected_snapshot_values"],
            "master_snapshot_values": EXPECTED_VALUES,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"{path}: temporal config {key} mismatch.")
    for split in ("train", "val", "test"):
        if config.get(f"{split}_ids") != manifest[f"{split}_ids"]:
            raise ValueError(f"{path}: ordered {split} IDs differ from manifest.")
        rows = csv_rows(path / "predictions" / f"{split}_predictions.csv")
        if len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{path}: {split} prediction count mismatch.")
        if [row["universe_id"] for row in rows] != manifest[f"{split}_ids"]:
            raise ValueError(f"{path}: ordered {split} prediction IDs mismatch.")
        reproduced = reproduced_metrics(rows)
        saved = metrics[split]
        for metric, value in reproduced.items():
            if not math.isclose(float(saved[metric]), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{path}: saved {split}.{metric} does not reproduce.")
    checkpoint_path = path / "checkpoints" / "best_model.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("config") != config:
        raise ValueError(f"{path}: checkpoint config identity differs from config.json.")
    if state_schema(checkpoint["model_state_dict"]) != expected_schema:
        raise ValueError(f"{path}: checkpoint state-dict schema mismatch.")
    with (path / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        log = list(csv.DictReader(handle))
    minimum = min(float(row["val_mse"]) for row in log)
    if not math.isclose(float(checkpoint["best_val_mse"]), minimum, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{path}: checkpoint is not the minimum-validation-MSE checkpoint.")
    if int(checkpoint["epoch"]) != int(metrics["best_epoch"]):
        raise ValueError(f"{path}: checkpoint epoch differs from metrics best_epoch.")
    return {
        "T": T,
        "seed": run["seed"],
        "origin": run["origin"],
        "source_commit": config.get("training_git_commit"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_mae": metrics["test"]["mae"],
    }


def verify_static_references(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Confirm Static GCN is present only as unchanged external context."""
    results = []
    split_sha_by_seed = {
        int(run["seed"]): run["split_manifest_sha256"]
        for run in spec["runs"]
        if int(run["T"]) == 5
    }
    references = spec.get("static_external_references", [])
    if len(references) != 3:
        raise ValueError("Expected exactly three external Static GCN references.")
    for reference in references:
        seed = int(reference["seed"])
        path = REPO_ROOT / reference["experiment_path"]
        config = read_json(path / "config.json")
        metrics = read_json(path / "metrics.json")
        checkpoint = path / "checkpoints" / "best_model.pt"
        if reference.get("role") != "external_context_only":
            raise ValueError(f"Static reference has incorrect role: {path}")
        if config.get("dataset_identity") != EXPECTED_SHA:
            raise ValueError(f"Static reference dataset identity mismatch: {path}")
        if config.get("split_manifest_sha256") != split_sha_by_seed[seed]:
            raise ValueError(f"Static reference split identity mismatch: {path}")
        results.append({
            "seed": seed,
            "experiment_path": reference["experiment_path"],
            "source_commit": config.get("training_git_commit"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "test_mae": metrics["test"]["mae"],
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-spec",
        default="configs/experiment_families/u1000_top1500_temporal_context_ablation_evolvegcn_h.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-real-data-audit", action="store_true")
    args = parser.parse_args()

    try:
        spec = read_json(REPO_ROOT / args.family_spec)
        verify_family_structure(spec)
        dataset_path = REPO_ROOT / spec["master_dataset"]["path"]
        if sha256_file(dataset_path) != EXPECTED_SHA:
            raise ValueError("Top1500 master dataset SHA-256 mismatch.")

        data = None
        audit = None
        if not args.skip_real_data_audit:
            try:
                data = torch.load(dataset_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(dataset_path, map_location="cpu")
            audit = audit_real_dataset(data)

        manifests = verify_manifests(spec, data.keys() if data is not None else None)
        static_references = verify_static_references(spec)
        reference_model = model()
        if count_parameters(reference_model) != EXPECTED_PARAMS:
            raise ValueError("Reference model parameter count mismatch.")
        expected_schema = state_schema(reference_model.state_dict())

        verified = []
        missing_new = []
        collisions = []
        for run in spec["runs"]:
            path = REPO_ROOT / run["experiment_path"]
            required = [path / item for item in spec["expected_artifacts"]]
            present = [item.is_file() and item.stat().st_size > 0 for item in required]
            if not path.exists():
                if run["action"] == "reuse":
                    raise ValueError(f"Missing required historical T5 cell: {path}")
                missing_new.append(run)
                continue
            if not all(present):
                collisions.append((run, [str(item.relative_to(path)) for item, ok in zip(required, present) if not ok]))
                continue
            verified.append(verify_artifacts(run, manifests[int(run["seed"])], expected_schema))

        if collisions:
            raise ValueError(f"Partial/unexpected experiment collisions: {collisions}")
        if args.require_complete and missing_new:
            raise ValueError(f"Full verification requested but {len(missing_new)} cells are missing.")
        t5 = [item for item in verified if item["T"] == 5]
        if len(t5) != 3:
            raise ValueError(f"Expected 3 verified historical T5 cells, got {len(t5)}.")

        print("VERIFIED temporal family structure and frozen 5x3 matrix.")
        print(f"Conceptual matrix: 15 cells")
        print(f"Reusable: 3 T5 cells")
        print(f"Runnable/new: {len(missing_new)} T1-T4 cells")
        print("Unexpected collisions: 0")
        print("Missing required T5 reused cells: 0")
        print(f"Completed cells verified: {len(verified)}")
        if audit is not None:
            print(f"Real-data audit: {json.dumps(audit, sort_keys=True)}")
        for item in t5:
            print(f"T5 provenance: {json.dumps(item, sort_keys=True)}")
        for item in static_references:
            print(f"Static external reference: {json.dumps(item, sort_keys=True)}")
        print("PREFLIGHT PASS" if not args.require_complete else "FULL 15-CELL VERIFICATION PASS")
        return 0
    except Exception as exc:  # noqa: BLE001 - verifier reports a single explicit failure.
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
