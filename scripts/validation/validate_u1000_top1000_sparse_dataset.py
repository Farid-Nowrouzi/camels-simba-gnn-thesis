#!/usr/bin/env python3
"""Read-only post-build validator for the approved sparse U1000 dataset."""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.source_manifest import source_manifest_sha256


EXPECTED_RELATIVE_DATASET = Path(
    "data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt"
)
EXPECTED_RELATIVE_TARGET = Path("outputs/target_inspection_1000u.csv")
EXPECTED_TARGET_SHA256 = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
EXPECTED_SCHEMA = "camels_temporal_sparse_v1"
EXPECTED_STORAGE = "sparse_edge_index"
EXPECTED_LOGICAL_ID = (
    "camels_simba_u1000_top1000_temporal5_none_periodic_knn_k8_box25_sparse_v1"
)
EXPECTED_SNAPSHOTS = [0.20000, 0.25000, 0.51209, 0.75065, 1.00000]
EXPECTED_FEATURES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]
EXPECTED_TOP_N = 1000


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_targets(path: Path) -> dict[str, float]:
    targets: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "target table has no header")
        universe_column = next(
            (name for name in ("universe_id", "Universe", "universe", "Universe_ID", "lh_id", "LH")
             if name in reader.fieldnames),
            None,
        )
        target_column = next(
            (name for name in ("omega_m", "Omega_m", "Omega_M", "target", "Target", "omega_m_value")
             if name in reader.fieldnames),
            None,
        )
        require(universe_column is not None, "target table has no supported universe-ID column")
        require(target_column is not None, "target table has no supported target column")
        for row in reader:
            raw_id = str(row[universe_column]).strip()
            universe_id = f"LH_{int(raw_id.split('_', 1)[1] if raw_id.lower().startswith('lh_') else raw_id)}"
            require(universe_id not in targets, f"duplicate target-table ID: {universe_id}")
            target = float(row[target_column])
            require(math.isfinite(target), f"nonfinite target-table value for {universe_id}")
            targets[universe_id] = target
    return targets


def check_exact_metadata(metadata: Mapping[str, Any], complete: Mapping[str, Any], checksum: str) -> None:
    required_fields = {
        "dataset_type", "preprocessing_version", "num_universes_requested",
        "num_universes_successful", "num_universes_failed", "allow_partial",
        "num_snapshots", "num_nodes", "normalization", "target_normalization",
        "graph_mode", "graph_storage", "dataset_schema_version", "k", "radius",
        "periodic_boundary", "periodic_boundary_knn", "box_size", "feature_names",
        "feature_columns", "mass_column", "mass_feature", "node_selection",
        "selection_method", "tie_breaking_policy", "ordered_universe_ids",
        "ordered_universe_ids_hash", "snapshot_ids", "top_n", "edge_policy",
        "source_manifest_policy", "source_manifest_schema_version", "source_manifest",
        "source_manifest_sha256", "source_manifest_hash", "source_manifest_verification",
        "source_manifest_entry_count", "source_manifest_catalogue_count",
        "source_manifest_target_source_count", "target_source_relative_path",
        "target_source_sha256", "selected_halo_hash", "builder_config_hash",
        "git_commit", "python_version", "pytorch_version", "creation_timestamp_utc",
        "checksum_algorithm", "checksum", "completion_status", "targets_csv",
        "target_mode", "used_dummy_target", "dummy_target", "device_used_for_building",
        "saved_device", "source_suite", "raw_catalogue_roots", "position_columns",
        "velocity_columns", "graph_positions", "node_padding_statistics",
        "edge_statistics", "target_summary", "failed_universes",
    }
    missing = sorted(required_fields.difference(metadata))
    require(not missing, f"metadata/provenance fields missing: {missing}")
    require(metadata["dataset_type"] == "temporal_graph_sequences", "wrong dataset type")
    require(metadata["dataset_schema_version"] == EXPECTED_SCHEMA, "wrong sparse schema")
    require(metadata["graph_storage"] == EXPECTED_STORAGE, "wrong graph storage")
    require(metadata["num_universes_requested"] == 1000, "wrong requested universe count")
    require(metadata["num_universes_successful"] == 1000, "wrong successful universe count")
    require(metadata["num_universes_failed"] == 0, "build records failed universes")
    require(metadata["allow_partial"] is False, "partial-build mode was enabled")
    require(metadata["num_snapshots"] == 5, "wrong snapshot count")
    require(metadata["num_nodes"] == EXPECTED_TOP_N and metadata["top_n"] == EXPECTED_TOP_N, "wrong Top-N")
    require(metadata["normalization"] == "none", "wrong node normalization")
    require(metadata["target_normalization"] == "none", "wrong target normalization")
    require(metadata["graph_mode"] == "knn" and metadata["k"] == 8, "wrong kNN protocol")
    require(metadata["radius"] is None, "radius must be unset for kNN")
    require(metadata["periodic_boundary"] is True, "periodic boundary is not enabled")
    require(metadata["periodic_boundary_knn"] is True, "periodic kNN is not enabled")
    require(float(metadata["box_size"]) == 25.0, "wrong box size")
    require(metadata["feature_names"] == EXPECTED_FEATURES, "wrong feature names/order")
    require(metadata["mass_feature"] == "log10_Mvir", "wrong mass feature")
    require(metadata["node_selection"] == "top_num_nodes_by_raw_Mvir_descending",
            "wrong node-selection rule")
    require(metadata["selection_method"] == "raw_Mvir_desc_stable_then_tie_key_asc",
            "wrong selection method")
    require(metadata["tie_breaking_policy"] ==
            "authoritative_halo_id_ascending_else_original_row_index",
            "wrong tie-breaking policy")
    require(metadata["snapshot_ids"] == EXPECTED_SNAPSHOTS, "wrong snapshot order")
    require(metadata["source_manifest_policy"] == "full_sha256", "wrong source-manifest policy")
    require(metadata["source_manifest_schema_version"] == "camels_source_manifest_v1", "wrong source-manifest schema")
    require(metadata["targets_csv"] == EXPECTED_RELATIVE_TARGET.as_posix(), "wrong target-table path")
    require(metadata["target_mode"] == "real_targets_csv", "dataset did not use real target CSV")
    require(metadata["used_dummy_target"] is False and metadata["dummy_target"] is None,
            "dummy target was used or recorded")
    require(metadata["device_used_for_building"] == "cpu" and metadata["saved_device"] == "cpu",
            "dataset was not built/saved on CPU")
    require(metadata["source_suite"] == "CAMELS-SIMBA", "wrong source suite")
    require(metadata["failed_universes"] == [], "failed-universe list is not empty")
    target_summary = metadata["target_summary"]
    require(isinstance(target_summary, dict) and target_summary.get("count") == 1000,
            "target summary count is wrong")
    require(all(math.isfinite(float(target_summary[key])) for key in ("min", "max", "mean")),
            "target summary contains NaN/Inf")
    require(metadata["checksum_algorithm"] == "sha256", "wrong checksum algorithm")
    require(metadata["checksum"] == checksum, "dataset checksum does not match metadata")
    require(metadata["completion_status"] == "complete", "metadata is not complete")
    require(complete.get("status") == "complete", "completion marker is not complete")
    require(complete.get("sha256") == checksum, "dataset checksum does not match completion marker")
    require(complete.get("dataset") == EXPECTED_RELATIVE_DATASET.name, "completion marker names wrong dataset")
    require(complete.get("metadata") == EXPECTED_RELATIVE_DATASET.with_suffix(".metadata.json").name,
            "completion marker names wrong metadata")

    ordered_ids = metadata["ordered_universe_ids"]
    require(len(ordered_ids) == 1000 and len(set(ordered_ids)) == 1000,
            "metadata universe IDs are missing or duplicated")
    expected_ids = [f"LH_{index}" for index in range(1000)]
    require(ordered_ids == expected_ids, "metadata universe IDs/order are not exactly LH_0..LH_999")
    ordered_hash = hashlib.sha256("".join(f"{key}\n" for key in ordered_ids).encode()).hexdigest()
    require(metadata["ordered_universe_ids_hash"] == ordered_hash, "ordered universe-ID hash mismatch")

    logical_id = (
        f"camels_simba_u{metadata['num_universes_successful']}_top{metadata['top_n']}_"
        f"temporal{metadata['num_snapshots']}_{metadata['normalization']}_periodic_"
        f"{metadata['graph_mode']}_k{metadata['k']}_box{float(metadata['box_size']):g}_sparse_v1"
    )
    require(logical_id == EXPECTED_LOGICAL_ID, "logical dataset identity is wrong")
    require(len(checksum) == 64, "invalid immutable dataset checksum identity")


def check_manifest(metadata: Mapping[str, Any], target_path: Path) -> None:
    manifest = metadata["source_manifest"]
    require(isinstance(manifest, dict), "source manifest is not an object")
    require(manifest.get("schema_version") == "camels_source_manifest_v1", "wrong manifest schema")
    require(manifest.get("source_manifest_policy") == "full_sha256", "manifest is not full_sha256")
    require(manifest.get("manifest_sha256") == source_manifest_sha256(manifest),
            "source-manifest top-level hash mismatch")
    require(metadata["source_manifest_sha256"] == manifest["manifest_sha256"],
            "metadata source-manifest hash mismatch")
    require(metadata["source_manifest_hash"] == manifest["manifest_sha256"],
            "metadata source-manifest hash alias mismatch")
    require(manifest.get("entry_count") == 5001, "source manifest must contain 5001 entries")
    require(manifest.get("catalogue_count") == 5000, "source manifest must contain 5000 catalogues")
    require(manifest.get("target_source_count") == 1, "source manifest must contain one target source")
    require(metadata["source_manifest_entry_count"] == 5001, "wrong metadata manifest entry count")
    require(metadata["source_manifest_catalogue_count"] == 5000, "wrong metadata catalogue count")
    require(metadata["source_manifest_target_source_count"] == 1, "wrong metadata target-source count")
    verification = metadata["source_manifest_verification"]
    require(isinstance(verification, dict) and verification.get("verified") is True,
            "source manifest is not recorded as verified")
    require(verification.get("verification_result") == "verified_full_sha256",
            "wrong source-manifest verification result")
    target_entries = [entry for entry in manifest.get("entries", [])
                      if entry.get("source_role") == "target_table"]
    require(len(target_entries) == 1, "source manifest target entry missing or duplicated")
    target_entry = target_entries[0]
    require(target_entry.get("sha256") == EXPECTED_TARGET_SHA256, "manifest target SHA-256 mismatch")
    require(metadata["target_source_sha256"] == EXPECTED_TARGET_SHA256,
            "metadata target SHA-256 mismatch")
    require(target_entry.get("row_count") == 1000, "manifest target row count is wrong")
    require(target_entry.get("relative_path") == target_path.name, "manifest target relative path is wrong")
    require(metadata["target_source_relative_path"] == target_path.name,
            "metadata target relative path is wrong")


def check_dataset(dataset: Any, metadata: Mapping[str, Any], targets: Mapping[str, float]) -> None:
    require(isinstance(dataset, dict), "serialized dataset is not a dictionary")
    expected_ids = [f"LH_{index}" for index in range(1000)]
    ids = list(dataset)
    require(len(ids) == 1000 and len(set(ids)) == 1000, "dataset IDs are missing or duplicated")
    require(ids == expected_ids, "dataset IDs/order are not exactly LH_0..LH_999")
    require(set(targets) == set(expected_ids), "target table IDs are not exactly LH_0..LH_999")

    for universe_id in expected_ids:
        sample = dataset[universe_id]
        require(isinstance(sample, dict), f"{universe_id}: sample is not a dictionary")
        require(sample.get("schema_version") == EXPECTED_SCHEMA, f"{universe_id}: wrong schema")
        require(sample.get("graph_storage") == EXPECTED_STORAGE, f"{universe_id}: wrong storage")
        dense_keys = {"a", "a_list", "adjacency", "adjacency_matrix", "dense_adjacency"}
        require(not any(str(key).lower() in dense_keys for key in sample),
                f"{universe_id}: serialized dense-adjacency key found")
        require(not any(torch.is_tensor(value) and value.ndim == 2 and
                        tuple(value.shape) == (EXPECTED_TOP_N, EXPECTED_TOP_N) for value in sample.values()),
                f"{universe_id}: serialized dense-adjacency tensor found")
        require(sample.get("feature_names") == EXPECTED_FEATURES,
                f"{universe_id}: wrong feature names/order")
        require(sample.get("normalization") == "none", f"{universe_id}: wrong normalization")
        require(sample.get("num_nodes") == EXPECTED_TOP_N and sample.get("num_snapshots") == 5,
                f"{universe_id}: wrong node or snapshot count")
        require(sample.get("graph_mode") == "knn", f"{universe_id}: wrong graph mode")
        require(sample.get("periodic_boundary") is True, f"{universe_id}: periodic disabled")
        require(float(sample.get("box_size")) == 25.0, f"{universe_id}: wrong box size")

        edges = sample.get("edge_index_list")
        nodes = sample.get("Nodes_list")
        masks = sample.get("mask_list")
        snapshots = sample.get("snapshots")
        edge_weights = sample.get("edge_weight_list")
        require(all(isinstance(value, list) and len(value) == 5
                    for value in (edges, nodes, masks, snapshots)),
                f"{universe_id}: temporal lists must each have five entries")
        require(edge_weights is None or (isinstance(edge_weights, list) and len(edge_weights) == 5),
                f"{universe_id}: edge_weight_list must be null or have five entries")
        snapshot_values = [float(snapshot.get("snapshot_value")) for snapshot in snapshots]
        require(snapshot_values == EXPECTED_SNAPSHOTS, f"{universe_id}: wrong snapshot order")

        target = sample.get("target")
        require(torch.is_tensor(target) and target.numel() == 1,
                f"{universe_id}: target is not a scalar tensor")
        target_value = float(target.item())
        require(math.isfinite(target_value), f"{universe_id}: target is nonfinite")
        expected_float32 = float(torch.tensor(targets[universe_id], dtype=torch.float32).item())
        require(target_value == expected_float32, f"{universe_id}: target does not match target table")

        for snapshot_index, (edge_index, features, mask, snapshot) in enumerate(
            zip(edges, nodes, masks, snapshots)
        ):
            label = f"{universe_id} snapshot {snapshot_index}"
            expected_snapshot_name = f"{universe_id}_hlist_{EXPECTED_SNAPSHOTS[snapshot_index]:.5f}.list"
            require(Path(str(snapshot.get("path"))).name == expected_snapshot_name,
                    f"{label}: source filename is wrong")
            require(torch.is_tensor(features) and tuple(features.shape) == (EXPECTED_TOP_N, 7),
                    f"{label}: node-feature shape is not [{EXPECTED_TOP_N},7]")
            require(features.is_floating_point() and bool(torch.isfinite(features).all()),
                    f"{label}: node features contain NaN/Inf or are not floating point")
            require(torch.is_tensor(mask) and tuple(mask.shape) == (EXPECTED_TOP_N, 1),
                    f"{label}: mask shape is not [{EXPECTED_TOP_N},1]")
            require(mask.is_floating_point() and bool(torch.isfinite(mask).all()),
                    f"{label}: mask contains NaN/Inf or is not floating point")
            flat_mask = mask.reshape(-1)
            require(bool(((flat_mask == 0) | (flat_mask == 1)).all()),
                    f"{label}: mask is not binary")
            real_nodes = int(flat_mask.sum().item())
            require(0 < real_nodes <= EXPECTED_TOP_N, f"{label}: invalid real-node count")
            require(bool((flat_mask[:real_nodes] == 1).all()) and
                    bool((flat_mask[real_nodes:] == 0).all()),
                    f"{label}: mask padding is not contiguous")
            require(snapshot.get("num_real_nodes") == real_nodes,
                    f"{label}: snapshot real-node count disagrees with mask")
            require(snapshot.get("selected_num_halos_before_padding") == real_nodes,
                    f"{label}: selected-node count disagrees with mask")
            if real_nodes < EXPECTED_TOP_N:
                require(bool((features[real_nodes:] == 0).all()),
                        f"{label}: padded node features are not zero")
            require(snapshot.get("feature_names") == EXPECTED_FEATURES,
                    f"{label}: snapshot feature names/order are wrong")
            require(snapshot.get("normalization") == "none", f"{label}: wrong normalization")
            require(snapshot.get("graph_mode") == "knn" and snapshot.get("k") == 8,
                    f"{label}: wrong kNN settings")
            require(snapshot.get("periodic_boundary") is True and
                    snapshot.get("periodic_boundary_knn") is True,
                    f"{label}: periodic kNN is not enabled")
            require(float(snapshot.get("box_size")) == 25.0, f"{label}: wrong box size")

            require(torch.is_tensor(edge_index) and edge_index.dtype == torch.long,
                    f"{label}: edge_index must be an int64 tensor")
            require(edge_index.ndim == 2 and edge_index.shape[0] == 2,
                    f"{label}: edge_index shape is not [2,E]")
            require(edge_index.shape[1] > 0, f"{label}: edge_index is empty")
            require(int(edge_index.min().item()) >= 0 and int(edge_index.max().item()) < real_nodes,
                    f"{label}: edge index is out of bounds or reaches padding")
            require(bool((edge_index[0] != edge_index[1]).all()), f"{label}: self-loop found")
            encoded = edge_index[0] * EXPECTED_TOP_N + edge_index[1]
            require(torch.unique(encoded).numel() == encoded.numel(), f"{label}: duplicate edge found")
            reverse = edge_index[1] * EXPECTED_TOP_N + edge_index[0]
            require(bool(torch.isin(reverse, encoded).all()), f"{label}: asymmetric edge found")
            if edge_weights is not None:
                weights = edge_weights[snapshot_index]
                require(torch.is_tensor(weights) and weights.ndim == 1 and
                        weights.numel() == edge_index.shape[1],
                        f"{label}: edge weights do not match edges")
                require(weights.is_floating_point() and bool(torch.isfinite(weights).all()),
                        f"{label}: edge weights contain NaN/Inf or are not floating point")

    # The production Static conversion must be a zero-copy final-snapshot view.
    from src.training.train_static_gcn import convert_temporal_final_snapshot_to_static

    static_dataset = convert_temporal_final_snapshot_to_static(dataset)
    require(list(static_dataset) == expected_ids, "Static extraction changed universe order")
    for universe_id in expected_ids:
        temporal = dataset[universe_id]
        static = static_dataset[universe_id]
        require(static["X"] is temporal["Nodes_list"][-1], f"{universe_id}: Static X is not exact final X")
        require(static["edge_index"] is temporal["edge_index_list"][-1],
                f"{universe_id}: Static edges are not exact final edges")
        require(static["mask"] is temporal["mask_list"][-1],
                f"{universe_id}: Static mask is not exact final mask")
        require(static["target"] is temporal["target"], f"{universe_id}: Static target was copied")
        require(static["snapshot"] is temporal["snapshots"][-1],
                f"{universe_id}: Static metadata is not exact final metadata")
    del static_dataset


def validate(repo_root: Path, dataset_path: Path, target_path: Path) -> None:
    repo_root = repo_root.resolve()
    dataset_path = dataset_path.resolve()
    target_path = target_path.resolve()
    expected_dataset = (repo_root / EXPECTED_RELATIVE_DATASET).resolve()
    expected_target = (repo_root / EXPECTED_RELATIVE_TARGET).resolve()
    require(dataset_path == expected_dataset, f"dataset path must be {expected_dataset}")
    require(target_path == expected_target, f"target path must be {expected_target}")
    require(dataset_path.is_file(), f"expected output does not exist: {dataset_path}")
    require(target_path.is_file(), f"target table does not exist: {target_path}")

    metadata_path = dataset_path.with_suffix(".metadata.json")
    complete_path = dataset_path.with_suffix(".complete")
    lock_path = dataset_path.with_suffix(dataset_path.suffix + ".lock")
    require(metadata_path.is_file(), f"atomic metadata is missing: {metadata_path}")
    require(complete_path.is_file(), f"atomic completion marker is missing: {complete_path}")
    require(not lock_path.exists(), f"atomic lock is still present: {lock_path}")
    temp_paths = list(dataset_path.parent.glob(f".{dataset_path.name}.*.tmp"))
    temp_paths += list(dataset_path.parent.glob(f".{metadata_path.name}.*.tmp"))
    require(not temp_paths, f"temporary atomic files are present: {temp_paths}")

    require(sha256_file(target_path) == EXPECTED_TARGET_SHA256, "authoritative target SHA-256 mismatch")
    checksum = sha256_file(dataset_path)
    metadata = load_json(metadata_path)
    complete = load_json(complete_path)
    check_exact_metadata(metadata, complete, checksum)
    check_manifest(metadata, target_path)
    targets = load_targets(target_path)

    try:
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(dataset_path, map_location="cpu")
    check_dataset(dataset, metadata, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only post-build integrity validator for the approved U1000 Top1000 sparse dataset."
    )
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root,
                        help="Repository root (default: inferred from this script).")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset path (default: the approved production path under repo-root).")
    parser.add_argument("--target", type=Path, default=None,
                        help="Target CSV path (default: outputs/target_inspection_1000u.csv).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset = args.dataset or repo_root / EXPECTED_RELATIVE_DATASET
    target = args.target or repo_root / EXPECTED_RELATIVE_TARGET
    try:
        validate(repo_root, dataset, target)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
