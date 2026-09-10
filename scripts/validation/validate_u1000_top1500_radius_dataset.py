#!/usr/bin/env python3
"""Fail-closed read-only validator for the frozen U1000 Top1500 radius artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.camels_graph_utils import build_sparse_radius_edge_index
from src.data.source_manifest import sha256_file_streaming
from scripts.validation.validate_u1000_top1500_knn_variants import (
    require_matching_source_manifests,
)

FREEZE = Path("reports/experiment_registry/u1000_top1500_radius_selection_freeze.json")
CONTROL = Path(
    "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
)
TARGET = Path("outputs/target_inspection_1000u.csv")
EXPECTED_TARGET_SHA256 = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_torch(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), f"dataset is not a dictionary: {path}")
    return value


def check_edge_index(edge_index: torch.Tensor, real_nodes: int, radius: float, features: torch.Tensor) -> None:
    require(torch.is_tensor(edge_index) and edge_index.dtype == torch.int64,
            "edge_index is not torch.int64")
    require(edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index shape is not [2,E]")
    if edge_index.numel():
        require(int(edge_index.min()) >= 0 and int(edge_index.max()) < real_nodes,
                "edge_index is out of bounds or reaches padding")
        require(bool((edge_index[0] != edge_index[1]).all()), "self-loop found")
    encoded = edge_index[0] * features.shape[0] + edge_index[1]
    require(torch.unique(encoded).numel() == encoded.numel(), "duplicate directed edge found")
    require(bool(torch.isin(edge_index[1] * features.shape[0] + edge_index[0], encoded).all()),
            "reverse edge missing")
    if encoded.numel() > 1:
        require(bool((encoded[1:] > encoded[:-1]).all()), "edge_index is not lexicographically sorted")
    positions = features[:real_nodes, 1:4].numpy()
    expected = torch.from_numpy(build_sparse_radius_edge_index(
        positions, torch.ones(real_nodes, 1).numpy(), radius,
        periodic_boundary=True, box_size=25.0,
    ))
    require(torch.equal(edge_index, expected), "edge_index differs from exact radius reconstruction")


def compare_dataset_pair(control: dict[str, Any], radius_data: dict[str, Any], radius: float) -> None:
    require(list(radius_data) == list(control), "universe IDs/order differs from frozen k8 control")
    for universe_id in control:
        left, right = control[universe_id], radius_data[universe_id]
        for key in (
            "feature_names", "feature_columns", "raw_feature_columns", "position_columns",
            "velocity_columns", "mass_column", "mass_feature", "node_selection", "normalization",
            "graph_positions", "num_nodes", "num_snapshots", "periodic_boundary", "box_size",
            "graph_storage", "schema_version",
        ):
            require(right.get(key) == left.get(key), f"{universe_id}: scientific field differs: {key}")
        require(right.get("graph_mode") == "radius", f"{universe_id}: graph mode is not radius")
        require(torch.equal(right["target"], left["target"]), f"{universe_id}: target/dtype differs")
        require(len(right["edge_index_list"]) == len(left["edge_index_list"]) == 5,
                f"{universe_id}: snapshot graph count differs")
        for index in range(5):
            label = f"{universe_id} snapshot {index}"
            require(torch.equal(right["Nodes_list"][index], left["Nodes_list"][index]),
                    f"{label}: node features/dtype differ")
            require(torch.equal(right["mask_list"][index], left["mask_list"][index]),
                    f"{label}: masks/dtype differ")
            control_meta, radius_meta = left["snapshots"][index], right["snapshots"][index]
            for key in (
                "path", "snapshot_value", "feature_names", "mass_feature", "node_selection",
                "normalization", "graph_positions", "original_num_halos", "valid_num_halos",
                "selected_num_halos_before_padding", "periodic_boundary", "box_size",
                "num_real_nodes", "selection_hash_sha256", "tie_breaking_policy",
                "selected_halo_keys",
            ):
                require(radius_meta.get(key) == control_meta.get(key), f"{label}: field differs: {key}")
            require(radius_meta.get("graph_mode") == "radius" and float(radius_meta.get("radius")) == radius,
                    f"{label}: radius metadata mismatch")
            real_nodes = int(right["mask_list"][index].sum().item())
            check_edge_index(right["edge_index_list"][index], real_nodes, radius,
                             right["Nodes_list"][index])


def git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=root, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"provenance blob unavailable: {commit}:{relative}") from exc
    return hashlib.sha256(content).hexdigest()


def validate(root: Path, dataset_path: Path, control_path: Path, freeze_path: Path) -> None:
    freeze = load_json(freeze_path)
    radius = freeze.get("selected_radius")
    require(isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0,
            "freeze radius is invalid")
    identity = freeze["production_identity"]
    expected_dataset = (root / identity["dataset_path"]).resolve()
    require(dataset_path.resolve() == expected_dataset, "radius dataset path is not frozen identity")
    require(control_path.resolve() == (root / CONTROL).resolve(), "control path is not canonical k8")
    metadata_path = dataset_path.with_suffix(".metadata.json")
    marker_path = dataset_path.with_suffix(".complete")
    require(dataset_path.is_file() and metadata_path.is_file() and marker_path.is_file(),
            "dataset, metadata, or completion marker is missing")
    require(not dataset_path.with_suffix(dataset_path.suffix + ".lock").exists(), "atomic lock remains")
    metadata, marker = load_json(metadata_path), load_json(marker_path)
    checksum = sha256_file_streaming(dataset_path)
    require(metadata.get("checksum") == marker.get("sha256") == checksum, "artifact checksum mismatch")
    exact = {
        "dataset_type": "temporal_graph_sequences", "num_universes_requested": 1000,
        "num_universes_successful": 1000, "num_universes_failed": 0, "allow_partial": False,
        "num_snapshots": 5, "num_nodes": 1500, "top_n": 1500, "normalization": "none",
        "target_normalization": "none", "graph_mode": "radius", "graph_storage": "sparse_edge_index",
        "dataset_schema_version": "camels_temporal_sparse_v1", "radius": radius,
        "radius_units": "h^-1 Mpc", "threshold_rule": "distance <= radius",
        "periodic_boundary": True, "box_size": 25.0,
        "logical_dataset_id": identity["logical_dataset_id"],
        "self_loop_policy": "excluded_in_builder_model_may_add_self_loops",
        "symmetric_directed_policy": "store_both_directions_for_each_unordered_pair",
        "duplicate_policy": "no_duplicate_directed_edges",
        "edge_ordering_policy": "lexicographic_source_then_target",
        "masking_padding_policy": "mask_gt_0_real_nodes_only_padding_excluded",
    }
    for key, value in exact.items():
        require(metadata.get(key) == value, f"radius metadata mismatch: {key}")
    require(metadata.get("source_manifest_policy") == "full_sha256", "source manifest is not full SHA-256")
    require(metadata.get("target_source_sha256") == EXPECTED_TARGET_SHA256, "target source hash mismatch")
    require(metadata.get("edge_statistics", {}).get("snapshot_graph_count") == 5000,
            "edge statistics do not cover all graphs")
    control_metadata = load_json(control_path.with_suffix(".metadata.json"))
    for key in ("source_manifest_sha256", "source_manifest", "target_source_sha256",
                "ordered_universe_ids", "snapshot_ids", "selected_halo_hash"):
        if key == "source_manifest":
            require_matching_source_manifests(metadata.get(key), control_metadata.get(key))
            continue
        require(metadata.get(key) == control_metadata.get(key), f"control provenance differs: {key}")
    commit = metadata.get("source_git_commit")
    require(isinstance(commit, str) and len(commit) == 40, "source Git commit missing")
    for path_key, hash_key in (
        ("builder_source_path", "builder_source_sha256"),
        ("graph_utility_path", "graph_utility_sha256"),
        ("build_launcher_path", "build_launcher_sha256"),
    ):
        relative = metadata.get(path_key)
        require(isinstance(relative, str), f"missing provenance path: {path_key}")
        require(metadata.get(hash_key) == git_blob_sha256(root, commit, relative),
                f"recorded source hash mismatch: {relative}")
    require(sha256_file_streaming(root / TARGET) == EXPECTED_TARGET_SHA256,
            "authoritative target table hash mismatch")
    compare_dataset_pair(load_torch(control_path), load_torch(dataset_path), float(radius))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    freeze_path = args.freeze if args.freeze.is_absolute() else root / args.freeze
    freeze = load_json(freeze_path)
    dataset = args.dataset or Path(freeze["production_identity"]["dataset_path"])
    dataset_path = dataset if dataset.is_absolute() else root / dataset
    control_path = args.control if args.control.is_absolute() else root / args.control
    try:
        validate(root, dataset_path, control_path, freeze_path)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
