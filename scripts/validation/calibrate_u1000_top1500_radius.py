#!/usr/bin/env python3
"""Freeze a periodic radius from common-training final-snapshot geometry only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.camels_graph_utils import build_sparse_radius_edge_index
from src.data.source_manifest import sha256_file_streaming

CONTROL = Path(
    "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
)
SPLITS = {
    seed: Path(f"configs/splits/u1000_top1500_none_k8_sparse/seed{seed}_train700.json")
    for seed in (42, 123, 2025)
}
OUTPUT = Path("reports/experiment_registry/u1000_top1500_radius_selection_freeze.json")
BOX_SIZE = 25.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ordered_id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def load_membership(root: Path) -> tuple[dict[int, list[str]], list[str], dict[str, Any]]:
    train: dict[int, list[str]] = {}
    manifests: dict[str, Any] = {}
    for seed, relative in SPLITS.items():
        path = root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        ids = value.get("train_ids")
        require(isinstance(ids, list) and len(ids) == 700 and len(set(ids)) == 700,
                f"seed{seed}: expected 700 unique ordered training IDs")
        train[seed] = ids
        manifests[str(seed)] = {
            "path": relative.as_posix(),
            "sha256": sha256_file_streaming(path),
            "ordered_train_id_hash": ordered_id_hash(ids),
        }
    common_set = set.intersection(*(set(ids) for ids in train.values()))
    common_ordered = [value for value in train[42] if value in common_set]
    audit = {
        "train_sizes": {str(seed): len(ids) for seed, ids in train.items()},
        "pairwise_intersections": {
            "42_123": len(set(train[42]) & set(train[123])),
            "42_2025": len(set(train[42]) & set(train[2025])),
            "123_2025": len(set(train[123]) & set(train[2025])),
        },
        "union": len(set.union(*(set(ids) for ids in train.values()))),
        "three_way_intersection": len(common_ordered),
    }
    require(len(common_ordered) == 409, f"common training intersection is {len(common_ordered)}, expected 409")
    return train, common_ordered, {"audit": audit, "manifests": manifests}


def load_control(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), "control dataset is not a dictionary")
    return value


def final_geometry(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = sample["Nodes_list"][-1]
    node_mask = sample["mask_list"][-1]
    edge_index = sample["edge_index_list"][-1]
    real = int(node_mask.sum().item())
    require(sample.get("normalization") == "none", "calibration requires raw, unnormalized features")
    require(sample.get("graph_mode") == "knn", "control graph must be kNN")
    require(sample["snapshots"][-1].get("k") == 8, "control final snapshot is not k=8")
    require(float(sample["snapshots"][-1]["snapshot_value"]) == 1.0,
            "calibration snapshot is not final snapshot 1.0")
    positions = features[:real, 1:4].numpy().astype(np.float32, copy=False)
    require(np.isfinite(positions).all(), "calibration positions contain NaN/Inf")
    require(bool((positions >= 0).all()) and bool((positions <= BOX_SIZE).all()),
            "calibration positions fall outside the periodic box")
    return positions, np.ones((real, 1), dtype=np.float32), edge_index.numpy()


def exact_radius_edges(positions: np.ndarray, radius: float) -> np.ndarray:
    return build_sparse_radius_edge_index(
        positions, np.ones((len(positions), 1), dtype=np.float32), radius,
        periodic_boundary=True, box_size=BOX_SIZE,
    )


def periodic_tree(positions: np.ndarray) -> cKDTree:
    return cKDTree(np.mod(positions.astype(np.float64), BOX_SIZE), boxsize=BOX_SIZE)


def exact_radius_count(positions: np.ndarray, tree: cKDTree, radius: float) -> int:
    # query_pairs is only a candidate generator. The final comparison exactly
    # reproduces the builder's float32-distance inclusive threshold.
    query_radius = float(np.nextafter(np.float32(radius), np.float32(np.inf)))
    pairs = tree.query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return 0
    diff = np.abs(positions[pairs[:, 0]].astype(np.float64) - positions[pairs[:, 1]].astype(np.float64))
    diff = np.minimum(diff, BOX_SIZE - diff)
    distances = np.sqrt(np.einsum("ij,ij->i", diff, diff)).astype(np.float32)
    return int((distances <= radius).sum())


def count_at_radius(
    geometries: list[np.ndarray], trees: list[cKDTree], radius: float,
) -> int:
    return sum(
        exact_radius_count(positions, tree, radius)
        for positions, tree in zip(geometries, trees)
    )


def select_float32_radius(geometries: list[np.ndarray], target: int) -> tuple[float, list[dict[str, Any]]]:
    """Binary-search positive finite float32 radii and apply the smaller-radius tie rule."""
    low_bits = int(np.asarray(np.float32(0.0)).view(np.uint32))
    high_bits = int(np.asarray(np.float32(math.sqrt(3.0) * BOX_SIZE / 2.0)).view(np.uint32))
    cache: dict[int, int] = {}
    trees = [periodic_tree(value) for value in geometries]

    def evaluate(bits: int) -> int:
        if bits not in cache:
            radius = float(np.asarray(np.uint32(bits)).view(np.float32))
            cache[bits] = count_at_radius(geometries, trees, radius)
        return cache[bits]

    require(evaluate(high_bits) >= target, "radius search upper bound does not reach target")
    while low_bits + 1 < high_bits:
        middle = (low_bits + high_bits) // 2
        if evaluate(middle) >= target:
            high_bits = middle
        else:
            low_bits = middle
    candidates = [(low_bits, evaluate(low_bits)), (high_bits, evaluate(high_bits))]
    selected_bits, selected_count = min(candidates, key=lambda item: (abs(item[1] - target), item[0]))
    selected = float(np.asarray(np.uint32(selected_bits)).view(np.float32))

    # Human-readable points surrounding the exact crossing. These do not drive selection.
    nearby = []
    for offset in (-0.0002, -0.0001, 0.0, 0.0001, 0.0002):
        radius = float(np.float32(max(0.0, selected + offset)))
        count = count_at_radius(geometries, trees, radius)
        nearby.append({
            "radius": radius,
            "total_unordered_edges": count,
            "difference_from_target": count - target,
            "absolute_difference_from_target": abs(count - target),
            "mean_unordered_edges": count / len(geometries),
            "mean_degree": 2.0 * count / sum(len(value) for value in geometries),
        })
    require(selected_count == count_at_radius(geometries, trees, selected), "selected radius is not reproducible")
    return selected, nearby


def quantiles(values: list[float]) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.25, 0.5, 0.75, 0.9])
    return {key: float(value) for key, value in zip(("q10", "q25", "q50", "q75", "q90"), result)}


def structural_statistics(edge_sets: list[np.ndarray], node_counts: list[int]) -> dict[str, Any]:
    unordered_counts = [edges.shape[1] // 2 for edges in edge_sets]
    directed_total = sum(edges.shape[1] for edges in edge_sets)
    all_degrees: list[int] = []
    isolated = 0
    graphs_with_isolates = 0
    component_counts: list[int] = []
    largest_fractions: list[float] = []
    for edges, nodes in zip(edge_sets, node_counts):
        degree = np.bincount(edges[0], minlength=nodes) if edges.size else np.zeros(nodes, dtype=np.int64)
        all_degrees.extend(int(value) for value in degree)
        graph_isolated = int((degree == 0).sum())
        isolated += graph_isolated
        graphs_with_isolates += int(graph_isolated > 0)
        adjacency = coo_matrix(
            (np.ones(edges.shape[1], dtype=np.uint8), (edges[0], edges[1])), shape=(nodes, nodes),
        )
        count, labels = connected_components(adjacency, directed=False, return_labels=True)
        component_counts.append(int(count))
        largest_fractions.append(float(np.bincount(labels).max() / nodes))
    return {
        "graph_count": len(edge_sets),
        "total_directed_edges": directed_total,
        "total_unordered_edges": sum(unordered_counts),
        "mean_directed_edges": directed_total / len(edge_sets),
        "mean_unordered_edges": statistics.mean(unordered_counts),
        "median_unordered_edges": statistics.median(unordered_counts),
        "unordered_edge_count_sample_sd": statistics.stdev(unordered_counts),
        "edge_count_quantiles": quantiles([float(value) for value in unordered_counts]),
        "mean_degree": statistics.mean(all_degrees),
        "median_node_degree": statistics.median(all_degrees),
        "degree_sample_sd": statistics.stdev(all_degrees),
        "isolated_node_fraction": isolated / sum(node_counts),
        "graphs_with_isolated_nodes": graphs_with_isolates,
        "mean_connected_component_count": statistics.mean(component_counts),
        "max_connected_component_count": max(component_counts),
        "mean_largest_component_fraction": statistics.mean(largest_fractions),
        "min_largest_component_fraction": min(largest_fractions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--implementation-commit", default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    control_path = args.control if args.control.is_absolute() else root / args.control
    output_path = args.output if args.output.is_absolute() else root / args.output
    _, calibration_ids, membership = load_membership(root)
    control = load_control(control_path)
    require(all(value in control for value in calibration_ids), "calibration ID missing from control dataset")

    geometries: list[np.ndarray] = []
    knn_edges: list[np.ndarray] = []
    for universe_id in calibration_ids:
        positions, _, edges = final_geometry(control[universe_id])
        geometries.append(positions)
        knn_edges.append(edges)
    node_counts = [len(value) for value in geometries]
    target = sum(edges.shape[1] // 2 for edges in knn_edges)
    radius, candidates = select_float32_radius(geometries, target)
    radius_edges = [exact_radius_edges(positions, radius) for positions in geometries]
    radius_token = f"r{radius:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    output_dir = Path(f"data/processed/temporal_1000u_none_top1500_periodic_radius_{radius_token}_sparse")
    dataset_name = f"camels_1000u_temporal_logmass_none_top1500_periodic_radius_{radius_token}_sparse.pt"
    implementation_commit = args.implementation_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    script_relative = Path(__file__).resolve().relative_to(root)
    graph_relative = Path("src/data/camels_graph_utils.py")
    builder_relative = Path("src/data/build_temporal_sequences.py")
    launcher_relative = Path("scripts/production/run_u1000_top1500_radius_sparse_build.sh")
    record = {
        "schema_version": "u1000_top1500_radius_selection_freeze_v1",
        "experiment_id": "static_gcn_u1000_top1500_knn8_vs_fixed_radius",
        "calibration_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "control": {"graph_mode": "knn", "k": 8, "dataset_path": CONTROL.as_posix(),
                    "dataset_sha256": sha256_file_streaming(control_path)},
        "selected_radius": radius,
        "radius_units": "h^-1 Mpc",
        "radius_token": radius_token,
        "periodic_box_size_h^-1_mpc": BOX_SIZE,
        "threshold_convention": "float32(minimum-image Euclidean distance) <= radius",
        "calibration_snapshot": "final (scale factor 1.00000)",
        "calibration_ids": calibration_ids,
        "calibration_id_count": len(calibration_ids),
        "ordered_calibration_id_hash": ordered_id_hash(calibration_ids),
        "split_membership_audit": membership["audit"],
        "source_split_manifests": membership["manifests"],
        "structural_target": "aggregate unordered kNN(k=8) edge count",
        "structural_target_value": target,
        "candidate_table": candidates,
        "selected_structural_error": abs(sum(value.shape[1] // 2 for value in radius_edges) - target),
        "selection_rule": "minimize absolute aggregate unordered-edge difference from frozen kNN(k=8)",
        "tie_breaking_policy": "smaller radius wins",
        "knn_structural_statistics": structural_statistics(knn_edges, node_counts),
        "radius_structural_statistics": structural_statistics(radius_edges, node_counts),
        "leakage_audit": {
            "omega_m_used": False, "validation_ids_used": False,
            "validation_predictions_used": False, "validation_metrics_used": False,
            "test_ids_used": False, "test_predictions_used": False,
            "test_metrics_used": False, "predictive_performance_used": False,
        },
        "code_provenance": {
            "implementation_git_commit": implementation_commit,
            "calibration_tool_path": script_relative.as_posix(),
            "calibration_tool_sha256": sha256_file_streaming(root / script_relative),
            "graph_utility_path": graph_relative.as_posix(),
            "graph_utility_sha256": sha256_file_streaming(root / graph_relative),
            "builder_path": builder_relative.as_posix(),
            "builder_sha256": sha256_file_streaming(root / builder_relative),
            "launcher_path": launcher_relative.as_posix(),
            "launcher_sha256": sha256_file_streaming(root / launcher_relative),
        },
        "production_identity": {
            "output_directory": output_dir.as_posix(),
            "dataset_path": (output_dir / dataset_name).as_posix(),
            "metadata_path": (output_dir / dataset_name).with_suffix(".metadata.json").as_posix(),
            "completion_marker_path": (output_dir / dataset_name).with_suffix(".complete").as_posix(),
            "logical_dataset_id": f"camels_simba_u1000_top1500_temporal5_none_periodic_radius_{radius_token}_box25_sparse_v1",
        },
        "limitations": [
            "Radius matches aggregate connectivity, not each graph or node degree.",
            "Calibration uses the 409-universe common-training intersection and final snapshot only.",
            "No predictive quantity was evaluated during selection.",
        ],
        "next_action": "build U1000 Top1500 temporal5 periodic sparse radius dataset",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "selected_radius": radius, "radius_token": radius_token,
        "target_unordered_edges": target, "selected_unordered_edges": record["radius_structural_statistics"]["total_unordered_edges"],
        "output": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
