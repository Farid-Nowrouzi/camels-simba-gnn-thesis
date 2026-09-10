#!/usr/bin/env python3
"""Prepare and audit the three matched radius Static-GCN runs without training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import prepare_u1000_top1500_knn_training as knn
from scripts.validation import validate_u1000_top1500_radius_dataset as radius_validator
from src.training.split_manifest import canonical_manifest_sha256


SEEDS = (42, 123, 2025)
RADIUS = 1.9628219604492188
RADIUS_TOKEN = "r1p962822"
DATASET_SHA256 = "b719eed2b09de1e91e132a357c39100cae835489042639ccffb51aa91250db42"
METADATA_SHA256 = "3c3d978e29ed7f86cdffaf39cb502896692eaa40c6f0e6dd44de87c2dc046891"
MARKER_SHA256 = "05bda9e9cf9cfde7cd3619c4d696848064b24c4f63c0be7ebde43063ef18999b"
LOGICAL_DATASET_ID = (
    "camels_simba_u1000_top1500_temporal5_none_periodic_"
    "radius_r1p962822_box25_sparse_v1"
)
DATASET = Path(
    "data/processed/temporal_1000u_none_top1500_periodic_radius_r1p962822_sparse/"
    "camels_1000u_temporal_logmass_none_top1500_periodic_radius_r1p962822_sparse.pt"
)
SPLIT_DIRECTORY = Path("configs/splits/u1000_top1500_none_radius_r1p962822_sparse")
CONFIG_DIRECTORY = Path("configs/production/u1000_top1500_radius_graph_ablation")
TOOL_PATH = "scripts/validation/prepare_u1000_top1500_radius_training.py"
EXPECTED_K8_TEST_MAE = {
    42: 0.038196001405739664,
    123: 0.036829303894469985,
    2025: 0.03863455318099824,
}

SPLIT_ALLOWED_DIFFERENCES = {
    "canonical_manifest_sha256",
    "creation_metadata.dataset_rebinding.method",
    "creation_metadata.dataset_rebinding.ordered_partition_authority",
    "creation_metadata.dataset_rebinding.tool",
    "dataset_binding.completion_marker_path",
    "dataset_binding.completion_marker_sha256",
    "dataset_binding.dataset_path",
    "dataset_binding.dataset_sha256",
    "dataset_binding.logical_dataset_id",
    "dataset_binding.metadata_path",
    "dataset_binding.metadata_sha256",
    "dataset_identity",
    "graph_protocol_summary.k",
    "graph_protocol_summary.mode",
    "graph_protocol_summary.radius",
    "graph_protocol_summary.radius_units",
    "graph_protocol_summary.threshold_rule",
    "partition_source_manifest",
}
CONFIG_ALLOWED_DIFFERENCES = set(knn.CONFIG_ALLOWED_DIFFERENCES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def split_relative_path(seed: int) -> Path:
    return SPLIT_DIRECTORY / f"seed{seed}_train700.json"


def experiment_name(seed: int) -> str:
    return (
        f"static_gcn_u1000_top1500_radius_{RADIUS_TOKEN}_sparse_train700_seed{seed}"
        "_none_h32_l3_mean_mlp_final"
    )


def config_relative_path(seed: int) -> Path:
    return CONFIG_DIRECTORY / f"{experiment_name(seed)}.json"


def control_split_relative_path(seed: int) -> Path:
    return knn.split_relative_path(8, seed)


def control_config_relative_path(seed: int) -> Path:
    return knn.control_config_relative_path(seed)


def radius_graph_protocol() -> dict[str, Any]:
    return {
        "box_size": 25.0,
        "mode": "radius",
        "periodic_boundary": True,
        "radius": RADIUS,
        "radius_units": "h^-1 Mpc",
        "source_suite": "CAMELS-SIMBA",
        "threshold_rule": "distance <= radius",
        "top_n": 1500,
    }


def validate_radius_artifact(*, hash_dataset: bool) -> dict[str, Any]:
    dataset = ROOT / DATASET
    metadata_path = dataset.with_suffix(".metadata.json")
    marker_path = dataset.with_suffix(".complete")
    require(dataset.is_file() and metadata_path.is_file() and marker_path.is_file(),
            "radius artifact set is incomplete")
    require(not Path(f"{dataset}.lock").exists(), "radius dataset lock remains")
    if hash_dataset:
        require(knn.sha256_file(dataset) == DATASET_SHA256, "radius dataset SHA-256 mismatch")
    require(knn.sha256_file(metadata_path) == METADATA_SHA256, "radius metadata SHA-256 mismatch")
    require(knn.sha256_file(marker_path) == MARKER_SHA256, "radius marker SHA-256 mismatch")

    metadata = knn.read_json(metadata_path)
    marker = knn.read_json(marker_path)
    require(marker == {
        "status": "complete",
        "dataset": dataset.name,
        "metadata": metadata_path.name,
        "sha256": DATASET_SHA256,
    }, "radius completion marker binding mismatch")
    expected = {
        "checksum": DATASET_SHA256,
        "logical_dataset_id": LOGICAL_DATASET_ID,
        "source_manifest_sha256": knn.SOURCE_MANIFEST_SHA256,
        "target_source_sha256": knn.TARGET_SHA256,
        "graph_mode": "radius",
        "radius": RADIUS,
        "radius_units": "h^-1 Mpc",
        "threshold_rule": "distance <= radius",
        "num_universes_requested": 1000,
        "num_universes_successful": 1000,
        "num_universes_failed": 0,
        "num_snapshots": 5,
        "snapshot_ids": [0.2, 0.25, 0.51209, 0.75065, 1.0],
        "num_nodes": 1500,
        "feature_names": knn.EXPECTED_FEATURES,
        "normalization": "none",
        "target_normalization": "none",
        "periodic_boundary": True,
        "box_size": 25.0,
        "graph_storage": "sparse_edge_index",
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "output_path": DATASET.as_posix(),
        "completion_status": "complete",
    }
    for field, value in expected.items():
        require(metadata.get(field) == value,
                f"radius metadata {field} expected {value!r}, got {metadata.get(field)!r}")
    return {
        "dataset_path": DATASET.as_posix(),
        "dataset_sha256": DATASET_SHA256,
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "metadata_sha256": METADATA_SHA256,
        "completion_marker_path": marker_path.relative_to(ROOT).as_posix(),
        "completion_marker_sha256": MARKER_SHA256,
        "logical_dataset_id": LOGICAL_DATASET_ID,
        "source_manifest_sha256": knn.SOURCE_MANIFEST_SHA256,
        "target_table_sha256": knn.TARGET_SHA256,
    }


def dataset_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": binding["dataset_path"],
        "dataset_sha256": binding["dataset_sha256"],
        "metadata_path": binding["metadata_path"],
        "metadata_sha256": binding["metadata_sha256"],
        "completion_marker_path": binding["completion_marker_path"],
        "completion_marker_sha256": binding["completion_marker_sha256"],
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "graph_storage_mode": "sparse_edge_index",
        "logical_dataset_id": binding["logical_dataset_id"],
        "source_manifest_policy": "full_sha256",
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "target_table_path": "outputs/target_inspection_1000u.csv",
        "target_table_sha256": binding["target_table_sha256"],
    }


def make_split(control: Mapping[str, Any], binding: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(control))
    result["dataset_identity"] = binding["dataset_sha256"]
    result["dataset_binding"] = dataset_binding(binding)
    result["graph_protocol_summary"] = radius_graph_protocol()
    result["partition_source_manifest"] = control["canonical_manifest_sha256"]
    result["creation_metadata"]["dataset_rebinding"] = {
        "tool": TOOL_PATH,
        "ordered_partition_authority": control_split_relative_path(seed).as_posix(),
        "method": "exact ordered ID preservation with radius dataset-identity rebinding only",
    }
    result["canonical_manifest_sha256"] = canonical_manifest_sha256(result)
    knn.check_split(result, seed=seed, label=f"candidate radius seed={seed}")
    return result


def make_config(
    control: Mapping[str, Any], binding: Mapping[str, Any], split_path: Path, *, seed: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(control))
    result.update({
        "experiment_name": experiment_name(seed),
        "dataset_path": binding["dataset_path"],
        "dataset_sha256": binding["dataset_sha256"],
        "metadata_sha256": binding["metadata_sha256"],
        "completion_marker_sha256": binding["completion_marker_sha256"],
        "split_manifest_path": split_path.relative_to(ROOT).as_posix(),
        "split_manifest_sha256": knn.sha256_file(split_path),
    })
    return result


def validate_k8_control_identity(seed: int, control_config: Mapping[str, Any]) -> None:
    knn.validate_k8_control(seed, control_config)
    metrics_path = ROOT / "experiments" / control_config["experiment_name"] / "metrics.json"
    metrics = knn.read_json(metrics_path)
    actual = metrics.get("test", {}).get("mae")
    require(isinstance(actual, (int, float)) and
            math.isclose(float(actual), EXPECTED_K8_TEST_MAE[seed], rel_tol=0.0, abs_tol=1e-15),
            f"k8 seed={seed}: historical test-MAE identity mismatch")


def prepare() -> None:
    binding = validate_radius_artifact(hash_dataset=True)
    destinations = [ROOT / split_relative_path(seed) for seed in SEEDS]
    destinations += [ROOT / config_relative_path(seed) for seed in SEEDS]
    existing = [path.relative_to(ROOT) for path in destinations if path.exists()]
    require(not existing, f"refusing to overwrite destinations: {existing}")

    splits: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        control = knn.read_json(ROOT / control_split_relative_path(seed))
        knn.check_split(control, seed=seed, label=f"authoritative k8 seed={seed}")
        splits[seed] = make_split(control, binding, seed=seed)

    for seed, manifest in splits.items():
        path = ROOT / split_relative_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

    for seed in SEEDS:
        split_path = ROOT / split_relative_path(seed)
        control = knn.read_json(ROOT / control_config_relative_path(seed))
        config = make_config(control, binding, split_path, seed=seed)
        path = ROOT / config_relative_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
    print("PASS: created exactly 3 radius split manifests and 3 matched Static-GCN configs")


def canonical_and_scientific_audit() -> None:
    dataset = ROOT / DATASET
    control = ROOT / radius_validator.CONTROL
    freeze = ROOT / radius_validator.FREEZE
    radius_validator.validate(ROOT, dataset, control, freeze)
    print("PASS: canonical radius validator")

    radius_data = radius_validator.load_torch(dataset)
    control_data = radius_validator.load_torch(control)
    require(list(radius_data) == list(control_data) == [f"LH_{i}" for i in range(1000)],
            "radius/k8 ordered universe identity mismatch")
    for universe_id in control_data:
        k8_sample, radius_sample = control_data[universe_id], radius_data[universe_id]
        require(k8_sample["target"].dtype == radius_sample["target"].dtype and
                torch.equal(k8_sample["target"], radius_sample["target"]),
                f"{universe_id}: target value/dtype mismatch")
        require(len(k8_sample["snapshots"]) == len(radius_sample["snapshots"]) == 5,
                f"{universe_id}: snapshot count mismatch")
        for index in range(5):
            require(k8_sample["Nodes_list"][index].dtype == radius_sample["Nodes_list"][index].dtype and
                    torch.equal(k8_sample["Nodes_list"][index], radius_sample["Nodes_list"][index]),
                    f"{universe_id} snapshot={index}: feature value/dtype mismatch")
            require(k8_sample["mask_list"][index].dtype == radius_sample["mask_list"][index].dtype and
                    torch.equal(k8_sample["mask_list"][index], radius_sample["mask_list"][index]),
                    f"{universe_id} snapshot={index}: mask value/dtype mismatch")
            k8_meta = k8_sample["snapshots"][index]
            radius_meta = radius_sample["snapshots"][index]
            require(k8_meta["snapshot_value"] == radius_meta["snapshot_value"],
                    f"{universe_id} snapshot={index}: snapshot identity mismatch")
            require(k8_meta["selected_halo_keys"] == radius_meta["selected_halo_keys"],
                    f"{universe_id} snapshot={index}: selected halo ordering mismatch")

    from src.models.static_gcn import StaticGCNRegressor
    from src.training.train_static_gcn import CamelsStaticGraphDataset, collate_fn

    selected_ids = ["LH_0", "LH_42", "LH_999"]
    topology_rows = []
    static_samples: dict[str, dict[str, Any]] = {}
    for universe_id in selected_ids:
        source = radius_data[universe_id]
        radius_edge = source["edge_index_list"][-1]
        k8_edge = control_data[universe_id]["edge_index_list"][-1]
        require(not torch.equal(radius_edge, k8_edge),
                f"{universe_id}: radius and k8 final topology unexpectedly identical")
        real_nodes = int(source["mask_list"][-1].sum().item())
        topology_rows.append((universe_id, int(k8_edge.shape[1]), int(radius_edge.shape[1]), real_nodes))
        static_samples[universe_id] = {
            "X": source["Nodes_list"][-1],
            "mask": source["mask_list"][-1],
            "target": source["target"],
            "edge_index": radius_edge,
        }

    dataset_fixture = CamelsStaticGraphDataset(static_samples, selected_ids[:2])
    _, graph, _, _, _ = collate_fn([dataset_fixture[0], dataset_fixture[1]])
    model = StaticGCNRegressor(
        node_features=7, hidden_dim=32, num_layers=3, dropout=0.2,
        graph_pooling="mean", conv_type="gcn", add_self_loops=True,
        use_layer_norm=True, residual=True,
    ).cpu().eval()
    with torch.no_grad():
        output = model(graph)
    require(tuple(output.shape) == (2, 1), f"CPU forward output shape changed: {tuple(output.shape)}")
    require(tuple(static_samples["LH_0"]["X"].shape) == (1500, 7), "feature shape mismatch")
    require(radius_data["LH_0"]["edge_index_list"][-1].shape[0] == 2, "edge shape mismatch")
    print(
        "PASS: scientific identity across 1000 universes x 5 snapshots; "
        f"CPU batch x={tuple(graph['x'].shape)} edge_index={tuple(graph['edge_index'].shape)} "
        f"output={tuple(output.shape)}"
    )
    for universe_id, k8_edges, radius_edges, real_nodes in topology_rows:
        print(
            f"TOPOLOGY {universe_id}: nodes={real_nodes} k8_directed_edges={k8_edges} "
            f"radius_directed_edges={radius_edges} "
            f"k8_mean_degree={k8_edges / real_nodes:.6f} "
            f"radius_mean_degree={radius_edges / real_nodes:.6f}"
        )


def audit(*, hash_dataset: bool, run_canonical: bool) -> None:
    binding = validate_radius_artifact(hash_dataset=hash_dataset)
    k8_binding = knn.validate_dataset(8, hash_dataset=False)
    if run_canonical:
        canonical_and_scientific_audit()

    rows = 0
    for seed in SEEDS:
        control_split_path = ROOT / control_split_relative_path(seed)
        control_split = knn.read_json(control_split_path)
        knn.check_split(control_split, seed=seed, label=f"authoritative k8 seed={seed}")
        control_config = knn.read_json(ROOT / control_config_relative_path(seed))
        require(control_config["dataset_sha256"] == k8_binding["dataset_sha256"],
                f"k8 seed={seed}: authoritative dataset binding mismatch")
        require(control_config["split_manifest_sha256"] == knn.sha256_file(control_split_path),
                f"k8 seed={seed}: authoritative split binding mismatch")
        validate_k8_control_identity(seed, control_config)

        split_path = ROOT / split_relative_path(seed)
        manifest = knn.read_json(split_path)
        knn.check_split(manifest, seed=seed, label=f"radius seed={seed}")
        for partition in ("train", "val", "test", "unused"):
            require(manifest[f"{partition}_ids"] == control_split[f"{partition}_ids"],
                    f"radius seed={seed}: ordered {partition} IDs differ from k8")
        split_differences = knn.differing_paths(manifest, control_split)
        require(split_differences == SPLIT_ALLOWED_DIFFERENCES,
                f"radius seed={seed}: split differences changed: {sorted(split_differences)}")
        require(manifest["dataset_binding"] == dataset_binding(binding),
                f"radius seed={seed}: split dataset binding mismatch")
        require(manifest["graph_protocol_summary"] == radius_graph_protocol(),
                f"radius seed={seed}: graph protocol mismatch")

        config_path = ROOT / config_relative_path(seed)
        config = knn.read_json(config_path)
        differences = knn.differing_paths(config, control_config)
        require(differences == CONFIG_ALLOWED_DIFFERENCES,
                f"radius seed={seed}: unexpected config differences: {sorted(differences)}")
        require(config == make_config(control_config, binding, split_path, seed=seed),
                f"radius seed={seed}: config is not the exact frozen derivation")
        require(config["seed"] == manifest["seed"] == seed, f"radius seed={seed}: seed mismatch")
        require(config["counts"] == knn.EXPECTED_COUNTS, f"radius seed={seed}: counts mismatch")
        require(config["dataset_sha256"] == manifest["dataset_identity"] == DATASET_SHA256,
                f"radius seed={seed}: dataset cross-binding mismatch")
        require(config["split_manifest_sha256"] == knn.sha256_file(split_path),
                f"radius seed={seed}: split checksum mismatch")
        require(config["snapshot_selection"] == "exact_final_snapshot_index_minus_one",
                f"radius seed={seed}: final snapshot semantics mismatch")
        require(config["node_features"] == 7 and config["node_normalization"] == "none",
                f"radius seed={seed}: feature protocol mismatch")
        require(config["target"] == "Omega_m" and config["target_normalization"] == "none",
                f"radius seed={seed}: target protocol mismatch")
        knn.validate_model_resolution(config, label=f"radius seed={seed}")
        knn.audit_collision(config["experiment_name"])
        rows += 1
        print(
            f"PASS radius seed={seed}: ordered splits, dataset binding, config fairness, "
            "model resolution, historical control identity, collision clear"
        )
    require(rows == 3, f"expected 3 audited radius runs, got {rows}")
    print("PASS: radius pre-training audit 3/3; unexpected config differences=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Create the six requested JSON files.")
    mode.add_argument("--audit", action="store_true", help="Run the read-only fail-closed audit.")
    parser.add_argument("--skip-dataset-rehash", action="store_true")
    parser.add_argument("--skip-canonical-validator", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.prepare:
            require(not args.skip_dataset_rehash, "--prepare requires full dataset rehashing")
            require(not args.skip_canonical_validator,
                    "--skip-canonical-validator is valid only for focused audit tests")
            prepare()
        else:
            audit(
                hash_dataset=not args.skip_dataset_rehash,
                run_canonical=not args.skip_canonical_validator,
            )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
