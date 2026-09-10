#!/usr/bin/env python3
"""Prepare and audit the matched Top1500 kNN Static-GCN training matrix.

This utility never imports a trainer, loads a graph tensor, allocates a GPU, or
creates an experiment directory.  ``--prepare`` deterministically rebinds the
authoritative k=8 ordered Train700 partitions to k=4,6,12 and copies the
corresponding frozen Static-GCN configs with dataset/split/run identity changes
only.  ``--audit`` is read-only and fails closed on any identity, protocol, or
output-collision mismatch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.split_manifest import canonical_manifest_sha256, ordered_id_hash


SEEDS = (42, 123, 2025)
VARIANT_K = (4, 6, 12)
EXPECTED_COUNTS = {"train": 700, "val": 99, "test": 201, "unused": 0}
EXPECTED_FEATURES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]
TARGET_SHA256 = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
SOURCE_MANIFEST_SHA256 = "ba22c3611a70763566ffb38a20f9b5a36fb6c1a27c3ad8030c4a7e189ce87618"
K8_DATASET_SHA256 = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
K8_METADATA_SHA256 = "b76e1ae9c049ce3cff9a625a8acb6af7b53018b7bf0d02308b5f526148aef0ba"
K8_MARKER_SHA256 = "cb8a14789bb84b96ab5274ba659fe58d9db4adb3a92965fc1dce29c2a9cd85a6"
CONFIG_DIRECTORY = Path("configs/production/u1000_top1500_knn_k_ablation")
TOOL_PATH = "scripts/validation/prepare_u1000_top1500_knn_training.py"
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")

CONFIG_ALLOWED_DIFFERENCES = {
    "completion_marker_sha256": "REQUIRED DATASET BINDING",
    "dataset_path": "REQUIRED DATASET BINDING",
    "dataset_sha256": "REQUIRED DATASET BINDING",
    "experiment_name": "REQUIRED RUN IDENTITY",
    "metadata_sha256": "REQUIRED DATASET BINDING",
    "split_manifest_path": "REQUIRED SPLIT BINDING",
    "split_manifest_sha256": "REQUIRED SPLIT BINDING",
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
    "partition_source_manifest",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_relative_path(k: int) -> Path:
    if k == 8:
        directory = "temporal_1000u_none_top1500_periodic_knn_sparse"
        filename = "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
    else:
        directory = f"temporal_1000u_none_top1500_periodic_knn_k{k}_sparse"
        filename = f"camels_1000u_temporal_logmass_none_top1500_periodic_knn_k{k}_sparse.pt"
    return Path("data/processed") / directory / filename


def logical_dataset_id(k: int) -> str:
    return f"camels_simba_u1000_top1500_temporal5_none_periodic_knn_k{k}_box25_sparse_v1"


def split_relative_path(k: int, seed: int) -> Path:
    return Path(f"configs/splits/u1000_top1500_none_k{k}_sparse/seed{seed}_train700.json")


def control_config_relative_path(seed: int) -> Path:
    name = f"static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final.json"
    return Path("configs/production/u1000_top1500_training_scaling") / name


def experiment_name(k: int, seed: int) -> str:
    return (
        f"static_gcn_u1000_top1500_knn_k{k}_sparse_train700_seed{seed}"
        "_none_h32_l3_mean_mlp_final"
    )


def config_relative_path(k: int, seed: int) -> Path:
    return CONFIG_DIRECTORY / f"{experiment_name(k, seed)}.json"


def validate_dataset(k: int, *, hash_dataset: bool) -> dict[str, Any]:
    relative = dataset_relative_path(k)
    dataset = ROOT / relative
    metadata_path = dataset.with_suffix(".metadata.json")
    marker_path = dataset.with_suffix(".complete")
    lock_path = dataset.with_suffix(".lock")
    require(dataset.is_file(), f"missing dataset: {relative}")
    require(metadata_path.is_file(), f"missing metadata: {metadata_path.relative_to(ROOT)}")
    require(marker_path.is_file(), f"missing completion marker: {marker_path.relative_to(ROOT)}")
    require(not lock_path.exists(), f"dataset lock exists: {lock_path.relative_to(ROOT)}")
    temporary = sorted(
        path for path in dataset.parent.iterdir()
        if path.name.startswith(dataset.name) and
        (".tmp" in path.name or ".partial" in path.name)
    )
    require(not temporary, f"temporary dataset publication files exist: {temporary}")

    metadata = read_json(metadata_path)
    marker = read_json(marker_path)
    dataset_sha = metadata.get("checksum")
    require(isinstance(dataset_sha, str) and SHA_PATTERN.fullmatch(dataset_sha), "invalid dataset SHA")
    if hash_dataset:
        require(sha256_file(dataset) == dataset_sha, f"k={k}: dataset checksum mismatch")
    metadata_sha = sha256_file(metadata_path)
    marker_sha = sha256_file(marker_path)
    require(marker == {
        "status": "complete",
        "dataset": dataset.name,
        "metadata": metadata_path.name,
        "sha256": dataset_sha,
    }, f"k={k}: invalid completion marker")

    expected = {
        "k": k,
        "num_universes_requested": 1000,
        "num_universes_successful": 1000,
        "num_universes_failed": 0,
        "num_snapshots": 5,
        "snapshot_ids": [0.2, 0.25, 0.51209, 0.75065, 1.0],
        "num_nodes": 1500,
        "feature_names": EXPECTED_FEATURES,
        "normalization": "none",
        "target_mode": "real_targets_csv",
        "target_normalization": "none",
        "used_dummy_target": False,
        "graph_mode": "knn",
        "periodic_boundary": True,
        "periodic_boundary_knn": True,
        "box_size": 25.0,
        "graph_storage": "sparse_edge_index",
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "source_suite": "CAMELS-SIMBA",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "target_source_sha256": TARGET_SHA256,
        "failed_universes": [],
    }
    for field, expected_value in expected.items():
        require(metadata.get(field) == expected_value,
                f"k={k}: metadata {field} expected {expected_value!r}, got {metadata.get(field)!r}")
    require(metadata.get("output_path") == relative.as_posix(), f"k={k}: output path mismatch")
    require(metadata.get("completion_status") == "complete", f"k={k}: metadata not complete")
    require(metadata.get("source_manifest_hash") == SOURCE_MANIFEST_SHA256,
            f"k={k}: source-manifest alias mismatch")
    require(metadata.get("logical_dataset_id", logical_dataset_id(k)) == logical_dataset_id(k),
            f"k={k}: logical dataset ID mismatch")
    require(metadata.get("builder_source_sha256") ==
            "e7eba09615c6965d8d03d1004346dc87ab0fcbf1ebe9e335787f74852a171955",
            f"k={k}: builder SHA mismatch")

    if k == 8:
        require(dataset.resolve() ==
                (Path("/home/ml/thesis-camels") / relative).resolve(),
                "authoritative k8 symlink target changed")
        require(dataset_sha == K8_DATASET_SHA256, "authoritative k8 dataset identity changed")
        require(metadata_sha == K8_METADATA_SHA256, "authoritative k8 metadata identity changed")
        require(marker_sha == K8_MARKER_SHA256, "authoritative k8 marker identity changed")

    return {
        "dataset_path": relative.as_posix(),
        "dataset_sha256": dataset_sha,
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "metadata_sha256": metadata_sha,
        "completion_marker_path": marker_path.relative_to(ROOT).as_posix(),
        "completion_marker_sha256": marker_sha,
        "logical_dataset_id": logical_dataset_id(k),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "target_table_sha256": TARGET_SHA256,
        "source_git_commit": metadata["source_git_commit"],
        "builder_source_sha256": metadata["builder_source_sha256"],
        "build_launcher_sha256": metadata["build_launcher_sha256"],
    }


def check_split(manifest: Mapping[str, Any], *, seed: int, label: str) -> None:
    require(manifest.get("seed") == seed, f"{label}: seed mismatch")
    counts = manifest.get("counts")
    require(isinstance(counts, Mapping), f"{label}: missing counts")
    populations: list[set[str]] = []
    all_ids: list[str] = []
    for partition in ("train", "val", "test"):
        ids = manifest.get(f"{partition}_ids")
        require(isinstance(ids, list), f"{label}: {partition}_ids must be a list")
        require(len(ids) == EXPECTED_COUNTS[partition], f"{label}: wrong {partition} count")
        require(counts.get(partition) == len(ids), f"{label}: stored {partition} count mismatch")
        require(len(ids) == len(set(ids)), f"{label}: duplicate {partition} IDs")
        require(manifest.get("split_hashes", {}).get(partition) == ordered_id_hash(ids),
                f"{label}: ordered {partition} hash mismatch")
        populations.append(set(ids))
        all_ids.extend(ids)
    require(counts.get("unused") == 0 and manifest.get("unused_ids") == [],
            f"{label}: unused partition mismatch")
    require(all(not left & right for index, left in enumerate(populations)
                for right in populations[index + 1:]), f"{label}: split overlap")
    require(len(all_ids) == len(set(all_ids)) == 1000, f"{label}: population is not 1000 unique IDs")
    require(set(all_ids) == {f"LH_{index}" for index in range(1000)},
            f"{label}: universe population mismatch")
    require(manifest.get("canonical_manifest_sha256") == canonical_manifest_sha256(dict(manifest)),
            f"{label}: canonical manifest hash mismatch")


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


def make_split(control: Mapping[str, Any], binding: Mapping[str, Any], *, k: int, seed: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(control))
    result["dataset_identity"] = binding["dataset_sha256"]
    result["dataset_binding"] = dataset_binding(binding)
    result["graph_protocol_summary"]["k"] = k
    result["partition_source_manifest"] = control["canonical_manifest_sha256"]
    result["creation_metadata"]["dataset_rebinding"] = {
        "tool": TOOL_PATH,
        "ordered_partition_authority": split_relative_path(8, seed).as_posix(),
        "method": "exact ordered ID preservation with dataset-identity rebinding only",
    }
    result["canonical_manifest_sha256"] = canonical_manifest_sha256(result)
    check_split(result, seed=seed, label=f"candidate k={k} seed={seed}")
    return result


def make_config(
    control: Mapping[str, Any], binding: Mapping[str, Any], split_path: Path, *, k: int, seed: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(control))
    result.update({
        "experiment_name": experiment_name(k, seed),
        "dataset_path": binding["dataset_path"],
        "dataset_sha256": binding["dataset_sha256"],
        "metadata_sha256": binding["metadata_sha256"],
        "completion_marker_sha256": binding["completion_marker_sha256"],
        "split_manifest_path": split_path.relative_to(ROOT).as_posix(),
        "split_manifest_sha256": sha256_file(split_path),
    })
    return result


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    return {prefix: value}


def differing_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    left_flat, right_flat = flatten(left), flatten(right)
    return {path for path in set(left_flat) | set(right_flat)
            if left_flat.get(path) != right_flat.get(path)}


def audit_collision(run_name: str) -> str:
    matches: list[Path] = []
    for root_name in ("experiments", "outputs", "checkpoints", "logs"):
        root = ROOT / root_name
        if root.is_dir():
            matches.extend(path for path in root.rglob("*") if run_name in path.name)
    require(not matches, f"run collision for {run_name}: {[str(path) for path in matches]}")
    return "ABSENT"


def validate_model_resolution(config: Mapping[str, Any], *, label: str) -> None:
    """Resolve the frozen model on CPU without loading data or running a forward pass."""
    from src.models.static_gcn import StaticGCNRegressor, count_parameters

    model = StaticGCNRegressor(
        node_features=config["node_features"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        graph_pooling=config["graph_pooling"],
        conv_type=config["conv_type"],
        add_self_loops=config["add_self_loops"],
        use_layer_norm=config["use_layer_norm"],
        residual=config["residual"],
    )
    linear_shapes = [
        (module.in_features, module.out_features)
        for module in model.regressor
        if hasattr(module, "in_features") and hasattr(module, "out_features")
    ]
    require(linear_shapes == [(32, 32), (32, 16), (16, 1)],
            f"{label}: regression head did not resolve to 32->32->16->1")
    require(count_parameters(model) == 5281, f"{label}: unexpected trainable parameter count")


def validate_k8_control(seed: int, control_config: Mapping[str, Any]) -> None:
    """Verify reusable k8 artifacts without reading any metric values."""
    run_name = control_config["experiment_name"]
    local_run = ROOT / "experiments" / run_name
    canonical_run = Path("/home/ml/thesis-camels/experiments") / run_name
    required_local = (
        "config.json", "metrics.json", "train_log.csv",
        "predictions/train_predictions.csv", "predictions/val_predictions.csv",
        "predictions/test_predictions.csv",
    )
    for relative in required_local:
        require((local_run / relative).is_file(), f"k8 seed={seed}: missing {local_run / relative}")
    require((canonical_run / "checkpoints/best_model.pt").is_file(),
            f"k8 seed={seed}: canonical checkpoint missing")
    produced = read_json(local_run / "config.json")
    expected = {
        "model": control_config["model"],
        "dataset_path": control_config["dataset_path"],
        "dataset_format": control_config["dataset_format"],
        "experiment_name": run_name,
        "seed": seed,
        "batch_size": control_config["batch_size"],
        "epochs": control_config["epochs"],
        "patience": control_config["patience"],
        "learning_rate": control_config["learning_rate"],
        "weight_decay": control_config["weight_decay"],
        "hidden_dim": control_config["hidden_dim"],
        "num_layers": control_config["num_layers"],
        "dropout": control_config["dropout"],
        "graph_pooling": control_config["graph_pooling"],
        "conv_type": control_config["conv_type"],
        "split_source": control_config["split_manifest_path"],
        "dataset_identity": control_config["dataset_sha256"],
        "trainer_invocation_seed": seed,
        "split_manifest_sha256": control_config["split_manifest_sha256"],
        "split_manifest_seed": seed,
        "grad_clip_norm": control_config["grad_clip_norm"],
        "optimizer": control_config["optimizer"],
        "scheduler": control_config["scheduler"],
        "checkpoint_criterion": control_config["checkpoint_criterion"],
        "deterministic_seed_handling": control_config["deterministic_seed_handling"],
        "num_total_universes": 1000,
        "num_train_universes": 700,
        "num_val_universes": 99,
        "num_test_universes": 201,
        "num_nodes": 1500,
        "node_features": 7,
        "trainable_parameters": 5281,
    }
    for field, expected_value in expected.items():
        require(produced.get(field) == expected_value,
                f"k8 seed={seed}: produced config {field} mismatch")


def prepare() -> None:
    bindings = {k: validate_dataset(k, hash_dataset=True) for k in (4, 6, 8, 12)}
    destinations = [ROOT / split_relative_path(k, seed) for k in VARIANT_K for seed in SEEDS]
    destinations += [ROOT / config_relative_path(k, seed) for k in VARIANT_K for seed in SEEDS]
    existing = [path.relative_to(ROOT) for path in destinations if path.exists()]
    require(not existing, f"refusing to overwrite destinations: {existing}")

    splits: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in SEEDS:
        control = read_json(ROOT / split_relative_path(8, seed))
        check_split(control, seed=seed, label=f"authoritative k8 seed={seed}")
        for k in VARIANT_K:
            splits[k, seed] = make_split(control, bindings[k], k=k, seed=seed)

    for (k, seed), manifest in splits.items():
        path = ROOT / split_relative_path(k, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

    for seed in SEEDS:
        control = read_json(ROOT / control_config_relative_path(seed))
        for k in VARIANT_K:
            split_path = ROOT / split_relative_path(k, seed)
            config = make_config(control, bindings[k], split_path, k=k, seed=seed)
            path = ROOT / config_relative_path(k, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
                handle.write("\n")
    print("PASS: created exactly 9 matched split manifests and 9 frozen-protocol configs")


def audit(*, hash_datasets: bool) -> None:
    bindings = {k: validate_dataset(k, hash_dataset=hash_datasets) for k in (4, 6, 8, 12)}
    rows = 0
    for seed in SEEDS:
        control_split_path = ROOT / split_relative_path(8, seed)
        control_split = read_json(control_split_path)
        check_split(control_split, seed=seed, label=f"authoritative k8 seed={seed}")
        control_config = read_json(ROOT / control_config_relative_path(seed))
        require(control_config["split_manifest_sha256"] == sha256_file(control_split_path),
                f"k8 seed={seed}: authoritative config/split checksum mismatch")
        require(control_config["dataset_sha256"] == bindings[8]["dataset_sha256"],
                f"k8 seed={seed}: authoritative config/dataset checksum mismatch")
        validate_model_resolution(control_config, label=f"authoritative k8 seed={seed}")
        validate_k8_control(seed, control_config)

        for k in VARIANT_K:
            split_path = ROOT / split_relative_path(k, seed)
            manifest = read_json(split_path)
            check_split(manifest, seed=seed, label=f"k={k} seed={seed}")
            for partition in ("train", "val", "test"):
                require(manifest[f"{partition}_ids"] == control_split[f"{partition}_ids"],
                        f"k={k} seed={seed}: ordered {partition} IDs differ from k8")
            split_differences = differing_paths(manifest, control_split)
            require(split_differences == SPLIT_ALLOWED_DIFFERENCES,
                    f"k={k} seed={seed}: split differences changed: {sorted(split_differences)}")
            require(manifest["dataset_binding"] == dataset_binding(bindings[k]),
                    f"k={k} seed={seed}: split dataset binding mismatch")

            config_path = ROOT / config_relative_path(k, seed)
            config = read_json(config_path)
            config_differences = differing_paths(config, control_config)
            require(config_differences == set(CONFIG_ALLOWED_DIFFERENCES),
                    f"k={k} seed={seed}: unexpected config differences: {sorted(config_differences)}")
            expected_config = make_config(control_config, bindings[k], split_path, k=k, seed=seed)
            require(config == expected_config, f"k={k} seed={seed}: config is not the exact frozen derivation")
            require(config["seed"] == manifest["seed"] == seed, f"k={k} seed={seed}: seed mismatch")
            require(f"seed{seed}" in config["experiment_name"], f"k={k} seed={seed}: run-name seed mismatch")
            require(f"knn_k{k}" in config["experiment_name"], f"k={k} seed={seed}: run-name k mismatch")
            require(config["snapshot_selection"] == "exact_final_snapshot_index_minus_one",
                    f"k={k} seed={seed}: final snapshot not selected")
            require(config["node_features"] == 7 and config["node_normalization"] == "none",
                    f"k={k} seed={seed}: feature protocol mismatch")
            require(config["target"] == "Omega_m" and config["target_normalization"] == "none",
                    f"k={k} seed={seed}: target protocol mismatch")
            require(config["counts"] == EXPECTED_COUNTS, f"k={k} seed={seed}: config counts mismatch")
            require(config["dataset_sha256"] == manifest["dataset_identity"] == bindings[k]["dataset_sha256"],
                    f"k={k} seed={seed}: dataset/split/config cross-binding mismatch")
            require(config["split_manifest_sha256"] == sha256_file(split_path),
                    f"k={k} seed={seed}: config split checksum mismatch")
            validate_model_resolution(config, label=f"k={k} seed={seed}")
            audit_collision(config["experiment_name"])
            rows += 1
            print(
                f"PASS k={k} seed={seed}: IDs/order/counts, dataset/split/config binding, "
                "frozen protocol, and no run collision"
            )
    require(rows == 9, f"expected 9 audited runs, got {rows}")
    print("PASS: 9/9 pre-training cells audited; unexpected config differences=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Create the 18 requested JSON files.")
    mode.add_argument("--audit", action="store_true", help="Read-only fail-closed audit.")
    parser.add_argument(
        "--skip-dataset-rehash", action="store_true",
        help="Use metadata checksum binding after an earlier full streaming-hash audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.prepare:
            require(not args.skip_dataset_rehash, "--prepare requires full dataset rehashing")
            prepare()
        else:
            audit(hash_datasets=not args.skip_dataset_rehash)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
