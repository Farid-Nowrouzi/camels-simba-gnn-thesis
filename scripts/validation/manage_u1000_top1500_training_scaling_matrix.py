#!/usr/bin/env python3
"""Prepare, bind, preflight, validate, and aggregate the Top1500 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.source_manifest import sha256_file_streaming, source_manifest_sha256
from src.training.split_manifest import canonical_manifest_sha256, ordered_id_hash

DATASET = Path("data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt")
TARGET = Path("outputs/target_inspection_1000u.csv")
SOURCE_SPLITS = Path("configs/splits/u1000_top1000_none_k8_sparse")
SPLITS = Path("configs/splits/u1000_top1500_none_k8_sparse")
CONFIG_DIR = Path("configs/production/u1000_top1500_training_scaling")
REGISTRY = Path("configs/experiment_registry/u1000_top1500_training_scaling_matrix.json")
MASTER = Path("reports/experiment_registry/master_experiment_registry.csv")
REPORTS = Path("reports/experiment_registry")
SEEDS = (42, 123, 2025)
LEVELS = (20, 50, 100, 200, 450, 700)
MODELS = ("evolve", "static")
PENDING = "PENDING_POST_BUILD"
TARGET_SHA = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
PILOT = REPORTS / "u1000_top1500_cuda_pilot_result.json"
PILOT_MANIFEST = SPLITS / "seed42_train700.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return sha256_file_streaming(path)


def read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def partition_identity(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        "".join(part + "\n" + ordered_id_hash(manifest[part + "_ids"])
                for part in ("train", "val", "test", "unused")).encode()
    ).hexdigest()


def require_bound_hash(value: Any, expected: str, artifact: str, location: str) -> None:
    require(value != PENDING, f"PENDING HASH: {artifact} at {location}")
    require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
            f"INVALID HASH: {artifact} at {location}")
    require(value == expected, f"ARTIFACT HASH MISMATCH: {artifact} at {location}")


def current_artifact_identities(root: Path = ROOT) -> dict[str, str]:
    dataset = root / DATASET
    metadata_path = dataset.with_suffix(".metadata.json")
    marker_path = dataset.with_suffix(".complete")
    target_path = root / TARGET
    for label, path in (("dataset", dataset), ("metadata", metadata_path),
                        ("completion marker", marker_path), ("target source", target_path)):
        require(path.is_file(), f"missing {label}: {path}")
    identities = {
        "dataset_sha256": sha(dataset),
        "metadata_sha256": sha(metadata_path),
        "completion_marker_sha256": sha(marker_path),
        "target_table_sha256": sha(target_path),
    }
    metadata = read(metadata_path)
    marker = read(marker_path)
    manifest = metadata.get("source_manifest")
    require(isinstance(manifest, dict), "metadata: raw-source manifest is missing")
    raw_identity = source_manifest_sha256(manifest)
    require_bound_hash(manifest.get("manifest_sha256"), raw_identity,
                       "raw-source manifest", "metadata.source_manifest.manifest_sha256")
    require_bound_hash(metadata.get("source_manifest_sha256"), raw_identity,
                       "raw-source manifest", "metadata.source_manifest_sha256")
    require_bound_hash(metadata.get("target_source_sha256"), identities["target_table_sha256"],
                       "target source", "metadata.target_source_sha256")
    require_bound_hash(metadata.get("checksum"), identities["dataset_sha256"],
                       "dataset", "metadata.checksum")
    require_bound_hash(marker.get("sha256"), identities["dataset_sha256"],
                       "dataset", "completion marker.sha256")
    require(marker.get("dataset") == dataset.name, "completion marker: dataset filename mismatch")
    require(marker.get("metadata") == metadata_path.name, "completion marker: metadata filename mismatch")
    require(marker.get("status") == "complete", "completion marker: status is not complete")
    identities["source_manifest_sha256"] = raw_identity
    return identities


def verify_artifact_bindings(root: Path = ROOT) -> dict[str, str]:
    identities = current_artifact_identities(root)
    registry_path = root / REGISTRY
    registry = read(registry_path)
    entries = registry.get("entries", [])
    require(entries, f"registry has no entries: {REGISTRY}")
    artifact_fields = (
        ("dataset_sha256", "dataset"), ("metadata_sha256", "metadata"),
        ("completion_marker_sha256", "completion marker"),
        ("target_table_sha256", "target source"),
        ("source_manifest_sha256", "raw-source manifest"),
    )
    seen_manifests: set[str] = set()
    seen_configs: set[str] = set()
    for index, item in enumerate(entries):
        location = f"registry {REGISTRY.as_posix()} entries[{index}]"
        for field, artifact in artifact_fields[:-1]:
            require_bound_hash(item.get(field), identities[field], artifact, f"{location}.{field}")
        require(item.get("master_dataset_path") == DATASET.as_posix(),
                f"dataset path mismatch at {location}.master_dataset_path")
        manifest_rel = item.get("split_manifest_path")
        require(isinstance(manifest_rel, str) and manifest_rel != "", f"missing split manifest at {location}")
        manifest_path = root / manifest_rel
        require(manifest_path.is_file(), f"missing split manifest: {manifest_rel}")
        manifest_file_sha = sha(manifest_path)
        require_bound_hash(item.get("split_manifest_sha256"), manifest_file_sha,
                           "split manifest", f"{location}.split_manifest_sha256")
        manifest = read(manifest_path)
        calculated_partition = partition_identity(manifest)
        require_bound_hash(item.get("partition_identity"), calculated_partition,
                           "ordered partition", f"{location}.partition_identity")
        require_bound_hash(item.get("split_binding_identity"), canonical_manifest_sha256(manifest),
                           "split binding", f"{location}.split_binding_identity")
        if manifest_rel not in seen_manifests:
            seen_manifests.add(manifest_rel)
            binding_location = f"split binding {manifest_rel}.dataset_binding"
            binding = manifest.get("dataset_binding")
            require(isinstance(binding, dict), f"missing dataset binding at {manifest_rel}")
            for field, artifact in artifact_fields:
                require_bound_hash(binding.get(field), identities[field], artifact,
                                   f"{binding_location}.{field}")
            require_bound_hash(manifest.get("dataset_identity"), identities["dataset_sha256"],
                               "dataset", f"split binding {manifest_rel}.dataset_identity")
            expected_paths = {
                "dataset_path": DATASET.as_posix(),
                "metadata_path": DATASET.with_suffix(".metadata.json").as_posix(),
                "completion_marker_path": DATASET.with_suffix(".complete").as_posix(),
                "target_table_path": TARGET.as_posix(),
            }
            for field, expected in expected_paths.items():
                require(binding.get(field) == expected, f"path mismatch at {binding_location}.{field}")
            require_bound_hash(manifest.get("partition_identity"), calculated_partition,
                               "ordered partition", f"split binding {manifest_rel}.partition_identity")
            require_bound_hash(manifest.get("canonical_manifest_sha256"), canonical_manifest_sha256(manifest),
                               "split binding", f"split binding {manifest_rel}.canonical_manifest_sha256")
        config_rel = item.get("configuration_path")
        require(isinstance(config_rel, str) and config_rel != "", f"missing config path at {location}")
        if config_rel not in seen_configs:
            seen_configs.add(config_rel)
            config_path = root / config_rel
            require(config_path.is_file(), f"missing model configuration: {config_rel}")
            config = read(config_path)
            config_location = f"model configuration {config_rel}"
            for field, artifact in artifact_fields:
                require_bound_hash(config.get(field), identities[field], artifact,
                                   f"{config_location}.{field}")
            require(config.get("dataset_path") == DATASET.as_posix(),
                    f"dataset path mismatch at {config_location}.dataset_path")
            require(config.get("split_manifest_path") == manifest_rel,
                    f"split manifest path mismatch at {config_location}.split_manifest_path")
            require_bound_hash(config.get("split_manifest_sha256"), manifest_file_sha,
                               "split manifest", f"{config_location}.split_manifest_sha256")
    return identities


def verify_cuda_pilot(
    identities: dict[str, str], root: Path = ROOT, pilot_path: Path | None = None,
) -> None:
    path = pilot_path or root / PILOT
    require(path.is_file(), f"Top1500 CUDA pilot evidence missing: {path}")
    pilot = read(path)
    required = {
        "schema_version", "status", "dataset_path", "dataset_sha256", "metadata_path",
        "metadata_sha256", "completion_marker_path", "completion_marker_sha256",
        "raw_source_identity", "target_source_identity", "top_n", "universe_count",
        "snapshot_count", "feature_dimension", "normalization", "periodic_flag", "k",
        "box_size", "model_names_tested", "production_batch_sizes", "seed",
        "train700_seed42_manifest_path", "manifest_sha256", "ordered_partition_identity",
        "source_git_commit", "execution_timestamp", "cuda_device_identity",
        "forward_backward_result", "finite_loss_result", "finite_gradient_result",
        "peak_memory_mib", "results",
    }
    missing = sorted(required.difference(pilot))
    require(not missing, f"CUDA PILOT MISSING REQUIRED FIELDS: {missing}")
    require(pilot["schema_version"] == "u1000_top1500_cuda_pilot_v2", "unsupported CUDA pilot schema")
    require(pilot["status"] == "PASS", "Top1500 CUDA pilot status is not PASS")
    comparisons = (
        ("dataset_sha256", "dataset SHA", "dataset_sha256"),
        ("metadata_sha256", "metadata SHA", "metadata_sha256"),
        ("completion_marker_sha256", "completion-marker SHA", "completion_marker_sha256"),
        ("raw_source_identity", "raw-source identity", "source_manifest_sha256"),
        ("target_source_identity", "target-source identity", "target_table_sha256"),
    )
    for field, label, identity_field in comparisons:
        value = pilot[field]
        require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
                f"INVALID CUDA PILOT: {label} is malformed")
        require(value == identities[identity_field],
                f"STALE CUDA PILOT: {label} does not match current Top1500 dataset")
    expected_paths = {
        "dataset_path": DATASET.as_posix(),
        "metadata_path": DATASET.with_suffix(".metadata.json").as_posix(),
        "completion_marker_path": DATASET.with_suffix(".complete").as_posix(),
        "train700_seed42_manifest_path": PILOT_MANIFEST.as_posix(),
    }
    for field, expected in expected_paths.items():
        require(pilot[field] == expected, f"STALE CUDA PILOT: {field} does not match {expected}")
    manifest_path = root / PILOT_MANIFEST
    manifest = read(manifest_path)
    require_bound_hash(pilot["manifest_sha256"], sha(manifest_path), "pilot split manifest",
                       "CUDA pilot.manifest_sha256")
    require_bound_hash(pilot["ordered_partition_identity"], partition_identity(manifest),
                       "pilot ordered partition", "CUDA pilot.ordered_partition_identity")
    require(pilot["seed"] == 42 and manifest.get("seed") == 42,
            "STALE CUDA PILOT: pilot is not bound to seed42")
    require(pilot["top_n"] == 1500 and pilot["universe_count"] == 1000 and
            pilot["snapshot_count"] == 5 and pilot["feature_dimension"] == 7,
            "STALE CUDA PILOT: dataset dimensions/protocol mismatch")
    require(pilot["normalization"] == "none" and pilot["periodic_flag"] is True and
            pilot["k"] == 8 and float(pilot["box_size"]) == 25.0,
            "STALE CUDA PILOT: graph protocol mismatch")
    require(pilot["model_names_tested"] == ["evolve", "static"] and
            pilot["production_batch_sizes"] == {"evolve": 4, "static": 8},
            "CUDA pilot did not test both exact production batch sizes")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(pilot["source_git_commit"] == commit,
            "STALE CUDA PILOT: source Git commit does not match current commit")
    require(isinstance(pilot["execution_timestamp"], str) and pilot["execution_timestamp"],
            "CUDA pilot execution timestamp is missing")
    require(isinstance(pilot["cuda_device_identity"], dict) and pilot["cuda_device_identity"],
            "CUDA pilot device identity is missing")
    for field in ("forward_backward_result", "finite_loss_result", "finite_gradient_result"):
        require(pilot[field] == "PASS", f"CUDA pilot {field} is not PASS")
    require(set(pilot["results"]) == {"evolve", "static"}, "CUDA pilot model results are incomplete")


def name(model: str, count: int, seed: int) -> str:
    if model == "evolve":
        return f"evolvegcn_h_u1000_top1500_sparse_train{count}_seed{seed}_none_h32_l2_mean_temporal_mean_linear"
    return f"static_gcn_u1000_top1500_sparse_train{count}_seed{seed}_none_h32_l3_mean_mlp_final"


def identity(model: str, count: int, seed: int) -> str:
    return f"u1000-top1500-sparse-{model}-train{count}-seed{seed}"


def template(model: str) -> dict[str, Any]:
    filename = ("evolvegcn_h_u1000_top1000_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear.json"
                if model == "evolve" else
                "static_gcn_u1000_top1000_sparse_train700_seed42_none_h32_l3_mean_mlp_final.json")
    return read(ROOT / "configs/pilots" / filename)


def create_manifest(source: dict[str, Any], dataset_binding: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(source)
    if dataset_binding is None:
        result["dataset_identity"] = PENDING
        binding = deepcopy(result["dataset_binding"])
        for key in ("dataset_sha256", "metadata_sha256", "completion_marker_sha256", "source_manifest_sha256"):
            binding[key] = PENDING
        binding.update({"dataset_path": DATASET.as_posix(),
                        "metadata_path": DATASET.with_suffix(".metadata.json").as_posix(),
                        "completion_marker_path": DATASET.with_suffix(".complete").as_posix(),
                        "logical_dataset_id": "camels_simba_u1000_top1500_temporal5_none_periodic_knn_k8_box25_sparse_v1"})
        result["dataset_binding"] = binding
    else:
        result["dataset_identity"] = dataset_binding["dataset_sha256"]
        result["dataset_binding"] = dataset_binding
    result["graph_protocol_summary"]["top_n"] = 1500
    result["partition_source_manifest"] = source.get("canonical_manifest_sha256")
    result["partition_identity"] = partition_identity(source)
    result["canonical_manifest_sha256"] = canonical_manifest_sha256(result)
    return result


def config_for(model: str, count: int, seed: int, manifest_path: Path, manifest_sha: str,
               binding: dict[str, str] | None = None) -> dict[str, Any]:
    config = template(model)
    exp_name = name(model, count, seed)
    config.update({"experiment_name": exp_name, "seed": seed,
                   "counts": {"train": count, "val": 99, "test": 201, "unused": 700-count},
                   "dataset_path": DATASET.as_posix(), "split_manifest_path": manifest_path.as_posix(),
                   "split_manifest_sha256": manifest_sha})
    for key in ("dataset_sha256", "metadata_sha256", "completion_marker_sha256", "source_manifest_sha256"):
        config[key] = binding[key] if binding else PENDING
    return config


def entry(model: str, count: int, seed: int, manifest: Path, manifest_sha: str,
          config: Path, binding: dict[str, str] | None) -> dict[str, Any]:
    exp_name = name(model, count, seed)
    return {
        "canonical_experiment_id": identity(model, count, seed), "experiment_name": exp_name,
        "model": "EvolveGCNHRegressor" if model == "evolve" else "StaticGCNRegressor",
        "model_key": model, "training_universe_count": count, "validation_count": 99,
        "test_count": 201, "unused_count": 700-count, "master_dataset_path": DATASET.as_posix(),
        "dataset_sha256": binding["dataset_sha256"] if binding else PENDING,
        "metadata_sha256": binding["metadata_sha256"] if binding else PENDING,
        "completion_marker_sha256": binding["completion_marker_sha256"] if binding else PENDING,
        "target_table_sha256": TARGET_SHA, "split_manifest_path": manifest.as_posix(),
        "split_manifest_sha256": manifest_sha, "partition_identity": read(ROOT/manifest)["partition_identity"],
        "split_binding_identity": canonical_manifest_sha256(read(ROOT/manifest)),
        "configuration_path": config.as_posix(), "experiment_directory": f"experiments/{exp_name}",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "graph_storage_mode": "sparse_edge_index", "normalization": "none", "top_n": 1500, "k": 8,
        "snapshot_input_protocol": "all_five_temporal_snapshots" if model == "evolve" else "temporal_final_snapshot_exact_a1.0",
        "seed": seed, "status": "planned", "planned_timestamp": now(), "launch_timestamp": "",
        "completion_timestamp": "", "validation_timestamp": "", "failure_reason": "",
        "checkpoint_path": "", "metrics_path": "", "predictions_path": "", "best_epoch": "",
        "epochs_executed": "", "runtime_seconds": "", "peak_gpu_memory_mib": "", "validation_result": "",
        "test_mae": "", "test_rmse": "", "test_mse": "", "test_r2": "", "prediction_sd_ratio": "",
        "repeated_prediction_fraction": "", "residual_mean": "", "residual_standard_deviation": "",
        "low_target_bias": "", "high_target_bias": "", "collapse_flag": "", "matrix_included": True,
    }


def sync_master(entries: list[dict[str, Any]]) -> None:
    path = ROOT / MASTER
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = list(reader)
    rows = [row for row in rows if not row.get("canonical_experiment_id", "").startswith("u1000-top1500-sparse-")]
    for item in entries:
        row = {key: "" for key in fields}
        row.update({"experiment_name": item["experiment_name"], "experiment_path": item["experiment_directory"],
                    "canonical_experiment_id": item["canonical_experiment_id"], "status": item["status"],
                    "experiment_type": "controlled_ablation", "experiment_family": "topn_halo_scaling",
                    "comparison_quality": "controlled", "scientific_role": "Top1000 versus Top1500 scaling slice",
                    "original_or_reproduction": "original", "notes": "U1000 Top1500 planned lifecycle entry",
                    "matrix_included": "true", "training_universe_count": item["training_universe_count"],
                    "master_dataset_path": item["master_dataset_path"], "dataset_sha256": item["dataset_sha256"],
                    "metadata_sha256": item["metadata_sha256"], "completion_marker_sha256": item["completion_marker_sha256"],
                    "target_table_sha256": TARGET_SHA, "split_manifest_path": item["split_manifest_path"],
                    "split_manifest_sha256": item["split_manifest_sha256"], "configuration_path": item["configuration_path"],
                    "experiment_directory": item["experiment_directory"], "repository_commit": item["repository_commit"],
                    "graph_storage_mode": "sparse_edge_index", "snapshot_input_protocol": item["snapshot_input_protocol"],
                    "planned_timestamp": item["planned_timestamp"], "launch_timestamp": item["launch_timestamp"],
                    "completion_timestamp": item["completion_timestamp"], "validation_timestamp": item["validation_timestamp"],
                    "failure_reason": item["failure_reason"], "checkpoint_path": item["checkpoint_path"],
                    "metrics_path": item["metrics_path"], "predictions_path": item["predictions_path"],
                    "epochs_executed": item["epochs_executed"], "runtime_seconds": item["runtime_seconds"],
                    "peak_gpu_memory_mib": item["peak_gpu_memory_mib"], "validation_result": item["validation_result"],
                    "test_mae": item["test_mae"], "test_rmse": item["test_rmse"], "test_mse": item["test_mse"],
                    "test_r2": item["test_r2"], "prediction_sd_ratio_matrix": item["prediction_sd_ratio"],
                    "repeated_prediction_fraction_matrix": item["repeated_prediction_fraction"],
                    "residual_mean": item["residual_mean"], "residual_standard_deviation": item["residual_standard_deviation"],
                    "low_target_bias": item["low_target_bias"], "high_target_bias": item["high_target_bias"],
                    "collapse_flag": item["collapse_flag"], "dataset_path": item["master_dataset_path"],
                    "dataset_filename": DATASET.name, "dataset_format": "temporal_graph_sequences", "universes": 1000,
                    "snapshots": 5, "top_n": 1500, "normalization": "none", "k": 8, "periodic": "true",
                    "box_size": 25, "node_features": 7, "target_name": "Omega_m",
                    "final_snapshot_only": "false" if item["model_key"] == "evolve" else "true",
                    "model": item["model"], "hidden_dim": 32, "num_layers": 2 if item["model_key"] == "evolve" else 3,
                    "dropout": 0.2, "activation": "relu", "graph_pooling": "mean",
                    "temporal_pooling": "mean" if item["model_key"] == "evolve" else "",
                    "head_type": "linear" if item["model_key"] == "evolve" else "mlp", "target_normalization": "false",
                    "add_self_loops": "true", "batch_size": 4 if item["model_key"] == "evolve" else 8,
                    "epochs": 300, "patience": 40, "learning_rate": 0.001, "weight_decay": 0.00001,
                    "optimizer": "AdamW", "grad_clip_norm": 1, "device": "cuda", "seed": item["seed"],
                    "train_ratio": 0.7, "val_ratio": 0.099, "test_ratio": 0.201,
                    "train_count": item["training_universe_count"], "val_count": 99, "test_count": 201,
                    "expected_seed_set": "42,123,2025", "intended_ablation_variable": "top_n",
                    "changed_variables": "top_n", "hypothesis": "Lower-mass halos may improve Omega_m prediction."})
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def prepare(binding: dict[str, str] | None = None) -> None:
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
            == "thesis-sparse-integrity-hardening", "wrong branch")
    entries = []
    for seed in SEEDS:
        for count in LEVELS:
            source_path = ROOT / SOURCE_SPLITS / f"seed{seed}_train{count}.json"
            source = read(source_path)
            manifest = create_manifest(source, binding)
            manifest_path = SPLITS / f"seed{seed}_train{count}.json"
            write_json(ROOT / manifest_path, manifest)
            manifest_sha = sha(ROOT / manifest_path)
            for model in MODELS:
                config_path = CONFIG_DIR / f"{name(model,count,seed)}.json"
                write_json(ROOT / config_path, config_for(model, count, seed, manifest_path, manifest_sha, binding))
                entries.append(entry(model, count, seed, manifest_path, manifest_sha, config_path, binding))
    registry = {"schema_version": "u1000_top1500_training_scaling_registry_v1",
                "allowed_statuses": ["planned", "running", "completed", "failed"],
                "updated_at": now(), "entries": entries}
    write_json(ROOT / REGISTRY, registry)
    inventory = ROOT / REPORTS / "u1000_top1500_matrix_inventory.csv"
    fields = ["canonical_experiment_id", "experiment_name", "model", "training_universe_count", "seed", "status",
              "dataset_sha256", "partition_identity", "split_binding_identity", "split_manifest_path",
              "split_manifest_sha256", "configuration_path", "experiment_directory"]
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(entries)
    sync_master(entries)
    validate_preparation(allow_pending=binding is None)
    print(f"PASS: prepared {len(entries)} Top1500 cells and 18 immutable partition bindings")


def dataset_binding() -> dict[str, str]:
    dataset = ROOT / DATASET; metadata = dataset.with_suffix(".metadata.json"); marker = dataset.with_suffix(".complete")
    subprocess.run([str(ROOT/"envs/camels-gnn/bin/python"), "scripts/validation/validate_u1000_top1500_sparse_dataset.py"], cwd=ROOT, check=True)
    meta = read(metadata)
    return {"dataset_path": DATASET.as_posix(), "dataset_sha256": sha(dataset),
            "metadata_path": DATASET.with_suffix(".metadata.json").as_posix(), "metadata_sha256": sha(metadata),
            "completion_marker_path": DATASET.with_suffix(".complete").as_posix(), "completion_marker_sha256": sha(marker),
            "dataset_schema_version": "camels_temporal_sparse_v1", "graph_storage_mode": "sparse_edge_index",
            "logical_dataset_id": "camels_simba_u1000_top1500_temporal5_none_periodic_knn_k8_box25_sparse_v1",
            "source_manifest_policy": "full_sha256", "source_manifest_sha256": meta["source_manifest_sha256"],
            "target_table_path": TARGET.as_posix(), "target_table_sha256": TARGET_SHA}


def validate_preparation(allow_pending: bool = True) -> None:
    reg = read(ROOT / REGISTRY); entries = reg["entries"]
    require(len(entries) == 36 and len({e["canonical_experiment_id"] for e in entries}) == 36, "registry count/IDs")
    require(len({(e["model_key"], e["training_universe_count"], e["seed"]) for e in entries}) == 36, "duplicate cells")
    for seed in SEEDS:
        for count in LEVELS:
            source = read(ROOT/SOURCE_SPLITS/f"seed{seed}_train{count}.json")
            bound_path = ROOT/SPLITS/f"seed{seed}_train{count}.json"; bound = read(bound_path)
            for part in ("train", "val", "test", "unused"):
                require(bound[f"{part}_ids"] == source[f"{part}_ids"], f"partition changed: {bound_path} {part}")
            require(bound["seed"] == seed and bound["counts"] == source["counts"], "seed/count changed")
            require(bound["canonical_manifest_sha256"] == canonical_manifest_sha256(bound), "binding canonical hash")
    for item in entries:
        cfg = read(ROOT/item["configuration_path"])
        require(cfg["experiment_name"] == item["experiment_name"] and cfg["seed"] == item["seed"], "config identity")
        require(cfg["counts"]["train"] == item["training_universe_count"], "config training count")
        require(cfg["batch_size"] == (4 if item["model_key"] == "evolve" else 8), "batch size drift")
        require(cfg["hidden_dim"] == 32 and cfg["num_layers"] == (2 if item["model_key"] == "evolve" else 3), "architecture drift")
        require(cfg["dataset_sha256"] == item["dataset_sha256"], "dataset binding drift")
        if not allow_pending: require(item["dataset_sha256"] != PENDING, "pending dataset binding")
    print("PASS: 36 unique configs; Evolve=18 Static=18; partitions identical 18/18")


def preflight() -> None:
    validate_preparation(allow_pending=True)
    dataset = ROOT / DATASET
    if not dataset.is_file():
        print(f"TOP1500 BUILD REQUIRED BEFORE CUDA: missing {DATASET}")
        raise SystemExit(3)
    identities = verify_artifact_bindings()
    verify_cuda_pilot(identities)
    print("READY FOR TOP1500 MATRIX")


def status() -> None:
    entries = read(ROOT/REGISTRY)["entries"]; counts = Counter(e["status"] for e in entries)
    duplicates = len(entries) - len({e["canonical_experiment_id"] for e in entries})
    print(f"planned={counts['planned']} running={counts['running']} completed={counts['completed']} failed={counts['failed']} duplicates={duplicates}")


def find_entry(registry: dict[str, Any], ident: str) -> dict[str, Any]:
    matches = [item for item in registry["entries"] if item["canonical_experiment_id"] == ident]
    require(len(matches) == 1, f"registry identity not unique: {ident}")
    return matches[0]


def validate_run(ident: str) -> dict[str, Any]:
    registry = read(ROOT/REGISTRY); item = find_entry(registry, ident); exp = ROOT/item["experiment_directory"]
    config_path = exp/"config.json"; metrics_path = exp/"metrics.json"; predictions_path = exp/"predictions/test_predictions.csv"
    checkpoint_path = exp/"checkpoints/best_model.pt"; log_path = exp/"train_log.csv"
    for path in (config_path, metrics_path, predictions_path, checkpoint_path, log_path):
        require(path.is_file(), f"missing run artifact: {path}")
    config = read(config_path); metrics = read(metrics_path); manifest = read(ROOT/item["split_manifest_path"])
    require(config["model"] == item["model"] and config["seed"] == item["seed"], "model/seed mismatch")
    provenance = config.get("dataset_provenance", config)
    require(provenance.get("dataset_sha256") == item["dataset_sha256"], "dataset identity mismatch")
    require(config["train_ids"] == manifest["train_ids"] and config["val_ids"] == manifest["val_ids"]
            and config["test_ids"] == manifest["test_ids"], "ordered partitions mismatch")
    with predictions_path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    ids = [row["universe_id"] for row in rows]
    require(len(rows) == 201 and ids == manifest["test_ids"] and len(set(ids)) == 201, "prediction IDs/order mismatch")
    with (ROOT/TARGET).open(newline="", encoding="utf-8") as handle:
        targets = {row["universe_id"]: float(row["omega_m"]) for row in csv.DictReader(handle)}
    true = [float(row["true_omega_m"]) for row in rows]; predicted = [float(row["pred_omega_m"]) for row in rows]
    require(all(math.isfinite(value) for value in true+predicted), "nonfinite prediction/target")
    require(all(math.isclose(value, targets[uid], rel_tol=0, abs_tol=1e-6) for uid,value in zip(ids,true)), "target table mismatch")
    residual = [p-y for p,y in zip(predicted,true)]; mse = statistics.mean(value*value for value in residual)
    mae = statistics.mean(abs(value) for value in residual); rmse = math.sqrt(mse); mean_true = statistics.mean(true)
    r2 = 1-sum(value*value for value in residual)/sum((value-mean_true)**2 for value in true)
    saved = metrics["test"]
    for key,value in (("mae",mae),("mse",mse),("rmse",rmse),("r2",r2)):
        require(math.isclose(float(saved[key]), value, rel_tol=1e-7, abs_tol=1e-9), f"metric mismatch: {key}")
    pred_sd=statistics.stdev(predicted); target_sd=statistics.stdev(true); repeated=len(predicted)-len(set(predicted))
    order=sorted(range(len(true)),key=true.__getitem__); low=set(order[:len(order)//5]); high=set(order[-len(order)//5:])
    with log_path.open(newline="", encoding="utf-8") as handle: epochs=list(csv.DictReader(handle))
    return {"validation_result":"PASS", "validation_timestamp":now(), "checkpoint_path":str(checkpoint_path.relative_to(ROOT)),
            "metrics_path":str(metrics_path.relative_to(ROOT)), "predictions_path":str(predictions_path.relative_to(ROOT)),
            "test_mae":mae,"test_mse":mse,"test_rmse":rmse,"test_r2":r2,"prediction_sd_ratio":pred_sd/target_sd,
            "repeated_prediction_fraction":repeated/len(predicted),"residual_mean":statistics.mean(residual),
            "residual_standard_deviation":statistics.stdev(residual),"low_target_bias":statistics.mean(residual[i] for i in low),
            "high_target_bias":statistics.mean(residual[i] for i in high),"collapse_flag":pred_sd/target_sd<0.1 or repeated==len(predicted)-1,
            "best_epoch":int(metrics["best_epoch"]),"epochs_executed":len(epochs)}


def set_status(ident: str, value: str, reason: str = "") -> None:
    registry=read(ROOT/REGISTRY); item=find_entry(registry,ident)
    require(value in registry["allowed_statuses"], "invalid lifecycle status")
    if value == "running": require(item["status"] == "planned", "only planned may run"); item["launch_timestamp"] = now()
    elif value == "completed": require(item["status"] == "running", "only running may complete"); item.update(validate_run(ident)); item["completion_timestamp"] = now()
    elif value == "failed": item["failure_reason"] = reason
    item["status"] = value; registry["updated_at"] = now(); write_json(ROOT/REGISTRY,registry); sync_master(registry["entries"])
    print(f"PASS: {ident} -> {value}")


def aggregate() -> None:
    entries=read(ROOT/REGISTRY)["entries"]
    require(all(item["status"]=="completed" and item["validation_result"]=="PASS" for item in entries), "aggregation requires 36/36 validated completions")
    per_run_fields=["model","training_universe_count","seed","test_mae","test_mse","test_rmse","test_r2","prediction_sd_ratio",
                    "repeated_prediction_fraction","residual_mean","residual_standard_deviation","low_target_bias","high_target_bias",
                    "collapse_flag","best_epoch","epochs_executed","runtime_seconds","peak_gpu_memory_mib"]
    with (ROOT/REPORTS/"u1000_top1500_final_per_run_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=per_run_fields,extrasaction="ignore",lineterminator="\n"); writer.writeheader(); writer.writerows(entries)
    summary=[]
    for model in ("EvolveGCNHRegressor","StaticGCNRegressor"):
        for count in LEVELS:
            group=[item for item in entries if item["model"]==model and item["training_universe_count"]==count]
            row={"model":model,"training_count":count,"seed_count":len(group)}
            for field in ("test_mae","test_mse","test_rmse","test_r2","prediction_sd_ratio"):
                values=[float(item[field]) for item in group]; row[field+"_mean"]=statistics.mean(values); row[field+"_sample_std"]=statistics.stdev(values)
            summary.append(row)
    with (ROOT/REPORTS/"u1000_top1500_final_mean_std_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(summary)
    print("PASS: generated Top1500 final per-run and mean/std tables")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true"); group.add_argument("--bind-dataset", action="store_true")
    group.add_argument("--validate-configs", action="store_true"); group.add_argument("--preflight", action="store_true")
    group.add_argument("--status", action="store_true"); group.add_argument("--validate-run")
    group.add_argument("--set-status"); group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--status-value", choices=("planned","running","completed","failed")); parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.bind_dataset: prepare(dataset_binding())
    elif args.validate_configs: validate_preparation()
    elif args.preflight: preflight()
    elif args.status: status()
    elif args.validate_run: print(json.dumps(validate_run(args.validate_run),indent=2))
    elif args.set_status: require(args.status_value is not None,"--status-value required"); set_status(args.set_status,args.status_value,args.reason)
    else: aggregate()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
