#!/usr/bin/env python3
"""Fail-closed EvolveGCN-O pre-training and completed-run auditor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evolvegcn_o import EvolveGCNORegressor, count_parameters
from src.training.split_manifest import load_split_manifest
from src.training.train_evolvegcn_h import CamelsTemporalDataset, collate_fn, load_temporal_dataset
from src.training.train_evolvegcn_o import FROZEN_O_VALUES, build_model
from src.training.temporal_architecture_common import load_and_validate_config
from src.training.temporal_architecture_common import validate_finalizable_run


DATASET = Path("data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt")
DATASET_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
SCALE_FACTORS = [0.2, 0.25, 0.51209, 0.75065, 1.0]
CONFIG_DIR = Path("configs/production/u1000_top1500_evolvegcn_o")
PROTOCOL = Path("reports/experiment_registry/u1000_top1500_evolvegcn_o_temporal_protocol.json")
PROTOCOL_SHA = "45a21cc4500199a87116f72c707967a721cf9ead832cb04cde6f2dedf112cb2e"
EXPECTED_WORKTREE = Path("/home/ml/thesis-camels")
EXPECTED_BRANCH = "thesis-sparse-integrity-hardening"
CONTROL_ROOT = Path("/home/ml/thesis-camels/experiments")
EXPECTED_PARAMETERS = 11_527


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_configs() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((ROOT / CONFIG_DIR).glob("*.json"))
    require(len(paths) == 3, f"expected exactly three O configs, found {len(paths)}")
    result = []
    for path in paths:
        config = load_and_validate_config(
            path, "EvolveGCNORegressor", frozen_values=FROZEN_O_VALUES
        )
        result.append((path, config))
    require(sorted(item["seed"] for _, item in result) == [42, 123, 2025], "wrong seeds")
    return result


def audit_repository() -> None:
    require(ROOT.resolve() == EXPECTED_WORKTREE, f"auditor must run from {EXPECTED_WORKTREE}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    require(branch == EXPECTED_BRANCH, f"wrong branch: {branch!r}")
    require((ROOT / PROTOCOL).is_file(), "protocol record missing")
    require(sha256(ROOT / PROTOCOL) == PROTOCOL_SHA, "protocol SHA-256 mismatch")


def audit_metadata() -> dict[str, Any]:
    dataset_path = ROOT / DATASET
    metadata_path = dataset_path.with_suffix(".metadata.json")
    require(dataset_path.is_file() and metadata_path.is_file(), "dataset or metadata missing")
    require(sha256(dataset_path) == DATASET_SHA, "dataset SHA-256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "num_universes_successful": 1000, "num_snapshots": 5, "num_nodes": 1500,
        "snapshot_ids": SCALE_FACTORS,
        "feature_names": ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"],
        "normalization": "none", "graph_mode": "knn", "k": 8,
        "periodic_boundary": True, "box_size": 25.0, "target_mode": "real_targets_csv",
        "target_normalization": "none", "graph_storage": "sparse_edge_index",
        "completion_status": "complete",
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, f"metadata mismatch for {key}: {metadata.get(key)!r}")
    require(metadata.get("used_dummy_target") is False, "dummy targets are forbidden")
    return metadata


def audit_h_match(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    seed = config["seed"]
    h_run = CONTROL_ROOT / (
        f"evolvegcn_h_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_"
        "mean_temporal_mean_linear"
    )
    h = json.loads((h_run / "config.json").read_text(encoding="utf-8"))
    matched = (
        "dataset_path", "seed", "batch_size", "epochs", "patience", "learning_rate",
        "weight_decay", "hidden_dim", "num_layers", "dropout", "activation",
        "temporal_pooling", "graph_pooling", "head_type", "add_self_loops",
        "grad_clip_norm", "checkpoint_criterion",
    )
    for key in matched:
        require(config[key] == h[key], f"seed {seed}: unexpected H/O difference for {key}")
    require(config["split_manifest_path"] == h["split_source"], f"seed {seed}: H split path")
    require(config["split_manifest_sha256"] == h["split_manifest_sha256"], f"seed {seed}: H split SHA")
    for split in ("train", "val", "test"):
        require(manifest[f"{split}_ids"] == h[f"{split}_ids"], f"seed {seed}: ordered H/O {split}")
    require(config["optimizer"] == h["optimizer"] == "AdamW", "H/O optimizer")
    require(config["loss"] == h["loss"] == "MSELoss", "H/O loss")
    require(config["scheduler"] == h["scheduler"], "H/O scheduler")
    require(h["scheduler"] == {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                                "patience": 10, "min_lr": 1e-6}, "H scheduler")
    require(h["trainable_parameters"] == 3_408_097, "H parameter count")


def audit_configs_splits_and_smoke(
    data: dict[str, Any], metadata: dict[str, Any], configs: list[tuple[Path, dict[str, Any]]],
    *, enforce_clear_run_directories: bool = True,
) -> list[dict[str, Any]]:
    dataset_ids = list(data)
    require(dataset_ids == metadata["ordered_universe_ids"], "dataset order differs from metadata")
    require(len(dataset_ids) == len(set(dataset_ids)) == 1000, "dataset IDs invalid")
    results = []
    for path, config in configs:
        require(config["dataset_sha256"] == DATASET_SHA, f"{path.name}: dataset binding")
        require(config["snapshots"] == 5 and config["scale_factors"] == SCALE_FACTORS, f"{path.name}: snapshots")
        require(config["node_features"] == 7 and config["node_feature_names"] == metadata["feature_names"], f"{path.name}: features")
        require(config["graph"] == "periodic_sparse_knn" and config["k"] == 8 and config["box_size"] == 25.0, f"{path.name}: graph")
        require(config["target"] == "Omega_m" and config["target_normalization"] == "none", f"{path.name}: target")
        require(config["weight_evolution"] == "canonical_matrix_gru_previous_weight_only", f"{path.name}: formulation")
        manifest_path = ROOT / config["split_manifest_path"]
        require(sha256(manifest_path) == config["split_manifest_sha256"], f"{path.name}: split SHA")
        manifest = load_split_manifest(manifest_path, dataset_ids, DATASET_SHA, expected_seed=config["seed"])
        require(manifest["counts"] == {"population": 1000, "train": 700, "val": 99, "test": 201, "unused": 0}, f"{path.name}: counts")
        sets = [set(manifest[f"{split}_ids"]) for split in ("train", "val", "test")]
        require(all(len(values) == expected for values, expected in zip(sets, (700, 99, 201))), f"{path.name}: duplicates")
        require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), f"{path.name}: overlap")
        require(set.union(*sets) == set(dataset_ids), f"{path.name}: incomplete split coverage")
        audit_h_match(config, manifest)
        run_dir = ROOT / config["output_root"] / config["experiment_name"]
        if enforce_clear_run_directories:
            require(not run_dir.exists(), f"run-directory collision: {run_dir}")

        tiny = CamelsTemporalDataset(data, manifest["train_ids"][:2])
        batch = next(iter(DataLoader(tiny, batch_size=2, shuffle=False, collate_fn=collate_fn)))
        _, temporal, _, _, _ = batch
        require(temporal["num_timesteps"] == 5, f"{path.name}: smoke snapshot count")
        values = [float(data[manifest["train_ids"][0]]["snapshots"][i]["snapshot_value"]) for i in range(5)]
        require(values == SCALE_FACTORS, f"{path.name}: real-data chronology")
        torch.manual_seed(config["seed"])
        model = build_model(config).cpu().eval()
        require(isinstance(model, EvolveGCNORegressor), f"{path.name}: model resolution")
        require(count_parameters(model) == config["parameter_count"] == EXPECTED_PARAMETERS, f"{path.name}: parameters")
        with torch.no_grad():
            output = model(temporal)
        require(tuple(output.shape) == (2, 1) and bool(torch.isfinite(output).all()), f"{path.name}: CPU forward")
        results.append({
            "seed": config["seed"],
            "snapshot_shapes": [{"x": list(item["x"].shape), "edge_index": list(item["edge_index"].shape)} for item in temporal["snapshots"]],
            "scale_factors": values, "output_shape": list(output.shape), "finite": True,
            "split_counts": {"train": 700, "validation": 99, "test": 201},
            "duplicates": False, "overlap": False, "h_o_ordered_match": True,
        })
    return results


def validate_run(config_path: Path) -> None:
    config = load_and_validate_config(config_path, "EvolveGCNORegressor", FROZEN_O_VALUES)
    validate_finalizable_run(
        config_path, "EvolveGCNORegressor", build_model, frozen_values=FROZEN_O_VALUES
    )
    run = ROOT / config["output_root"] / config["experiment_name"]
    required = ("config.json", "metrics.json", "train_log.csv", "run_metadata.json",
                "checkpoints/best_model.pt", "predictions/val_predictions.csv",
                "predictions/test_predictions.csv")
    for relative in required:
        path = run / relative
        require(path.is_file() and path.stat().st_size > 0, f"incomplete run: {relative}")
    manifest = json.loads((ROOT / config["split_manifest_path"]).read_text(encoding="utf-8"))
    for split, expected, key in (("val", 99, "val_ids"), ("test", 201, "test_ids")):
        with (run / f"predictions/{split}_predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == expected, f"{split}: wrong prediction row count")
        require([row["universe_id"] for row in rows] == manifest[key], f"{split}: IDs differ from manifest")
        require(len({row["universe_id"] for row in rows}) == expected, f"{split}: duplicate IDs")
        require(all(math.isfinite(float(row["target"])) and math.isfinite(float(row["prediction"])) for row in rows), f"{split}: non-finite values")
    print("PASS: EvolveGCN-O completed-run artifacts verified")


def recovery_preflight() -> None:
    audit_repository()
    metadata = audit_metadata()
    configs = load_configs()
    data = load_temporal_dataset(ROOT / DATASET)
    smoke = audit_configs_splits_and_smoke(
        data, metadata, configs, enforce_clear_run_directories=False
    )
    by_seed = {config["seed"]: (path, config) for path, config in configs}
    validate_run(by_seed[42][0])

    seed123_path, seed123 = by_seed[123]
    evidence = validate_finalizable_run(
        seed123_path, "EvolveGCNORegressor", build_model, frozen_values=FROZEN_O_VALUES
    )
    seed123_run = ROOT / seed123["output_root"] / seed123["experiment_name"]
    require(
        not (seed123_run / "metrics.json").exists()
        and not (seed123_run / "predictions/val_predictions.csv").exists()
        and not (seed123_run / "predictions/test_predictions.csv").exists(),
        "seed123 is not in the forensically approved finalization-missing state",
    )
    seed2025 = by_seed[2025][1]
    seed2025_run = ROOT / seed2025["output_root"] / seed2025["experiment_name"]
    require(not seed2025_run.exists(), f"seed2025 canonical collision: {seed2025_run}")
    print(json.dumps({
        "status": "READY_FOR_RECOVERY",
        "seed42": "complete_and_validated",
        "seed123": {
            "status": "TRAINING_COMPLETE_FINALIZATION_MISSING",
            "train_log_rows": evidence["train_log_rows"],
            "last_epoch": evidence["last_epoch"],
            "best_epoch": evidence["best_epoch"],
            "termination": evidence["termination"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
        },
        "seed2025": "not_started",
        "real_data_cpu_smoke": smoke,
        "production_training_performed": False,
        "test_metrics_used_for_selection": False,
    }, indent=2))
    print("READY — HUMAN MAY RECOVER EVOLVEGCN-O SEEDS 123 AND 2025")


def validate_all() -> None:
    for path, _ in load_configs():
        validate_run(path)
    print("PASS: all three EvolveGCN-O production runs are complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-run", type=Path)
    mode.add_argument("--validate-all", action="store_true")
    mode.add_argument("--recovery-preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_run:
            validate_run(args.validate_run)
            return 0
        if args.validate_all:
            validate_all()
            return 0
        if args.recovery_preflight:
            recovery_preflight()
            return 0
        audit_repository()
        metadata = audit_metadata()
        configs = load_configs()
        data = load_temporal_dataset(ROOT / DATASET)
        smoke = audit_configs_splits_and_smoke(data, metadata, configs)
        print(json.dumps({
            "status": "READY", "dataset_sha256": DATASET_SHA,
            "parameter_count": EXPECTED_PARAMETERS, "real_data_cpu_smoke": smoke,
            "production_training_performed": False, "test_metrics_used_for_selection": False,
        }, indent=2))
        print("READY — HUMAN MAY START THREE EVOLVEGCN-O TRAININGS")
        return 0
    except Exception as exc:
        print(f"BLOCKED — FIX EVOLVEGCN-O PREPARATION FIRST: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
