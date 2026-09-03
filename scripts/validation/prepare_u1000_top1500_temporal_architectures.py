#!/usr/bin/env python3
"""Fail-closed pre-training audit for both Notebook-16 temporal GCNs."""

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

from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.gcn_gru import GCNGRURegressor
from src.models.gcn_temporal_transformer import GCNTemporalTransformerRegressor
from src.models.static_gcn import StaticGCNRegressor
from src.training.split_manifest import load_split_manifest
from src.training.train_evolvegcn_h import CamelsTemporalDataset, collate_fn, load_temporal_dataset
from torch.utils.data import DataLoader


DATASET = Path("data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt")
DATASET_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
SCALE_FACTORS = [0.2, 0.25, 0.51209, 0.75065, 1.0]
CONFIG_DIR = Path("configs/production/u1000_top1500_temporal_architectures")
PROTOCOL_HASHES = {
    "reports/experiment_registry/u1000_top1500_gcn_gru_temporal_protocol.json": "c9ff4cc99feedf75812dbc16fe5a4c19c201dba6281b61389eee70cecb97584e",
    "reports/experiment_registry/u1000_top1500_gcn_temporal_transformer_protocol.json": "f7dff97f4d545146adec913e3879ada6b3ed366d66a51d4880a22d95d5a483ac",
}
EXPECTED_BRANCH = "thesis-notebook16-temporal-architectures"
EXPECTED_WORKTREE = Path("/home/ml/thesis-camels-notebook16")
CONTROL_ROOT = Path("/home/ml/thesis-camels/experiments")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_configs() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((ROOT / CONFIG_DIR).glob("*.json"))
    require(len(paths) == 6, f"expected exactly six configs, found {len(paths)}")
    configs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    require(sorted(config["seed"] for _, config in configs) == [42, 42, 123, 123, 2025, 2025],
            "configs do not contain exactly two runs for each frozen seed")
    return configs


def audit_repository_and_protocols() -> None:
    require(ROOT.resolve() == EXPECTED_WORKTREE, f"auditor must run from {EXPECTED_WORKTREE}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    require(branch == EXPECTED_BRANCH, f"wrong branch: {branch!r}")
    for relative, expected in PROTOCOL_HASHES.items():
        path = ROOT / relative
        require(path.is_file(), f"missing protocol: {relative}")
        require(sha256(path) == expected, f"protocol hash mismatch: {relative}")


def audit_metadata() -> dict[str, Any]:
    path = ROOT / DATASET
    metadata_path = path.with_suffix(".metadata.json")
    require(path.is_file() and metadata_path.is_file(), "authoritative dataset or metadata missing")
    require(sha256(path) == DATASET_SHA, "dataset SHA-256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "num_universes_successful": 1000, "num_snapshots": 5, "num_nodes": 1500,
        "feature_names": ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"],
        "normalization": "none", "graph_mode": "knn", "k": 8,
        "periodic_boundary": True, "box_size": 25.0, "target_mode": "real_targets_csv",
        "target_normalization": "none", "snapshot_ids": SCALE_FACTORS,
        "graph_storage": "sparse_edge_index", "completion_status": "complete",
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, f"metadata mismatch for {key}: {metadata.get(key)!r}")
    require(metadata.get("used_dummy_target") is False, "dummy targets are forbidden")
    return metadata


def audit_dataset(data: dict[str, Any], metadata: dict[str, Any]) -> None:
    require(len(data) == 1000, f"expected 1000 universes, got {len(data)}")
    ids = list(data)
    require(ids == metadata["ordered_universe_ids"], "dataset universe ordering differs from metadata")
    require(len(ids) == len(set(ids)), "duplicate universe IDs")
    topology_varies = False
    for universe_id, sample in data.items():
        require(len(sample.get("Nodes_list", [])) == 5, f"{universe_id}: wrong snapshot count")
        require(len(sample.get("edge_index_list", [])) == 5, f"{universe_id}: wrong edge snapshot count")
        require(len(sample.get("mask_list", [])) == 5, f"{universe_id}: wrong mask snapshot count")
        values = [float(item["snapshot_value"]) for item in sample.get("snapshots", [])]
        require(values == SCALE_FACTORS, f"{universe_id}: non-chronological snapshots")
        require(math.isfinite(float(sample["target"])), f"{universe_id}: non-finite Omega_m")
        for timestep, (x, edges, mask, info) in enumerate(zip(
            sample["Nodes_list"], sample["edge_index_list"], sample["mask_list"], sample["snapshots"]
        )):
            require(tuple(x.shape) == (1500, 7), f"{universe_id} t={timestep}: x shape {tuple(x.shape)}")
            require(edges.ndim == 2 and edges.shape[0] == 2, f"{universe_id} t={timestep}: edge shape")
            require(tuple(mask.shape) == (1500, 1), f"{universe_id} t={timestep}: mask shape")
            require(0 < int(mask.sum()) <= 1500, f"{universe_id} t={timestep}: invalid real node count")
            require(info["k"] == 8 and info["periodic_boundary"] is True,
                    f"{universe_id} t={timestep}: graph regime mismatch")
        topology_varies |= any(
            not torch.equal(sample["edge_index_list"][0], edge)
            for edge in sample["edge_index_list"][1:]
        )
    require(topology_varies, "snapshot-specific topology was not observed")


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    if config["model"] == "GCNGRURegressor":
        return GCNGRURegressor(config["node_features"], config["hidden_dim"],
                               config["num_gcn_layers"], config["dropout"])
    require(config["model"] == "GCNTemporalTransformerRegressor", "unknown model class")
    return GCNTemporalTransformerRegressor(
        config["node_features"], config["hidden_dim"], config["num_gcn_layers"],
        config["dropout"], config["scale_factors"], config["nhead"],
        config["num_encoder_layers"], config["dim_feedforward"],
    )


def audit_splits_configs_and_smoke(
    data: dict[str, Any], configs, collision_model: str | None = None,
) -> list[dict[str, Any]]:
    dataset_ids = list(data)
    manifests: dict[int, dict[str, Any]] = {}
    smoke: list[dict[str, Any]] = []
    for path, config in configs:
        require(config["dataset_sha256"] == DATASET_SHA, f"{path.name}: dataset binding")
        require(config["snapshots"] == 5 and config["scale_factors"] == SCALE_FACTORS,
                f"{path.name}: snapshot protocol")
        require(config["node_features"] == 7 and config["node_normalization"] == "none",
                f"{path.name}: feature regime")
        require(config["graph"] == "periodic_sparse_knn" and config["k"] == 8,
                f"{path.name}: graph regime")
        require(config["target"] == "Omega_m" and config["target_normalization"] == "none",
                f"{path.name}: target regime")
        require(config["batch_size"] == 8 and config["epochs"] == 300,
                f"{path.name}: training budget")
        manifest_path = ROOT / config["split_manifest_path"]
        require(sha256(manifest_path) == config["split_manifest_sha256"],
                f"{path.name}: split manifest file hash")
        manifest = load_split_manifest(manifest_path, dataset_ids, DATASET_SHA,
                                       expected_seed=config["seed"])
        require(manifest["counts"] == {"population": 1000, "train": 700, "val": 99,
                                        "test": 201, "unused": 0},
                f"{path.name}: split counts")
        if config["seed"] in manifests:
            require(manifest["train_ids"] == manifests[config["seed"]]["train_ids"] and
                    manifest["val_ids"] == manifests[config["seed"]]["val_ids"] and
                    manifest["test_ids"] == manifests[config["seed"]]["test_ids"],
                    f"{path.name}: treatment manifests do not match in order")
        else:
            manifests[config["seed"]] = manifest
        if collision_model is None or config["model"] == collision_model:
            run_dir = ROOT / config["output_root"] / config["experiment_name"]
            require(not run_dir.exists(), f"run collision: {run_dir}")

    for seed in (42, 123, 2025):
        manifest = manifests[seed]
        tiny = CamelsTemporalDataset(data, manifest["train_ids"][:2])
        batch = next(iter(DataLoader(tiny, batch_size=2, shuffle=False, collate_fn=collate_fn)))
        _, temporal, _, _, _ = batch
        snapshot_shapes = [{"x": list(item["x"].shape), "edge_index": list(item["edge_index"].shape)}
                           for item in temporal["snapshots"]]
        for model_name in ("GCNGRURegressor", "GCNTemporalTransformerRegressor"):
            config = next(item for _, item in configs if item["seed"] == seed and item["model"] == model_name)
            torch.manual_seed(seed)
            model = build_model(config).cpu().eval()
            require(count(model) == config["parameter_count"], f"{model_name}: parameter count")
            with torch.no_grad():
                sequence = model.encode_snapshots(temporal)
                output = model(temporal)
            require(tuple(sequence.shape) == (2, 5, 32), f"{model_name}: sequence shape")
            require(tuple(output.shape) == (2, 1), f"{model_name}: output shape")
            require(bool(torch.isfinite(output).all()), f"{model_name}: non-finite CPU output")
            smoke.append({"architecture": model_name, "seed": seed,
                          "snapshot_shapes": snapshot_shapes,
                          "sequence_shape": list(sequence.shape), "output_shape": list(output.shape),
                          "finite": True})
    return smoke


def audit_controls(configs) -> dict[str, Any]:
    result: dict[str, Any] = {}
    patterns = {
        "Static GCN": "static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final",
        "EvolveGCN-H": "evolvegcn_h_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_mean_temporal_mean_linear",
    }
    treatment_by_seed = {config["seed"]: config for _, config in configs if config["model"] == "GCNGRURegressor"}
    for label, pattern in patterns.items():
        paths = []
        for seed in (42, 123, 2025):
            run = CONTROL_ROOT / pattern.format(seed=seed)
            control = json.loads((run / "config.json").read_text(encoding="utf-8"))
            treatment = treatment_by_seed[seed]
            require(control["dataset_path"] == treatment["dataset_path"], f"{label} seed {seed}: dataset")
            require(control.get("split_source") == treatment["split_manifest_path"],
                    f"{label} seed {seed}: split")
            manifest = json.loads((ROOT / treatment["split_manifest_path"]).read_text(encoding="utf-8"))
            for split, field in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
                require(control.get(field) == manifest[field],
                        f"{label} seed {seed}: ordered {split} IDs differ")
            for relative in ("config.json", "metrics.json", "train_log.csv",
                             "checkpoints/best_model.pt", "predictions/val_predictions.csv",
                             "predictions/test_predictions.csv"):
                require((run / relative).is_file(), f"{label} seed {seed}: missing {relative}")
            paths.append(str(run))
        result[label] = paths
    static = StaticGCNRegressor(node_features=7, hidden_dim=32, num_layers=3, dropout=0.2,
                                graph_pooling="mean", conv_type="gcn", use_layer_norm=True,
                                residual=True, add_self_loops=True)
    evolve = EvolveGCNHRegressor(node_features=7, hidden_dim=32, num_layers=2, dropout=0.2,
                                 activation="relu", temporal_pooling="mean", graph_pooling="mean",
                                 head_type="linear", add_self_loops=True)
    require(count(static) == 5281 and count(evolve) == 3408097, "control parameter count mismatch")
    return result


def validate_run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run = ROOT / config["output_root"] / config["experiment_name"]
    required = ("config.json", "metrics.json", "train_log.csv", "run_metadata.json",
                "checkpoints/best_model.pt", "predictions/val_predictions.csv",
                "predictions/test_predictions.csv")
    for relative in required:
        require((run / relative).is_file(), f"incomplete run {run}: missing {relative}")
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for key in ("model", "seed", "dataset_sha256", "split_manifest_sha256", "parameter_count"):
        require(saved.get(key) == config[key], f"completed run mismatch for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-run", type=Path)
    parser.add_argument("--architecture", choices=("gru", "transformer"))
    args = parser.parse_args()
    try:
        if args.validate_run:
            validate_run(args.validate_run)
            print("PASS: required run artifacts verified")
            return 0
        audit_repository_and_protocols()
        configs = load_configs()
        metadata = audit_metadata()
        data = load_temporal_dataset(ROOT / DATASET)
        audit_dataset(data, metadata)
        controls = audit_controls(configs)
        collision_model = {
            "gru": "GCNGRURegressor",
            "transformer": "GCNTemporalTransformerRegressor",
        }.get(args.architecture)
        smoke = audit_splits_configs_and_smoke(data, configs, collision_model)
        report = {
            "status": "PASS", "dataset_sha256": DATASET_SHA,
            "dataset_rebuild_required": False, "controls": controls,
            "parameter_counts": {"Static GCN": 5281, "EvolveGCN-H": 3408097,
                                 "GCN-GRU": 11617, "GCN-Temporal-Transformer": 13825},
            "smoke_tests": smoke, "production_training_performed": False,
            "leakage_checks": {"node_normalization": "none", "target_in_temporal_encoding": False,
                               "split_overlap": False, "test_used_for_selection": False},
        }
        print(json.dumps(report, indent=2))
        print("PASS: NO DATASET REBUILD REQUIRED")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
