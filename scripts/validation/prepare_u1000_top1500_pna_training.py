#!/usr/bin/env python3
"""Prepare and fail-closed audit the controlled U1000 Top1500 Static PNA runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.static_gcn import StaticGCNRegressor, count_parameters
from src.models.static_pna import (
    CANONICAL_AGGREGATORS,
    CANONICAL_SCALERS,
    StaticPNARegressor,
    effective_pna_edge_index,
)
from src.training.sparse_batch import collate_sparse_static
from src.training.split_manifest import ordered_id_hash
from src.training.train_static_gcn import load_dataset
from src.training.train_static_pna import histogram_content_sha256, load_degree_histogram


SEEDS = (42, 123, 2025)
DATASET_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
DATASET = Path("data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt")
HIST_DIR = Path("reports/experiment_registry/pna_degree_histograms")
CONFIG_DIR = Path("configs/production/u1000_top1500_pna_ablation")
PROTOCOL = Path("reports/experiment_registry/u1000_top1500_static_pna_protocol.json")
TOOL = Path("scripts/validation/prepare_u1000_top1500_pna_training.py")
EXPECTED_SPLIT_SHAS = {
    42: "f5556ec5c193e7cb80f2231705edbdae32d4de206889dec308a819bdde427ab7",
    123: "18a295106ec844848053f3040d2be3cdf73a443010be31e6e2cd962362982471",
    2025: "c233c0631b1a24d963ffccc7c6389054fddf135a7aabc4dc4ab7bf5976fab3a9",
}
EXPECTED_COUNTS = {"train": 700, "val": 99, "test": 201, "unused": 0}
FEATURES = ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]
PNA_FIELDS = {
    "aggregators", "conv_type", "degree_histogram_content_sha256",
    "degree_histogram_path", "divide_input", "edge_dim", "experiment_name",
    "model", "post_layers", "pre_layers", "scalers", "schema_version", "self_loop_policy",
    "towers", "train_norm",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def split_path(seed: int) -> Path:
    return Path(f"configs/splits/u1000_top1500_none_k8_sparse/seed{seed}_train700.json")


def control_config_path(seed: int) -> Path:
    name = f"static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final.json"
    return Path("configs/production/u1000_top1500_training_scaling") / name


def experiment_name(seed: int) -> str:
    return f"static_pna_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final"


def histogram_path(seed: int) -> Path:
    return HIST_DIR / f"u1000_top1500_k8_seed{seed}_train700.json"


def config_path(seed: int) -> Path:
    return CONFIG_DIR / f"{experiment_name(seed)}.json"


def validate_split(seed: int) -> dict[str, Any]:
    path = ROOT / split_path(seed)
    require(sha256_file(path) == EXPECTED_SPLIT_SHAS[seed], f"seed {seed}: split file SHA mismatch")
    value = read_json(path)
    require(value.get("seed") == seed and value.get("dataset_identity") == DATASET_SHA,
            f"seed {seed}: split binding mismatch")
    sets = []
    for partition in ("train", "val", "test"):
        ids = value.get(f"{partition}_ids")
        require(isinstance(ids, list) and len(ids) == EXPECTED_COUNTS[partition],
                f"seed {seed}: wrong {partition} IDs")
        require(len(ids) == len(set(ids)), f"seed {seed}: duplicate {partition} IDs")
        require(value.get("split_hashes", {}).get(partition) == ordered_id_hash(ids),
                f"seed {seed}: ordered {partition} hash mismatch")
        sets.append(set(ids))
    require(not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2],
            f"seed {seed}: split leakage/overlap")
    require(all(value.get("counts", {}).get(key) == count for key, count in EXPECTED_COUNTS.items())
            and value.get("counts", {}).get("population") == 1000 and value.get("unused_ids") == [],
            f"seed {seed}: split counts/unused mismatch")
    return value


def compute_histogram(data: Mapping[str, Any], train_ids: list[str]) -> tuple[list[int], int]:
    histogram = torch.zeros(1, dtype=torch.long)
    total_edges = 0
    for universe_id in train_ids:
        sample = data[universe_id]
        x = sample["X"]
        raw_edge = sample["edge_index"]
        require(not bool((raw_edge[0] == raw_edge[1]).any()),
                f"{universe_id}: frozen dataset unexpectedly contains self-loops")
        edge = effective_pna_edge_index(raw_edge, int(x.shape[0]))
        require(int((edge[0] == edge[1]).sum()) == int(x.shape[0]),
                f"{universe_id}: effective PNA edge_index does not contain exactly one loop per node")
        degrees = torch.bincount(edge[1], minlength=int(x.shape[0]))
        current = torch.bincount(degrees, minlength=int(degrees.max()) + 1)
        if current.numel() > histogram.numel():
            histogram = torch.nn.functional.pad(histogram, (0, current.numel() - histogram.numel()))
        histogram[:current.numel()] += current
        total_edges += int(edge.shape[1])
    return histogram.tolist(), total_edges


def histogram_record(seed: int, split: Mapping[str, Any], histogram: list[int],
                     total_edges: int) -> dict[str, Any]:
    train_ids = list(split["train_ids"])
    return {
        "schema_version": "camels_static_pna_degree_histogram_v1",
        "seed": seed,
        "dataset_path": DATASET.as_posix(),
        "dataset_sha256": DATASET_SHA,
        "split_manifest_path": split_path(seed).as_posix(),
        "split_manifest_sha256": EXPECTED_SPLIT_SHAS[seed],
        "training_id_order_sha256": ordered_id_hash(train_ids),
        "number_of_training_universes": len(train_ids),
        "snapshot_policy": "final_snapshot_index_minus_one_only",
        "graph": {"family": "periodic_sparse_knn", "k": 8, "box_size_h_inverse_mpc": 25.0},
        "self_loop_policy": "torch_geometric.add_remaining_self_loops_once_before_all_layers",
        "degree_direction": "incoming_target_index_edge_index_1",
        "histogram": histogram,
        "histogram_content_sha256": histogram_content_sha256(histogram),
        "max_degree": len(histogram) - 1,
        "total_node_count": sum(histogram),
        "total_degree_count": total_edges,
        "partition_hygiene": {
            "source": "train_ids_only",
            "validation_ids_contributed": 0,
            "test_ids_contributed": 0,
            "targets_read": False,
            "predictions_or_metrics_read": False,
        },
        "creation_provenance": {
            "tool": TOOL.as_posix(),
            "tool_sha256": sha256_file(ROOT / TOOL),
            "torch_version": torch.__version__,
            "torch_geometric_version": __import__("torch_geometric").__version__,
        },
    }


def make_config(seed: int, record: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(read_json(ROOT / control_config_path(seed)))
    value.update({
        "schema_version": "camels_u1000_static_pna_v1",
        "experiment_name": experiment_name(seed),
        "model": "StaticPNARegressor",
        "conv_type": "pna",
        "aggregators": list(CANONICAL_AGGREGATORS),
        "scalers": list(CANONICAL_SCALERS),
        "towers": 1,
        "pre_layers": 1,
        "post_layers": 1,
        "divide_input": False,
        "train_norm": False,
        "edge_dim": None,
        "self_loop_policy": record["self_loop_policy"],
        "degree_histogram_path": histogram_path(seed).as_posix(),
        "degree_histogram_content_sha256": record["histogram_content_sha256"],
    })
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def config_differences(control: Mapping[str, Any], pna: Mapping[str, Any]) -> set[str]:
    left, right = flatten(control), flatten(pna)
    missing = object()
    return {key for key in left.keys() | right.keys()
            if left.get(key, missing) != right.get(key, missing)}


def protocol_document(records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    static_count = count_parameters(StaticGCNRegressor(7, 32, 3, 0.2, "mean", "gcn"))
    pna_count = count_parameters(StaticPNARegressor(deg=records[42]["histogram"]))
    return {
        "schema_version": "camels_u1000_top1500_static_pna_protocol_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "research_question": "Holding data, graph, features, splits, training, pooling, and head fixed, does Static PNA improve Omega_m prediction over Static GCN?",
        "control": "StaticGCNRegressor",
        "treatment": "StaticPNARegressor",
        "dataset": {"path": DATASET.as_posix(), "sha256": DATASET_SHA, "rebuilt": False},
        "data": {"suite": "CAMELS-SIMBA", "universes": 1000, "top_n": 1500,
                 "snapshot": "final", "normalization": "none", "features": FEATURES,
                 "target": "Omega_m"},
        "graph": {"family": "periodic_sparse_knn", "k": 8,
                  "box_size_h_inverse_mpc": 25.0, "edge_features": False},
        "seeds": list(SEEDS),
        "splits": {str(seed): {"path": split_path(seed).as_posix(),
                                "sha256": EXPECTED_SPLIT_SHAS[seed]} for seed in SEEDS},
        "degree_histograms": {str(seed): {"path": histogram_path(seed).as_posix(),
                  "content_sha256": records[seed]["histogram_content_sha256"]} for seed in SEEDS},
        "architecture": {"input_dim": 7, "hidden_dim": 32, "num_layers": 3,
            "activation": "relu", "dropout": 0.2, "layer_norm": True, "residual": True,
            "graph_pooling": "mean", "head": [32, 32, 16, 1],
            "aggregators": list(CANONICAL_AGGREGATORS), "scalers": list(CANONICAL_SCALERS),
            "towers": 1, "pre_layers": 1, "post_layers": 1, "divide_input": False,
            "train_norm": False, "edge_dim": None,
            "self_loop_policy": records[42]["self_loop_policy"]},
        "parameter_counts": {"static_gcn": static_count, "static_pna": pna_count,
                             "pna_to_gcn_ratio": pna_count / static_count},
        "training": {"loss": "MSE", "optimizer": "AdamW", "learning_rate": 0.001,
            "weight_decay": 1e-5, "batch_size": 8, "max_epochs": 300,
            "early_stopping": {"criterion": "validation MSE", "patience": 40},
            "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5,
                          "patience": 10, "min_lr": 1e-6},
            "gradient_clip_norm": 1.0, "deterministic_seeding": True, "device": "cuda"},
        "selection": {"primary": "validation MAE",
            "supporting": ["validation RMSE", "validation R2", "prediction dispersion", "calibration slope"],
            "test_metrics_must_not_be_used_for_architecture_selection": True},
        "no_pna_hyperparameter_tuning_performed": True,
        "allowed_control_treatment_differences": sorted(PNA_FIELDS),
        "source_symmetry_closure_commit": "dc22f0af21c15611c2c010f83534a972e0fe6629",
    }


def prepare() -> None:
    require(git("rev-parse", "HEAD") == "dc22f0af21c15611c2c010f83534a972e0fe6629",
            "preparation must start at the symmetry closure commit")
    require(not git("diff", "--name-only") or set(git("diff", "--name-only").splitlines()) <= {
        "src/models/static_pna.py", "src/training/train_static_gcn.py",
        "src/training/train_static_pna.py", "tests/test_static_pna.py", TOOL.as_posix()},
        "unexpected tracked changes before PNA preparation")
    require(sha256_file(ROOT / DATASET) == DATASET_SHA, "dataset SHA mismatch")
    splits = {seed: validate_split(seed) for seed in SEEDS}
    data = load_dataset(ROOT / DATASET, "temporal_final_snapshot")
    records = {}
    for seed in SEEDS:
        histogram, total_edges = compute_histogram(data, list(splits[seed]["train_ids"]))
        record = histogram_record(seed, splits[seed], histogram, total_edges)
        require(record["total_node_count"] == 700 * 1500, f"seed {seed}: node total mismatch")
        write_json(ROOT / histogram_path(seed), record)
        records[seed] = record
        write_json(ROOT / config_path(seed), make_config(seed, record))
    write_json(ROOT / PROTOCOL, protocol_document(records))
    print("PNA PREPARATION COMPLETE")


def audit(expected_head: str | None) -> None:
    require(ROOT.resolve() == Path("/home/ml/thesis-camels-notebook15"), "wrong worktree")
    require(git("branch", "--show-current") == "thesis-notebook15-graph-representation", "wrong branch")
    if expected_head:
        require(git("rev-parse", "HEAD") == expected_head, "wrong PNA preparation HEAD")
    require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"),
            "tracked or staged changes present")
    require(sha256_file(ROOT / DATASET) == DATASET_SHA, "dataset SHA mismatch")
    data = load_dataset(ROOT / DATASET, "temporal_final_snapshot")
    records = {}
    for seed in SEEDS:
        split = validate_split(seed)
        deg, record = load_degree_histogram(ROOT / histogram_path(seed))
        require(record.get("split_manifest_sha256") == EXPECTED_SPLIT_SHAS[seed],
                f"seed {seed}: histogram split identity mismatch")
        require(record.get("training_id_order_sha256") == ordered_id_hash(split["train_ids"]),
                f"seed {seed}: histogram training ID identity mismatch")
        require(record.get("partition_hygiene", {}).get("validation_ids_contributed") == 0 and
                record.get("partition_hygiene", {}).get("test_ids_contributed") == 0,
                f"seed {seed}: degree leakage declaration mismatch")
        histogram, total_edges = compute_histogram(data, list(split["train_ids"]))
        require(histogram == record["histogram"] and total_edges == record["total_degree_count"],
                f"seed {seed}: independently recomputed degree histogram mismatch")
        require(sum(histogram) == record["total_node_count"] == 700 * 1500,
                f"seed {seed}: histogram node sum mismatch")
        control, config = read_json(ROOT / control_config_path(seed)), read_json(ROOT / config_path(seed))
        require(config == make_config(seed, record), f"seed {seed}: production config mismatch")
        diffs = config_differences(control, config)
        require(diffs == PNA_FIELDS, f"seed {seed}: unexpected config differences: {sorted(diffs ^ PNA_FIELDS)}")
        require(not (ROOT / "experiments" / experiment_name(seed)).exists(), f"seed {seed}: run collision")
        sample_ids = split["train_ids"][:2]
        samples = [{"graph_storage": "sparse_edge_index", "x": data[uid]["X"],
                    "edge_index": data[uid]["edge_index"], "edge_weight": data[uid].get("edge_weight"),
                    "mask": data[uid]["mask"]} for uid in sample_ids]
        batch = collate_sparse_static(samples)
        original = batch["edge_index"].clone()
        model = StaticPNARegressor(7, 32, 3, 0.2, "mean", deg).eval()
        with torch.inference_mode():
            output = model(batch)
        require(torch.equal(original, batch["edge_index"]), f"seed {seed}: stored edge_index mutated")
        require(tuple(output.shape) == (2, 1) and bool(torch.isfinite(output).all()),
                f"seed {seed}: non-finite/invalid CPU forward")
        records[seed] = record
        print(f"PASS PNA seed={seed} x={tuple(batch['x'].shape)} edge_index={tuple(batch['edge_index'].shape)} output={tuple(output.shape)} finite=True collision=CLEAR leakage=ZERO")
    require(read_json(ROOT / PROTOCOL) == protocol_document(records), "PNA protocol mismatch")
    print("PNA PRE-TRAINING AUDIT PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    prepare() if args.prepare else audit(args.expected_head)


if __name__ == "__main__":
    main()
