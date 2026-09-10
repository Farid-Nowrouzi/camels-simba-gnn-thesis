"""Thin training entrypoint for the controlled final-snapshot Static PNA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.models.static_pna import (
    CANONICAL_AGGREGATORS,
    CANONICAL_SCALERS,
    StaticPNARegressor,
    validate_degree_histogram,
)
from src.training.train_static_gcn import train_static_gcn


def histogram_content_sha256(histogram: list[int]) -> str:
    payload = json.dumps(histogram, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_degree_histogram(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PNA degree histogram not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("histogram"), list):
        raise ValueError(f"Invalid PNA degree histogram record: {path}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value["histogram"]):
        raise ValueError(f"PNA degree histogram bins must be integers: {path}")
    histogram = list(value["histogram"])
    actual = histogram_content_sha256(histogram)
    if value.get("histogram_content_sha256") != actual:
        raise ValueError(f"PNA degree histogram content SHA mismatch: {path}")
    if sum(histogram) != value.get("total_node_count"):
        raise ValueError(f"PNA degree histogram sum mismatch: {path}")
    return validate_degree_histogram(histogram), value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled Static PNA on frozen sparse graphs.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_format", choices=["temporal_final_snapshot"],
                        default="temporal_final_snapshot")
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--dataset_identity", required=True)
    parser.add_argument("--degree_histogram_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph_pooling", choices=["mean"], default="mean")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.099)
    parser.add_argument("--test_ratio", type=float, default=0.201)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output_root) / args.experiment_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing PNA run: {output}")
    degree_histogram, degree_record = load_degree_histogram(args.degree_histogram_path)
    if degree_record.get("seed") != args.seed:
        raise ValueError("Degree histogram seed does not match training seed.")
    if degree_record.get("dataset_sha256") != args.dataset_identity:
        raise ValueError("Degree histogram dataset identity does not match training dataset.")

    def build_model(node_features: int) -> StaticPNARegressor:
        return StaticPNARegressor(
            node_features=node_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            graph_pooling=args.graph_pooling,
            deg=degree_histogram,
            add_self_loops=True,
            use_layer_norm=True,
            residual=True,
        )

    train_static_gcn(
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        split_manifest_path=args.split_manifest_path,
        dataset_identity=args.dataset_identity,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
        conv_type="pna",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        grad_clip_norm=args.grad_clip_norm,
        device_name=args.device,
        model_factory=build_model,
        model_name="StaticPNARegressor",
        training_label="Static PNA",
        config_overrides={
            "activation": "relu",
            "aggregators": list(CANONICAL_AGGREGATORS),
            "scalers": list(CANONICAL_SCALERS),
            "towers": 1,
            "pre_layers": 1,
            "post_layers": 1,
            "divide_input": False,
            "train_norm": False,
            "edge_dim": None,
            "add_self_loops": True,
            "self_loop_policy": "torch_geometric.add_remaining_self_loops_once_before_all_layers",
            "use_layer_norm": True,
            "residual": True,
            "degree_histogram_path": str(args.degree_histogram_path),
            "degree_histogram_content_sha256": degree_record["histogram_content_sha256"],
        },
    )


if __name__ == "__main__":
    main()
