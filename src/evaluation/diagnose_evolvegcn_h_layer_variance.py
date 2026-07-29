from __future__ import annotations

"""
diagnose_evolvegcn_h_layer_variance.py

Pure diagnostic for measuring variance flow through EvolveGCN-H layers before
graph pooling.

This script does not train, change preprocessing, alter dataset construction,
modify model behavior, edit checkpoints, change splits, or alter architecture.
It loads a trained experiment config/checkpoint, reuses saved split IDs, and
reports variance retention from the raw layer input through every EvolveGCN
layer output.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.evaluation.diagnose_evolvegcn_h_representations import (
    build_model_from_config,
    choose_device,
    load_checkpoint_state,
    load_json,
    save_json,
    validate_split_ids,
)
from src.training.train_evolvegcn_h import (
    CamelsTemporalDataset,
    collate_fn,
    load_temporal_dataset,
    unpack_batch,
)


def sample_matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    return tensor.reshape(tensor.shape[0], -1)


def stage_metrics(sample_vectors: torch.Tensor) -> Dict[str, Any]:
    sample_vectors = sample_vectors.detach().cpu().float()

    if sample_vectors.ndim != 2:
        raise ValueError(
            f"Expected sample vectors [B, D], got {tuple(sample_vectors.shape)}"
        )

    num_samples, width = sample_vectors.shape

    if num_samples < 2 or width == 0:
        return {
            "shape": list(sample_vectors.shape),
            "num_samples": int(num_samples),
            "width": int(width),
            "avg_feature_variance": float("nan"),
            "avg_pairwise_squared_distance_per_feature": float("nan"),
        }

    feature_variance = sample_vectors.var(dim=0, unbiased=False).mean()
    pairwise_distances = torch.pdist(sample_vectors, p=2).pow(2) / float(width)

    return {
        "shape": list(sample_vectors.shape),
        "num_samples": int(num_samples),
        "width": int(width),
        "avg_feature_variance": float(feature_variance.item()),
        "avg_pairwise_squared_distance_per_feature": float(
            pairwise_distances.mean().item()
        ),
    }


def safe_ratio(current: float, previous: float) -> float:
    if previous == 0.0 or previous != previous:
        return float("nan")

    return current / previous


def add_retention_ratios(stage_reports: List[Dict[str, Any]]) -> None:
    for index, report in enumerate(stage_reports):
        if index == 0:
            report["variance_retention_from_previous"] = 1.0
            report["pairwise_retention_from_previous"] = 1.0
            continue

        previous = stage_reports[index - 1]

        report["variance_retention_from_previous"] = safe_ratio(
            report["avg_feature_variance"],
            previous["avg_feature_variance"],
        )
        report["pairwise_retention_from_previous"] = safe_ratio(
            report["avg_pairwise_squared_distance_per_feature"],
            previous["avg_pairwise_squared_distance_per_feature"],
        )


@torch.no_grad()
def extract_layer_outputs(
    model,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
) -> List[torch.Tensor]:
    outputs = [X_seq]
    H_seq = X_seq

    for layer in model.layers:
        H_seq = layer(
            A_seq=A_seq,
            X_seq=H_seq,
            mask_seq=mask_seq,
        )
        outputs.append(H_seq)

    return outputs


def print_layer_table(stage_reports: List[Dict[str, Any]]) -> None:
    print()
    print("Layer Variance Flow Before Graph Pooling")
    print("-" * 126)
    print(
        f"{'Stage':<32}"
        f"{'Shape':<18}"
        f"{'Avg Feature Var':>18}"
        f"{'Pairwise Dist/Feat':>22}"
        f"{'Var Retention':>18}"
        f"{'Pairwise Retention':>22}"
    )
    print("-" * 126)

    for report in stage_reports:
        print(
            f"{report['stage']:<32}"
            f"{str(report['shape']):<18}"
            f"{report['avg_feature_variance']:>18.8f}"
            f"{report['avg_pairwise_squared_distance_per_feature']:>22.8f}"
            f"{report['variance_retention_from_previous']:>18.8f}"
            f"{report['pairwise_retention_from_previous']:>22.8f}"
        )

    print("-" * 126)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose EvolveGCN-H per-layer variance flow before graph pooling."
    )

    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_path = (
        Path(args.config_path)
        if args.config_path is not None
        else experiment_dir / "config.json"
    )
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else experiment_dir / "checkpoints" / "best_model.pt"
    )

    config = load_json(config_path)
    dataset_path = Path(
        args.dataset_path if args.dataset_path is not None else config["dataset_path"]
    )
    device = choose_device(args.device)

    data = load_temporal_dataset(dataset_path)

    split_key = f"{args.split}_ids"
    if split_key not in config:
        raise KeyError(f"Config does not contain split key: {split_key}")

    split_ids = list(config[split_key])
    validate_split_ids(
        split_ids=split_ids,
        dataset_ids=list(data.keys()),
        split_name=args.split,
    )

    batch_size = int(args.batch_size or config.get("batch_size", 4))

    dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids,
        use_summary_features=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    first_batch = next(iter(loader))
    _, _, X_example, _, _, _ = unpack_batch(first_batch)
    node_features = int(X_example.shape[-1])

    model = build_model_from_config(
        config=config,
        node_features=node_features,
        device=device,
    )
    checkpoint = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stage_names = ["Layer 0 input"] + [
        f"Layer {index + 1} output"
        for index in range(len(model.layers))
    ]

    stage_vectors: List[List[torch.Tensor]] = [
        [] for _ in range(len(stage_names))
    ]
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, _, _ = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        layer_outputs = extract_layer_outputs(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )

        for index, tensor in enumerate(layer_outputs):
            stage_vectors[index].append(sample_matrix(tensor))

        universe_ids_all.extend(universe_ids)

    stage_reports: List[Dict[str, Any]] = []

    for name, vectors in zip(stage_names, stage_vectors):
        sample_vectors = torch.cat(vectors, dim=0)
        report = stage_metrics(sample_vectors)
        report["stage"] = name
        stage_reports.append(report)

    add_retention_ratios(stage_reports)

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "num_samples": len(universe_ids_all),
        "universe_ids": universe_ids_all,
        "model": {
            "hidden_dim": int(config["hidden_dim"]),
            "num_layers": int(config["num_layers"]),
            "dropout": float(config["dropout"]),
            "temporal_pooling": str(config["temporal_pooling"]),
            "graph_pooling": str(config["graph_pooling"]),
            "summary_feature_dim": int(config.get("summary_feature_dim", 0)),
            "use_summary_features": bool(config.get("use_summary_features", False)),
        },
        "stages": stage_reports,
    }

    print("=" * 126)
    print("EVOLVEGCN-H LAYER VARIANCE DIAGNOSTIC")
    print("=" * 126)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Device:         {device}")

    print_layer_table(stage_reports)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_layer_variance.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
