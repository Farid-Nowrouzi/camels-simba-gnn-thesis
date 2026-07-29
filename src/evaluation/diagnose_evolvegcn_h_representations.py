from __future__ import annotations

"""
diagnose_evolvegcn_h_representations.py

Inspect representation collapse in a trained EvolveGCN-H checkpoint.

This script does not train, modify checkpoints, change preprocessing, or change
splits. It loads an experiment config/checkpoint, reuses the saved split IDs,
and reports representation statistics at several stages:

    1. final node embeddings
    2. graph embeddings after graph pooling
    3. universe embeddings after temporal pooling
    4. regressor inputs
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.training.train_evolvegcn_h import (
    CamelsTemporalDataset,
    collate_fn,
    load_temporal_dataset,
    unpack_batch,
)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def validate_split_ids(
    split_ids: List[str],
    dataset_ids: List[str],
    split_name: str,
) -> None:
    dataset_id_set = set(dataset_ids)
    split_id_set = set(split_ids)

    if len(split_ids) == 0:
        raise ValueError(f"{split_name} split is empty.")

    if len(split_ids) != len(split_id_set):
        raise ValueError(f"{split_name} split contains duplicate IDs.")

    missing_ids = sorted(split_id_set - dataset_id_set)
    if missing_ids:
        raise ValueError(
            f"{split_name} split has IDs missing from dataset: {missing_ids[:20]}"
        )


def build_model_from_config(
    config: Dict[str, Any],
    node_features: int,
    device: torch.device,
) -> EvolveGCNHRegressor:
    model = EvolveGCNHRegressor(
        node_features=int(config.get("node_features", node_features)),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        activation=str(config.get("activation", "relu")),
        temporal_pooling=str(config["temporal_pooling"]),
        graph_pooling=str(config["graph_pooling"]),
        add_self_loops=bool(config.get("add_self_loops", True)),
        summary_feature_dim=int(config.get("summary_feature_dim", 0)),
        head_type=str(config.get("head_type", "mlp")),
    )

    return model.to(device)


def load_checkpoint_state(path: str | Path, device: torch.device) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def extract_representations(
    model: EvolveGCNHRegressor,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
    summary_features: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    H_seq = X_seq

    for layer in model.layers:
        H_seq = layer(
            A_seq=A_seq,
            X_seq=H_seq,
            mask_seq=mask_seq,
        )

    node_embeddings = H_seq

    graph_embeddings = model.masked_graph_pool(
        X_seq=node_embeddings,
        mask_seq=mask_seq,
        mode=model.graph_pooling,
    )

    universe_embeddings = model.temporal_pool(graph_embeddings)
    regressor_inputs = universe_embeddings

    if model.summary_feature_dim > 0:
        if summary_features is None:
            raise ValueError(
                "Checkpoint expects summary features, but loader returned None."
            )

        regressor_inputs = torch.cat(
            [regressor_inputs, summary_features.float()],
            dim=-1,
        )

    predictions = model.regressor(regressor_inputs)

    return {
        "node_embeddings": node_embeddings,
        "graph_embeddings": graph_embeddings,
        "universe_embeddings": universe_embeddings,
        "regressor_inputs": regressor_inputs,
        "predictions": predictions,
    }


def finite_stats(values: torch.Tensor) -> Dict[str, float | int | List[int]]:
    values = values.detach().cpu().float()

    if values.numel() == 0:
        return {
            "shape": list(values.shape),
            "num_values": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "shape": list(values.shape),
        "num_values": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def sample_matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    return tensor.reshape(tensor.shape[0], -1)


def pairwise_variance_stats(sample_vectors: torch.Tensor) -> Dict[str, float]:
    """
    Summarize variation across samples.

    avg_feature_variance_across_samples is the average variance of each feature
    dimension across universes. avg_pairwise_squared_distance_per_feature is the
    mean pairwise squared distance normalized by representation width.
    """
    sample_vectors = sample_vectors.detach().cpu().float()

    if sample_vectors.ndim != 2:
        raise ValueError(
            f"Expected sample vectors [B, D], got {tuple(sample_vectors.shape)}"
        )

    num_samples, width = sample_vectors.shape

    if num_samples < 2 or width == 0:
        return {
            "avg_feature_variance_across_samples": float("nan"),
            "avg_pairwise_squared_distance_per_feature": float("nan"),
        }

    feature_variance = sample_vectors.var(dim=0, unbiased=False).mean()
    pairwise_distances = torch.pdist(sample_vectors, p=2).pow(2) / float(width)

    return {
        "avg_feature_variance_across_samples": float(feature_variance.item()),
        "avg_pairwise_squared_distance_per_feature": float(
            pairwise_distances.mean().item()
        ),
    }


def stage_report(
    values: torch.Tensor,
    sample_vectors: torch.Tensor,
) -> Dict[str, Any]:
    report = finite_stats(values)
    report.update(pairwise_variance_stats(sample_vectors))
    return report


def print_stage_report(name: str, report: Dict[str, Any]) -> None:
    print()
    print(name)
    print("-" * 80)
    print(f"shape:                                   {report['shape']}")
    print(f"num_values:                              {report['num_values']}")
    print(f"mean:                                    {report['mean']:.8f}")
    print(f"std:                                     {report['std']:.8f}")
    print(f"min:                                     {report['min']:.8f}")
    print(f"max:                                     {report['max']:.8f}")
    print(
        "avg_feature_variance_across_samples:    "
        f"{report['avg_feature_variance_across_samples']:.8f}"
    )
    print(
        "avg_pairwise_squared_distance_per_feature: "
        f"{report['avg_pairwise_squared_distance_per_feature']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose representation collapse in EvolveGCN-H."
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

    use_summary_features = bool(config.get("use_summary_features", False))
    batch_size = int(args.batch_size or config.get("batch_size", 4))

    dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids,
        use_summary_features=use_summary_features,
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

    node_values = []
    node_sample_vectors = []
    graph_values = []
    graph_sample_vectors = []
    universe_values = []
    regressor_input_values = []
    predictions = []
    targets = []
    universe_ids_all = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)
        target = target.to(device)

        if summary_features is not None:
            summary_features = summary_features.to(device)

        reps = extract_representations(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )

        node_embeddings = reps["node_embeddings"]
        valid_node_mask = mask_seq.bool().expand_as(node_embeddings)
        node_values.append(node_embeddings[valid_node_mask].detach().cpu())
        node_sample_vectors.append(sample_matrix(node_embeddings))

        graph_embeddings = reps["graph_embeddings"]
        graph_values.append(graph_embeddings.detach().cpu().reshape(-1))
        graph_sample_vectors.append(sample_matrix(graph_embeddings))

        universe_embeddings = reps["universe_embeddings"]
        regressor_inputs = reps["regressor_inputs"]

        universe_values.append(universe_embeddings.detach().cpu())
        regressor_input_values.append(regressor_inputs.detach().cpu())
        predictions.append(reps["predictions"].detach().cpu())
        targets.append(target.detach().cpu())
        universe_ids_all.extend(universe_ids)

    node_values_tensor = torch.cat(node_values, dim=0)
    node_sample_matrix = torch.cat(node_sample_vectors, dim=0)
    graph_values_tensor = torch.cat(graph_values, dim=0)
    graph_sample_matrix = torch.cat(graph_sample_vectors, dim=0)
    universe_tensor = torch.cat(universe_values, dim=0)
    regressor_input_tensor = torch.cat(regressor_input_values, dim=0)
    prediction_tensor = torch.cat(predictions, dim=0).view(-1)
    target_tensor = torch.cat(targets, dim=0).view(-1)

    reports = {
        "node_embeddings_valid_nodes": stage_report(
            values=node_values_tensor,
            sample_vectors=node_sample_matrix,
        ),
        "graph_embeddings_after_graph_pooling": stage_report(
            values=graph_values_tensor,
            sample_vectors=graph_sample_matrix,
        ),
        "universe_embeddings_after_temporal_pooling": stage_report(
            values=universe_tensor,
            sample_vectors=universe_tensor,
        ),
        "regressor_inputs": stage_report(
            values=regressor_input_tensor,
            sample_vectors=regressor_input_tensor,
        ),
        "predictions": finite_stats(prediction_tensor),
        "targets": finite_stats(target_tensor),
    }

    if prediction_tensor.numel() > 1:
        reports["prediction_target"] = {
            "prediction_std": float(prediction_tensor.std(unbiased=False).item()),
            "target_std": float(target_tensor.std(unbiased=False).item()),
            "prediction_mean": float(prediction_tensor.mean().item()),
            "target_mean": float(target_tensor.mean().item()),
        }

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
            "use_summary_features": use_summary_features,
        },
        "reports": reports,
    }

    print("=" * 80)
    print("EVOLVEGCN-H REPRESENTATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Device:         {device}")

    for name in [
        "node_embeddings_valid_nodes",
        "graph_embeddings_after_graph_pooling",
        "universe_embeddings_after_temporal_pooling",
        "regressor_inputs",
    ]:
        print_stage_report(name, reports[name])

    print()
    print("predictions / targets")
    print("-" * 80)
    print(f"prediction mean: {reports['prediction_target']['prediction_mean']:.8f}")
    print(f"prediction std:  {reports['prediction_target']['prediction_std']:.8f}")
    print(f"target mean:     {reports['prediction_target']['target_mean']:.8f}")
    print(f"target std:      {reports['prediction_target']['target_std']:.8f}")

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_representation_stats.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
