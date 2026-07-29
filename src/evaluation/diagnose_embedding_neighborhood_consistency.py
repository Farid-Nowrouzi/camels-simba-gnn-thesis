from __future__ import annotations

"""
diagnose_embedding_neighborhood_consistency.py

Pure diagnostic for testing whether EvolveGCN-H temporal embedding space is
organized by Omega_m.

This script does not train, change preprocessing, alter datasets, modify
splits, edit checkpoints, or change model behavior. It loads a trained
experiment, reuses saved split IDs, extracts temporal embeddings for
train/val/test, and evaluates nearest-neighbor target consistency.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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


K_VALUES = [1, 3, 5, 10]


@torch.no_grad()
def extract_temporal_embeddings(
    model,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
) -> torch.Tensor:
    H_seq = X_seq

    for layer in model.layers:
        H_seq = layer(
            A_seq=A_seq,
            X_seq=H_seq,
            mask_seq=mask_seq,
        )

    graph_embeddings = model.masked_graph_pool(
        X_seq=H_seq,
        mask_seq=mask_seq,
        mode=model.graph_pooling,
    )

    return model.temporal_pool(graph_embeddings)


def extract_split_embeddings(
    model,
    data: Dict[str, Dict[str, Any]],
    split_ids: List[str],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

    embedding_batches = []
    target_batches = []
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, _ = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        embeddings = extract_temporal_embeddings(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )

        embedding_batches.append(embeddings.detach().cpu().float().numpy())
        target_batches.append(target.detach().cpu().float().view(-1).numpy())
        universe_ids_all.extend(universe_ids)

    return (
        np.concatenate(embedding_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        universe_ids_all,
    )


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1

        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[sorter[start:end]] = average_rank
        start = end

    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson_corr(predictions, targets),
    }


def knn_predict_from_distance_matrix(
    distances: np.ndarray,
    source_targets: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, int]:
    effective_k = min(k, distances.shape[1])
    neighbor_indices = np.argsort(distances, axis=1)[:, :effective_k]
    predictions = source_targets[neighbor_indices].mean(axis=1)

    return predictions, effective_k


def same_split_knn(
    embeddings: np.ndarray,
    targets: np.ndarray,
    k_values: List[int],
) -> Dict[str, Any]:
    distances = pairwise_distances(embeddings, embeddings)
    np.fill_diagonal(distances, np.inf)

    results = {}
    for k in k_values:
        predictions, effective_k = knn_predict_from_distance_matrix(
            distances=distances,
            source_targets=targets,
            k=k,
        )
        results[str(k)] = {
            "requested_k": k,
            "effective_k": effective_k,
            **regression_metrics(predictions, targets),
        }

    return results


def cross_split_knn(
    query_embeddings: np.ndarray,
    query_targets: np.ndarray,
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    k_values: List[int],
) -> Dict[str, Any]:
    distances = pairwise_distances(query_embeddings, train_embeddings)

    results = {}
    for k in k_values:
        predictions, effective_k = knn_predict_from_distance_matrix(
            distances=distances,
            source_targets=train_targets,
            k=k,
        )
        results[str(k)] = {
            "requested_k": k,
            "effective_k": effective_k,
            **regression_metrics(predictions, query_targets),
        }

    return results


def distance_target_difference_correlations(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float | int]:
    if len(targets) < 2:
        return {
            "num_pairs": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    distances = pairwise_distances(embeddings, embeddings)
    target_diffs = np.abs(targets[:, None] - targets[None, :])
    upper = np.triu_indices(len(targets), k=1)

    distance_values = distances[upper]
    target_diff_values = target_diffs[upper]

    return {
        "num_pairs": int(len(distance_values)),
        "pearson": pearson_corr(distance_values, target_diff_values),
        "spearman": spearman_corr(distance_values, target_diff_values),
    }


def print_knn_table(title: str, results_by_split: Dict[str, Dict[str, Any]]) -> None:
    print()
    print(title)
    print("-" * 104)
    print(
        f"{'Split':<12}"
        f"{'k':>5}"
        f"{'eff_k':>7}"
        f"{'R2':>12}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'Pearson':>12}"
    )
    print("-" * 104)

    for split, results in results_by_split.items():
        for k in K_VALUES:
            row = results[str(k)]
            print(
                f"{split:<12}"
                f"{k:>5}"
                f"{row['effective_k']:>7}"
                f"{row['r2']:>12.6f}"
                f"{row['mae']:>12.6f}"
                f"{row['rmse']:>12.6f}"
                f"{row['pearson']:>12.6f}"
            )


def print_distance_correlation_table(correlations: Dict[str, Dict[str, Any]]) -> None:
    print()
    print("Embedding distance vs absolute Omega_m difference")
    print("-" * 72)
    print(f"{'Split':<12}{'Pairs':>10}{'Pearson':>16}{'Spearman':>16}")
    print("-" * 72)

    for split, row in correlations.items():
        print(
            f"{split:<12}"
            f"{row['num_pairs']:>10}"
            f"{row['pearson']:>16.6f}"
            f"{row['spearman']:>16.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose temporal embedding neighborhood consistency."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
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
    dataset_ids = list(data.keys())

    split_ids = {}
    for split in ["train", "val", "test"]:
        key = f"{split}_ids"
        if key not in config:
            raise KeyError(f"Config does not contain split key: {key}")
        split_ids[split] = list(config[key])
        validate_split_ids(
            split_ids=split_ids[split],
            dataset_ids=dataset_ids,
            split_name=split,
        )

    batch_size = int(args.batch_size or config.get("batch_size", 4))

    first_dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids["train"],
        use_summary_features=False,
    )
    first_loader = DataLoader(
        first_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    first_batch = next(iter(first_loader))
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

    embeddings_by_split = {}
    targets_by_split = {}
    ids_by_split = {}

    for split in ["train", "val", "test"]:
        embeddings, targets, ids = extract_split_embeddings(
            model=model,
            data=data,
            split_ids=split_ids[split],
            batch_size=batch_size,
            device=device,
        )
        embeddings_by_split[split] = embeddings
        targets_by_split[split] = targets
        ids_by_split[split] = ids

    same_split_results = {
        split: same_split_knn(
            embeddings=embeddings_by_split[split],
            targets=targets_by_split[split],
            k_values=K_VALUES,
        )
        for split in ["train", "val", "test"]
    }
    distance_correlations = {
        split: distance_target_difference_correlations(
            embeddings=embeddings_by_split[split],
            targets=targets_by_split[split],
        )
        for split in ["train", "val", "test"]
    }
    cross_split_results = {
        split: cross_split_knn(
            query_embeddings=embeddings_by_split[split],
            query_targets=targets_by_split[split],
            train_embeddings=embeddings_by_split["train"],
            train_targets=targets_by_split["train"],
            k_values=K_VALUES,
        )
        for split in ["val", "test"]
    }

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "model": {
            "hidden_dim": int(config["hidden_dim"]),
            "num_layers": int(config["num_layers"]),
            "dropout": float(config["dropout"]),
            "activation": str(config.get("activation", "relu")),
            "temporal_pooling": str(config["temporal_pooling"]),
            "graph_pooling": str(config["graph_pooling"]),
            "head_type": str(config.get("head_type", "mlp")),
            "summary_feature_dim": int(config.get("summary_feature_dim", 0)),
            "use_summary_features": bool(config.get("use_summary_features", False)),
        },
        "splits": {
            split: {
                "num_samples": int(len(ids_by_split[split])),
                "embedding_shape": list(embeddings_by_split[split].shape),
                "target_mean": float(np.mean(targets_by_split[split])),
                "target_std": float(np.std(targets_by_split[split])),
                "universe_ids": ids_by_split[split],
            }
            for split in ["train", "val", "test"]
        },
        "same_split_knn": same_split_results,
        "distance_target_difference_correlations": distance_correlations,
        "train_neighbor_knn": cross_split_results,
    }

    print("=" * 104)
    print("TEMPORAL EMBEDDING NEIGHBORHOOD CONSISTENCY")
    print("=" * 104)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Device:         {device}")

    print_knn_table("Same-split kNN regression", same_split_results)
    print_distance_correlation_table(distance_correlations)
    print_knn_table("Validation/Test using TRAIN neighbors only", cross_split_results)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "embedding_neighborhood_consistency.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
