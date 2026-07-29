from __future__ import annotations

"""
diagnose_embedding_probe_splits.py

Pure diagnostic for evaluating whether temporal embeddings contain Omega_m
signal that generalizes across train/validation/test splits.

This script does not train the GNN, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads a trained
EvolveGCN-H experiment, extracts temporal embeddings for all saved splits, fits
a linear probe on TRAIN embeddings only, and evaluates that same probe on
train/validation/test embeddings.
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = predictions - targets
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "r2": r2,
        "rmse": rmse,
        "pearson": pearson_corr(predictions, targets),
    }


def fit_linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    X_design = np.concatenate(
        [
            embeddings.astype(np.float64),
            np.ones((embeddings.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(X_design, targets.astype(np.float64), rcond=None)
    return coefficients


def predict_linear_probe(
    embeddings: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    X_design = np.concatenate(
        [
            embeddings.astype(np.float64),
            np.ones((embeddings.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    return X_design @ coefficients


def split_summary(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    own_coefficients = fit_linear_probe(embeddings, targets)
    own_predictions = predict_linear_probe(embeddings, own_coefficients)
    own_metrics = regression_metrics(own_predictions, targets)

    return {
        "num_samples": int(embeddings.shape[0]),
        "embedding_shape": list(embeddings.shape),
        "embedding_mean": float(np.mean(embeddings)),
        "embedding_std": float(np.std(embeddings)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "within_split_linear_probe": own_metrics,
    }


def extract_split(
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


def print_summary_table(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("TEMPORAL EMBEDDING LINEAR PROBE SPLIT DIAGNOSTIC")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Checkpoint:     {output['checkpoint_path']}")
    print(f"Dataset:        {output['dataset_path']}")
    print()
    print("Within-split probes")
    print("-" * 112)
    print(
        f"{'Split':<12}"
        f"{'N':>6}"
        f"{'Emb Std':>12}"
        f"{'Target Std':>12}"
        f"{'Probe R2':>12}"
        f"{'Probe RMSE':>14}"
        f"{'Probe Pearson':>16}"
    )
    print("-" * 112)

    for split in ["train", "val", "test"]:
        summary = output["splits"][split]
        metrics = summary["within_split_linear_probe"]
        print(
            f"{split:<12}"
            f"{summary['num_samples']:>6}"
            f"{summary['embedding_std']:>12.6f}"
            f"{summary['target_std']:>12.6f}"
            f"{metrics['r2']:>12.6f}"
            f"{metrics['rmse']:>14.6f}"
            f"{metrics['pearson']:>16.6f}"
        )

    print()
    print("One probe fit on TRAIN embeddings only")
    print("-" * 112)
    print(
        f"{'Eval Split':<12}"
        f"{'R2':>12}"
        f"{'RMSE':>14}"
        f"{'Pearson':>16}"
    )
    print("-" * 112)

    for split in ["train", "val", "test"]:
        metrics = output["train_probe_evaluation"][split]
        print(
            f"{split:<12}"
            f"{metrics['r2']:>12.6f}"
            f"{metrics['rmse']:>14.6f}"
            f"{metrics['pearson']:>16.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal embedding linear probes across saved splits."
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
        embeddings, targets, ids = extract_split(
            model=model,
            data=data,
            split_ids=split_ids[split],
            batch_size=batch_size,
            device=device,
        )
        embeddings_by_split[split] = embeddings
        targets_by_split[split] = targets
        ids_by_split[split] = ids

    train_probe_coefficients = fit_linear_probe(
        embeddings_by_split["train"],
        targets_by_split["train"],
    )

    train_probe_evaluation = {}
    for split in ["train", "val", "test"]:
        predictions = predict_linear_probe(
            embeddings_by_split[split],
            train_probe_coefficients,
        )
        train_probe_evaluation[split] = regression_metrics(
            predictions,
            targets_by_split[split],
        )

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
                **split_summary(embeddings_by_split[split], targets_by_split[split]),
                "universe_ids": ids_by_split[split],
            }
            for split in ["train", "val", "test"]
        },
        "train_probe_evaluation": train_probe_evaluation,
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "embedding_probe_splits.json"
    )
    save_json(output, output_path)

    print_summary_table(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
