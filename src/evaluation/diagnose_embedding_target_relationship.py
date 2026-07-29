from __future__ import annotations

"""
diagnose_embedding_target_relationship.py

Pure diagnostic for measuring whether EvolveGCN-H graph/temporal embeddings
contain information about Omega_m.

This script does not train, change preprocessing, alter datasets, modify
splits, edit checkpoints, or change model behavior. It loads a trained
experiment, reuses saved split IDs, extracts embeddings, and evaluates their
relationship to the target.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

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
def extract_graph_and_temporal_embeddings(
    model,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
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
    temporal_embeddings = model.temporal_pool(graph_embeddings)

    return {
        "graph_embeddings": graph_embeddings,
        "temporal_embeddings": temporal_embeddings,
    }


def sample_matrix(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    return tensor.reshape(tensor.shape[0], -1).numpy()


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def per_dimension_correlations(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    correlations = []

    for dim in range(embeddings.shape[1]):
        correlations.append(pearson_corr(embeddings[:, dim], targets))

    corr_array = np.asarray(correlations, dtype=np.float64)
    abs_corr = np.abs(corr_array)

    # Treat NaN correlations from constant dimensions as zero signal for ranking.
    ranking_scores = np.nan_to_num(abs_corr, nan=0.0)
    top_indices = np.argsort(-ranking_scores)[:10]

    top_dimensions = []
    for dim in top_indices:
        top_dimensions.append(
            {
                "dimension": int(dim),
                "correlation": float(corr_array[dim]),
                "absolute_correlation": float(abs_corr[dim])
                if not np.isnan(abs_corr[dim])
                else float("nan"),
            }
        )

    return {
        "num_dimensions": int(embeddings.shape[1]),
        "max_absolute_correlation": float(np.nanmax(abs_corr)),
        "mean_absolute_correlation": float(np.nanmean(abs_corr)),
        "median_absolute_correlation": float(np.nanmedian(abs_corr)),
        "top_10_dimensions_by_absolute_correlation": top_dimensions,
        "per_dimension_correlation": [
            float(value) if not np.isnan(value) else float("nan")
            for value in corr_array
        ],
    }


def fit_linear_probe(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    X_design = np.concatenate(
        [X, np.ones((X.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    predictions = X_design @ coefficients

    errors = predictions - y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson_correlation": pearson_corr(predictions, y),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def analyze_embedding_stage(
    name: str,
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    return {
        "name": name,
        "shape": list(embeddings.shape),
        "per_dimension_correlations": per_dimension_correlations(
            embeddings=embeddings,
            targets=targets,
        ),
        "linear_probe": fit_linear_probe(
            embeddings=embeddings,
            targets=targets,
        ),
    }


def print_stage_summary(stage: Dict[str, Any]) -> None:
    corr = stage["per_dimension_correlations"]
    probe = stage["linear_probe"]

    print()
    print(stage["name"])
    print("-" * 100)
    print(f"shape:                       {stage['shape']}")
    print(f"max abs dim corr:            {corr['max_absolute_correlation']:.6f}")
    print(f"mean abs dim corr:           {corr['mean_absolute_correlation']:.6f}")
    print(f"median abs dim corr:         {corr['median_absolute_correlation']:.6f}")
    print(f"linear probe R2:             {probe['r2']:.6f}")
    print(f"linear probe MAE:            {probe['mae']:.6f}")
    print(f"linear probe RMSE:           {probe['rmse']:.6f}")
    print(f"linear probe Pearson:        {probe['pearson_correlation']:.6f}")

    print()
    print("Top 10 dimensions by absolute target correlation")
    print("-" * 100)
    print(f"{'dim':>6}{'corr':>16}{'abs_corr':>16}")

    for row in corr["top_10_dimensions_by_absolute_correlation"]:
        print(
            f"{row['dimension']:>6}"
            f"{row['correlation']:>16.6f}"
            f"{row['absolute_correlation']:>16.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose target relationship in EvolveGCN-H embeddings."
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

    graph_embeddings_all: List[np.ndarray] = []
    temporal_embeddings_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, _ = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        embeddings = extract_graph_and_temporal_embeddings(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )

        graph_embeddings_all.append(sample_matrix(embeddings["graph_embeddings"]))
        temporal_embeddings_all.append(sample_matrix(embeddings["temporal_embeddings"]))
        targets_all.append(target.detach().cpu().float().view(-1).numpy())
        universe_ids_all.extend(universe_ids)

    graph_embeddings = np.concatenate(graph_embeddings_all, axis=0)
    temporal_embeddings = np.concatenate(temporal_embeddings_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)

    graph_stage = analyze_embedding_stage(
        name="GRAPH EMBEDDINGS",
        embeddings=graph_embeddings,
        targets=targets,
    )
    temporal_stage = analyze_embedding_stage(
        name="TEMPORAL EMBEDDINGS",
        embeddings=temporal_embeddings,
        targets=targets,
    )

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "num_samples": len(universe_ids_all),
        "universe_ids": universe_ids_all,
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "model": {
            "hidden_dim": int(config["hidden_dim"]),
            "num_layers": int(config["num_layers"]),
            "dropout": float(config["dropout"]),
            "activation": str(config.get("activation", "relu")),
            "temporal_pooling": str(config["temporal_pooling"]),
            "graph_pooling": str(config["graph_pooling"]),
            "summary_feature_dim": int(config.get("summary_feature_dim", 0)),
            "use_summary_features": use_summary_features,
        },
        "graph_embeddings": graph_stage,
        "temporal_embeddings": temporal_stage,
    }

    print("=" * 100)
    print("EMBEDDING TARGET RELATIONSHIP DIAGNOSTIC")
    print("=" * 100)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Target mean:    {np.mean(targets):.6f}")
    print(f"Target std:     {np.std(targets):.6f}")
    print(f"Device:         {device}")

    print_stage_summary(graph_stage)
    print_stage_summary(temporal_stage)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_embedding_target_relationship.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
