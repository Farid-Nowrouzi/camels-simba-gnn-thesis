from __future__ import annotations

"""
diagnose_evolvegcn_h_feature_variance.py

Pure diagnostic for measuring per-dimension variance in EvolveGCN-H graph and
temporal embeddings.

This script does not train, change preprocessing, alter dataset construction,
modify model architecture, edit checkpoints, change splits, or alter model
behavior. It loads a trained experiment config/checkpoint, reuses saved split
IDs, and reports whether variance collapse is uniform across embedding
dimensions or concentrated in a subset.
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


def sample_matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    return tensor.reshape(tensor.shape[0], -1)


def per_dimension_stats(sample_vectors: torch.Tensor) -> Dict[str, Any]:
    sample_vectors = sample_vectors.detach().cpu().float()

    if sample_vectors.ndim != 2:
        raise ValueError(
            f"Expected sample vectors [B, D], got {tuple(sample_vectors.shape)}"
        )

    variance = sample_vectors.var(dim=0, unbiased=False)
    std = sample_vectors.std(dim=0, unbiased=False)
    min_values = sample_vectors.min(dim=0).values
    max_values = sample_vectors.max(dim=0).values

    return {
        "shape": list(sample_vectors.shape),
        "num_samples": int(sample_vectors.shape[0]),
        "num_dimensions": int(sample_vectors.shape[1]),
        "variance": variance.tolist(),
        "std": std.tolist(),
        "min": min_values.tolist(),
        "max": max_values.tolist(),
    }


def ranked_dimensions(
    stats: Dict[str, Any],
    largest: bool,
    limit: int = 10,
) -> List[Dict[str, float | int]]:
    variance = torch.tensor(stats["variance"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    min_values = torch.tensor(stats["min"], dtype=torch.float32)
    max_values = torch.tensor(stats["max"], dtype=torch.float32)

    k = min(limit, int(variance.numel()))
    values, indices = torch.topk(variance, k=k, largest=largest)

    rows: List[Dict[str, float | int]] = []

    for value, index in zip(values, indices):
        dim = int(index.item())
        rows.append(
            {
                "dimension": dim,
                "variance": float(value.item()),
                "std": float(std[dim].item()),
                "min": float(min_values[dim].item()),
                "max": float(max_values[dim].item()),
            }
        )

    return rows


def collapse_percentages(stats: Dict[str, Any]) -> Dict[str, float]:
    variance = torch.tensor(stats["variance"], dtype=torch.float32)

    if variance.numel() == 0:
        return {
            "variance_lt_1e-6": float("nan"),
            "variance_lt_1e-5": float("nan"),
            "variance_lt_1e-4": float("nan"),
        }

    return {
        "variance_lt_1e-6": float((variance < 1e-6).float().mean().item() * 100.0),
        "variance_lt_1e-5": float((variance < 1e-5).float().mean().item() * 100.0),
        "variance_lt_1e-4": float((variance < 1e-4).float().mean().item() * 100.0),
    }


def summarize_stage(stats: Dict[str, Any]) -> Dict[str, Any]:
    percentages = collapse_percentages(stats)

    return {
        "shape": stats["shape"],
        "num_samples": stats["num_samples"],
        "num_dimensions": stats["num_dimensions"],
        "top_10_highest_variance_dimensions": ranked_dimensions(
            stats,
            largest=True,
            limit=10,
        ),
        "bottom_10_lowest_variance_dimensions": ranked_dimensions(
            stats,
            largest=False,
            limit=10,
        ),
        "percent_dimensions_with_variance_lt_1e-6": percentages["variance_lt_1e-6"],
        "percent_dimensions_with_variance_lt_1e-5": percentages["variance_lt_1e-5"],
        "percent_dimensions_with_variance_lt_1e-4": percentages["variance_lt_1e-4"],
        "per_dimension": stats,
    }


def print_rank_table(title: str, rows: List[Dict[str, float | int]]) -> None:
    print()
    print(title)
    print("-" * 78)
    print(
        f"{'dim':>6}"
        f"{'variance':>18}"
        f"{'std':>18}"
        f"{'min':>18}"
        f"{'max':>18}"
    )
    print("-" * 78)

    for row in rows:
        print(
            f"{int(row['dimension']):>6}"
            f"{float(row['variance']):>18.8f}"
            f"{float(row['std']):>18.8f}"
            f"{float(row['min']):>18.8f}"
            f"{float(row['max']):>18.8f}"
        )


def print_stage_summary(stage_name: str, summary: Dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(stage_name)
    print("=" * 78)
    print(f"shape:                                  {summary['shape']}")
    print(f"samples:                                {summary['num_samples']}")
    print(f"dimensions:                             {summary['num_dimensions']}")
    print(
        "percent dims variance < 1e-6:          "
        f"{summary['percent_dimensions_with_variance_lt_1e-6']:.2f}%"
    )
    print(
        "percent dims variance < 1e-5:          "
        f"{summary['percent_dimensions_with_variance_lt_1e-5']:.2f}%"
    )
    print(
        "percent dims variance < 1e-4:          "
        f"{summary['percent_dimensions_with_variance_lt_1e-4']:.2f}%"
    )

    print_rank_table(
        "Top 10 highest-variance dimensions",
        summary["top_10_highest_variance_dimensions"],
    )
    print_rank_table(
        "Bottom 10 lowest-variance dimensions",
        summary["bottom_10_lowest_variance_dimensions"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose per-dimension variance in EvolveGCN-H embeddings."
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

    graph_vectors: List[torch.Tensor] = []
    temporal_vectors: List[torch.Tensor] = []
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, _, _ = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        embeddings = extract_graph_and_temporal_embeddings(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )

        graph_vectors.append(sample_matrix(embeddings["graph_embeddings"]))
        temporal_vectors.append(sample_matrix(embeddings["temporal_embeddings"]))
        universe_ids_all.extend(universe_ids)

    graph_matrix = torch.cat(graph_vectors, dim=0)
    temporal_matrix = torch.cat(temporal_vectors, dim=0)

    graph_summary = summarize_stage(per_dimension_stats(graph_matrix))
    temporal_summary = summarize_stage(per_dimension_stats(temporal_matrix))

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
        "graph_embeddings": graph_summary,
        "temporal_embeddings": temporal_summary,
    }

    print("=" * 78)
    print("EVOLVEGCN-H FEATURE VARIANCE DIAGNOSTIC")
    print("=" * 78)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Device:         {device}")

    print_stage_summary("GRAPH EMBEDDINGS", graph_summary)
    print_stage_summary("TEMPORAL EMBEDDINGS", temporal_summary)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_feature_variance.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
