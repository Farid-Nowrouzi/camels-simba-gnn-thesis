from __future__ import annotations

"""
diagnose_embedding_feature_stability.py

Pure diagnostic for checking whether temporal embedding dimensions that
correlate with Omega_m on the training split keep the same relationship on
validation and test splits.

This script does not train, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior.
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def per_dimension_correlations(
    embeddings: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    correlations = []

    for dim in range(embeddings.shape[1]):
        correlations.append(pearson_corr(embeddings[:, dim], targets))

    return np.asarray(correlations, dtype=np.float64)


def safe_abs(value: float) -> float:
    if np.isnan(value):
        return float("nan")

    return abs(value)


def sign_agreement(left: float, right: float) -> bool:
    if np.isnan(left) or np.isnan(right):
        return False

    if left == 0.0 or right == 0.0:
        return False

    return bool(np.sign(left) == np.sign(right))


def dimension_records(
    train_corr: np.ndarray,
    val_corr: np.ndarray,
    test_corr: np.ndarray,
    strong_threshold: float,
    collapse_ratio: float,
) -> List[Dict[str, Any]]:
    records = []

    for dim in range(len(train_corr)):
        train_value = float(train_corr[dim])
        val_value = float(val_corr[dim])
        test_value = float(test_corr[dim])
        abs_train = safe_abs(train_value)
        abs_val = safe_abs(val_value)
        abs_test = safe_abs(test_value)

        val_sign_agreement = sign_agreement(train_value, val_value)
        test_sign_agreement = sign_agreement(train_value, test_value)
        strong_on_train = bool(not np.isnan(abs_train) and abs_train >= strong_threshold)
        val_sign_flip = bool(strong_on_train and not val_sign_agreement)
        test_sign_flip = bool(strong_on_train and not test_sign_agreement)
        val_collapsed = bool(
            strong_on_train
            and not np.isnan(abs_val)
            and abs_val < collapse_ratio * abs_train
        )
        test_collapsed = bool(
            strong_on_train
            and not np.isnan(abs_test)
            and abs_test < collapse_ratio * abs_train
        )

        stability_score = min(
            0.0 if np.isnan(abs_val) else abs_val,
            0.0 if np.isnan(abs_test) else abs_test,
        )
        instability_score = (
            (0.0 if np.isnan(abs_train) else abs_train)
            + abs(train_value - val_value)
            + abs(train_value - test_value)
            + (1.0 if val_sign_flip else 0.0)
            + (1.0 if test_sign_flip else 0.0)
            + (0.5 if val_collapsed else 0.0)
            + (0.5 if test_collapsed else 0.0)
        )

        records.append(
            {
                "dimension": int(dim),
                "train_correlation": train_value,
                "val_correlation": val_value,
                "test_correlation": test_value,
                "abs_train_correlation": abs_train,
                "abs_val_correlation": abs_val,
                "abs_test_correlation": abs_test,
                "val_sign_agreement": val_sign_agreement,
                "test_sign_agreement": test_sign_agreement,
                "val_correlation_difference": val_value - train_value,
                "test_correlation_difference": test_value - train_value,
                "strong_on_train": strong_on_train,
                "val_sign_flip": val_sign_flip,
                "test_sign_flip": test_sign_flip,
                "val_collapsed_toward_zero": val_collapsed,
                "test_collapsed_toward_zero": test_collapsed,
                "stability_score": float(stability_score),
                "instability_score": float(instability_score),
            }
        )

    return records


def top_stable_dimensions(
    records: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    stable = [
        record
        for record in records
        if record["val_sign_agreement"]
        and record["test_sign_agreement"]
        and not record["val_collapsed_toward_zero"]
        and not record["test_collapsed_toward_zero"]
    ]

    stable.sort(
        key=lambda record: (
            record["stability_score"],
            record["abs_train_correlation"],
        ),
        reverse=True,
    )

    return stable[:limit]


def top_unstable_dimensions(
    records: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    unstable = [
        record
        for record in records
        if record["strong_on_train"]
        and (
            record["val_sign_flip"]
            or record["test_sign_flip"]
            or record["val_collapsed_toward_zero"]
            or record["test_collapsed_toward_zero"]
        )
    ]

    unstable.sort(
        key=lambda record: record["instability_score"],
        reverse=True,
    )

    return unstable[:limit]


def summarize_flags(records: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "num_dimensions": len(records),
        "strong_on_train": sum(record["strong_on_train"] for record in records),
        "val_sign_flip": sum(record["val_sign_flip"] for record in records),
        "test_sign_flip": sum(record["test_sign_flip"] for record in records),
        "val_collapsed_toward_zero": sum(
            record["val_collapsed_toward_zero"] for record in records
        ),
        "test_collapsed_toward_zero": sum(
            record["test_collapsed_toward_zero"] for record in records
        ),
    }


def print_dimension_table(title: str, records: List[Dict[str, Any]]) -> None:
    print()
    print(title)
    print("-" * 112)
    print(
        f"{'Dim':>6}"
        f"{'Train r':>14}"
        f"{'Val r':>14}"
        f"{'Test r':>14}"
        f"{'Val Sign':>12}"
        f"{'Test Sign':>12}"
        f"{'Val Diff':>14}"
        f"{'Test Diff':>14}"
    )
    print("-" * 112)

    for record in records:
        print(
            f"{record['dimension']:>6}"
            f"{record['train_correlation']:>14.6f}"
            f"{record['val_correlation']:>14.6f}"
            f"{record['test_correlation']:>14.6f}"
            f"{str(record['val_sign_agreement']):>12}"
            f"{str(record['test_sign_agreement']):>12}"
            f"{record['val_correlation_difference']:>14.6f}"
            f"{record['test_correlation_difference']:>14.6f}"
        )


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("TEMPORAL EMBEDDING FEATURE STABILITY DIAGNOSTIC")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Checkpoint:     {output['checkpoint_path']}")
    print(f"Dataset:        {output['dataset_path']}")
    print()
    print("Correlation pattern agreement")
    print("-" * 112)
    print(
        f"corr(train dimension correlations, val dimension correlations):  "
        f"{output['correlation_vector_agreement']['train_vs_val']:.6f}"
    )
    print(
        f"corr(train dimension correlations, test dimension correlations): "
        f"{output['correlation_vector_agreement']['train_vs_test']:.6f}"
    )
    print()
    print("Flag counts")
    print("-" * 112)
    for key, value in output["flag_counts"].items():
        print(f"{key:<32} {value}")

    print_dimension_table("Top 20 stable dimensions", output["top_stable_dimensions"])
    print_dimension_table("Top 20 unstable dimensions", output["top_unstable_dimensions"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose split stability of embedding-dimension correlations."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--strong_threshold", type=float, default=0.30)
    parser.add_argument("--collapse_ratio", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=20)

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
        embeddings, targets, universe_ids = extract_split_embeddings(
            model=model,
            data=data,
            split_ids=split_ids[split],
            batch_size=batch_size,
            device=device,
        )
        embeddings_by_split[split] = embeddings
        targets_by_split[split] = targets
        ids_by_split[split] = universe_ids

    train_corr = per_dimension_correlations(
        embeddings_by_split["train"],
        targets_by_split["train"],
    )
    val_corr = per_dimension_correlations(
        embeddings_by_split["val"],
        targets_by_split["val"],
    )
    test_corr = per_dimension_correlations(
        embeddings_by_split["test"],
        targets_by_split["test"],
    )

    records = dimension_records(
        train_corr=train_corr,
        val_corr=val_corr,
        test_corr=test_corr,
        strong_threshold=float(args.strong_threshold),
        collapse_ratio=float(args.collapse_ratio),
    )

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "thresholds": {
            "strong_threshold_abs_train_corr": float(args.strong_threshold),
            "collapse_ratio": float(args.collapse_ratio),
        },
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
                "num_samples": int(embeddings_by_split[split].shape[0]),
                "embedding_shape": list(embeddings_by_split[split].shape),
                "target_mean": float(np.mean(targets_by_split[split])),
                "target_std": float(np.std(targets_by_split[split])),
                "universe_ids": ids_by_split[split],
            }
            for split in ["train", "val", "test"]
        },
        "correlation_vector_agreement": {
            "train_vs_val": pearson_corr(train_corr, val_corr),
            "train_vs_test": pearson_corr(train_corr, test_corr),
        },
        "flag_counts": summarize_flags(records),
        "all_dimensions": records,
        "top_stable_dimensions": top_stable_dimensions(
            records=records,
            limit=int(args.top_k),
        ),
        "top_unstable_dimensions": top_unstable_dimensions(
            records=records,
            limit=int(args.top_k),
        ),
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "embedding_feature_stability.json"
    )
    save_json(output, output_path)

    print_summary(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
