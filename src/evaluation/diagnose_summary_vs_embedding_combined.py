from __future__ import annotations

"""
diagnose_summary_vs_embedding_combined.py

Pure diagnostic for testing whether trained EvolveGCN-H temporal embeddings add
useful information beyond simple temporal summary features.

This script does not train the GNN, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads an existing
experiment config/checkpoint, reuses saved split IDs, extracts summary features
and temporal embeddings, then fits Ridge probes on TRAIN only.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.evaluation.diagnose_evolvegcn_h_representations import (
    build_model_from_config,
    choose_device,
    load_checkpoint_state,
    load_json,
    save_json,
    validate_split_ids,
)
from src.evaluation.run_summary_feature_baseline import (
    build_arrays_for_ids,
    extract_temporal_summary,
    get_target,
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


def build_summary_arrays(
    data: Dict[str, Dict[str, Any]],
    universe_ids: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    features_by_id = {}
    targets_by_id = {}

    for universe_id in universe_ids:
        sample = data[universe_id]
        features_by_id[universe_id] = extract_temporal_summary(sample)
        targets_by_id[universe_id] = get_target(sample)

    return features_by_id, targets_by_id


def check_same_order(expected_ids: List[str], actual_ids: List[str], split: str) -> None:
    if expected_ids != actual_ids:
        raise ValueError(
            f"{split} embedding extraction changed ID order. "
            f"Expected first IDs {expected_ids[:5]}, got {actual_ids[:5]}"
        )


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "pearson": pearson_corr(predictions, targets),
    }


def fit_and_evaluate_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(X_train, y_train)

    results = {}
    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        predictions = model.predict(X)
        results[split] = regression_metrics(
            predictions=np.asarray(predictions, dtype=np.float64),
            targets=np.asarray(y, dtype=np.float64),
        )

    return results


def format_metric(value: float) -> str:
    if value != value:
        return "nan"

    return f"{value:.6f}"


def table_row(feature_set: str, results: Dict[str, Dict[str, float]]) -> str:
    return (
        f"{feature_set:<24}"
        f"{format_metric(results['train']['r2']):>12}"
        f"{format_metric(results['val']['r2']):>12}"
        f"{format_metric(results['test']['r2']):>12}"
        f"{format_metric(results['test']['mae']):>14}"
        f"{format_metric(results['test']['rmse']):>14}"
        f"{format_metric(results['test']['pearson']):>16}"
    )


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("SUMMARY FEATURES VS EVOLVEGCN-H TEMPORAL EMBEDDINGS")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Checkpoint:     {output['checkpoint_path']}")
    print(f"Dataset:        {output['dataset_path']}")
    print(f"Summary dim:    {output['feature_dimensions']['summary']}")
    print(f"Embedding dim:  {output['feature_dimensions']['embedding']}")
    print()
    print(
        f"{'Feature Set':<24}"
        f"{'Train R2':>12}"
        f"{'Val R2':>12}"
        f"{'Test R2':>12}"
        f"{'Test MAE':>14}"
        f"{'Test RMSE':>14}"
        f"{'Test Pearson':>16}"
    )
    print("-" * 112)

    for feature_set, results in output["ridge_results"].items():
        print(table_row(feature_set, results))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Ridge probes on summary, embedding, and combined features."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)

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
    universe_ids = sorted(data.keys())

    split_ids = {}
    for split in ["train", "val", "test"]:
        key = f"{split}_ids"
        if key not in config:
            raise KeyError(f"Config does not contain split key: {key}")
        split_ids[split] = list(config[key])
        validate_split_ids(
            split_ids=split_ids[split],
            dataset_ids=universe_ids,
            split_name=split,
        )

    batch_size = int(args.batch_size or config.get("batch_size", 4))

    summary_features_by_id, summary_targets_by_id = build_summary_arrays(
        data=data,
        universe_ids=universe_ids,
    )

    summary_arrays = {}
    targets_from_summary = {}
    for split in ["train", "val", "test"]:
        X_summary, y_summary = build_arrays_for_ids(
            split_ids[split],
            summary_features_by_id,
            summary_targets_by_id,
        )
        summary_arrays[split] = X_summary
        targets_from_summary[split] = y_summary

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

    embedding_arrays = {}
    targets_from_embeddings = {}
    embedding_ids = {}
    for split in ["train", "val", "test"]:
        embeddings, targets, ids = extract_split_embeddings(
            model=model,
            data=data,
            split_ids=split_ids[split],
            batch_size=batch_size,
            device=device,
        )
        check_same_order(split_ids[split], ids, split)
        embedding_arrays[split] = embeddings
        targets_from_embeddings[split] = targets
        embedding_ids[split] = ids

        if not np.allclose(targets_from_summary[split], targets):
            raise ValueError(f"{split} targets differ between summary and embedding loaders.")

    combined_arrays = {
        split: np.concatenate(
            [
                summary_arrays[split].astype(np.float64),
                embedding_arrays[split].astype(np.float64),
            ],
            axis=1,
        )
        for split in ["train", "val", "test"]
    }

    y_train = targets_from_summary["train"]
    y_val = targets_from_summary["val"]
    y_test = targets_from_summary["test"]

    feature_sets = {
        "summary_only": summary_arrays,
        "embedding_only": embedding_arrays,
        "summary_plus_embedding": combined_arrays,
    }

    ridge_results = {}
    for feature_set, arrays in feature_sets.items():
        ridge_results[feature_set] = fit_and_evaluate_ridge(
            X_train=arrays["train"],
            y_train=y_train,
            X_val=arrays["val"],
            y_val=y_val,
            X_test=arrays["test"],
            y_test=y_test,
            alpha=float(args.ridge_alpha),
        )

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "ridge_alpha": float(args.ridge_alpha),
        "split_source": str(config_path),
        "split_sizes": {
            split: len(split_ids[split])
            for split in ["train", "val", "test"]
        },
        "feature_dimensions": {
            "summary": int(summary_arrays["train"].shape[1]),
            "embedding": int(embedding_arrays["train"].shape[1]),
            "combined": int(combined_arrays["train"].shape[1]),
        },
        "target_summary": {
            split: {
                "mean": float(np.mean(targets_from_summary[split])),
                "std": float(np.std(targets_from_summary[split])),
                "min": float(np.min(targets_from_summary[split])),
                "max": float(np.max(targets_from_summary[split])),
            }
            for split in ["train", "val", "test"]
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
        "ridge_results": ridge_results,
        "train_ids": split_ids["train"],
        "val_ids": split_ids["val"],
        "test_ids": split_ids["test"],
        "interpretation": {
            "summary_plus_embedding_beats_summary": (
                "If summary_plus_embedding clearly exceeds summary_only on val/test, "
                "the GNN embedding adds complementary signal."
            ),
            "summary_plus_embedding_matches_summary": (
                "If summary_plus_embedding is approximately equal to summary_only, "
                "the embedding adds little or no useful signal beyond summaries."
            ),
            "embedding_only_weak": (
                "If embedding_only is weak, the current GNN representation is not "
                "enough alone under a linear readout."
            ),
        },
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "summary_vs_embedding_combined.json"
    )
    save_json(output, output_path)

    print_summary(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
