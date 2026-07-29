from __future__ import annotations

"""
diagnose_embedding_distribution_shift.py

Pure diagnostic for testing whether EvolveGCN-H temporal embeddings from
train/validation/test splits come from different distributions.

This script does not train, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads a trained
experiment, reuses saved split IDs, extracts temporal embeddings, and reports
split distribution statistics, MMD, and train-vs-test classifier separability.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def check_split_overlap(split_ids: Dict[str, List[str]]) -> None:
    split_names = list(split_ids)

    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            if overlap:
                raise ValueError(
                    f"{left}/{right} splits overlap: {overlap[:20]}"
                )


def covariance_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [N, D], got {embeddings.shape}")

    num_samples, width = embeddings.shape

    if num_samples < 2:
        return {
            "num_samples": int(num_samples),
            "embedding_dim": int(width),
            "mean_diagonal_variance": float("nan"),
            "trace": float("nan"),
            "frobenius_norm": float("nan"),
            "min_eigenvalue": float("nan"),
            "max_eigenvalue": float("nan"),
            "effective_rank": float("nan"),
        }

    covariance = np.cov(embeddings.astype(np.float64), rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))

    if total > 0.0:
        probabilities = eigenvalues / total
        probabilities = probabilities[probabilities > 0.0]
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0

    return {
        "num_samples": int(num_samples),
        "embedding_dim": int(width),
        "mean_diagonal_variance": float(np.mean(np.diag(covariance))),
        "trace": float(np.trace(covariance)),
        "frobenius_norm": float(np.linalg.norm(covariance, ord="fro")),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "effective_rank": effective_rank,
    }


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def median_heuristic_sigma(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.concatenate([a, b], axis=0).astype(np.float64)

    if len(combined) < 2:
        return 1.0

    distances = np.sqrt(squared_distances(combined, combined))
    upper = distances[np.triu_indices(len(combined), k=1)]
    positive = upper[upper > 0.0]

    if len(positive) == 0:
        return 1.0

    return float(np.median(positive))


def rbf_mmd(a: np.ndarray, b: np.ndarray, sigma: float | None = None) -> Dict[str, float]:
    if sigma is None:
        sigma = median_heuristic_sigma(a, b)

    sigma = max(float(sigma), 1e-12)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * squared_distances(a, a))
    k_yy = np.exp(-gamma * squared_distances(b, b))
    k_xy = np.exp(-gamma * squared_distances(a, b))
    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))

    return {
        "mmd2_biased": max(mmd2, 0.0),
        "rbf_sigma": sigma,
    }


def split_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    return float(np.linalg.norm(mean_a - mean_b, ord=2))


def classifier_separability(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    random_state: int,
) -> Dict[str, Any]:
    features = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(train_embeddings), dtype=np.int64),
            np.ones(len(test_embeddings), dtype=np.int64),
        ],
        axis=0,
    )

    class_counts = np.bincount(labels)
    min_class_count = int(np.min(class_counts))

    if min_class_count < 2:
        return {
            "method": "standardized_logistic_regression_cv",
            "num_samples": int(len(labels)),
            "num_train_class": int(class_counts[0]),
            "num_test_class": int(class_counts[1]),
            "accuracy": float("nan"),
            "roc_auc": float("nan"),
            "n_splits": 0,
        }

    n_splits = min(5, min_class_count)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    return {
        "method": "standardized_logistic_regression_cv",
        "num_samples": int(len(labels)),
        "num_train_class": int(class_counts[0]),
        "num_test_class": int(class_counts[1]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "n_splits": int(n_splits),
    }


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("TEMPORAL EMBEDDING DISTRIBUTION SHIFT DIAGNOSTIC")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Checkpoint:     {output['checkpoint_path']}")
    print(f"Dataset:        {output['dataset_path']}")

    print()
    print("Split covariance statistics")
    print("-" * 112)
    print(
        f"{'Split':<10}"
        f"{'N':>6}"
        f"{'Dim':>6}"
        f"{'Mean Var':>14}"
        f"{'Trace':>14}"
        f"{'Fro Norm':>14}"
        f"{'Max Eig':>14}"
        f"{'Eff Rank':>14}"
    )
    print("-" * 112)
    for split in ["train", "val", "test"]:
        stats = output["splits"][split]["covariance"]
        print(
            f"{split:<10}"
            f"{stats['num_samples']:>6}"
            f"{stats['embedding_dim']:>6}"
            f"{stats['mean_diagonal_variance']:>14.6e}"
            f"{stats['trace']:>14.6e}"
            f"{stats['frobenius_norm']:>14.6e}"
            f"{stats['max_eigenvalue']:>14.6e}"
            f"{stats['effective_rank']:>14.6f}"
        )

    print()
    print("Split mean distances and MMD")
    print("-" * 112)
    print(
        f"{'Pair':<18}"
        f"{'Mean Distance':>18}"
        f"{'MMD^2':>18}"
        f"{'RBF Sigma':>18}"
    )
    print("-" * 112)
    for pair_name, stats in output["split_comparisons"].items():
        print(
            f"{pair_name:<18}"
            f"{stats['mean_distance']:>18.6e}"
            f"{stats['mmd']['mmd2_biased']:>18.6e}"
            f"{stats['mmd']['rbf_sigma']:>18.6e}"
        )

    classifier = output["train_vs_test_classifier"]
    print()
    print("Train-vs-test classifier")
    print("-" * 112)
    print(f"Method:    {classifier['method']}")
    print(f"CV folds:  {classifier['n_splits']}")
    print(f"Accuracy:  {classifier['accuracy']:.6f}")
    print(f"ROC-AUC:   {classifier['roc_auc']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose distribution shift among temporal embedding splits."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random_state", type=int, default=42)

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
    check_split_overlap(split_ids)

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

    split_summaries = {}
    for split in ["train", "val", "test"]:
        embeddings = embeddings_by_split[split]
        split_summaries[split] = {
            "num_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "mean_embedding_vector": np.mean(embeddings, axis=0).tolist(),
            "embedding_global_mean": float(np.mean(embeddings)),
            "embedding_global_std": float(np.std(embeddings)),
            "target_mean": float(np.mean(targets_by_split[split])),
            "target_std": float(np.std(targets_by_split[split])),
            "covariance": covariance_stats(embeddings),
            "universe_ids": ids_by_split[split],
        }

    pair_specs = [
        ("train_vs_val", "train", "val"),
        ("train_vs_test", "train", "test"),
        ("val_vs_test", "val", "test"),
    ]
    split_comparisons = {}
    for pair_name, left, right in pair_specs:
        left_embeddings = embeddings_by_split[left]
        right_embeddings = embeddings_by_split[right]
        split_comparisons[pair_name] = {
            "left_split": left,
            "right_split": right,
            "mean_distance": split_mean_distance(left_embeddings, right_embeddings),
            "mmd": rbf_mmd(left_embeddings, right_embeddings),
        }

    classifier_results = classifier_separability(
        train_embeddings=embeddings_by_split["train"],
        test_embeddings=embeddings_by_split["test"],
        random_state=int(args.random_state),
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
        "splits": split_summaries,
        "split_comparisons": split_comparisons,
        "train_vs_test_classifier": classifier_results,
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "embedding_distribution_shift.json"
    )
    save_json(output, output_path)

    print_summary(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
