from __future__ import annotations

"""
diagnose_head_vs_optimal_linear_solution.py

Pure diagnostic comparing a trained EvolveGCN-H regression head against the
ordinary least squares solution on the exact same head inputs.

This script does not train, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads a trained
experiment, reuses saved split IDs, extracts temporal embeddings, optionally
concatenates summary features if the trained head expects them, and compares:

    1. trained regression head predictions
    2. OLS fit on TRAIN head inputs only
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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


def summary_scaler_from_config(
    config: Dict[str, Any],
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    mean = config.get("summary_feature_mean")
    std = config.get("summary_feature_std")

    if mean is None and std is None:
        return None, None

    if mean is None or std is None:
        raise ValueError(
            "Config contains only one of summary_feature_mean/summary_feature_std."
        )

    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def target_scaler_from_config(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    normalize_target = bool(config.get("normalize_target", False))

    if not normalize_target:
        return False, 0.0, 1.0

    if "target_mean" not in config or "target_std" not in config:
        raise KeyError(
            "Config has normalize_target=True but is missing target_mean/target_std."
        )

    return True, float(config["target_mean"]), float(config["target_std"])


def maybe_denormalize_predictions(
    predictions: np.ndarray,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    if not normalize_target:
        return predictions

    return predictions * target_std + target_mean


def extract_split_inputs_and_predictions(
    model,
    data: Dict[str, Dict[str, Any]],
    split_ids: List[str],
    batch_size: int,
    device: torch.device,
    use_summary_features: bool,
    summary_feature_mean: torch.Tensor | None,
    summary_feature_std: torch.Tensor | None,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids,
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_feature_mean,
        summary_feature_std=summary_feature_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    temporal_embedding_batches = []
    head_input_batches = []
    target_batches = []
    prediction_batches = []
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        temporal_embeddings = extract_temporal_embeddings(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )
        head_inputs = temporal_embeddings

        if model.summary_feature_dim > 0:
            if summary_features is None:
                raise ValueError(
                    "Model expects summary features, but dataset did not return them."
                )
            head_inputs = torch.cat(
                [head_inputs, summary_features.to(device).float()],
                dim=-1,
            )

        predictions = model.regressor(head_inputs)
        predictions_np = predictions.detach().cpu().float().view(-1).numpy()
        predictions_np = maybe_denormalize_predictions(
            predictions=predictions_np,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
        )

        temporal_embedding_batches.append(
            temporal_embeddings.detach().cpu().float().numpy()
        )
        head_input_batches.append(head_inputs.detach().cpu().float().numpy())
        target_batches.append(target.detach().cpu().float().view(-1).numpy())
        prediction_batches.append(predictions_np)
        universe_ids_all.extend(universe_ids)

    return (
        np.concatenate(temporal_embedding_batches, axis=0),
        np.concatenate(head_input_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        np.concatenate(prediction_batches, axis=0),
        universe_ids_all,
    )


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson_corr(predictions, targets),
    }


def fit_ols(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [
            features.astype(np.float64),
            np.ones((features.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(design, targets.astype(np.float64), rcond=None)
    return coefficients


def predict_ols(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        [
            features.astype(np.float64),
            np.ones((features.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    return design @ coefficients


def get_direct_linear_head_parameters(
    regressor: nn.Module,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
) -> Tuple[np.ndarray | None, float | None, str]:
    if not isinstance(regressor, nn.Linear):
        return None, None, "trained head is not a direct nn.Linear"

    if regressor.out_features != 1:
        return None, None, "trained linear head does not have one output"

    weight = regressor.weight.detach().cpu().float().view(-1).numpy().astype(np.float64)
    bias = float(regressor.bias.detach().cpu().float().view(-1)[0].item())

    if normalize_target:
        weight = weight * target_std
        bias = bias * target_std + target_mean

    return weight, bias, "ok"


def weight_comparison(
    model,
    ols_coefficients: np.ndarray,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
) -> Dict[str, Any]:
    trained_weight, trained_bias, status = get_direct_linear_head_parameters(
        regressor=model.regressor,
        normalize_target=normalize_target,
        target_mean=target_mean,
        target_std=target_std,
    )

    ols_weight = ols_coefficients[:-1]
    ols_bias = float(ols_coefficients[-1])

    if trained_weight is None or trained_bias is None:
        return {
            "available": False,
            "reason": status,
            "ols_weight_norm": float(np.linalg.norm(ols_weight)),
            "ols_bias": ols_bias,
        }

    if trained_weight.shape != ols_weight.shape:
        return {
            "available": False,
            "reason": (
                "trained head weight and OLS weight have different shapes: "
                f"{trained_weight.shape} vs {ols_weight.shape}"
            ),
            "trained_weight_norm": float(np.linalg.norm(trained_weight)),
            "ols_weight_norm": float(np.linalg.norm(ols_weight)),
            "trained_bias": trained_bias,
            "ols_bias": ols_bias,
        }

    trained_norm = float(np.linalg.norm(trained_weight))
    ols_norm = float(np.linalg.norm(ols_weight))

    if trained_norm == 0.0 or ols_norm == 0.0:
        cosine_similarity = float("nan")
    else:
        cosine_similarity = float(
            np.dot(trained_weight, ols_weight) / (trained_norm * ols_norm)
        )

    return {
        "available": True,
        "cosine_similarity": cosine_similarity,
        "trained_weight_norm": trained_norm,
        "ols_weight_norm": ols_norm,
        "norm_ratio_trained_over_ols": (
            float("nan") if ols_norm == 0.0 else trained_norm / ols_norm
        ),
        "trained_bias": trained_bias,
        "ols_bias": ols_bias,
        "bias_difference_trained_minus_ols": trained_bias - ols_bias,
    }


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("TRAINED HEAD VS OPTIMAL LINEAR SOLUTION")
    print("=" * 112)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Checkpoint:     {output['checkpoint_path']}")
    print(f"Dataset:        {output['dataset_path']}")
    print(f"Head input:     {output['head_input_type']}")

    print()
    print("Performance on the same extracted head inputs")
    print("-" * 112)
    print(
        f"{'Split':<12}"
        f"{'Model':<16}"
        f"{'R2':>12}"
        f"{'RMSE':>14}"
        f"{'MAE':>14}"
        f"{'Pearson':>14}"
    )
    print("-" * 112)

    for split in ["train", "val", "test"]:
        for model_name, metrics in [
            ("trained_head", output["splits"][split]["trained_head"]),
            ("ols_train_fit", output["splits"][split]["ols_train_fit"]),
        ]:
            print(
                f"{split:<12}"
                f"{model_name:<16}"
                f"{metrics['r2']:>12.6f}"
                f"{metrics['rmse']:>14.6f}"
                f"{metrics['mae']:>14.6f}"
                f"{metrics['pearson']:>14.6f}"
            )

    print()
    print("Trained direct-linear head vs OLS geometry")
    print("-" * 112)
    comparison = output["weight_comparison"]
    if comparison["available"]:
        print(f"Cosine similarity:      {comparison['cosine_similarity']:.6f}")
        print(f"Norm ratio trained/OLS: {comparison['norm_ratio_trained_over_ols']:.6f}")
        print(f"Bias difference:        {comparison['bias_difference_trained_minus_ols']:.6e}")
    else:
        print(f"Unavailable: {comparison['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trained EvolveGCN-H head to OLS on the same inputs."
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
    summary_feature_dim = int(config.get("summary_feature_dim", 0))
    use_summary_features = summary_feature_dim > 0
    summary_feature_mean, summary_feature_std = summary_scaler_from_config(config)
    normalize_target, target_mean, target_std = target_scaler_from_config(config)

    first_dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids["train"],
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_feature_mean,
        summary_feature_std=summary_feature_std,
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

    temporal_embeddings_by_split = {}
    head_inputs_by_split = {}
    targets_by_split = {}
    trained_predictions_by_split = {}
    ids_by_split = {}

    for split in ["train", "val", "test"]:
        (
            temporal_embeddings,
            head_inputs,
            targets,
            trained_predictions,
            ids,
        ) = extract_split_inputs_and_predictions(
            model=model,
            data=data,
            split_ids=split_ids[split],
            batch_size=batch_size,
            device=device,
            use_summary_features=use_summary_features,
            summary_feature_mean=summary_feature_mean,
            summary_feature_std=summary_feature_std,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
        )
        temporal_embeddings_by_split[split] = temporal_embeddings
        head_inputs_by_split[split] = head_inputs
        targets_by_split[split] = targets
        trained_predictions_by_split[split] = trained_predictions
        ids_by_split[split] = ids

    ols_coefficients = fit_ols(
        features=head_inputs_by_split["train"],
        targets=targets_by_split["train"],
    )

    split_results = {}
    for split in ["train", "val", "test"]:
        ols_predictions = predict_ols(
            features=head_inputs_by_split[split],
            coefficients=ols_coefficients,
        )
        split_results[split] = {
            "num_samples": int(len(targets_by_split[split])),
            "temporal_embedding_shape": list(temporal_embeddings_by_split[split].shape),
            "head_input_shape": list(head_inputs_by_split[split].shape),
            "target_mean": float(np.mean(targets_by_split[split])),
            "target_std": float(np.std(targets_by_split[split])),
            "trained_head": regression_metrics(
                trained_predictions_by_split[split],
                targets_by_split[split],
            ),
            "ols_train_fit": regression_metrics(
                ols_predictions,
                targets_by_split[split],
            ),
            "universe_ids": ids_by_split[split],
        }

    head_input_type = (
        "temporal_embedding_plus_summary_features"
        if use_summary_features
        else "temporal_embedding"
    )

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "head_input_type": head_input_type,
        "target_normalization": {
            "normalize_target": normalize_target,
            "target_mean": target_mean if normalize_target else None,
            "target_std": target_std if normalize_target else None,
        },
        "model": {
            "hidden_dim": int(config["hidden_dim"]),
            "num_layers": int(config["num_layers"]),
            "dropout": float(config["dropout"]),
            "activation": str(config.get("activation", "relu")),
            "temporal_pooling": str(config["temporal_pooling"]),
            "graph_pooling": str(config["graph_pooling"]),
            "head_type": str(config.get("head_type", "mlp")),
            "summary_feature_dim": summary_feature_dim,
            "use_summary_features": bool(config.get("use_summary_features", False)),
        },
        "splits": split_results,
        "ols": {
            "fit_split": "train",
            "coefficient_shape": list(ols_coefficients.shape),
            "weight_norm": float(np.linalg.norm(ols_coefficients[:-1])),
            "bias": float(ols_coefficients[-1]),
        },
        "weight_comparison": weight_comparison(
            model=model,
            ols_coefficients=ols_coefficients,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
        ),
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "head_vs_optimal_linear_solution.json"
    )
    save_json(output, output_path)

    print_summary(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
