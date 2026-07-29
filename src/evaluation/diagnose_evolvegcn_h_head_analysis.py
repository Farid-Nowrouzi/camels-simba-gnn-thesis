from __future__ import annotations

"""
diagnose_evolvegcn_h_head_analysis.py

Pure diagnostic for analyzing whether the trained EvolveGCN-H regression head
is discarding target signal already present in temporal embeddings.

This script does not train, change preprocessing, alter datasets, modify
splits, edit checkpoints, or change model behavior.
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
def extract_temporal_and_regressor_input(
    model,
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

    graph_embeddings = model.masked_graph_pool(
        X_seq=H_seq,
        mask_seq=mask_seq,
        mode=model.graph_pooling,
    )
    temporal_embeddings = model.temporal_pool(graph_embeddings)
    regressor_inputs = temporal_embeddings

    if model.summary_feature_dim > 0:
        if summary_features is None:
            raise ValueError(
                "Checkpoint expects summary features, but loader returned None."
            )
        regressor_inputs = torch.cat(
            [regressor_inputs, summary_features.float()],
            dim=-1,
        )

    return {
        "temporal_embeddings": temporal_embeddings,
        "regressor_inputs": regressor_inputs,
    }


@torch.no_grad()
def run_head_with_intermediates(model, regressor_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    if len(model.regressor) != 4:
        raise ValueError(
            "Expected regressor architecture: Linear, ReLU, Dropout, Linear. "
            f"Got {model.regressor}"
        )

    first_linear_pre = model.regressor[0](regressor_inputs)
    hidden_after_activation = model.regressor[1](first_linear_pre)
    hidden_before_final = model.regressor[2](hidden_after_activation)
    final_linear_output = model.regressor[3](hidden_before_final)

    return {
        "regressor_inputs": regressor_inputs,
        "first_linear_pre_activation": first_linear_pre,
        "hidden_after_activation": hidden_after_activation,
        "hidden_before_final_linear": hidden_before_final,
        "final_linear_output": final_linear_output,
    }


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
        "mae": mae,
        "rmse": rmse,
        "pearson_correlation": pearson_corr(predictions, targets),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def fit_linear_probe(embeddings: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    X_design = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)

    coefficients, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    predictions = X_design @ coefficients

    return {
        "metrics": regression_metrics(predictions, y),
        "coefficient_norm": float(np.linalg.norm(coefficients[:-1])),
        "intercept": float(coefficients[-1]),
    }


def tensor_stats(values: torch.Tensor) -> Dict[str, float | List[int]]:
    values = values.detach().cpu().float()
    return {
        "shape": list(values.shape),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "variance": float(values.var(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def effective_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> float:
    singular_values = singular_values.detach().cpu().float()
    total = singular_values.sum()

    if float(total.item()) <= eps:
        return 0.0

    probs = singular_values / total
    probs = probs[probs > eps]
    entropy = -(probs * torch.log(probs)).sum()
    return float(torch.exp(entropy).item())


def linear_layer_report(layer: torch.nn.Linear, name: str) -> Dict[str, Any]:
    weight = layer.weight.detach().cpu().float()
    bias = layer.bias.detach().cpu().float() if layer.bias is not None else None
    singular_values = torch.linalg.svdvals(weight)

    report: Dict[str, Any] = {
        "name": name,
        "weight_shape": list(weight.shape),
        "weight_mean": float(weight.mean().item()),
        "weight_std": float(weight.std(unbiased=False).item()),
        "weight_min": float(weight.min().item()),
        "weight_max": float(weight.max().item()),
        "percent_near_zero_weights_abs_lt_1e-6": float(
            (weight.abs() < 1e-6).float().mean().item() * 100.0
        ),
        "percent_near_zero_weights_abs_lt_1e-5": float(
            (weight.abs() < 1e-5).float().mean().item() * 100.0
        ),
        "percent_near_zero_weights_abs_lt_1e-4": float(
            (weight.abs() < 1e-4).float().mean().item() * 100.0
        ),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "max_singular_value": float(singular_values.max().item()),
        "min_singular_value": float(singular_values.min().item()),
        "effective_rank": effective_rank(singular_values),
    }

    if bias is not None:
        report.update(
            {
                "bias_mean": float(bias.mean().item()),
                "bias_std": float(bias.std(unbiased=False).item()),
                "bias_min": float(bias.min().item()),
                "bias_max": float(bias.max().item()),
            }
        )
    else:
        report.update(
            {
                "bias_mean": None,
                "bias_std": None,
                "bias_min": None,
                "bias_max": None,
            }
        )

    return report


def hidden_unit_correlations(hidden: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    correlations = []

    for dim in range(hidden.shape[1]):
        correlations.append(pearson_corr(hidden[:, dim], targets))

    corr_array = np.asarray(correlations, dtype=np.float64)
    abs_corr = np.abs(corr_array)
    ranking_scores = np.nan_to_num(abs_corr, nan=0.0)
    top_indices = np.argsort(-ranking_scores)[:10]

    return {
        "max_absolute_correlation": float(np.nanmax(abs_corr)),
        "mean_absolute_correlation": float(np.nanmean(abs_corr)),
        "median_absolute_correlation": float(np.nanmedian(abs_corr)),
        "top_10_hidden_units_by_absolute_correlation": [
            {
                "unit": int(index),
                "correlation": float(corr_array[index]),
                "absolute_correlation": float(abs_corr[index])
                if not np.isnan(abs_corr[index])
                else float("nan"),
            }
            for index in top_indices
        ],
        "per_hidden_unit_correlation": [
            float(value) if not np.isnan(value) else float("nan")
            for value in corr_array
        ],
    }


def activation_report(
    name: str,
    pre_values: torch.Tensor,
    post_values: torch.Tensor,
    targets: np.ndarray,
) -> Dict[str, Any]:
    pre_cpu = pre_values.detach().cpu().float()
    post_cpu = post_values.detach().cpu().float()
    post_np = post_cpu.numpy()

    entirely_zero = (post_cpu == 0).all(dim=0)

    return {
        "name": name,
        "output_before_activation": tensor_stats(pre_cpu),
        "output_after_activation": tensor_stats(post_cpu),
        "percent_dead_activations": float((post_cpu == 0).float().mean().item() * 100.0),
        "num_units_entirely_zero": int(entirely_zero.sum().item()),
        "percent_units_entirely_zero": float(entirely_zero.float().mean().item() * 100.0),
        "hidden_unit_target_correlations": hidden_unit_correlations(post_np, targets),
    }


def final_output_report(values: torch.Tensor, targets: np.ndarray) -> Dict[str, Any]:
    values_cpu = values.detach().cpu().float()
    predictions = values_cpu.view(-1).numpy()
    return {
        "output": tensor_stats(values_cpu),
        "metrics": regression_metrics(predictions, targets),
        "correlation_with_target": pearson_corr(predictions, targets),
    }


def get_summary_scaler_from_config(config: Dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mean = config.get("summary_feature_mean")
    std = config.get("summary_feature_std")

    if mean is None and std is None:
        return None, None

    if mean is None or std is None:
        raise ValueError("Config has only one of summary_feature_mean/summary_feature_std.")

    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def print_head_summary(output: Dict[str, Any]) -> None:
    comparison = output["comparison"]
    first = output["head_analysis"]["first_linear"]
    final_metrics = output["head_analysis"]["final_linear"]["metrics"]

    print("=" * 100)
    print("EVOLVEGCN-H REGRESSION HEAD ANALYSIS")
    print("=" * 100)
    print(f"Experiment dir: {output['experiment_dir']}")
    print(f"Split:          {output['split']}")
    print(f"Samples:        {output['num_samples']}")
    print()
    print("Embedding -> Linear Probe vs Embedding -> Trained Head")
    print("-" * 100)
    print(
        f"{'Model':<28}"
        f"{'R2':>12}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'Pearson':>12}"
        f"{'Pred Std':>12}"
    )
    print("-" * 100)

    for name, metrics in [
        ("linear_probe_temporal", comparison["linear_probe_temporal"]["metrics"]),
        ("trained_head", final_metrics),
    ]:
        print(
            f"{name:<28}"
            f"{metrics['r2']:>12.6f}"
            f"{metrics['mae']:>12.6f}"
            f"{metrics['rmse']:>12.6f}"
            f"{metrics['pearson_correlation']:>12.6f}"
            f"{metrics['prediction_std']:>12.6f}"
        )

    if "linear_probe_regressor_input" in comparison:
        metrics = comparison["linear_probe_regressor_input"]["metrics"]
        print(
            f"{'linear_probe_reg_input':<28}"
            f"{metrics['r2']:>12.6f}"
            f"{metrics['mae']:>12.6f}"
            f"{metrics['rmse']:>12.6f}"
            f"{metrics['pearson_correlation']:>12.6f}"
            f"{metrics['prediction_std']:>12.6f}"
        )

    print()
    print("First Linear / Hidden Activation")
    print("-" * 100)
    print(f"weight std:                         {first['weights']['weight_std']:.8f}")
    print(f"effective rank:                     {first['weights']['effective_rank']:.4f}")
    print(
        "pre-activation variance:            "
        f"{first['activations']['output_before_activation']['variance']:.8f}"
    )
    print(
        "post-activation variance:           "
        f"{first['activations']['output_after_activation']['variance']:.8f}"
    )
    print(
        "percent dead activations:           "
        f"{first['activations']['percent_dead_activations']:.2f}%"
    )
    print(
        "percent units entirely zero:        "
        f"{first['activations']['percent_units_entirely_zero']:.2f}%"
    )
    print(
        "max abs hidden-target corr:         "
        f"{first['activations']['hidden_unit_target_correlations']['max_absolute_correlation']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze trained EvolveGCN-H regression head."
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
    config_path = Path(args.config_path) if args.config_path else experiment_dir / "config.json"
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else experiment_dir / "checkpoints" / "best_model.pt"
    )

    config = load_json(config_path)
    dataset_path = Path(args.dataset_path if args.dataset_path else config["dataset_path"])
    device = choose_device(args.device)

    data = load_temporal_dataset(dataset_path)
    split_key = f"{args.split}_ids"
    if split_key not in config:
        raise KeyError(f"Config does not contain split key: {split_key}")

    split_ids = list(config[split_key])
    validate_split_ids(split_ids=split_ids, dataset_ids=list(data.keys()), split_name=args.split)

    use_summary_features = bool(config.get("use_summary_features", False))
    summary_mean, summary_std = get_summary_scaler_from_config(config)
    batch_size = int(args.batch_size or config.get("batch_size", 4))

    dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=split_ids,
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_mean,
        summary_feature_std=summary_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    first_batch = next(iter(loader))
    _, _, X_example, _, _, _ = unpack_batch(first_batch)
    node_features = int(X_example.shape[-1])

    model = build_model_from_config(config=config, node_features=node_features, device=device)
    checkpoint = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    temporal_embeddings_all = []
    regressor_inputs_all = []
    head_values = {
        "first_linear_pre_activation": [],
        "hidden_after_activation": [],
        "hidden_before_final_linear": [],
        "final_linear_output": [],
    }
    targets_all = []
    universe_ids_all = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)
        target = target.to(device)

        if summary_features is not None:
            summary_features = summary_features.to(device)

        reps = extract_temporal_and_regressor_input(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )
        head = run_head_with_intermediates(model, reps["regressor_inputs"])

        temporal_embeddings_all.append(reps["temporal_embeddings"].detach().cpu())
        regressor_inputs_all.append(reps["regressor_inputs"].detach().cpu())

        for name in head_values:
            head_values[name].append(head[name].detach().cpu())

        targets_all.append(target.detach().cpu().view(-1))
        universe_ids_all.extend(universe_ids)

    temporal_embeddings = torch.cat(temporal_embeddings_all, dim=0).float().numpy()
    regressor_inputs = torch.cat(regressor_inputs_all, dim=0).float().numpy()
    targets = torch.cat(targets_all, dim=0).float().numpy()
    first_pre = torch.cat(head_values["first_linear_pre_activation"], dim=0)
    hidden_after_activation = torch.cat(head_values["hidden_after_activation"], dim=0)
    hidden_before_final = torch.cat(head_values["hidden_before_final_linear"], dim=0)
    final_output = torch.cat(head_values["final_linear_output"], dim=0)

    comparison = {
        "linear_probe_temporal": fit_linear_probe(temporal_embeddings, targets),
        "trained_head": regression_metrics(final_output.view(-1).numpy(), targets),
    }
    if regressor_inputs.shape[1] != temporal_embeddings.shape[1]:
        comparison["linear_probe_regressor_input"] = fit_linear_probe(regressor_inputs, targets)

    first_linear = model.regressor[0]
    final_linear = model.regressor[3]

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
            "activation": str(config.get("activation", "relu")),
            "temporal_pooling": str(config["temporal_pooling"]),
            "graph_pooling": str(config["graph_pooling"]),
            "summary_feature_dim": int(config.get("summary_feature_dim", 0)),
            "use_summary_features": use_summary_features,
        },
        "target": {
            "mean": float(np.mean(targets)),
            "std": float(np.std(targets)),
            "min": float(np.min(targets)),
            "max": float(np.max(targets)),
        },
        "comparison": comparison,
        "head_analysis": {
            "first_linear": {
                "weights": linear_layer_report(first_linear, "first_linear"),
                "activations": activation_report(
                    name="first_linear",
                    pre_values=first_pre,
                    post_values=hidden_after_activation,
                    targets=targets,
                ),
            },
            "dropout_eval_output_before_final_linear": tensor_stats(hidden_before_final),
            "final_linear": {
                "weights": linear_layer_report(final_linear, "final_linear"),
                "output": final_output_report(final_output, targets),
                "metrics": regression_metrics(final_output.view(-1).numpy(), targets),
            },
        },
    }

    print_head_summary(output)

    output_path = (
        Path(args.output_path)
        if args.output_path
        else experiment_dir / "diagnostics" / f"{args.split}_head_analysis.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
