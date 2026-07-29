from __future__ import annotations

"""
diagnose_evolvegcn_h_regressor_head.py

Pure diagnostic for locating prediction-variance collapse inside the final
EvolveGCN-H regression head.

This script does not train, change preprocessing, alter dataset construction,
modify model behavior, or change splits. It loads a trained experiment config
and checkpoint, reuses the saved split IDs, and reports variance through the
regressor head only.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.evaluation.diagnose_evolvegcn_h_representations import (
    build_model_from_config,
    choose_device,
    finite_stats,
    load_checkpoint_state,
    load_json,
    pairwise_variance_stats,
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
def extract_regressor_input(
    model,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
    summary_features: torch.Tensor | None,
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

    universe_embeddings = model.temporal_pool(graph_embeddings)
    regressor_input = universe_embeddings

    if model.summary_feature_dim > 0:
        if summary_features is None:
            raise ValueError(
                "Checkpoint expects summary features, but loader returned None."
            )

        regressor_input = torch.cat(
            [regressor_input, summary_features.float()],
            dim=-1,
        )

    return regressor_input


@torch.no_grad()
def run_regressor_head(model, regressor_input: torch.Tensor) -> Dict[str, torch.Tensor]:
    if len(model.regressor) != 4:
        raise ValueError(
            "Expected regressor architecture: Linear, ReLU, Dropout, Linear. "
            f"Got {model.regressor}"
        )

    first_linear = model.regressor[0](regressor_input)
    after_relu = model.regressor[1](first_linear)
    before_final_linear = model.regressor[2](after_relu)
    after_final_linear = model.regressor[3](before_final_linear)

    return {
        "regressor_inputs": regressor_input,
        "after_first_linear": first_linear,
        "after_relu": after_relu,
        "before_final_linear": before_final_linear,
        "after_final_linear": after_final_linear,
    }


def sample_matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    return tensor.reshape(tensor.shape[0], -1)


def regressor_stage_report(values: torch.Tensor) -> Dict[str, Any]:
    values_cpu = values.detach().cpu().float()
    report = finite_stats(values_cpu)
    report["variance"] = float(values_cpu.var(unbiased=False).item())
    report.update(pairwise_variance_stats(sample_matrix(values_cpu)))
    return report


def print_stage_report(name: str, report: Dict[str, Any]) -> None:
    print()
    print(name)
    print("-" * 80)
    print(f"shape:                                      {report['shape']}")
    print(f"num_values:                                 {report['num_values']}")
    print(f"mean:                                       {report['mean']:.8f}")
    print(f"std:                                        {report['std']:.8f}")
    print(f"variance:                                   {report['variance']:.8f}")
    print(f"min:                                        {report['min']:.8f}")
    print(f"max:                                        {report['max']:.8f}")
    print(
        "avg_feature_variance_across_samples:       "
        f"{report['avg_feature_variance_across_samples']:.8f}"
    )
    print(
        "avg_pairwise_squared_distance_per_feature: "
        f"{report['avg_pairwise_squared_distance_per_feature']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose variance collapse inside EvolveGCN-H regressor head."
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

    stage_values = {
        "regressor_inputs": [],
        "after_first_linear": [],
        "after_relu": [],
        "before_final_linear": [],
        "after_final_linear": [],
    }
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

        regressor_input = extract_regressor_input(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )

        batch_stages = run_regressor_head(
            model=model,
            regressor_input=regressor_input,
        )

        for name, values in batch_stages.items():
            stage_values[name].append(values.detach().cpu())

        targets.append(target.detach().cpu())
        universe_ids_all.extend(universe_ids)

    stage_tensors = {
        name: torch.cat(values, dim=0)
        for name, values in stage_values.items()
    }
    target_tensor = torch.cat(targets, dim=0).view(-1)
    prediction_tensor = stage_tensors["after_final_linear"].view(-1)

    reports = {
        name: regressor_stage_report(values)
        for name, values in stage_tensors.items()
    }
    reports["predictions"] = finite_stats(prediction_tensor)
    reports["targets"] = finite_stats(target_tensor)

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
    print("EVOLVEGCN-H REGRESSOR HEAD DIAGNOSTIC")
    print("=" * 80)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Device:         {device}")

    for name in [
        "regressor_inputs",
        "after_first_linear",
        "after_relu",
        "before_final_linear",
        "after_final_linear",
    ]:
        print_stage_report(name, reports[name])

    prediction_report = reports["predictions"]
    print()
    print("prediction summary")
    print("-" * 80)
    print(f"mean: {prediction_report['mean']:.8f}")
    print(f"std:  {prediction_report['std']:.8f}")
    print(f"min:  {prediction_report['min']:.8f}")
    print(f"max:  {prediction_report['max']:.8f}")

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_regressor_head_stats.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
