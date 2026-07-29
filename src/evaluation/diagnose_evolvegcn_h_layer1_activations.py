from __future__ import annotations

"""
diagnose_evolvegcn_h_layer1_activations.py

Pure diagnostic for inspecting why the first EvolveGCN-H layer collapses
variance.

This script does not train, change preprocessing, alter dataset construction,
modify model architecture, edit checkpoints, change splits, or alter model
behavior. It loads a trained experiment config/checkpoint, reuses saved split
IDs, and reports Layer 1 pre/post activation variance, ReLU sparsity, dead
hidden dimensions, and node cosine similarity before/after Layer 1.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.diagnose_evolvegcn_h_representations import (
    build_model_from_config,
    choose_device,
    load_checkpoint_state,
    load_json,
    save_json,
    validate_split_ids,
)
from src.models.evolvegcn_h import masked_mean_pool, normalize_dense_adjacency
from src.training.train_evolvegcn_h import (
    CamelsTemporalDataset,
    collate_fn,
    load_temporal_dataset,
    unpack_batch,
)


@torch.no_grad()
def run_first_layer_with_activations(
    model,
    A_seq: torch.Tensor,
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if len(model.layers) < 1:
        raise ValueError("Model has no EvolveGCN-H layers.")

    layer = model.layers[0]
    batch_size, num_timesteps, _, _ = A_seq.shape

    current_weight = layer.initial_weight.reshape(1, -1).expand(
        batch_size,
        -1,
    )

    pre_activations = []
    post_activations = []

    for timestep in range(num_timesteps):
        A_t = A_seq[:, timestep, :, :]
        X_t = X_seq[:, timestep, :, :]
        mask_t = mask_seq[:, timestep, :, :].float()

        graph_summary = masked_mean_pool(
            X=X_t,
            mask=mask_t,
            dim=1,
        )

        current_weight = layer.weight_evolver(
            graph_summary,
            current_weight,
        )

        W_t = current_weight.reshape(
            batch_size,
            layer.in_features,
            layer.out_features,
        )

        A_norm_t = normalize_dense_adjacency(
            A=A_t,
            add_self_loops=layer.add_self_loops,
        )

        support = torch.bmm(X_t, W_t)
        pre_t = torch.bmm(A_norm_t, support) + layer.bias

        if layer.activation:
            post_t = F.relu(pre_t)
        else:
            post_t = pre_t

        post_t = layer.dropout(post_t)
        post_t = post_t * mask_t

        pre_activations.append(pre_t)
        post_activations.append(post_t)

    return {
        "pre_activations": torch.stack(pre_activations, dim=1),
        "post_activations": torch.stack(post_activations, dim=1),
    }


def valid_values(tensor: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    mask = mask_seq.bool().expand_as(tensor)
    return tensor[mask].detach().cpu().float()


def activation_stats(
    pre_activations: torch.Tensor,
    post_activations: torch.Tensor,
    mask_seq: torch.Tensor,
) -> Dict[str, Any]:
    pre_values = valid_values(pre_activations, mask_seq)
    post_values = valid_values(post_activations, mask_seq)

    valid_post = post_activations[mask_seq.bool().expand_as(post_activations)]
    zero_fraction = (valid_post == 0).float().mean()

    valid_hidden = post_activations[mask_seq.squeeze(-1).bool()]
    dead_neurons = (valid_hidden == 0).all(dim=0)

    return {
        "pre_activation_mean": float(pre_values.mean().item()),
        "pre_activation_variance": float(pre_values.var(unbiased=False).item()),
        "pre_activation_std": float(pre_values.std(unbiased=False).item()),
        "pre_activation_min": float(pre_values.min().item()),
        "pre_activation_max": float(pre_values.max().item()),
        "post_activation_mean": float(post_values.mean().item()),
        "post_activation_variance": float(post_values.var(unbiased=False).item()),
        "post_activation_std": float(post_values.std(unbiased=False).item()),
        "post_activation_min": float(post_values.min().item()),
        "post_activation_max": float(post_values.max().item()),
        "percent_activations_equal_zero": float(zero_fraction.item() * 100.0),
        "num_hidden_dimensions": int(dead_neurons.numel()),
        "num_neurons_entirely_zero": int(dead_neurons.sum().item()),
        "percent_neurons_entirely_zero": float(
            dead_neurons.float().mean().item() * 100.0
        ),
        "dead_neuron_indices": [
            int(index)
            for index in torch.nonzero(dead_neurons, as_tuple=False).view(-1).tolist()
        ],
    }


def average_pairwise_cosine_for_matrix(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return float("nan")

    matrix = F.normalize(matrix.float(), p=2, dim=-1, eps=1e-12)
    cosine = matrix @ matrix.T
    num_nodes = cosine.shape[0]
    keep = ~torch.eye(num_nodes, dtype=torch.bool, device=cosine.device)

    return float(cosine[keep].mean().item())


def average_node_cosine_similarity(
    embeddings: torch.Tensor,
    mask_seq: torch.Tensor,
) -> float:
    embeddings = embeddings.detach().cpu().float()
    mask_seq = mask_seq.detach().cpu().bool()

    batch_size, num_timesteps, _, _ = embeddings.shape
    similarities: List[float] = []

    for batch_index in range(batch_size):
        for timestep in range(num_timesteps):
            valid_mask = mask_seq[batch_index, timestep, :, 0]
            valid_embeddings = embeddings[batch_index, timestep, valid_mask, :]

            if valid_embeddings.shape[0] < 2:
                continue

            similarities.append(
                average_pairwise_cosine_for_matrix(valid_embeddings)
            )

    if len(similarities) == 0:
        return float("nan")

    return float(sum(similarities) / len(similarities))


def print_report(report: Dict[str, Any]) -> None:
    print()
    print("Layer 1 Activation Diagnostic")
    print("-" * 80)
    print(f"pre_activation_variance:             {report['pre_activation_variance']:.8f}")
    print(f"pre_activation_std:                  {report['pre_activation_std']:.8f}")
    print(f"pre_activation_min:                  {report['pre_activation_min']:.8f}")
    print(f"pre_activation_max:                  {report['pre_activation_max']:.8f}")
    print(f"post_activation_variance:            {report['post_activation_variance']:.8f}")
    print(f"post_activation_std:                 {report['post_activation_std']:.8f}")
    print(f"post_activation_min:                 {report['post_activation_min']:.8f}")
    print(f"post_activation_max:                 {report['post_activation_max']:.8f}")
    print(
        "percent_activations_equal_zero:      "
        f"{report['percent_activations_equal_zero']:.2f}%"
    )
    print(
        "percent_neurons_entirely_zero:       "
        f"{report['percent_neurons_entirely_zero']:.2f}%"
    )
    print(f"num_neurons_entirely_zero:           {report['num_neurons_entirely_zero']}")
    print(
        "avg_node_cosine_before_layer1:       "
        f"{report['avg_node_cosine_before_layer1']:.8f}"
    )
    print(
        "avg_node_cosine_after_layer1:        "
        f"{report['avg_node_cosine_after_layer1']:.8f}"
    )
    print(
        "cosine_similarity_change:            "
        f"{report['cosine_similarity_change']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose first-layer EvolveGCN-H activations."
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

    batch_size = int(args.batch_size or config.get("batch_size", 4))

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

    pre_batches = []
    post_batches = []
    input_batches = []
    mask_batches = []
    universe_ids_all: List[str] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, _, _ = unpack_batch(batch)

        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)

        activations = run_first_layer_with_activations(
            model=model,
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
        )

        input_batches.append(X_seq.detach().cpu())
        pre_batches.append(activations["pre_activations"].detach().cpu())
        post_batches.append(activations["post_activations"].detach().cpu())
        mask_batches.append(mask_seq.detach().cpu())
        universe_ids_all.extend(universe_ids)

    input_tensor = torch.cat(input_batches, dim=0)
    pre_tensor = torch.cat(pre_batches, dim=0)
    post_tensor = torch.cat(post_batches, dim=0)
    mask_tensor = torch.cat(mask_batches, dim=0)

    report = activation_stats(
        pre_activations=pre_tensor,
        post_activations=post_tensor,
        mask_seq=mask_tensor,
    )
    before_cosine = average_node_cosine_similarity(
        embeddings=input_tensor,
        mask_seq=mask_tensor,
    )
    after_cosine = average_node_cosine_similarity(
        embeddings=post_tensor,
        mask_seq=mask_tensor,
    )
    report["avg_node_cosine_before_layer1"] = before_cosine
    report["avg_node_cosine_after_layer1"] = after_cosine
    report["cosine_similarity_change"] = after_cosine - before_cosine

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
            "use_summary_features": bool(config.get("use_summary_features", False)),
        },
        "layer1": report,
    }

    print("=" * 80)
    print("EVOLVEGCN-H LAYER 1 ACTIVATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Experiment dir: {experiment_dir}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Split:          {args.split}")
    print(f"Samples:        {len(universe_ids_all)}")
    print(f"Device:         {device}")

    print_report(report)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / f"{args.split}_layer1_activations.json"
    )
    save_json(output, output_path)

    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
