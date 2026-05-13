from __future__ import annotations

"""
train_static_gcn.py

Train a Static GCN regressor on CAMELS-SIMBA static graph datasets.

Purpose
-------
This script trains a graph neural network baseline where each universe is
represented by ONE static graph, usually from the final snapshot a = 1.0.

This model is used after the mean Omega_m baseline and before the temporal
EvolveGCN-H model.

Comparison ladder:
    1. Mean Omega_m baseline
    2. Static GCN baseline
    3. Temporal EvolveGCN-H model

Expected dataset format
-----------------------
The static graph dataset should be created by:

    src.data.build_static_graphs

Expected saved format:

    {
        "LH_0": {
            "A": tensor [N, N],
            "X": tensor [N, F],
            "mask": tensor [N, 1],
            "target": tensor scalar,
            "snapshot": {...metadata...},
            ...
        },
        ...
    }

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn

Official node features:
    [log10_Mvir, X, Y, Z, VX, VY, VZ]

Example command
---------------
python -m src.training.train_static_gcn \
  --dataset_path data/processed/static_100u_logmass_minmax_top100_periodic_knn/camels_100u_static_logmass_minmax_top100_periodic_knn.pt \
  --experiment_name static_gcn_100u_seed123 \
  --output_root experiments \
  --seed 123 \
  --batch_size 8 \
  --epochs 300 \
  --patience 40 \
  --learning_rate 0.001 \
  --weight_decay 0.00001 \
  --hidden_dim 64 \
  --num_layers 3 \
  --dropout 0.2 \
  --graph_pooling mean \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --device auto
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.static_gcn import StaticGCNRegressor, count_parameters


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Make experiments more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset wrapper
# ============================================================

class CamelsStaticGraphDataset(Dataset):
    """
    PyTorch Dataset wrapper for CAMELS-SIMBA static graph samples.

    Each item returned:
        universe_id: str
        A: Tensor [N, N]
        X: Tensor [N, F]
        mask: Tensor [N, 1]
        target: Tensor [1]
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        universe_ids: List[str],
    ) -> None:
        self.data_dict = data_dict
        self.universe_ids = universe_ids

    def __len__(self) -> int:
        return len(self.universe_ids)

    def __getitem__(self, index: int):
        universe_id = self.universe_ids[index]
        sample = self.data_dict[universe_id]

        A = sample["A"].float()
        X = sample["X"].float()
        mask = sample["mask"].float()

        target = sample["target"]

        if not torch.is_tensor(target):
            target = torch.tensor(float(target), dtype=torch.float32)

        target = target.float().view(1)

        return universe_id, A, X, mask, target


def collate_fn(batch):
    """
    Custom collate function because universe_id is a string.
    """
    universe_ids = [item[0] for item in batch]

    A = torch.stack([item[1] for item in batch], dim=0)
    X = torch.stack([item[2] for item in batch], dim=0)
    mask = torch.stack([item[3] for item in batch], dim=0)
    target = torch.stack([item[4] for item in batch], dim=0)

    return universe_ids, A, X, mask, target


# ============================================================
# Loading and splitting
# ============================================================

def load_static_dataset(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a saved static graph dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Static dataset file not found: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset to be a dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Loaded static dataset is empty.")

    return data


def universe_sort_key(universe_id: str) -> int:
    """
    Sort universe IDs like LH_0, LH_1, LH_2, ...
    """
    text = str(universe_id)

    if text.lower().startswith("lh_"):
        return int(text.split("_", 1)[1])

    return int(text)


def split_universes(
    universe_ids: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split universe IDs into train, validation, and test sets.

    The same seed and ratios should be used across:
        - mean baseline
        - static GCN
        - temporal EvolveGCN-H

    This keeps comparisons fair.
    """
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1. "
            f"Got {total_ratio}"
        )

    universe_ids = list(universe_ids)
    rng = random.Random(seed)
    rng.shuffle(universe_ids)

    n = len(universe_ids)

    if n < 5:
        raise ValueError(
            "Dataset is too small for train/val/test splitting. "
            "Use at least 5 universes."
        )

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = n - n_train - n_val

    if n_test < 1:
        n_train -= 1
        n_test = 1

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}"
        )

    train_ids = universe_ids[:n_train]
    val_ids = universe_ids[n_train:n_train + n_val]
    test_ids = universe_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def create_loaders(
    data: Dict[str, Dict[str, Any]],
    seed: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    """
    Create train, validation, and test DataLoaders.
    """
    universe_ids = sorted(data.keys(), key=universe_sort_key)

    train_ids, val_ids, test_ids = split_universes(
        universe_ids=universe_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_dataset = CamelsStaticGraphDataset(data, train_ids)
    val_dataset = CamelsStaticGraphDataset(data, val_ids)
    test_dataset = CamelsStaticGraphDataset(data, test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, train_ids, val_ids, test_ids


# ============================================================
# Training helpers
# ============================================================

def move_batch_to_device(
    A: torch.Tensor,
    X: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
):
    """
    Move batch tensors to selected device.
    """
    A = A.to(device)
    X = X.to(device)
    mask = mask.to(device)
    target = target.to(device)

    return A, X, mask, target


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    """
    Run one training epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for _, A, X, mask, target in loader:
        A, X, mask, target = move_batch_to_device(
            A=A,
            X=X,
            mask=mask,
            target=target,
            device=device,
        )

        optimizer.zero_grad()

        prediction = model(A=A, X=X, mask=mask)
        loss = criterion(prediction, target)

        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate average MSE loss.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for _, A, X, mask, target in loader:
        A, X, mask, target = move_batch_to_device(
            A=A,
            X=X,
            mask=mask,
            target=target,
            device=device,
        )

        prediction = model(A=A, X=X, mask=mask)
        loss = criterion(prediction, target)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, float | str]]:
    """
    Collect prediction rows for CSV saving.
    """
    model.eval()

    rows: List[Dict[str, float | str]] = []

    for universe_ids, A, X, mask, target in loader:
        A, X, mask, target = move_batch_to_device(
            A=A,
            X=X,
            mask=mask,
            target=target,
            device=device,
        )

        prediction = model(A=A, X=X, mask=mask)

        prediction_cpu = prediction.detach().cpu().view(-1)
        target_cpu = target.detach().cpu().view(-1)

        for universe_id, pred_value, true_value in zip(
            universe_ids,
            prediction_cpu,
            target_cpu,
        ):
            pred_float = float(pred_value)
            true_float = float(true_value)

            abs_error = abs(pred_float - true_float)
            sq_error = (pred_float - true_float) ** 2

            rows.append(
                {
                    "universe_id": universe_id,
                    "true_omega_m": true_float,
                    "pred_omega_m": pred_float,
                    "absolute_error": abs_error,
                    "squared_error": sq_error,
                }
            )

    return rows


def compute_metrics(rows: List[Dict[str, float | str]]) -> Dict[str, float | int]:
    """
    Compute regression metrics from prediction rows.
    """
    if len(rows) == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "num_samples": 0,
        }

    squared_errors = [float(row["squared_error"]) for row in rows]
    absolute_errors = [float(row["absolute_error"]) for row in rows]

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_errors))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "num_samples": len(rows),
    }


# ============================================================
# Saving helpers
# ============================================================

def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_predictions_csv(
    rows: List[Dict[str, float | str]],
    path: str | Path,
) -> None:
    """
    Save prediction rows as CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "universe_id",
        "true_omega_m",
        "pred_omega_m",
        "absolute_error",
        "squared_error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_train_log_csv(
    log_rows: List[Dict[str, float | int]],
    path: str | Path,
) -> None:
    """
    Save training log as CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_mse",
        "val_mse",
        "best_val_mse",
        "best_epoch",
        "learning_rate",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


# ============================================================
# Main training function
# ============================================================

def train_static_gcn(
    dataset_path: str | Path,
    experiment_name: str,
    output_root: str | Path = "experiments",
    seed: int = 123,
    batch_size: int = 8,
    epochs: int = 300,
    patience: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
    graph_pooling: str = "mean",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    grad_clip_norm: float = 1.0,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Train the Static GCN regressor.
    """
    set_seed(seed)

    dataset_path = Path(dataset_path)
    output_root = Path(output_root)

    experiment_dir = output_root / experiment_name
    checkpoints_dir = experiment_dir / "checkpoints"
    predictions_dir = experiment_dir / "predictions"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 90)
    print("CAMELS-SIMBA Static GCN Training")
    print("=" * 90)
    print(f"Dataset path:       {dataset_path}")
    print(f"Experiment name:    {experiment_name}")
    print(f"Experiment dir:     {experiment_dir}")
    print(f"Device:             {device}")
    print(f"Seed:               {seed}")
    print(f"Batch size:         {batch_size}")
    print(f"Epochs:             {epochs}")
    print(f"Patience:           {patience}")
    print(f"Learning rate:      {learning_rate}")
    print(f"Weight decay:       {weight_decay}")
    print(f"Hidden dim:         {hidden_dim}")
    print(f"Num layers:         {num_layers}")
    print(f"Dropout:            {dropout}")
    print(f"Graph pooling:      {graph_pooling}")
    print(f"Grad clip norm:     {grad_clip_norm}")
    print("=" * 90)

    data = load_static_dataset(dataset_path)

    train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = create_loaders(
        data=data,
        seed=seed,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    _, A_example, X_example, mask_example, target_example = next(iter(train_loader))

    batch_size_real, num_nodes, _ = A_example.shape
    _, num_nodes_x, node_features = X_example.shape

    if num_nodes != num_nodes_x:
        raise ValueError(
            f"Node mismatch between A and X: A nodes={num_nodes}, X nodes={num_nodes_x}"
        )

    print()
    print("Data summary")
    print("-" * 90)
    print(f"Total universes:     {len(data)}")
    print(f"Train universes:     {len(train_ids)}")
    print(f"Val universes:       {len(val_ids)}")
    print(f"Test universes:      {len(test_ids)}")
    print(f"Num nodes:           {num_nodes}")
    print(f"Node features:       {node_features}")
    print(f"Example A:           {tuple(A_example.shape)}")
    print(f"Example X:           {tuple(X_example.shape)}")
    print(f"Example mask:        {tuple(mask_example.shape)}")
    print(f"Example target:      {tuple(target_example.shape)}")

    print()
    print("Split details")
    print("-" * 90)
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs:   {val_ids}")
    print(f"Test IDs:  {test_ids}")

    model = StaticGCNRegressor(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        graph_pooling=graph_pooling,
        add_self_loops=True,
        use_layer_norm=True,
        residual=True,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(patience // 4, 5),
        min_lr=1e-6,
    )

    trainable_params = count_parameters(model)

    print()
    print("Model summary")
    print("-" * 90)
    print(model)
    print(f"Trainable parameters: {trainable_params}")

    config = {
        "model": "StaticGCNRegressor",
        "dataset_path": str(dataset_path),
        "experiment_name": experiment_name,
        "output_root": str(output_root),
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "graph_pooling": graph_pooling,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "grad_clip_norm": grad_clip_norm,
        "device": str(device),
        "num_total_universes": len(data),
        "num_train_universes": len(train_ids),
        "num_val_universes": len(val_ids),
        "num_test_universes": len(test_ids),
        "num_nodes": num_nodes,
        "node_features": node_features,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "trainable_parameters": trainable_params,
    }

    save_json(config, experiment_dir / "config.json")

    best_val_mse = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_log_rows: List[Dict[str, float | int]] = []

    best_checkpoint_path = checkpoints_dir / "best_model.pt"

    print()
    print("Training")
    print("-" * 90)

    for epoch in range(1, epochs + 1):
        train_mse = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_mse = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_mse)

        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = val_mse < best_val_mse

        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mse": best_val_mse,
                    "config": config,
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1

        train_log_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "best_val_mse": best_val_mse,
                "best_epoch": best_epoch,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_mse:.8f} | "
            f"Val MSE: {val_mse:.8f} | "
            f"Best Val: {best_val_mse:.8f} at epoch {best_epoch} | "
            f"LR: {current_lr:.8f}"
        )

        if patience_counter >= patience:
            print()
            print(
                f"Early stopping triggered after {patience} "
                f"epochs without validation improvement."
            )
            break

    train_log_path = experiment_dir / "train_log.csv"
    save_train_log_csv(train_log_rows, train_log_path)

    print()
    print("Loading best checkpoint for final evaluation")
    print("-" * 90)

    try:
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    train_rows = collect_predictions(model, train_loader, device)
    val_rows = collect_predictions(model, val_loader, device)
    test_rows = collect_predictions(model, test_loader, device)

    train_metrics = compute_metrics(train_rows)
    val_metrics = compute_metrics(val_rows)
    test_metrics = compute_metrics(test_rows)

    save_predictions_csv(train_rows, predictions_dir / "train_predictions.csv")
    save_predictions_csv(val_rows, predictions_dir / "val_predictions.csv")
    save_predictions_csv(test_rows, predictions_dir / "test_predictions.csv")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "trainable_parameters": trainable_params,
    }

    save_json(metrics, experiment_dir / "metrics.json")

    print()
    print("Final metrics")
    print("-" * 90)
    print(json.dumps(metrics, indent=2))

    print()
    print("=" * 90)
    print("STATIC GCN TRAINING COMPLETE")
    print("=" * 90)
    print(f"Best checkpoint:     {best_checkpoint_path}")
    print(f"Train log:           {train_log_path}")
    print(f"Metrics:             {experiment_dir / 'metrics.json'}")
    print(f"Predictions folder:  {predictions_dir}")

    return metrics


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Static GCN on CAMELS-SIMBA static graph dataset."
    )

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="experiments")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "mean_max"],
    )

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Use 'auto', 'cpu', or 'cuda'.",
    )

    args = parser.parse_args()

    train_static_gcn(
        dataset_path=args.dataset_path,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        grad_clip_norm=args.grad_clip_norm,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()