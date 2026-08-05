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
from src.training.sparse_batch import collate_sparse_static, sparse_batch_to
from src.training.split_manifest import (
    current_repository_commit,
    load_dataset_provenance,
    load_split_manifest,
    validate_split_manifest_seed,
)


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

        target = sample["target"]
        if not torch.is_tensor(target):
            target = torch.tensor(float(target), dtype=torch.float32)
        target = target.float().view(1)

        if "edge_index" in sample:
            graph = {
                "graph_storage": "sparse_edge_index",
                "x": sample["X"].float(),
                "edge_index": sample["edge_index"].long(),
                "edge_weight": sample.get("edge_weight"),
                "mask": sample["mask"].float(),
            }
            return universe_id, graph, None, None, target

        A = sample["A"].float()
        X = sample["X"].float()
        mask = sample["mask"].float()

        return universe_id, A, X, mask, target


def collate_fn(batch):
    """
    Custom collate function because universe_id is a string.
    """
    universe_ids = [item[0] for item in batch]

    if isinstance(batch[0][1], dict):
        graph = collate_sparse_static([item[1] for item in batch])
        target = torch.stack([item[4] for item in batch], dim=0)
        return universe_ids, graph, None, None, target

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


def convert_temporal_final_snapshot_to_static(
    data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a temporal graph dataset to static samples in memory.

    The final snapshot is selected for each universe:
        A     = A_list[-1]
        X     = Nodes_list[-1]
        mask  = mask_list[-1]
        target is preserved unchanged.
    """
    static_data: Dict[str, Dict[str, Any]] = {}

    for universe_id, sample in data.items():
        sparse = "edge_index_list" in sample
        required_keys = ["Nodes_list", "mask_list", "target"]
        required_keys.append("edge_index_list" if sparse else "A_list")
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"{universe_id}: missing required key {key}")

        graph_list = sample["edge_index_list"] if sparse else sample["A_list"]
        if len(graph_list) == 0:
            raise ValueError(f"{universe_id}: A_list is empty.")

        if len(sample["Nodes_list"]) == 0:
            raise ValueError(f"{universe_id}: Nodes_list is empty.")

        if len(sample["mask_list"]) == 0:
            raise ValueError(f"{universe_id}: mask_list is empty.")

        converted = {
            "X": sample["Nodes_list"][-1],
            "mask": sample["mask_list"][-1],
            "target": sample["target"],
            "universe_id": universe_id,
            "snapshot": sample.get("snapshots", [None])[-1],
        }
        if sparse:
            converted["edge_index"] = sample["edge_index_list"][-1]
            weights = sample.get("edge_weight_list")
            converted["edge_weight"] = None if weights is None else weights[-1]
            converted["graph_storage"] = "sparse_edge_index"
        else:
            converted["A"] = sample["A_list"][-1]
        static_data[universe_id] = converted

    return static_data


def load_dataset(
    path: str | Path,
    dataset_format: str = "static",
) -> Dict[str, Dict[str, Any]]:
    """
    Load either a native static dataset or a temporal dataset converted to
    final-snapshot static samples.
    """
    if dataset_format not in {"static", "temporal_final_snapshot"}:
        raise ValueError(
            "dataset_format must be one of: 'static', 'temporal_final_snapshot'."
        )

    data = load_static_dataset(path)

    if dataset_format == "temporal_final_snapshot":
        data = convert_temporal_final_snapshot_to_static(data)

    return data


def load_split_config(path: str | Path) -> Dict[str, Any]:
    """
    Load train/validation/test split IDs from an existing experiment config.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Split config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["train_ids", "val_ids", "test_ids"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Split config missing required key: {key}")

        if not isinstance(config[key], list):
            raise TypeError(f"Split config key {key} must be a list.")

    return config


def validate_split_ids(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    dataset_ids: List[str],
) -> None:
    """
    Validate externally provided train/validation/test split IDs.
    """
    dataset_id_set = set(dataset_ids)
    split_name_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, split_ids in split_name_to_ids.items():
        if len(split_ids) == 0:
            raise ValueError(f"{split_name} split is empty.")

        duplicate_count = len(split_ids) - len(set(split_ids))
        if duplicate_count > 0:
            raise ValueError(
                f"{split_name} split contains {duplicate_count} duplicate IDs."
            )

        missing_ids = sorted(set(split_ids) - dataset_id_set)
        if missing_ids:
            raise ValueError(
                f"{split_name} split contains IDs not present in dataset: "
                f"{missing_ids[:20]}"
            )

    overlaps = {
        "train_val": sorted(set(train_ids) & set(val_ids)),
        "train_test": sorted(set(train_ids) & set(test_ids)),
        "val_test": sorted(set(val_ids) & set(test_ids)),
    }
    nonempty_overlaps = {
        name: ids for name, ids in overlaps.items() if len(ids) > 0
    }

    if nonempty_overlaps:
        raise ValueError(f"Split IDs overlap: {nonempty_overlaps}")


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
    split_config_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    dataset_identity: str | None = None,
):
    """
    Create train, validation, and test DataLoaders.
    """
    universe_ids = sorted(data.keys(), key=universe_sort_key)

    if split_config_path is not None and split_manifest_path is not None:
        raise ValueError("Use only one of split_config_path and split_manifest_path.")
    if split_manifest_path is not None:
        manifest = load_split_manifest(
            split_manifest_path, universe_ids, dataset_identity or "", expected_seed=seed,
        )
        train_ids = list(manifest["train_ids"])
        val_ids = list(manifest["val_ids"])
        test_ids = list(manifest["test_ids"])
    elif split_config_path is not None:
        split_config = load_split_config(split_config_path)
        train_ids = list(split_config["train_ids"])
        val_ids = list(split_config["val_ids"])
        test_ids = list(split_config["test_ids"])
        validate_split_ids(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            dataset_ids=universe_ids,
        )
    else:
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
        shuffle=split_manifest_path is None,
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
    A,
    X,
    mask,
    target: torch.Tensor,
    device: torch.device,
):
    """
    Move batch tensors to selected device.
    """
    if isinstance(A, dict):
        A = sparse_batch_to(A, device)
    else:
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
    dataset_format: str = "static",
    split_config_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    dataset_identity: str | None = None,
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
    conv_type: str = "gcn",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    grad_clip_norm: float = 1.0,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Train the Static GCN regressor.
    """
    split_provenance = (
        validate_split_manifest_seed(split_manifest_path, seed)
        if split_manifest_path is not None else None
    )
    set_seed(seed)

    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    dataset_provenance = load_dataset_provenance(dataset_path)

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
    print(f"Dataset format:     {dataset_format}")
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
    print(f"Conv type:          {conv_type}")
    print(f"Grad clip norm:     {grad_clip_norm}")
    print(f"Split config path:  {split_config_path}")
    print(f"Split manifest:     {split_manifest_path}")
    print("=" * 90)

    data = load_dataset(
        path=dataset_path,
        dataset_format=dataset_format,
    )

    train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = create_loaders(
        data=data,
        seed=seed,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_config_path=split_config_path,
        split_manifest_path=split_manifest_path,
        dataset_identity=dataset_identity,
    )

    _, A_example, X_example, mask_example, target_example = next(iter(train_loader))

    if isinstance(A_example, dict):
        batch_size_real = int(A_example["num_graphs"])
        num_nodes = int(A_example["x"].shape[0])
        node_features = int(A_example["x"].shape[1])
    else:
        batch_size_real, num_nodes, _ = A_example.shape
        _, num_nodes_x, node_features = X_example.shape
        if num_nodes != num_nodes_x:
            raise ValueError(f"Node mismatch between A and X: A nodes={num_nodes}, X nodes={num_nodes_x}")

    print()
    print("Data summary")
    print("-" * 90)
    print(f"Total universes:     {len(data)}")
    print(f"Train universes:     {len(train_ids)}")
    print(f"Val universes:       {len(val_ids)}")
    print(f"Test universes:      {len(test_ids)}")
    print(f"Num nodes:           {num_nodes}")
    print(f"Node features:       {node_features}")
    print(f"Example graph:       {'sparse_edge_index' if isinstance(A_example, dict) else tuple(A_example.shape)}")
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
        conv_type=conv_type,
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
        "dataset_format": dataset_format,
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
        "conv_type": conv_type,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_source": (
            str(split_manifest_path or split_config_path)
            if split_manifest_path is not None or split_config_path is not None
            else "generated_from_seed_and_ratios"
        ),
        "dataset_identity": dataset_identity,
        "dataset_provenance": dataset_provenance,
        "training_git_commit": current_repository_commit(),
        "trainer_invocation_seed": seed,
        "split_manifest_sha256": (
            split_provenance["split_manifest_sha256"] if split_provenance else None
        ),
        "split_manifest_seed": (
            split_provenance["split_manifest_seed"] if split_provenance else None
        ),
        "ordered_split_hashes": (
            split_provenance["split_hashes"] if split_provenance else None
        ),
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
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="static",
        choices=["static", "temporal_final_snapshot"],
        help="Use a native static dataset or convert a temporal dataset to final snapshots in memory.",
    )
    parser.add_argument(
        "--split_config_path",
        type=str,
        default=None,
        help="Optional config.json containing train_ids, val_ids, and test_ids to reuse.",
    )
    parser.add_argument("--split_manifest_path", type=str, default=None,
                        help="Optional immutable split manifest with hashes and dataset identity.")
    parser.add_argument("--dataset_identity", type=str, default=None,
                        help="Dataset checksum/identity required by --split_manifest_path.")
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

    parser.add_argument(
        "--conv_type",
        type=str,
        default="gcn",
        choices=["gcn", "graphsage"],
        help="Message-passing layer type. Default preserves the original Static GCN.",
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
        dataset_format=args.dataset_format,
        split_config_path=args.split_config_path,
        split_manifest_path=args.split_manifest_path,
        dataset_identity=args.dataset_identity,
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
        conv_type=args.conv_type,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        grad_clip_norm=args.grad_clip_norm,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
