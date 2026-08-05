from __future__ import annotations

"""
train_evolvegcn_h.py

Train an EvolveGCN-H temporal graph regressor on CAMELS-SIMBA temporal graph datasets.

Purpose
-------
This script trains the temporal graph model in the thesis pipeline.

Comparison ladder:
    1. Mean Omega_m baseline
    2. Static GCN baseline
    3. Temporal EvolveGCN-H model

Expected temporal dataset format
--------------------------------
The temporal dataset should be created by:

    src.data.build_temporal_sequences

Expected saved format:

    {
        "LH_0": {
            "A_list": [Tensor(N, N), ...],
            "Nodes_list": [Tensor(N, F), ...],
            "mask_list": [Tensor(N, 1), ...],
            "target": Tensor scalar,
            "snapshots": [...metadata...],
            ...
        },
        ...
    }

Each universe becomes one training sample:
    A_seq:    [T, N, N]
    X_seq:    [T, N, F]
    mask_seq: [T, N, 1]
    target:   [1]

Official preprocessing:
    v2_logmass_minmax_top100_periodic_knn

Official node features:
    [log10_Mvir, X, Y, Z, VX, VY, VZ]

Example command
---------------
python -m src.training.train_evolvegcn_h \
  --dataset_path data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt \
  --experiment_name evolvegcn_h_100u_seed123 \
  --output_root experiments \
  --seed 123 \
  --batch_size 4 \
  --epochs 300 \
  --patience 40 \
  --learning_rate 0.001 \
  --weight_decay 0.00001 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.2 \
  --temporal_pooling mean \
  --graph_pooling mean \
  --add_self_loops \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --grad_clip_norm 1.0 \
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

from src.models.evolvegcn_h import EvolveGCNHRegressor, count_parameters
from src.training.sparse_batch import collate_sparse_temporal, sparse_batch_to
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

class CamelsTemporalDataset(Dataset):
    """
    PyTorch Dataset wrapper for CAMELS-SIMBA temporal graph sequences.

    Each item returned:
        universe_id: str
        A_seq: Tensor [T, N, N]
        X_seq: Tensor [T, N, F]
        mask_seq: Tensor [T, N, 1]
        target: Tensor [1]
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        universe_ids: List[str],
        use_summary_features: bool = False,
        summary_feature_mean: torch.Tensor | None = None,
        summary_feature_std: torch.Tensor | None = None,
    ) -> None:
        self.data_dict = data_dict
        self.universe_ids = universe_ids
        self.use_summary_features = use_summary_features
        self.summary_feature_mean = summary_feature_mean
        self.summary_feature_std = summary_feature_std

        if (summary_feature_mean is None) != (summary_feature_std is None):
            raise ValueError(
                "summary_feature_mean and summary_feature_std must both be "
                "provided or both be None."
            )

    def __len__(self) -> int:
        return len(self.universe_ids)

    def __getitem__(self, index: int):
        universe_id = self.universe_ids[index]
        sample = self.data_dict[universe_id]

        sparse = "edge_index_list" in sample
        required_keys = ["Nodes_list", "mask_list", "target"]
        required_keys.append("edge_index_list" if sparse else "A_list")
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"{universe_id}: missing required key {key}")

        A_seq = sample if sparse else torch.stack(sample["A_list"], dim=0).float()
        X_seq = torch.stack(sample["Nodes_list"], dim=0).float()
        mask_seq = torch.stack(sample["mask_list"], dim=0).float()

        target = sample["target"]

        if not torch.is_tensor(target):
            target = torch.tensor(float(target), dtype=torch.float32)

        target = target.float().view(1)

        if self.use_summary_features:
            summary_features = compute_temporal_summary_features(
                X_seq=X_seq,
                mask_seq=mask_seq,
            )

            if self.summary_feature_mean is not None:
                summary_features = (
                    (summary_features - self.summary_feature_mean)
                    / self.summary_feature_std
                )

            return universe_id, A_seq, X_seq, mask_seq, target, summary_features

        return universe_id, A_seq, X_seq, mask_seq, target


def compute_temporal_summary_features(
    X_seq: torch.Tensor,
    mask_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the same per-snapshot summary features as the summary baseline.

    For 5 snapshots, this returns 100 features:
        20 summary statistics per snapshot * 5 snapshots.
    """
    features: List[float] = []

    for timestep in range(X_seq.shape[0]):
        X_t = X_seq[timestep]
        mask_t = mask_seq[timestep].squeeze(-1) > 0

        valid = X_t[mask_t]

        if valid.shape[0] == 0:
            features.extend([0.0] * 20)
            continue

        mass = valid[:, 0]
        pos = valid[:, 1:4]
        vel = valid[:, 4:7]

        speed = torch.linalg.norm(vel, dim=1)

        summary = [
            float(valid.shape[0]),

            float(mass.mean()),
            float(mass.std(unbiased=False)),
            float(mass.min()),
            float(mass.max()),
            float(torch.quantile(mass, 0.5)),

            float(pos[:, 0].mean()),
            float(pos[:, 1].mean()),
            float(pos[:, 2].mean()),
            float(pos[:, 0].std(unbiased=False)),
            float(pos[:, 1].std(unbiased=False)),
            float(pos[:, 2].std(unbiased=False)),

            float(vel[:, 0].mean()),
            float(vel[:, 1].mean()),
            float(vel[:, 2].mean()),
            float(vel[:, 0].std(unbiased=False)),
            float(vel[:, 1].std(unbiased=False)),
            float(vel[:, 2].std(unbiased=False)),

            float(speed.mean()),
            float(speed.std(unbiased=False)),
        ]

        features.extend(summary)

    return torch.tensor(features, dtype=torch.float32)


def compute_summary_feature_scaler(
    data: Dict[str, Dict[str, Any]],
    train_ids: List[str],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit summary-feature normalization statistics on the train split only.
    """
    if len(train_ids) == 0:
        raise ValueError("Cannot compute summary feature scaler with empty train_ids.")

    summary_rows = []

    for universe_id in train_ids:
        sample = data[universe_id]

        required_keys = ["Nodes_list", "mask_list"]
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"{universe_id}: missing required key {key}")

        X_seq = torch.stack(sample["Nodes_list"], dim=0).float()
        mask_seq = torch.stack(sample["mask_list"], dim=0).float()

        summary_rows.append(
            compute_temporal_summary_features(
                X_seq=X_seq,
                mask_seq=mask_seq,
            )
        )

    summary_matrix = torch.stack(summary_rows, dim=0).float()

    if torch.isnan(summary_matrix).any() or torch.isinf(summary_matrix).any():
        raise ValueError("Raw train summary features contain NaN or Inf values.")

    mean = summary_matrix.mean(dim=0)
    std = summary_matrix.std(dim=0, unbiased=False).clamp(min=eps)

    return mean, std


def compute_target_scaler(
    data: Dict[str, Dict[str, Any]],
    train_ids: List[str],
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Fit target normalization statistics on the train split only.
    """
    if len(train_ids) == 0:
        raise ValueError("Cannot compute target scaler with empty train_ids.")

    targets = []

    for universe_id in train_ids:
        sample = data[universe_id]

        if "target" not in sample:
            raise KeyError(f"{universe_id}: missing required key target")

        target = sample["target"]

        if torch.is_tensor(target):
            target_value = float(target.detach().cpu().view(-1)[0])
        else:
            target_value = float(target)

        targets.append(target_value)

    target_tensor = torch.tensor(targets, dtype=torch.float32)

    if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
        raise ValueError("Train targets contain NaN or Inf values.")

    target_mean = float(target_tensor.mean().item())
    target_std = float(target_tensor.std(unbiased=False).clamp(min=eps).item())

    return target_mean, target_std


def normalize_target_tensor(
    target: torch.Tensor,
    target_mean: float | None,
    target_std: float | None,
) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return target

    return (target - target_mean) / target_std


def denormalize_prediction_tensor(
    prediction: torch.Tensor,
    target_mean: float | None,
    target_std: float | None,
) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return prediction

    return prediction * target_std + target_mean


def collate_fn(batch):
    """
    Custom collate function because universe_id is a string.
    """
    universe_ids = [item[0] for item in batch]

    if isinstance(batch[0][1], dict):
        sparse_temporal = collate_sparse_temporal([item[1] for item in batch])
        target = torch.stack([item[4] for item in batch], dim=0)
        if len(batch[0]) == 6:
            summary_features = torch.stack([item[5] for item in batch], dim=0)
            return universe_ids, sparse_temporal, None, None, target, summary_features
        return universe_ids, sparse_temporal, None, None, target

    A_seq = torch.stack([item[1] for item in batch], dim=0)
    X_seq = torch.stack([item[2] for item in batch], dim=0)
    mask_seq = torch.stack([item[3] for item in batch], dim=0)
    target = torch.stack([item[4] for item in batch], dim=0)

    if len(batch[0]) == 6:
        summary_features = torch.stack([item[5] for item in batch], dim=0)
        return universe_ids, A_seq, X_seq, mask_seq, target, summary_features

    return universe_ids, A_seq, X_seq, mask_seq, target


# ============================================================
# Data loading and splitting
# ============================================================

def load_temporal_dataset(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a saved temporal graph dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Temporal dataset file not found: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset to be a dict, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Loaded temporal dataset is empty.")

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

    Use the same seed/ratios across mean baseline, static GCN, and EvolveGCN-H
    for fair comparison.
    """
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1. Got {total_ratio}"
        )

    universe_ids = list(universe_ids)

    if len(universe_ids) < 5:
        raise ValueError(
            "Dataset is too small for train/val/test splitting. "
            "Use at least 5 universes."
        )

    rng = random.Random(seed)
    rng.shuffle(universe_ids)

    n = len(universe_ids)

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
    use_summary_features: bool = False,
    summary_feature_scaler_eps: float = 1e-6,
    split_manifest_path: str | Path | None = None,
    dataset_identity: str | None = None,
):
    """
    Create train, validation, and test DataLoaders.
    """
    universe_ids = sorted(data.keys(), key=universe_sort_key)

    if split_manifest_path is not None:
        manifest = load_split_manifest(
            split_manifest_path, universe_ids, dataset_identity or "", expected_seed=seed,
        )
        train_ids = list(manifest["train_ids"])
        val_ids = list(manifest["val_ids"])
        test_ids = list(manifest["test_ids"])
    else:
        train_ids, val_ids, test_ids = split_universes(
            universe_ids=universe_ids, seed=seed, train_ratio=train_ratio,
            val_ratio=val_ratio, test_ratio=test_ratio,
        )

    summary_feature_mean = None
    summary_feature_std = None

    if use_summary_features:
        summary_feature_mean, summary_feature_std = compute_summary_feature_scaler(
            data=data,
            train_ids=train_ids,
            eps=summary_feature_scaler_eps,
        )

    train_dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=train_ids,
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_feature_mean,
        summary_feature_std=summary_feature_std,
    )
    val_dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=val_ids,
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_feature_mean,
        summary_feature_std=summary_feature_std,
    )
    test_dataset = CamelsTemporalDataset(
        data_dict=data,
        universe_ids=test_ids,
        use_summary_features=use_summary_features,
        summary_feature_mean=summary_feature_mean,
        summary_feature_std=summary_feature_std,
    )

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

    return (
        train_loader,
        val_loader,
        test_loader,
        train_ids,
        val_ids,
        test_ids,
        summary_feature_mean,
        summary_feature_std,
    )


# ============================================================
# Batch validation and movement
# ============================================================

def validate_example_batch(
    A_seq,
    X_seq,
    mask_seq,
    target: torch.Tensor,
    summary_features: torch.Tensor | None = None,
) -> None:
    """
    Validate one example batch before training.
    """
    if isinstance(A_seq, dict):
        snapshots = A_seq.get("snapshots", [])
        if not snapshots or any(graph["edge_index"].shape[0] != 2 for graph in snapshots):
            raise ValueError("Invalid sparse temporal example batch.")
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError(f"Expected target [B,1], got {tuple(target.shape)}")
        return
    if A_seq.ndim != 4:
        raise ValueError(f"Expected A_seq [B, T, N, N], got {tuple(A_seq.shape)}")

    if X_seq.ndim != 4:
        raise ValueError(f"Expected X_seq [B, T, N, F], got {tuple(X_seq.shape)}")

    if mask_seq.ndim != 4:
        raise ValueError(
            f"Expected mask_seq [B, T, N, 1], got {tuple(mask_seq.shape)}"
        )

    if target.ndim != 2 or target.shape[1] != 1:
        raise ValueError(f"Expected target [B, 1], got {tuple(target.shape)}")

    batch_a, time_a, nodes_a, nodes_a_2 = A_seq.shape
    batch_x, time_x, nodes_x, _ = X_seq.shape
    batch_m, time_m, nodes_m, mask_features = mask_seq.shape

    if nodes_a != nodes_a_2:
        raise ValueError(f"A_seq must contain square matrices, got {tuple(A_seq.shape)}")

    if batch_a != batch_x or time_a != time_x or nodes_a != nodes_x:
        raise ValueError(
            f"A_seq and X_seq mismatch: A_seq={tuple(A_seq.shape)}, "
            f"X_seq={tuple(X_seq.shape)}"
        )

    if batch_m != batch_a or time_m != time_a or nodes_m != nodes_a:
        raise ValueError(
            f"mask_seq mismatch: mask_seq={tuple(mask_seq.shape)}, "
            f"A_seq={tuple(A_seq.shape)}"
        )

    if mask_features != 1:
        raise ValueError(f"Expected mask last dimension 1, got {mask_features}")

    if torch.isnan(X_seq).any() or torch.isinf(X_seq).any():
        raise ValueError("Example X_seq contains NaN or Inf values.")

    if torch.isnan(A_seq).any() or torch.isinf(A_seq).any():
        raise ValueError("Example A_seq contains NaN or Inf values.")

    if torch.isnan(mask_seq).any() or torch.isinf(mask_seq).any():
        raise ValueError("Example mask_seq contains NaN or Inf values.")

    if summary_features is not None:
        if summary_features.ndim != 2:
            raise ValueError(
                "Expected summary_features [B, S], "
                f"got {tuple(summary_features.shape)}"
            )

        if summary_features.shape[0] != batch_a:
            raise ValueError(
                "summary_features batch size does not match A_seq batch: "
                f"summary_features={tuple(summary_features.shape)}, "
                f"A_seq={tuple(A_seq.shape)}"
            )

        if torch.isnan(summary_features).any() or torch.isinf(summary_features).any():
            raise ValueError("Example summary_features contains NaN or Inf values.")


def move_batch_to_device(
    A_seq,
    X_seq,
    mask_seq,
    target: torch.Tensor,
    device: torch.device,
    summary_features: torch.Tensor | None = None,
):
    """
    Move batch tensors to selected device.
    """
    if isinstance(A_seq, dict):
        A_seq = sparse_batch_to(A_seq, device)
    else:
        A_seq = A_seq.to(device)
        X_seq = X_seq.to(device)
        mask_seq = mask_seq.to(device)
    target = target.to(device)

    if summary_features is not None:
        summary_features = summary_features.to(device)

    return A_seq, X_seq, mask_seq, target, summary_features


def unpack_batch(batch):
    """
    Support the original 5-item batch and the optional 6-item hybrid batch.
    """
    if len(batch) == 6:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = batch
        return universe_ids, A_seq, X_seq, mask_seq, target, summary_features

    universe_ids, A_seq, X_seq, mask_seq, target = batch
    return universe_ids, A_seq, X_seq, mask_seq, target, None


# ============================================================
# Training and evaluation helpers
# ============================================================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> float:
    """
    Run one training epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        _, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq, X_seq, mask_seq, target, summary_features = move_batch_to_device(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            target=target,
            device=device,
            summary_features=summary_features,
        )

        optimizer.zero_grad()

        prediction = model(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )

        loss_target = normalize_target_tensor(
            target=target,
            target_mean=target_mean,
            target_std=target_std,
        )

        loss = criterion(prediction, loss_target)

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
    target_mean: float | None = None,
    target_std: float | None = None,
) -> float:
    """
    Evaluate average MSE loss.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        _, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq, X_seq, mask_seq, target, summary_features = move_batch_to_device(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            target=target,
            device=device,
            summary_features=summary_features,
        )

        prediction = model(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )

        loss_target = normalize_target_tensor(
            target=target,
            target_mean=target_mean,
            target_std=target_std,
        )

        loss = criterion(prediction, loss_target)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> List[Dict[str, float | str]]:
    """
    Collect prediction rows for CSV saving.
    """
    model.eval()

    rows: List[Dict[str, float | str]] = []

    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)

        A_seq, X_seq, mask_seq, target, summary_features = move_batch_to_device(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            target=target,
            device=device,
            summary_features=summary_features,
        )

        prediction = model(
            A_seq=A_seq,
            X_seq=X_seq,
            mask_seq=mask_seq,
            summary_features=summary_features,
        )
        prediction = denormalize_prediction_tensor(
            prediction=prediction,
            target_mean=target_mean,
            target_std=target_std,
        )

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
            "r2": float("nan"),
            "num_samples": 0,
        }

    squared_errors = [float(row["squared_error"]) for row in rows]
    absolute_errors = [float(row["absolute_error"]) for row in rows]

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_errors))
    targets = np.asarray([float(row["true_omega_m"]) for row in rows], dtype=np.float64)
    ss_res = float(np.sum(squared_errors, dtype=np.float64))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2, dtype=np.float64))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
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
    Save epoch training log as CSV.
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

def train_evolvegcn_h(
    dataset_path: str | Path,
    experiment_name: str,
    output_root: str | Path = "experiments",
    seed: int = 123,
    batch_size: int = 4,
    epochs: int = 300,
    patience: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    activation: str = "relu",
    temporal_pooling: str = "mean",
    graph_pooling: str = "mean",
    head_type: str = "mlp",
    add_self_loops: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    grad_clip_norm: float = 1.0,
    use_summary_features: bool = False,
    normalize_target: bool = False,
    split_manifest_path: str | Path | None = None,
    dataset_identity: str | None = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Train EvolveGCN-H regressor on a CAMELS-SIMBA temporal graph dataset.
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
    print("CAMELS-SIMBA EvolveGCN-H Training")
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
    print(f"Activation:         {activation}")
    print(f"Temporal pooling:   {temporal_pooling}")
    print(f"Graph pooling:      {graph_pooling}")
    print(f"Head type:          {head_type}")
    print(f"Add self loops:     {add_self_loops}")
    print(f"Grad clip norm:     {grad_clip_norm}")
    print(f"Summary features:   {use_summary_features}")
    print(f"Normalize target:   {normalize_target}")
    print(f"Split manifest:     {split_manifest_path}")
    print("=" * 90)

    data = load_temporal_dataset(dataset_path)

    summary_feature_scaler_eps = 1e-6

    (
        train_loader,
        val_loader,
        test_loader,
        train_ids,
        val_ids,
        test_ids,
        summary_feature_mean,
        summary_feature_std,
    ) = create_loaders(
        data=data,
        seed=seed,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        use_summary_features=use_summary_features,
        summary_feature_scaler_eps=summary_feature_scaler_eps,
        split_manifest_path=split_manifest_path,
        dataset_identity=dataset_identity,
    )

    target_mean = None
    target_std = None

    if normalize_target:
        target_mean, target_std = compute_target_scaler(
            data=data,
            train_ids=train_ids,
        )

    example_batch = next(iter(train_loader))
    _, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(example_batch)

    validate_example_batch(
        A_seq=A_seq,
        X_seq=X_seq,
        mask_seq=mask_seq,
        target=target,
        summary_features=summary_features,
    )

    if isinstance(A_seq, dict):
        num_snapshots = int(A_seq["num_timesteps"])
        num_nodes = int(data[train_ids[0]]["Nodes_list"][0].shape[0])
        node_features = int(A_seq["snapshots"][0]["x"].shape[1])
    else:
        _, num_snapshots, num_nodes, node_features = X_seq.shape
    summary_feature_dim = 0

    if summary_features is not None:
        summary_feature_dim = int(summary_features.shape[1])

    print()
    print("Data summary")
    print("-" * 90)
    print(f"Total universes:     {len(data)}")
    print(f"Train universes:     {len(train_ids)}")
    print(f"Val universes:       {len(val_ids)}")
    print(f"Test universes:      {len(test_ids)}")
    print(f"Num snapshots:       {num_snapshots}")
    print(f"Num nodes:           {num_nodes}")
    print(f"Node features:       {node_features}")
    print(f"Example graph:       {'sparse_edge_index' if isinstance(A_seq, dict) else tuple(A_seq.shape)}")
    print(f"Example target:      {tuple(target.shape)}")
    print(f"Summary features:    {use_summary_features}")
    print(f"Summary feature dim: {summary_feature_dim}")
    print(f"Normalize target:    {normalize_target}")

    if normalize_target:
        print(f"Target mean:         {target_mean:.8f}")
        print(f"Target std:          {target_std:.8f}")

    print()
    print("Split details")
    print("-" * 90)
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs:   {val_ids}")
    print(f"Test IDs:  {test_ids}")

    model = EvolveGCNHRegressor(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        temporal_pooling=temporal_pooling,
        graph_pooling=graph_pooling,
        add_self_loops=add_self_loops,
        summary_feature_dim=summary_feature_dim,
        head_type=head_type,
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
        "model": "EvolveGCNHRegressor",
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
        "activation": activation,
        "temporal_pooling": temporal_pooling,
        "graph_pooling": graph_pooling,
        "head_type": head_type,
        "add_self_loops": add_self_loops,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "grad_clip_norm": grad_clip_norm,
        "optimizer": "AdamW",
        "loss": "MSELoss",
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": max(patience // 4, 5),
            "min_lr": 1e-6,
        },
        "checkpoint_criterion": "minimum_validation_mse",
        "deterministic_seed_handling": True,
        "use_summary_features": use_summary_features,
        "summary_feature_dim": summary_feature_dim,
        "summary_features_normalized": bool(use_summary_features),
        "summary_feature_mean": (
            summary_feature_mean.detach().cpu().tolist()
            if summary_feature_mean is not None
            else None
        ),
        "summary_feature_std": (
            summary_feature_std.detach().cpu().tolist()
            if summary_feature_std is not None
            else None
        ),
        "summary_feature_scaler_source": (
            "train_split_only" if use_summary_features else None
        ),
        "summary_feature_scaler_eps": (
            summary_feature_scaler_eps if use_summary_features else None
        ),
        "normalize_target": normalize_target,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_scaler_source": (
            "train_split_only" if normalize_target else None
        ),
        "device": str(device),
        "num_total_universes": len(data),
        "num_train_universes": len(train_ids),
        "num_val_universes": len(val_ids),
        "num_test_universes": len(test_ids),
        "num_snapshots": num_snapshots,
        "num_nodes": num_nodes,
        "node_features": node_features,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "split_source": str(split_manifest_path) if split_manifest_path else "generated_from_seed_and_ratios",
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
            target_mean=target_mean,
            target_std=target_std,
        )

        val_mse = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
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

    train_rows = collect_predictions(
        model,
        train_loader,
        device,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_rows = collect_predictions(
        model,
        val_loader,
        device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_rows = collect_predictions(
        model,
        test_loader,
        device,
        target_mean=target_mean,
        target_std=target_std,
    )

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
    print("EVOLVEGCN-H TRAINING COMPLETE")
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
        description="Train EvolveGCN-H on CAMELS-SIMBA temporal graph sequences."
    )

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split_manifest_path", type=str, default=None,
                        help="Optional immutable split manifest with ordered IDs and verified hashes.")
    parser.add_argument("--dataset_identity", type=str, default=None,
                        help="Dataset checksum/identity required by --split_manifest_path.")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="experiments")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "elu"],
        help="Activation function used inside EvolveGCN-H layers.",
    )

    parser.add_argument(
        "--temporal_pooling",
        type=str,
        default="mean",
        choices=["mean", "last"],
    )

    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["mean", "sum", "mean_max"],
    )

    parser.add_argument(
        "--head_type",
        type=str,
        default="mlp",
        choices=["mlp", "linear"],
        help="Regression head type after graph/temporal pooling.",
    )

    parser.add_argument(
        "--add_self_loops",
        action="store_true",
        default=True,
        help="Add self-loops inside EvolveGCN-H adjacency normalization.",
    )

    parser.add_argument(
        "--no_self_loops",
        action="store_false",
        dest="add_self_loops",
        help="Disable self-loops inside EvolveGCN-H.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--use_summary_features",
        action="store_true",
        help="Concatenate universe-level summary features with the EvolveGCN-H embedding.",
    )

    parser.add_argument(
        "--normalize_target",
        action="store_true",
        help="Train on train-split normalized targets and report metrics in original scale.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Use 'auto', 'cpu', or 'cuda'.",
    )

    args = parser.parse_args()

    train_evolvegcn_h(
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
        activation=args.activation,
        temporal_pooling=args.temporal_pooling,
        graph_pooling=args.graph_pooling,
        head_type=args.head_type,
        add_self_loops=args.add_self_loops,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        grad_clip_norm=args.grad_clip_norm,
        use_summary_features=args.use_summary_features,
        normalize_target=args.normalize_target,
        split_manifest_path=args.split_manifest_path,
        dataset_identity=args.dataset_identity,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
