from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.run_summary_feature_baseline import (
    load_dataset,
    extract_temporal_summary,
    get_target,
    load_split_config,
    validate_split_ids,
    build_arrays_for_ids,
)


class SummaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def mae_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(mae), float(rmse)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X_t).cpu().numpy().reshape(-1)
    return mae_rmse(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--split_config_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(Path(args.dataset_path))
    universe_ids = sorted(data.keys())

    X_list = []
    y_list = []

    for uid in universe_ids:
        X_list.append(extract_temporal_summary(data[uid]))
        y_list.append(get_target(data[uid]))

    features_by_id = dict(zip(universe_ids, X_list))
    targets_by_id = dict(zip(universe_ids, y_list))

    split_cfg = load_split_config(Path(args.split_config_path))
    train_ids = split_cfg["train_ids"]
    val_ids = split_cfg["val_ids"]
    test_ids = split_cfg["test_ids"]

    validate_split_ids(train_ids, val_ids, test_ids, universe_ids, split_cfg)

    X_train, y_train = build_arrays_for_ids(train_ids, features_by_id, targets_by_id)
    X_val, y_val = build_arrays_for_ids(val_ids, features_by_id, targets_by_id)
    X_test, y_test = build_arrays_for_ids(test_ids, features_by_id, targets_by_id)

    # Train-only standardization: no leakage.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SummaryMLP(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        val_mae, val_rmse = evaluate(model, X_val, y_val, device)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Val MAE={val_mae:.6f} | Val RMSE={val_rmse:.6f}")

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    train_mae, train_rmse = evaluate(model, X_train, y_train, device)
    val_mae, val_rmse = evaluate(model, X_val, y_val, device)
    test_mae, test_rmse = evaluate(model, X_test, y_test, device)

    results = {
        "dataset_path": args.dataset_path,
        "split_config_path": args.split_config_path,
        "seed": args.seed,
        "input_dim": int(X_train.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "train": {"mae": train_mae, "rmse": train_rmse},
        "val": {"mae": val_mae, "rmse": val_rmse},
        "test": {"mae": test_mae, "rmse": test_rmse},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

    out_dir = Path(args.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSUMMARY MLP COMPLETE")
    print("=" * 80)
    print(f"Train MAE={train_mae:.6f} RMSE={train_rmse:.6f}")
    print(f"Val   MAE={val_mae:.6f} RMSE={val_rmse:.6f}")
    print(f"Test  MAE={test_mae:.6f} RMSE={test_rmse:.6f}")
    print("Saved to:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
