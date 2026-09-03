"""Frozen, backward-compatible trainer shared by the two Notebook-16 models."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.training.split_manifest import current_repository_commit, load_dataset_provenance
from src.training.train_evolvegcn_h import (
    create_loaders,
    evaluate_loss,
    load_temporal_dataset,
    move_batch_to_device,
    run_one_epoch,
    set_seed,
    unpack_batch,
    validate_example_batch,
)


FROZEN_VALUES = {
    "batch_size": 8, "epochs": 300, "patience": 40,
    "learning_rate": 0.001, "weight_decay": 1e-5,
    "hidden_dim": 32, "num_gcn_layers": 3, "dropout": 0.2,
    "scheduler_patience": 10, "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6, "grad_clip_norm": 1.0,
    "target": "Omega_m", "target_normalization": "none",
    "node_normalization": "none", "graph_pooling": "mean",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_config(
    path: str | Path,
    expected_model: str,
    frozen_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model") != expected_model:
        raise ValueError(f"Expected model {expected_model}, got {config.get('model')!r}")
    for key, expected in (FROZEN_VALUES if frozen_values is None else frozen_values).items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen config mismatch for {key}: {config.get(key)!r} != {expected!r}")
    if config.get("seed") not in {42, 123, 2025}:
        raise ValueError("seed must be one of 42, 123, 2025")
    if config.get("scale_factors") != [0.2, 0.25, 0.51209, 0.75065, 1.0]:
        raise ValueError("authoritative scale_factors mismatch")
    return config


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        universe_ids, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(batch)
        A_seq, X_seq, mask_seq, target, summary_features = move_batch_to_device(
            A_seq, X_seq, mask_seq, target, device, summary_features,
        )
        prediction = model(A_seq=A_seq, X_seq=X_seq, mask_seq=mask_seq,
                           summary_features=summary_features)
        for universe_id, truth, estimate in zip(
            universe_ids, target.detach().cpu().view(-1), prediction.detach().cpu().view(-1)
        ):
            rows.append({"universe_id": universe_id, "target": float(truth),
                         "prediction": float(estimate)})
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    residual = prediction - target
    mse = float(np.mean(residual ** 2))
    target_ss = float(np.sum((target - target.mean()) ** 2))
    return {
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / target_ss),
        "num_samples": len(rows),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_from_config(
    config_path: str | Path,
    expected_model: str,
    model_factory: Callable[[dict[str, Any]], nn.Module],
    frozen_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one explicitly configured production run; never overwrite a run."""
    config = load_and_validate_config(config_path, expected_model, frozen_values)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    if experiment_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {experiment_dir}")

    dataset_path = Path(config["dataset_path"])
    if sha256_file(dataset_path) != config["dataset_sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    if sha256_file(Path(config["split_manifest_path"])) != config["split_manifest_sha256"]:
        raise ValueError("split manifest SHA-256 mismatch")
    set_seed(int(config["seed"]))
    data = load_temporal_dataset(dataset_path)
    loaders = create_loaders(
        data=data, seed=config["seed"], batch_size=config["batch_size"],
        train_ratio=0.7, val_ratio=0.099, test_ratio=0.201,
        split_manifest_path=config["split_manifest_path"],
        dataset_identity=config["dataset_sha256"], use_summary_features=False,
    )
    train_loader, val_loader, test_loader, train_ids, val_ids, test_ids, _, _ = loaders
    example = next(iter(train_loader))
    _, A_seq, X_seq, mask_seq, target, summary_features = unpack_batch(example)
    validate_example_batch(A_seq, X_seq, mask_seq, target, summary_features)

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    model = model_factory(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != config["parameter_count"]:
        raise ValueError(f"parameter count mismatch: {parameter_count} != {config['parameter_count']}")

    experiment_dir.mkdir(parents=True)
    checkpoints = experiment_dir / "checkpoints"
    predictions = experiment_dir / "predictions"
    checkpoints.mkdir()
    predictions.mkdir()
    resolved = dict(config)
    resolved.update({
        "source_config": str(config_path), "training_git_commit": current_repository_commit(),
        "dataset_provenance": load_dataset_provenance(dataset_path),
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "production_training_performed": True,
    })
    _write_json(experiment_dir / "config.json", resolved)
    _write_json(experiment_dir / "run_metadata.json", {
        "training_git_commit": resolved["training_git_commit"],
        "dataset_sha256": config["dataset_sha256"],
        "split_manifest_sha256": config["split_manifest_sha256"],
        "source_config": str(config_path),
    })

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["scheduler_factor"],
        patience=config["scheduler_patience"], min_lr=config["scheduler_min_lr"],
    )
    best = float("inf")
    best_epoch = -1
    stale = 0
    log: list[dict[str, Any]] = []
    checkpoint_path = checkpoints / "best_model.pt"
    for epoch in range(1, config["epochs"] + 1):
        train_mse = run_one_epoch(model, train_loader, optimizer, criterion, device,
                                  config["grad_clip_norm"])
        val_mse = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_mse)
        if val_mse < best:
            best, best_epoch, stale = val_mse, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_mse": best, "config": resolved}, checkpoint_path)
        else:
            stale += 1
        log.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                    "best_val_mse": best, "best_epoch": best_epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"]})
        if stale >= config["patience"]:
            break

    with (experiment_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_rows = {
        "train": collect_predictions(model, train_loader, device),
        "val": collect_predictions(model, val_loader, device),
        "test": collect_predictions(model, test_loader, device),
    }
    for split, rows in split_rows.items():
        _write_csv(predictions / f"{split}_predictions.csv", rows)
    result = {split: metrics(rows) for split, rows in split_rows.items()}
    result.update({"best_epoch": best_epoch, "best_val_mse": best,
                   "trainable_parameters": parameter_count})
    _write_json(experiment_dir / "metrics.json", result)
    return result
