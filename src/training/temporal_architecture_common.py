"""Frozen, backward-compatible trainer shared by the two Notebook-16 models."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
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


def _atomic_publish_bytes(path: Path, payload: bytes, *, allow_identical: bool = False) -> None:
    """Publish one complete artifact without ever replacing differing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if allow_identical and path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any, *, allow_identical: bool = False) -> None:
    payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    _atomic_publish_bytes(path, payload, allow_identical=allow_identical)


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, allow_identical: bool = False
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty prediction artifact: {path}")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_publish_bytes(path, handle.getvalue().encode("utf-8"), allow_identical=allow_identical)


def validate_finalizable_run(
    config_path: str | Path,
    expected_model: str,
    model_factory: Callable[[dict[str, Any]], nn.Module],
    frozen_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that training ended normally and that its selected checkpoint is final."""
    config = load_and_validate_config(config_path, expected_model, frozen_values)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    required = (
        experiment_dir / "config.json",
        experiment_dir / "run_metadata.json",
        experiment_dir / "train_log.csv",
        experiment_dir / "checkpoints/best_model.pt",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Training completion cannot be proven: missing {path}")

    resolved = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    identity_keys = (
        "model", "experiment_name", "dataset_sha256", "split_manifest_sha256", "seed",
        "hidden_dim", "num_layers", "graph_pooling", "temporal_pooling", "head_type",
    )
    for key in identity_keys:
        if resolved.get(key) != config.get(key):
            raise RuntimeError(f"Run/config identity mismatch for {key}")

    with (experiment_dir / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Training completion cannot be proven: empty train_log.csv")
    expected_fields = {
        "epoch", "train_mse", "val_mse", "best_val_mse", "best_epoch", "learning_rate"
    }
    if set(rows[0]) != expected_fields:
        raise RuntimeError("Training completion cannot be proven: unexpected train_log schema")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError("Training completion cannot be proven: non-contiguous epochs")
    numeric_fields = ("train_mse", "val_mse", "best_val_mse", "learning_rate")
    if not all(np.isfinite(float(row[key])) for row in rows for key in numeric_fields):
        raise RuntimeError("Training completion cannot be proven: non-finite train log")
    last_epoch = epochs[-1]
    best_epoch = int(rows[-1]["best_epoch"])
    normal_termination = (
        last_epoch == int(config["epochs"])
        or last_epoch - best_epoch >= int(config["patience"])
    )
    if not normal_termination:
        raise RuntimeError("Training completion cannot be proven: no normal stop condition")

    checkpoint_path = experiment_dir / "checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != best_epoch:
        raise RuntimeError("Selected checkpoint does not match final logged best epoch")
    checkpoint_config = checkpoint.get("config", {})
    for key in identity_keys:
        if checkpoint_config.get(key) != config.get(key):
            raise RuntimeError(f"Checkpoint/config identity mismatch for {key}")
    model = model_factory(config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise RuntimeError("Selected checkpoint contains non-finite tensors")
    return {
        "config": config,
        "experiment_dir": experiment_dir,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_log_rows": len(rows),
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "termination": "max_epochs" if last_epoch == int(config["epochs"]) else "early_stopping",
    }


def finalize_existing_run(
    config_path: str | Path,
    expected_model: str,
    model_factory: Callable[[dict[str, Any]], nn.Module],
    *,
    expected_checkpoint_sha256: str,
    frozen_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create only missing post-training artifacts from the already selected checkpoint."""
    evidence = validate_finalizable_run(
        config_path, expected_model, model_factory, frozen_values=frozen_values
    )
    config = evidence["config"]
    experiment_dir = evidence["experiment_dir"]
    checkpoint_path = experiment_dir / "checkpoints/best_model.pt"
    if evidence["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA-256 differs from the forensic recovery binding")

    preserved = {
        relative: sha256_file(experiment_dir / relative)
        for relative in ("config.json", "run_metadata.json", "train_log.csv", "checkpoints/best_model.pt")
    }
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
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    model = model_factory(config).to(device)
    model.load_state_dict(evidence["checkpoint"]["model_state_dict"], strict=True)
    split_rows = {
        "train": collect_predictions(model, train_loader, device),
        "val": collect_predictions(model, val_loader, device),
        "test": collect_predictions(model, test_loader, device),
    }
    expected_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split, rows in split_rows.items():
        if [row["universe_id"] for row in rows] != expected_ids[split]:
            raise RuntimeError(f"{split} prediction identities differ from frozen manifest")
        if not all(np.isfinite(row["target"]) and np.isfinite(row["prediction"]) for row in rows):
            raise RuntimeError(f"{split} predictions contain non-finite values")

    result = {split: metrics(rows) for split, rows in split_rows.items()}
    result.update({
        "best_epoch": evidence["best_epoch"],
        "best_val_mse": evidence["checkpoint"]["best_val_mse"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    for split, rows in split_rows.items():
        _write_csv(
            experiment_dir / "predictions" / f"{split}_predictions.csv",
            rows,
            allow_identical=True,
        )
    _write_json(experiment_dir / "metrics.json", result, allow_identical=True)

    after = {
        relative: sha256_file(experiment_dir / relative)
        for relative in preserved
    }
    if after != preserved or after["checkpoints/best_model.pt"] != expected_checkpoint_sha256:
        raise RuntimeError("Recovery altered preserved training evidence")
    recovery = {
        "schema_version": "evolvegcn_o_post_training_recovery_v1",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path),
        "checkpoint_sha256_before": expected_checkpoint_sha256,
        "checkpoint_sha256_after": after["checkpoints/best_model.pt"],
        "no_additional_optimization_performed": True,
        "checkpoint_unchanged": True,
        "checkpoint_selection_unchanged": True,
        "selected_checkpoint_epoch": evidence["best_epoch"],
        "training_termination": evidence["termination"],
        "train_log_preserved": True,
    }
    _write_json(experiment_dir / "recovery_metadata.json", recovery, allow_identical=False)
    return result


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
