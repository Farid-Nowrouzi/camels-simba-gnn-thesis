"""Run the controlled modern U1000 final-snapshot Summary MLP family."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.baseline_common import (
    SPLIT_FILENAMES,
    build_prediction_rows,
    compute_metrics,
    git_provenance,
    sha256_file,
    write_json,
    write_prediction_csv,
)
from src.evaluation.run_modern_summary_baselines import (
    load_bound_data,
    load_family as load_classical_family,
    load_summary_data,
)
from src.evaluation.summary_features import (
    SUMMARY_DEFINITION_VERSION,
    SUMMARY_FEATURE_NAMES,
    arrays_for_ids,
    extract_dataset_summaries,
)


MODEL_FAMILY = "summary_mlp"
ARCHITECTURE_VERSION = "summary_mlp_20x64x64x1_relu_drop0p2_v1"
INPUT_DIM = 20
HIDDEN_DIMS = [64, 64]
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 40
NUM_WORKERS = 0


class SummaryMLP(nn.Module):
    """Fixed neural comparator for the versioned 20-feature summary."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[1], 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected input [B, {INPUT_DIM}], got {tuple(inputs.shape)}")
        return self.net(inputs)


def seed_everything(seed: int) -> torch.Generator:
    """Seed every stochastic source used by this trainer."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def fit_feature_scaler(training_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != INPUT_DIM or not np.isfinite(values).all():
        raise ValueError(f"Training summaries must be finite [N, {INPUT_DIM}].")
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, ddof=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    return mean, scale


def transform_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    transformed = (values - np.asarray(mean, dtype=np.float64)) / np.asarray(scale, dtype=np.float64)
    if transformed.ndim != 2 or transformed.shape[1] != INPUT_DIM or not np.isfinite(transformed).all():
        raise ValueError(f"Scaled summaries must be finite [N, {INPUT_DIM}].")
    return transformed.astype(np.float32)


def experiment_name(top_n: int, seed: int) -> str:
    return (
        f"summary_mlp_u1000_top{top_n}_summary20_final_train700_seed{seed}"
        "_trainzscore_h64x64_drop0p2"
    )


def load_family(path: str | Path) -> dict[str, Any]:
    family_path = Path(path)
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if family.get("schema_version") != "1.0":
        raise ValueError("Unsupported modern Summary MLP family schema.")
    if family.get("model_family") != MODEL_FAMILY:
        raise ValueError("Family must contain only the modern Summary MLP.")
    if family.get("top_n") != [500, 1000, 1500, 2000]:
        raise ValueError("Family must declare exactly Top500/1000/1500/2000.")
    if family.get("seeds") != [42, 123, 2025] or family.get("expected_jobs") != 12:
        raise ValueError("Family must declare exactly three controlled seeds and 12 jobs.")
    if family.get("summary_definition_version") != SUMMARY_DEFINITION_VERSION:
        raise ValueError("Summary definition does not match the shared extractor.")
    if family.get("snapshot_protocol") != "final" or family.get("feature_count") != INPUT_DIM:
        raise ValueError("Family must use the exact final-snapshot 20-feature protocol.")
    expected_fixed = {
        "feature_scaling": "training_rows_zscore_ddof0_zero_scale_to_one",
        "architecture": "20-64-64-1",
        "activation": "ReLU",
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "MSELoss",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_criterion": "minimum_validation_mse",
        "scheduler": "none",
        "gradient_clipping": "none",
        "num_workers": NUM_WORKERS,
    }
    if family.get("fixed_configuration") != expected_fixed:
        raise ValueError("Family fixed configuration does not match the predeclared implementation.")
    return family


def expand_family_jobs(family: Mapping[str, Any], repo_root: str | Path = ".") -> list[dict[str, Any]]:
    repo = Path(repo_root)
    source_path = repo / str(family["dataset_source_family"])
    source_family = load_classical_family(source_path)
    jobs: list[dict[str, Any]] = []
    for top_n in family["top_n"]:
        source = source_family["datasets"][str(top_n)]
        for seed in family["seeds"]:
            manifest_info = source["split_manifests"][str(seed)]
            manifest_path = repo / str(manifest_info["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            jobs.append({
                "model_family": MODEL_FAMILY,
                "top_n": int(top_n),
                "seed": int(seed),
                "experiment_name": experiment_name(int(top_n), int(seed)),
                "dataset_path": str(source["path"]),
                "dataset_sha256": str(source["sha256"]),
                "split_manifest_path": str(manifest_info["path"]),
                "split_manifest_sha256": str(manifest_info["sha256"]),
                "split_manifest_canonical_sha256": manifest.get("canonical_manifest_sha256"),
                "split_counts": dict(manifest["counts"]),
            })
    if len(jobs) != 12 or any(job["top_n"] == 750 for job in jobs):
        raise AssertionError("Expected exactly 12 core Summary MLP jobs and no Top750.")
    return jobs


def resolve_device(requested: str) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def build_run_config(job: Mapping[str, Any], repo_root: str | Path, device: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_name": str(job["experiment_name"]),
        "model_family": MODEL_FAMILY,
        "representation": "engineered_summary",
        "summary_definition_version": SUMMARY_DEFINITION_VERSION,
        "feature_names": SUMMARY_FEATURE_NAMES,
        "feature_count": INPUT_DIM,
        "snapshot_protocol": "final",
        "final_scale_factor": 1.0,
        "top_n": int(job["top_n"]),
        "seed": int(job["seed"]),
        "dataset_path": str(job["dataset_path"]),
        "dataset_sha256": str(job["dataset_sha256"]),
        "split_manifest_path": str(job["split_manifest_path"]),
        "split_manifest_sha256": str(job["split_manifest_sha256"]),
        "split_manifest_canonical_sha256": job.get("split_manifest_canonical_sha256"),
        "split_counts": dict(job["split_counts"]),
        "target": "Omega_m",
        "target_normalization": "none",
        "feature_scaling": "training_rows_zscore_ddof0_zero_scale_to_one",
        "architecture": {
            "version": ARCHITECTURE_VERSION,
            "input_dim": INPUT_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "activation": "ReLU",
            "dropout": DROPOUT,
            "output_dim": 1,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "MSELoss",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "checkpoint_criterion": "minimum_validation_mse",
            "scheduler": "none",
            "gradient_clipping": "none",
            "num_workers": NUM_WORKERS,
            "test_used_for_model_selection": False,
        },
        "device": device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        **git_provenance(Path(repo_root).resolve()),
    }


def required_artifacts() -> list[str]:
    return [
        "config.json", "metrics.json", "train_log.csv", "checkpoints/best_model.pt",
        *[f"predictions/{name}" for name in SPLIT_FILENAMES.values()],
    ]


def completion_state(experiment_dir: str | Path, expected_config: Mapping[str, Any]) -> str:
    path = Path(experiment_dir)
    if not path.exists():
        return "missing"
    if not path.is_dir():
        raise FileExistsError(f"Experiment destination is not a directory: {path}")
    present = [relative for relative in required_artifacts() if (path / relative).is_file()]
    if len(present) != len(required_artifacts()):
        raise RuntimeError(f"Partial experiment destination must not be overwritten: {path}")
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    identity_fields = (
        "experiment_name", "model_family", "representation", "summary_definition_version",
        "feature_names", "snapshot_protocol", "top_n", "seed", "dataset_sha256",
        "split_manifest_sha256", "target", "target_normalization", "feature_scaling",
        "architecture", "training", "device",
    )
    mismatches = [key for key in identity_fields if saved.get(key) != expected_config.get(key)]
    if mismatches:
        raise RuntimeError(f"Completed destination has incompatible config fields {mismatches}: {path}")
    return "complete"


def predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        values = model(tensor).detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite predictions.")
    return values.astype(np.float64)


def train_model(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[SummaryMLP, list[dict[str, Any]], int, float]:
    """Train using train data; choose the checkpoint using validation MSE only."""
    generator = seed_everything(seed)
    train_x = torch.as_tensor(training_features, dtype=torch.float32)
    train_y = torch.as_tensor(training_targets, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, generator=generator,
    )
    model = SummaryMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    log: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}.")
            optimizer.step()

        train_predictions = predict(model, training_features, device)
        validation_predictions = predict(model, validation_features, device)
        train_mse = float(np.mean((train_predictions - training_targets) ** 2, dtype=np.float64))
        validation_errors = validation_predictions - validation_targets
        validation_mse = float(np.mean(validation_errors ** 2, dtype=np.float64))
        validation_rmse = float(np.sqrt(validation_mse))
        validation_mae = float(np.mean(np.abs(validation_errors), dtype=np.float64))
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_counter = 0
        else:
            patience_counter += 1
        log.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_rmse": validation_rmse,
            "validation_mae": validation_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
            "patience_counter": patience_counter,
        })
        if patience_counter >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, log, best_epoch, best_validation_mse


def checkpoint_payload(
    model: SummaryMLP,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    best_epoch: int,
    best_validation_mse: float,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
        "scaler_mean": torch.as_tensor(scaler_mean, dtype=torch.float64),
        "scaler_scale": torch.as_tensor(scaler_scale, dtype=torch.float64),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_validation_mse),
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_names": SUMMARY_FEATURE_NAMES,
        "dataset_sha256": str(job["dataset_sha256"]),
        "split_manifest_sha256": str(job["split_manifest_sha256"]),
        "seed": int(job["seed"]),
        "top_n": int(job["top_n"]),
    }


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def model_from_checkpoint(checkpoint: Mapping[str, Any]) -> SummaryMLP:
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint architecture is incompatible.")
    model = SummaryMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_train_log(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    columns = [
        "epoch", "train_mse", "validation_mse", "validation_rmse",
        "validation_mae", "learning_rate", "improved", "patience_counter",
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_job(
    job: Mapping[str, Any],
    output_root: str | Path = "experiments",
    repo_root: str | Path = ".",
    requested_device: str = "auto",
    bound_data: tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> str:
    device = resolve_device(requested_device)
    config = build_run_config(job, repo_root, str(device))
    destination = Path(output_root) / str(job["experiment_name"])
    state = completion_state(destination, config)
    if state == "complete":
        return "skipped_complete"

    ids, features, targets, manifest = bound_data or load_bound_data(job)
    split_ids: dict[str, list[str]] = {}
    raw_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, key in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
        split_ids[split] = list(manifest[key])
        raw_arrays[split] = arrays_for_ids(split_ids[split], ids, features, targets)
    scaler_mean, scaler_scale = fit_feature_scaler(raw_arrays["train"][0])
    scaled_features = {
        split: transform_features(values[0], scaler_mean, scaler_scale)
        for split, values in raw_arrays.items()
    }
    model, train_log, best_epoch, best_validation_mse = train_model(
        scaled_features["train"], raw_arrays["train"][1],
        scaled_features["val"], raw_arrays["val"][1], int(job["seed"]), device,
    )
    # Test data is first evaluated only after validation has selected and restored the checkpoint.
    predictions = {split: predict(model, scaled_features[split], device) for split in ("train", "val", "test")}
    prediction_rows = {
        split: build_prediction_rows(split_ids[split], raw_arrays[split][1], predictions[split])
        for split in ("train", "val", "test")
    }
    metrics = {
        "train": compute_metrics(prediction_rows["train"]),
        "validation": compute_metrics(prediction_rows["val"]),
        "test": compute_metrics(prediction_rows["test"]),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
    }
    checkpoint = checkpoint_payload(
        model, scaler_mean, scaler_scale, best_epoch, best_validation_mse, job,
    )

    Path(output_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{job['experiment_name']}.tmp-", dir=output_root))
    try:
        write_json(config, temporary / "config.json")
        write_json(metrics, temporary / "metrics.json")
        write_train_log(train_log, temporary / "train_log.csv")
        checkpoint_path = temporary / "checkpoints" / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        for split, filename in SPLIT_FILENAMES.items():
            write_prediction_csv(prediction_rows[split], temporary / "predictions" / filename)
        os.rename(temporary, destination)
    except Exception:
        # Preserve the isolated temporary directory for diagnosis; never publish a partial destination.
        raise
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-spec", default="configs/experiment_families/u1000_modern_summary_mlp.json",
    )
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--execute", action="store_true", help="Execute jobs; default is a selection-only dry run.")
    parser.add_argument("--experiment-name", help="Restrict to one exact family experiment.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    family = load_family(args.family_spec)
    jobs = expand_family_jobs(family, repo_root)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
        if not jobs:
            raise ValueError(f"Unknown family experiment: {args.experiment_name}")
    print(f"Selected {len(jobs)} modern Summary MLP jobs.")
    for job in jobs:
        state = "unchecked"
        if not args.execute:
            state = "collision" if (Path(args.output_root) / job["experiment_name"]).exists() else "missing"
        print(
            f"summary_mlp seed={job['seed']:4d} top_n={job['top_n']:4d} "
            f"state={state:9s} {job['experiment_name']}"
        )
    if not args.execute:
        if any((Path(args.output_root) / job["experiment_name"]).exists() for job in jobs):
            raise RuntimeError("Dry run found one or more experiment destination collisions.")
        print("DRY RUN ONLY: no datasets loaded and no experiment artifacts created.")
        return

    summary_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for job in jobs:
        key = str(job["dataset_path"])
        if key not in summary_cache:
            summary_cache[key] = load_summary_data(job)
        bound = load_bound_data(job, summary_data=summary_cache[key])
        result = run_job(job, args.output_root, repo_root, args.device, bound)
        print(f"{result}: {job['experiment_name']}")


if __name__ == "__main__":
    main()
