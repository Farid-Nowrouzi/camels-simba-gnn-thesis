"""Run the controlled modern U1000 final-snapshot raw-halo DeepSets family."""

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
from torch.utils.data import DataLoader, Dataset

from src.evaluation.baseline_common import (
    SPLIT_FILENAMES, build_prediction_rows, compute_metrics, git_provenance,
    sha256_file, write_json, write_prediction_csv,
)
from src.models.deepsets import FEATURE_NAMES, DeepSetsRegressor, count_parameters
from src.training.split_manifest import load_split_manifest


MODEL_FAMILY = "deepsets"
ARCHITECTURE_VERSION = "deepsets_raw7_phi32x3_masked_mean_static_head_v1"
INPUT_DIM = 7
HIDDEN_DIM = 32
DROPOUT = 0.2
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 40
GRAD_CLIP_NORM = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
MIN_LR = 1e-6
NUM_WORKERS = 0


def seed_everything(seed: int) -> torch.Generator:
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


def experiment_name(top_n: int, seed: int) -> str:
    return f"deepsets_u1000_top{top_n}_raw7_final_train700_seed{seed}_none_h32_phi3_mean_statichead"


def load_family(path: str | Path) -> dict[str, Any]:
    family = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fixed = {
        "architecture": "7-32-32-32_masked_mean_32-32-16-1",
        "activation": "ReLU", "dropout": DROPOUT, "layer_normalization": True,
        "same_width_residuals": True, "pooling": "masked_mean", "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "loss": "MSELoss",
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "checkpoint_criterion": "minimum_validation_mse",
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": SCHEDULER_FACTOR,
                      "patience": SCHEDULER_PATIENCE, "min_lr": MIN_LR},
        "gradient_clipping": GRAD_CLIP_NORM, "num_workers": NUM_WORKERS, "device": "cuda",
    }
    required = {
        "schema_version": "1.0", "model_family": MODEL_FAMILY, "model": "DeepSets",
        "representation": "raw_halo_set", "uses_graph_edges": False,
        "snapshot_protocol": "final", "final_scale_factor": 1.0,
        "feature_names": FEATURE_NAMES, "feature_normalization": "none",
        "target": "Omega_m", "target_normalization": "none",
        "top_n": [500, 1000, 1500, 2000], "seeds": [42, 123, 2025], "expected_jobs": 12,
        "fixed_configuration": expected_fixed,
    }
    mismatches = [key for key, value in required.items() if family.get(key) != value]
    if mismatches:
        raise ValueError(f"DeepSets family specification mismatch: {mismatches}")
    return family


def expand_family_jobs(family: Mapping[str, Any], repo_root: str | Path = ".") -> list[dict[str, Any]]:
    repo = Path(repo_root)
    source = json.loads((repo / str(family["dataset_source_family"])).read_text(encoding="utf-8"))
    jobs = []
    for top_n in family["top_n"]:
        dataset = source["datasets"][str(top_n)]
        for seed in family["seeds"]:
            manifest_info = dataset["split_manifests"][str(seed)]
            manifest = json.loads((repo / manifest_info["path"]).read_text(encoding="utf-8"))
            jobs.append({
                "model_family": MODEL_FAMILY, "top_n": int(top_n), "seed": int(seed),
                "experiment_name": experiment_name(int(top_n), int(seed)),
                "dataset_path": dataset["path"], "dataset_sha256": dataset["sha256"],
                "split_manifest_path": manifest_info["path"],
                "split_manifest_sha256": manifest_info["sha256"],
                "split_manifest_canonical_sha256": manifest.get("canonical_manifest_sha256"),
                "split_counts": dict(manifest["counts"]),
            })
    if len(jobs) != 12 or any(job["top_n"] == 750 for job in jobs):
        raise AssertionError("Expected exactly 12 DeepSets jobs and no Top750.")
    return jobs


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def build_run_config(job: Mapping[str, Any], repo_root: str | Path, device: str) -> dict[str, Any]:
    model = DeepSetsRegressor()
    return {
        "schema_version": "1.0", "experiment_name": str(job["experiment_name"]),
        "model_family": MODEL_FAMILY, "model": "DeepSets", "representation": "raw_halo_set",
        "uses_graph_edges": False, "snapshot_protocol": "final", "final_scale_factor": 1.0,
        "top_n": int(job["top_n"]), "seed": int(job["seed"]),
        "dataset_path": str(job["dataset_path"]), "dataset_sha256": str(job["dataset_sha256"]),
        "split_manifest_path": str(job["split_manifest_path"]),
        "split_manifest_sha256": str(job["split_manifest_sha256"]),
        "split_manifest_canonical_sha256": job.get("split_manifest_canonical_sha256"),
        "split_counts": dict(job["split_counts"]), "feature_names": FEATURE_NAMES,
        "feature_count": INPUT_DIM, "feature_normalization": "none", "target": "Omega_m",
        "target_normalization": "none", "pooling": "masked_mean",
        "architecture": {
            "version": ARCHITECTURE_VERSION, "input_dim": INPUT_DIM,
            "per_halo_encoder_dims": [7, 32, 32, 32], "activation": "ReLU",
            "dropout": DROPOUT, "layer_normalization": True, "same_width_residuals": True,
            "pooling": "masked_mean", "regression_head_dims": [32, 32, 16, 1],
        },
        "trainable_parameters": count_parameters(model),
        "training": {
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "loss": "MSELoss", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "checkpoint_criterion": "minimum_validation_mse",
            "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": SCHEDULER_FACTOR,
                          "patience": SCHEDULER_PATIENCE, "min_lr": MIN_LR},
            "gradient_clipping": GRAD_CLIP_NORM, "num_workers": NUM_WORKERS,
            "test_used_for_training": False, "test_used_for_model_selection": False,
        },
        "device": device, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        **git_provenance(Path(repo_root).resolve()),
    }


def required_artifacts() -> list[str]:
    return ["config.json", "metrics.json", "train_log.csv", "checkpoints/best_model.pt",
            *[f"predictions/{name}" for name in SPLIT_FILENAMES.values()]]


def completion_state(experiment_dir: str | Path, expected_config: Mapping[str, Any]) -> str:
    path = Path(experiment_dir)
    if not path.exists():
        return "missing"
    if not path.is_dir():
        raise FileExistsError(f"Experiment destination is not a directory: {path}")
    present = [name for name in required_artifacts() if (path / name).is_file()]
    if len(present) != len(required_artifacts()):
        raise RuntimeError(f"Partial experiment destination must not be overwritten: {path}")
    saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
    identity = ("experiment_name", "model_family", "representation", "uses_graph_edges",
                "snapshot_protocol", "final_scale_factor", "top_n", "seed", "dataset_sha256",
                "split_manifest_sha256", "feature_names", "feature_normalization", "target",
                "target_normalization", "pooling", "architecture", "training", "device")
    mismatches = [key for key in identity if saved.get(key) != expected_config.get(key)]
    if mismatches:
        raise RuntimeError(f"Completed destination has incompatible config fields {mismatches}: {path}")
    return "complete"


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or not data:
        raise TypeError("Processed temporal dataset must be a non-empty dictionary.")
    return data


def _snapshot_value(sample: Mapping[str, Any]) -> float:
    snapshots = sample.get("snapshots")
    if not isinstance(snapshots, Sequence) or not snapshots:
        raise ValueError("Sample is missing snapshot scale factors.")
    value = snapshots[-1]
    if isinstance(value, Mapping):
        for key in ("snapshot_value", "scale_factor", "a", "snapshot"):
            if key in value:
                value = value[key]
                break
    return float(value)


def validate_raw_dataset(data: Mapping[str, Mapping[str, Any]], top_n: int) -> None:
    if len(data) != 1000:
        raise ValueError(f"Expected 1000 universes, got {len(data)}")
    for universe_id, sample in data.items():
        for key in ("Nodes_list", "mask_list", "target", "snapshots"):
            if key not in sample:
                raise KeyError(f"{universe_id}: missing {key}")
        x = torch.as_tensor(sample["Nodes_list"][-1])
        mask = torch.as_tensor(sample["mask_list"][-1]).squeeze(-1)
        if list(sample.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError(f"{universe_id}: raw feature names/order mismatch")
        if sample.get("normalization") != "none":
            raise ValueError(f"{universe_id}: feature normalization must be none")
        if tuple(x.shape) != (top_n, INPUT_DIM) or tuple(mask.shape) != (top_n,):
            raise ValueError(f"{universe_id}: invalid final shapes x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        if not torch.isfinite(x).all() or not torch.isfinite(torch.as_tensor(sample["target"]).float()).all():
            raise ValueError(f"{universe_id}: non-finite raw values or target")
        if not torch.all((mask == 0) | (mask == 1)) or not bool(mask.bool().any()):
            raise ValueError(f"{universe_id}: invalid or all-false mask")
        if not np.isclose(_snapshot_value(sample), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"{universe_id}: final snapshot is not a=1.0")


def load_bound_data(job: Mapping[str, Any], data: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(job["dataset_path"]) != job["dataset_sha256"]:
        raise ValueError(f"Dataset SHA-256 mismatch: {job['dataset_path']}")
    loaded = data if data is not None else _torch_load(job["dataset_path"])
    validate_raw_dataset(loaded, int(job["top_n"]))
    if sha256_file(job["split_manifest_path"]) != job["split_manifest_sha256"]:
        raise ValueError(f"Split manifest SHA-256 mismatch: {job['split_manifest_path']}")
    manifest = load_split_manifest(
        job["split_manifest_path"], list(loaded), str(job["dataset_sha256"]), int(job["seed"]),
    )
    expected = {"train": 700, "val": 99, "test": 201, "unused": 0}
    if any(int(manifest["counts"].get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("Manifest does not have exact 700/99/201/0 counts.")
    return loaded, manifest


class RawHaloSetDataset(Dataset):
    def __init__(self, data: Mapping[str, Mapping[str, Any]], universe_ids: Sequence[str]) -> None:
        self.data, self.universe_ids = data, list(universe_ids)

    def __len__(self) -> int:
        return len(self.universe_ids)

    def __getitem__(self, index: int):
        universe_id = self.universe_ids[index]
        sample = self.data[universe_id]
        x = torch.as_tensor(sample["Nodes_list"][-1], dtype=torch.float32)
        mask = torch.as_tensor(sample["mask_list"][-1]).squeeze(-1).bool()
        target = torch.as_tensor(sample["target"], dtype=torch.float32).reshape(1)
        return universe_id, x, mask, target


def make_loader(data: Mapping[str, Any], ids: Sequence[str], shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(RawHaloSetDataset(data, ids), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, generator=generator)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    all_ids, truths, predictions = [], [], []
    for universe_ids, x, mask, target in loader:
        prediction = model(x.to(device), mask.to(device)).cpu().numpy().reshape(-1)
        all_ids.extend(universe_ids)
        truths.extend(target.numpy().reshape(-1))
        predictions.extend(prediction)
    values = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite predictions.")
    return build_prediction_rows(all_ids, truths, values)


def train_model(data: Mapping[str, Any], train_ids: Sequence[str], val_ids: Sequence[str], seed: int,
                device: torch.device) -> tuple[DeepSetsRegressor, list[dict[str, Any]], int, float]:
    generator = seed_everything(seed)
    train_loader = make_loader(data, train_ids, True, generator)
    train_eval_loader = make_loader(data, train_ids, False)
    val_loader = make_loader(data, val_ids, False)
    model = DeepSetsRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )
    criterion = nn.MSELoss()
    best_state, best_mse, best_epoch, stale, log = None, float("inf"), 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for _, x, mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device), mask.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        train_rows = collect_predictions(model, train_eval_loader, device)
        val_rows = collect_predictions(model, val_loader, device)
        train_metrics, val_metrics = compute_metrics(train_rows), compute_metrics(val_rows)
        val_mse = float(val_metrics["mse"])
        scheduler.step(val_mse)
        improved = val_mse < best_mse
        if improved:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        log.append({"epoch": epoch, "train_mse": train_metrics["mse"],
                    "validation_mse": val_mse, "validation_rmse": val_metrics["rmse"],
                    "validation_mae": val_metrics["mae"], "learning_rate": optimizer.param_groups[0]["lr"],
                    "improved": improved, "patience_counter": stale})
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, log, best_epoch, best_mse


def checkpoint_payload(model: DeepSetsRegressor, best_epoch: int, best_validation_mse: float,
                       job: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_state_dict": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            "best_epoch": int(best_epoch), "best_validation_mse": float(best_validation_mse),
            "architecture_version": ARCHITECTURE_VERSION, "feature_names": FEATURE_NAMES,
            "dataset_sha256": str(job["dataset_sha256"]),
            "split_manifest_sha256": str(job["split_manifest_sha256"]),
            "seed": int(job["seed"]), "top_n": int(job["top_n"])}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_from_checkpoint(checkpoint: Mapping[str, Any]) -> DeepSetsRegressor:
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint architecture is incompatible.")
    model = DeepSetsRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_train_log(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    columns = ["epoch", "train_mse", "validation_mse", "validation_rmse", "validation_mae",
               "learning_rate", "improved", "patience_counter"]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def run_job(job: Mapping[str, Any], output_root: str | Path = "experiments", repo_root: str | Path = ".",
            requested_device: str = "auto", bound_data=None) -> str:
    device = resolve_device(requested_device)
    config = build_run_config(job, repo_root, str(device))
    destination = Path(output_root) / str(job["experiment_name"])
    if completion_state(destination, config) == "complete":
        return "skipped_complete"
    data, manifest = bound_data or load_bound_data(job)
    model, log, best_epoch, best_mse = train_model(
        data, manifest["train_ids"], manifest["val_ids"], int(job["seed"]), device,
    )
    # Test is touched only after validation-only model selection and best-state restoration.
    loaders = {split: make_loader(data, manifest[key], False) for split, key in
               (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids"))}
    rows = {split: collect_predictions(model, loader, device) for split, loader in loaders.items()}
    metrics = {"train": compute_metrics(rows["train"]), "validation": compute_metrics(rows["val"]),
               "test": compute_metrics(rows["test"]), "best_epoch": best_epoch,
               "best_validation_mse": best_mse}
    payload = checkpoint_payload(model, best_epoch, best_mse, job)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{job['experiment_name']}.tmp-", dir=output_root))
    try:
        write_json(config, temporary / "config.json"); write_json(metrics, temporary / "metrics.json")
        write_train_log(log, temporary / "train_log.csv")
        checkpoint = temporary / "checkpoints/best_model.pt"; checkpoint.parent.mkdir(parents=True)
        torch.save(payload, checkpoint)
        for split, filename in SPLIT_FILENAMES.items():
            write_prediction_csv(rows[split], temporary / "predictions" / filename)
        os.rename(temporary, destination)
    except Exception:
        raise
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", default="configs/experiment_families/u1000_modern_deepsets.json")
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true", help="Execute jobs; default is selection-only dry run.")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    jobs = expand_family_jobs(load_family(args.family_spec), repo)
    if args.experiment_name:
        jobs = [job for job in jobs if job["experiment_name"] == args.experiment_name]
        if not jobs:
            raise ValueError(f"Unknown family experiment: {args.experiment_name}")
    print(f"Selected {len(jobs)} modern DeepSets jobs.")
    for job in jobs:
        destination = Path(args.output_root) / job["experiment_name"]
        state = "unchecked" if args.execute else ("collision" if destination.exists() else "missing")
        print(f"deepsets seed={job['seed']:4d} top_n={job['top_n']:4d} state={state:9s} {job['experiment_name']}")
    if not args.execute:
        if any((Path(args.output_root) / job["experiment_name"]).exists() for job in jobs):
            raise RuntimeError("Dry run found one or more experiment destination collisions.")
        print("DRY RUN ONLY: no datasets loaded and no experiment artifacts created.")
        return
    data_cache = {}
    for job in jobs:
        key = str(job["dataset_path"])
        if key not in data_cache:
            data_cache[key] = _torch_load(key)
        result = run_job(job, args.output_root, repo, args.device, load_bound_data(job, data_cache[key]))
        print(f"{result}: {job['experiment_name']}")


if __name__ == "__main__":
    main()
