"""Protected smoke, train, verify, and finalize workflow for frozen CNN3D."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.halo_voxelization import EXPERIMENT_1_SPEC, RAW7_FEATURE_NAMES, representation_fingerprint, voxelize_halo_features
from src.evaluation.baseline_common import SPLIT_FILENAMES, build_prediction_rows, compute_metrics, sha256_file, write_json, write_prediction_csv
from src.models.static_cnn3d import EXPECTED_PARAMETERS, StaticCNN3DRegressor, count_parameters
from src.training.split_manifest import load_split_manifest


REPO = Path(__file__).resolve().parents[2]
FAMILY_PATH = Path("configs/experiment_families/u1000_top1500_cnn3d.json")
SCIENTIFIC_FREEZE_PATH = Path("reports/experiment_registry/u1000_top1500_cnn3d_protocol_freeze.json")
IMPLEMENTATION_FREEZE_PATH = Path("reports/experiment_registry/u1000_top1500_cnn3d_implementation_freeze.json")
SCIENTIFIC_FREEZE_SHA256 = "ce934560d21b7823924f235787dd972cdb72abd8f0ebcf452ddd85db11e5b490"
SEEDS = (42, 123, 2025)
SMOKE_PATH = Path("outputs/smoke/u1000_top1500_cnn3d")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> torch.Generator:
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


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def experiment_name(seed: int) -> str:
    if seed not in SEEDS:
        raise ValueError("Seed must be 42, 123, or 2025.")
    return f"cnn3d_u1000_top1500_cic5_r32_final_train700_seed{seed}_fixedscale_c8x16x32_avg2_gap"


def load_family(repo: Path = REPO) -> dict[str, Any]:
    family = read_json(repo / FAMILY_PATH)
    if family.get("protocol_freeze_sha256") != SCIENTIFIC_FREEZE_SHA256:
        raise ValueError("Family config is not bound to the frozen scientific protocol.")
    if family.get("seeds") != list(SEEDS) or family.get("architecture", {}).get("expected_trainable_parameters") != EXPECTED_PARAMETERS:
        raise ValueError("Family config violates the frozen seed or parameter contract.")
    if family.get("voxelization", {}).get("representation_fingerprint") != representation_fingerprint(EXPERIMENT_1_SPEC):
        raise ValueError("Family config representation fingerprint mismatch.")
    return family


def verify_scientific_freeze(repo: Path = REPO) -> None:
    path = repo / SCIENTIFIC_FREEZE_PATH
    if not path.is_file() or sha256_file(path) != SCIENTIFIC_FREEZE_SHA256:
        raise ValueError("Scientific protocol freeze SHA-256 mismatch.")


def expand_family_jobs(family: Mapping[str, Any], repo: Path = REPO) -> list[dict[str, Any]]:
    jobs = []
    for seed in SEEDS:
        info = family["splits"][str(seed)]
        path = repo / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise ValueError("Exact split binding mismatch.")
        raw = read_json(path)
        manifest = load_split_manifest(path, raw["train_ids"] + raw["val_ids"] + raw["test_ids"], family["dataset"]["sha256"], seed)
        if [len(manifest[f"{name}_ids"]) for name in ("train", "val", "test")] != [700, 99, 201]:
            raise ValueError("Expected exact 700/99/201 split.")
        jobs.append({"seed": seed, "experiment_name": experiment_name(seed), "manifest": manifest,
                     "split_manifest_path": info["path"], "split_manifest_sha256": info["sha256"]})
    return jobs


def load_dataset_index(family: Mapping[str, Any], repo: Path = REPO) -> Mapping[str, Any]:
    path = repo / family["dataset"]["path"]
    if sha256_file(path) != family["dataset"]["sha256"]:
        raise ValueError("Dataset SHA-256 mismatch.")
    try:
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or len(data) != 1000 or set(data) != {f"LH_{index}" for index in range(1000)}:
        raise ValueError("Frozen dataset population contract mismatch.")
    return data


class VoxelHaloDataset(Dataset):
    """Partition-scoped final-snapshot dataset; only requested IDs expose features."""

    def __init__(self, data: Mapping[str, Any], universe_ids: Sequence[str]) -> None:
        self.data = data
        self.universe_ids = list(universe_ids)
        if not self.universe_ids or len(self.universe_ids) != len(set(self.universe_ids)):
            raise ValueError("Allowed universe IDs must be non-empty and unique.")
        if any(identifier not in data for identifier in self.universe_ids):
            raise ValueError("Allowed universe ID is absent from dataset.")

    def __len__(self) -> int:
        return len(self.universe_ids)

    def __getitem__(self, index: int):
        universe_id = self.universe_ids[index]
        sample = self.data[universe_id]
        if tuple(sample.get("feature_names", ())) != RAW7_FEATURE_NAMES:
            raise ValueError(f"{universe_id}: frozen raw7 feature order mismatch.")
        if float(sample["snapshots"][-1]["snapshot_value"]) != 1.0:
            raise ValueError(f"{universe_id}: final snapshot must be a=1.0.")
        x = torch.as_tensor(sample["Nodes_list"][-1], dtype=torch.float32)
        mask = torch.as_tensor(sample["mask_list"][-1]).reshape(-1).bool()
        if tuple(x.shape) != (1500, 7) or tuple(mask.shape) != (1500,) or int(mask.sum()) != 1500:
            raise ValueError(f"{universe_id}: expected exactly 1500 valid raw7 halos.")
        target = torch.as_tensor(sample["target"], dtype=torch.float32).reshape(1)
        if not torch.isfinite(target).all():
            raise ValueError(f"{universe_id}: target must be finite.")
        return universe_id, voxelize_halo_features(x, mask), target


def make_loader(data: Mapping[str, Any], ids: Sequence[str], shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(VoxelHaloDataset(data, ids), batch_size=8, shuffle=shuffle, num_workers=0, generator=generator)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    ids: list[str] = []
    truth: list[float] = []
    predicted: list[float] = []
    for universe_ids, grids, targets in loader:
        values = model(grids.to(device)).cpu().numpy().reshape(-1)
        ids.extend(universe_ids)
        truth.extend(targets.numpy().reshape(-1))
        predicted.extend(values)
    if not np.isfinite(np.asarray(predicted)).all():
        raise ValueError("Model produced non-finite predictions.")
    return build_prediction_rows(ids, truth, predicted)


def train_model(data: Mapping[str, Any], train_ids: Sequence[str], val_ids: Sequence[str], seed: int,
                device: torch.device, *, epochs: int = 300) -> tuple[nn.Module, list[dict[str, Any]], int, float]:
    generator = seed_everything(seed)
    train_loader = make_loader(data, train_ids, True, generator)
    train_eval_loader = make_loader(data, train_ids, False)
    val_loader = make_loader(data, val_ids, False)
    model = StaticCNN3DRegressor().to(device)
    if count_parameters(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen CNN parameter-count contract violated.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_state, best_mse, best_epoch, stale, log = None, float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for _, grids, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(grids.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_metrics = compute_metrics(collect_predictions(model, train_eval_loader, device))
        val_metrics = compute_metrics(collect_predictions(model, val_loader, device))
        value = float(val_metrics["mse"])
        scheduler.step(value)
        improved = value < best_mse
        if improved:
            best_state, best_mse, best_epoch, stale = copy.deepcopy(model.state_dict()), value, epoch, 0
        else:
            stale += 1
        log.append({"epoch": epoch, "train_mse": train_metrics["mse"], "validation_mse": value,
                    "validation_mae": val_metrics["mae"], "validation_rmse": val_metrics["rmse"],
                    "learning_rate": optimizer.param_groups[0]["lr"], "improved": improved, "patience_counter": stale})
        if stale >= 40:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, log, best_epoch, best_mse


def write_train_log(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_config(family: Mapping[str, Any], job: Mapping[str, Any], implementation_freeze_sha256: str | None,
               device: torch.device, *, smoke: bool) -> dict[str, Any]:
    return {**family, "seed": job["seed"], "experiment_name": job["experiment_name"],
            "split_manifest_path": job["split_manifest_path"], "split_manifest_sha256": job["split_manifest_sha256"],
            "family_config_sha256": sha256_file(REPO / FAMILY_PATH),
            "implementation_freeze_sha256": implementation_freeze_sha256, "device": str(device),
            "scientific_run": not smoke, "smoke_protocol": {"seed": 42, "train_count": 16, "val_count": 8, "epochs": 2} if smoke else None,
            "trainable_parameters": EXPECTED_PARAMETERS}


def require_implementation_freeze(expected_hash: str, repo: Path = REPO) -> dict[str, Any]:
    path = repo / IMPLEMENTATION_FREEZE_PATH
    if not expected_hash or not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError("Implementation/source freeze SHA-256 mismatch or missing approval binding.")
    freeze = read_json(path)
    if freeze.get("scientific_protocol_freeze", {}).get("sha256") != SCIENTIFIC_FREEZE_SHA256:
        raise ValueError("Implementation freeze is bound to the wrong scientific protocol.")
    for record in freeze.get("source_files", []) + freeze.get("tests", []):
        if sha256_file(repo / record["path"]) != record["sha256"]:
            raise ValueError(f"Implementation source SHA-256 mismatch: {record['path']}")
    return freeze


def _partition_rows(model: nn.Module, data: Mapping[str, Any], manifest: Mapping[str, Any], partition: str,
                    device: torch.device, stage: str) -> list[dict[str, Any]]:
    allowed = {"smoke": {"train", "val"}, "train": {"train", "val"}, "finalize": {"test"}}
    if partition not in allowed[stage]:
        raise ValueError(f"{stage} cannot access {partition} features.")
    return collect_predictions(model, make_loader(data, manifest[f"{partition}_ids"], False), device)


def train_run(seed: int, implementation_freeze_sha256: str | None, device: torch.device, *, smoke: bool = False) -> Path:
    verify_scientific_freeze()
    if not smoke:
        require_implementation_freeze(str(implementation_freeze_sha256))
    family = load_family()
    job = next(item for item in expand_family_jobs(family) if item["seed"] == seed)
    manifest = job["manifest"]
    if smoke:
        if seed != 42:
            raise ValueError("Smoke uses seed 42 only.")
        manifest = {"train_ids": manifest["train_ids"][:16], "val_ids": manifest["val_ids"][:8]}
        destination = REPO / SMOKE_PATH
    else:
        destination = REPO / "experiments" / job["experiment_name"]
    destination.mkdir(parents=True, exist_ok=False)
    config = run_config(family, job, implementation_freeze_sha256, device, smoke=smoke)
    write_json(config, destination / "config.json")
    write_json({"state": "training", "scientific_run": not smoke, "test_status": "forbidden" if smoke else "pending"}, destination / "run_metadata.json")
    data = load_dataset_index(family)
    model, log, epoch, best_mse = train_model(data, manifest["train_ids"], manifest["val_ids"], seed, device, epochs=2 if smoke else 300)
    checkpoint = destination / "checkpoints" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "best_epoch": epoch, "best_validation_mse": best_mse}, checkpoint)
    write_train_log(log, destination / "train_log.csv")
    metrics: dict[str, Any] = {"best_epoch": epoch, "best_validation_mse": best_mse, "test_status": "forbidden" if smoke else "pending"}
    for partition in ("train", "val"):
        rows = _partition_rows(model, data, manifest, partition, device, "smoke" if smoke else "train")
        write_prediction_csv(rows, destination / "predictions" / SPLIT_FILENAMES[partition])
        metrics["validation" if partition == "val" else "train"] = compute_metrics(rows)
    write_json(metrics, destination / "metrics.json")
    hashes = {str(item.relative_to(destination)): sha256_file(item) for item in (destination / "config.json", checkpoint, destination / "train_log.csv", destination / "predictions" / "train_predictions.csv", destination / "predictions" / "val_predictions.csv")}
    write_json({"state": "smoke_complete" if smoke else "trained", "scientific_run": not smoke,
                "test_status": "forbidden" if smoke else "pending", "training_artifact_hashes": hashes,
                "completed_training_utc": datetime.now(timezone.utc).isoformat()}, destination / "run_metadata.json")
    return destination


def verify_all_trained(implementation_freeze_sha256: str, repo: Path = REPO) -> list[tuple[dict[str, Any], Path, dict[str, Any], nn.Module]]:
    require_implementation_freeze(implementation_freeze_sha256, repo)
    family = load_family(repo)
    records = []
    for job in expand_family_jobs(family, repo):
        destination = repo / "experiments" / job["experiment_name"]
        metadata_path = destination / "run_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError("Finalize requires completed training for all three seeds.")
        metadata = read_json(metadata_path)
        if metadata.get("state") not in {"trained", "finalized"} or metadata.get("scientific_run") is not True:
            raise RuntimeError("Finalize requires completed scientific training for all three seeds.")
        required = ("config.json", "checkpoints/best_model.pt", "train_log.csv", "predictions/train_predictions.csv", "predictions/val_predictions.csv")
        for relative in required:
            if not (destination / relative).is_file() or sha256_file(destination / relative) != metadata.get("training_artifact_hashes", {}).get(relative):
                raise ValueError(f"Training artifact hash mismatch: {destination / relative}")
        config = read_json(destination / "config.json")
        if config.get("implementation_freeze_sha256") != implementation_freeze_sha256 or config.get("trainable_parameters") != EXPECTED_PARAMETERS:
            raise ValueError("Run configuration does not match frozen implementation.")
        checkpoint = torch.load(destination / "checkpoints/best_model.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("config") != config:
            raise ValueError("Checkpoint configuration mismatch.")
        model = StaticCNN3DRegressor(); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if count_parameters(model) != EXPECTED_PARAMETERS:
            raise ValueError("Checkpoint parameter-count mismatch.")
        records.append((job, destination, metadata, model))
    return records


def finalize(implementation_freeze_sha256: str, device: torch.device) -> None:
    records = verify_all_trained(implementation_freeze_sha256)
    family = load_family()
    for job, destination, metadata, model in records:
        marker = destination / "test_evaluation_started.json"
        prediction = destination / "predictions" / "test_predictions.csv"
        if marker.exists() or prediction.exists():
            raise RuntimeError("Test evaluation already attempted; automatic retry is forbidden.")
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"implementation_freeze_sha256": implementation_freeze_sha256}, handle, sort_keys=True)
        data = load_dataset_index(family)
        rows = _partition_rows(model.to(device), data, job["manifest"], "test", device, "finalize")
        write_prediction_csv(rows, prediction)
        metrics = read_json(destination / "metrics.json")
        metrics.update(test=compute_metrics(rows), test_status="evaluated_once")
        write_json(metrics, destination / "metrics.json")
        metadata.update(state="finalized", test_status="evaluated_once")
        write_json(metadata, destination / "run_metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "verify", "finalize"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    if args.mode == "smoke":
        if args.seed not in (None, 42): parser.error("Smoke uses seed 42 only.")
        train_run(42, None, resolve_device(args.device), smoke=True)
    else:
        if not args.freeze_sha256: parser.error("Production stages require --freeze-sha256.")
        if args.mode == "train":
            if args.seed is None: parser.error("Train requires --seed.")
            train_run(args.seed, args.freeze_sha256, resolve_device(args.device))
        elif args.mode == "verify":
            if args.seed is not None: parser.error("Verify does not select a seed.")
            verify_all_trained(args.freeze_sha256)
        else:
            if args.seed is not None: parser.error("Finalize does not select a seed.")
            finalize(args.freeze_sha256, resolve_device(args.device))


if __name__ == "__main__":
    main()
