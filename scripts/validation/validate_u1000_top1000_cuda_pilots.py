#!/usr/bin/env python3
"""Bounded CUDA forward/backward pilots for the approved sparse U1000 dataset.

This is deliberately not a trainer: it has no epoch loop, optimizer, checkpoint,
prediction export, or parameter update.  The only backward calls are one per
batch-size-1 model pilot; larger feasibility checks are forward-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.static_gcn import StaticGCNRegressor
from src.training.sparse_batch import (
    SPARSE_STORAGE,
    collate_sparse_static,
    collate_sparse_temporal,
    sparse_batch_to,
    temporal_sample_snapshots,
)
from src.training.split_manifest import load_split_manifest
from src.training.train_static_gcn import convert_temporal_final_snapshot_to_static


DATASET_SHA256 = "6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a"
TARGET_SHA256 = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
DEFAULT_DATASET = REPOSITORY_ROOT / (
    "data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt"
)
DEFAULT_MANIFEST = REPOSITORY_ROOT / (
    "configs/splits/u1000_top1000_none_k8_sparse/seed42_train700.json"
)
SEED = 42


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def current_rss_mib() -> float:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("Could not read current process RSS.")


def nvidia_snapshot() -> dict[str, Any]:
    fields = "name,driver_version,memory.total,memory.used,memory.free"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi result: {result.stdout!r}")
    return {
        "gpu_model": values[0], "driver_version": values[1],
        "total_mib": int(values[2]), "used_mib": int(values[3]),
        "free_mib": int(values[4]),
    }


def cuda_memory() -> dict[str, float]:
    gib = 1024.0 * 1024.0
    return {
        "allocated_mib": torch.cuda.memory_allocated() / gib,
        "reserved_mib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / gib,
    }


def tensor_shapes(value: Any, prefix: str = "") -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    if torch.is_tensor(value):
        found[prefix or "tensor"] = list(value.shape)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.update(tensor_shapes(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(tensor_shapes(child, f"{prefix}[{index}]"))
    return found


def assert_sparse_batch(batch: dict[str, Any], expected_graphs: int) -> None:
    if batch.get("graph_storage") != SPARSE_STORAGE:
        raise ValueError("Batch is not marked sparse_edge_index.")
    if int(batch.get("num_graphs", -1)) != expected_graphs:
        raise ValueError("Sparse batch graph count mismatch.")
    for name, shape in tensor_shapes(batch).items():
        if len(shape) >= 2 and shape[-2:] == [1000, 1000]:
            raise ValueError(f"Dense [N,N] tensor detected at {name}: {shape}")


def validate_source_sample(universe_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    if sample.get("graph_storage") != SPARSE_STORAGE:
        raise ValueError(f"{universe_id}: wrong graph storage")
    if sample.get("feature_names") != ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"]:
        raise ValueError(f"{universe_id}: feature protocol mismatch")
    if len(sample["Nodes_list"]) != 5 or len(sample["edge_index_list"]) != 5:
        raise ValueError(f"{universe_id}: expected five snapshots")
    shapes, edges, nodes = [], [], []
    for index, (x, edge_index, mask) in enumerate(zip(
        sample["Nodes_list"], sample["edge_index_list"], sample["mask_list"]
    )):
        if tuple(x.shape) != (1000, 7):
            raise ValueError(f"{universe_id} snapshot {index}: feature shape mismatch")
        real_nodes = int((mask.reshape(-1) > 0).sum())
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
            raise ValueError(f"{universe_id} snapshot {index}: invalid sparse edges")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= real_nodes:
            raise ValueError(f"{universe_id} snapshot {index}: edge reaches padding")
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"{universe_id} snapshot {index}: nonfinite feature")
        shapes.append(list(x.shape))
        edges.append(int(edge_index.shape[1]))
        nodes.append(real_nodes)
    target = sample["target"].float().reshape(1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"{universe_id}: nonfinite target")
    return {"input_shapes": shapes, "edge_counts": edges, "node_counts": nodes}


def gradients_result(model: nn.Module) -> dict[str, Any]:
    required = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    missing = [name for name, parameter in required if parameter.grad is None]
    nonfinite = [name for name, parameter in required
                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    if missing or nonfinite:
        raise ValueError(f"Gradient validation failed: missing={missing}, nonfinite={nonfinite}")
    return {"status": "PASS", "parameter_tensors": len(required), "missing": [], "nonfinite": []}


def timed_forward_backward(model: nn.Module, batch: dict[str, Any], target: torch.Tensor) -> dict[str, Any]:
    criterion = nn.MSELoss()
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    output = model(batch)
    loss = criterion(output, target)
    forward_end.record()
    backward_start.record()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize()
    if tuple(output.shape) != (target.shape[0], 1):
        raise ValueError(f"Wrong output shape: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite model output or loss.")
    return {
        "output_shape": list(output.shape), "output_finite": True,
        "loss": float(loss.detach()), "loss_finite": True,
        "forward_ms": forward_start.elapsed_time(forward_end),
        "backward_ms": backward_start.elapsed_time(backward_end),
        "gradients": gradients_result(model),
    }


def cleanup(*objects: Any) -> dict[str, Any]:
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"cuda": cuda_memory(), "host_rss_mib": current_rss_mib(), "nvidia_smi": nvidia_snapshot()}


def evolve_pilot(data: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    set_seed()
    source = validate_source_sample(ids[0], data[ids[0]])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = {"cuda": cuda_memory(), "host_rss_mib": current_rss_mib(), "nvidia_smi": nvidia_snapshot()}
    batch = collate_sparse_temporal([data[ids[0]]])
    assert_sparse_batch(batch, 1)
    batch = sparse_batch_to(batch, "cuda")
    target = data[ids[0]]["target"].float().reshape(1, 1).cuda()
    model = EvolveGCNHRegressor(
        node_features=7, hidden_dim=32, num_layers=2, dropout=0.2,
        graph_pooling="mean", temporal_pooling="mean", head_type="linear",
    ).cuda().train()
    measured = timed_forward_backward(model, batch, target)
    measured.update({
        "status": "PASS", "universe_ids": [ids[0]], "batch_size": 1,
        "snapshot_count": 5, "source": source, "batch_shapes": tensor_shapes(batch),
        "edge_counts": [int(graph["edge_index"].shape[1]) for graph in batch["snapshots"]],
        "node_counts": [int(graph["x"].shape[0]) for graph in batch["snapshots"]],
        "cuda": cuda_memory(), "host_rss_mib": current_rss_mib(), "before": before,
        "sparse_only": True, "device": str(next(model.parameters()).device),
    })
    del model, batch, target
    measured["after_cleanup"] = cleanup()
    return measured


def static_pilot(data: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    set_seed()
    universe_id = ids[0]
    temporal = data[universe_id]
    source = validate_source_sample(universe_id, temporal)
    static = convert_temporal_final_snapshot_to_static({universe_id: temporal})[universe_id]
    identity = {
        "x_same_storage": static["X"].data_ptr() == temporal["Nodes_list"][-1].data_ptr(),
        "edge_same_storage": static["edge_index"].data_ptr() == temporal["edge_index_list"][-1].data_ptr(),
        "mask_same_storage": static["mask"].data_ptr() == temporal["mask_list"][-1].data_ptr(),
        "snapshot_value": float(static["snapshot"]["snapshot_value"]),
    }
    if not all(identity[key] for key in ("x_same_storage", "edge_same_storage", "mask_same_storage")):
        raise ValueError("Static conversion is not the exact final-snapshot tensor view.")
    graph = temporal_sample_snapshots(temporal)[-1]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = {"cuda": cuda_memory(), "host_rss_mib": current_rss_mib(), "nvidia_smi": nvidia_snapshot()}
    batch = collate_sparse_static([graph])
    assert_sparse_batch(batch, 1)
    batch = sparse_batch_to(batch, "cuda")
    target = temporal["target"].float().reshape(1, 1).cuda()
    model = StaticGCNRegressor(
        node_features=7, hidden_dim=32, num_layers=3, dropout=0.2,
        graph_pooling="mean", conv_type="gcn",
    ).cuda().train()
    measured = timed_forward_backward(model, batch, target)
    measured.update({
        "status": "PASS", "universe_ids": [universe_id], "batch_size": 1,
        "snapshot_count": 1, "input_shapes": [source["input_shapes"][-1]],
        "edge_counts": [int(batch["edge_index"].shape[1])],
        "node_counts": [int(batch["x"].shape[0])], "batch_shapes": tensor_shapes(batch),
        "final_snapshot_identity": identity, "cuda": cuda_memory(),
        "host_rss_mib": current_rss_mib(), "before": before,
        "sparse_only": True, "device": str(next(model.parameters()).device),
    })
    del model, batch, target, graph, static
    measured["after_cleanup"] = cleanup()
    return measured


def feasibility(model_name: str, data: dict[str, Any], ids: list[str], sizes: list[int]) -> list[dict[str, Any]]:
    results = []
    for size in sizes:
        set_seed()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            if model_name == "evolve":
                batch = collate_sparse_temporal([data[item] for item in ids[:size]])
                model: nn.Module = EvolveGCNHRegressor(
                    7, 32, 2, 0.2, graph_pooling="mean", temporal_pooling="mean", head_type="linear"
                )
            else:
                graphs = [temporal_sample_snapshots(data[item])[-1] for item in ids[:size]]
                batch = collate_sparse_static(graphs)
                model = StaticGCNRegressor(7, 32, 3, 0.2, "mean", "gcn")
            assert_sparse_batch(batch, size)
            batch = sparse_batch_to(batch, "cuda")
            model = model.cuda().eval()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                output = model(batch)
            end.record()
            torch.cuda.synchronize()
            if tuple(output.shape) != (size, 1) or not bool(torch.isfinite(output).all()):
                raise ValueError("Forward-only feasibility output validation failed.")
            results.append({
                "model": model_name, "batch_size": size, "status": "PASS",
                "forward_only": True, "forward_ms": start.elapsed_time(end),
                "output_shape": list(output.shape), "cuda": cuda_memory(),
                "host_rss_mib": current_rss_mib(), "sparse_only": True,
            })
            del output, model, batch
            cleanup()
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"model": model_name, "batch_size": size, "status": "OOM", "error": str(exc)})
            cleanup()
            break
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only bounded CUDA pilots; never trains or updates parameters.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional JSON audit result path (use /tmp for transient output).")
    parser.add_argument("--skip-feasibility", action="store_true",
                        help="Run only the required batch-size-1 forward/backward pilots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("CUDA unavailable: run in the production host environment with an NVIDIA device visible.", file=sys.stderr)
        return 2
    started = perf_counter()
    before = nvidia_snapshot()
    try:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)
        manifest = load_split_manifest(args.manifest, list(data), DATASET_SHA256, expected_seed=SEED)
        binding = manifest.get("dataset_binding", {})
        if binding.get("target_table_sha256") != TARGET_SHA256:
            raise ValueError("Manifest target-table identity mismatch.")
        if manifest["counts"] != {"population": 1000, "train": 700, "val": 99, "test": 201, "unused": 0}:
            raise ValueError("Manifest population accounting mismatch.")
        train_ids = list(manifest["train_ids"])
        evolve = evolve_pilot(data, train_ids)
        static = static_pilot(data, train_ids)
        checks = {"evolve": [], "static": []}
        if not args.skip_feasibility:
            checks["evolve"] = feasibility("evolve", data, train_ids, [2, 4])
            checks["static"] = feasibility("static", data, train_ids, [4, 8])
        result = {
            "status": "PASS", "decision": "GO FOR TRAIN700 PILOTS",
            "seed": SEED, "manifest_counts": manifest["counts"],
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_identity": manifest["dataset_identity"],
            "target_identity": binding["target_table_sha256"],
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cuda_available": True, "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0), "nvidia_before": before,
            "evolve": evolve, "static": static, "feasibility": checks,
            "nvidia_after": nvidia_snapshot(), "elapsed_seconds": perf_counter() - started,
            "safety": {"epoch_loops": 0, "optimizer_steps": 0, "checkpoints": 0, "prediction_files": 0},
        }
    except Exception as exc:
        result = {"status": "FAIL", "decision": "NO-GO", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        if args.output_json:
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output_json:
        args.output_json.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
