#!/usr/bin/env python3
"""Bounded Top1500 CUDA pilots at exact production batch sizes; no optimizer or epoch loop."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.static_gcn import StaticGCNRegressor
from src.training.sparse_batch import (collate_sparse_static, collate_sparse_temporal,
                                       sparse_batch_to, temporal_sample_snapshots)
from src.training.split_manifest import load_split_manifest

DATASET = ROOT / "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
MANIFEST = ROOT / "configs/splits/u1000_top1500_none_k8_sparse/seed42_train700.json"
OUTPUT = ROOT / "reports/experiment_registry/u1000_top1500_cuda_pilot_result.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def run_model(model_key: str, data: dict, ids: list[str], batch_size: int) -> dict:
    samples = [data[item] for item in ids[:batch_size]]
    for sample in samples:
        for features in sample["Nodes_list"]:
            if tuple(features.shape) != (1500, 7):
                raise ValueError("actual Top1500 feature shape mismatch")
    if model_key == "evolve":
        batch = collate_sparse_temporal(samples)
        model: nn.Module = EvolveGCNHRegressor(7, 32, 2, 0.2, graph_pooling="mean",
                                               temporal_pooling="mean", head_type="linear")
        snapshots = 5
    else:
        graphs = [temporal_sample_snapshots(sample)[-1] for sample in samples]
        batch = collate_sparse_static(graphs)
        model = StaticGCNRegressor(7, 32, 3, 0.2, "mean", "gcn")
        snapshots = 1
        for sample, graph in zip(samples, graphs):
            if graph["x"].data_ptr() != sample["Nodes_list"][-1].data_ptr():
                raise ValueError("Static graph is not the exact final temporal snapshot")
    if batch.get("graph_storage") != "sparse_edge_index":
        raise ValueError("batch lost sparse storage identity")
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[-2:]) == (1500, 1500):
            raise ValueError("dense adjacency conversion detected")
    targets = torch.stack([sample["target"].float().reshape(1) for sample in samples]).cuda()
    batch = sparse_batch_to(batch, "cuda"); model = model.cuda().train()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    start_f = torch.cuda.Event(enable_timing=True); end_f = torch.cuda.Event(enable_timing=True)
    start_b = torch.cuda.Event(enable_timing=True); end_b = torch.cuda.Event(enable_timing=True)
    start_f.record(); predictions = model(batch); loss = nn.MSELoss()(predictions, targets); end_f.record()
    start_b.record(); loss.backward(); end_b.record(); torch.cuda.synchronize()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite prediction or MSE loss")
    if not gradients or any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError("missing/nonfinite gradients")
    if not any(bool((gradient != 0).any()) for gradient in gradients if gradient is not None):
        raise ValueError("all gradients are zero")
    result = {"status": "PASS", "model": model_key, "batch_size": batch_size,
              "snapshots": snapshots, "prediction_shape": list(predictions.shape),
              "loss": float(loss.detach()), "forward_ms": start_f.elapsed_time(end_f),
              "backward_ms": start_b.elapsed_time(end_b),
              "peak_allocated_cuda_mib": torch.cuda.max_memory_allocated()/1024**2,
              "peak_reserved_cuda_mib": torch.cuda.max_memory_reserved()/1024**2,
              "host_rss_mib": rss_mib(), "sparse_only": True,
              "gradient_tensors": len(gradients), "nonzero_gradients": True}
    del predictions, loss, targets, batch, model, samples
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    result["post_cleanup_allocated_cuda_mib"] = torch.cuda.memory_allocated()/1024**2
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-json", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.dataset.is_file():
        print("Top1500 dataset is missing; CUDA pilot correctly gated.", file=sys.stderr); return 3
    if not torch.cuda.is_available():
        print("CUDA unavailable.", file=sys.stderr); return 2
    try:
        data = torch.load(args.dataset, map_location="cpu", weights_only=False)
        manifest = load_split_manifest(args.manifest, list(data), sha(args.dataset), expected_seed=42)
        ids = manifest["train_ids"]
        result = {"status": "PASS", "dataset_sha256": sha(args.dataset), "manifest_sha256": sha(args.manifest),
                  "cuda_device": torch.cuda.get_device_name(0), "evolve": run_model("evolve", data, ids, 4),
                  "static": run_model("static", data, ids, 8),
                  "safety": {"epoch_loops": 0, "optimizer_steps": 0, "checkpoints": 0, "predictions": 0}}
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "FAIL", "reason": "OOM at required production batch size; no fallback used", "error": str(exc)}
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
