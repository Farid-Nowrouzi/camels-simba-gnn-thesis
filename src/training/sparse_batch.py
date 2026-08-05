"""Dependency-free batching for CAMELS sparse static and temporal graphs."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


SPARSE_STORAGE = "sparse_edge_index"


def is_sparse_sample(sample: Dict[str, Any]) -> bool:
    return sample.get("graph_storage") == SPARSE_STORAGE or "edge_index_list" in sample


def _real_node_view(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_mask = mask.reshape(-1) > 0
    return x[flat_mask].float(), flat_mask


def collate_sparse_static(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one disconnected sparse graph batch without adjacency padding."""
    if not samples:
        raise ValueError("Cannot collate an empty sparse batch.")

    xs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    ptr = [0]
    has_weights = all(sample.get("edge_weight") is not None for sample in samples)

    for graph_index, sample in enumerate(samples):
        x = sample["x"].float()
        mask = sample.get("mask", torch.ones(x.shape[0], 1)).float()
        x_real, real_mask = _real_node_view(x, mask)
        real_indices = torch.nonzero(real_mask, as_tuple=False).reshape(-1)
        remap = torch.full((x.shape[0],), -1, dtype=torch.long)
        remap[real_indices] = torch.arange(real_indices.numel(), dtype=torch.long)

        edge_index = sample["edge_index"].long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
        if edge_index.numel():
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]:
                raise ValueError("edge_index contains an out-of-bounds node index.")
            mapped = remap[edge_index]
            if (mapped < 0).any():
                raise ValueError("Sparse edge references a padded node.")
            edges.append(mapped + ptr[-1])
        else:
            edges.append(edge_index)

        if has_weights:
            edge_weight = sample["edge_weight"].float().reshape(-1)
            if edge_weight.numel() != edge_index.shape[1]:
                raise ValueError("edge_weight length does not match edge_index.")
            weights.append(edge_weight)

        xs.append(x_real)
        masks.append(torch.ones((x_real.shape[0], 1), dtype=torch.float32))
        batch_parts.append(torch.full((x_real.shape[0],), graph_index, dtype=torch.long))
        ptr.append(ptr[-1] + x_real.shape[0])

    return {
        "graph_storage": SPARSE_STORAGE,
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "edge_weight": torch.cat(weights) if has_weights else None,
        "mask": torch.cat(masks, dim=0),
        "batch": torch.cat(batch_parts),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "num_graphs": len(samples),
    }


def temporal_sample_snapshots(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expose serialized temporal fields as per-snapshot sparse records by reference."""
    required = ["Nodes_list", "edge_index_list", "mask_list"]
    for key in required:
        if key not in sample:
            raise KeyError(f"Sparse temporal sample missing {key}.")
    count = len(sample["Nodes_list"])
    if len(sample["edge_index_list"]) != count or len(sample["mask_list"]) != count:
        raise ValueError("Sparse temporal field lengths do not match.")
    weight_list = sample.get("edge_weight_list")
    if weight_list is not None and len(weight_list) != count:
        raise ValueError("edge_weight_list length does not match snapshots.")
    result = []
    for index in range(count):
        result.append({
            "graph_storage": SPARSE_STORAGE,
            "x": sample["Nodes_list"][index],
            "edge_index": sample["edge_index_list"][index],
            "edge_weight": None if weight_list is None else weight_list[index],
            "mask": sample["mask_list"][index],
        })
    return result


def collate_sparse_temporal(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch equal-length temporal sequences while allowing variable node counts."""
    sequences = [temporal_sample_snapshots(sample) for sample in samples]
    timesteps = len(sequences[0])
    if timesteps == 0 or any(len(sequence) != timesteps for sequence in sequences):
        raise ValueError("Sparse temporal samples must have one equal non-zero snapshot count.")
    return {
        "graph_storage": SPARSE_STORAGE,
        "snapshots": [
            collate_sparse_static([sequence[t] for sequence in sequences])
            for t in range(timesteps)
        ],
        "num_graphs": len(samples),
        "num_timesteps": timesteps,
    }


def sparse_batch_to(batch: Any, device: torch.device | str) -> Any:
    """Recursively move a sparse batch while retaining non-tensor metadata."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: sparse_batch_to(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [sparse_batch_to(value, device) for value in batch]
    return batch
