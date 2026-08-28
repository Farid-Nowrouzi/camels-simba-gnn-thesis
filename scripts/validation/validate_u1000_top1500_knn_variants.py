#!/usr/bin/env python3
"""Read-only validator for matched U1000 Top1500 kNN variants (k=4,6,12)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import validate_u1000_top1000_sparse_dataset as base
from src.data.camels_graph_utils import (
    build_sparse_knn_edge_index,
    preprocessing_version_for_config,
)
from src.data.source_manifest import verify_full_source_manifest


SUPPORTED_K = (4, 6, 12)
K8_RELATIVE = Path(
    "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
)
K8_DATASET_SHA256 = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
K8_METADATA_SHA256 = "b76e1ae9c049ce3cff9a625a8acb6af7b53018b7bf0d02308b5f526148aef0ba"
K8_GRAPH_SOURCE_SHA256 = "209a3b15ba49d57d1bbbf5d8cf46a22ad985c9301b556970c4313d73fc94c533"
SOURCE_MANIFEST_IDENTITY_FIELDS = (
    "schema_version",
    "source_manifest_policy",
    "hash_algorithm",
    "hash_chunk_size_bytes",
    "sorting_key",
    "entry_count",
    "catalogue_count",
    "target_source_count",
    "source_root_identity",
    "entries",
    "manifest_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def variant_relative_path(k: int) -> Path:
    require(k in SUPPORTED_K, f"unsupported k={k}; expected one of {SUPPORTED_K}")
    directory = f"temporal_1000u_none_top1500_periodic_knn_k{k}_sparse"
    filename = f"camels_1000u_temporal_logmass_none_top1500_periodic_knn_k{k}_sparse.pt"
    return Path("data/processed") / directory / filename


def logical_dataset_id(k: int) -> str:
    return f"camels_simba_u1000_top1500_temporal5_none_periodic_knn_k{k}_box25_sparse_v1"


def expected_preprocessing_version(k: int) -> str:
    return preprocessing_version_for_config(
        num_nodes=1500,
        normalization="none",
        graph_mode="knn",
        k=k,
        radius=None,
        periodic_boundary=True,
        box_size=25.0,
        graph_storage="sparse_edge_index",
    )


def require_preprocessing_version(actual: Any, *, k: int, label: str) -> None:
    expected = expected_preprocessing_version(k)
    require(actual == expected,
            f"{label}: wrong k-dependent preprocessing version; expected {expected}, actual {actual}")


def require_same_tensor(candidate: Any, control: Any, *, label: str) -> None:
    require(torch.is_tensor(candidate) and torch.is_tensor(control), f"{label}: values must be tensors")
    require(
        candidate.dtype == control.dtype,
        f"{label}: dtype changed; expected {control.dtype}, actual {candidate.dtype}",
    )
    require(torch.equal(candidate, control), f"{label}: tensor values changed")


def require_same_snapshot_path(candidate: Any, control: Any, *, label: str) -> None:
    require(isinstance(candidate, str) and isinstance(control, str),
            f"{label}: stored source paths must be strings")
    require(candidate == control, f"{label}: stored source catalogue path changed")


def require_matching_source_manifests(
    variant_manifest: Any,
    k8_manifest: Any,
) -> None:
    """Require portable source identity while verifying each recorded checkout root."""
    require(isinstance(variant_manifest, Mapping) and isinstance(k8_manifest, Mapping),
            "source manifests must be objects")
    require(set(variant_manifest) == set(k8_manifest), "source manifest field set changed")
    missing = set(SOURCE_MANIFEST_IDENTITY_FIELDS).difference(k8_manifest)
    require(not missing, f"source manifest identity fields missing: {sorted(missing)}")

    for field in sorted(set(k8_manifest).difference({"source_roots"})):
        require(
            variant_manifest.get(field) == k8_manifest.get(field),
            f"source manifest scientific identity changed: {field}",
        )

    variant_roots = variant_manifest.get("source_roots")
    k8_roots = k8_manifest.get("source_roots")
    require(isinstance(variant_roots, Mapping) and isinstance(k8_roots, Mapping),
            "source manifest roots must be objects")
    require(set(variant_roots) == set(k8_roots), "source manifest root roles changed")

    for label, manifest, roots in (
        ("authoritative k8", k8_manifest, k8_roots),
        ("variant", variant_manifest, variant_roots),
    ):
        try:
            verify_full_source_manifest(manifest, source_roots=roots)
        except Exception as exc:
            raise ValueError(f"{label} recorded source roots failed verification: {exc}") from exc

    for role in sorted(k8_roots):
        if k8_roots[role] != variant_roots[role]:
            print(
                "Operational source root differs but independently verified: "
                f"role={role} k8={k8_roots[role]} variant={variant_roots[role]}"
            )


def require_matching_graph_source(
    repo_root: Path,
    variant_commit: str,
    k8_commit: str,
    *,
    authoritative_sha256: str,
) -> None:
    require(base.GIT_COMMIT_PATTERN.fullmatch(k8_commit) is not None,
            "authoritative k8 source commit is invalid")
    require(base.GIT_COMMIT_PATTERN.fullmatch(variant_commit) is not None,
            "variant source commit is invalid")
    k8_hash = base._git_blob_sha256(repo_root, k8_commit, base.EXPECTED_GRAPH_SOURCE)
    require(k8_hash == authoritative_sha256,
            "authoritative k8 graph utility blob SHA-256 mismatch")
    variant_hash = base._git_blob_sha256(repo_root, variant_commit, base.EXPECTED_GRAPH_SOURCE)
    require(variant_hash == k8_hash,
            "variant graph utility differs from the authoritative k8 graph mathematics")


def load_dataset(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), f"dataset is not a dictionary: {path}")
    return value


def check_k8_identity(k8_path: Path) -> Mapping[str, Any]:
    expected = (ROOT / K8_RELATIVE).resolve()
    require(k8_path.resolve() == expected, f"authoritative k8 path must resolve to {expected}")
    metadata_path = k8_path.with_suffix(".metadata.json")
    marker_path = k8_path.with_suffix(".complete")
    require(k8_path.is_file() and metadata_path.is_file() and marker_path.is_file(),
            "authoritative k8 artifact set is incomplete")
    require(base.sha256_file(k8_path) == K8_DATASET_SHA256, "authoritative k8 dataset SHA-256 mismatch")
    require(base.sha256_file(metadata_path) == K8_METADATA_SHA256, "authoritative k8 metadata SHA-256 mismatch")
    metadata = base.load_json(metadata_path)
    marker = base.load_json(marker_path)
    require(metadata.get("checksum") == K8_DATASET_SHA256, "k8 metadata checksum binding mismatch")
    require(marker.get("sha256") == K8_DATASET_SHA256 and marker.get("status") == "complete",
            "k8 completion marker binding mismatch")
    return metadata


def check_edge_index(
    edge_index: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int,
    label: str,
) -> None:
    require(torch.is_tensor(edge_index) and edge_index.dtype == torch.long,
            f"{label}: edge_index must be an int64 tensor")
    require(edge_index.ndim == 2 and edge_index.shape[0] == 2,
            f"{label}: edge_index shape must be [2,E]")
    real_nodes = int(mask.reshape(-1).sum().item())
    require(real_nodes > 1 and edge_index.shape[1] > 0, f"{label}: graph is unexpectedly empty")
    require(int(edge_index.min().item()) >= 0 and int(edge_index.max().item()) < real_nodes,
            f"{label}: edge reaches padding or is out of bounds")
    require(bool((edge_index[0] != edge_index[1]).all()), f"{label}: builder self-loop found")

    encoded = edge_index[0] * features.shape[0] + edge_index[1]
    require(torch.unique(encoded).numel() == encoded.numel(), f"{label}: duplicate directed edge found")
    require(encoded.numel() < 2 or bool((encoded[1:] > encoded[:-1]).all()),
            f"{label}: edge_index is not lexicographically ordered")
    reverse = edge_index[1] * features.shape[0] + edge_index[0]
    require(bool(torch.isin(reverse, encoded).all()), f"{label}: asymmetric edge found")

    effective_k = min(k, real_nodes - 1)
    degree = torch.bincount(edge_index[0], minlength=real_nodes)
    require(bool((degree >= effective_k).all()), f"{label}: degree below effective_k={effective_k}")
    if effective_k == real_nodes - 1:
        require(edge_index.shape[1] == real_nodes * (real_nodes - 1),
                f"{label}: effective-k complete graph has the wrong edge count")

    expected = build_sparse_knn_edge_index(
        positions=features.detach().cpu().numpy()[:, 1:4],
        mask=mask.detach().cpu().numpy(),
        k=k,
        periodic_boundary=True,
        box_size=25.0,
        tie_keys=np.arange(real_nodes, dtype=np.int64),
    )
    require(torch.equal(edge_index.detach().cpu(), torch.from_numpy(expected)),
            f"{label}: edges disagree with authoritative periodic kNN reconstruction")


def check_matched_data(
    variant: Mapping[str, Any],
    k8: Mapping[str, Any],
    variant_metadata: Mapping[str, Any],
    k8_metadata: Mapping[str, Any],
    *,
    k: int,
) -> None:
    require(list(variant) == list(k8) == [f"LH_{index}" for index in range(1000)],
            "variant and k8 universe IDs/order are not exactly LH_0..LH_999")

    allowed_metadata_differences = {
        "preprocessing_version", "output_path", "k", "creation_timestamp_utc",
        "builder_config_hash", "git_commit", "source_git_commit",
        "build_launcher_sha256", "checksum", "edge_statistics",
    }
    require(set(variant_metadata) == set(k8_metadata), "variant metadata field set changed")
    for field in set(k8_metadata).difference(allowed_metadata_differences):
        if field == "source_manifest":
            require_matching_source_manifests(
                variant_metadata.get(field), k8_metadata.get(field),
            )
            continue
        require(variant_metadata.get(field) == k8_metadata.get(field),
                f"metadata scientific control changed: {field}")

    for universe_id in variant:
        candidate = variant[universe_id]
        control = k8[universe_id]
        require(set(candidate) == set(control), f"{universe_id}: sample field set changed")
        allowed_sample_differences = {
            "preprocessing_version", "edge_index_list", "Nodes_list", "mask_list",
            "target", "snapshots",
        }
        for field in set(control).difference(allowed_sample_differences):
            require(candidate.get(field) == control.get(field),
                    f"{universe_id}: sample scientific control changed: {field}")
        require(candidate.get("edge_weight_list") == control.get("edge_weight_list") is None,
                f"{universe_id}: edge-weight policy changed")
        require_preprocessing_version(candidate.get("preprocessing_version"), k=k,
                                      label=universe_id)
        require_same_tensor(candidate["target"], control["target"],
                            label=f"{universe_id} target")
        require(len(candidate["Nodes_list"]) == len(control["Nodes_list"]) == 5,
                f"{universe_id}: node-feature snapshot count changed")

        for snapshot_index in range(5):
            label = f"{universe_id} snapshot {snapshot_index}"
            candidate_x = candidate["Nodes_list"][snapshot_index]
            control_x = control["Nodes_list"][snapshot_index]
            candidate_mask = candidate["mask_list"][snapshot_index]
            control_mask = control["mask_list"][snapshot_index]
            require_same_tensor(candidate_x, control_x, label=f"{label} node features")
            require_same_tensor(candidate_mask, control_mask, label=f"{label} node mask")

            candidate_snapshot = candidate["snapshots"][snapshot_index]
            control_snapshot = control["snapshots"][snapshot_index]
            require(set(candidate_snapshot) == set(control_snapshot),
                    f"{label}: snapshot field set changed")
            require_same_snapshot_path(candidate_snapshot.get("path"), control_snapshot.get("path"),
                                       label=label)
            for field in set(control_snapshot).difference({"path", "preprocessing_version", "k"}):
                require(candidate_snapshot.get(field) == control_snapshot.get(field),
                        f"{label}: snapshot scientific control changed: {field}")
            require(candidate_snapshot.get("k") == k, f"{label}: variant k metadata mismatch")
            require_preprocessing_version(candidate_snapshot.get("preprocessing_version"), k=k,
                                          label=label)

            check_edge_index(
                candidate["edge_index_list"][snapshot_index], candidate_x, candidate_mask,
                k=k, label=label,
            )


def configure_base(k: int, relative_dataset: Path) -> None:
    base.EXPECTED_RELATIVE_DATASET = relative_dataset
    base.EXPECTED_TOP_N = 1500
    base.EXPECTED_K = k
    base.EXPECTED_LOGICAL_ID = logical_dataset_id(k)


def validate(k: int, variant_path: Path, k8_path: Path) -> None:
    relative = variant_relative_path(k)
    expected_variant = (ROOT / relative).resolve()
    require(variant_path.resolve() == expected_variant, f"variant path must resolve to {expected_variant}")
    configure_base(k, relative)
    base.validate(ROOT, variant_path, ROOT / base.EXPECTED_RELATIVE_TARGET)

    variant_metadata = base.load_json(variant_path.with_suffix(".metadata.json"))
    k8_metadata = check_k8_identity(k8_path)
    require(variant_metadata.get("k") == k, "variant metadata k mismatch")
    require(variant_metadata.get("builder_source_sha256") == k8_metadata.get("builder_source_sha256"),
            "graph builder source changed relative to authoritative k8")
    require_matching_graph_source(
        ROOT,
        variant_metadata.get("source_git_commit"),
        k8_metadata.get("source_git_commit"),
        authoritative_sha256=K8_GRAPH_SOURCE_SHA256,
    )

    variant = load_dataset(variant_path)
    k8 = load_dataset(k8_path)
    check_matched_data(variant, k8, variant_metadata, k8_metadata, k=k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, choices=SUPPORTED_K, required=True)
    parser.add_argument("--variant-dataset", type=Path, required=True)
    parser.add_argument("--authoritative-k8", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate(args.k, args.variant_dataset, args.authoritative_k8)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
