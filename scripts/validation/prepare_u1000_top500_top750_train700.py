#!/usr/bin/env python3
"""Preflight or bind exact U1000 Train700 splits to validated Top500/Top750 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import validate_u1000_top500_top750_sparse_dataset as dataset_validator
from src.data.source_manifest import sha256_file_streaming, source_manifest_sha256
from src.training.split_manifest import canonical_manifest_sha256, ordered_id_hash


SUPPORTED_TOP_N = dataset_validator.SUPPORTED_TOP_N
SUPPORTED_SEEDS = (42, 123, 2025)
TARGET = Path("outputs/target_inspection_1000u.csv")
TARGET_SHA256 = dataset_validator.TARGET_SHA256
TOP1000_SPLITS = Path("configs/splits/u1000_top1000_none_k8_sparse")
TOP1500_SPLITS = Path("configs/splits/u1000_top1500_none_k8_sparse")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PARTITIONS = ("train", "val", "test", "unused")
EXPECTED_COUNTS = {"train": 700, "val": 99, "test": 201, "unused": 0}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_seed(seed: int) -> int:
    if type(seed) is not int or seed not in SUPPORTED_SEEDS:
        raise ValueError(
            f"seed must be exactly one of {SUPPORTED_SEEDS}; received {seed!r}"
        )
    return seed


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def reference_paths(seed: int) -> tuple[Path, Path]:
    seed = require_seed(seed)
    filename = f"seed{seed}_train700.json"
    return TOP1000_SPLITS / filename, TOP1500_SPLITS / filename


def output_manifest_path(top_n: int, seed: int) -> Path:
    layout = dataset_validator.dataset_layout(top_n)
    return layout.split_directory / f"seed{require_seed(seed)}_train700.json"


def partition_identity(manifest: Mapping[str, Any]) -> str:
    payload = "".join(
        partition + "\n" + ordered_id_hash(manifest[f"{partition}_ids"])
        for partition in PARTITIONS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_manifest_internally(manifest: Mapping[str, Any], seed: int, label: str) -> None:
    require(manifest.get("seed") == seed, f"{label}: wrong seed")
    counts = manifest.get("counts")
    require(isinstance(counts, dict), f"{label}: counts missing")
    all_ids: list[str] = []
    for partition in PARTITIONS:
        key = f"{partition}_ids"
        values = manifest.get(key)
        require(isinstance(values, list), f"{label}: {key} is not a list")
        require(len(values) == EXPECTED_COUNTS[partition], f"{label}: wrong {partition} count")
        require(len(values) == len(set(values)), f"{label}: duplicate {partition} IDs")
        require(counts.get(partition) == len(values), f"{label}: {partition} count field mismatch")
        expected_hash = manifest.get("split_hashes", {}).get(partition)
        require(expected_hash == ordered_id_hash(values), f"{label}: {partition} ordered hash mismatch")
        all_ids.extend(values)
    require(len(all_ids) == 1000 and len(set(all_ids)) == 1000, f"{label}: partitions do not cover 1000 unique IDs")
    require(set(all_ids) == {f"LH_{index}" for index in range(1000)}, f"{label}: wrong universe population")
    canonical = manifest.get("canonical_manifest_sha256")
    require(canonical == canonical_manifest_sha256(dict(manifest)), f"{label}: canonical hash mismatch")


def load_reference_pair(root: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    top1000_relative, top1500_relative = reference_paths(seed)
    top1000_path = root / top1000_relative
    top1500_path = root / top1500_relative
    require(top1000_path.is_file(), f"missing Top1000 reference: {top1000_relative}")
    require(top1500_path.is_file(), f"missing Top1500 reference: {top1500_relative}")
    top1000 = read_json(top1000_path)
    top1500 = read_json(top1500_path)
    _check_manifest_internally(top1000, seed, top1000_relative.as_posix())
    _check_manifest_internally(top1500, seed, top1500_relative.as_posix())
    for partition in PARTITIONS:
        key = f"{partition}_ids"
        require(
            top1000[key] == top1500[key],
            f"seed {seed}: Top1000/Top1500 ordered {key} differ",
        )
    return top1000, top1500


def _require_real_hash(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} is missing or is not a real SHA-256 identity",
    )
    return value


def build_dataset_binding(root: Path, top_n: int) -> dict[str, str]:
    root = root.resolve()
    layout = dataset_validator.validate_dataset(root, top_n)
    dataset = root / layout.dataset
    metadata_path = root / layout.metadata
    marker_path = root / layout.completion_marker
    target_path = root / TARGET
    metadata = read_json(metadata_path)
    marker = read_json(marker_path)
    manifest = metadata.get("source_manifest")
    require(isinstance(manifest, dict), "metadata source manifest is missing")

    dataset_sha = sha256_file_streaming(dataset)
    metadata_sha = sha256_file_streaming(metadata_path)
    marker_sha = sha256_file_streaming(marker_path)
    target_sha = sha256_file_streaming(target_path)
    source_sha = source_manifest_sha256(manifest)
    require(metadata.get("checksum") == dataset_sha, "metadata/dataset SHA-256 mismatch")
    require(marker.get("sha256") == dataset_sha, "completion-marker/dataset SHA-256 mismatch")
    require(manifest.get("manifest_sha256") == source_sha, "source-manifest SHA-256 mismatch")
    require(metadata.get("source_manifest_sha256") == source_sha, "metadata source identity mismatch")
    require(target_sha == TARGET_SHA256, "authoritative target SHA-256 mismatch")

    binding = {
        "dataset_path": layout.dataset.as_posix(),
        "dataset_sha256": dataset_sha,
        "metadata_path": layout.metadata.as_posix(),
        "metadata_sha256": metadata_sha,
        "completion_marker_path": layout.completion_marker.as_posix(),
        "completion_marker_sha256": marker_sha,
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "graph_storage_mode": "sparse_edge_index",
        "logical_dataset_id": layout.logical_dataset_id,
        "source_manifest_policy": "full_sha256",
        "source_manifest_sha256": source_sha,
        "target_table_path": TARGET.as_posix(),
        "target_table_sha256": target_sha,
    }
    for key in (
        "dataset_sha256",
        "metadata_sha256",
        "completion_marker_sha256",
        "source_manifest_sha256",
        "target_table_sha256",
    ):
        _require_real_hash(binding[key], key)
    return binding


def create_bound_manifest(
    source: Mapping[str, Any], binding: Mapping[str, str], top_n: int
) -> dict[str, Any]:
    layout = dataset_validator.dataset_layout(top_n)
    required_binding = {
        "dataset_path": layout.dataset.as_posix(),
        "metadata_path": layout.metadata.as_posix(),
        "completion_marker_path": layout.completion_marker.as_posix(),
        "logical_dataset_id": layout.logical_dataset_id,
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "graph_storage_mode": "sparse_edge_index",
        "source_manifest_policy": "full_sha256",
        "target_table_path": TARGET.as_posix(),
        "target_table_sha256": TARGET_SHA256,
    }
    for key, expected in required_binding.items():
        require(binding.get(key) == expected, f"dataset binding {key} mismatch")
    for key in (
        "dataset_sha256",
        "metadata_sha256",
        "completion_marker_sha256",
        "source_manifest_sha256",
    ):
        _require_real_hash(binding.get(key), key)

    result = deepcopy(dict(source))
    result["dataset_identity"] = binding["dataset_sha256"]
    result["dataset_binding"] = dict(binding)
    result["graph_protocol_summary"]["top_n"] = top_n
    result["partition_source_manifest"] = source.get("canonical_manifest_sha256")
    result["partition_identity"] = partition_identity(source)
    result["canonical_manifest_sha256"] = canonical_manifest_sha256(result)
    return result


def verify_preserved_ids(
    candidate: Mapping[str, Any], top1000: Mapping[str, Any], top1500: Mapping[str, Any], seed: int
) -> None:
    _check_manifest_internally(candidate, seed, "candidate")
    for partition in PARTITIONS:
        key = f"{partition}_ids"
        require(candidate[key] == top1000[key], f"candidate changed Top1000 {key}")
        require(candidate[key] == top1500[key], f"candidate changed Top1500 {key}")


def preflight(root: Path, top_n: int) -> list[tuple[int, Path, Path, Path]]:
    root = root.resolve()
    dataset_validator.require_top_n(top_n)
    rows: list[tuple[int, Path, Path, Path]] = []
    for seed in SUPPORTED_SEEDS:
        load_reference_pair(root, seed)
        top1000, top1500 = reference_paths(seed)
        output = output_manifest_path(top_n, seed)
        require(not (root / output).exists(), f"destination manifest already exists: {output}")
        rows.append((seed, top1000, top1500, output))
    return rows


def bind(root: Path, top_n: int) -> list[Path]:
    root = root.resolve()
    rows = preflight(root, top_n)
    binding = build_dataset_binding(root, top_n)
    prepared: list[tuple[Path, dict[str, Any]]] = []
    for seed, _, _, output in rows:
        top1000, top1500 = load_reference_pair(root, seed)
        candidate = create_bound_manifest(top1000, binding, top_n)
        verify_preserved_ids(candidate, top1000, top1500, seed)
        prepared.append((root / output, candidate))

    for output, _ in prepared:
        require(not output.exists(), f"destination manifest already exists: {output}")
    written: list[Path] = []
    for output, manifest in prepared:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        written.append(output)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, required=True, choices=SUPPORTED_TOP_N)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify exact references and destinations without loading data or writing manifests.",
    )
    mode.add_argument(
        "--bind",
        action="store_true",
        help="Validate the built dataset and create immutable manifests with real hashes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.preflight_only:
            rows = preflight(args.repo_root, args.top_n)
            print(f"PREFLIGHT PASS: U1000 Top{args.top_n} Train700")
            for seed, top1000, top1500, output in rows:
                print(f"seed {seed}: {top1000} == {top1500} -> {output}")
            print("Exact ordered IDs: train=700 val=99 test=201 unused=0")
            print("No random splitting, dataset loading, hashing, or manifest writing occurred.")
        else:
            written = bind(args.repo_root, args.top_n)
            for path in written:
                print(path.relative_to(args.repo_root.resolve()))
            print(f"PASS: wrote {len(written)} validated dataset-bound manifests")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
