#!/usr/bin/env python3
"""Read-only integrity validator for the canonical U1000 Top2000 dataset."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import validate_u1000_top1000_sparse_dataset as base
from src.data.camels_graph_utils import (
    MASS_COLUMN,
    build_node_features,
    build_positions,
    build_sparse_knn_edge_index,
    clean_halo_dataframe,
    read_hlist_file,
    select_top_halos,
    selection_provenance,
)


TOP_N = 2000
EXPECTED_REAL_NODES = 9_861_546
EXPECTED_PADDED_SLOTS = 138_454
EXPECTED_PADDED_PAIRS = 180
EXPECTED_PADDED_FINAL_SNAPSHOTS = 0
EXPECTED_SNAPSHOTS = ("0.20000", "0.25000", "0.51209", "0.75065", "1.00000")
DATASET = Path(
    "data/processed/temporal_1000u_none_top2000_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top2000_periodic_knn_sparse.pt"
)
TARGET = Path("outputs/target_inspection_1000u.csv")
TARGET_SHA256 = base.EXPECTED_TARGET_SHA256
RAW_COUNT_AUDIT = Path(
    "reports/experiment_registry/u1000_top1500_raw_halo_count_distribution.csv"
)
BUILDER_MODULE = "src.data.build_temporal_sequences_top2000"
BUILDER_SOURCE = Path("src/data/build_temporal_sequences_top2000.py")
BUILD_LAUNCHER = Path("scripts/production/run_u1000_top2000_sparse_build.sh")
LOGICAL_DATASET_ID = (
    "camels_simba_u1000_top2000_temporal5_none_periodic_knn_k8_box25_sparse_v1"
)


@dataclass(frozen=True)
class DatasetLayout:
    top_n: int
    directory: Path
    dataset: Path
    metadata: Path
    completion_marker: Path
    lock: Path
    split_directory: Path
    logical_dataset_id: str


def require_top_n(top_n: int) -> int:
    if type(top_n) is not int or top_n != TOP_N:
        raise ValueError(f"Top-N must be exactly {TOP_N}; received {top_n!r}")
    return top_n


def dataset_layout(top_n: int = TOP_N) -> DatasetLayout:
    require_top_n(top_n)
    return DatasetLayout(
        top_n=TOP_N,
        directory=DATASET.parent,
        dataset=DATASET,
        metadata=DATASET.with_suffix(".metadata.json"),
        completion_marker=DATASET.with_suffix(".complete"),
        lock=DATASET.with_suffix(DATASET.suffix + ".lock"),
        split_directory=Path("configs/splits/u1000_top2000_none_k8_sparse"),
        logical_dataset_id=LOGICAL_DATASET_ID,
    )


def configure_base() -> None:
    base.EXPECTED_RELATIVE_DATASET = DATASET
    base.EXPECTED_TOP_N = TOP_N
    base.EXPECTED_LOGICAL_ID = LOGICAL_DATASET_ID
    base.EXPECTED_BUILDER_MODULE = BUILDER_MODULE
    base.EXPECTED_BUILDER_SOURCE = BUILDER_SOURCE
    base.EXPECTED_TOP1500_LAUNCHER = BUILD_LAUNCHER
    base.check_builder_provenance = check_top2000_builder_provenance


def check_top2000_builder_provenance(
    metadata: Mapping[str, Any], repo_root: Path, *, required: bool,
) -> None:
    """Validate the dedicated builder and launcher without Top1500 aliases."""
    fields = {
        "builder_provenance_schema_version", "builder_entrypoint", "builder_module",
        "builder_source_path", "builder_source_sha256", "build_launcher_path",
        "build_launcher_sha256", "source_git_commit",
    }
    present = fields.intersection(metadata)
    if not required and not present:
        return
    missing = sorted(fields.difference(metadata))
    base.require(not missing, f"Top2000 builder provenance fields missing: {missing}")
    base.require(metadata["builder_provenance_schema_version"] == "camels_builder_provenance_v1",
                 "wrong Top2000 builder-provenance schema")
    base.require(metadata["builder_entrypoint"] == BUILDER_MODULE,
                 "metadata claims a different Top2000 builder entrypoint")
    base.require(metadata["builder_module"] == BUILDER_MODULE,
                 "metadata claims a different Top2000 builder module")
    base.require(metadata["builder_source_path"] == BUILDER_SOURCE.as_posix(),
                 "metadata claims a different Top2000 builder source")
    base.require(metadata["build_launcher_path"] == BUILD_LAUNCHER.as_posix(),
                 "metadata claims a different Top2000 production launcher")
    builder = base._canonical_repository_path(
        repo_root, metadata["builder_source_path"], "builder_source_path"
    )
    launcher = base._canonical_repository_path(
        repo_root, metadata["build_launcher_path"], "build_launcher_path"
    )
    base.require(builder.is_file(), f"Top2000 builder source is missing: {builder}")
    base.require(launcher.is_file(), f"Top2000 launcher is missing: {launcher}")
    for key in ("builder_source_sha256", "build_launcher_sha256"):
        base.require(isinstance(metadata[key], str) and
                     base.SHA256_PATTERN.fullmatch(metadata[key]) is not None,
                     f"{key} is not a valid SHA-256")
    base.require(metadata["builder_source_sha256"] == base.sha256_file_streaming(builder),
                 "Top2000 builder source SHA-256 mismatch")
    base.require(metadata["build_launcher_sha256"] == base.sha256_file_streaming(launcher),
                 "Top2000 launcher SHA-256 mismatch")
    base.require(isinstance(metadata["source_git_commit"], str) and
                 base.GIT_COMMIT_PATTERN.fullmatch(metadata["source_git_commit"]) is not None,
                 "source_git_commit is not a full Git identity")
    base.require(metadata.get("git_commit") == metadata["source_git_commit"],
                 "source_git_commit disagrees with git_commit")


def load_authoritative_counts(root: Path) -> dict[tuple[str, str], int]:
    path = root / RAW_COUNT_AUDIT
    base.require(path.is_file(), f"authoritative raw-count audit is missing: {path}")
    counts: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            base.require(row.get("status") == "PASS", f"raw-count audit failure: {row}")
            key = (str(row["universe_id"]), str(row["snapshot"]))
            base.require(key not in counts, f"duplicate raw-count audit row: {key}")
            counts[key] = int(row["valid_real_halo_count"])
    base.require(len(counts) == 5000, "raw-count audit does not contain 5,000 rows")
    return counts


def selection_hash(keys: list[str]) -> str:
    payload = "".join(f"{rank}\t{key}\n" for rank, key in enumerate(keys, start=1))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_edge_checks(
    counts: dict[tuple[str, str], int],
) -> set[tuple[str, str]]:
    checks = {
        (f"LH_{universe}", snapshot)
        for universe in (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)
        for snapshot in EXPECTED_SNAPSHOTS
    }
    for snapshot in EXPECTED_SNAPSHOTS:
        checks.add(min(
            ((universe, snap) for universe, snap in counts if snap == snapshot),
            key=lambda key: (counts[key], key[0]),
        ))
    return checks


def check_raw_selection_padding_and_edges(root: Path, dataset_path: Path) -> None:
    counts = load_authoritative_counts(root)
    edge_checks = deterministic_edge_checks(counts)
    try:
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(dataset_path, map_location="cpu")

    total_real = 0
    padded_pairs = 0
    padded_final = 0
    selection_hashes: list[str] = []
    for universe_id, sample in dataset.items():
        for index, snapshot in enumerate(sample["snapshots"]):
            snapshot_id = f"{float(snapshot['snapshot_value']):.5f}"
            key = (universe_id, snapshot_id)
            expected_real = min(TOP_N, counts[key])
            features = sample["Nodes_list"][index]
            mask = sample["mask_list"][index]
            edges = sample["edge_index_list"][index]
            real = int(mask.sum().item())
            base.require(real == expected_real, f"{key}: real-node count disagrees with raw audit")
            total_real += real
            padded_pairs += int(real < TOP_N)
            padded_final += int(snapshot_id == "1.00000" and real < TOP_N)

            raw_path = Path(str(snapshot["path"]))
            raw_path = raw_path if raw_path.is_absolute() else root / raw_path
            clean = clean_halo_dataframe(read_hlist_file(raw_path))
            selected = select_top_halos(clean, num_nodes=TOP_N, mass_column=MASS_COLUMN)
            provenance = selection_provenance(selected)
            expected_features = torch.from_numpy(build_node_features(selected))
            base.require(len(selected) == real, f"{key}: selected raw-halo count mismatch")
            base.require(torch.equal(features[:real].cpu(), expected_features),
                         f"{key}: stored features do not match stable raw-Mvir Top2000 selection")
            keys = [str(value) for value in snapshot.get("selected_halo_keys", [])]
            base.require(keys == provenance["selected_halo_keys"],
                         f"{key}: selected halo keys/tie order mismatch")
            if "raw_mass_rank" in snapshot:
                base.require(snapshot["raw_mass_rank"] == list(range(1, real + 1)),
                             f"{key}: raw-mass ranks are not canonical")
            expected_selection_hash = selection_hash(keys)
            base.require(snapshot.get("selection_hash_sha256") == expected_selection_hash,
                         f"{key}: selection hash mismatch")
            selection_hashes.append(expected_selection_hash)
            if real > 1:
                mass = features[:real, 0]
                base.require(bool((mass[:-1] >= mass[1:]).all()),
                             f"{key}: selected raw-mass order is not descending")

            encoded = edges[0] * TOP_N + edges[1]
            base.require(bool((encoded[1:] > encoded[:-1]).all()),
                         f"{key}: edge_index is not unique lexicographic order")
            effective_k = min(8, real - 1)
            base.require(real * effective_k <= edges.shape[1] <= 2 * real * effective_k,
                         f"{key}: symmetric kNN edge count is outside its exact bounds")

            if key in edge_checks:
                positions = build_positions(selected)
                positions_padded = np.zeros((TOP_N, 3), dtype=np.float32)
                positions_padded[:real] = positions
                mask_np = np.zeros((TOP_N, 1), dtype=np.float32)
                mask_np[:real] = 1.0
                expected_edges = build_sparse_knn_edge_index(
                    positions_padded,
                    mask_np,
                    k=8,
                    periodic_boundary=True,
                    box_size=25.0,
                    tie_keys=np.arange(real, dtype=np.int64),
                )
                base.require(np.array_equal(edges.cpu().numpy(), expected_edges),
                             f"{key}: edges differ from exact periodic kNN reconstruction")

    padded_slots = 5_000 * TOP_N - total_real
    base.require(total_real == EXPECTED_REAL_NODES, "unexpected retained-real-node total")
    base.require(padded_slots == EXPECTED_PADDED_SLOTS, "unexpected padded-slot total")
    base.require(padded_pairs == EXPECTED_PADDED_PAIRS, "unexpected padded-pair total")
    base.require(padded_final == EXPECTED_PADDED_FINAL_SNAPSHOTS,
                 "unexpected final-snapshot padding")
    metadata = base.load_json(dataset_path.with_suffix(".metadata.json"))
    combined = hashlib.sha256(
        "".join(f"{value}\n" for value in selection_hashes).encode("utf-8")
    ).hexdigest()
    base.require(metadata.get("selected_halo_hash") == combined,
                 "metadata selected-halo aggregate hash mismatch")
    padding = metadata.get("node_padding_statistics", {})
    base.require(padding.get("padded_total") == EXPECTED_PADDED_SLOTS,
                 "metadata padded-slot total mismatch")
    base.require(abs(100.0 * padded_slots / 10_000_000 - 1.38454) < 1e-12,
                 "Top2000 padding percentage mismatch")


def validate_dataset(root: Path, top_n: int = TOP_N) -> DatasetLayout:
    require_top_n(top_n)
    layout = dataset_layout(top_n)
    configure_base()
    base.validate(root.resolve(), root.resolve() / layout.dataset, root.resolve() / TARGET)
    gc.collect()
    check_raw_selection_padding_and_edges(root.resolve(), root.resolve() / layout.dataset)
    return layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        validate_dataset(args.repo_root)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
