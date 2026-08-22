#!/usr/bin/env python3
"""Validate or preflight the controlled U1000 Top500/Top750 sparse datasets."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import validate_u1000_top1000_sparse_dataset as base


SUPPORTED_TOP_N = (500, 750)
TARGET = Path("outputs/target_inspection_1000u.csv")
TARGET_SHA256 = "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
RAW_COUNT_AUDIT = Path(
    "reports/experiment_registry/u1000_top1500_raw_halo_count_distribution.csv"
)
BUILD_LAUNCHER = Path(
    "scripts/production/run_u1000_top500_top750_sparse_build.sh"
)
BUILDER_SOURCE = Path("src/data/build_temporal_sequences.py")


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
    if type(top_n) is not int or top_n not in SUPPORTED_TOP_N:
        raise ValueError(
            f"Top-N must be exactly one of {SUPPORTED_TOP_N}; received {top_n!r}"
        )
    return top_n


def dataset_layout(top_n: int) -> DatasetLayout:
    top_n = require_top_n(top_n)
    directory = Path(
        f"data/processed/temporal_1000u_none_top{top_n}_periodic_knn_sparse"
    )
    dataset = directory / (
        f"camels_1000u_temporal_logmass_none_top{top_n}_periodic_knn_sparse.pt"
    )
    return DatasetLayout(
        top_n=top_n,
        directory=directory,
        dataset=dataset,
        metadata=dataset.with_suffix(".metadata.json"),
        completion_marker=dataset.with_suffix(".complete"),
        lock=dataset.with_suffix(dataset.suffix + ".lock"),
        split_directory=Path(f"configs/splits/u1000_top{top_n}_none_k8_sparse"),
        logical_dataset_id=(
            f"camels_simba_u1000_top{top_n}_temporal5_none_periodic_"
            "knn_k8_box25_sparse_v1"
        ),
    )


def destination_artifacts(dataset_path: Path) -> list[Path]:
    artifacts = [
        dataset_path,
        dataset_path.with_suffix(".metadata.json"),
        dataset_path.with_suffix(".complete"),
        dataset_path.with_suffix(dataset_path.suffix + ".lock"),
    ]
    if dataset_path.parent.exists():
        artifacts.append(dataset_path.parent)
    if dataset_path.parent.is_dir():
        artifacts.extend(dataset_path.parent.glob(f".{dataset_path.name}.*.tmp"))
        metadata_name = dataset_path.with_suffix(".metadata.json").name
        artifacts.extend(dataset_path.parent.glob(f".{metadata_name}.*.tmp"))
    return sorted(set(artifacts), key=lambda path: path.as_posix())


def assert_destination_clean(dataset_path: Path) -> None:
    collisions = [path for path in destination_artifacts(dataset_path) if path.exists()]
    if collisions:
        rendered = ", ".join(path.as_posix() for path in collisions)
        raise FileExistsError(
            "destination is not empty; refusing replacement or recovery: " + rendered
        )


def configure_base(top_n: int) -> DatasetLayout:
    layout = dataset_layout(top_n)
    base.EXPECTED_RELATIVE_DATASET = layout.dataset
    base.EXPECTED_TOP_N = top_n
    base.EXPECTED_LOGICAL_ID = layout.logical_dataset_id
    base.EXPECTED_TOP1500_LAUNCHER = BUILD_LAUNCHER
    return layout


def _load_raw_counts(root: Path) -> dict[tuple[str, str], int]:
    audit_path = root / RAW_COUNT_AUDIT
    base.require(audit_path.is_file(), f"raw halo-count audit missing: {audit_path}")
    counts: dict[tuple[str, str], int] = {}
    with audit_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        base.require(reader.fieldnames is not None, "raw-count audit has no header")
        for row in reader:
            if row.get("status") != "PASS":
                continue
            key = (str(row["universe_id"]), str(row["snapshot"]))
            base.require(key not in counts, f"duplicate raw-count audit row: {key}")
            counts[key] = int(row["valid_real_halo_count"])
    base.require(len(counts) == 5000, "raw-count audit lacks 5,000 successful catalogues")
    return counts


def _load_dataset(dataset_path: Path) -> dict[str, Any]:
    try:
        value = base.torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        value = base.torch.load(dataset_path, map_location="cpu")
    base.require(isinstance(value, dict), "serialized dataset is not a dictionary")
    return value


def check_raw_counts_and_loaders(root: Path, dataset_path: Path, top_n: int) -> None:
    counts = _load_raw_counts(root)
    data = _load_dataset(dataset_path)
    for universe_id, sample in data.items():
        for snapshot in sample["snapshots"]:
            key = (universe_id, f"{float(snapshot['snapshot_value']):.5f}")
            base.require(key in counts, f"raw-count audit has no row for {key}")
            expected = min(top_n, counts[key])
            base.require(
                snapshot["num_real_nodes"] == expected,
                f"{key}: real-node count disagrees with raw catalogue audit",
            )
            base.require(
                snapshot["selected_num_halos_before_padding"] == expected,
                f"{key}: selected-node count disagrees with raw catalogue audit",
            )

    from src.training.train_evolvegcn_h import CamelsTemporalDataset

    first_id = "LH_0"
    temporal = CamelsTemporalDataset(data, [first_id])
    item = temporal[0]
    sparse_sample, x_seq, mask_seq = item[1], item[2], item[3]
    base.require(isinstance(sparse_sample, dict), "EvolveGCN sparse loader did not retain sample")
    base.require(len(sparse_sample["edge_index_list"]) == 5, "EvolveGCN did not receive five edge lists")
    base.require(tuple(x_seq.shape) == (5, top_n, 7), "EvolveGCN feature sequence shape mismatch")
    base.require(tuple(mask_seq.shape) == (5, top_n, 1), "EvolveGCN mask sequence shape mismatch")
    del temporal, item, data
    gc.collect()


def validate_dataset(root: Path, top_n: int) -> DatasetLayout:
    root = root.resolve()
    layout = configure_base(top_n)
    dataset = (root / layout.dataset).resolve()
    target = (root / TARGET).resolve()
    base.validate(root, dataset, target)
    metadata = base.load_json(dataset.with_suffix(".metadata.json"))
    base.check_builder_provenance(metadata, root, required=True)
    check_raw_counts_and_loaders(root, dataset, top_n)
    return layout


def preflight(root: Path, top_n: int) -> DatasetLayout:
    root = root.resolve()
    layout = configure_base(top_n)
    required = (TARGET, RAW_COUNT_AUDIT, BUILD_LAUNCHER, BUILDER_SOURCE)
    for relative in required:
        base.require((root / relative).is_file(), f"required input missing: {relative}")
    base.require(
        base.sha256_file(root / TARGET) == TARGET_SHA256,
        "authoritative target SHA-256 mismatch",
    )
    assert_destination_clean(root / layout.dataset)
    return layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, required=True, choices=SUPPORTED_TOP_N)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check protocol paths and collision gates without requiring/loading a dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.preflight_only:
            layout = preflight(args.repo_root, args.top_n)
            print("PREFLIGHT PASS")
            print(f"Top-N: {layout.top_n}")
            print(f"Dataset: {layout.dataset}")
            print(f"Metadata: {layout.metadata}")
            print(f"Completion marker: {layout.completion_marker}")
            print("Protocol: U1000 temporal5 none periodic-knn k8 box25 sparse CPU")
            print("No dataset was loaded or created.")
        else:
            layout = validate_dataset(args.repo_root, args.top_n)
            print(f"PASS: validated U1000 Top{layout.top_n} dataset")
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
