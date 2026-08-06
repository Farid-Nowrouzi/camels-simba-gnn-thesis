#!/usr/bin/env python3
"""Read-only integrity validator for the canonical U1000 Top1500 dataset."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation import validate_u1000_top1000_sparse_dataset as validator


DATASET = Path(
    "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/"
    "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
)
AUDIT = Path("reports/experiment_registry/u1000_top1500_raw_halo_count_distribution.csv")


def configure() -> None:
    validator.EXPECTED_RELATIVE_DATASET = DATASET
    validator.EXPECTED_TOP_N = 1500
    validator.EXPECTED_LOGICAL_ID = (
        "camels_simba_u1000_top1500_temporal5_none_periodic_knn_k8_box25_sparse_v1"
    )


def check_padding(dataset_path: Path) -> None:
    audit_path = ROOT / AUDIT
    validator.require(audit_path.is_file(), f"raw halo-count audit missing: {audit_path}")
    with audit_path.open(newline="", encoding="utf-8") as handle:
        audit = {(row["universe_id"], row["snapshot"]): int(row["valid_real_halo_count"])
                 for row in csv.DictReader(handle) if row["status"] == "PASS"}
    validator.require(len(audit) == 5000, "raw audit does not contain 5,000 successful catalogues")
    try:
        data = validator.torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = validator.torch.load(dataset_path, map_location="cpu")
    for universe_id, sample in data.items():
        for snapshot in sample["snapshots"]:
            key = (universe_id, f"{float(snapshot['snapshot_value']):.5f}")
            expected = min(1500, audit[key])
            validator.require(snapshot["num_real_nodes"] == expected,
                              f"{key}: mask/padding disagrees with raw audit")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args()
    configure()
    root = args.repo_root.resolve()
    dataset = (args.dataset or root / DATASET).resolve()
    target = (args.target or root / validator.EXPECTED_RELATIVE_TARGET).resolve()
    try:
        validator.validate(root, dataset, target)
        check_padding(dataset)
    except Exception as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
