#!/usr/bin/env python3
"""
inspect_targets.py

Purpose
-------
Inspect CAMELS-SIMBA raw halo catalog files and extract the real cosmological
target value Omega_M for each universe.

This script verifies that temporal/static datasets use REAL targets from the
raw CAMELS files, not dummy targets.

Expected raw file naming pattern:
    LH_0_hlist_0.20000.list
    LH_0_hlist_0.25000.list
    LH_0_hlist_0.51209.list
    LH_0_hlist_0.75065.list
    LH_0_hlist_1.00000.list

Output CSV format is compatible with build_temporal_sequences.py:

    universe_id, universe_index, omega_m, ...

Example command:
    python -m src.data.inspect_targets \
        --raw_dir data/raw/CAMELS_SIMBA_100U \
        --num_universes 20 \
        --required_snapshots 5 \
        --save_csv outputs/target_inspection_20u.csv \
        --save_json outputs/target_inspection_20u.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    PREPROCESSING_VERSION,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

TARGET_KEY = "Omega_M"

EXPECTED_OMEGA_M_MIN = 0.0
EXPECTED_OMEGA_M_MAX = 1.0


# ---------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------

def read_header_lines(path: Path, max_lines: int = 160) -> List[str]:
    """
    Read only the header/comment lines from a CAMELS .list file.

    CAMELS halo files usually start with comment lines beginning with '#'.
    We stop after max_lines or when normal data rows clearly begin.
    """
    header_lines: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            stripped = line.strip()

            if stripped.startswith("#"):
                header_lines.append(stripped)
            elif stripped == "":
                continue
            else:
                # First non-comment data row reached.
                break

    return header_lines


def extract_float_from_header(header_lines: List[str], key: str) -> Optional[float]:
    """
    Extract a floating-point value from header lines.

    Handles formats like:
        #Omega_M = 0.146200
        # Omega_M = 0.146200
        #Omega_M=0.146200
    """
    pattern = re.compile(
        rf"^\s*#\s*{re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    for line in header_lines:
        match = pattern.search(line)
        if match:
            return float(match.group(1))

    return None


def extract_omega_m(path: Path) -> Optional[float]:
    """
    Extract Omega_M from one raw CAMELS .list file.
    """
    header_lines = read_header_lines(path)
    return extract_float_from_header(header_lines, TARGET_KEY)


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------

def parse_universe_id(path: Path) -> Optional[int]:
    """
    Extract universe id from filename.

    Example:
        LH_0_hlist_1.00000.list -> 0
        LH_57_hlist_0.51209.list -> 57
    """
    match = re.match(r"LH_(\d+)_hlist_", path.name)
    if not match:
        return None

    return int(match.group(1))


def parse_snapshot_value(path: Path) -> Optional[float]:
    """
    Extract snapshot scale factor from filename.

    Example:
        LH_0_hlist_1.00000.list -> 1.00000
        LH_0_hlist_0.51209.list -> 0.51209
    """
    match = re.match(r"LH_\d+_hlist_([0-9.]+)\.list$", path.name)
    if not match:
        return None

    return float(match.group(1))


def find_universe_files(raw_dir: Path, universe_id: int) -> List[Path]:
    """
    Find all snapshot files for one universe.
    """
    files = sorted(
        raw_dir.glob(f"LH_{universe_id}_hlist_*.list"),
        key=lambda p: (
            parse_snapshot_value(p)
            if parse_snapshot_value(p) is not None
            else -1.0
        ),
    )

    return files


def find_available_universe_ids(raw_dir: Path) -> List[int]:
    """
    Find all available LH universe ids in raw directory.
    """
    universe_ids = set()

    for path in raw_dir.glob("LH_*_hlist_*.list"):
        uid = parse_universe_id(path)
        if uid is not None:
            universe_ids.add(uid)

    return sorted(universe_ids)


# ---------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------

def classify_target_status(
    selected_target: Optional[float],
    has_missing_target: bool,
    has_consistent_target: bool,
    enough_snapshots: bool,
) -> str:
    """
    Classify whether one universe target is safe to use.
    """
    if selected_target is None:
        return "CHECK"

    if has_missing_target:
        return "CHECK"

    if not has_consistent_target:
        return "CHECK"

    if not enough_snapshots:
        return "CHECK"

    if not (EXPECTED_OMEGA_M_MIN < selected_target < EXPECTED_OMEGA_M_MAX):
        return "CHECK"

    return "OK"


def inspect_targets(
    raw_dir: Path,
    num_universes: Optional[int] = None,
    required_snapshots: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    Inspect Omega_M targets for multiple universes.

    Returns one row per universe.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    universe_ids = find_available_universe_ids(raw_dir)

    if not universe_ids:
        raise FileNotFoundError(
            f"No CAMELS files found in {raw_dir}. "
            "Expected files like LH_0_hlist_1.00000.list"
        )

    if num_universes is not None:
        universe_ids = universe_ids[:num_universes]

    rows: List[Dict[str, object]] = []

    for uid in universe_ids:
        files = find_universe_files(raw_dir, uid)

        omega_values: List[Tuple[str, Optional[float]]] = []
        snapshot_values: List[float] = []

        for file_path in files:
            snapshot_value = parse_snapshot_value(file_path)
            if snapshot_value is not None:
                snapshot_values.append(snapshot_value)

            omega_m = extract_omega_m(file_path)
            omega_values.append((file_path.name, omega_m))

        valid_omega_values = [v for _, v in omega_values if v is not None]
        unique_omega_values = sorted(set(valid_omega_values))

        has_consistent_target = len(unique_omega_values) == 1
        has_missing_target = len(valid_omega_values) != len(files)

        selected_target: Optional[float]
        if has_consistent_target and unique_omega_values:
            selected_target = unique_omega_values[0]
        else:
            selected_target = None

        enough_snapshots = True
        if required_snapshots is not None:
            enough_snapshots = len(files) >= required_snapshots

        target_in_expected_range = (
            selected_target is not None
            and EXPECTED_OMEGA_M_MIN < selected_target < EXPECTED_OMEGA_M_MAX
        )

        status = classify_target_status(
            selected_target=selected_target,
            has_missing_target=has_missing_target,
            has_consistent_target=has_consistent_target,
            enough_snapshots=enough_snapshots,
        )

        rows.append(
            {
                "universe_id": f"LH_{uid}",
                "universe_index": uid,
                "omega_m": selected_target,
                "num_files": len(files),
                "required_snapshots": required_snapshots,
                "enough_snapshots": enough_snapshots,
                "snapshots": ",".join(str(x) for x in snapshot_values),
                "first_snapshot": snapshot_values[0] if snapshot_values else None,
                "last_snapshot": snapshot_values[-1] if snapshot_values else None,
                "target_key": TARGET_KEY,
                "target_consistent": has_consistent_target,
                "missing_target": has_missing_target,
                "target_in_expected_range": target_in_expected_range,
                "num_unique_targets_found": len(unique_omega_values),
                "unique_targets_found": ",".join(str(x) for x in unique_omega_values),
                "status": status,
                "example_file": files[-1].name if files else "",
                "preprocessing_version_context": PREPROCESSING_VERSION,
                "camels_box_size_context": DEFAULT_BOX_SIZE,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------

def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """
    Save inspection rows to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: List[Dict[str, object]], path: Path, raw_dir: Path) -> None:
    """
    Save inspection rows to JSON with metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src.data.inspect_targets",
        "raw_dir": str(raw_dir),
        "target_key": TARGET_KEY,
        "preprocessing_version_context": PREPROCESSING_VERSION,
        "camels_box_size_context": DEFAULT_BOX_SIZE,
        "num_universes": len(rows),
        "rows": rows,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------

def omega_statistics(rows: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
    """
    Compute summary statistics of valid Omega_m targets.
    """
    values = [
        row["omega_m"]
        for row in rows
        if isinstance(row.get("omega_m"), float)
    ]

    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    arr = np.array(values, dtype=np.float64)

    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def print_summary(rows: List[Dict[str, object]], max_show: int = 20) -> None:
    """
    Print readable target inspection summary.
    """
    print("=" * 100)
    print("CAMELS-SIMBA TARGET INSPECTION")
    print("=" * 100)

    print(f"Target key searched:       {TARGET_KEY}")
    print(f"Preprocessing context:     {PREPROCESSING_VERSION}")
    print(f"CAMELS box size context:   {DEFAULT_BOX_SIZE}")
    print(f"Total universes inspected: {len(rows)}")

    ok_rows = [r for r in rows if r["status"] == "OK"]
    check_rows = [r for r in rows if r["status"] != "OK"]

    print(f"OK universes:              {len(ok_rows)}")
    print(f"Universes needing check:   {len(check_rows)}")

    stats = omega_statistics(rows)

    print("\nOmega_m distribution from valid rows")
    print("-" * 100)
    print(f"Count:                     {stats['count']}")
    print(f"Min:                       {stats['min']}")
    print(f"Max:                       {stats['max']}")
    print(f"Mean:                      {stats['mean']}")
    print(f"Std:                       {stats['std']}")

    print("\nFirst inspected universes")
    print("-" * 100)
    print(
        f"{'Universe':<10} "
        f"{'Files':<7} "
        f"{'First':<9} "
        f"{'Last':<9} "
        f"{'Omega_m':<14} "
        f"{'Consistent':<11} "
        f"{'Status':<8} "
        f"{'Example file'}"
    )
    print("-" * 100)

    for row in rows[:max_show]:
        omega = row["omega_m"]
        omega_str = f"{omega:.8f}" if isinstance(omega, float) else "None"

        print(
            f"{row['universe_id']:<10} "
            f"{row['num_files']:<7} "
            f"{str(row['first_snapshot']):<9} "
            f"{str(row['last_snapshot']):<9} "
            f"{omega_str:<14} "
            f"{str(row['target_consistent']):<11} "
            f"{row['status']:<8} "
            f"{row['example_file']}"
        )

    if check_rows:
        print("\nWARNING: Some universes need checking")
        print("-" * 100)
        for row in check_rows[:max_show]:
            print(
                f"{row['universe_id']} | "
                f"files={row['num_files']} | "
                f"omega_m={row['omega_m']} | "
                f"target_consistent={row['target_consistent']} | "
                f"missing_target={row['missing_target']} | "
                f"enough_snapshots={row['enough_snapshots']} | "
                f"target_in_expected_range={row['target_in_expected_range']} | "
                f"unique_targets={row['unique_targets_found']}"
            )
    else:
        print("\n✅ All inspected universes have valid, consistent Omega_m targets.")

    print("\nInspection complete.")
    print("=" * 100)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect CAMELS-SIMBA raw files and extract Omega_M targets."
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw CAMELS-SIMBA directory containing LH_*_hlist_*.list files.",
    )

    parser.add_argument(
        "--num_universes",
        type=int,
        default=None,
        help="Number of universes to inspect. If omitted, inspect all available universes.",
    )

    parser.add_argument(
        "--required_snapshots",
        type=int,
        default=5,
        help="Minimum number of snapshot files required per universe.",
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional path to save target inspection CSV.",
    )

    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save target inspection JSON.",
    )

    parser.add_argument(
        "--max_show",
        type=int,
        default=20,
        help="Maximum number of universe rows to print.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    rows = inspect_targets(
        raw_dir=raw_dir,
        num_universes=args.num_universes,
        required_snapshots=args.required_snapshots,
    )

    print_summary(rows, max_show=args.max_show)

    if args.save_csv:
        save_csv(rows, Path(args.save_csv))
        print(f"CSV saved to: {args.save_csv}")

    if args.save_json:
        save_json(rows, Path(args.save_json), raw_dir=raw_dir)
        print(f"JSON saved to: {args.save_json}")


if __name__ == "__main__":
    main()