"""
Inspect one raw CAMELS-SIMBA halo catalog snapshot.

This script is used before graph construction.

It checks:
- whether the raw .list file exists
- header/comment metadata
- number of rows and columns
- confirmed physical feature columns
- missing / NaN / infinite values
- raw Mvir and derived log10(Mvir)
- top-N halo selection by raw Mvir
- position ranges relative to CAMELS box size
- velocity ranges
- first few halo rows

Official preprocessing context:
    v2_logmass_minmax_top100_periodic_knn

Example usage:

python -m src.data.inspect_raw_snapshot \
  --snapshot_path data/raw/CAMELS_SIMBA_100U/LH_0_hlist_1.00000.list \
  --max_rows 5 \
  --top_n 100
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.camels_graph_utils import DEFAULT_BOX_SIZE, PREPROCESSING_VERSION


# ---------------------------------------------------------------------
# Confirmed CAMELS-SIMBA / Rockstar column mapping from the file header
# ---------------------------------------------------------------------

RAW_FEATURE_COLUMNS: Dict[str, int] = {
    "Mvir_raw": 10,
    "X": 17,
    "Y": 18,
    "Z": 19,
    "VX": 20,
    "VY": 21,
    "VZ": 22,
}

FINAL_NODE_FEATURES = [
    "log10_Mvir",
    "X",
    "Y",
    "Z",
    "VX",
    "VY",
    "VZ",
]

POSITION_COLUMNS: Dict[str, int] = {
    "X": 17,
    "Y": 18,
    "Z": 19,
}

VELOCITY_COLUMNS: Dict[str, int] = {
    "VX": 20,
    "VY": 21,
    "VZ": 22,
}

MASS_COLUMN_NAME = "Mvir_raw"
MASS_COLUMN_INDEX = 10


def read_header_lines(path: Path, max_lines: int = 160) -> List[str]:
    """
    Read comment/header lines from a CAMELS-SIMBA .list file.
    """
    header_lines: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for _ in range(max_lines):
            line = file.readline()

            if not line:
                break

            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
            elif line.strip() == "":
                continue
            else:
                break

    return header_lines


def extract_float_from_header(header_lines: List[str], key: str) -> Optional[float]:
    """
    Extract a floating-point metadata value from header lines.

    Handles:
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


def read_raw_snapshot(path: Path) -> pd.DataFrame:
    """
    Read the raw snapshot as a numeric dataframe.

    CAMELS-SIMBA / Rockstar files contain comment lines beginning with '#'.
    The actual halo table is whitespace-separated.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    return df


def require_columns(df: pd.DataFrame) -> None:
    """
    Ensure all columns needed by the thesis preprocessing exist.
    """
    required_indices = list(RAW_FEATURE_COLUMNS.values())
    max_required = max(required_indices)

    if df.shape[1] <= max_required:
        raise ValueError(
            f"Snapshot has {df.shape[1]} columns, but column index "
            f"{max_required} is required."
        )


def clean_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean only the required thesis columns.

    Removes rows with:
        - NaN / Inf in required columns
        - Mvir <= 0
    """
    require_columns(df)

    required_indices = list(RAW_FEATURE_COLUMNS.values())

    clean = df.copy()
    clean[required_indices] = clean[required_indices].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    clean = clean.dropna(subset=required_indices)
    clean = clean[clean.iloc[:, MASS_COLUMN_INDEX].astype(float) > 0]

    return clean.reset_index(drop=True)


def select_top_by_raw_mvir(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Select top-N halos by raw Mvir.
    """
    require_columns(df)

    selected = (
        df.sort_values(by=MASS_COLUMN_INDEX, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return selected


def print_header_summary(header_lines: List[str]) -> None:
    """
    Print important header lines from the raw snapshot.
    """
    print("\nHeader / metadata preview")
    print("-" * 90)

    if not header_lines:
        print("No header lines found.")
        return

    omega_m = extract_float_from_header(header_lines, "Omega_M")
    sigma_8 = extract_float_from_header(header_lines, "sigma_8")

    print(f"Extracted Omega_M: {omega_m}")
    print(f"Extracted sigma_8: {sigma_8}")
    print()

    for line in header_lines[:40]:
        print(line)

    print(f"\nTotal header/comment lines read: {len(header_lines)}")


def print_dataset_summary(path: Path, df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """
    Print basic dataframe information.
    """
    print("\nRaw snapshot summary")
    print("-" * 90)
    print(f"File: {path}")
    print(f"Rows / halos before cleaning: {df.shape[0]}")
    print(f"Rows / halos after cleaning:  {clean_df.shape[0]}")
    print(f"Rows removed:                {df.shape[0] - clean_df.shape[0]}")
    print(f"Columns:                     {df.shape[1]}")
    print(f"Memory usage:                {df.memory_usage(deep=True).sum() / 1024**2:.3f} MB")

    numeric_values = df.to_numpy(dtype=np.float64, copy=False)

    nan_count = np.isnan(numeric_values).sum()
    inf_count = np.isinf(numeric_values).sum()

    print(f"Raw NaN count:               {nan_count}")
    print(f"Raw Inf count:               {inf_count}")


def print_selected_feature_summary(df: pd.DataFrame) -> None:
    """
    Print statistics for the selected thesis features.
    """
    print("\nSelected thesis feature columns")
    print("-" * 90)
    print("Raw columns read from CAMELS/Rockstar:")
    print(", ".join(RAW_FEATURE_COLUMNS.keys()))

    print("\nFinal node feature order after preprocessing:")
    print(", ".join(FINAL_NODE_FEATURES))

    print("\nImportant distinction:")
    print("  Selection uses: raw Mvir")
    print("  Node feature uses: log10(Mvir)")

    print("\nColumn mapping")
    print("-" * 90)

    for name, col_idx in RAW_FEATURE_COLUMNS.items():
        print(f"{name:<10} -> raw column index {col_idx}")

    print("\nRaw feature statistics")
    print("-" * 90)

    rows = []

    for name, col_idx in RAW_FEATURE_COLUMNS.items():
        if col_idx >= df.shape[1]:
            rows.append(
                {
                    "feature": name,
                    "column": col_idx,
                    "status": "MISSING",
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
            continue

        values = df.iloc[:, col_idx].astype(float)

        rows.append(
            {
                "feature": name,
                "column": col_idx,
                "status": "OK",
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(ddof=0),
            }
        )

    stats_df = pd.DataFrame(rows)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.6g}".format,
    ):
        print(stats_df.to_string(index=False))


def print_logmass_summary(df: pd.DataFrame, label: str) -> None:
    """
    Print raw Mvir and log10(Mvir) statistics.
    """
    raw_mvir = df.iloc[:, MASS_COLUMN_INDEX].astype(float).to_numpy()
    log_mvir = np.log10(raw_mvir)

    print(f"\nMass / log-mass summary: {label}")
    print("-" * 90)
    print(f"Raw Mvir min:        {raw_mvir.min():.6g}")
    print(f"Raw Mvir max:        {raw_mvir.max():.6g}")
    print(f"Raw Mvir mean:       {raw_mvir.mean():.6g}")
    print(f"log10(Mvir) min:     {log_mvir.min():.6g}")
    print(f"log10(Mvir) max:     {log_mvir.max():.6g}")
    print(f"log10(Mvir) mean:    {log_mvir.mean():.6g}")
    print(f"log10(Mvir) std:     {log_mvir.std():.6g}")


def print_first_rows(df: pd.DataFrame, max_rows: int) -> None:
    """
    Print first rows for quick inspection.
    """
    print(f"\nFirst {max_rows} rows, selected raw columns + derived log10(Mvir)")
    print("-" * 90)

    selected = pd.DataFrame()

    raw_mvir = df.iloc[:, MASS_COLUMN_INDEX].astype(float)
    selected["log10_Mvir"] = np.log10(raw_mvir)

    for name, col_idx in RAW_FEATURE_COLUMNS.items():
        selected[name] = df.iloc[:, col_idx]

    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.6g}".format,
    ):
        print(selected.head(max_rows).to_string(index=False))


def print_position_velocity_ranges(df: pd.DataFrame) -> None:
    """
    Print position and velocity range checks.
    """
    print("\nPosition range check")
    print("-" * 90)

    for name, col_idx in POSITION_COLUMNS.items():
        values = df.iloc[:, col_idx].astype(float)
        vmin = values.min()
        vmax = values.max()
        print(
            f"{name}: min={vmin:.6g}, "
            f"max={vmax:.6g}, "
            f"mean={values.mean():.6g}, "
            f"inside [0, {DEFAULT_BOX_SIZE}] = {vmin >= 0 and vmax <= DEFAULT_BOX_SIZE}"
        )

    print("\nPeriodic box context")
    print("-" * 90)
    print(f"CAMELS box size assumed by preprocessing: {DEFAULT_BOX_SIZE}")
    print("Graph construction uses periodic boundary-aware distances in v2.")

    print("\nVelocity range check")
    print("-" * 90)

    for name, col_idx in VELOCITY_COLUMNS.items():
        values = df.iloc[:, col_idx].astype(float)
        print(
            f"{name}: min={values.min():.6g}, "
            f"max={values.max():.6g}, "
            f"mean={values.mean():.6g}, "
            f"std={values.std(ddof=0):.6g}"
        )


def print_top_selection_summary(clean_df: pd.DataFrame, top_df: pd.DataFrame, top_n: int) -> None:
    """
    Print information about top-N halo selection by raw Mvir.
    """
    print(f"\nTop-{top_n} selection by raw Mvir")
    print("-" * 90)

    print(f"Valid halos available:       {len(clean_df)}")
    print(f"Selected halos:              {len(top_df)}")
    print("Selection rule:              descending raw Mvir")
    print("Feature mass after select:   log10(Mvir)")

    if len(top_df) > 0:
        threshold = top_df.iloc[:, MASS_COLUMN_INDEX].astype(float).min()
        print(f"Lowest raw Mvir in selected: {threshold:.6g}")


def inspect_raw_snapshot(
    snapshot_path: Path,
    max_rows: int,
    top_n: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run the complete raw snapshot inspection.
    """
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    print("=" * 90)
    print("CAMELS-SIMBA RAW SNAPSHOT INSPECTION")
    print("=" * 90)
    print(f"Preprocessing context: {PREPROCESSING_VERSION}")

    header_lines = read_header_lines(snapshot_path)
    df = read_raw_snapshot(snapshot_path)

    require_columns(df)

    clean_df = clean_required_columns(df)
    top_df = select_top_by_raw_mvir(clean_df, top_n=top_n)

    print_header_summary(header_lines)
    print_dataset_summary(snapshot_path, df, clean_df)
    print_selected_feature_summary(clean_df)
    print_logmass_summary(clean_df, label="all valid halos")
    print_top_selection_summary(clean_df, top_df, top_n=top_n)
    print_logmass_summary(top_df, label=f"top {top_n} halos by raw Mvir")
    print_position_velocity_ranges(clean_df)
    print_first_rows(top_df, max_rows=max_rows)

    print("\nInterpretation checklist")
    print("-" * 90)

    if df.shape[1] > max(RAW_FEATURE_COLUMNS.values()):
        print("✅ Snapshot has enough columns for selected thesis features.")
    else:
        print("❌ Snapshot does not have enough columns for selected thesis features.")

    numeric_values = df.to_numpy(dtype=np.float64, copy=False)

    if np.isnan(numeric_values).sum() == 0:
        print("✅ No NaN values found in raw snapshot.")
    else:
        print("⚠️ NaN values found in raw snapshot.")

    if np.isinf(numeric_values).sum() == 0:
        print("✅ No infinite values found in raw snapshot.")
    else:
        print("⚠️ Infinite values found in raw snapshot.")

    if len(top_df) == top_n:
        print(f"✅ Top-{top_n} halo selection is available without padding.")
    else:
        print(f"⚠️ Only {len(top_df)} valid halos available; padding would be needed.")

    if top_df.iloc[:, MASS_COLUMN_INDEX].astype(float).min() > 0:
        print("✅ All selected halos have positive raw Mvir, so log10(Mvir) is valid.")
    else:
        print("❌ Non-positive mass found in selected halos.")

    print("✅ Raw snapshot inspection complete.")
    print("=" * 90)

    return df, header_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one raw CAMELS-SIMBA halo snapshot."
    )

    parser.add_argument(
        "--snapshot_path",
        type=str,
        required=True,
        help="Path to one raw CAMELS-SIMBA .list file.",
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=5,
        help="Number of first selected rows to print.",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Number of halos selected by raw Mvir for graph construction.",
    )

    args = parser.parse_args()

    inspect_raw_snapshot(
        snapshot_path=Path(args.snapshot_path),
        max_rows=args.max_rows,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()