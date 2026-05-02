from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    MASS_COLUMN,
    POSITION_COLUMNS,
    PREPROCESSING_VERSION,
    VELOCITY_COLUMNS,
    build_universe_sequence,
)


def parse_universe_id(value: object) -> int:
    """
    Convert universe identifiers into integer IDs.

    Supported examples:
        0       -> 0
        "0"     -> 0
        "LH_0"  -> 0
        "lh_0"  -> 0
    """
    value_str = str(value).strip()

    if value_str.lower().startswith("lh_"):
        value_str = value_str.split("_", 1)[1]

    return int(value_str)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Find a column using flexible candidate names.

    This avoids breaking when CSV headers are slightly different.
    """
    normalized_to_original = {
        col.strip().lower(): col for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_to_original:
            return normalized_to_original[key]

    raise ValueError(
        f"None of the expected columns were found.\n"
        f"Expected one of: {candidates}\n"
        f"Actual columns: {list(df.columns)}"
    )


def load_targets_csv(path: str | Path) -> Dict[int, float]:
    """
    Load Omega_m labels from a CSV file.

    Accepted universe columns:
        universe_id
        Universe
        universe
        Universe_ID
        lh_id
        LH

    Accepted Omega_m columns:
        omega_m
        Omega_m
        Omega_M
        target
        Target

    Accepted universe ID values:
        0
        1
        2
        LH_0
        LH_1
        LH_2
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Target CSV not found: {path}")

    df = pd.read_csv(path)

    universe_col = find_column(
        df,
        candidates=[
            "universe_id",
            "Universe",
            "universe",
            "Universe_ID",
            "lh_id",
            "LH",
        ],
    )

    omega_col = find_column(
        df,
        candidates=[
            "omega_m",
            "Omega_m",
            "Omega_M",
            "target",
            "Target",
        ],
    )

    targets: Dict[int, float] = {}

    for _, row in df.iterrows():
        universe_id = parse_universe_id(row[universe_col])
        omega_m = float(row[omega_col])
        targets[universe_id] = omega_m

    print(f"Loaded {len(targets)} Omega_m targets from: {path}")
    print(f"Universe column: {universe_col}")
    print(f"Omega_m column:  {omega_col}")

    return targets


def save_json(path: str | Path, data: Dict[str, object]) -> None:
    """
    Save metadata/configuration as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_temporal_dataset(
    raw_dir: str | Path,
    output_path: str | Path,
    num_universes: int,
    num_snapshots: int = 5,
    num_nodes: int = 100,
    normalization: str = "none",
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    targets_csv: Optional[str | Path] = None,
    dummy_target: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Build temporal graph sequences from raw CAMELS-SIMBA hlist files.

    Each universe becomes one sample:

        LH_0:
            A_list      -> temporal list of adjacency matrices
            Nodes_list  -> temporal list of node-feature matrices
            mask_list   -> temporal list of node masks
            target      -> Omega_m
            snapshots   -> selected snapshot metadata

    Official preprocessing version:
        v2_logmass_minmax_top100_periodic_knn

    Node selection:
        Top num_nodes halos by raw Mvir per snapshot.

    Node features:
        [log10_Mvir, X, Y, Z, VX, VY, VZ]

    Graph construction:
        kNN or radius graph from raw physical X/Y/Z positions.

    Periodic boundary:
        Enabled by default.
        For CAMELS, the default box size is 25.0 h^-1 Mpc.

    Important:
        Use --dummy_target only for testing.
        Use --targets_csv for real Omega_m training datasets.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if periodic_boundary and (box_size is None or box_size <= 0):
        raise ValueError(
            "box_size must be positive when periodic_boundary=True."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if targets_csv is not None:
        targets = load_targets_csv(targets_csv)
        target_mode = "real_targets_csv"
    else:
        targets = {}
        target_mode = "dummy_target"

    if targets_csv is None and dummy_target is None:
        raise ValueError(
            "You must provide either --targets_csv for real Omega_m values "
            "or --dummy_target for testing only."
        )

    dataset: Dict[str, object] = {}
    failed_universes = []

    print("=" * 90)
    print("CAMELS-SIMBA TEMPORAL SEQUENCE BUILDER")
    print("=" * 90)
    print(f"Preprocessing version: {PREPROCESSING_VERSION}")
    print(f"Raw directory:         {raw_dir}")
    print(f"Output path:           {output_path}")
    print(f"Number universes:      {num_universes}")
    print(f"Number snapshots:      {num_snapshots}")
    print(f"Number nodes:          {num_nodes}")
    print(f"Normalization:         {normalization}")
    print(f"Graph mode:            {graph_mode}")
    print(f"k:                     {k}")
    print(f"Radius:                {radius}")
    print(f"Periodic boundary:     {periodic_boundary}")
    print(f"Box size:              {box_size}")
    print(f"Device:                {device}")
    print(f"Target mode:           {target_mode}")
    print()
    print("Feature definition:")
    print(f"  Feature names:       {FEATURE_NAMES}")
    print(f"  Feature columns:     {FEATURE_COLUMNS}")
    print(f"  Mass column:         {MASS_COLUMN}")
    print("  Mass feature:        log10_Mvir")
    print("  Node selection:      top_num_nodes_by_raw_Mvir_descending")
    print("  Graph positions:     raw_physical_XYZ_before_feature_normalization")
    print(f"  Position columns:    {POSITION_COLUMNS}")
    print(f"  Velocity columns:    {VELOCITY_COLUMNS}")
    print("=" * 90)

    if dummy_target is not None:
        print()
        print("WARNING:")
        print("  You are using --dummy_target.")
        print("  This is acceptable only for testing the pipeline structure.")
        print("  Do not use dummy targets for scientific results.")
        print()

    for universe_id in range(num_universes):
        universe_key = f"LH_{universe_id}"

        if universe_id in targets:
            target = targets[universe_id]
        else:
            target = dummy_target

        if target is None:
            error_message = "Missing Omega_m target"
            failed_universes.append((universe_key, error_message))
            print(f"[FAILED] {universe_key}: {error_message}")
            continue

        try:
            sequence = build_universe_sequence(
                raw_dir=raw_dir,
                universe_id=universe_id,
                target=target,
                num_snapshots=num_snapshots,
                num_nodes=num_nodes,
                normalization=normalization,
                graph_mode=graph_mode,
                k=k,
                radius=radius,
                periodic_boundary=periodic_boundary,
                box_size=box_size,
                device=device,
            )

            # --------------------------------------------------------
            # Safety checks for official preprocessing version.
            # --------------------------------------------------------
            if sequence.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError(
                    f"Unexpected preprocessing version for {universe_key}: "
                    f"{sequence.get('preprocessing_version')}. "
                    f"Expected: {PREPROCESSING_VERSION}"
                )

            if sequence.get("feature_names") != FEATURE_NAMES:
                raise ValueError(
                    f"Unexpected feature names for {universe_key}: "
                    f"{sequence.get('feature_names')}. "
                    f"Expected: {FEATURE_NAMES}"
                )

            if sequence.get("mass_feature") != "log10_Mvir":
                raise ValueError(
                    f"Unexpected mass feature for {universe_key}: "
                    f"{sequence.get('mass_feature')}. Expected: log10_Mvir"
                )

            if sequence.get("node_selection") != "top_num_nodes_by_raw_Mvir_descending":
                raise ValueError(
                    f"Unexpected node selection for {universe_key}: "
                    f"{sequence.get('node_selection')}"
                )

            if (
                sequence.get("graph_positions")
                != "raw_physical_XYZ_before_feature_normalization"
            ):
                raise ValueError(
                    f"Unexpected graph position rule for {universe_key}: "
                    f"{sequence.get('graph_positions')}"
                )

            if sequence.get("periodic_boundary") != periodic_boundary:
                raise ValueError(
                    f"Unexpected periodic_boundary value for {universe_key}: "
                    f"{sequence.get('periodic_boundary')}. "
                    f"Expected: {periodic_boundary}"
                )

            if graph_mode.lower() == "knn":
                expected_periodic_knn = bool(periodic_boundary)
                if sequence.get("periodic_boundary_knn") != expected_periodic_knn:
                    raise ValueError(
                        f"Unexpected periodic_boundary_knn value for {universe_key}: "
                        f"{sequence.get('periodic_boundary_knn')}. "
                        f"Expected: {expected_periodic_knn}"
                    )

            if float(sequence.get("box_size")) != float(box_size):
                raise ValueError(
                    f"Unexpected box_size for {universe_key}: "
                    f"{sequence.get('box_size')}. Expected: {box_size}"
                )

            dataset[universe_key] = sequence

            first_a = sequence["A_list"][0]
            first_x = sequence["Nodes_list"][0]
            first_mask = sequence["mask_list"][0]

            snapshot_values = [
                item["snapshot_value"] for item in sequence["snapshots"]
            ]

            print(
                f"[OK] {universe_key} | "
                f"snapshots={snapshot_values} | "
                f"A={tuple(first_a.shape)} | "
                f"X={tuple(first_x.shape)} | "
                f"mask={tuple(first_mask.shape)} | "
                f"target={float(sequence['target']):.6f} | "
                f"features={sequence['feature_names']} | "
                f"version={sequence['preprocessing_version']} | "
                f"periodic={sequence['periodic_boundary']} | "
                f"box={sequence['box_size']}"
            )

        except Exception as exc:
            failed_universes.append((universe_key, str(exc)))
            print(f"[FAILED] {universe_key}: {exc}")

    if len(dataset) == 0:
        raise RuntimeError(
            "No universes were successfully processed. "
            "Check raw_dir, file names, column mapping, and targets."
        )

    torch.save(dataset, output_path)

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_dir": str(raw_dir),
        "output_path": str(output_path),
        "num_universes_requested": num_universes,
        "num_universes_successful": len(dataset),
        "num_universes_failed": len(failed_universes),
        "num_snapshots": num_snapshots,
        "num_nodes": num_nodes,
        "normalization": normalization,
        "graph_mode": graph_mode,
        "k": k,
        "radius": radius,
        "periodic_boundary": periodic_boundary,
        "periodic_boundary_knn": bool(periodic_boundary and graph_mode.lower() == "knn"),
        "box_size": box_size,
        "targets_csv": str(targets_csv) if targets_csv is not None else None,
        "target_mode": target_mode,
        "used_dummy_target": dummy_target is not None,
        "dummy_target": dummy_target,
        "device": device,
        "feature_names": FEATURE_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "mass_column": MASS_COLUMN,
        "mass_feature": "log10_Mvir",
        "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "position_columns": POSITION_COLUMNS,
        "velocity_columns": VELOCITY_COLUMNS,
        "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "failed_universes": failed_universes,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    save_json(metadata_path, metadata)

    print()
    print("=" * 90)
    print("TEMPORAL DATASET BUILD COMPLETE")
    print("=" * 90)
    print(f"Saved dataset:        {output_path}")
    print(f"Saved metadata:       {metadata_path}")
    print(f"Successful universes: {len(dataset)}")
    print(f"Failed universes:     {len(failed_universes)}")
    print(f"Preprocessing:        {PREPROCESSING_VERSION}")
    print(f"Feature names:        {FEATURE_NAMES}")
    print("Mass feature:         log10_Mvir")
    print("Node selection:       top_num_nodes_by_raw_Mvir_descending")
    print("Graph positions:      raw_physical_XYZ_before_feature_normalization")
    print(f"Periodic boundary:    {periodic_boundary}")
    print(f"Periodic kNN:         {bool(periodic_boundary and graph_mode.lower() == 'knn')}")
    print(f"Box size:             {box_size}")

    if failed_universes:
        print()
        print("Failed universe details:")
        for universe_key, error_message in failed_universes:
            print(f"  - {universe_key}: {error_message}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CAMELS-SIMBA temporal graph sequences."
    )

    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--num_universes", type=int, required=True)
    parser.add_argument("--num_snapshots", type=int, default=5)
    parser.add_argument("--num_nodes", type=int, default=100)

    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "minmax", "zscore"],
        help=(
            "Use minmax for the official "
            "v2_logmass_minmax_top100_periodic_knn preprocessing configuration."
        ),
    )

    parser.add_argument(
        "--graph_mode",
        type=str,
        default="knn",
        choices=["knn", "radius"],
    )

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--radius", type=float, default=None)

    parser.add_argument(
        "--periodic_boundary",
        action="store_true",
        default=True,
        help=(
            "Use periodic boundary-aware distances. "
            "Enabled by default for CAMELS."
        ),
    )

    parser.add_argument(
        "--no_periodic_boundary",
        action="store_false",
        dest="periodic_boundary",
        help=(
            "Disable periodic boundary-aware distances. "
            "Use only for ablation/debugging."
        ),
    )

    parser.add_argument(
        "--box_size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="CAMELS box size. Default: 25.0 h^-1 Mpc.",
    )

    parser.add_argument("--targets_csv", type=str, default=None)
    parser.add_argument("--dummy_target", type=float, default=None)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    build_temporal_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        num_universes=args.num_universes,
        num_snapshots=args.num_snapshots,
        num_nodes=args.num_nodes,
        normalization=args.normalization,
        graph_mode=args.graph_mode,
        k=args.k,
        radius=args.radius,
        periodic_boundary=args.periodic_boundary,
        box_size=args.box_size,
        targets_csv=args.targets_csv,
        dummy_target=args.dummy_target,
        device=args.device,
    )


if __name__ == "__main__":
    main()