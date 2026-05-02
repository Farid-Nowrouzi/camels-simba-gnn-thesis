from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from src.data.camels_graph_utils import (
    DEFAULT_BOX_SIZE,
    FEATURE_NAMES,
    PREPROCESSING_VERSION,
)


def describe_value(name: str, value: Any, indent: str = "") -> None:
    """
    Recursively describe a Python object from the saved dataset.
    """
    prefix = f"{indent}{name}"

    if torch.is_tensor(value):
        print(
            f"{prefix}: tensor shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )

    elif isinstance(value, list):
        print(f"{prefix}: list length={len(value)}")
        if len(value) > 0:
            first = value[0]
            if torch.is_tensor(first):
                print(
                    f"{indent}  first item: tensor shape={tuple(first.shape)}, "
                    f"dtype={first.dtype}, device={first.device}"
                )
            elif isinstance(first, dict):
                print(f"{indent}  first item: dict keys={list(first.keys())}")
            else:
                print(f"{indent}  first item type={type(first)}")

    elif isinstance(value, dict):
        print(f"{prefix}: dict with keys={list(value.keys())}")
        for k, v in value.items():
            describe_value(str(k), v, indent=indent + "  ")

    else:
        display_value = value if isinstance(value, (int, float, str, bool)) else ""
        print(f"{prefix}: type={type(value)} {display_value}")


def check_sample(sample_key: str, sample: dict) -> None:
    """
    Print thesis-specific checks for one temporal sequence sample.
    """
    print("\n" + "=" * 80)
    print(f"THESIS PIPELINE CHECKS FOR SAMPLE: {sample_key}")
    print("=" * 80)

    preprocessing_version = sample.get("preprocessing_version")
    feature_names = sample.get("feature_names")
    mass_feature = sample.get("mass_feature")
    node_selection = sample.get("node_selection")
    graph_positions = sample.get("graph_positions")
    periodic_boundary = sample.get("periodic_boundary")
    periodic_boundary_knn = sample.get("periodic_boundary_knn")
    box_size = sample.get("box_size")
    normalization = sample.get("normalization")
    graph_mode = sample.get("graph_mode")

    print(f"Preprocessing version: {preprocessing_version}")
    print(f"Expected version:      {PREPROCESSING_VERSION}")
    print(f"Feature names:         {feature_names}")
    print(f"Expected features:     {FEATURE_NAMES}")
    print(f"Mass feature:          {mass_feature}")
    print(f"Node selection:        {node_selection}")
    print(f"Graph positions:       {graph_positions}")
    print(f"Normalization:         {normalization}")
    print(f"Graph mode:            {graph_mode}")
    print(f"Periodic boundary:     {periodic_boundary}")
    print(f"Periodic boundary kNN: {periodic_boundary_knn}")
    print(f"Box size:              {box_size}")

    A_list = sample.get("A_list")
    X_list = sample.get("Nodes_list")
    mask_list = sample.get("mask_list")
    target = sample.get("target")

    if A_list is not None:
        print(f"A_list length:         {len(A_list)}")
    if X_list is not None:
        print(f"Nodes_list length:     {len(X_list)}")
    if mask_list is not None:
        print(f"mask_list length:      {len(mask_list)}")

    if target is not None:
        print(f"Target:                {target}")

    if A_list and X_list and mask_list:
        A0 = A_list[0].detach().cpu()
        X0 = X_list[0].detach().cpu()
        M0 = mask_list[0].detach().cpu()

        print("\nFirst snapshot tensor checks")
        print("-" * 80)
        print(f"A0 shape:              {tuple(A0.shape)}")
        print(f"X0 shape:              {tuple(X0.shape)}")
        print(f"mask0 shape:           {tuple(M0.shape)}")
        print(f"X0 min/max:            {X0.min().item():.6g} / {X0.max().item():.6g}")
        print(f"A0 symmetric:          {bool((A0 == A0.T).all())}")
        print(f"Real nodes:            {int(M0.sum().item())}")
        print(f"Undirected edges:      {int(A0.sum().item() / 2)}")
        print(f"NaN in X0:             {bool(torch.isnan(X0).any())}")
        print(f"Inf in X0:             {bool(torch.isinf(X0).any())}")

    snapshots = sample.get("snapshots")
    if isinstance(snapshots, list) and len(snapshots) > 0:
        first_snapshot_metadata = snapshots[0]
        print("\nFirst snapshot metadata")
        print("-" * 80)
        print(first_snapshot_metadata)

    print("\nChecklist")
    print("-" * 80)

    if preprocessing_version == PREPROCESSING_VERSION:
        print("✅ Preprocessing version matches current official version.")
    else:
        print("⚠️ Preprocessing version mismatch.")

    if feature_names == FEATURE_NAMES:
        print("✅ Feature names match expected feature order.")
    else:
        print("⚠️ Feature names do not match expected feature order.")

    if mass_feature == "log10_Mvir":
        print("✅ Mass feature is log10_Mvir.")
    else:
        print("⚠️ Mass feature is not log10_Mvir.")

    if node_selection == "top_num_nodes_by_raw_Mvir_descending":
        print("✅ Node selection uses raw Mvir ranking.")
    else:
        print("⚠️ Node selection rule is unexpected.")

    if graph_positions == "raw_physical_XYZ_before_feature_normalization":
        print("✅ Graph construction uses raw physical positions.")
    else:
        print("⚠️ Graph position rule is unexpected.")

    if periodic_boundary is True:
        print("✅ Periodic boundary-aware distances are enabled.")
    else:
        print("⚠️ Periodic boundary-aware distances are disabled.")

    if graph_mode == "knn":
        if periodic_boundary_knn is True:
            print("✅ kNN uses periodic boundary-aware distances.")
        else:
            print("⚠️ kNN does not report periodic boundary-aware distances.")

    if box_size == DEFAULT_BOX_SIZE:
        print(f"✅ Box size matches CAMELS default: {DEFAULT_BOX_SIZE}.")
    else:
        print(f"⚠️ Box size differs from CAMELS default: {DEFAULT_BOX_SIZE}.")

    if A_list and X_list and mask_list:
        if len(A_list) == len(X_list) == len(mask_list):
            print("✅ Temporal lists have matching lengths.")
        else:
            print("⚠️ Temporal list lengths do not match.")

        if X_list[0].shape[1] == 7:
            print("✅ Node feature dimension is 7.")
        else:
            print(f"⚠️ Node feature dimension is {X_list[0].shape[1]}, expected 7.")

        if bool((A_list[0].detach().cpu() == A_list[0].detach().cpu().T).all()):
            print("✅ First adjacency matrix is symmetric.")
        else:
            print("⚠️ First adjacency matrix is not symmetric.")

        if not torch.isnan(X_list[0].detach().cpu()).any():
            print("✅ No NaN values in first node feature matrix.")
        else:
            print("⚠️ NaN values found in first node feature matrix.")

        if not torch.isinf(X_list[0].detach().cpu()).any():
            print("✅ No Inf values in first node feature matrix.")
        else:
            print("⚠️ Inf values found in first node feature matrix.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect saved CAMELS temporal graph sequence files."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a saved .pt file",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Print recursive details for the first sample.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"File not found: {args.path}")

    print("=" * 80)
    print(f"Loading file: {args.path}")
    print("=" * 80)

    data = torch.load(args.path, map_location="cpu", weights_only=False)

    print(f"Top-level object type: {type(data)}")

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Number of top-level keys: {len(keys)}")
        print(f"First 10 keys: {keys[:10]}")

        if len(keys) == 0:
            print("The dictionary is empty.")
            return

        first_key = keys[0]
        first_sample = data[first_key]

        print("\n" + "=" * 80)
        print(f"First sample key: {first_key}")
        print(f"First sample type: {type(first_sample)}")
        print("=" * 80)

        if isinstance(first_sample, dict):
            check_sample(first_key, first_sample)

            if args.deep:
                print("\n" + "=" * 80)
                print("DEEP STRUCTURE")
                print("=" * 80)
                describe_value("sample", first_sample)
        else:
            describe_value("sample", first_sample)

    elif isinstance(data, list):
        print(f"List length: {len(data)}")

        if len(data) == 0:
            print("The list is empty.")
            return

        first_sample = data[0]

        print("\n" + "=" * 80)
        print("First list item")
        print(f"First sample type: {type(first_sample)}")
        print("=" * 80)

        if isinstance(first_sample, dict):
            check_sample("list[0]", first_sample)

            if args.deep:
                print("\n" + "=" * 80)
                print("DEEP STRUCTURE")
                print("=" * 80)
                describe_value("sample", first_sample)
        else:
            describe_value("sample", first_sample)

    else:
        print("Unknown saved structure.")
        print(data)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()