from __future__ import annotations

"""
summarize_experiments.py

Create a clean comparison table for CAMELS-SIMBA thesis experiments.

Purpose
-------
This script collects experiment results from folders such as:

    experiments/mean_baseline_20u_seed123/metrics.json
    experiments/mean_baseline_50u_seed123/metrics.json
    experiments/mean_baseline_100u_seed123/metrics.json

    experiments/static_gcn_20u_seed123/metrics.json
    experiments/static_gcn_50u_seed123/metrics.json
    experiments/static_gcn_100u_seed123/metrics.json

Later it will also support:

    experiments/evolvegcn_h_20u_seed123/metrics.json
    experiments/evolvegcn_h_50u_seed123/metrics.json
    experiments/evolvegcn_h_100u_seed123/metrics.json

The output is a reproducible comparison table saved as:

    outputs/experiment_summary_seed123.csv
    outputs/experiment_summary_seed123.json
    outputs/experiment_summary_seed123.md

This is used for thesis tables, professor updates, and experiment tracking.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Helpers
# ============================================================

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load JSON file safely.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """
    Save data as formatted JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_get_metric(
    metrics: Dict[str, Any],
    split: str,
    key: str,
) -> Optional[float]:
    """
    Safely read a metric value.

    Example:
        safe_get_metric(metrics, "test", "mae")
    """
    try:
        value = metrics.get(split, {}).get(key, None)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_get_int(
    metrics: Dict[str, Any],
    split: str,
    key: str,
) -> Optional[int]:
    """
    Safely read an integer metric value.
    """
    try:
        value = metrics.get(split, {}).get(key, None)
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def infer_model_name(experiment_name: str) -> str:
    """
    Infer model family from experiment folder name.
    """
    name = experiment_name.lower()

    if name.startswith("mean_baseline"):
        return "Mean Baseline"

    if name.startswith("static_gcn"):
        return "Static GCN"

    if name.startswith("evolvegcn_h") or name.startswith("evolve_gcn_h"):
        return "EvolveGCN-H"

    if name.startswith("evolvegcn") or name.startswith("evolve_gcn"):
        return "EvolveGCN"

    if name.startswith("temporal"):
        return "Temporal GNN"

    return "Unknown"


def infer_model_key(experiment_name: str) -> str:
    """
    Infer machine-readable model key from experiment folder name.
    """
    name = experiment_name.lower()

    if name.startswith("mean_baseline"):
        return "mean_baseline"

    if name.startswith("static_gcn"):
        return "static_gcn"

    if name.startswith("evolvegcn_h") or name.startswith("evolve_gcn_h"):
        return "evolvegcn_h"

    if name.startswith("evolvegcn") or name.startswith("evolve_gcn"):
        return "evolvegcn"

    if name.startswith("temporal"):
        return "temporal_gnn"

    return "unknown"


def infer_num_universes(experiment_name: str) -> Optional[int]:
    """
    Extract number of universes from names like:

        mean_baseline_20u_seed123
        static_gcn_100u_seed123
        evolvegcn_h_500u_seed42
    """
    match = re.search(r"_(\d+)u(?:_|$)", experiment_name.lower())

    if match is None:
        return None

    return int(match.group(1))


def infer_seed(experiment_name: str) -> Optional[int]:
    """
    Extract seed from names like:

        mean_baseline_20u_seed123
        static_gcn_100u_seed42
    """
    match = re.search(r"seed(\d+)", experiment_name.lower())

    if match is None:
        return None

    return int(match.group(1))


def infer_dataset_size_label(num_universes: Optional[int]) -> str:
    """
    Create a readable dataset-size label.
    """
    if num_universes is None:
        return "unknown"

    return f"{num_universes}U"


def find_metrics_files(
    experiments_root: str | Path,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """
    Find metrics.json files under the experiments folder.

    By default, this finds every:

        experiments/*/metrics.json

    Optional include/exclude filters can be used later.
    """
    experiments_root = Path(experiments_root)

    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments folder not found: {experiments_root}")

    metrics_files = sorted(experiments_root.glob("*/metrics.json"))

    if include_patterns:
        lowered_patterns = [p.lower() for p in include_patterns]
        metrics_files = [
            path for path in metrics_files
            if any(pattern in path.parent.name.lower() for pattern in lowered_patterns)
        ]

    if exclude_patterns:
        lowered_patterns = [p.lower() for p in exclude_patterns]
        metrics_files = [
            path for path in metrics_files
            if not any(pattern in path.parent.name.lower() for pattern in lowered_patterns)
        ]

    return metrics_files


# ============================================================
# Main parsing logic
# ============================================================

def parse_experiment_metrics(metrics_path: str | Path) -> Dict[str, Any]:
    """
    Parse one experiment metrics.json into one table row.
    """
    metrics_path = Path(metrics_path)
    experiment_dir = metrics_path.parent
    experiment_name = experiment_dir.name

    metrics = load_json(metrics_path)

    model_name = infer_model_name(experiment_name)
    model_key = infer_model_key(experiment_name)
    num_universes = infer_num_universes(experiment_name)
    seed = infer_seed(experiment_name)

    row = {
        # Identification
        "experiment_name": experiment_name,
        "model": model_name,
        "model_key": model_key,
        "num_universes": num_universes,
        "dataset_size": infer_dataset_size_label(num_universes),
        "seed": seed,

        # Train metrics
        "train_mse": safe_get_metric(metrics, "train", "mse"),
        "train_rmse": safe_get_metric(metrics, "train", "rmse"),
        "train_mae": safe_get_metric(metrics, "train", "mae"),
        "train_samples": safe_get_int(metrics, "train", "num_samples"),

        # Validation metrics
        "val_mse": safe_get_metric(metrics, "val", "mse"),
        "val_rmse": safe_get_metric(metrics, "val", "rmse"),
        "val_mae": safe_get_metric(metrics, "val", "mae"),
        "val_samples": safe_get_int(metrics, "val", "num_samples"),

        # Test metrics
        "test_mse": safe_get_metric(metrics, "test", "mse"),
        "test_rmse": safe_get_metric(metrics, "test", "rmse"),
        "test_mae": safe_get_metric(metrics, "test", "mae"),
        "test_samples": safe_get_int(metrics, "test", "num_samples"),

        # Optional model-specific fields
        "best_epoch": metrics.get("best_epoch", None),
        "best_val_mse": metrics.get("best_val_mse", None),
        "trainable_parameters": metrics.get("trainable_parameters", None),
        "train_mean_omega_m": metrics.get("train_mean_omega_m", None),

        # Paths
        "experiment_dir": str(experiment_dir),
        "metrics_path": str(metrics_path),
    }

    return row


def build_summary_table(
    experiments_root: str | Path = "experiments",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build full experiment comparison table.
    """
    metrics_files = find_metrics_files(
        experiments_root=experiments_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    rows: List[Dict[str, Any]] = []

    for metrics_path in metrics_files:
        try:
            row = parse_experiment_metrics(metrics_path)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "experiment_name": metrics_path.parent.name,
                    "model": "ERROR",
                    "model_key": "error",
                    "num_universes": None,
                    "dataset_size": "unknown",
                    "seed": None,
                    "error": str(exc),
                    "experiment_dir": str(metrics_path.parent),
                    "metrics_path": str(metrics_path),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Ensure expected columns exist even if some experiments failed.
    expected_columns = [
        "experiment_name",
        "model",
        "model_key",
        "num_universes",
        "dataset_size",
        "seed",
        "train_mse",
        "train_rmse",
        "train_mae",
        "train_samples",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_samples",
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_samples",
        "best_epoch",
        "best_val_mse",
        "trainable_parameters",
        "train_mean_omega_m",
        "experiment_dir",
        "metrics_path",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    # Sort professionally: dataset size first, then model order.
    model_order = {
        "mean_baseline": 0,
        "static_gcn": 1,
        "evolvegcn_h": 2,
        "evolvegcn": 3,
        "temporal_gnn": 4,
        "unknown": 99,
        "error": 100,
    }

    df["_model_order"] = df["model_key"].map(model_order).fillna(99)

    df = df.sort_values(
        by=["num_universes", "seed", "_model_order", "experiment_name"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    df = df.drop(columns=["_model_order"])

    return df


# ============================================================
# Reporting helpers
# ============================================================

def create_clean_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a smaller thesis-friendly table.
    """
    if df.empty:
        return df

    columns = [
        "dataset_size",
        "model",
        "seed",
        "train_mae",
        "val_mae",
        "test_mae",
        "test_rmse",
        "best_epoch",
        "trainable_parameters",
        "experiment_name",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = None

    display_df = df[columns].copy()

    # Round numeric columns for readability.
    numeric_cols = [
        "train_mae",
        "val_mae",
        "test_mae",
        "test_rmse",
    ]

    for col in numeric_cols:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(6)

    return display_df


def create_ranked_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models within each dataset size by test MAE.
    """
    if df.empty:
        return df

    ranked_df = df.copy()

    ranked_df["test_mae"] = pd.to_numeric(ranked_df["test_mae"], errors="coerce")
    ranked_df["test_rmse"] = pd.to_numeric(ranked_df["test_rmse"], errors="coerce")

    ranked_df = ranked_df.dropna(subset=["test_mae"])

    if ranked_df.empty:
        return ranked_df

    ranked_df["rank_by_test_mae"] = (
        ranked_df
        .groupby(["num_universes", "seed"])["test_mae"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    ranked_df["rank_by_test_rmse"] = (
        ranked_df
        .groupby(["num_universes", "seed"])["test_rmse"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    ranked_df = ranked_df.sort_values(
        by=["num_universes", "seed", "rank_by_test_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return ranked_df


def dataframe_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to Markdown safely.

    Pandas to_markdown requires the optional 'tabulate' package.
    If tabulate is not installed, this function creates a simple Markdown table manually.
    """
    if df.empty:
        return "No rows available."

    try:
        return df.to_markdown(index=False)
    except ImportError:
        columns = list(df.columns)

        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        rows = []
        for _, row in df.iterrows():
            row_values = []
            for col in columns:
                value = row[col]
                if pd.isna(value):
                    row_values.append("")
                else:
                    row_values.append(str(value))
            rows.append("| " + " | ".join(row_values) + " |")

        return "\n".join([header, separator] + rows)


def create_markdown_report(
    display_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> str:
    """
    Create a Markdown report that can be copied into README or thesis notes.
    This version works even if the optional 'tabulate' package is missing.
    """
    lines = []

    lines.append("# CAMELS-SIMBA Experiment Summary")
    lines.append("")
    lines.append("This report summarizes the current experiment results.")
    lines.append("")
    lines.append("Lower MAE and RMSE are better.")
    lines.append("")

    if display_df.empty:
        lines.append("No experiments found.")
        return "\n".join(lines)

    lines.append("## Main Comparison Table")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(display_df))
    lines.append("")

    if not ranked_df.empty:
        ranked_display = ranked_df[
            [
                "dataset_size",
                "model",
                "seed",
                "test_mae",
                "test_rmse",
                "rank_by_test_mae",
                "rank_by_test_rmse",
                "experiment_name",
            ]
        ].copy()

        ranked_display["test_mae"] = pd.to_numeric(
            ranked_display["test_mae"],
            errors="coerce",
        ).round(6)

        ranked_display["test_rmse"] = pd.to_numeric(
            ranked_display["test_rmse"],
            errors="coerce",
        ).round(6)

        lines.append("## Ranking by Test Performance")
        lines.append("")
        lines.append(dataframe_to_markdown_safe(ranked_display))
        lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- The mean baseline predicts the training-set mean of `Omega_m`.")
    lines.append("- Static GCN uses one final-snapshot graph per universe.")
    lines.append("- EvolveGCN-H will use temporal graph sequences across multiple snapshots.")
    lines.append("- The key thesis question is whether graph structure and temporal evolution improve prediction over the simple mean baseline.")
    lines.append("")

    return "\n".join(lines)


def print_console_summary(df: pd.DataFrame, display_df: pd.DataFrame, ranked_df: pd.DataFrame) -> None:
    """
    Print a readable summary in terminal.
    """
    print("=" * 100)
    print("CAMELS-SIMBA EXPERIMENT SUMMARY")
    print("=" * 100)

    print(f"Total experiments found: {len(df)}")

    if df.empty:
        print("No experiments found.")
        return

    print()
    print("Models found:")
    print("-" * 100)
    for model in sorted(df["model"].dropna().unique()):
        print(f"- {model}")

    print()
    print("Dataset sizes found:")
    print("-" * 100)
    for size in sorted(df["num_universes"].dropna().unique()):
        print(f"- {int(size)} universes")

    print()
    print("Main comparison:")
    print("-" * 100)
    print(display_df.to_string(index=False))

    if not ranked_df.empty:
        print()
        print("Best model per dataset size and seed by test MAE:")
        print("-" * 100)

        best_rows = (
            ranked_df[ranked_df["rank_by_test_mae"] == 1]
            .sort_values(by=["num_universes", "seed"])
        )

        cols = [
            "dataset_size",
            "seed",
            "model",
            "test_mae",
            "test_rmse",
            "experiment_name",
        ]

        print(best_rows[cols].to_string(index=False))

    print()
    print("=" * 100)


# ============================================================
# Main execution
# ============================================================

def summarize_experiments(
    experiments_root: str | Path = "experiments",
    output_root: str | Path = "outputs",
    output_name: str = "experiment_summary",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Build and save experiment summaries.

    Returns paths to saved files.
    """
    experiments_root = Path(experiments_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = build_summary_table(
        experiments_root=experiments_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    display_df = create_clean_display_table(df)
    ranked_df = create_ranked_table(df)
    markdown_report = create_markdown_report(display_df, ranked_df)

    csv_path = output_root / f"{output_name}.csv"
    clean_csv_path = output_root / f"{output_name}_clean.csv"
    ranked_csv_path = output_root / f"{output_name}_ranked.csv"
    json_path = output_root / f"{output_name}.json"
    markdown_path = output_root / f"{output_name}.md"

    df.to_csv(csv_path, index=False)
    display_df.to_csv(clean_csv_path, index=False)
    ranked_df.to_csv(ranked_csv_path, index=False)

    save_json(df.to_dict(orient="records"), json_path)

    with markdown_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report)

    print_console_summary(df, display_df, ranked_df)

    print()
    print("Saved files:")
    print("-" * 100)
    print(f"Full CSV:       {csv_path}")
    print(f"Clean CSV:      {clean_csv_path}")
    print(f"Ranked CSV:     {ranked_csv_path}")
    print(f"JSON:           {json_path}")
    print(f"Markdown:       {markdown_path}")

    return {
        "full_csv": str(csv_path),
        "clean_csv": str(clean_csv_path),
        "ranked_csv": str(ranked_csv_path),
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize CAMELS-SIMBA experiment metrics."
    )

    parser.add_argument(
        "--experiments_root",
        type=str,
        default="experiments",
        help="Folder containing experiment subfolders.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Folder where summary files will be saved.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="experiment_summary_seed123",
        help="Base name for output summary files.",
    )

    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional list of substrings. Only experiments containing one of these are included.",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional list of substrings. Experiments containing one of these are excluded.",
    )

    args = parser.parse_args()

    summarize_experiments(
        experiments_root=args.experiments_root,
        output_root=args.output_root,
        output_name=args.output_name,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )


if __name__ == "__main__":
    main()