from __future__ import annotations

"""
diagnose_split_target_distribution.py

Pure diagnostic for checking whether train/validation/test splits cover the
same Omega_m target distribution.

This script does not train, change preprocessing, alter datasets, edit
checkpoints, modify splits, or change model behavior. It loads split IDs from
an experiment config and reports target distribution overlap and coverage.
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.evaluation.diagnose_evolvegcn_h_representations import (
    load_json,
    save_json,
    validate_split_ids,
)
from src.training.train_evolvegcn_h import load_temporal_dataset


DEFAULT_QUANTILES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]


def get_target(sample: Dict[str, Any]) -> float:
    target = sample["target"]

    if hasattr(target, "detach"):
        return float(target.detach().cpu().view(-1)[0].item())

    if isinstance(target, (list, tuple)):
        return float(target[0])

    return float(target)


def extract_targets(
    data: Dict[str, Dict[str, Any]],
    universe_ids: List[str],
) -> np.ndarray:
    return np.asarray([get_target(data[universe_id]) for universe_id in universe_ids])


def distribution_summary(values: np.ndarray) -> Dict[str, Any]:
    return {
        "num_samples": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": {
            f"{quantile:.2f}": float(np.quantile(values, quantile))
            for quantile in DEFAULT_QUANTILES
        },
    }


def ks_2samp(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test with an asymptotic p-value.
    """
    left = np.sort(left)
    right = np.sort(right)
    values = np.concatenate([left, right])

    left_cdf = np.searchsorted(left, values, side="right") / len(left)
    right_cdf = np.searchsorted(right, values, side="right") / len(right)
    statistic = float(np.max(np.abs(left_cdf - right_cdf)))

    effective_n = len(left) * len(right) / (len(left) + len(right))
    lambda_value = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic

    terms = [
        ((-1) ** (j - 1)) * math.exp(-2.0 * (lambda_value ** 2) * (j ** 2))
        for j in range(1, 101)
    ]
    p_value = max(0.0, min(1.0, 2.0 * sum(terms)))

    return {
        "statistic": statistic,
        "p_value_asymptotic": float(p_value),
    }


def histogram_counts(
    values_by_split: Dict[str, np.ndarray],
    num_bins: int,
) -> Tuple[List[float], Dict[str, List[int]]]:
    all_values = np.concatenate(list(values_by_split.values()))
    bin_edges = np.linspace(float(np.min(all_values)), float(np.max(all_values)), num_bins + 1)

    counts = {}
    for split, values in values_by_split.items():
        split_counts, _ = np.histogram(values, bins=bin_edges)
        counts[split] = split_counts.astype(int).tolist()

    return bin_edges.tolist(), counts


def histogram_overlap(
    left: np.ndarray,
    right: np.ndarray,
    bin_edges: np.ndarray,
) -> Dict[str, float]:
    left_counts, _ = np.histogram(left, bins=bin_edges)
    right_counts, _ = np.histogram(right, bins=bin_edges)

    left_prob = left_counts / max(1, int(np.sum(left_counts)))
    right_prob = right_counts / max(1, int(np.sum(right_counts)))
    overlap = float(np.minimum(left_prob, right_prob).sum())

    return {
        "histogram_overlap_fraction": overlap,
        "histogram_overlap_percent": 100.0 * overlap,
    }


def interval_overlap(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    left_min = float(np.min(left))
    left_max = float(np.max(left))
    right_min = float(np.min(right))
    right_max = float(np.max(right))

    overlap_min = max(left_min, right_min)
    overlap_max = min(left_max, right_max)
    overlap_width = max(0.0, overlap_max - overlap_min)

    left_width = max(0.0, left_max - left_min)
    right_width = max(0.0, right_max - right_min)
    union_width = max(left_max, right_max) - min(left_min, right_min)

    return {
        "interval_overlap_min": overlap_min if overlap_width > 0.0 else float("nan"),
        "interval_overlap_max": overlap_max if overlap_width > 0.0 else float("nan"),
        "overlap_width": overlap_width,
        "left_range_covered_percent": (
            float("nan") if left_width == 0.0 else 100.0 * overlap_width / left_width
        ),
        "right_range_covered_percent": (
            float("nan") if right_width == 0.0 else 100.0 * overlap_width / right_width
        ),
        "union_range_overlap_percent": (
            float("nan") if union_width == 0.0 else 100.0 * overlap_width / union_width
        ),
    }


def nearest_train_target_coverage(
    query_values: np.ndarray,
    train_values: np.ndarray,
    query_ids: List[str],
    tolerance: float,
) -> Dict[str, Any]:
    distances = np.abs(query_values[:, None] - train_values[None, :])
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(query_values)), nearest_indices]

    poorly_covered_mask = nearest_distances > tolerance
    poorly_covered = []

    for index in np.where(poorly_covered_mask)[0]:
        poorly_covered.append(
            {
                "universe_id": query_ids[index],
                "omega_m": float(query_values[index]),
                "nearest_train_omega_m": float(train_values[nearest_indices[index]]),
                "nearest_train_abs_diff": float(nearest_distances[index]),
            }
        )

    poorly_covered.sort(
        key=lambda item: item["nearest_train_abs_diff"],
        reverse=True,
    )

    return {
        "tolerance": float(tolerance),
        "nearest_abs_diff_mean": float(np.mean(nearest_distances)),
        "nearest_abs_diff_std": float(np.std(nearest_distances)),
        "nearest_abs_diff_min": float(np.min(nearest_distances)),
        "nearest_abs_diff_max": float(np.max(nearest_distances)),
        "nearest_abs_diff_quantiles": {
            f"{quantile:.2f}": float(np.quantile(nearest_distances, quantile))
            for quantile in DEFAULT_QUANTILES
        },
        "percent_within_tolerance": float(100.0 * np.mean(nearest_distances <= tolerance)),
        "num_poorly_covered": int(np.sum(poorly_covered_mask)),
        "poorly_covered_regions": poorly_covered,
    }


def print_summary(output: Dict[str, Any]) -> None:
    print("=" * 112)
    print("SPLIT TARGET DISTRIBUTION DIAGNOSTIC")
    print("=" * 112)
    print(f"Config:  {output['config_path']}")
    print(f"Dataset: {output['dataset_path']}")

    print()
    print("Omega_m distributions")
    print("-" * 112)
    print(
        f"{'Split':<10}"
        f"{'N':>6}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Min':>12}"
        f"{'Q25':>12}"
        f"{'Median':>12}"
        f"{'Q75':>12}"
        f"{'Max':>12}"
    )
    print("-" * 112)

    for split in ["train", "val", "test"]:
        stats = output["splits"][split]
        quantiles = stats["quantiles"]
        print(
            f"{split:<10}"
            f"{stats['num_samples']:>6}"
            f"{stats['mean']:>12.6f}"
            f"{stats['std']:>12.6f}"
            f"{stats['min']:>12.6f}"
            f"{quantiles['0.25']:>12.6f}"
            f"{quantiles['0.50']:>12.6f}"
            f"{quantiles['0.75']:>12.6f}"
            f"{stats['max']:>12.6f}"
        )

    print()
    print("KS tests and histogram overlap")
    print("-" * 112)
    print(
        f"{'Pair':<18}"
        f"{'KS stat':>12}"
        f"{'KS p approx':>14}"
        f"{'Hist overlap %':>18}"
        f"{'Union overlap %':>18}"
    )
    print("-" * 112)

    for pair, stats in output["pairwise_comparisons"].items():
        print(
            f"{pair:<18}"
            f"{stats['ks_test']['statistic']:>12.6f}"
            f"{stats['ks_test']['p_value_asymptotic']:>14.6f}"
            f"{stats['histogram_overlap']['histogram_overlap_percent']:>18.2f}"
            f"{stats['interval_overlap']['union_range_overlap_percent']:>18.2f}"
        )

    print()
    print("Nearest train target coverage")
    print("-" * 112)
    print(
        f"{'Split':<10}"
        f"{'Tolerance':>12}"
        f"{'Mean NN Diff':>16}"
        f"{'Max NN Diff':>16}"
        f"{'Within Tol %':>16}"
        f"{'Poorly Covered':>18}"
    )
    print("-" * 112)

    for split in ["val", "test"]:
        coverage = output["nearest_train_target_coverage"][split]
        print(
            f"{split:<10}"
            f"{coverage['tolerance']:>12.6f}"
            f"{coverage['nearest_abs_diff_mean']:>16.6f}"
            f"{coverage['nearest_abs_diff_max']:>16.6f}"
            f"{coverage['percent_within_tolerance']:>16.2f}"
            f"{coverage['num_poorly_covered']:>18}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Omega_m target distributions across saved splits."
    )
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--coverage_tolerance", type=float, default=0.02)

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_path = (
        Path(args.config_path)
        if args.config_path is not None
        else experiment_dir / "config.json"
    )
    config = load_json(config_path)
    dataset_path = Path(
        args.dataset_path if args.dataset_path is not None else config["dataset_path"]
    )

    data = load_temporal_dataset(dataset_path)
    dataset_ids = list(data.keys())

    split_ids = {}
    for split in ["train", "val", "test"]:
        key = f"{split}_ids"
        if key not in config:
            raise KeyError(f"Config does not contain split key: {key}")
        split_ids[split] = list(config[key])
        validate_split_ids(
            split_ids=split_ids[split],
            dataset_ids=dataset_ids,
            split_name=split,
        )

    targets_by_split = {
        split: extract_targets(data, split_ids[split])
        for split in ["train", "val", "test"]
    }

    bin_edges, histogram_by_split = histogram_counts(
        values_by_split=targets_by_split,
        num_bins=int(args.num_bins),
    )
    bin_edges_array = np.asarray(bin_edges)

    pair_specs = [
        ("train_vs_val", "train", "val"),
        ("train_vs_test", "train", "test"),
        ("val_vs_test", "val", "test"),
    ]
    pairwise_comparisons = {}

    for pair_name, left, right in pair_specs:
        left_values = targets_by_split[left]
        right_values = targets_by_split[right]
        pairwise_comparisons[pair_name] = {
            "left_split": left,
            "right_split": right,
            "ks_test": ks_2samp(left_values, right_values),
            "histogram_overlap": histogram_overlap(
                left_values,
                right_values,
                bin_edges_array,
            ),
            "interval_overlap": interval_overlap(left_values, right_values),
        }

    nearest_train_coverage = {
        split: nearest_train_target_coverage(
            query_values=targets_by_split[split],
            train_values=targets_by_split["train"],
            query_ids=split_ids[split],
            tolerance=float(args.coverage_tolerance),
        )
        for split in ["val", "test"]
    }

    output = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "num_bins": int(args.num_bins),
        "splits": {
            split: {
                **distribution_summary(targets_by_split[split]),
                "universe_ids": split_ids[split],
            }
            for split in ["train", "val", "test"]
        },
        "histogram": {
            "bin_edges": bin_edges,
            "counts": histogram_by_split,
        },
        "pairwise_comparisons": pairwise_comparisons,
        "nearest_train_target_coverage": nearest_train_coverage,
        "poorly_represented_test_regions": nearest_train_coverage["test"][
            "poorly_covered_regions"
        ],
    }

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else experiment_dir / "diagnostics" / "split_target_distribution.json"
    )
    save_json(output, output_path)

    print_summary(output)
    print()
    print(f"Saved diagnostic JSON: {output_path}")


if __name__ == "__main__":
    main()
