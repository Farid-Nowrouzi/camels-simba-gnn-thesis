#!/usr/bin/env python3
"""Read-only verification of the canonical EvolveGCN-H scaling family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


UNIVERSES = (20, 50, 100, 200, 500)
SEEDS = (42, 123, 2025)
PLANNED = {
    (20, 42): "evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42",
    (20, 2025): "evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025",
    (50, 42): "evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42",
    (50, 2025): "evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025",
    (100, 42): "evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42",
    (100, 2025): "evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025",
}
EXISTING = {
    (20, 123): "evolvegcn_h_20u_seed123_final32",
    (50, 123): "evolvegcn_h_50u_seed123_final32",
    (100, 123): "evolvegcn_h_100u_seed123_final32",
    **{
        (u, s): f"evolvegcn_h_{u}u_seed{s}_final32"
        for u in (200, 500)
        for s in SEEDS
    },
}
DATASETS = {
    u: f"data/processed/temporal_{u}u_minmax/"
    f"camels_{u}u_temporal_logmass_minmax_top100_periodic_knn.pt"
    for u in UNIVERSES
}
REQUIRED_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "train_log.csv",
    "predictions/test_predictions.csv",
    "checkpoints/best_model.pt",
)
EXPECTED_CONFIG = {
    "model": "EvolveGCNHRegressor",
    "batch_size": 4,
    "epochs": 300,
    "patience": 40,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.2,
    "activation": "relu",
    "temporal_pooling": "mean",
    "graph_pooling": "mean",
    "head_type": "mlp",
    "add_self_loops": True,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "grad_clip_norm": 1.0,
    "use_summary_features": False,
    "normalize_target": False,
}
LEGACY_DEFAULTS = {
    "activation": "relu",
    "head_type": "mlp",
    "use_summary_features": False,
    "normalize_target": False,
}
METADATA_EXPECTED = {
    "num_snapshots": 5,
    "num_nodes": 100,
    "normalization": "minmax",
    "graph_mode": "knn",
    "k": 8,
    "periodic_boundary": True,
    "periodic_boundary_knn": True,
    "box_size": 25.0,
    "feature_names": ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 15 canonical EvolveGCN-H Top100/minmax/k8/h32 "
            "universe-scaling experiments without loading datasets/checkpoints."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Nothing is written when omitted.",
    )
    return parser.parse_args()


def equal_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".metadata.json")


def expected_counts(universes: int) -> tuple[int, int, int]:
    train = int(universes * 0.70)
    val = int(universes * 0.15)
    return train, val, universes - train - val


def prediction_metrics(path: Path) -> dict[str, float]:
    targets: list[float] = []
    predictions: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"prediction CSV has no header: {path}")
        if "true_omega_m" not in reader.fieldnames or "pred_omega_m" not in reader.fieldnames:
            raise ValueError(f"prediction CSV lacks canonical columns: {path}")
        for row in reader:
            targets.append(float(row["true_omega_m"]))
            predictions.append(float(row["pred_omega_m"]))
    if not targets:
        raise ValueError(f"prediction CSV has no rows: {path}")
    errors = [p - y for p, y in zip(predictions, targets)]
    mae = sum(abs(e) for e in errors) / len(errors)
    mse = sum(e * e for e in errors) / len(errors)
    target_mean = statistics.mean(targets)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((y - target_mean) ** 2 for y in targets)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    if len(targets) > 1:
        pred_mean = statistics.mean(predictions)
        covariance = sum(
            (p - pred_mean) * (y - target_mean)
            for p, y in zip(predictions, targets)
        )
        pred_ss = sum((p - pred_mean) ** 2 for p in predictions)
        target_ss = sum((y - target_mean) ** 2 for y in targets)
        pearson = covariance / math.sqrt(pred_ss * target_ss) if pred_ss and target_ss else float("nan")
        target_std = statistics.stdev(targets)
        prediction_std = statistics.stdev(predictions)
    else:
        pearson = target_std = prediction_std = float("nan")
    return {
        "mae": mae,
        "rmse": math.sqrt(mse),
        "r2": r2,
        "pearson": pearson,
        "target_std": target_std,
        "prediction_std_ratio": prediction_std / target_std if target_std else float("nan"),
        "count": float(len(targets)),
    }


def split_signature(config: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "train_ids": config.get("train_ids"),
            "val_ids": config.get("val_ids"),
            "test_ids": config.get("test_ids"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    errors: list[str] = []
    verified: list[dict[str, Any]] = []

    print("Canonical EvolveGCN-H scaling verification")
    print(f"Repository: {root}")
    print("Protocol: Top100, minmax, periodic kNN k=8, h32/L2, mean/mean, MLP")

    for universes in UNIVERSES:
        dataset = root / DATASETS[universes]
        meta_path = metadata_path(dataset)
        if not dataset.is_file():
            errors.append(f"{universes}U: missing dataset {dataset}")
        if not meta_path.is_file():
            errors.append(f"{universes}U: missing metadata {meta_path}")
            metadata = {}
        else:
            metadata = load_json(meta_path)
            for field, expected in METADATA_EXPECTED.items():
                if not equal_value(metadata.get(field), expected):
                    errors.append(
                        f"{universes}U metadata: {field}={metadata.get(field)!r}, "
                        f"expected {expected!r}"
                    )
            if metadata.get("num_universes_successful") != universes:
                errors.append(f"{universes}U metadata: universe count mismatch")

        for seed in SEEDS:
            name = PLANNED.get((universes, seed), EXISTING.get((universes, seed)))
            if name is None:
                errors.append(f"{universes}U seed {seed}: no canonical manifest entry")
                continue
            directory = root / "experiments" / name
            missing = [
                artifact for artifact in REQUIRED_ARTIFACTS
                if not (directory / artifact).is_file()
            ]
            if missing:
                errors.append(
                    f"{universes}U seed {seed}: missing {', '.join(missing)} in {name}"
                )
                continue
            config = load_json(directory / "config.json")
            for field, expected in EXPECTED_CONFIG.items():
                actual = config.get(field, LEGACY_DEFAULTS.get(field))
                if not equal_value(actual, expected):
                    errors.append(
                        f"{universes}U seed {seed}: {field}={actual!r}, expected {expected!r}"
                    )
            if config.get("seed") != seed:
                errors.append(f"{universes}U seed {seed}: config seed mismatch")
            if config.get("dataset_path") != DATASETS[universes]:
                errors.append(f"{universes}U seed {seed}: dataset path mismatch")
            train_ids = config.get("train_ids", [])
            val_ids = config.get("val_ids", [])
            test_ids = config.get("test_ids", [])
            counts = (len(train_ids), len(val_ids), len(test_ids))
            if counts != expected_counts(universes):
                errors.append(
                    f"{universes}U seed {seed}: split counts {counts}, "
                    f"expected {expected_counts(universes)}"
                )
            split_sets = [set(train_ids), set(val_ids), set(test_ids)]
            if (
                split_sets[0] & split_sets[1]
                or split_sets[0] & split_sets[2]
                or split_sets[1] & split_sets[2]
            ):
                errors.append(f"{universes}U seed {seed}: split overlap detected")
            try:
                metrics = prediction_metrics(directory / "predictions/test_predictions.csv")
            except (OSError, ValueError) as exc:
                errors.append(f"{universes}U seed {seed}: {exc}")
                continue
            verified.append(
                {
                    "universes": universes,
                    "seed": seed,
                    "experiment_name": name,
                    "split_signature": split_signature(config),
                    **metrics,
                }
            )

    observed_pairs = {(row["universes"], row["seed"]) for row in verified}
    expected_pairs = {(u, s) for u in UNIVERSES for s in SEEDS}
    if observed_pairs != expected_pairs:
        errors.append(
            f"verified universe/seed set is incomplete: "
            f"{sorted(expected_pairs - observed_pairs)}"
        )

    summaries: list[dict[str, Any]] = []
    print()
    print("U    seeds        MAE mean±SD       RMSE mean±SD      R² mean±SD       best")
    for universes in UNIVERSES:
        rows = [row for row in verified if row["universes"] == universes]
        if len(rows) != 3:
            print(f"{universes:<4} INCOMPLETE ({len(rows)}/3)")
            continue
        summary = {"universes": universes, "seeds": list(SEEDS)}
        for metric in ("mae", "rmse", "r2", "pearson"):
            values = [float(row[metric]) for row in rows]
            summary[f"mean_{metric}"] = statistics.mean(values)
            summary[f"std_{metric}"] = statistics.stdev(values)
        summary["target_std"] = statistics.mean(
            float(row["target_std"]) for row in rows
        )
        summary["prediction_std_ratio"] = statistics.mean(
            float(row["prediction_std_ratio"]) for row in rows
        )
        best = min(rows, key=lambda row: float(row["mae"]))
        summary["best_seed"] = best["seed"]
        summaries.append(summary)
        print(
            f"{universes:<4} 42,123,2025  "
            f"{summary['mean_mae']:.6f}±{summary['std_mae']:.6f}  "
            f"{summary['mean_rmse']:.6f}±{summary['std_rmse']:.6f}  "
            f"{summary['mean_r2']:.6f}±{summary['std_r2']:.6f}  "
            f"{summary['best_seed']}"
        )

    result = {
        "valid": not errors,
        "protocol": {**EXPECTED_CONFIG, "top_n": 100, "normalization": "minmax", "k": 8},
        "experiments": verified,
        "summary": summaries,
        "errors": errors,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote: {output}")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nPASS: all 15 canonical experiments are complete and compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
