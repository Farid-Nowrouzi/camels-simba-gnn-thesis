#!/usr/bin/env python3
"""Validate and optionally write the frozen final-thesis artifact index.

This utility is read-only with respect to experiments.  It recomputes test
metrics from saved predictions, verifies split/config/model artifacts, and
writes only the two small CSV reports named below when ``--write`` is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 2025)
MANIFEST = ROOT / "reports/reproducibility/final_thesis_artifact_manifest.csv"
SELECTION = ROOT / "reports/experiment_registry/thesis_final_selection.csv"
LEGACY_REPLACEMENT = ROOT / "reports/reproducibility/evolvegcn_500u_top500_vs_750u_top1000_authoritative.csv"

FAMILIES = (
    ("Gradient Boosting", "gradient_boosting_u1000_top1500_summary20_final_train700_seed{seed}_none_t300_lr0p03_d3", "frozen configuration", "BEST SUMMARY", "engineered summary20", "static", 1500, "N/A"),
    ("Static GCN", "static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final", "validation-ranked", "BEST STATIC GRAPH", "raw7 graph", "static", 1500, "kNN k=8"),
    ("Static PNA", "static_pna_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final", "post-freeze evaluation", "SPATIAL COMPLEXITY ABLATION", "raw7 graph", "static", 1500, "kNN k=8"),
    ("EvolveGCN-H", "evolvegcn_h_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_mean_temporal_mean_linear", "frozen configuration", "EVOLVING-WEIGHT TEMPORAL", "raw7 graph sequence", "temporal", 1500, "kNN k=8"),
    ("EvolveGCN-O", "evolvegcn_o_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_mean_temporal_mean_linear", "validation-ranked; recovered seed 123", "EVOLVING-WEIGHT TEMPORAL", "raw7 graph sequence", "temporal", 1500, "kNN k=8"),
    ("GCN-GRU", "gcn_gru_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_final", "validation-ranked", "BEST TEMPORAL GRAPH", "raw7 graph sequence", "temporal", 1500, "kNN k=8"),
    ("GCN Temporal Transformer", "gcn_temporal_transformer_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_final", "validation-ranked", "TEMPORAL ARCHITECTURE COMPARISON", "raw7 graph sequence", "temporal", 1500, "kNN k=8"),
    ("DeepSets", "deepsets_u1000_top1000_raw7_final_train700_seed{seed}_none_h32_phi3_mean_statichead", "frozen configuration; Top1000 headline documented in Notebook 13", "RAW SET BASELINE", "raw7 halo set", "static", 1000, "no graph"),
)

MANIFEST_COLUMNS = (
    "model_family", "configuration", "seed", "result_directory", "config_path",
    "metrics_path", "test_prediction_path", "checkpoint_or_model_path", "split_manifest",
    "sha256_config", "sha256_metrics", "sha256_test_predictions", "sha256_model_or_checkpoint",
    "source_commit_if_known", "status", "selection_basis", "notes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def metric_block(metrics: dict, split: str) -> dict:
    if split == "val":
        return metrics.get("val") or metrics.get("validation") or {}
    return metrics.get(split) or {}


def prediction_metrics(path: Path, expected_count: int | None = 201) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(f"{path}: expected {expected_count} test predictions, found {len(rows)}")
    fields = set(rows[0])
    true_key = "true_omega_m" if "true_omega_m" in fields else "target"
    pred_key = "pred_omega_m" if "pred_omega_m" in fields else "prediction"
    truth = [float(row[true_key]) for row in rows]
    pred = [float(row[pred_key]) for row in rows]
    if not all(math.isfinite(value) for value in truth + pred):
        raise RuntimeError(f"{path}: non-finite prediction data")
    residual = [estimate - target for target, estimate in zip(truth, pred)]
    mse = statistics.fmean(value * value for value in residual)
    denom = sum((value - statistics.fmean(truth)) ** 2 for value in truth)
    return {
        "mae": statistics.fmean(abs(value) for value in residual),
        "rmse": math.sqrt(mse),
        "r2": 1.0 - sum(value * value for value in residual) / denom,
    }


def resolve_split(config: dict) -> Path:
    value = config.get("split_manifest_path") or config.get("split_source")
    if not value:
        raise RuntimeError("config does not identify a split manifest")
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError as exc:
            raise RuntimeError(f"non-canonical split path: {path}") from exc
    return ROOT / path


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, str]] = []
    measurements: list[dict[str, object]] = []
    for family, pattern, basis, role, representation, temporal, top_n, graph in FAMILIES:
        for seed in SEEDS:
            run = ROOT / "experiments" / pattern.format(seed=seed)
            config_path = run / "config.json"
            metrics_path = run / "metrics.json"
            prediction_path = run / "predictions/test_predictions.csv"
            model_path = run / "checkpoints/best_model.pt"
            if not model_path.is_file():
                model_path = run / "model.joblib"
            required = (config_path, metrics_path, prediction_path, model_path)
            if not all(path.is_file() and path.stat().st_size > 0 for path in required):
                missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
                raise RuntimeError(f"{family}/{seed}: missing artifacts: {missing}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            split_path = resolve_split(config)
            if not split_path.is_file():
                raise RuntimeError(f"{family}/{seed}: missing split manifest {split_path}")
            recomputed = prediction_metrics(prediction_path)
            saved = metric_block(metrics, "test")
            for key in ("mae", "rmse", "r2"):
                if not math.isclose(recomputed[key], float(saved[key]), rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError(f"{family}/{seed}: saved/recomputed {key} mismatch")
            val = metric_block(metrics, "val")
            source_commit = next((str(config[key]) for key in ("git_head", "git_commit", "repository_commit", "commit_hash") if config.get(key)), "UNKNOWN")
            status = "RECOVERED" if family == "EvolveGCN-O" and seed == 123 else "COMPLETED"
            notes = "inference-only recovery; checkpoint unchanged" if status == "RECOVERED" else "frozen artifact; raw test metrics independently validated"
            manifest_rows.append({
                "model_family": family, "configuration": run.name, "seed": str(seed),
                "result_directory": relative(run), "config_path": relative(config_path),
                "metrics_path": relative(metrics_path), "test_prediction_path": relative(prediction_path),
                "checkpoint_or_model_path": relative(model_path), "split_manifest": relative(split_path),
                "sha256_config": sha256(config_path), "sha256_metrics": sha256(metrics_path),
                "sha256_test_predictions": sha256(prediction_path),
                "sha256_model_or_checkpoint": sha256(model_path), "source_commit_if_known": source_commit,
                "status": status, "selection_basis": basis, "notes": notes,
            })
            measurements.append({
                "family": family, "seed": seed, "val_mae": float(val["mae"]),
                "test_mae": recomputed["mae"], "test_rmse": recomputed["rmse"],
                "test_r2": recomputed["r2"], "path": relative(run), "basis": basis,
                "role": role, "representation": representation, "temporal": temporal,
                "top_n": top_n, "graph": graph,
            })
    return manifest_rows, measurements


def selection_rows(measurements: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for family, *_ in FAMILIES:
        rows = [row for row in measurements if row["family"] == family]
        first = rows[0]
        result.append({
            "category": first["role"], "model": family,
            "input_representation": first["representation"], "temporal_or_static": first["temporal"],
            "top_n": first["top_n"], "k_or_graph_type": first["graph"],
            "snapshots": "5" if first["temporal"] == "temporal" else "final",
            "seeds": ";".join(str(seed) for seed in SEEDS), "selection_basis": first["basis"],
            "val_mae_mean": statistics.fmean(float(row["val_mae"]) for row in rows),
            "test_mae_mean": statistics.fmean(float(row["test_mae"]) for row in rows),
            "test_rmse_mean": statistics.fmean(float(row["test_rmse"]) for row in rows),
            "test_r2_mean": statistics.fmean(float(row["test_r2"]) for row in rows),
            "result_path": "experiments/" + FAMILIES[[item[0] for item in FAMILIES].index(family)][1].replace("{seed}", "{42,123,2025}"),
            "status": "COMPLETED", "thesis_role": first["role"],
        })
    return result


def legacy_replacement_rows() -> list[dict[str, object]]:
    rows = []
    for group, universes, top_n in (("EvolveGCN_500U_Top500_none", 500, 500),
                                    ("EvolveGCN_750U_Top1000_none", 750, 1000)):
        for seed in (42, 123, 777, 999, 2025):
            suffix = "_norm" if universes == 500 and seed == 42 else ""
            name = f"evolvegcn_h_{universes}u_top{top_n}_h32_seed{seed}_none{suffix}"
            run = ROOT / "experiments" / name
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            recomputed = prediction_metrics(run / "predictions/test_predictions.csv", expected_count=None)
            saved = metric_block(metrics, "test")
            for key in ("mae", "rmse", "r2"):
                if key in saved and not math.isclose(recomputed[key], float(saved[key]), rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError(f"legacy replacement {name}: saved/recomputed {key} mismatch")
            rows.append({
                "group": group, "seed": seed, "experiment": name,
                "best_epoch": metrics.get("best_epoch", "UNKNOWN"),
                "best_val_mse": metrics.get("best_val_loss", metrics.get("best_val_mse", "UNKNOWN")),
                "trainable_parameters": metrics.get("trainable_parameters", "UNKNOWN"),
                "test_mae_from_predictions": recomputed["mae"],
                "test_rmse_from_predictions": recomputed["rmse"],
                "test_r2_from_predictions": recomputed["r2"],
            })
    return rows


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest_rows, measurements = build_rows()
    selections = selection_rows(measurements)
    replacement = legacy_replacement_rows()
    if args.write:
        write_csv(MANIFEST, MANIFEST_COLUMNS, manifest_rows)
        write_csv(SELECTION, tuple(selections[0]), selections)
        write_csv(LEGACY_REPLACEMENT, tuple(replacement[0]), replacement)
    print(f"PASS: {len(manifest_rows)} headline artifacts, {len(selections)} final selections, and {len(replacement)} deprecated-table replacement rows validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
