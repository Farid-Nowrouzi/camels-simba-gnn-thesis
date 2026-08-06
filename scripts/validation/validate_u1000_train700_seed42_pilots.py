#!/usr/bin/env python3
"""Validate the two authorized U1000 Train700 seed42 pilots and write reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_SHA = "6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a"
MANIFEST_SHA = "b56f35e4cbf1307344beaaf5b26cf181004d23d04fd719b678f1edb9e9924571"
MANIFEST = ROOT / "configs/splits/u1000_top1000_none_k8_sparse/seed42_train700.json"
TARGETS = ROOT / "outputs/target_inspection_1000u.csv"
REPORT_DIR = ROOT / "reports/experiment_registry"
SPECS = {
    "evolve": ROOT / "configs/pilots/evolvegcn_h_u1000_top1000_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear.json",
    "static": ROOT / "configs/pilots/static_gcn_u1000_top1000_sparse_train700_seed42_none_h32_l3_mean_mlp_final.json",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_specs() -> None:
    expected_common = {
        "dataset_sha256": DATASET_SHA, "split_manifest_sha256": MANIFEST_SHA,
        "seed": 42, "counts": {"train": 700, "val": 99, "test": 201, "unused": 0},
        "node_features": 7, "node_normalization": "none", "target_normalization": "none",
        "learning_rate": 0.001, "weight_decay": 0.00001, "loss": "MSE",
        "epochs": 300, "patience": 40, "grad_clip_norm": 1.0,
        "optimizer": "AdamW", "deterministic_seed_handling": True,
        "ordered_predictions": True, "device": "cuda",
    }
    for name, path in SPECS.items():
        require(path.is_file(), f"missing pilot spec: {path}")
        spec = read_json(path)
        for key, expected in expected_common.items():
            require(spec.get(key) == expected, f"{name} spec mismatch: {key}")
        scheduler = spec.get("scheduler", {})
        require(scheduler == {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                              "patience": 10, "min_lr": 0.000001}, f"{name} scheduler mismatch")
    evolve, static = read_json(SPECS["evolve"]), read_json(SPECS["static"])
    require((evolve["model"], evolve["hidden_dim"], evolve["num_layers"], evolve["batch_size"],
             evolve["dropout"], evolve["activation"], evolve["temporal_pooling"], evolve["graph_pooling"],
             evolve["head_type"], evolve["use_summary_features"]) ==
            ("EvolveGCNHRegressor", 32, 2, 4, 0.2, "relu", "mean", "mean", "linear", False),
            "Evolve spec does not match the verified linear-head anchor")
    require((static["model"], static["dataset_format"], static["hidden_dim"], static["num_layers"],
             static["batch_size"], static["dropout"], static["graph_pooling"], static["conv_type"],
             static["use_layer_norm"], static["residual"]) ==
            ("StaticGCNRegressor", "temporal_final_snapshot", 32, 3, 8, 0.2, "mean", "gcn", True, True),
            "Static spec does not match the verified final-snapshot anchor")
    require(sha256(MANIFEST) == MANIFEST_SHA, "split manifest byte hash mismatch")
    manifest = read_json(MANIFEST)
    require(manifest["dataset_identity"] == DATASET_SHA and manifest["seed"] == 42,
            "manifest identity/seed mismatch")
    require(manifest["counts"] == {"population": 1000, "test": 201, "train": 700, "unused": 0, "val": 99},
            "manifest counts mismatch")


def load_targets() -> dict[str, float]:
    with TARGETS.open(newline="", encoding="utf-8") as handle:
        return {row["universe_id"]: float(row["omega_m"]) for row in csv.DictReader(handle)}


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def experiment_result(name: str) -> dict:
    spec = read_json(SPECS[name])
    exp = ROOT / "experiments" / spec["experiment_name"]
    config_path, metrics_path = exp / "config.json", exp / "metrics.json"
    pred_path, checkpoint = exp / "predictions/test_predictions.csv", exp / "checkpoints/best_model.pt"
    for path in (config_path, metrics_path, pred_path, checkpoint, exp / "train_log.csv"):
        require(path.is_file(), f"{name}: missing artifact {path}")
    config, reported, manifest = read_json(config_path), read_json(metrics_path), read_json(MANIFEST)
    require(config["dataset_provenance"]["dataset_sha256"] == DATASET_SHA, f"{name}: dataset hash mismatch")
    require(config["split_manifest_sha256"] == MANIFEST_SHA, f"{name}: manifest hash mismatch")
    require(config["seed"] == config["trainer_invocation_seed"] == config["split_manifest_seed"] == 42,
            f"{name}: seed mismatch")
    require((config["num_train_universes"], config["num_val_universes"], config["num_test_universes"]) ==
            (700, 99, 201), f"{name}: split counts mismatch")
    require(config["num_nodes"] == 1000 and config["node_features"] == 7,
            f"{name}: saved Top1000/feature dimensions mismatch")
    require(config["train_ids"] == manifest["train_ids"] and config["val_ids"] == manifest["val_ids"] and
            config["test_ids"] == manifest["test_ids"], f"{name}: config split ordering mismatch")
    expected_saved = {
        "evolve": {"model": "EvolveGCNHRegressor", "batch_size": 4, "epochs": 300, "patience": 40,
                   "learning_rate": 0.001, "weight_decay": 0.00001, "hidden_dim": 32, "num_layers": 2,
                   "dropout": 0.2, "activation": "relu", "temporal_pooling": "mean", "graph_pooling": "mean",
                   "head_type": "linear", "add_self_loops": True, "grad_clip_norm": 1.0,
                   "use_summary_features": False, "normalize_target": False},
        "static": {"model": "StaticGCNRegressor", "dataset_format": "temporal_final_snapshot", "batch_size": 8,
                   "epochs": 300, "patience": 40, "learning_rate": 0.001, "weight_decay": 0.00001,
                   "hidden_dim": 32, "num_layers": 3, "dropout": 0.2, "graph_pooling": "mean",
                   "conv_type": "gcn", "grad_clip_norm": 1.0},
    }[name]
    for key, expected in expected_saved.items():
        require(config.get(key) == expected, f"{name}: saved config mismatch for {key}")
    require(config.get("optimizer") == "AdamW" and config.get("loss") == "MSELoss" and
            config.get("checkpoint_criterion") == "minimum_validation_mse" and
            config.get("deterministic_seed_handling") is True, f"{name}: saved training behavior mismatch")
    require(config.get("scheduler") == {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                                         "patience": 10, "min_lr": 1e-6}, f"{name}: saved scheduler mismatch")
    with pred_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["universe_id"] for row in rows]
    require(len(rows) == 201 and ids == manifest["test_ids"], f"{name}: test prediction ordering mismatch")
    require(len(set(ids)) == 201, f"{name}: duplicate prediction IDs")
    targets_table = load_targets()
    true = [float(row["true_omega_m"]) for row in rows]
    pred = [float(row["pred_omega_m"]) for row in rows]
    require(all(math.isfinite(value) for value in true + pred), f"{name}: nonfinite predictions/targets")
    for universe_id, value in zip(ids, true):
        expected = float(__import__("struct").unpack("f", __import__("struct").pack("f", targets_table[universe_id]))[0])
        require(value == expected, f"{name}: target table mismatch for {universe_id}")
    require(len(set(pred)) > 1, f"{name}: exactly constant prediction vector")
    with (exp / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        log_rows = list(csv.DictReader(handle))
    require(0 < len(log_rows) <= 300, f"{name}: invalid epoch count")
    require(all(math.isfinite(float(row[key])) for row in log_rows for key in ("train_mse", "val_mse", "best_val_mse")),
            f"{name}: nonfinite training/validation loss")
    best_epoch = int(reported["best_epoch"])
    require(best_epoch == int(log_rows[-1]["best_epoch"]) and 1 <= best_epoch <= len(log_rows),
            f"{name}: best-epoch accounting mismatch")
    if len(log_rows) < 300:
        require(len(log_rows) - best_epoch >= 40, f"{name}: early stopping occurred before patience 40")
    errors = [p - y for p, y in zip(pred, true)]
    mse = sum(e * e for e in errors) / len(errors)
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(mse)
    target_mean = sum(true) / len(true)
    pred_mean = sum(pred) / len(pred)
    ss_tot = sum((y - target_mean) ** 2 for y in true)
    r2 = 1.0 - sum(e * e for e in errors) / ss_tot
    test = reported["test"]
    for key, got in (("mse", mse), ("rmse", rmse), ("mae", mae), ("r2", r2)):
        require(math.isclose(float(test[key]), got, rel_tol=1e-9, abs_tol=1e-10), f"{name}: {key} mismatch")
    repeated = len(pred) - len(set(pred))
    pred_sd, target_sd = sample_sd(pred), sample_sd(true)
    residual_mean = sum(errors) / len(errors)
    residual_sd = sample_sd(errors)
    sorted_targets = sorted(true)
    low_cut, high_cut = sorted_targets[len(true) // 4], sorted_targets[(3 * len(true)) // 4]
    low_errors = [e for e, y in zip(errors, true) if y <= low_cut]
    high_errors = [e for e, y in zip(errors, true) if y >= high_cut]
    runtime_path = ROOT / "logs/u1000_train700_seed42_pilots" / f"{name}_runtime.json"
    telemetry_path = ROOT / "logs/u1000_train700_seed42_pilots" / f"{name}_gpu_telemetry.csv"
    runtime = read_json(runtime_path) if runtime_path.is_file() else {}
    peak_gpu = None
    if telemetry_path.is_file():
        with telemetry_path.open(newline="", encoding="utf-8") as handle:
            vals = [float(row["used_gpu_memory_mib"]) for row in csv.DictReader(handle)
                    if row.get("used_gpu_memory_mib") not in (None, "", "N/A")]
        peak_gpu = max(vals) if vals else None
    return {
        "model": name, "experiment_name": spec["experiment_name"], "experiment_dir": str(exp.relative_to(ROOT)),
        "status": "PASS", "best_epoch": int(reported["best_epoch"]), "epochs_run": len(log_rows),
        "mae": mae, "rmse": rmse, "r2": r2, "mse": mse,
        "prediction_sd": pred_sd, "target_sd": target_sd, "prediction_sd_ratio": pred_sd / target_sd,
        "repeated_prediction_count": repeated, "repeated_prediction_fraction": repeated / len(pred),
        "mean_prediction": pred_mean, "mean_target": target_mean, "residual_mean": residual_mean,
        "residual_sd": residual_sd, "low_target_bias": sum(low_errors) / len(low_errors),
        "high_target_bias": sum(high_errors) / len(high_errors), "runtime_seconds": runtime.get("runtime_seconds"),
        "max_host_rss_kib": runtime.get("max_host_rss_kib"), "peak_gpu_memory_mib": peak_gpu,
        "ordered_test_ids": "PASS", "metric_recomputation": "PASS", "prediction_rows": len(rows),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def finalize(results: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    severe = any(r["prediction_sd_ratio"] < 0.1 or r["prediction_sd_ratio"] > 2.0 or
                 r["repeated_prediction_fraction"] > 0.5 or r["r2"] < 0.0 for r in results)
    decision = "CONDITIONAL GO" if severe else "GO FOR REMAINING MATRIX"
    metrics_fields = ["model", "experiment_name", "status", "best_epoch", "epochs_run", "mae", "rmse", "r2", "mse"]
    diag_fields = ["model", "prediction_sd", "target_sd", "prediction_sd_ratio", "repeated_prediction_count",
                   "repeated_prediction_fraction", "mean_prediction", "mean_target", "residual_mean", "residual_sd",
                   "low_target_bias", "high_target_bias", "ordered_test_ids", "metric_recomputation"]
    runtime_fields = ["model", "runtime_seconds", "max_host_rss_kib", "peak_gpu_memory_mib"]
    write_csv(REPORT_DIR / "u1000_train700_seed42_full_pilot_metrics.csv", results, metrics_fields)
    write_csv(REPORT_DIR / "u1000_train700_seed42_prediction_diagnostics.csv", results, diag_fields)
    write_csv(REPORT_DIR / "u1000_train700_seed42_runtime_memory.csv", results, runtime_fields)
    lines = ["# U1000 Train700 Seed42 Full Pilot Audit", "", f"Run finalized UTC: {datetime.now(timezone.utc).isoformat()}",
             "", f"## Final decision", "", f"**{decision}**", "",
             "Both authorized pilots passed infrastructure and artifact validation. Weak scientific performance does not alter infrastructure PASS.", ""]
    for r in results:
        lines += [f"## {r['model'].title()}", "", f"- Experiment: `{r['experiment_dir']}`",
                  f"- Best epoch / epochs run: {r['best_epoch']} / {r['epochs_run']}",
                  f"- Test MAE / RMSE / R2: {r['mae']:.10g} / {r['rmse']:.10g} / {r['r2']:.10g}",
                  f"- Prediction SD ratio: {r['prediction_sd_ratio']:.10g}",
                  f"- Repeated-prediction fraction: {r['repeated_prediction_fraction']:.10g}",
                  f"- Runtime seconds: {r['runtime_seconds']}", f"- Peak GPU memory MiB: {r['peak_gpu_memory_mib']}",
                  "- Exact ordered test IDs: PASS", "- Metric recomputation: PASS", ""]
    (REPORT_DIR / "u1000_train700_seed42_full_pilot_audit.md").write_text("\n".join(lines), encoding="utf-8")
    next_text = f"""# U1000 Train700 Seed42 Next-Matrix Decision

## Decision

**{decision}**

Both authorized full pilots completed and passed identity, ordering, artifact, finiteness, and metric-recomputation gates. Prediction collapse diagnostics are preserved in `u1000_train700_seed42_prediction_diagnostics.csv`.

No additional training-size or seed run was launched by this workflow. Any remaining matrix requires a separate explicit authorization.
"""
    (REPORT_DIR / "u1000_train700_seed42_next_matrix_decision.md").write_text(next_text, encoding="utf-8")
    print(f"PASS: finalized reports; decision={decision}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model", choices=("evolve", "static"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    validate_specs()
    if args.preflight:
        print("PASS: exact pilot specifications and Train700 seed42 manifest")
        return 0
    if args.model:
        result = experiment_result(args.model)
        print(json.dumps(result, indent=2))
        return 0
    if args.finalize:
        finalize([experiment_result("evolve"), experiment_result("static")])
        return 0
    parser.error("choose --preflight, --model, or --finalize")


if __name__ == "__main__":
    raise SystemExit(main())
