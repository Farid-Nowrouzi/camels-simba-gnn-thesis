#!/usr/bin/env python3
"""Prepare, audit, validate, and aggregate the controlled U1000 scaling matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "envs/camels-gnn/bin/python"
REGISTRY = ROOT / "configs/experiment_registry/u1000_top1000_training_scaling_matrix.json"
CONFIG_DIR = Path("configs/production/u1000_top1000_training_scaling")
REPORT_DIR = ROOT / "reports/experiment_registry"
DATASET = Path("data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt")
TARGETS = Path("outputs/target_inspection_1000u.csv")
SPLITS = Path("configs/splits/u1000_top1000_none_k8_sparse")
HASHES = {
    str(DATASET): "6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a",
    str(DATASET.with_suffix(".metadata.json")): "d4ea0ba0c3a1abc6f49d6856be86c7fc1226090daac8924eb6b72262d22753b9",
    str(DATASET.with_suffix(".complete")): "4eea1a4bbbfc57d0c3420a115ae436240e0dcb1588cf47588ab2ee5809edd85a",
    str(TARGETS): "9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2",
}
SEEDS = (42, 123, 2025)
LEVELS = (20, 50, 100, 200, 450, 700)
MODELS = ("evolve", "static")
PILOT_NAMES = {
    "evolve": "evolvegcn_h_u1000_top1000_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear",
    "static": "static_gcn_u1000_top1000_sparse_train700_seed42_none_h32_l3_mean_mlp_final",
}
PILOT_CONFIGS = {
    key: Path("configs/pilots") / f"{name}.json" for key, name in PILOT_NAMES.items()
}
ALLOWED = {"planned", "running", "completed", "failed", "invalid"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def name_for(model: str, count: int, seed: int) -> str:
    if model == "evolve":
        return f"evolvegcn_h_u1000_top1000_sparse_train{count}_seed{seed}_none_h32_l2_mean_temporal_mean_linear"
    return f"static_gcn_u1000_top1000_sparse_train{count}_seed{seed}_none_h32_l3_mean_mlp_final"


def identity(model: str, count: int, seed: int) -> str:
    return f"u1000-top1000-sparse-{model}-train{count}-seed{seed}"


def config_path(model: str, count: int, seed: int) -> Path:
    if count == 700 and seed == 42:
        return PILOT_CONFIGS[model]
    return CONFIG_DIR / f"{name_for(model, count, seed)}.json"


def common_entry(model: str, count: int, seed: int, commit: str) -> dict[str, Any]:
    manifest = SPLITS / f"seed{seed}_train{count}.json"
    name = name_for(model, count, seed)
    status = "completed" if (count, seed) == (700, 42) else "planned"
    return {
        "canonical_experiment_id": identity(model, count, seed), "experiment_name": name,
        "model": "EvolveGCNHRegressor" if model == "evolve" else "StaticGCNRegressor",
        "model_key": model, "training_universe_count": count, "validation_count": 99,
        "test_count": 201, "unused_count": 700 - count,
        "master_dataset_path": str(DATASET), "dataset_sha256": HASHES[str(DATASET)],
        "metadata_sha256": HASHES[str(DATASET.with_suffix('.metadata.json'))],
        "completion_marker_sha256": HASHES[str(DATASET.with_suffix('.complete'))],
        "target_table_sha256": HASHES[str(TARGETS)], "split_manifest_path": str(manifest),
        "split_manifest_sha256": sha(ROOT / manifest), "configuration_path": str(config_path(model, count, seed)),
        "experiment_directory": f"experiments/{name}", "repository_commit": commit,
        "graph_storage_mode": "sparse_edge_index", "normalization": "none", "top_n": 1000, "k": 8,
        "snapshot_input_protocol": "all_five_temporal_snapshots" if model == "evolve" else "temporal_final_snapshot_exact_a1.0",
        "seed": seed, "status": status, "planned_timestamp": now(), "launch_timestamp": "",
        "completion_timestamp": "", "validation_timestamp": "", "failure_reason": "", "invalid_reason": "",
        "checkpoint_path": "", "metrics_path": "", "predictions_path": "", "best_epoch": "",
        "epochs_executed": "", "runtime_seconds": "", "peak_gpu_memory_mib": "",
        "validation_result": "", "test_mae": "", "test_rmse": "", "test_mse": "", "test_r2": "",
        "prediction_mean": "", "prediction_standard_deviation": "", "target_standard_deviation": "",
        "prediction_sd_ratio": "", "repeated_prediction_count": "", "repeated_prediction_fraction": "",
        "residual_mean": "", "residual_standard_deviation": "", "low_target_bias": "",
        "high_target_bias": "", "collapse_flag": "", "matrix_included": True,
    }


def validate_manifest(path: Path, seed: int, count: int) -> dict[str, Any]:
    m = read_json(path)
    expected = {"population": 1000, "train": count, "val": 99, "test": 201, "unused": 700-count}
    require(m.get("dataset_identity") == HASHES[str(DATASET)], f"dataset identity: {path}")
    require(m.get("seed") == seed, f"seed: {path}")
    require(m.get("counts") == expected, f"counts: {path}")
    parts = {k: m[f"{k}_ids"] for k in ("train", "val", "test", "unused")}
    require(all(len(v) == expected[k] and len(v) == len(set(v)) for k, v in parts.items()), f"partition sizes: {path}")
    require(len(set().union(*(set(v) for v in parts.values()))) == 1000, f"partition coverage: {path}")
    return m


def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)


def prepare() -> None:
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "thesis-sparse-integrity-hardening", "wrong branch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    templates = {key: read_json(ROOT / path) for key, path in PILOT_CONFIGS.items()}
    (ROOT / CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in SEEDS:
        for count in LEVELS:
            manifest_path = ROOT / SPLITS / f"seed{seed}_train{count}.json"
            validate_manifest(manifest_path, seed, count)
            for model in MODELS:
                entry = common_entry(model, count, seed, commit)
                if not (count == 700 and seed == 42):
                    config = dict(templates[model])
                    config.update({"experiment_name": entry["experiment_name"], "seed": seed,
                                   "counts": {"train": count, "val": 99, "test": 201, "unused": 700-count},
                                   "split_manifest_path": entry["split_manifest_path"],
                                   "split_manifest_sha256": entry["split_manifest_sha256"]})
                    atomic_json(ROOT / entry["configuration_path"], config)
                entries.append(entry)
    for entry in entries:
        if entry["status"] == "completed":
            result = validate_run_entry(entry, update=False)
            entry.update(result); entry["completion_timestamp"] = "2026-08-06T00:00:00+00:00"
    invalid_dir = Path("experiments/failed_invalid_config_evolvegcn_h_u1000_train700_seed42_20260805T235518Z")
    invalid = {"canonical_experiment_id": "invalid-excluded-evolve-train700-seed42-20260805t235518z",
               "experiment_name": invalid_dir.name, "experiment_directory": str(invalid_dir),
               "model": "EvolveGCNHRegressor", "model_key": "evolve", "training_universe_count": 700,
               "seed": 42, "status": "invalid", "invalid_reason": "preserved failed invalid configuration attempt",
               "failure_reason": "configuration invalid; excluded from scientific matrix", "matrix_included": False,
               "planned_timestamp": "", "launch_timestamp": "", "completion_timestamp": "", "validation_timestamp": ""}
    atomic_json(REGISTRY, {"schema_version": "u1000_training_scaling_registry_v1", "allowed_statuses": sorted(ALLOWED),
                           "updated_at": now(), "entries": entries + [invalid]})
    refresh_reports()
    print("PASS: prepared 34 configurations and 36 valid registry entries")


def load_registry() -> dict[str, Any]:
    registry = read_json(REGISTRY)
    require(set(registry.get("allowed_statuses", [])) == ALLOWED, "registry status schema mismatch")
    return registry


def validate_configs() -> None:
    registry = load_registry(); entries = [e for e in registry["entries"] if e.get("matrix_included")]
    require(len(entries) == 36, "valid entry count is not 36")
    require(len({e["canonical_experiment_id"] for e in entries}) == 36, "duplicate canonical IDs")
    require(len({(e["model_key"], e["training_universe_count"], e["seed"]) for e in entries}) == 36, "duplicate cells")
    templates = {m: read_json(ROOT / p) for m, p in PILOT_CONFIGS.items()}
    mutable = {"experiment_name", "seed", "counts", "split_manifest_path", "split_manifest_sha256"}
    generated = 0
    for e in entries:
        manifest_path = ROOT / e["split_manifest_path"]
        validate_manifest(manifest_path, e["seed"], e["training_universe_count"])
        require(sha(manifest_path) == e["split_manifest_sha256"], f"manifest hash mismatch: {manifest_path}")
        cfg = read_json(ROOT / e["configuration_path"])
        expected = dict(templates[e["model_key"]]); expected.update({k: cfg[k] for k in mutable})
        require(cfg == expected, f"template equivalence failed: {e['configuration_path']}")
        require(cfg["seed"] == e["seed"] and cfg["counts"]["train"] == e["training_universe_count"], "config identity mismatch")
        require(cfg["split_manifest_sha256"] == e["split_manifest_sha256"], "config manifest mismatch")
        if not (e["seed"] == 42 and e["training_universe_count"] == 700): generated += 1
    require(generated == 34, "remaining config count is not 34")
    statuses = Counter(e["status"] for e in entries)
    require(statuses == {"completed": 2, "planned": 34}, f"unexpected prelaunch statuses: {dict(statuses)}")
    invalid = [e for e in registry["entries"] if e.get("status") == "invalid" and not e.get("matrix_included")]
    require(invalid, "invalid attempt not preserved/excluded")
    print("PASS: 36 unique cells, 2 completed, 34 planned, 34 generated configs, templates equivalent")


def targets() -> dict[str, float]:
    with (ROOT / TARGETS).open(newline="", encoding="utf-8") as handle:
        return {r["universe_id"]: float(r["omega_m"]) for r in csv.DictReader(handle)}


def validate_run_entry(entry: dict[str, Any], update: bool = True, runtime_json: str = "", telemetry: str = "") -> dict[str, Any]:
    exp = ROOT / entry["experiment_directory"]
    config_p, metrics_p = exp / "config.json", exp / "metrics.json"
    pred_p, checkpoint_p, log_p = exp / "predictions/test_predictions.csv", exp / "checkpoints/best_model.pt", exp / "train_log.csv"
    for p in (config_p, metrics_p, pred_p, checkpoint_p, log_p): require(p.is_file(), f"missing artifact: {p}")
    cfg, metrics, manifest = read_json(config_p), read_json(metrics_p), read_json(ROOT / entry["split_manifest_path"])
    require(cfg["model"] == entry["model"], "saved model mismatch")
    require(cfg["seed"] == cfg["trainer_invocation_seed"] == cfg["split_manifest_seed"] == entry["seed"], "saved seed mismatch")
    require(cfg["split_manifest_sha256"] == entry["split_manifest_sha256"], "saved manifest mismatch")
    require(cfg["dataset_provenance"]["dataset_sha256"] == entry["dataset_sha256"], "saved dataset mismatch")
    require((cfg["num_train_universes"], cfg["num_val_universes"], cfg["num_test_universes"]) == (entry["training_universe_count"],99,201), "saved counts mismatch")
    require(cfg["train_ids"] == manifest["train_ids"] and cfg["val_ids"] == manifest["val_ids"] and cfg["test_ids"] == manifest["test_ids"], "saved split order mismatch")
    with pred_p.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    ids = [r["universe_id"] for r in rows]
    require(len(rows) == 201 and ids == manifest["test_ids"] and len(set(ids)) == 201, "prediction ID/order mismatch")
    table = targets(); y = [float(r["true_omega_m"]) for r in rows]; p = [float(r["pred_omega_m"]) for r in rows]
    require(all(math.isfinite(v) for v in y+p), "nonfinite prediction/target")
    import struct
    require(all(v == struct.unpack("f", struct.pack("f", table[i]))[0] for i,v in zip(ids,y)), "target-table mismatch")
    err = [a-b for a,b in zip(p,y)]; mse=sum(v*v for v in err)/len(err); mae=sum(abs(v) for v in err)/len(err); rmse=math.sqrt(mse)
    ym=sum(y)/len(y); r2=1-sum(v*v for v in err)/sum((v-ym)**2 for v in y)
    for key,value in (("mse",mse),("mae",mae),("rmse",rmse),("r2",r2)):
        require(math.isclose(float(metrics["test"][key]),value,rel_tol=1e-9,abs_tol=1e-10), f"metric mismatch: {key}")
    with log_p.open(newline="", encoding="utf-8") as handle: logs=list(csv.DictReader(handle))
    best=int(metrics["best_epoch"]); require(0 < len(logs) <= 300 and best == int(logs[-1]["best_epoch"]), "epoch accounting mismatch")
    require(len(logs) == 300 or len(logs)-best >= 40, "early stopping mismatch")
    pred_sd=statistics.stdev(p); target_sd=statistics.stdev(y); repeated=len(p)-len(set(p)); residual_sd=statistics.stdev(err)
    sy=sorted(y); lo,hi=sy[len(y)//4],sy[3*len(y)//4]
    runtime = read_json(Path(runtime_json)) if runtime_json and Path(runtime_json).is_file() else {}
    peak=""
    if telemetry and Path(telemetry).is_file():
        with Path(telemetry).open(newline="", encoding="utf-8") as handle:
            vals=[float(r["used_gpu_memory_mib"]) for r in csv.DictReader(handle) if r.get("used_gpu_memory_mib")]
        peak=max(vals) if vals else ""
    if not runtime_json and entry["model_key"] in PILOT_NAMES:
        rp=ROOT/f"logs/u1000_train700_seed42_pilots/{entry['model_key']}_runtime.json"; runtime=read_json(rp) if rp.is_file() else {}
        tp=ROOT/f"logs/u1000_train700_seed42_pilots/{entry['model_key']}_gpu_telemetry.csv"
        if tp.is_file():
            with tp.open(newline="",encoding="utf-8") as handle: vals=[float(r["used_gpu_memory_mib"]) for r in csv.DictReader(handle) if r.get("used_gpu_memory_mib")]
            peak=max(vals) if vals else ""
    result={"checkpoint_path":str(checkpoint_p.relative_to(ROOT)),"metrics_path":str(metrics_p.relative_to(ROOT)),"predictions_path":str(pred_p.relative_to(ROOT)),
            "best_epoch":best,"epochs_executed":len(logs),"runtime_seconds":runtime.get("runtime_seconds",""),"peak_gpu_memory_mib":peak,
            "validation_result":"PASS","validation_timestamp":now(),"test_mae":mae,"test_rmse":rmse,"test_mse":mse,"test_r2":r2,
            "prediction_mean":sum(p)/len(p),"prediction_standard_deviation":pred_sd,"target_standard_deviation":target_sd,
            "prediction_sd_ratio":pred_sd/target_sd,"repeated_prediction_count":repeated,"repeated_prediction_fraction":repeated/len(p),
            "residual_mean":sum(err)/len(err),"residual_standard_deviation":residual_sd,
            "low_target_bias":statistics.mean(v for v,t in zip(err,y) if t<=lo),"high_target_bias":statistics.mean(v for v,t in zip(err,y) if t>=hi),
            "collapse_flag": bool(len(set(p))==1 or pred_sd/target_sd < .1)}
    return result


def find_entry(registry: dict[str, Any], ident: str) -> dict[str, Any]:
    found=[e for e in registry["entries"] if e["canonical_experiment_id"]==ident]
    require(len(found)==1, f"identity not unique: {ident}"); return found[0]


def set_status(ident: str, status: str, reason: str, runtime_json: str, telemetry: str) -> None:
    require(status in ALLOWED, "invalid status")
    registry=load_registry(); e=find_entry(registry,ident); require(e.get("matrix_included"), "cannot operate on excluded entry")
    if status=="running":
        require(e["status"]=="planned", "only planned runs may start"); e["launch_timestamp"]=now()
        e["repository_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    elif status=="completed":
        require(e["status"]=="running", "only running runs may complete"); e.update(validate_run_entry(e, runtime_json=runtime_json, telemetry=telemetry)); e["completion_timestamp"]=now()
    elif status=="failed": e["failure_reason"]=reason
    e["status"]=status; registry["updated_at"]=now(); atomic_json(REGISTRY,registry); refresh_reports()
    print(f"PASS: {ident} -> {status}")


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io
    out=io.StringIO(newline=""); w=csv.DictWriter(out,fieldnames=fields,extrasaction="ignore",lineterminator="\n"); w.writeheader(); w.writerows(rows); return out.getvalue()


def refresh_reports() -> None:
    reg=load_registry(); valid=[e for e in reg["entries"] if e.get("matrix_included")]; invalid=[e for e in reg["entries"] if not e.get("matrix_included")]
    fields=["canonical_experiment_id","experiment_name","model","training_universe_count","seed","status","validation_result","configuration_path","experiment_directory","split_manifest_path","split_manifest_sha256","dataset_sha256","repository_commit","matrix_included","invalid_reason"]
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_matrix_inventory.csv",csv_text(valid+invalid,fields))
    status_fields=["canonical_experiment_id","status","planned_timestamp","launch_timestamp","completion_timestamp","validation_timestamp","failure_reason","invalid_reason","validation_result"]
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_matrix_registry_status.csv",csv_text(valid+invalid,status_fields))
    c=Counter(e["status"] for e in valid)
    prep=f"""# U1000 Top1000 Training-Scaling Matrix Preparation

Canonical lifecycle source: `configs/experiment_registry/u1000_top1000_training_scaling_matrix.json`.
Canonical repository registry: `reports/experiment_registry/master_experiment_registry.csv`.

- Valid matrix cells: {len(valid)}
- Completed: {c['completed']}
- Planned: {c['planned']}
- Running: {c['running']}
- Failed: {c['failed']}
- Invalid/excluded preserved attempts: {len(invalid)}
- Generated remaining configurations: 34
- Models: EvolveGCN-H and Static GCN
- Training levels: Train20, Train50, Train100, Train200, Train450, Train700
- Seeds: 42, 123, 2025

The matrix is a fixed-holdout training-universe learning curve. Invalid attempts are retained as provenance and excluded from scientific aggregates.
"""
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_matrix_preparation.md",prep)


def status() -> None:
    reg=load_registry(); valid=[e for e in reg["entries"] if e.get("matrix_included")]; c=Counter(e["status"] for e in valid)
    next_e=next((e for e in execution_order(valid) if e["status"]=="planned"),None)
    completed=[e for e in valid if e["status"]=="completed"]
    last=max(completed,key=lambda e:e.get("completion_timestamp", ""),default=None)
    print(f"completed={c['completed']}/36 planned={c['planned']} running={c['running']} failed={c['failed']} invalid={sum(e['status']=='invalid' for e in reg['entries'])}")
    print(f"last_completed={last['canonical_experiment_id'] if last else 'none'} mae={last.get('test_mae','n/a') if last else 'n/a'} rmse={last.get('test_rmse','n/a') if last else 'n/a'} r2={last.get('test_r2','n/a') if last else 'n/a'}")
    print(f"next_planned={next_e['canonical_experiment_id'] if next_e else 'none'}")


def execution_order(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by={(e["model_key"],e["training_universe_count"],e["seed"]):e for e in entries}; out=[]
    for seed, levels in ((42,(450,200,100,50,20)),(123,(700,450,200,100,50,20)),(2025,(700,450,200,100,50,20))):
        for count in levels:
            for model in MODELS: out.append(by[(model,count,seed)])
    return out


def preflight(skip_environment: bool=False) -> None:
    require(ROOT == Path.cwd().resolve(), "run preflight from repository root")
    require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="thesis-sparse-integrity-hardening","wrong branch")
    require(subprocess.run(["git","diff","--quiet"],cwd=ROOT).returncode==0 and subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode==0,"tracked/staged worktree not clean")
    for rel,expected in HASHES.items(): p=ROOT/rel; require(p.is_file() and sha(p)==expected,f"hash mismatch: {rel}")
    validate_configs()
    subprocess.run([str(PYTHON),"scripts/validation/validate_u1000_top1000_sparse_dataset.py"],cwd=ROOT,check=True)
    subprocess.run([str(PYTHON),"-m","src.training.train_evolvegcn_h","--help"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([str(PYTHON),"-m","src.training.train_static_gcn","--help"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    gpu="not checked"; free="not checked"
    if not skip_environment:
        code="import torch,sys; assert torch.cuda.is_available(); n=torch.cuda.get_device_name(0); print(n); assert n=='NVIDIA L40'"
        gpu=subprocess.check_output([str(PYTHON),"-c",code],cwd=ROOT,text=True).strip()
        free=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0].strip()
    disk=subprocess.check_output(["df","-h","--output=avail",str(ROOT)],text=True).splitlines()[-1].strip()
    pilots=sum(sum(p.stat().st_size for p in (ROOT/f"experiments/{n}").rglob("*") if p.is_file()) for n in PILOT_NAMES.values())
    estimate=pilots/2*36/(1024**3)
    decision="READY FOR MANUAL TMUX LAUNCH" if not skip_environment else "CONDITIONAL READY"
    report=f"""# U1000 Top1000 Training-Scaling Matrix Preflight

Generated: {now()}

- Decision: **{decision}**
- Branch: thesis-sparse-integrity-hardening
- HEAD: {subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
- Dataset and target hashes: PASS
- Dataset validator: PASS
- Manifests: PASS (18/18)
- Registry: PASS (36 unique valid; 2 completed; 34 planned; invalid excluded)
- Configuration equivalence: PASS (34 generated; immutable templates)
- Trainer CLI: PASS
- CUDA/GPU: {gpu}
- Free GPU memory MiB: {free}
- Available disk: {disk}
- Estimated 36-run artifacts from pilots: {estimate:.2f} GiB
- No trainer or epoch loop was started by preflight.
"""
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_matrix_preflight.md",report)
    print(decision)


def aggregate() -> None:
    reg=load_registry(); rows=[e for e in reg["entries"] if e.get("matrix_included")]
    require(all(e["status"]=="completed" and e["validation_result"]=="PASS" for e in rows),"aggregation requires 36 validated completions")
    fields=["model","seed","training_universe_count","validation_count","test_count","unused_count","canonical_experiment_id","configuration_path","experiment_directory","split_manifest_path","split_manifest_sha256","dataset_sha256","repository_commit","status","validation_result","best_epoch","epochs_executed","runtime_seconds","peak_gpu_memory_mib","test_mae","test_rmse","test_mse","test_r2","prediction_mean","prediction_standard_deviation","target_standard_deviation","prediction_sd_ratio","repeated_prediction_count","repeated_prediction_fraction","residual_mean","residual_standard_deviation","low_target_bias","high_target_bias","collapse_flag"]
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_all_runs.csv",csv_text(rows,fields))
    summaries=[]
    for model in ("EvolveGCNHRegressor","StaticGCNRegressor"):
        for count in LEVELS:
            g=[e for e in rows if e["model"]==model and e["training_universe_count"]==count]
            s={"model":model,"training_universe_count":count,"successful_seed_count":len(g),"failed_seed_count":0}
            for field in ("test_mae","test_rmse","test_r2","prediction_sd_ratio","runtime_seconds"):
                vals=[float(e[field]) for e in g]; s[f"mean_{field}"]=statistics.mean(vals); s[f"sample_sd_{field}"]=statistics.stdev(vals); s[f"median_{field}"]=statistics.median(vals)
            summaries.append(s)
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_model_summary.csv",csv_text(summaries,list(summaries[0])))
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_seed_summary.csv",csv_text(rows,fields))
    diag=[k for k in fields if k in {"model","seed","training_universe_count","prediction_mean","prediction_standard_deviation","target_standard_deviation","prediction_sd_ratio","repeated_prediction_count","repeated_prediction_fraction","residual_mean","residual_standard_deviation","low_target_bias","high_target_bias","collapse_flag"}]
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_prediction_diagnostics.csv",csv_text(rows,diag))
    atomic_text(REPORT_DIR/"u1000_top1000_training_scaling_runtime_summary.csv",csv_text(summaries,[k for k in summaries[0] if "runtime" in k or k in {"model","training_universe_count","successful_seed_count","failed_seed_count"}]))
    for fn,title in (("u1000_top1000_training_scaling_registry_audit.md","Registry Audit"),("u1000_top1000_training_scaling_final_audit.md","Final Audit"),("u1000_top1000_training_scaling_scientific_interpretation.md","Scientific Interpretation")):
        atomic_text(REPORT_DIR/fn,f"# U1000 Top1000 Training Scaling {title}\n\nAll 36 intended cells completed and passed artifact validation. Interpretations must be updated from the generated validated tables.\n")


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true"); g.add_argument("--validate-configs",action="store_true"); g.add_argument("--refresh",action="store_true")
    g.add_argument("--status",action="store_true"); g.add_argument("--preflight",action="store_true"); g.add_argument("--validate-run")
    g.add_argument("--set-status"); g.add_argument("--aggregate",action="store_true")
    p.add_argument("--status-value",choices=sorted(ALLOWED)); p.add_argument("--reason",default=""); p.add_argument("--runtime-json",default=""); p.add_argument("--telemetry",default=""); p.add_argument("--skip-environment",action="store_true")
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.validate_configs: validate_configs()
    elif a.refresh: refresh_reports(); print("PASS: matrix reports refreshed")
    elif a.status: status()
    elif a.preflight: preflight(a.skip_environment)
    elif a.validate_run:
        e=find_entry(load_registry(),a.validate_run); print(json.dumps(validate_run_entry(e,False,a.runtime_json,a.telemetry),indent=2))
    elif a.set_status:
        require(a.status_value is not None,"--status-value required"); set_status(a.set_status,a.status_value,a.reason,a.runtime_json,a.telemetry)
    elif a.aggregate: aggregate()
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc: print(f"FAIL: {exc}",file=sys.stderr); raise SystemExit(1)
