#!/usr/bin/env python3
"""Read-only symmetry diagnostic for the frozen U1000 Top1500 k=8 Static GCNs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.camels_graph_utils import build_sparse_knn_edge_index  # noqa: E402
from src.evaluation.symmetry import (  # noqa: E402
    FEATURE_NAMES,
    identity_rotation_id,
    proper_cube_rotations,
    rotation_metadata,
    transform_raw7_features,
    translation_vectors,
)
from src.models.static_gcn import StaticGCNRegressor  # noqa: E402
from src.training.sparse_batch import collate_sparse_static, sparse_batch_to  # noqa: E402
from src.training.split_manifest import load_split_manifest, ordered_id_hash  # noqa: E402
from src.training.train_static_gcn import load_dataset  # noqa: E402


DATASET_SHA = "ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
CHECKPOINT_SHAS = {
    42: "49ea8189bec30852bfa0879012144df8b7982291892b976368970c2c0926a487",
    123: "dce2c8cac85ab8a3f8c3e6ae3247fcaa3cf818c9ae6364b3adf5598bd3ef50eb",
    2025: "8be14a8dc8e7bb9a6ffb2f20ab5afe9df80e98740dc9db12584426d656b3809f",
}
SPLIT_SHAS = {
    42: "f5556ec5c193e7cb80f2231705edbdae32d4de206889dec308a819bdde427ab7",
    123: "18a295106ec844848053f3040d2be3cdf73a443010be31e6e2cd962362982471",
    2025: "c233c0631b1a24d963ffccc7c6389054fddf135a7aabc4dc4ab7bf5976fab3a9",
}
BOX_SIZE = 25.0
K = 8
ATOL = 1e-6
RTOL = 1e-5
RELATIVE_EPSILON = 1e-12
PILOT_NAMESPACE = "symmetry-pilot-v1"
DEFAULT_DATASET = ROOT / "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
DEFAULT_PROTOCOL = ROOT / "reports/experiment_registry/u1000_top1500_knn_k8_symmetry_protocol.json"
DEFAULT_PILOT_OUTPUT = ROOT / "reports/symmetry/u1000_top1500_knn_k8_static_gcn/pilot"
DEFAULT_FULL_OUTPUT = ROOT / "reports/symmetry/u1000_top1500_knn_k8_static_gcn/full"
CANONICAL_ROOT = Path("/home/ml/thesis-camels")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_paths() -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    checkpoints: dict[int, Path] = {}
    configs: dict[int, Path] = {}
    splits: dict[int, Path] = {}
    for seed in CHECKPOINT_SHAS:
        experiment = CANONICAL_ROOT / (
            f"experiments/static_gcn_u1000_top1500_sparse_train700_seed{seed}_"
            "none_h32_l3_mean_mlp_final"
        )
        checkpoints[seed] = experiment / "checkpoints/best_model.pt"
        configs[seed] = experiment / "config.json"
        splits[seed] = ROOT / f"configs/splits/u1000_top1500_none_k8_sparse/seed{seed}_train700.json"
    return checkpoints, configs, splits


def audit_repository(expected_head: str | None) -> str:
    require(ROOT.resolve() == Path("/home/ml/thesis-camels-notebook15"), f"wrong worktree: {ROOT}")
    require(git("branch", "--show-current") == "thesis-notebook15-graph-representation", "wrong branch")
    head = git("rev-parse", "HEAD")
    if expected_head:
        require(head == expected_head, f"HEAD mismatch: expected {expected_head}, got {head}")
    require(not git("diff", "--name-only"), "tracked unstaged changes are present")
    require(not git("diff", "--cached", "--name-only"), "staged changes are present")
    return head


def audit_frozen_inputs(dataset_path: Path, checkpoints: dict[int, Path],
                        configs: dict[int, Path], splits: dict[int, Path]) -> dict[str, Any]:
    require(dataset_path.is_file(), f"dataset missing: {dataset_path}")
    dataset_sha = sha256_file(dataset_path)
    require(dataset_sha == DATASET_SHA, f"dataset SHA mismatch: {dataset_sha}")
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = read_json(metadata_path)
    expected_metadata = {
        "checksum": DATASET_SHA, "box_size": BOX_SIZE, "k": K,
        "graph_mode": "knn", "graph_storage": "sparse_edge_index",
        "normalization": "none", "num_nodes": 1500,
        "feature_names": list(FEATURE_NAMES), "num_universes_successful": 1000,
    }
    for key, expected in expected_metadata.items():
        require(metadata.get(key) == expected,
                f"dataset metadata {key} mismatch: {metadata.get(key)!r} != {expected!r}")

    records = {}
    for seed in CHECKPOINT_SHAS:
        checkpoint_sha = sha256_file(checkpoints[seed])
        split_sha = sha256_file(splits[seed])
        require(checkpoint_sha == CHECKPOINT_SHAS[seed], f"seed {seed} checkpoint SHA mismatch")
        require(split_sha == SPLIT_SHAS[seed], f"seed {seed} split SHA mismatch")
        config = read_json(configs[seed])
        expected_config = {
            "model": "StaticGCNRegressor", "dataset_identity": DATASET_SHA,
            "dataset_format": "temporal_final_snapshot", "seed": seed,
            "hidden_dim": 32, "num_layers": 3, "dropout": 0.2,
            "graph_pooling": "mean", "conv_type": "gcn", "node_features": 7,
            "num_nodes": 1500, "num_test_universes": 201,
            "split_manifest_sha256": SPLIT_SHAS[seed],
        }
        for key, expected in expected_config.items():
            require(config.get(key) == expected,
                    f"seed {seed} config {key} mismatch: {config.get(key)!r} != {expected!r}")
        require(config.get("dataset_provenance", {}).get("dataset_sha256") == DATASET_SHA,
                f"seed {seed} config dataset provenance mismatch")
        records[seed] = {
            "checkpoint_path": str(checkpoints[seed]), "checkpoint_sha256": checkpoint_sha,
            "config_path": str(configs[seed]), "split_manifest_path": str(splits[seed]),
            "split_manifest_sha256": split_sha,
        }
    return {"dataset_path": str(dataset_path), "dataset_sha256": dataset_sha,
            "metadata_path": str(metadata_path), "metadata": metadata, "seeds": records}


def protocol_document(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "camels_knn_symmetry_protocol_v1",
        "status": "FROZEN_BEFORE_PILOT",
        "source_starting_commit": "d6094a2c26479256572c9724475b46d1fafa3039",
        "dataset": {"path": audit["dataset_path"], "sha256": DATASET_SHA},
        "graph_representation": {"family": "periodic_sparse_knn", "k": K,
                                 "box_size_h_inverse_mpc": BOX_SIZE, "reuse_edges_for_inference": True,
                                 "pilot_rebuild_exact_equality_required": True},
        "features": {"order": list(FEATURE_NAMES), "dimension": 7, "normalization": "none"},
        "model": {"class": "StaticGCNRegressor", "hidden_dim": 32, "num_layers": 3,
                  "activation": "ReLU", "dropout": 0.2, "layer_norm": True,
                  "residual": True, "model_side_self_loops": True, "pooling": "mean",
                  "head": [32, 32, 16, 1]},
        "checkpoints": {str(seed): record for seed, record in audit["seeds"].items()},
        "translations": [{"id": key, "delta": list(value)}
                         for key, value in translation_vectors(BOX_SIZE).items()],
        "rotations": rotation_metadata(),
        "row_vector_convention": {"position": "(p @ R.T) mod L", "velocity": "v @ R.T"},
        "conditions_per_universe": {"baseline": 1, "translations": 4, "rotations": 24, "total": 29},
        "pilot": {"size_per_seed": 32, "selection":
                  "lexicographically smallest SHA-256('symmetry-pilot-v1:<universe_id>')",
                  "target_independent": True},
        "full": {"test_universes_per_seed": 201, "seeds": [42, 123, 2025],
                 "core_forwards": 17487},
        "tolerance": {"atol": ATOL, "rtol": RTOL,
                      "failure": "abs(transformed-baseline) > atol + rtol*abs(baseline)",
                      "relative_epsilon_reporting_only": RELATIVE_EPSILON},
        "metrics": ["N", "mean_signed_delta", "mean_absolute_delta", "rms_delta",
                    "median_absolute_delta", "p95_absolute_delta", "maximum_absolute_delta",
                    "tolerance_failure_count", "tolerance_failure_fraction"],
        "controls": {"minimum_repeated_original_inferences": 5,
                     "stop_if_identity_materially_exceeds_tolerance": True,
                     "checkpoint_sha256_before_and_after": True},
        "prohibitions": ["training", "gradient_updates", "optimizer_creation", "model_selection",
                         "hyperparameter_tuning", "checkpoint_writes", "dataset_rebuild"],
    }


def pilot_ids(test_ids: Sequence[str], size: int) -> tuple[list[str], str]:
    require(0 < size <= len(test_ids), "pilot size must be positive and no larger than test partition")
    ranked = sorted(test_ids, key=lambda item: hashlib.sha256(
        f"{PILOT_NAMESPACE}:{item}".encode("utf-8")).hexdigest())
    selected = ranked[:size]
    return selected, ordered_id_hash(selected)


def graph_record(sample: dict[str, Any], features: torch.Tensor) -> dict[str, Any]:
    return {"graph_storage": "sparse_edge_index", "x": features,
            "edge_index": sample["edge_index"], "edge_weight": sample.get("edge_weight"),
            "mask": sample["mask"]}


def predict(model: StaticGCNRegressor, graphs: Sequence[dict[str, Any]], device: torch.device) -> np.ndarray:
    batch = sparse_batch_to(collate_sparse_static(graphs), device)
    with torch.inference_mode():
        values = model(A=batch).detach().cpu().reshape(-1).numpy().astype(np.float64)
    return values


def load_model(checkpoint_path: Path, device: torch.device) -> StaticGCNRegressor:
    model = StaticGCNRegressor(node_features=7, hidden_dim=32, num_layers=3, dropout=0.2,
                               graph_pooling="mean", conv_type="gcn", add_self_loops=True,
                               use_layer_norm=True, residual=True).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not model.training, "model failed to enter eval mode")
    require(all(parameter.grad is None for parameter in model.parameters()), "model has accumulated gradients")
    return model


def delta_row(seed: int, universe_id: str, family: str, transform_id: str,
              baseline: float, transformed: float) -> dict[str, Any]:
    signed = transformed - baseline
    absolute = abs(signed)
    tolerance = ATOL + RTOL * abs(baseline)
    return {"seed": seed, "universe_id": universe_id, "transform_family": family,
            "transform_id": transform_id, "baseline_prediction": baseline,
            "transformed_prediction": transformed, "signed_delta": signed,
            "absolute_delta": absolute,
            "relative_delta": absolute / max(abs(baseline), RELATIVE_EPSILON),
            "tolerance": tolerance, "tolerance_failure": absolute > tolerance}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    signed = np.asarray([row["signed_delta"] for row in rows], dtype=np.float64)
    absolute = np.abs(signed)
    failures = np.asarray([row["tolerance_failure"] for row in rows], dtype=bool)
    return {"N": len(rows), "mean_signed_delta": float(signed.mean()),
            "mean_absolute_delta": float(absolute.mean()),
            "rms_delta": float(np.sqrt(np.mean(signed ** 2))),
            "median_absolute_delta": float(np.median(absolute)),
            "p95_absolute_delta": float(np.percentile(absolute, 95)),
            "maximum_absolute_delta": float(absolute.max()),
            "tolerance_failure_count": int(failures.sum()),
            "tolerance_failure_fraction": float(failures.mean())}


def all_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["transform_family"], row["transform_id"])].append(row)
        grouped[(row["seed"], row["transform_family"], "AGGREGATE")].append(row)
        grouped[(row["seed"], "all", "AGGREGATE")].append(row)
        grouped[("all_seeds", row["transform_family"], row["transform_id"])].append(row)
        grouped[("all_seeds", row["transform_family"], "AGGREGATE")].append(row)
        grouped[("all_seeds", "all", "AGGREGATE")].append(row)
    return {f"seed={seed}|family={family}|transform={transform}": summarize(values)
            for (seed, family, transform), values in sorted(grouped.items(), key=lambda item: str(item[0]))}


def edge_mismatch_count(original: np.ndarray, rebuilt: np.ndarray) -> int:
    left = set(map(tuple, original.T.tolist()))
    right = set(map(tuple, rebuilt.T.tolist()))
    return len(left.symmetric_difference(right))


def topology_pilot(data: dict[str, Any], selections: dict[int, list[str]]) -> dict[str, Any]:
    rotations = proper_cube_rotations()
    translations = translation_vectors(BOX_SIZE)
    family_stats = {
        "translation": {"graphs": 0, "transformations": 4, "comparisons": 0,
                        "exact_matches": 0, "mismatches": 0, "maximum_edge_mismatch_count": 0},
        "rotation": {"graphs": 0, "transformations": 24, "comparisons": 0,
                     "exact_matches": 0, "mismatches": 0, "maximum_edge_mismatch_count": 0},
    }
    details = []
    for seed, universe_ids in selections.items():
        for universe_id in universe_ids:
            sample = data[universe_id]
            features = sample["X"]
            mask = sample["mask"].numpy()
            original = sample["edge_index"].numpy()
            for family, transforms in (("translation", translations), ("rotation", rotations)):
                family_stats[family]["graphs"] += 1
                for transform_id, transform in transforms.items():
                    kwargs = {family: transform}
                    changed = transform_raw7_features(features, box_size=BOX_SIZE, **kwargs)
                    rebuilt = build_sparse_knn_edge_index(changed[:, 1:4].numpy(), mask, k=K,
                                                          periodic_boundary=True, box_size=BOX_SIZE)
                    mismatch = edge_mismatch_count(original, rebuilt)
                    stats = family_stats[family]
                    stats["comparisons"] += 1
                    stats["exact_matches"] += int(mismatch == 0 and np.array_equal(original, rebuilt))
                    stats["mismatches"] += int(not np.array_equal(original, rebuilt))
                    stats["maximum_edge_mismatch_count"] = max(stats["maximum_edge_mismatch_count"], mismatch)
                    if mismatch:
                        details.append({"seed": seed, "universe_id": universe_id,
                                        "family": family, "transform_id": transform_id,
                                        "edge_mismatch_count": mismatch,
                                        "original_edges": int(original.shape[1]),
                                        "rebuilt_edges": int(rebuilt.shape[1])})
    return {"families": family_stats, "mismatch_details": details,
            "status": "PASS" if not details else "FAIL"}


def run_diagnostic(args: argparse.Namespace, audit: dict[str, Any],
                   checkpoints: dict[int, Path], splits: dict[int, Path]) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output destination already exists: {output_dir}")
    protocol_sha = sha256_file(args.protocol)
    if args.expected_protocol_sha:
        require(protocol_sha == args.expected_protocol_sha, "protocol SHA mismatch")
    protocol = read_json(args.protocol)
    require(protocol.get("status") == "FROZEN_BEFORE_PILOT", "symmetry protocol is not frozen")

    data = load_dataset(args.dataset, dataset_format="temporal_final_snapshot")
    require(len(data) == 1000, f"expected 1000 universes, got {len(data)}")
    manifests = {seed: load_split_manifest(path, list(data), DATASET_SHA, expected_seed=seed)
                 for seed, path in splits.items()}
    for seed, manifest in manifests.items():
        require(len(manifest["test_ids"]) == 201, f"seed {seed} test count is not 201")

    if args.full:
        pilot = read_json(args.pilot_summary)
        require(pilot.get("implementation_verdict") == "IMPLEMENTATION PASS — FULL DIAGNOSTIC AUTHORIZED",
                "pilot verdict does not authorize the full diagnostic")
        selections = {seed: list(manifest["test_ids"]) for seed, manifest in manifests.items()}
        selection_hashes = {seed: ordered_id_hash(ids) for seed, ids in selections.items()}
    else:
        selections_and_hashes = {seed: pilot_ids(manifest["test_ids"], args.pilot_size)
                                 for seed, manifest in manifests.items()}
        selections = {seed: value[0] for seed, value in selections_and_hashes.items()}
        selection_hashes = {seed: value[1] for seed, value in selections_and_hashes.items()}

    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    pre_hashes = {seed: sha256_file(path) for seed, path in checkpoints.items()}
    rows: list[dict[str, Any]] = []
    numerical_floor: dict[int, Any] = {}
    model_audit: dict[int, Any] = {}
    rotations = proper_cube_rotations()
    translations = translation_vectors(BOX_SIZE)

    for seed, universe_ids in selections.items():
        model = load_model(checkpoints[seed], device)
        originals = [graph_record(data[universe_id], data[universe_id]["X"]) for universe_id in universe_ids]
        repeats = [predict(model, originals, device) for _ in range(5)]
        baseline = repeats[0]
        repeat_deltas = np.stack(repeats[1:]) - baseline
        repeat_tolerances = ATOL + RTOL * np.abs(baseline)
        numerical_floor[seed] = {
            "repeated_original_count": 5,
            "maximum_absolute_repeat_delta": float(np.max(np.abs(repeat_deltas))),
            "rms_repeat_delta": float(np.sqrt(np.mean(repeat_deltas ** 2))),
            "mean_absolute_repeat_delta": float(np.mean(np.abs(repeat_deltas))),
            "repeat_tolerance_failures": int((np.abs(repeat_deltas) > repeat_tolerances).sum()),
        }
        require(numerical_floor[seed]["repeat_tolerance_failures"] == 0,
                f"seed {seed} repeated identity inference exceeds frozen tolerance")
        for family, transforms in (("translation", translations), ("rotation", rotations)):
            for transform_id, transform in transforms.items():
                graphs = []
                for universe_id in universe_ids:
                    sample = data[universe_id]
                    changed = transform_raw7_features(sample["X"], box_size=BOX_SIZE,
                                                      **{family: transform})
                    graphs.append(graph_record(sample, changed))
                transformed = predict(model, graphs, device)
                for index, universe_id in enumerate(universe_ids):
                    rows.append(delta_row(seed, universe_id, family, transform_id,
                                          float(baseline[index]), float(transformed[index])))
        identity_rows = [row for row in rows if row["seed"] == seed and
                         row["transform_family"] == "rotation" and
                         row["transform_id"] == identity_rotation_id()]
        numerical_floor[seed]["baseline_vs_identity_rotation"] = summarize(identity_rows)
        require(numerical_floor[seed]["baseline_vs_identity_rotation"]["tolerance_failure_count"] == 0,
                f"seed {seed} baseline versus identity rotation exceeds frozen tolerance")
        model_audit[seed] = {"eval_mode": not model.training, "gradients_enabled": False,
                             "parameter_grads_all_none": all(p.grad is None for p in model.parameters())}
        del model

    topology = None
    if args.pilot:
        topology = topology_pilot(data, selections)
        require(topology["status"] == "PASS", "pilot topology invariance failed; full run is blocked")
    post_hashes = {seed: sha256_file(path) for seed, path in checkpoints.items()}
    require(pre_hashes == post_hashes, "checkpoint bytes changed during diagnostic")
    summary = {
        "schema_version": "camels_knn_symmetry_pilot_v1" if args.pilot else "camels_knn_symmetry_full_v1",
        "mode": "pilot" if args.pilot else "full",
        "implementation_verdict": ("IMPLEMENTATION PASS — FULL DIAGNOSTIC AUTHORIZED"
                                   if args.pilot else "FULL DIAGNOSTIC COMPLETE"),
        "git_commit": git("rev-parse", "HEAD"), "execution_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started, "device": str(device),
        "dataset_sha256": DATASET_SHA, "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha, "checkpoint_sha256_before": pre_hashes,
        "checkpoint_sha256_after": post_hashes, "checkpoint_integrity_unchanged": pre_hashes == post_hashes,
        "split_manifest_sha256": SPLIT_SHAS, "selected_ids": selections,
        "selected_id_hashes": selection_hashes, "selection_target_independent": True,
        "numerical_floor": numerical_floor, "model_audit": model_audit,
        "topology_invariance": topology, "summaries": all_summaries(rows),
    }
    prefix = "symmetry_pilot" if args.pilot else "symmetry_full"
    with (output_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / f"{prefix}_summary.json", summary)
    (output_dir / f"{prefix}_report.md").write_text(
        f"# {'Pilot' if args.pilot else 'Full'} symmetry diagnostic\n\n"
        f"- Status: {summary['implementation_verdict']}\n"
        f"- Runtime seconds: {summary['runtime_seconds']:.3f}\n"
        f"- Device: {device}\n"
        f"- Machine-readable detail: `{prefix}_summary.json`\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    default_checkpoints, _, default_splits = canonical_paths()
    for seed in CHECKPOINT_SHAS:
        parser.add_argument(f"--checkpoint-seed{seed}", type=Path,
                            default=default_checkpoints[seed])
        parser.add_argument(f"--split-seed{seed}", type=Path, default=default_splits[seed])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--write-protocol", action="store_true",
                        help="With --audit, write the frozen protocol after all input checks pass.")
    parser.add_argument("--pilot-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot-summary", type=Path,
                        default=DEFAULT_PILOT_OUTPUT / "symmetry_pilot_summary.json")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-protocol-sha")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_FULL_OUTPUT if args.full else DEFAULT_PILOT_OUTPUT
    return args


def main() -> int:
    args = parse_args()
    try:
        head = audit_repository(args.expected_head)
        checkpoints = {seed: getattr(args, f"checkpoint_seed{seed}").resolve()
                       for seed in CHECKPOINT_SHAS}
        splits = {seed: getattr(args, f"split_seed{seed}").resolve()
                  for seed in CHECKPOINT_SHAS}
        configs = {seed: checkpoint.parent.parent / "config.json"
                   for seed, checkpoint in checkpoints.items()}
        audit = audit_frozen_inputs(args.dataset.resolve(), checkpoints, configs, splits)
        if args.audit:
            if args.write_protocol:
                require(not args.protocol.exists(), f"refusing to overwrite protocol: {args.protocol}")
                args.protocol.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.protocol, protocol_document(audit))
            elif args.protocol.exists() and args.expected_protocol_sha:
                require(sha256_file(args.protocol) == args.expected_protocol_sha, "protocol SHA mismatch")
            print(json.dumps({"status": "AUDIT PASS", "head": head,
                              "dataset_sha256": audit["dataset_sha256"],
                              "checkpoint_sha256": CHECKPOINT_SHAS,
                              "split_manifest_sha256": SPLIT_SHAS}, sort_keys=True))
            return 0
        summary = run_diagnostic(args, audit, checkpoints, splits)
        print(json.dumps({"status": summary["implementation_verdict"],
                          "output_directory": str(args.output_dir),
                          "runtime_seconds": summary["runtime_seconds"]}, sort_keys=True))
        return 0
    except Exception as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
