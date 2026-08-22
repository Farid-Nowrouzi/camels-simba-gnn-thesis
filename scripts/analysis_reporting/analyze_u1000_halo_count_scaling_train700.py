#!/usr/bin/env python3
"""Analyze controlled U1000 Train700 halo-count scaling after all runs exist."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
HALO_COUNTS = (500, 750, 1000, 1500)
SEEDS = (42, 123, 2025)
MODELS = ("Static GCN", "EvolveGCN-H")
OUTPUT_DIR = Path("reports/analysis/u1000_halo_count_scaling_train700")
EXPECTED_OUTPUTS = (
    "seed_level_results.csv",
    "halo_count_summary.csv",
    "paired_seed_differences.csv",
    "plotting_data.csv",
    "halo_count_scaling.png",
    "scientific_report.md",
    "validation.md",
)
MAE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class SeedResult:
    model: str
    halo_count: int
    seed: int
    test_mae: float
    reported_test_mae: float
    experiment_name: str
    experiment_path: str
    dataset_path: str
    split_manifest_path: str
    dataset_sha256: str
    split_manifest_sha256: str


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def experiment_name(model: str, halo_count: int, seed: int) -> str:
    require(model in MODELS, f"unsupported model: {model}")
    require(halo_count in HALO_COUNTS, f"unsupported halo count: {halo_count}")
    require(seed in SEEDS, f"unsupported seed: {seed}")
    if model == "Static GCN":
        return (
            f"static_gcn_u1000_top{halo_count}_sparse_train700_seed{seed}_"
            "none_h32_l3_mean_mlp_final"
        )
    return (
        f"evolvegcn_h_u1000_top{halo_count}_sparse_train700_seed{seed}_"
        "none_h32_l2_mean_temporal_mean_linear"
    )


def dataset_relative_path(halo_count: int) -> Path:
    directory = Path(
        f"data/processed/temporal_1000u_none_top{halo_count}_periodic_knn_sparse"
    )
    return directory / (
        f"camels_1000u_temporal_logmass_none_top{halo_count}_periodic_knn_sparse.pt"
    )


def split_relative_path(halo_count: int, seed: int) -> Path:
    return Path(
        f"configs/splits/u1000_top{halo_count}_none_k8_sparse/seed{seed}_train700.json"
    )


def _expected_model_config(model: str) -> dict[str, Any]:
    common = {
        "hidden_dim": 32,
        "dropout": 0.2,
        "graph_pooling": "mean",
        "batch_size": 8 if model == "Static GCN" else 4,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "epochs": 300,
        "patience": 40,
        "grad_clip_norm": 1.0,
        "checkpoint_criterion": "minimum_validation_mse",
        "num_layers": 3 if model == "Static GCN" else 2,
    }
    if model == "Static GCN":
        common.update(
            {
                "model": "StaticGCNRegressor",
                "conv_type": "gcn",
                "dataset_format": "temporal_final_snapshot",
            }
        )
    else:
        common.update(
            {
                "model": "EvolveGCNHRegressor",
                "temporal_pooling": "mean",
                "head_type": "linear",
                "activation": "relu",
                "num_snapshots": 5,
                "use_summary_features": False,
                "normalize_target": False,
            }
        )
    return common


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def recompute_mae(predictions_path: Path, expected_ids: list[str]) -> float:
    errors: list[float] = []
    ids: list[str] = []
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"prediction header missing: {predictions_path}")
        required = {"universe_id", "true_omega_m", "pred_omega_m"}
        require(required.issubset(reader.fieldnames), f"prediction columns missing: {predictions_path}")
        for row in reader:
            true_value = float(row["true_omega_m"])
            predicted = float(row["pred_omega_m"])
            require(math.isfinite(true_value) and math.isfinite(predicted), "nonfinite prediction value")
            ids.append(row["universe_id"])
            errors.append(abs(predicted - true_value))
    require(ids == expected_ids, f"prediction IDs/order do not match test manifest: {predictions_path}")
    require(len(errors) == 201, f"prediction row count is not 201: {predictions_path}")
    return statistics.fmean(errors)


def load_seed_result(root: Path, model: str, halo_count: int, seed: int) -> SeedResult:
    name = experiment_name(model, halo_count, seed)
    experiment = root / "experiments" / name
    required = (
        experiment / "config.json",
        experiment / "metrics.json",
        experiment / "train_log.csv",
        experiment / "predictions/test_predictions.csv",
        experiment / "checkpoints/best_model.pt",
    )
    missing = [path for path in required if not path.is_file()]
    require(not missing, f"{name}: missing artifacts: {[path.name for path in missing]}")

    config = read_json(experiment / "config.json")
    metrics = read_json(experiment / "metrics.json")
    expected_dataset = dataset_relative_path(halo_count)
    expected_split = split_relative_path(halo_count, seed)
    require(config.get("experiment_name") == name, f"{name}: config experiment name mismatch")
    require(config.get("seed") == seed, f"{name}: config seed mismatch")
    require(config.get("dataset_path") == expected_dataset.as_posix(), f"{name}: dataset path mismatch")
    require(config.get("split_source") == expected_split.as_posix(), f"{name}: split source mismatch")
    require(config.get("num_nodes") == halo_count, f"{name}: node count mismatch")
    require(config.get("num_total_universes") == 1000, f"{name}: universe count mismatch")
    require(config.get("num_train_universes") == 700, f"{name}: train count mismatch")
    require(config.get("num_val_universes") == 99, f"{name}: validation count mismatch")
    require(config.get("num_test_universes") == 201, f"{name}: test count mismatch")
    require(config.get("node_features") == 7, f"{name}: node feature count mismatch")
    for key, expected in _expected_model_config(model).items():
        require(_equal(config.get(key), expected), f"{name}: {key}={config.get(key)!r}, expected {expected!r}")

    scheduler = config.get("scheduler", {})
    require(
        scheduler == {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
        },
        f"{name}: scheduler mismatch",
    )
    split = read_json(root / expected_split)
    require(split.get("seed") == seed, f"{name}: split seed mismatch")
    require(split.get("train_ids") == config.get("train_ids"), f"{name}: train IDs mismatch")
    require(split.get("val_ids") == config.get("val_ids"), f"{name}: val IDs mismatch")
    require(split.get("test_ids") == config.get("test_ids"), f"{name}: test IDs mismatch")
    require(split.get("counts", {}).get("train") == 700, f"{name}: split train count mismatch")
    require(split.get("counts", {}).get("val") == 99, f"{name}: split val count mismatch")
    require(split.get("counts", {}).get("test") == 201, f"{name}: split test count mismatch")

    dataset_metadata = read_json(root / expected_dataset.with_suffix(".metadata.json"))
    dataset_sha = str(dataset_metadata.get("checksum", ""))
    require(config.get("dataset_identity") == dataset_sha, f"{name}: dataset identity mismatch")
    require(split.get("dataset_identity") == dataset_sha, f"{name}: split dataset identity mismatch")
    require(dataset_metadata.get("num_nodes") == halo_count, f"{name}: metadata Top-N mismatch")
    require(dataset_metadata.get("num_snapshots") == 5, f"{name}: metadata snapshot mismatch")
    require(dataset_metadata.get("normalization") == "none", f"{name}: metadata normalization mismatch")

    reported = float(metrics.get("test", {}).get("mae"))
    require(math.isfinite(reported), f"{name}: reported test MAE is nonfinite")
    recomputed = recompute_mae(required[3], split["test_ids"])
    require(math.isclose(recomputed, reported, rel_tol=0.0, abs_tol=MAE_TOLERANCE), f"{name}: recomputed MAE mismatch")
    return SeedResult(
        model=model,
        halo_count=halo_count,
        seed=seed,
        test_mae=recomputed,
        reported_test_mae=reported,
        experiment_name=name,
        experiment_path=experiment.relative_to(root).as_posix(),
        dataset_path=expected_dataset.as_posix(),
        split_manifest_path=expected_split.as_posix(),
        dataset_sha256=dataset_sha,
        split_manifest_sha256=str(config.get("split_manifest_sha256", "")),
    )


def collect(root: Path) -> tuple[list[SeedResult], list[str]]:
    rows: list[SeedResult] = []
    missing: list[str] = []
    for model in MODELS:
        for halo_count in HALO_COUNTS:
            for seed in SEEDS:
                name = experiment_name(model, halo_count, seed)
                if not (root / "experiments" / name).exists():
                    missing.append(name)
                    continue
                rows.append(load_seed_result(root, model, halo_count, seed))
    return rows, missing


def summary_rows(rows: Iterable[SeedResult]) -> list[dict[str, Any]]:
    indexed = {(row.model, row.halo_count, row.seed): row.test_mae for row in rows}
    output: list[dict[str, Any]] = []
    for model in MODELS:
        previous_mean: float | None = None
        for halo_count in HALO_COUNTS:
            values = [indexed[(model, halo_count, seed)] for seed in SEEDS]
            mean = statistics.fmean(values)
            delta = "" if previous_mean is None else mean - previous_mean
            percent = "" if previous_mean is None else 100.0 * (mean - previous_mean) / previous_mean
            output.append(
                {
                    "halo_count": halo_count,
                    "model": model,
                    "seed_42_mae": values[0],
                    "seed_123_mae": values[1],
                    "seed_2025_mae": values[2],
                    "mean_mae": mean,
                    "sample_sd": statistics.stdev(values),
                    "delta_mae_from_previous": delta,
                    "percent_change": percent,
                }
            )
            previous_mean = mean
    return output


def paired_rows(rows: Iterable[SeedResult]) -> list[dict[str, Any]]:
    indexed = {(row.model, row.halo_count, row.seed): row.test_mae for row in rows}
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for previous, current in zip(HALO_COUNTS, HALO_COUNTS[1:]):
            deltas = [indexed[(model, current, seed)] - indexed[(model, previous, seed)] for seed in SEEDS]
            previous_mean = statistics.fmean(indexed[(model, previous, seed)] for seed in SEEDS)
            mean_delta = statistics.fmean(deltas)
            output.append(
                {
                    "model": model,
                    "previous_halo_count": previous,
                    "new_halo_count": current,
                    "seed_42_delta_mae": deltas[0],
                    "seed_123_delta_mae": deltas[1],
                    "seed_2025_delta_mae": deltas[2],
                    "mean_delta_mae": mean_delta,
                    "percent_change": 100.0 * mean_delta / previous_mean,
                    "seeds_improving": sum(delta < 0 for delta in deltas),
                    "seeds_worsening": sum(delta > 0 for delta in deltas),
                    "seeds_unchanged": sum(delta == 0 for delta in deltas),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"refusing empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows: list[SeedResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = {(row.model, row.halo_count, row.seed): row.test_mae for row in rows}
    colors = {"Static GCN": "#1f77b4", "EvolveGCN-H": "#d62728"}
    markers = {"Static GCN": "o", "EvolveGCN-H": "s"}
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        means: list[float] = []
        sds: list[float] = []
        for halo_count in HALO_COUNTS:
            values = [indexed[(model, halo_count, seed)] for seed in SEEDS]
            means.append(statistics.fmean(values))
            sds.append(statistics.stdev(values))
            axis.scatter(
                [halo_count] * len(values), values, color=colors[model], alpha=0.42,
                marker=markers[model], s=34, zorder=2,
            )
        axis.errorbar(
            HALO_COUNTS, means, yerr=sds, color=colors[model], marker=markers[model],
            linewidth=1.8, capsize=4, label=f"{model} mean ± sample SD", zorder=3,
        )
    axis.set_xticks(HALO_COUNTS)
    axis.set_xlabel("Number of halos/nodes per snapshot")
    axis.set_ylabel("Test MAE for $\\Omega_m$")
    axis.set_title("U1000 Train700 controlled halo-count scaling")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def trend_text(values: list[float]) -> str:
    deltas = [new - old for old, new in zip(values, values[1:])]
    if all(delta < 0 for delta in deltas):
        return "mean MAE decreases monotonically"
    if all(delta > 0 for delta in deltas):
        return "mean MAE increases monotonically"
    return "mean MAE is non-monotonic"


def write_reports(output: Path, summaries: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    report = [
        "# U1000 Train700 halo-count scaling",
        "",
        "This controlled analysis varies only the number of raw-Mvir-ranked halos per snapshot.",
        "Negative ΔMAE denotes improvement; positive ΔMAE denotes worsening.",
        "",
    ]
    for model in MODELS:
        model_rows = [row for row in summaries if row["model"] == model]
        values = [float(row["mean_mae"]) for row in model_rows]
        report.extend([f"## {model}", "", f"Across the four points, {trend_text(values)}.", ""])
        for row in [item for item in paired if item["model"] == model]:
            report.append(
                f"- {row['previous_halo_count']}→{row['new_halo_count']}: "
                f"mean ΔMAE {row['mean_delta_mae']:.8f} "
                f"({row['percent_change']:.3f}%), "
                f"{row['seeds_improving']}/3 seeds improve."
            )
        report.extend(
            [
                "",
                "Any diminishing-returns interpretation must compare adjacent changes with "
                "sample-SD and paired-seed variability; this report does not establish a formal knee, "
                "saturation point, or statistical significance.",
                "",
            ]
        )
    with (output / "scientific_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    validation = (
        "# Validation\n\n"
        "PASS: all 24 model × halo-count × seed cells are complete.\n\n"
        "PASS: every test MAE was recomputed from 201 ordered predictions and matched metrics.json.\n\n"
        "PASS: U1000, Train700/Val99/Test201, exact dataset identities, split manifests, and frozen "
        "model configurations were checked.\n\n"
        "PASS: sample standard deviation uses n-1 and ΔMAE is new minus previous.\n"
    )
    with (output / "validation.md").open("x", encoding="utf-8") as handle:
        handle.write(validation)


def run_analysis(root: Path) -> None:
    rows, missing = collect(root)
    require(not missing, f"analysis inputs are incomplete; missing {len(missing)} runs: {missing}")
    require(len(rows) == 24, f"expected 24 seed results; found {len(rows)}")
    output = root / OUTPUT_DIR
    require(not output.exists(), f"output path already exists; refusing overwrite: {OUTPUT_DIR}")
    output.mkdir(parents=True)
    seed_rows = [asdict(row) for row in rows]
    summaries = summary_rows(rows)
    paired = paired_rows(rows)
    plotting = [
        {
            "model": row.model,
            "halo_count": row.halo_count,
            "seed": row.seed,
            "test_mae": row.test_mae,
            "mean_mae": next(
                item["mean_mae"] for item in summaries
                if item["model"] == row.model and item["halo_count"] == row.halo_count
            ),
            "sample_sd": next(
                item["sample_sd"] for item in summaries
                if item["model"] == row.model and item["halo_count"] == row.halo_count
            ),
        }
        for row in rows
    ]
    write_csv(output / "seed_level_results.csv", seed_rows)
    write_csv(output / "halo_count_summary.csv", summaries)
    write_csv(output / "paired_seed_differences.csv", paired)
    write_csv(output / "plotting_data.csv", plotting)
    make_plot(output / "halo_count_scaling.png", rows)
    write_reports(output, summaries, paired)
    require({path.name for path in output.iterdir()} == set(EXPECTED_OUTPUTS), "unexpected analysis outputs")
    print(f"PASS: wrote controlled analysis to {OUTPUT_DIR}")


def check_only(root: Path) -> int:
    rows, missing = collect(root)
    expected_missing = {
        experiment_name(model, halo_count, seed)
        for model in MODELS for halo_count in (500,) for seed in SEEDS
    }
    print(f"Validated existing complete inputs: {len(rows)}/24")
    print(f"Missing expected future inputs: {len(missing)}/6")
    for name in missing:
        print(f"MISSING: {name}")
    require(set(missing) == expected_missing, "missing input set is not exactly future Top500 runs")
    require(len(rows) == 18, "existing Top750/Top1000/Top1500 reference coverage is not exactly 18 runs")
    print("CHECK PASS: Top750/Top1000/Top1500 references validate; exactly 6 future Top500 runs are absent.")
    print("No analysis directory or report was created.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate references and expected missing inputs without writing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = args.repo_root.resolve()
        return check_only(root) if args.check else (run_analysis(root) or 0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
