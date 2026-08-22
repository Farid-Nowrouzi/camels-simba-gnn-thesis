#!/usr/bin/env python3
"""Post-completion Top-N analysis; refuses to create scientific outputs before 36/36."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_COUNTS = (20, 50, 100, 200, 450, 700)
SEEDS = (42, 123, 2025)
MODEL_LABELS = {
    "EvolveGCNHRegressor": "EvolveGCN-H",
    "StaticGCNRegressor": "Static GCN",
}
DISPLAY_MODELS = tuple(MODEL_LABELS.values())
SEED_MARKERS = {42: "o", 123: "s", 2025: "^"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, values: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(values)


def matrix_index(
    values: list[dict[str, str]],
    *,
    label: str,
    model_field: str,
    count_field: str,
    model_aliases: dict[str, str],
) -> dict[tuple[str, int, int], dict[str, str]]:
    expected = {
        (model, count, seed)
        for model in DISPLAY_MODELS
        for count in TRAINING_COUNTS
        for seed in SEEDS
    }
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    duplicates: list[tuple[str, int, int]] = []
    unexpected: list[tuple[str, int, int]] = []
    for row in values:
        raw_model = row.get(model_field, "")
        model = model_aliases.get(raw_model, raw_model)
        try:
            key = (model, int(row[count_field]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{label}: malformed comparison key in row {row}") from exc
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = row
        if key not in expected:
            unexpected.append(key)
    missing = sorted(expected.difference(indexed))
    if len(values) != len(expected) or duplicates or unexpected or missing:
        raise RuntimeError(
            f"{label}: expected exactly {len(expected)} unique model x training-size x seed rows; "
            f"rows={len(values)}, duplicates={sorted(set(duplicates))}, "
            f"unexpected={sorted(set(unexpected))}, missing={missing}"
        )
    return indexed


def describe(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise RuntimeError("sample standard deviation requires at least two observations")
    return statistics.mean(values), statistics.stdev(values)


def mean_sd(values: list[float], digits: int = 6) -> str:
    mean, sample_sd = describe(values)
    return f"{mean:.{digits}f} ± {sample_sd:.{digits}f}"


def markdown_table(headers: tuple[str, ...], values: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in values)
    return "\n".join(lines)


def validate_split_matching() -> dict[int, list[str]]:
    test_ids_by_seed: dict[int, list[str]] = {}
    for seed in SEEDS:
        seed_test_ids: list[str] | None = None
        for count in TRAINING_COUNTS:
            top1000 = json.loads(
                (ROOT / f"configs/splits/u1000_top1000_none_k8_sparse/seed{seed}_train{count}.json").read_text(
                    encoding="utf-8"
                )
            )
            top1500 = json.loads(
                (ROOT / f"configs/splits/u1000_top1500_none_k8_sparse/seed{seed}_train{count}.json").read_text(
                    encoding="utf-8"
                )
            )
            for partition in ("train_ids", "val_ids", "test_ids", "unused_ids"):
                if top1000[partition] != top1500[partition]:
                    raise RuntimeError(
                        f"split mismatch for seed={seed}, training_count={count}, partition={partition}"
                    )
            current = top1000["test_ids"]
            if len(current) != 201 or len(set(current)) != 201:
                raise RuntimeError(f"invalid test IDs for seed={seed}, training_count={count}")
            if seed_test_ids is None:
                seed_test_ids = current
            elif current != seed_test_ids:
                raise RuntimeError(f"test IDs change across training sizes for seed={seed}")
        if seed_test_ids is None:
            raise RuntimeError(f"missing test IDs for seed={seed}")
        test_ids_by_seed[seed] = seed_test_ids
    return test_ids_by_seed


def paired_topn_summaries(comparison: list[dict]) -> list[dict]:
    summaries = []
    for model in DISPLAY_MODELS:
        for count in TRAINING_COUNTS:
            group = [
                row
                for row in comparison
                if row["model"] == model and int(row["training_count"]) == count
            ]
            by_seed = {int(row["seed"]): float(row["top1500_minus_top1000_mae"]) for row in group}
            if len(group) != len(SEEDS) or set(by_seed) != set(SEEDS):
                raise RuntimeError(f"incomplete paired Top-N group for model={model}, training_count={count}")
            differences = [by_seed[seed] for seed in SEEDS]
            mean, sample_sd = describe(differences)
            summaries.append(
                {
                    "model": model,
                    "training_count": count,
                    "seed_count": len(differences),
                    "seed42_difference": by_seed[42],
                    "seed123_difference": by_seed[123],
                    "seed2025_difference": by_seed[2025],
                    "mean_paired_difference": mean,
                    "sample_std_paired_difference": sample_sd,
                    "top1500_wins": sum(value < 0 for value in differences),
                    "top1000_wins": sum(value > 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return summaries


def scientific_report(
    top1500: dict[tuple[str, int, int], dict[str, str]],
    static_evolve: list[dict],
    topn_summaries: list[dict],
    test_ids_by_seed: dict[int, list[str]],
) -> str:
    static_wins = sum(float(row["static_minus_evolve_mae"]) < 0 for row in static_evolve)
    evolve_wins = sum(float(row["static_minus_evolve_mae"]) > 0 for row in static_evolve)
    model_ties = sum(float(row["static_minus_evolve_mae"]) == 0 for row in static_evolve)
    test_sets = {seed: set(ids) for seed, ids in test_ids_by_seed.items()}
    pair_overlaps = {
        (left, right): len(test_sets[left] & test_sets[right])
        for index, left in enumerate(SEEDS)
        for right in SEEDS[index + 1 :]
    }
    triple_overlap = len(set.intersection(*(test_sets[seed] for seed in SEEDS)))
    test_union = len(set.union(*(test_sets[seed] for seed in SEEDS)))

    model_rows = []
    for count in TRAINING_COUNTS:
        evolve = [float(top1500[("EvolveGCN-H", count, seed)]["test_mae"]) for seed in SEEDS]
        static = [float(top1500[("Static GCN", count, seed)]["test_mae"]) for seed in SEEDS]
        differences = [static_value - evolve_value for static_value, evolve_value in zip(static, evolve)]
        model_rows.append(
            (
                count,
                mean_sd(evolve),
                mean_sd(static),
                sum(value < 0 for value in differences),
                sum(value > 0 for value in differences),
                sum(value == 0 for value in differences),
            )
        )

    train700_rows = []
    metric_fields = (
        ("MAE", "test_mae"),
        ("RMSE", "test_rmse"),
        ("R²", "test_r2"),
        ("Prediction SD / target SD", "prediction_sd_ratio"),
    )
    for model in DISPLAY_MODELS:
        train700_rows.append(
            (
                model,
                *(
                    mean_sd([float(top1500[(model, 700, seed)][field]) for seed in SEEDS])
                    for _, field in metric_fields
                ),
            )
        )

    topn_rows = []
    for row in topn_summaries:
        topn_rows.append(
            (
                row["model"],
                row["training_count"],
                f'{row["seed42_difference"]:.6f}',
                f'{row["seed123_difference"]:.6f}',
                f'{row["seed2025_difference"]:.6f}',
                f'{row["mean_paired_difference"]:.6f} ± {row["sample_std_paired_difference"]:.6f}',
                row["top1500_wins"],
                row["top1000_wins"],
                row["ties"],
            )
        )

    behavior = []
    dispersion = []
    for model in DISPLAY_MODELS:
        mae_summaries = {
            count: describe([float(top1500[(model, count, seed)]["test_mae"]) for seed in SEEDS])
            for count in TRAINING_COUNTS
        }
        best_count = min(TRAINING_COUNTS, key=lambda count: mae_summaries[count][0])
        monotonic = all(
            mae_summaries[right][0] <= mae_summaries[left][0]
            for left, right in zip(TRAINING_COUNTS, TRAINING_COUNTS[1:])
        )
        most_variable = max(TRAINING_COUNTS, key=lambda count: mae_summaries[count][1])
        behavior.append(
            f"- {model}: mean MAE changes from {mae_summaries[20][0]:.6f} at Train20 to "
            f"{mae_summaries[700][0]:.6f} at Train700; the lowest mean MAE is at Train{best_count}. "
            f"The mean learning curve is {'monotonic' if monotonic else 'not monotonic'}, and the largest "
            f"seed-to-seed MAE SD occurs at Train{most_variable} ({mae_summaries[most_variable][1]:.6f})."
        )
        ratios = [float(top1500[(model, 700, seed)]["prediction_sd_ratio"]) for seed in SEEDS]
        ratio_mean, ratio_sd = describe(ratios)
        relation = "compressed relative to" if ratio_mean < 1 else "more dispersed than"
        dispersion.append(
            f"- {model}: Train700 prediction-SD / target-SD is {ratio_mean:.6f} ± {ratio_sd:.6f}; "
            f"on average the predictions are {relation} the target distribution."
        )

    topn_conclusion = []
    for model in DISPLAY_MODELS:
        group = [row for row in topn_summaries if row["model"] == model]
        top1500_wins = sum(int(row["top1500_wins"]) for row in group)
        top1000_wins = sum(int(row["top1000_wins"]) for row in group)
        ties = sum(int(row["ties"]) for row in group)
        topn_conclusion.append(
            f"- {model}: Top1500 has lower MAE in {top1500_wins}/18 paired cells, Top1000 in "
            f"{top1000_wins}/18, with {ties} ties."
        )

    return (
        "# U1000 Top1500 training-scaling scientific interpretation\n\n"
        "## Scope and integrity\n\n"
        "This report uses all 36/36 completed and artifact-validated Top1500 runs: EvolveGCN-H and "
        "Static GCN at training sizes 20, 50, 100, 200, 450, and 700 with seeds 42, 123, and 2025. "
        "For every seed and training size, the ordered train, validation, test, and unused IDs match "
        "exactly between Top1000 and Top1500.\n\n"
        "The test populations differ across seeds. Each seed has 201 unique test universes; the pairwise "
        f"overlaps are 42/123={pair_overlaps[(42, 123)]}, 42/2025={pair_overlaps[(42, 2025)]}, and "
        f"123/2025={pair_overlaps[(123, 2025)]}. The three-way overlap is {triple_overlap}, and the union "
        f"contains {test_union} universes. Cross-seed prediction rows are therefore analyzed as distinct "
        "seed-specific test evaluations, not repeated predictions for one common test population.\n\n"
        "## Static GCN versus EvolveGCN-H\n\n"
        f"Across the 18 training-size/seed matched comparisons, Static GCN has lower MAE in {static_wins}, "
        f"EvolveGCN-H in {evolve_wins}, with {model_ties} ties.\n\n"
        + markdown_table(
            ("Training universes", "Evolve MAE mean ± SD", "Static MAE mean ± SD", "Static wins", "Evolve wins", "Ties"),
            model_rows,
        )
        + "\n\nThis is a matched model-protocol comparison, not an architecture-only comparison: EvolveGCN-H uses "
        "all five snapshots, two graph layers, temporal mean pooling, a linear head, and batch size 4, "
        "whereas Static GCN uses the exact final snapshot, three graph layers, an MLP head, and batch size 8.\n\n"
        "## Train700 metrics\n\n"
        + markdown_table(("Model", *(name + " mean ± SD" for name, _ in metric_fields)), train700_rows)
        + "\n\n## Top1000 versus Top1500\n\n"
        "The paired difference is defined as **Top1500 MAE − Top1000 MAE**. Negative values favor "
        "Top1500; positive values favor Top1000.\n\n"
        + markdown_table(
            ("Model", "Train", "Seed 42", "Seed 123", "Seed 2025", "Mean ± SD", "Top1500 wins", "Top1000 wins", "Ties"),
            topn_rows,
        )
        + "\n\n"
        + "\n".join(topn_conclusion)
        + "\n\n## Training-size behaviour\n\n"
        + "\n".join(behavior)
        + "\n\n## Prediction dispersion\n\n"
        "Prediction-SD / target-SD near one indicates similar marginal dispersion; values below one indicate "
        "prediction compression, and values above one indicate overdispersion. This ratio is a dispersion "
        "diagnostic and does not establish calibration by itself.\n\n"
        + "\n".join(dispersion)
        + "\n\n## Conclusions and limitations\n\n"
        "The analysis is descriptive and uses only three seeds per cell, so the reported means and sample "
        "standard deviations should not be treated as inferential uncertainty. Test populations differ across "
        "seeds, while Top1000/Top1500 comparisons are exactly paired within each seed. Static GCN and "
        "EvolveGCN-H also use different input and model protocols. No significance, causal, or architecture-only "
        "claim is made.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=1500, choices=(1500,))
    parser.parse_args()
    registry_path = ROOT / "configs/experiment_registry/u1000_top1500_training_scaling_matrix.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry["entries"]
    if len(entries) != 36 or not all(
        item["status"] == "completed" and item["validation_result"] == "PASS" for item in entries
    ):
        raise SystemExit("REFUSED: scientific analysis requires 36/36 completed, validated Top1500 runs")
    subprocess.run(
        [
            str(ROOT / "envs/camels-gnn/bin/python"),
            "scripts/validation/manage_u1000_top1500_training_scaling_matrix.py",
            "--aggregate",
        ],
        cwd=ROOT,
        check=True,
    )
    report = ROOT / "reports/experiment_registry"
    per_run = rows(report / "u1000_top1500_final_per_run_metrics.csv")
    top1500 = matrix_index(
        per_run,
        label="Top1500 per-run table",
        model_field="model",
        count_field="training_universe_count",
        model_aliases=MODEL_LABELS,
    )
    anchor = rows(report / "u1000_top1000_final_per_run_metrics.csv")
    top1000 = matrix_index(
        anchor,
        label="Top1000 per-run table",
        model_field="model",
        count_field="training_count",
        model_aliases={model: model for model in DISPLAY_MODELS},
    )
    test_ids_by_seed = validate_split_matching()

    paired = []
    for count in TRAINING_COUNTS:
        for seed in SEEDS:
            evolve = top1500[("EvolveGCN-H", count, seed)]
            static = top1500[("Static GCN", count, seed)]
            paired.append(
                {
                    "training_count": count,
                    "seed": seed,
                    "evolve_mae": evolve["test_mae"],
                    "static_mae": static["test_mae"],
                    "static_minus_evolve_mae": float(static["test_mae"]) - float(evolve["test_mae"]),
                }
            )
    write(report / "u1000_top1500_static_vs_evolve_paired_comparison.csv", paired, list(paired[0]))

    comparison = []
    for row in per_run:
        model = MODEL_LABELS[row["model"]]
        count = int(row["training_universe_count"])
        seed = int(row["seed"])
        match = top1000[(model, count, seed)]
        comparison.append(
            {
                "model": model,
                "training_count": row["training_universe_count"],
                "seed": row["seed"],
                "top1000_mae": match["mae"],
                "top1500_mae": row["test_mae"],
                "top1500_minus_top1000_mae": float(row["test_mae"]) - float(match["mae"]),
            }
        )
    write(report / "u1000_top1000_vs_top1500_comparison.csv", comparison, list(comparison[0]))
    topn_summaries = paired_topn_summaries(comparison)
    write(
        report / "u1000_top1000_vs_top1500_paired_summary.csv",
        topn_summaries,
        list(topn_summaries[0]),
    )

    import matplotlib.pyplot as plt

    output = ROOT / "presentation_assets/u1000_top1500_training_scaling"
    output.mkdir(parents=True, exist_ok=True)
    for metric, filename in (
        ("test_mae", "01_mae_learning_curve.png"),
        ("test_rmse", "02_rmse_learning_curve.png"),
        ("test_r2", "03_r2_learning_curve.png"),
        ("prediction_sd_ratio", "04_prediction_sd_ratio.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in DISPLAY_MODELS:
            seed_values = {
                count: [float(top1500[(model, count, seed)][metric]) for seed in SEEDS]
                for count in TRAINING_COUNTS
            }
            means = [describe(seed_values[count])[0] for count in TRAINING_COUNTS]
            sample_sds = [describe(seed_values[count])[1] for count in TRAINING_COUNTS]
            plotted = ax.errorbar(
                TRAINING_COUNTS,
                means,
                yerr=sample_sds,
                marker="o",
                capsize=3,
                label=f"{model} mean ± sample SD",
            )
            color = plotted.lines[0].get_color()
            for seed in SEEDS:
                ax.scatter(
                    TRAINING_COUNTS,
                    [float(top1500[(model, count, seed)][metric]) for count in TRAINING_COUNTS],
                    marker=SEED_MARKERS[seed],
                    color=color,
                    s=24,
                    alpha=0.5,
                    label=f"seed {seed}" if model == DISPLAY_MODELS[0] else None,
                )
        ax.set(xlabel="Training universes", ylabel=metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    train700 = [item for item in entries if item["training_universe_count"] == 700]
    predictions: dict[int, dict[str, list[dict[str, object]]]] = {seed: {} for seed in SEEDS}
    for item in train700:
        model = MODEL_LABELS.get(item["model"])
        seed = int(item["seed"])
        if model is None or seed not in predictions or model in predictions[seed]:
            raise RuntimeError(f"unexpected or duplicate Train700 prediction entry: {item['experiment_name']}")
        values = rows(ROOT / item["experiment_directory"] / "predictions/test_predictions.csv")
        parsed = [
            {
                "universe_id": value["universe_id"],
                "true": float(value["true_omega_m"]),
                "pred": float(value["pred_omega_m"]),
            }
            for value in values
        ]
        if len(parsed) != 201 or len({value["universe_id"] for value in parsed}) != 201:
            raise RuntimeError(f"invalid Train700 predictions for model={model}, seed={seed}")
        predictions[seed][model] = parsed
    for seed in SEEDS:
        if set(predictions[seed]) != set(DISPLAY_MODELS):
            raise RuntimeError(f"incomplete Train700 model predictions for seed={seed}")
        evolve = predictions[seed]["EvolveGCN-H"]
        static = predictions[seed]["Static GCN"]
        if [value["universe_id"] for value in evolve] != [value["universe_id"] for value in static]:
            raise RuntimeError(f"Static/Evolve ordered universe IDs differ for seed={seed}")
        if [value["true"] for value in evolve] != [value["true"] for value in static]:
            raise RuntimeError(f"Static/Evolve true Omega_m values differ for seed={seed}")
        if [value["universe_id"] for value in evolve] != test_ids_by_seed[seed]:
            raise RuntimeError(f"Train700 prediction IDs do not match split manifest for seed={seed}")

    for kind, filename in (
        ("scatter", "05_train700_true_vs_predicted.png"),
        ("sorted", "06_train700_sorted_low_to_high.png"),
        ("residual", "07_train700_residual_vs_true.png"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
        for ax, seed in zip(axes, SEEDS):
            for model in DISPLAY_MODELS:
                group = predictions[seed][model]
                if kind == "scatter":
                    ax.scatter(
                        [value["true"] for value in group],
                        [value["pred"] for value in group],
                        s=10,
                        alpha=0.45,
                        label=model,
                    )
                elif kind == "residual":
                    ax.scatter(
                        [value["true"] for value in group],
                        [value["pred"] - value["true"] for value in group],
                        s=10,
                        alpha=0.45,
                        label=model,
                    )
            if kind == "scatter":
                true_values = [value["true"] for value in predictions[seed][DISPLAY_MODELS[0]]]
                limits = [min(true_values), max(true_values)]
                ax.plot(limits, limits, "k--", lw=1, label="Identity")
                ax.set(xlabel="True Omega_m", ylabel="Predicted Omega_m")
            elif kind == "sorted":
                reference = predictions[seed][DISPLAY_MODELS[0]]
                order = sorted(range(len(reference)), key=lambda index: reference[index]["true"])
                ax.plot(
                    range(len(order)),
                    [reference[index]["true"] for index in order],
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label="True Omega_m",
                )
                for model in DISPLAY_MODELS:
                    group = predictions[seed][model]
                    ax.plot(
                        range(len(order)),
                        [group[index]["pred"] for index in order],
                        alpha=0.75,
                        label=model,
                    )
                ax.set(xlabel="Test samples sorted by true Omega_m", ylabel="Omega_m")
            else:
                ax.axhline(0, color="black", ls="--", lw=1)
                ax.set(xlabel="True Omega_m", ylabel="Prediction residual")
            ax.set_title(f"Seed {seed}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in DISPLAY_MODELS:
        group = [row for row in topn_summaries if row["model"] == model]
        means = [float(row["mean_paired_difference"]) for row in group]
        sample_sds = [float(row["sample_std_paired_difference"]) for row in group]
        plotted = ax.errorbar(
            TRAINING_COUNTS,
            means,
            yerr=sample_sds,
            marker="o",
            capsize=3,
            label=f"{model} mean ± sample SD",
        )
        color = plotted.lines[0].get_color()
        for seed in SEEDS:
            ax.scatter(
                TRAINING_COUNTS,
                [float(row[f"seed{seed}_difference"]) for row in group],
                marker=SEED_MARKERS[seed],
                color=color,
                s=24,
                alpha=0.5,
                label=f"seed {seed}" if model == DISPLAY_MODELS[0] else None,
            )
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(xlabel="Training universes", ylabel="Top1500 MAE − Top1000 MAE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "08_top1000_vs_top1500_paired.png", dpi=180)
    plt.close(fig)

    interpretation = scientific_report(top1500, paired, topn_summaries, test_ids_by_seed)
    (report / "u1000_top1500_training_scaling_scientific_interpretation.md").write_text(
        interpretation, encoding="utf-8"
    )
    print("PASS: final Top1500 tables/comparisons and eight presentation figures generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
