#!/usr/bin/env python3
"""Complete the kNN-specific outputs after the generic report builder runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANALYSIS_NAME = "controlled_static_vs_evolvegcn_knn_ablation_500u_top100_h64"
MODELS = ("EvolveGCN-H", "Static GCN")
K_VALUES = (4, 6, 8, 12)
SEEDS = (42, 123, 2025)
COLORS = {"EvolveGCN-H": "#1f77b4", "Static GCN": "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines)


def save_figure(figure: Any, output: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(output / "figures" / f"{name}.{extension}", bbox_inches="tight")
    plt.close(figure)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seed_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = read_csv(output / "seed_level_results.csv")
    numeric = (
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_pearson",
        "target_mean",
        "prediction_mean",
        "target_std",
        "prediction_std",
        "prediction_std_ratio",
        "repeated_prediction_fraction",
    )
    for row in rows:
        row["k"] = int(row["k"])
        row["seed"] = int(row["seed"])
        for field in numeric:
            row[field] = float(row[field])
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["k"])].append(row)
    result: list[dict[str, Any]] = []
    for model in MODELS:
        for k in K_VALUES:
            group = sorted(grouped[(model, k)], key=lambda row: row["seed"])
            record: dict[str, Any] = {
                "model": model,
                "k": k,
                "seed_count": len(group),
                "seed_list": ";".join(str(row["seed"]) for row in group),
            }
            for metric in (
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_pearson",
                "prediction_std_ratio",
                "repeated_prediction_fraction",
            ):
                values = [row[metric] for row in group]
                record[f"mean_{metric}"] = statistics.mean(values)
                record[f"std_{metric}"] = statistics.stdev(values)
                if metric in ("test_mae", "test_rmse", "test_r2", "test_pearson"):
                    record[f"min_{metric}"] = min(values)
                    record[f"max_{metric}"] = max(values)
            result.append(record)
    return result


def pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["k"], row["seed"]): row for row in rows}
    result: list[dict[str, Any]] = []
    for k in K_VALUES:
        for seed in SEEDS:
            evolve = lookup[("EvolveGCN-H", k, seed)]
            static = lookup[("Static GCN", k, seed)]
            result.append(
                {
                    "k": k,
                    "seed": seed,
                    "static_experiment_name": static["experiment_name"],
                    "evolve_experiment_name": evolve["experiment_name"],
                    "static_minus_evolve_mae": static["test_mae"]
                    - evolve["test_mae"],
                    "static_minus_evolve_rmse": static["test_rmse"]
                    - evolve["test_rmse"],
                    "static_minus_evolve_r2": static["test_r2"]
                    - evolve["test_r2"],
                    "static_minus_evolve_pearson": static["test_pearson"]
                    - evolve["test_pearson"],
                    "static_minus_evolve_prediction_std_ratio": static[
                        "prediction_std_ratio"
                    ]
                    - evolve["prediction_std_ratio"],
                }
            )
    return result


def write_best_k(output: Path, aggregated: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        ranked = sorted(
            (row for row in aggregated if row["model"] == model),
            key=lambda row: row["mean_test_mae"],
        )
        rows.append(
            {
                "model": model,
                "descriptive_best_k": ranked[0]["k"],
                "mean_test_mae": ranked[0]["mean_test_mae"],
                "sample_sd_test_mae": ranked[0]["std_test_mae"],
                "runner_up_k": ranked[1]["k"],
                "mean_mae_difference_to_runner_up": ranked[1]["mean_test_mae"]
                - ranked[0]["mean_test_mae"],
                "interpretation": (
                    "Descriptive only; four k values and three seeds do not "
                    "establish a universal optimum."
                ),
            }
        )
    fields = list(rows[0])
    write_csv(output / "tables/best_k_summary.csv", rows, fields)
    text = (
        "# Descriptive Best-k Summary\n\n"
        "Selecting the smallest mean MAE from four tested k values and three "
        "seeds does not prove a universally optimal k.\n\n"
        + markdown_table(rows, fields)
        + "\n"
    )
    (output / "tables/best_k_summary.md").write_text(text, encoding="utf-8")


def plot_diagnostics(
    root: Path,
    output: Path,
    seed_rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    plot_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in seed_rows if row["model"] == model]
        for row in model_rows:
            axis.scatter(
                row["k"],
                row["test_mae"],
                color=COLORS[model],
                alpha=0.48,
                s=27,
            )
            plot_rows.append(
                {
                    "model": model,
                    "k": row["k"],
                    "seed": row["seed"],
                    "test_mae": row["test_mae"],
                    "point_type": "seed",
                    "sample_standard_deviation": "",
                }
            )
        model_aggregates = [row for row in aggregated if row["model"] == model]
        means = [row["mean_test_mae"] for row in model_aggregates]
        deviations = [row["std_test_mae"] for row in model_aggregates]
        axis.errorbar(
            K_VALUES,
            means,
            yerr=deviations,
            color=COLORS[model],
            marker="o",
            capsize=4,
            label=model,
        )
        for k, mean, deviation in zip(K_VALUES, means, deviations):
            plot_rows.append(
                {
                    "model": model,
                    "k": k,
                    "seed": "",
                    "test_mae": mean,
                    "point_type": "mean",
                    "sample_standard_deviation": deviation,
                }
            )
    axis.set_xlabel("Number of periodic-kNN neighbours (k)")
    axis.set_ylabel("Test MAE")
    axis.set_title("Test MAE versus k — 500U Top100 h64")
    axis.grid(alpha=0.22)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output, "test_mae_vs_k")
    write_csv(output / "plot_data/test_mae_vs_k.csv", plot_rows, list(plot_rows[0]))

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    plot_rows = []
    for model in MODELS:
        for row in [item for item in seed_rows if item["model"] == model]:
            axis.scatter(
                row["k"],
                row["repeated_prediction_fraction"],
                color=COLORS[model],
                alpha=0.48,
                s=27,
            )
            plot_rows.append(
                {
                    "model": model,
                    "k": row["k"],
                    "seed": row["seed"],
                    "repeated_prediction_fraction": row[
                        "repeated_prediction_fraction"
                    ],
                    "point_type": "seed",
                    "sample_standard_deviation": "",
                }
            )
        model_aggregates = [row for row in aggregated if row["model"] == model]
        means = [
            row["mean_repeated_prediction_fraction"] for row in model_aggregates
        ]
        deviations = [
            row["std_repeated_prediction_fraction"] for row in model_aggregates
        ]
        axis.errorbar(
            K_VALUES,
            means,
            yerr=deviations,
            color=COLORS[model],
            marker="o",
            capsize=4,
            label=model,
        )
        for k, mean, deviation in zip(K_VALUES, means, deviations):
            plot_rows.append(
                {
                    "model": model,
                    "k": k,
                    "seed": "",
                    "repeated_prediction_fraction": mean,
                    "point_type": "mean",
                    "sample_standard_deviation": deviation,
                }
            )
    axis.set_xlabel("Number of periodic-kNN neighbours (k)")
    axis.set_ylabel("Exact repeated-prediction fraction")
    axis.set_title("Repeated-prediction behaviour versus k — h64")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.22)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output, "repeated_prediction_fraction_vs_k")
    write_csv(
        output / "plot_data/repeated_prediction_fraction_vs_k.csv",
        plot_rows,
        list(plot_rows[0]),
    )

    representatives = {
        (row["model"], int(row["k"])): row
        for row in manifest["representative_runs"]
    }
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.4), sharex=True, sharey=True)
    distribution_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        for k_index, k in enumerate(K_VALUES):
            representative = representatives[(model, k)]
            prediction_path = (
                root
                / representative["experiment_path"]
                / "predictions/test_predictions.csv"
            )
            predictions = read_csv(prediction_path)
            targets = [float(row["true_omega_m"]) for row in predictions]
            values = [float(row["pred_omega_m"]) for row in predictions]
            bins = [0.10 + index * (0.41 / 20) for index in range(21)]
            axis = axes[model_index][k_index]
            axis.hist(
                targets,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.7,
                color="black",
                label="Truth",
            )
            axis.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.48,
                color=COLORS[model],
                label="Prediction",
            )
            axis.set_title(
                f"{model}\nk={k}, seed={representative['seed']}", fontsize=9
            )
            axis.grid(alpha=0.15)
            if k_index == 0:
                axis.set_ylabel("Density")
            if model_index == 1:
                axis.set_xlabel(r"$\Omega_m$")
            for series, series_values in (("truth", targets), ("prediction", values)):
                distribution_rows.extend(
                    {
                        "model": model,
                        "k": k,
                        "seed": representative["seed"],
                        "series": series,
                        "value": value,
                    }
                    for value in series_values
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.suptitle(
        "Representative true and predicted distributions — median-MAE seeds, h64",
        y=0.985,
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(figure, output, "prediction_distribution_representative_runs")
    write_csv(
        output / "plot_data/prediction_distribution_representative_runs.csv",
        distribution_rows,
        list(distribution_rows[0]),
    )


def write_summaries(
    output: Path,
    seed_rows: list[dict[str, Any]],
    aggregated: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    by_group = {(row["model"], row["k"]): row for row in aggregated}
    static_wins = sum(row["static_minus_evolve_mae"] < 0 for row in paired)
    evolve_wins = len(paired) - static_wins
    ranges = {
        model: max(by_group[(model, k)]["mean_test_mae"] for k in K_VALUES)
        - min(by_group[(model, k)]["mean_test_mae"] for k in K_VALUES)
        for model in MODELS
    }
    seed_sds = {
        model: [by_group[(model, k)]["std_test_mae"] for k in K_VALUES]
        for model in MODELS
    }
    ratio_min = min(row["prediction_std_ratio"] for row in seed_rows)
    ratio_max = max(row["prediction_std_ratio"] for row in seed_rows)
    repeat_max = max(row["repeated_prediction_fraction"] for row in seed_rows)
    scientific = f"""# Scientific Summary: Controlled kNN Connectivity Ablation at h64

## Scope and question

How does changing periodic-kNN connectivity across k = 4, 6, 8, and 12 affect Omega_m regression for Static GCN and EvolveGCN-H at 500 universes, Top100 halos, minmax normalization, and hidden dimension 64?

The comparison is **mostly controlled**: population, graph protocol, k values, hidden dimension, seeds, and exact split IDs match. Temporal versus final-snapshot input, architecture, layer count, batch size, regression head, and temporal pooling intentionally differ. It is not a pure architecture-only ablation.

## Answers to the scientific questions

1. **Does mean Test MAE change materially across k?** The full mean-MAE range is {ranges['EvolveGCN-H']:.6f} for EvolveGCN-H and {ranges['Static GCN']:.6f} for Static GCN. These shifts are much smaller than the between-seed sample SDs ({min(seed_sds['EvolveGCN-H']):.6f}–{max(seed_sds['EvolveGCN-H']):.6f} and {min(seed_sds['Static GCN']):.6f}–{max(seed_sds['Static GCN']):.6f}, respectively), so the evidence does not support a material connectivity effect at this resolution.
2. **Is there a consistent best k across seeds?** No. Seed-level rankings change, and the mean curves are non-monotonic.
3. **Do both models favour the same k descriptively?** Yes: k=8 has the smallest mean MAE for both ({by_group[('EvolveGCN-H', 8)]['mean_test_mae']:.6f} and {by_group[('Static GCN', 8)]['mean_test_mae']:.6f}), but its advantage is tiny relative to seed variability and is not evidence of a universal optimum.
4. **Which model wins paired comparisons?** Static GCN has lower MAE in {static_wins}/12 matched k/seed pairs and EvolveGCN-H in {evolve_wins}/12. This descriptive split does not identify a causal architecture or temporal-input effect.
5. **How large is seed variability?** It dominates the differences among k means for both models.
6. **Does increasing k reduce prediction compression?** Not consistently. Prediction-SD ratios remain far below 1 and do not improve monotonically with k.
7. **Does increasing k reduce repeated predictions?** Not consistently. EvolveGCN-H has no exact repeats in these files, whereas Static GCN shows seed- and k-dependent repetition, including severe cases at k=4 and k=8 for seed 42.
8. **Does denser connectivity solve regression toward the mean?** No. Across the canonical rows, prediction-SD/target-SD spans {ratio_min:.3f}–{ratio_max:.3f}, remaining substantially below 1.
9. **Are apparent best-k results robust?** No; the differences are small and seed-dependent.

## Representative runs

Representative plots use the median-MAE seed within every model/k group, never the best seed automatically.

## Caveats

- Selecting the smallest mean MAE from four tested k values and three seeds does not prove a universally optimal k.
- Poor dispersion and repeated predictions identify behaviour, not its mechanism.
- Temporal processing cannot be claimed as the cause of cross-model differences because several model-specific protocol fields differ intentionally.
- This experiment tests kNN connectivity at hidden dimension 64 and does not automatically generalize to h32.
"""
    (output / "scientific_summary.md").write_text(scientific, encoding="utf-8")
    presentation = f"""# Presentation-ready Summary

## Experiment question

How does periodic-kNN connectivity affect Omega_m regression under the completed 500U Top100 h64 protocols?

## One-sentence protocol

Compare Static GCN and EvolveGCN-H at k = 4, 6, 8, and 12 over seeds 42, 123, and 2025 with matched population, preprocessing, graph construction, hidden dimension, and exact splits.

## Four numerical findings

- Both models have their smallest descriptive mean MAE at k=8: EvolveGCN-H {by_group[('EvolveGCN-H', 8)]['mean_test_mae']:.6f}; Static GCN {by_group[('Static GCN', 8)]['mean_test_mae']:.6f}.
- The mean-MAE range across k is only {ranges['EvolveGCN-H']:.6f} for EvolveGCN-H and {ranges['Static GCN']:.6f} for Static GCN.
- Static GCN wins {static_wins}/12 paired MAE comparisons; EvolveGCN-H wins {evolve_wins}/12.
- Prediction-SD ratios span {ratio_min:.3f}–{ratio_max:.3f}; the largest exact-repeat fraction is {repeat_max:.3f}.

## Recommended outputs

- **Main table:** `tables/main_results_table.md`, with mean ± sample SD by model and k.
- **Main figure:** `figures/test_mae_vs_k.png`, showing seed points and mean ± sample SD.
- **Diagnostic figure:** `figures/repeated_prediction_fraction_vs_k.png`, paired with `figures/prediction_std_ratio_vs_k.png`.

## Presentation-safe conclusions

- Connectivity effects are small relative to seed variability under this h64 protocol.
- k=8 is the descriptive minimum-mean-MAE setting for both models, but the result is not robust enough to claim a universal optimum.

## Important caveats

- Cross-model comparisons are mostly controlled, not architecture-only, because temporal input and several model-specific settings differ.
- Prediction compression and exact repeats are empirical diagnostics; their cause is not established.

## Suggested captions

- **Main figure:** “Test MAE across periodic-kNN connectivity at 500 universes, Top100 halos, and h64. Points are seeds; curves show mean ± sample SD.”
- **Diagnostic figure:** “Exact repeated-prediction fraction across k. Severe Static GCN repetition is seed-dependent and is not resolved monotonically by denser connectivity.”
"""
    (output / "presentation_ready_summary.md").write_text(
        presentation, encoding="utf-8"
    )


def update_manifest(
    root: Path,
    output: Path,
    seed_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    matrix = read_csv(
        root / "reports/experiment_registry/knn_ablation_candidate_matrix.csv"
    )
    excluded: list[dict[str, str]] = []
    for row in matrix:
        if row["hidden_dim"] != "32":
            continue
        reason = "Excluded: hidden_dim=32; the official analysis is restricted to h64."
        if row["compatibility"] == "incompatible_h32_batch4_alternative":
            reason = (
                "Excluded: historical Static h32 batch-size-4 alternative; "
                "incompatible with the official h64 protocol."
            )
        excluded.append(
            {
                "experiment_path": row["experiment_path"],
                "experiment_name": row["experiment_name"],
                "reason": reason,
            }
        )
    family_paths = (
        "configs/experiment_families/"
        "canonical_evolvegcn_knn_ablation_500u_top100_h64.json",
        "configs/experiment_families/"
        "canonical_static_gcn_knn_ablation_500u_top100_h64.json",
    )
    manifest["excluded_paths"] = excluded
    manifest["family_specification_hashes_sha256"] = {
        path: sha256(root / path) for path in family_paths
    }
    manifest["prediction_file_mappings"] = [
        {
            "experiment_path": row["experiment_path"],
            "prediction_file": (
                f"{row['experiment_path']}/predictions/test_predictions.csv"
            ),
        }
        for row in seed_rows
    ]
    manifest["generated_figures"] = [
        "test_mae_vs_k",
        "test_rmse_vs_k",
        "test_r2_vs_k",
        "paired_mae_difference_vs_k",
        "seed_variability_vs_k",
        "prediction_std_ratio_vs_k",
        "repeated_prediction_fraction_vs_k",
        "true_vs_predicted_representative_runs",
        "residuals_vs_true_representative_runs",
        "prediction_distribution_representative_runs",
    ]
    script = output / "rebuild_analysis_specific_outputs.py"
    manifest.setdefault("source_file_hashes_sha256", {})[
        str(script.relative_to(root))
    ] = sha256(script)
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else root / args.spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("analysis_name") != ANALYSIS_NAME:
        raise SystemExit(f"unexpected analysis specification: {spec_path}")
    output = root / spec["output_directory"]
    seed_rows = load_seed_rows(output)
    if len(seed_rows) != 24:
        raise SystemExit(f"expected 24 seed rows, found {len(seed_rows)}")
    aggregated = aggregate(seed_rows)
    paired = pair(seed_rows)
    aggregate_fields = ["model", "k", "seed_count", "seed_list"]
    for metric in ("test_mae", "test_rmse", "test_r2", "test_pearson"):
        aggregate_fields.extend(
            [
                f"mean_{metric}",
                f"std_{metric}",
                f"min_{metric}",
                f"max_{metric}",
            ]
        )
    aggregate_fields.extend(
        [
            "mean_prediction_std_ratio",
            "std_prediction_std_ratio",
            "mean_repeated_prediction_fraction",
            "std_repeated_prediction_fraction",
        ]
    )
    write_csv(output / "aggregated_results.csv", aggregated, aggregate_fields)
    write_csv(output / "paired_model_differences.csv", paired, list(paired[0]))
    write_best_k(output, aggregated)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    plot_diagnostics(root, output, seed_rows, aggregated, manifest)
    write_summaries(output, seed_rows, aggregated, paired)
    update_manifest(root, output, seed_rows, manifest)
    print(f"Completed analysis-specific outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
