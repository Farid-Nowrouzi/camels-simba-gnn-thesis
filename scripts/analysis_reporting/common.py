"""Reusable validation and reporting for verified experiment-family artifacts.

This module deliberately operates only on JSON and CSV artifacts. It never
loads datasets, checkpoints, notebooks, or training code.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment_pipeline.common import (
    PipelineError,
    deterministic_runs,
    format_markdown_table,
    load_family_spec,
    read_json,
    read_prediction_pairs,
    resolve_repo_path,
    sample_standard_deviation,
    values_equal,
    verify_family,
    write_csv,
    write_text,
)


ANALYSIS_TYPES = {
    "single_family",
    "paired_family",
    "multi_family",
    "ablation",
    "scaling",
}
REPRESENTATIVE_POLICIES = {
    "median_test_mae",
    "explicit_seed",
    "best_test_mae",
    "worst_test_mae",
    "all_seeds",
}
SUPPORTED_METRICS = {
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
}
SEED_COLUMNS = (
    "model",
    "seed",
    "experiment_name",
    "experiment_path",
    "split_signature",
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
    "best_epoch",
)
AGGREGATE_COLUMNS = (
    "model",
    "seed_count",
    "seed_list",
    "mean_test_mae",
    "std_test_mae",
    "min_test_mae",
    "max_test_mae",
    "mean_test_rmse",
    "std_test_rmse",
    "min_test_rmse",
    "max_test_rmse",
    "mean_test_r2",
    "std_test_r2",
    "min_test_r2",
    "max_test_r2",
    "mean_test_pearson",
    "std_test_pearson",
    "min_test_pearson",
    "max_test_pearson",
)
PAIRED_COLUMNS = (
    "seed",
    "static_experiment_name",
    "evolve_experiment_name",
    "static_minus_evolve_mae",
    "static_minus_evolve_rmse",
    "static_minus_evolve_r2",
    "static_minus_evolve_pearson",
)


@dataclass(frozen=True)
class AnalysisValidation:
    """Validated inputs and raw-derived family rows."""

    spec: dict[str, Any]
    spec_path: Path
    family_specs: tuple[tuple[str, Path, dict[str, Any]], ...]
    family_rows: tuple[tuple[str, tuple[dict[str, Any], ...]], ...]
    protocol_rows: tuple[dict[str, Any], ...]
    intentional_differences: tuple[dict[str, str], ...]
    errors: tuple[str, ...]
    excluded_paths: tuple[dict[str, str], ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _require(mapping: Mapping[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        raise PipelineError(f"analysis specification field {key!r} must be {kind.__name__}")
    return value


def load_analysis_spec(path: Path) -> dict[str, Any]:
    """Load and structurally validate one analysis specification."""

    spec = read_json(path)
    if spec.get("schema_version") != "1.0":
        raise PipelineError("analysis schema_version must be '1.0'")
    for key in (
        "analysis_name",
        "title",
        "scientific_question",
        "analysis_type",
        "grouping_field",
        "output_directory",
        "comparison_statement",
    ):
        _require(spec, key, str)
    for optional_text in (
        "grouping_axis_label",
        "grouping_value_label",
        "plot_grouping_suffix",
        "controlled_variables_summary",
        "limitations_summary",
    ):
        if optional_text in spec:
            _require(spec, optional_text, str)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", spec["analysis_name"]):
        raise PipelineError("analysis_name must contain only lowercase letters, digits, '_' or '-'")
    if spec["analysis_type"] not in ANALYSIS_TYPES:
        raise PipelineError(f"unsupported analysis_type: {spec['analysis_type']!r}")
    families = _require(spec, "families", list)
    if not families:
        raise PipelineError("families must contain at least one family")
    labels: set[str] = set()
    paths: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise PipelineError(f"families[{index}] must be an object")
        label = _require(family, "label", str)
        family_path = _require(family, "family_spec_path", str)
        if label in labels:
            raise PipelineError(f"duplicate family label: {label}")
        if family_path in paths:
            raise PipelineError(f"duplicate family specification: {family_path}")
        labels.add(label)
        paths.add(family_path)
    if spec["analysis_type"] == "single_family" and len(families) != 1:
        raise PipelineError("single_family analysis requires exactly one family")
    if spec["analysis_type"] == "paired_family" and len(families) != 2:
        raise PipelineError("paired_family analysis requires exactly two families")
    groups = _require(spec, "grouping_values", list)
    seeds = _require(spec, "required_seeds", list)
    metrics = _require(spec, "metrics", list)
    if not groups or len({json.dumps(v, sort_keys=True) for v in groups}) != len(groups):
        raise PipelineError("grouping_values must be non-empty and unique")
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise PipelineError("required_seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise PipelineError("required_seeds contains duplicates")
    if not metrics or not all(metric in SUPPORTED_METRICS for metric in metrics):
        raise PipelineError(f"metrics must be selected from {sorted(SUPPORTED_METRICS)}")
    _require(spec, "required_fixed_protocol", dict)
    allowed = _require(spec, "fields_allowed_to_differ", list)
    allowed_names: set[str] = set()
    for index, item in enumerate(allowed):
        if not isinstance(item, dict):
            raise PipelineError(f"fields_allowed_to_differ[{index}] must be an object")
        field = _require(item, "field", str)
        _require(item, "reason", str)
        if field in allowed_names:
            raise PipelineError(f"duplicate allowed-difference field: {field}")
        allowed_names.add(field)
    if not isinstance(spec.get("prediction_plots_enabled"), bool):
        raise PipelineError("prediction_plots_enabled must be boolean")
    formats = _require(spec, "figure_formats", list)
    if not formats or any(value not in {"png", "pdf"} for value in formats):
        raise PipelineError("figure_formats supports only 'png' and 'pdf'")
    representative = _require(spec, "representative_seed_policy", dict)
    policy = representative.get("policy")
    if policy not in REPRESENTATIVE_POLICIES:
        raise PipelineError(f"unsupported representative-seed policy: {policy!r}")
    if policy == "explicit_seed" and not isinstance(representative.get("seed"), int):
        raise PipelineError("explicit_seed policy requires an integer seed")
    tolerance = spec.get("metric_tolerance", 1e-6)
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise PipelineError("metric_tolerance must be positive")
    return spec


def _dotted_value(mapping: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def validate_analysis(repo_root: Path, spec_path: Path) -> AnalysisValidation:
    """Validate all families and cross-family compatibility without writing."""

    root = repo_root.expanduser().resolve()
    resolved_spec = resolve_repo_path(root, spec_path)
    spec = load_analysis_spec(resolved_spec)
    errors: list[str] = []
    family_specs: list[tuple[str, Path, dict[str, Any]]] = []
    family_rows: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    excluded: list[dict[str, str]] = []
    experiment_paths: dict[str, str] = {}
    expected_pairs = {
        (json.dumps(group, sort_keys=True), int(seed))
        for group in spec["grouping_values"]
        for seed in spec["required_seeds"]
    }

    for entry in spec["families"]:
        label = entry["label"]
        path = resolve_repo_path(root, entry["family_spec_path"])
        family = load_family_spec(path)
        family_specs.append((label, path, family))
        if family["grouping_field"] != spec["grouping_field"]:
            errors.append(
                f"{label}: grouping_field={family['grouping_field']!r}, "
                f"expected {spec['grouping_field']!r}"
            )
        if family["grouping_values"] != spec["grouping_values"]:
            errors.append(
                f"{label}: grouping_values={family['grouping_values']!r}, "
                f"expected {spec['grouping_values']!r}"
            )
        if family["required_seeds"] != spec["required_seeds"]:
            errors.append(
                f"{label}: required_seeds={family['required_seeds']!r}, "
                f"expected {spec['required_seeds']!r}"
            )
        verification = verify_family(root, family, allow_incomplete=False)
        if not verification.valid or not verification.complete:
            errors.extend(f"{label}: {message}" for message in verification.errors)
        rows = tuple(verification.rows)
        family_rows.append((label, rows))
        observed_pairs = {
            (json.dumps(row["grouping_value"], sort_keys=True), int(row["seed"]))
            for row in rows
        }
        if observed_pairs != expected_pairs:
            errors.append(
                f"{label}: required row coverage mismatch; "
                f"missing={sorted(expected_pairs - observed_pairs)}"
            )
        for run in deterministic_runs(family):
            if run["action"] == "exclude":
                excluded.append(
                    {
                        "family": label,
                        "experiment_path": str(run["experiment_path"]),
                        "reason": str(run.get("exclusion_reason", "excluded by family")),
                    }
                )
                continue
            path_key = str(run["experiment_path"])
            previous = experiment_paths.get(path_key)
            if previous is not None:
                errors.append(
                    f"experiment path counted more than once: {path_key} "
                    f"({previous} and {label})"
                )
            experiment_paths[path_key] = label

    tolerance = float(spec.get("metric_tolerance", 1e-6))
    for field, expected in spec["required_fixed_protocol"].items():
        for label, _, family in family_specs:
            present, actual = _dotted_value(family, field)
            if not present:
                errors.append(f"{label}: required fixed protocol field is missing: {field}")
            elif not values_equal(actual, expected, min(tolerance, 1e-12)):
                errors.append(
                    f"{label}: {field}={actual!r}, expected {expected!r}"
                )

    allowed = {
        item["field"]: item["reason"] for item in spec["fields_allowed_to_differ"]
    }
    flattened: dict[str, dict[str, Any]] = {}
    for label, _, family in family_specs:
        selected = {
            "fixed_scientific_settings.config": family["fixed_scientific_settings"].get(
                "config", {}
            ),
            "fixed_scientific_settings.dataset_metadata": family[
                "fixed_scientific_settings"
            ].get("dataset_metadata", {}),
        }
        flat: dict[str, Any] = {}
        for prefix, values in selected.items():
            flat.update(_flatten(values, prefix))
        flattened[label] = flat
    all_fields = sorted({field for values in flattened.values() for field in values})
    intentional: list[dict[str, str]] = []
    protocol_rows: list[dict[str, Any]] = []
    labels = [label for label, _, _ in family_specs]
    for field in all_fields:
        encoded_values = {
            json.dumps(flattened[label].get(field, "<missing>"), sort_keys=True)
            for label in labels
        }
        differs = len(encoded_values) > 1
        if differs and field not in allowed:
            rendered = ", ".join(
                f"{label}={flattened[label].get(field, '<missing>')!r}"
                for label in labels
            )
            errors.append(f"unexpected cross-family protocol difference: {field}: {rendered}")
        if differs and field in allowed:
            intentional.append({"field": field, "reason": allowed[field]})
        row: dict[str, Any] = {
            "field": field,
            "required_value": spec["required_fixed_protocol"].get(field, ""),
            "status": "intentional_difference" if differs else "matched",
        }
        for label in labels:
            row[label] = flattened[label].get(field, "<not applicable>")
        protocol_rows.append(row)

    if len(family_rows) > 1:
        signatures: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        for label, rows in family_rows:
            for row in rows:
                key = (json.dumps(row["grouping_value"], sort_keys=True), int(row["seed"]))
                signatures[key][label] = str(row["split_signature"])
        for key in sorted(signatures):
            values = signatures[key]
            if len(values) != len(family_rows):
                errors.append(f"split comparison is missing a family for {key}")
            elif len(set(values.values())) != 1:
                errors.append(f"split signature mismatch for group/seed {key}: {values}")

    return AnalysisValidation(
        spec=spec,
        spec_path=resolved_spec,
        family_specs=tuple(family_specs),
        family_rows=tuple(family_rows),
        protocol_rows=tuple(protocol_rows),
        intentional_differences=tuple(intentional),
        errors=tuple(errors),
        excluded_paths=tuple(excluded),
    )


def _prediction_path(
    repo_root: Path,
    family: Mapping[str, Any],
    experiment_path: str,
) -> Path:
    return resolve_repo_path(repo_root, experiment_path) / str(family["prediction_file"])


def _repeated_prediction_fraction(predictions: Sequence[float]) -> float:
    if not predictions:
        return float("nan")
    return (len(predictions) - len(set(predictions))) / len(predictions)


def build_seed_rows(
    repo_root: Path,
    validation: AnalysisValidation,
) -> tuple[list[dict[str, Any]], dict[tuple[str, Any, int], tuple[list[float], list[float]]]]:
    """Convert verified family rows to analysis rows and cache prediction pairs."""

    families = {label: family for label, _, family in validation.family_specs}
    prediction_pairs: dict[
        tuple[str, Any, int], tuple[list[float], list[float]]
    ] = {}
    result: list[dict[str, Any]] = []
    group_order = {
        json.dumps(value, sort_keys=True): index
        for index, value in enumerate(validation.spec["grouping_values"])
    }
    label_order = {
        entry["label"]: index for index, entry in enumerate(validation.spec["families"])
    }
    grouping_field = validation.spec["grouping_field"]
    for label, rows in validation.family_rows:
        family = families[label]
        for verified in rows:
            targets, predictions = read_prediction_pairs(
                _prediction_path(repo_root, family, verified["experiment_path"]),
                family["target_column_aliases"],
                family["prediction_column_aliases"],
            )
            key = (label, verified["grouping_value"], int(verified["seed"]))
            prediction_pairs[key] = (targets, predictions)
            result.append(
                {
                    "model": label,
                    grouping_field: verified["grouping_value"],
                    "seed": int(verified["seed"]),
                    "experiment_name": verified["experiment_name"],
                    "experiment_path": verified["experiment_path"],
                    "split_signature": verified["split_signature"],
                    "test_mae": verified["test_mae"],
                    "test_rmse": verified["test_rmse"],
                    "test_r2": verified["test_r2"],
                    "test_pearson": verified["test_pearson"],
                    "target_mean": verified["target_mean"],
                    "prediction_mean": verified["prediction_mean"],
                    "target_std": verified["target_std"],
                    "prediction_std": verified["prediction_std"],
                    "prediction_std_ratio": verified["prediction_std_ratio"],
                    "repeated_prediction_fraction": _repeated_prediction_fraction(
                        predictions
                    ),
                    "best_epoch": verified["best_epoch"],
                }
            )
    result.sort(
        key=lambda row: (
            label_order[row["model"]],
            group_order[json.dumps(row[grouping_field], sort_keys=True)],
            row["seed"],
        )
    )
    return result, prediction_pairs


def aggregate_seed_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
    grouping_values: Sequence[Any],
    grouping_field: str,
) -> list[dict[str, Any]]:
    """Aggregate each metric using sample standard deviation across seeds."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    originals: dict[str, Any] = {}
    for row in seed_rows:
        key = str(row[grouping_field])
        originals[key] = row[grouping_field]
        grouped[(str(row["model"]), key)].append(row)
    result: list[dict[str, Any]] = []
    for model in model_order:
        for group in grouping_values:
            rows = sorted(grouped[(model, str(group))], key=lambda row: int(row["seed"]))
            if not rows:
                continue
            record: dict[str, Any] = {
                "model": model,
                grouping_field: originals[str(group)],
                "seed_count": len(rows),
                "seed_list": ";".join(str(row["seed"]) for row in rows),
            }
            for metric in ("test_mae", "test_rmse", "test_r2", "test_pearson"):
                values = [float(row[metric]) for row in rows]
                record[f"mean_{metric}"] = statistics.mean(values)
                record[f"std_{metric}"] = sample_standard_deviation(values)
                record[f"min_{metric}"] = min(values)
                record[f"max_{metric}"] = max(values)
            result.append(record)
    return result


def pair_model_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    grouping_field: str,
) -> list[dict[str, Any]]:
    """Pair Static and Evolve rows by grouping value and seed."""

    static_rows = {
        (row[grouping_field], int(row["seed"])): row
        for row in seed_rows
        if "static" in str(row["model"]).lower()
    }
    evolve_rows = {
        (row[grouping_field], int(row["seed"])): row
        for row in seed_rows
        if "evolve" in str(row["model"]).lower()
    }
    if not static_rows or not evolve_rows:
        return []
    if set(static_rows) != set(evolve_rows):
        raise PipelineError("Static/Evolve pairing keys do not match")
    result: list[dict[str, Any]] = []
    for key in sorted(static_rows, key=lambda item: (float(item[0]), item[1])):
        static = static_rows[key]
        evolve = evolve_rows[key]
        result.append(
            {
                grouping_field: key[0],
                "seed": key[1],
                "static_experiment_name": static["experiment_name"],
                "evolve_experiment_name": evolve["experiment_name"],
                "static_minus_evolve_mae": float(static["test_mae"])
                - float(evolve["test_mae"]),
                "static_minus_evolve_rmse": float(static["test_rmse"])
                - float(evolve["test_rmse"]),
                "static_minus_evolve_r2": float(static["test_r2"])
                - float(evolve["test_r2"]),
                "static_minus_evolve_pearson": float(static["test_pearson"])
                - float(evolve["test_pearson"]),
            }
        )
    return result


def select_representative_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    policy_spec: Mapping[str, Any],
    grouping_field: str,
) -> list[dict[str, Any]]:
    """Select representative rows without automatically preferring the best seed."""

    grouped: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["model"]), row[grouping_field])].append(row)
    policy = policy_spec["policy"]
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], float(item[1]))):
        rows = sorted(grouped[key], key=lambda row: (float(row["test_mae"]), int(row["seed"])))
        if policy == "all_seeds":
            chosen = rows
        elif policy == "best_test_mae":
            chosen = [rows[0]]
        elif policy == "worst_test_mae":
            chosen = [rows[-1]]
        elif policy == "explicit_seed":
            chosen = [row for row in rows if int(row["seed"]) == int(policy_spec["seed"])]
            if not chosen:
                raise PipelineError(f"explicit representative seed is unavailable for {key}")
        else:
            chosen = [rows[len(rows) // 2]]
        selected.extend(dict(row) for row in chosen)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "dirty": None}


def build_manifest(
    repo_root: Path,
    validation: AnalysisValidation,
    seed_rows: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record complete lightweight provenance for one generated package."""

    source_paths: set[Path] = {
        validation.spec_path,
        repo_root / "scripts/analysis_reporting/common.py",
        repo_root / "scripts/build_analysis_report.py",
        repo_root / "scripts/experiment_pipeline/common.py",
    }
    experiment_paths: list[str] = []
    family_by_label = {
        label: (path, family) for label, path, family in validation.family_specs
    }
    for label, rows in validation.family_rows:
        family_path, family = family_by_label[label]
        source_paths.add(family_path)
        for row in rows:
            experiment_paths.append(str(row["experiment_path"]))
            experiment_dir = resolve_repo_path(repo_root, row["experiment_path"])
            source_paths.update(
                {
                    experiment_dir / "config.json",
                    experiment_dir / "metrics.json",
                    experiment_dir / str(family["prediction_file"]),
                }
            )
    hashes = {
        path.resolve().relative_to(repo_root.resolve()).as_posix(): _sha256(path)
        for path in sorted(source_paths)
    }
    return {
        "analysis_specification": validation.spec,
        "generation_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": _git_state(repo_root),
        "source_file_hashes_sha256": hashes,
        "experiment_paths_used": experiment_paths,
        "excluded_paths": list(validation.excluded_paths),
        "input_experiment_count": len(seed_rows),
        "representative_seed_policy": validation.spec["representative_seed_policy"],
        "representative_runs": [
            {
                "model": row["model"],
                validation.spec["grouping_field"]: row[
                    validation.spec["grouping_field"]
                ],
                "seed": row["seed"],
                "experiment_name": row["experiment_name"],
                "experiment_path": row["experiment_path"],
            }
            for row in representatives
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_protocol_outputs(
    output_dir: Path,
    validation: AnalysisValidation,
) -> None:
    labels = [entry["label"] for entry in validation.spec["families"]]
    columns = ("field", "required_value", *labels, "status")
    rows = [
        {key: _format_value(row.get(key, "")) for key in columns}
        for row in validation.protocol_rows
    ]
    write_csv(output_dir / "protocol_table.csv", rows, columns)
    write_text(output_dir / "protocol_table.md", format_markdown_table(rows, columns))


def write_compatibility_report(
    output_dir: Path,
    validation: AnalysisValidation,
) -> None:
    required_rows = [
        {"field": field, "required_value": _format_value(value)}
        for field, value in validation.spec["required_fixed_protocol"].items()
    ]
    matched = [row for row in validation.protocol_rows if row["status"] == "matched"]
    parts = [
        f"# Compatibility Report: {validation.spec['title']}",
        "",
        f"Verdict: **{'PASS' if validation.valid else 'FAIL'}**",
        "",
        validation.spec["comparison_statement"],
        "",
        "## Required settings",
        "",
        format_markdown_table(required_rows, ("field", "required_value")),
        "",
        "## Matched settings",
        "",
        format_markdown_table(matched, ("field", "status")),
        "",
        "## Intentional differences",
        "",
    ]
    if validation.intentional_differences:
        parts.append(
            format_markdown_table(
                validation.intentional_differences, ("field", "reason")
            )
        )
    else:
        parts.append("None.")
    parts.extend(
        [
            "",
            "## Unexpected differences",
            "",
            *([f"- {error}" for error in validation.errors] or ["None."]),
            "",
            "## Split compatibility",
            "",
            (
                "PASS: ordered train/validation/test split signatures match for every "
                "model, grouping value, and seed."
                if validation.valid and len(validation.family_specs) > 1
                else "Not applicable for a single-family analysis."
            ),
            "",
            "Raw saved MAE/RMSE/MSE and prediction-derived metrics were checked by "
            "the existing family verifier at the configured numerical tolerance.",
        ]
    )
    write_text(output_dir / "compatibility_report.md", "\n".join(parts))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def write_main_tables(
    output_dir: Path,
    aggregated: Sequence[Mapping[str, Any]],
    grouping_field: str,
    grouping_label: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in aggregated:
        rows.append(
            {
                "model": row["model"],
                grouping_field: row[grouping_field],
                "seeds": row["seed_list"],
                "test_mae_mean_sd": f"{float(row['mean_test_mae']):.6f} ± {float(row['std_test_mae']):.6f}",
                "test_rmse_mean_sd": f"{float(row['mean_test_rmse']):.6f} ± {float(row['std_test_rmse']):.6f}",
                "test_r2_mean_sd": f"{float(row['mean_test_r2']):.6f} ± {float(row['std_test_r2']):.6f}",
                "test_pearson_mean_sd": f"{float(row['mean_test_pearson']):.6f} ± {float(row['std_test_pearson']):.6f}",
            }
        )
    columns = tuple(rows[0]) if rows else (
        "model",
        grouping_field,
        "seeds",
        "test_mae_mean_sd",
        "test_rmse_mean_sd",
        "test_r2_mean_sd",
        "test_pearson_mean_sd",
    )
    table_dir = output_dir / "tables"
    write_csv(table_dir / "main_results_table.csv", rows, columns)
    write_text(table_dir / "main_results_table.md", format_markdown_table(rows, columns))
    latex = [
        r"\begin{tabular}{llccccc}",
        r"\hline",
        (
            r"Model & "
            + _latex_escape(grouping_label)
            + r" & Seeds & MAE & RMSE & $R^2$ & Pearson $r$ \\"
        ),
        r"\hline",
    ]
    for row in rows:
        latex.append(
            " & ".join(
                [
                    _latex_escape(str(row["model"])),
                    str(row[grouping_field]),
                    _latex_escape(str(row["seeds"])),
                    str(row["test_mae_mean_sd"]).replace("±", r"$\pm$"),
                    str(row["test_rmse_mean_sd"]).replace("±", r"$\pm$"),
                    str(row["test_r2_mean_sd"]).replace("±", r"$\pm$"),
                    str(row["test_pearson_mean_sd"]).replace("±", r"$\pm$"),
                ]
            )
            + r" \\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    write_text(table_dir / "main_results_table.tex", "\n".join(latex))


def _model_colors(models: Sequence[str]) -> dict[str, str]:
    palette = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    return {model: palette[index % len(palette)] for index, model in enumerate(models)}


def _save_figure(fig: Any, base: Path, formats: Sequence[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"dpi": 240} if fmt == "png" else {}
        fig.savefig(base.with_suffix(f".{fmt}"), bbox_inches="tight", **kwargs)


def _write_plot_data(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = tuple(rows[0].keys()) if rows else ("status",)
    write_csv(path, rows, columns)


def create_figures(
    output_dir: Path,
    spec: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    aggregated: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
    prediction_pairs: Mapping[
        tuple[str, Any, int], tuple[list[float], list[float]]
    ],
) -> list[str]:
    """Create reusable matplotlib-only figures and their exact plotting data."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PipelineError("matplotlib is required to build analysis figures") from exc

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    formats = spec["figure_formats"]
    groups = spec["grouping_values"]
    grouping_field = spec["grouping_field"]
    grouping_axis_label = spec.get(
        "grouping_axis_label", grouping_field.replace("_", " ").title()
    )
    grouping_value_label = spec.get(
        "grouping_value_label", grouping_field.replace("_", " ")
    )
    grouping_suffix = spec.get(
        "plot_grouping_suffix",
        re.sub(r"[^a-z0-9]+", "_", grouping_field.lower()).strip("_"),
    )
    models = [entry["label"] for entry in spec["families"]]
    colors = _model_colors(models)
    figure_dir = output_dir / "figures"
    plot_dir = output_dir / "plot_data"
    created: list[str] = []

    metric_specs = (
        ("test_mae", "Test MAE", False),
        ("test_rmse", "Test RMSE", False),
        ("test_r2", r"Test $R^2$", True),
    )
    for metric, ylabel, zero_line in metric_specs:
        fig, axis = plt.subplots(figsize=(7.4, 4.8))
        plot_rows: list[dict[str, Any]] = []
        for model in models:
            model_seed = [row for row in seed_rows if row["model"] == model]
            model_agg = [row for row in aggregated if row["model"] == model]
            for row in model_seed:
                axis.scatter(
                    row[grouping_field],
                    float(row[metric]),
                    color=colors[model],
                    alpha=0.45,
                    s=24,
                )
                plot_rows.append(
                    {
                        "model": model,
                        grouping_field: row[grouping_field],
                        "seed": row["seed"],
                        metric: row[metric],
                        "point_type": "seed",
                    }
                )
            xs = [row[grouping_field] for row in model_agg]
            means = [float(row[f"mean_{metric}"]) for row in model_agg]
            stds = [float(row[f"std_{metric}"]) for row in model_agg]
            axis.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                linewidth=1.7,
                color=colors[model],
                label=model,
            )
            for row in model_agg:
                plot_rows.append(
                    {
                        "model": model,
                        grouping_field: row[grouping_field],
                        "seed": "",
                        metric: row[f"mean_{metric}"],
                        "point_type": "mean",
                        "sample_standard_deviation": row[f"std_{metric}"],
                    }
                )
        if zero_line:
            axis.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.65)
        axis.set_xlabel(grouping_axis_label)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} versus {grouping_value_label}")
        axis.grid(alpha=0.22)
        axis.legend()
        fig.tight_layout()
        name = f"{metric}_vs_{grouping_suffix}"
        _save_figure(fig, figure_dir / name, formats)
        plt.close(fig)
        _write_plot_data(plot_dir / f"{name}.csv", plot_rows)
        created.append(name)

    if paired:
        fig, axis = plt.subplots(figsize=(7.4, 4.8))
        plot_rows = []
        grouped: dict[Any, list[float]] = defaultdict(list)
        for row in paired:
            value = float(row["static_minus_evolve_mae"])
            grouped[row[grouping_field]].append(value)
            axis.scatter(
                row[grouping_field], value, color="#6a3d9a", alpha=0.55, s=28
            )
            plot_rows.append(
                {
                    grouping_field: row[grouping_field],
                    "seed": row["seed"],
                    "static_minus_evolve_mae": value,
                    "point_type": "seed",
                }
            )
        means = [statistics.mean(grouped[group]) for group in groups]
        stds = [sample_standard_deviation(grouped[group]) for group in groups]
        axis.errorbar(groups, means, yerr=stds, color="#6a3d9a", marker="o", capsize=4)
        for group, mean, std in zip(groups, means, stds):
            plot_rows.append(
                {
                    grouping_field: group,
                    "seed": "",
                    "static_minus_evolve_mae": mean,
                    "point_type": "mean",
                    "sample_standard_deviation": std,
                }
            )
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel(grouping_axis_label)
        axis.set_ylabel("Static MAE − Evolve MAE")
        axis.set_title("Paired MAE difference (negative favors Static GCN)")
        axis.grid(alpha=0.22)
        fig.tight_layout()
        name = f"paired_mae_difference_vs_{grouping_suffix}"
        _save_figure(fig, figure_dir / name, formats)
        plt.close(fig)
        _write_plot_data(plot_dir / f"{name}.csv", plot_rows)
        created.append(name)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    plot_rows = []
    for model in models:
        rows = [row for row in aggregated if row["model"] == model]
        axis.plot(
            [row[grouping_field] for row in rows],
            [float(row["std_test_mae"]) for row in rows],
            marker="o",
            color=colors[model],
            label=model,
        )
        plot_rows.extend(
            {
                "model": model,
                grouping_field: row[grouping_field],
                "sample_standard_deviation_test_mae": row["std_test_mae"],
            }
            for row in rows
        )
    axis.set_xlabel(grouping_axis_label)
    axis.set_ylabel("Between-seed SD of test MAE")
    axis.set_title(f"Seed variability versus {grouping_value_label}")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    name = f"seed_variability_vs_{grouping_suffix}"
    _save_figure(fig, figure_dir / name, formats)
    plt.close(fig)
    _write_plot_data(plot_dir / f"{name}.csv", plot_rows)
    created.append(name)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    plot_rows = []
    for model in models:
        rows = [row for row in seed_rows if row["model"] == model]
        grouped_ratios: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            ratio = float(row["prediction_std_ratio"])
            grouped_ratios[row[grouping_field]].append(ratio)
            axis.scatter(
                row[grouping_field], ratio, color=colors[model], alpha=0.45, s=24
            )
            plot_rows.append(
                {
                    "model": model,
                    grouping_field: row[grouping_field],
                    "seed": row["seed"],
                    "prediction_std_ratio": ratio,
                    "point_type": "seed",
                }
            )
        means = [statistics.mean(grouped_ratios[group]) for group in groups]
        stds = [sample_standard_deviation(grouped_ratios[group]) for group in groups]
        axis.errorbar(
            groups,
            means,
            yerr=stds,
            color=colors[model],
            marker="o",
            capsize=4,
            label=model,
        )
        for group, mean, std in zip(groups, means, stds):
            plot_rows.append(
                {
                    "model": model,
                    grouping_field: group,
                    "seed": "",
                    "prediction_std_ratio": mean,
                    "point_type": "mean",
                    "sample_standard_deviation": std,
                }
            )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel(grouping_axis_label)
    axis.set_ylabel("Prediction SD / target SD")
    axis.set_title(f"Prediction dispersion versus {grouping_value_label}")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    name = f"prediction_std_ratio_vs_{grouping_suffix}"
    _save_figure(fig, figure_dir / name, formats)
    plt.close(fig)
    _write_plot_data(plot_dir / f"{name}.csv", plot_rows)
    created.append(name)

    if spec["prediction_plots_enabled"]:
        representative_lookup: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in representatives:
            representative_lookup[(str(row["model"]), row[grouping_field])].append(row)
        for residual_mode in (False, True):
            fig, axes = plt.subplots(
                1,
                len(groups),
                figsize=(3.15 * len(groups), 3.6),
                sharex=False,
                sharey=False,
            )
            if len(groups) == 1:
                axes = [axes]
            plot_rows = []
            for axis, group in zip(axes, groups):
                values_for_limits: list[float] = []
                annotation_index = 0
                for model in models:
                    for row in representative_lookup[(model, group)]:
                        targets, predictions = prediction_pairs[
                            (model, group, int(row["seed"]))
                        ]
                        y_values = [
                            prediction - target
                            for target, prediction in zip(targets, predictions)
                        ] if residual_mode else predictions
                        axis.scatter(
                            targets,
                            y_values,
                            color=colors[model],
                            alpha=0.72,
                            s=25,
                            label=model,
                        )
                        metrics_text = (
                            f"{model} s{row['seed']}\n"
                            f"MAE {float(row['test_mae']):.3f}  "
                            f"RMSE {float(row['test_rmse']):.3f}\n"
                            f"$R^2$ {float(row['test_r2']):.3f}  "
                            f"$r$ {float(row['test_pearson']):.3f}"
                        )
                        axis.text(
                            0.02,
                            0.98 - 0.20 * annotation_index,
                            metrics_text,
                            transform=axis.transAxes,
                            va="top",
                            ha="left",
                            fontsize=7,
                            color=colors[model],
                            bbox={
                                "boxstyle": "round,pad=0.22",
                                "facecolor": "white",
                                "edgecolor": colors[model],
                                "alpha": 0.78,
                            },
                        )
                        annotation_index += 1
                        values_for_limits.extend(targets)
                        if not residual_mode:
                            values_for_limits.extend(predictions)
                        for target, prediction, y_value in zip(
                            targets, predictions, y_values
                        ):
                            plot_rows.append(
                                {
                                    "model": model,
                                    grouping_field: group,
                                    "seed": row["seed"],
                                    "target": target,
                                    "prediction": prediction,
                                    "residual": prediction - target,
                                    "plotted_y": y_value,
                                    "test_mae": row["test_mae"],
                                    "test_rmse": row["test_rmse"],
                                    "test_r2": row["test_r2"],
                                    "test_pearson": row["test_pearson"],
                                }
                            )
                if residual_mode:
                    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
                    axis.set_ylabel(r"Residual ($\hat{\Omega}_m-\Omega_m$)")
                elif values_for_limits:
                    low, high = min(values_for_limits), max(values_for_limits)
                    margin = max((high - low) * 0.06, 0.005)
                    axis.plot(
                        [low - margin, high + margin],
                        [low - margin, high + margin],
                        color="black",
                        linewidth=1,
                        linestyle="--",
                    )
                    axis.set_ylabel(r"Predicted $\Omega_m$")
                axis.set_xlabel(r"True $\Omega_m$")
                axis.set_title(f"{group} {grouping_value_label}")
                axis.grid(alpha=0.18)
            handles, labels_found = axes[0].get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels_found, handles))
                fig.legend(
                    unique.values(),
                    unique.keys(),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=max(1, len(models)),
                )
            title = (
                "Residuals for representative runs"
                if residual_mode
                else "True versus predicted for representative runs"
            )
            fig.suptitle(title, y=1.10)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            name = (
                "residuals_vs_true_representative_runs"
                if residual_mode
                else "true_vs_predicted_representative_runs"
            )
            _save_figure(fig, figure_dir / name, formats)
            plt.close(fig)
            _write_plot_data(plot_dir / f"{name}.csv", plot_rows)
            created.append(name)
    return created


def _trend_text(
    aggregated: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    groups: Sequence[Any],
    grouping_field: str,
    grouping_value_label: str,
) -> list[str]:
    lines: list[str] = []
    for model in models:
        rows = [row for row in aggregated if row["model"] == model]
        maes = [float(row["mean_test_mae"]) for row in rows]
        monotonic = all(next_value <= value for value, next_value in zip(maes, maes[1:]))
        direction = "decreased" if maes[-1] < maes[0] else "increased"
        lines.append(
            f"- {model}: mean test MAE {direction} from {maes[0]:.6f} at "
            f"{groups[0]} {grouping_value_label} to {maes[-1]:.6f} at "
            f"{groups[-1]} {grouping_value_label}. "
            f"The sequence was {'monotonically non-increasing' if monotonic else 'non-monotonic'}."
        )
    return lines


def write_scientific_summary(
    output_dir: Path,
    validation: AnalysisValidation,
    seed_rows: Sequence[Mapping[str, Any]],
    aggregated: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    representatives: Sequence[Mapping[str, Any]],
) -> None:
    spec = validation.spec
    models = [entry["label"] for entry in spec["families"]]
    groups = spec["grouping_values"]
    grouping_field = spec["grouping_field"]
    grouping_value_label = spec.get(
        "grouping_value_label", grouping_field.replace("_", " ")
    )
    representative_rows = [
        {
            "model": row["model"],
            grouping_field: row[grouping_field],
            "seed": row["seed"],
            "test_mae": row["test_mae"],
        }
        for row in representatives
    ]
    if paired:
        static_mae_wins = sum(
            float(row["static_minus_evolve_mae"]) < 0 for row in paired
        )
        evolve_mae_wins = sum(
            float(row["static_minus_evolve_mae"]) > 0 for row in paired
        )
        ties = len(paired) - static_mae_wins - evolve_mae_wins
        architecture_text = (
            f"Across {len(paired)} paired universe/seed rows, Static GCN had lower "
            f"MAE in {static_mae_wins}, EvolveGCN-H in {evolve_mae_wins}, with "
            f"{ties} exact ties. This is descriptive evidence, not a causal test."
        )
    else:
        architecture_text = "No paired architecture comparison was requested."
    ratios = [float(row["prediction_std_ratio"]) for row in seed_rows]
    repeats = [float(row["repeated_prediction_fraction"]) for row in seed_rows]
    collapse_text = (
        f"Prediction-SD/target-SD ratios ranged from {min(ratios):.3f} to "
        f"{max(ratios):.3f}; the maximum exact repeated-prediction fraction was "
        f"{max(repeats):.3f}. Low dispersion can indicate regression toward the "
        "mean, but these diagnostics alone do not establish a collapse mechanism."
    )
    parts = [
        f"# Scientific Summary: {spec['title']}",
        "",
        "## Purpose",
        "",
        spec["scientific_question"],
        "",
        "## Controlled variables",
        "",
        (
            spec.get(
                "controlled_variables_summary",
                "All required fixed protocol fields, grouping values, seeds, and "
                "exact split IDs are matched.",
            )
            + " Error bars are sample standard deviations across the required "
            "seeds; test samples are not pooled."
        ),
        "",
        "## Intentional differences",
        "",
        spec["comparison_statement"],
        "",
        *[
            f"- `{item['field']}`: {item['reason']}"
            for item in validation.intentional_differences
        ],
        "",
        "## Numerical trends",
        "",
        *_trend_text(
            aggregated, models, groups, grouping_field, grouping_value_label
        ),
        "",
        architecture_text,
        "",
        "Negative R² values are retained in every table and figure.",
        "",
        "## Prediction-collapse diagnostics",
        "",
        collapse_text,
        "",
        "## Representative runs",
        "",
        (
            f"Policy: `{spec['representative_seed_policy']['policy']}`. The best "
            "seed is not selected automatically under the default policy."
        ),
        "",
        format_markdown_table(
            representative_rows, ("model", grouping_field, "seed", "test_mae")
        ),
        "",
        "## Limitations and interpretation",
        "",
        (
            "The comparison is controlled for data scale, graph construction, seeds, "
            "and splits, but architecture, temporal input, model-specific head, and "
            "batch size may intentionally differ. Consequently, observed differences "
            "should be interpreted as performance of the complete canonical model "
            "protocols, not as an isolated causal effect of temporal modeling or any "
            "single architectural component. "
            + spec.get(
                "limitations_summary",
                "Interpretation is limited to the populations and protocol encoded "
                "by the listed experiment families.",
            )
        ),
    ]
    write_text(output_dir / "scientific_summary.md", "\n".join(parts))


def build_analysis_package(
    repo_root: Path,
    validation: AnalysisValidation,
) -> dict[str, Any]:
    """Build a complete package only after a successful validation."""

    if not validation.valid:
        raise PipelineError("analysis validation failed; no package may be generated")
    output_dir = resolve_repo_path(repo_root, validation.spec["output_directory"])
    seed_rows, prediction_pairs = build_seed_rows(repo_root, validation)
    models = [entry["label"] for entry in validation.spec["families"]]
    grouping_field = validation.spec["grouping_field"]
    aggregated = aggregate_seed_rows(
        seed_rows, models, validation.spec["grouping_values"], grouping_field
    )
    paired = pair_model_rows(seed_rows, grouping_field)
    representatives = select_representative_rows(
        seed_rows, validation.spec["representative_seed_policy"], grouping_field
    )
    expected_seed_rows = (
        len(models)
        * len(validation.spec["grouping_values"])
        * len(validation.spec["required_seeds"])
    )
    if len(seed_rows) != expected_seed_rows:
        raise PipelineError(
            f"seed-level row count={len(seed_rows)}, expected={expected_seed_rows}"
        )
    expected_aggregates = len(models) * len(validation.spec["grouping_values"])
    if len(aggregated) != expected_aggregates:
        raise PipelineError(
            f"aggregate row count={len(aggregated)}, expected={expected_aggregates}"
        )
    if len(models) == 2 and len(paired) != len(validation.spec["grouping_values"]) * len(
        validation.spec["required_seeds"]
    ):
        raise PipelineError("paired row count does not match grouping/seed product")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_compatibility_report(output_dir, validation)
    write_protocol_outputs(output_dir, validation)
    seed_columns = ("model", grouping_field, *SEED_COLUMNS[1:])
    aggregate_columns = ("model", grouping_field, *AGGREGATE_COLUMNS[1:])
    paired_columns = (grouping_field, *PAIRED_COLUMNS)
    write_csv(output_dir / "seed_level_results.csv", seed_rows, seed_columns)
    write_csv(
        output_dir / "aggregated_results.csv", aggregated, aggregate_columns
    )
    if paired:
        write_csv(
            output_dir / "paired_model_differences.csv", paired, paired_columns
        )
    write_main_tables(
        output_dir,
        aggregated,
        grouping_field,
        validation.spec.get(
            "grouping_value_label", grouping_field.replace("_", " ").title()
        ),
    )
    write_scientific_summary(
        output_dir,
        validation,
        seed_rows,
        aggregated,
        paired,
        representatives,
    )
    figures = create_figures(
        output_dir,
        validation.spec,
        seed_rows,
        aggregated,
        paired,
        representatives,
        prediction_pairs,
    )
    manifest = build_manifest(repo_root, validation, seed_rows, representatives)
    manifest["generated_figures"] = figures
    manifest["generated_row_counts"] = {
        "seed_level_results": len(seed_rows),
        "aggregated_results": len(aggregated),
        "paired_model_differences": len(paired),
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    return {
        "output_directory": output_dir,
        "seed_rows": len(seed_rows),
        "aggregate_rows": len(aggregated),
        "paired_rows": len(paired),
        "figures": figures,
    }
