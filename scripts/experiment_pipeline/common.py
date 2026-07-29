"""Shared, dataset-safe helpers for controlled experiment families."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
VALID_ORIGINS = {"reusable_existing", "canonical_replacement", "planned_new"}
VALID_ACTIONS = {"reuse", "run_if_missing", "exclude"}


class PipelineError(RuntimeError):
    """Raised for malformed specifications or unsafe inspection conditions."""


@dataclass(frozen=True)
class RunInspection:
    """Live filesystem state for one family run."""

    grouping_value: Any
    seed: int
    experiment_name: str
    experiment_path: str
    dataset_path: str
    origin: str
    action: str
    status: str
    artifact_count: int
    artifact_total: int
    missing_artifacts: tuple[str, ...]
    compatibility: str
    compatibility_errors: tuple[str, ...]
    runnable: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    """Verifier result shared by the CLI and results builder."""

    valid: bool
    complete: bool
    rows: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    incomplete_messages: tuple[str, ...]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a useful path-specific error."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"expected JSON object at {path}")
    return value


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve a path and reject paths outside the repository root."""

    root = repo_root.expanduser().resolve()
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PipelineError(f"path escapes repository root: {value}") from exc
    return resolved


def repo_relative(repo_root: Path, path: Path) -> str:
    """Return a stable POSIX repository-relative path."""

    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _require_type(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type | tuple[type, ...],
) -> Any:
    if key not in mapping:
        raise PipelineError(f"family specification is missing required field: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise PipelineError(
            f"family specification field {key!r} has invalid type "
            f"{type(value).__name__}"
        )
    return value


def validate_family_spec(spec: dict[str, Any]) -> None:
    """Validate the supported schema and exact required run coverage."""

    version = _require_type(spec, "schema_version", str)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PipelineError(
            f"unsupported schema_version {version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    for key in (
        "family_id",
        "family_title",
        "scientific_question",
        "model_name",
        "training_module",
        "output_root",
        "grouping_field",
        "prediction_file",
    ):
        _require_type(spec, key, str)
    artifacts = _require_type(spec, "expected_artifacts", list)
    if not artifacts or not all(isinstance(item, str) and item for item in artifacts):
        raise PipelineError("expected_artifacts must be a non-empty list of paths")
    seeds = _require_type(spec, "required_seeds", list)
    groups = _require_type(spec, "grouping_values", list)
    runs = _require_type(spec, "runs", list)
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise PipelineError("required_seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise PipelineError("required_seeds contains duplicates")
    if not groups or len({json.dumps(item, sort_keys=True) for item in groups}) != len(groups):
        raise PipelineError("grouping_values must be non-empty and unique")
    _require_type(spec, "fixed_scientific_settings", dict)
    _require_type(spec, "allowed_varying_fields", list)
    _require_type(spec, "target_column_aliases", list)
    _require_type(spec, "prediction_column_aliases", list)
    _require_type(spec, "runner", dict)

    seen: set[tuple[str, int]] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise PipelineError(f"runs[{index}] must be an object")
        for key in (
            "group_value",
            "seed",
            "dataset_path",
            "experiment_name",
            "experiment_path",
            "origin",
            "action",
        ):
            if key not in run:
                raise PipelineError(f"runs[{index}] is missing {key!r}")
        if run["group_value"] not in groups:
            raise PipelineError(
                f"runs[{index}] group_value {run['group_value']!r} "
                "is not in grouping_values"
            )
        if run["seed"] not in seeds:
            raise PipelineError(
                f"runs[{index}] seed {run['seed']!r} is not in required_seeds"
            )
        if run["origin"] not in VALID_ORIGINS:
            raise PipelineError(f"runs[{index}] has invalid origin {run['origin']!r}")
        if run["action"] not in VALID_ACTIONS:
            raise PipelineError(f"runs[{index}] has invalid action {run['action']!r}")
        if run["action"] == "exclude" and not run.get("exclusion_reason"):
            raise PipelineError(f"runs[{index}] excludes a row without a reason")
        pair = (json.dumps(run["group_value"], sort_keys=True), int(run["seed"]))
        if pair in seen:
            raise PipelineError(
                f"duplicate run for group={run['group_value']!r}, seed={run['seed']}"
            )
        seen.add(pair)

    expected = {
        (json.dumps(group, sort_keys=True), seed)
        for group in groups
        for seed in seeds
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise PipelineError(
            f"run coverage is not the exact grouping-value/seed product; "
            f"missing={missing}, extra={extra}"
        )


def load_family_spec(path: Path) -> dict[str, Any]:
    """Load and validate a family specification."""

    spec = read_json(path)
    validate_family_spec(spec)
    return spec


def deterministic_runs(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return runs ordered by declared grouping values and seeds."""

    group_order = {
        json.dumps(value, sort_keys=True): index
        for index, value in enumerate(spec["grouping_values"])
    }
    seed_order = {int(seed): index for index, seed in enumerate(spec["required_seeds"])}
    return sorted(
        (dict(run) for run in spec["runs"]),
        key=lambda run: (
            group_order[json.dumps(run["group_value"], sort_keys=True)],
            seed_order[int(run["seed"])],
            run["experiment_name"],
        ),
    )


def values_equal(actual: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    """Compare JSON values, using strict absolute tolerance for numeric values."""

    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def dataset_metadata_path(dataset_path: Path) -> Path:
    """Return the lightweight sidecar path used by project datasets."""

    return dataset_path.with_suffix(".metadata.json")


def expected_config_for_run(
    spec: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exact expected config values for one run."""

    settings = spec["fixed_scientific_settings"]
    expected = dict(settings.get("config", {}))
    grouping_config_field = spec.get("grouping_config_field")
    if grouping_config_field:
        expected[str(grouping_config_field)] = run["group_value"]
    expected.update(
        {
            "seed": run["seed"],
            "dataset_path": run["dataset_path"],
            "experiment_name": run["experiment_name"],
            "output_root": spec["output_root"],
        }
    )
    overrides = run.get("expected_config_overrides", {})
    if not isinstance(overrides, dict):
        raise PipelineError(
            f"{run['experiment_name']}: expected_config_overrides must be an object"
        )
    expected.update(overrides)
    return expected


def compare_run_configuration(
    repo_root: Path,
    spec: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[str]:
    """Compare config and dataset metadata without loading the dataset."""

    errors: list[str] = []
    experiment_dir = resolve_repo_path(repo_root, run["experiment_path"])
    config_path = experiment_dir / "config.json"
    if not config_path.is_file():
        return [f"missing config: {repo_relative(repo_root, config_path)}"]
    config = read_json(config_path)
    settings = spec["fixed_scientific_settings"]
    legacy_defaults = settings.get("legacy_config_defaults", {})
    tolerance = float(spec.get("metric_tolerance", 1e-6))
    for field, expected in expected_config_for_run(spec, run).items():
        actual = config.get(field, legacy_defaults.get(field))
        if not values_equal(actual, expected, tolerance=min(tolerance, 1e-12)):
            errors.append(f"config {field}={actual!r}, expected {expected!r}")

    dataset_path = resolve_repo_path(repo_root, run["dataset_path"])
    metadata_path = dataset_metadata_path(dataset_path)
    if not metadata_path.is_file():
        errors.append(f"missing dataset metadata: {repo_relative(repo_root, metadata_path)}")
        return errors
    metadata = read_json(metadata_path)
    expected_metadata = dict(settings.get("dataset_metadata", {}))
    grouping_metadata_field = spec.get("grouping_metadata_field")
    if grouping_metadata_field:
        expected_metadata[str(grouping_metadata_field)] = run["group_value"]
    for field, expected in expected_metadata.items():
        actual = metadata.get(field)
        if not values_equal(actual, expected, tolerance=min(tolerance, 1e-12)):
            errors.append(f"metadata {field}={actual!r}, expected {expected!r}")
    return errors


def inspect_run(
    repo_root: Path,
    spec: Mapping[str, Any],
    run: Mapping[str, Any],
) -> RunInspection:
    """Classify one run from live artifacts."""

    expected_artifacts = tuple(str(item) for item in spec["expected_artifacts"])
    if run["action"] == "exclude":
        return RunInspection(
            grouping_value=run["group_value"],
            seed=int(run["seed"]),
            experiment_name=str(run["experiment_name"]),
            experiment_path=str(run["experiment_path"]),
            dataset_path=str(run["dataset_path"]),
            origin=str(run["origin"]),
            action=str(run["action"]),
            status="excluded",
            artifact_count=0,
            artifact_total=len(expected_artifacts),
            missing_artifacts=expected_artifacts,
            compatibility="not_applicable",
            compatibility_errors=(),
            runnable=False,
            notes=str(run.get("exclusion_reason") or run.get("notes", "")),
        )

    experiment_dir = resolve_repo_path(repo_root, run["experiment_path"])
    present = tuple(
        artifact
        for artifact in expected_artifacts
        if (experiment_dir / artifact).is_file()
    )
    missing = tuple(item for item in expected_artifacts if item not in present)
    if not experiment_dir.exists():
        status = "missing"
    elif experiment_dir.is_dir() and not missing:
        status = "complete"
    else:
        status = "partial"

    compatibility_errors: tuple[str, ...] = ()
    compatibility = "not_checked"
    if status in {"complete", "partial"} and (experiment_dir / "config.json").is_file():
        try:
            compatibility_errors = tuple(compare_run_configuration(repo_root, spec, run))
            compatibility = "compatible" if not compatibility_errors else "incompatible"
        except PipelineError as exc:
            compatibility_errors = (str(exc),)
            compatibility = "inspection_error"
    runnable = status == "missing" and run["action"] == "run_if_missing"
    return RunInspection(
        grouping_value=run["group_value"],
        seed=int(run["seed"]),
        experiment_name=str(run["experiment_name"]),
        experiment_path=str(run["experiment_path"]),
        dataset_path=str(run["dataset_path"]),
        origin=str(run["origin"]),
        action=str(run["action"]),
        status=status,
        artifact_count=len(present),
        artifact_total=len(expected_artifacts),
        missing_artifacts=missing,
        compatibility=compatibility,
        compatibility_errors=compatibility_errors,
        runnable=runnable,
        notes=str(run.get("notes", "")),
    )


def inspect_family(
    repo_root: Path,
    spec: Mapping[str, Any],
) -> list[RunInspection]:
    """Inspect all rows in deterministic order."""

    return [inspect_run(repo_root, spec, run) for run in deterministic_runs(spec)]


def find_column(fieldnames: Sequence[str], aliases: Sequence[str], kind: str) -> str:
    """Find the first configured CSV column alias."""

    for alias in aliases:
        if alias in fieldnames:
            return alias
    raise PipelineError(
        f"prediction CSV has no configured {kind} column; "
        f"aliases={list(aliases)}, columns={list(fieldnames)}"
    )


def read_prediction_pairs(
    path: Path,
    target_aliases: Sequence[str],
    prediction_aliases: Sequence[str],
) -> tuple[list[float], list[float]]:
    """Read finite target/prediction pairs from CSV."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise PipelineError(f"prediction CSV has no header: {path}")
            target_column = find_column(reader.fieldnames, target_aliases, "target")
            prediction_column = find_column(
                reader.fieldnames, prediction_aliases, "prediction"
            )
            targets: list[float] = []
            predictions: list[float] = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    target = float(row[target_column])
                    prediction = float(row[prediction_column])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PipelineError(
                        f"invalid prediction value at {path}:{line_number}"
                    ) from exc
                if not math.isfinite(target) or not math.isfinite(prediction):
                    raise PipelineError(
                        f"non-finite prediction value at {path}:{line_number}"
                    )
                targets.append(target)
                predictions.append(prediction)
    except OSError as exc:
        raise PipelineError(f"cannot read prediction CSV {path}: {exc}") from exc
    if not targets:
        raise PipelineError(f"prediction CSV has no rows: {path}")
    return targets, predictions


def recompute_metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    """Compute regression metrics using standard-library arithmetic."""

    if len(targets) != len(predictions) or not targets:
        raise PipelineError("target and prediction lengths must match and be nonzero")
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    count = len(targets)
    mse = sum(squared_errors) / count
    target_mean = statistics.mean(targets)
    prediction_mean = statistics.mean(predictions)
    ss_total = sum((target - target_mean) ** 2 for target in targets)
    ss_residual = sum(squared_errors)
    r2 = 1.0 - ss_residual / ss_total if ss_total else float("nan")
    if count > 1:
        target_std = statistics.stdev(targets)
        prediction_std = statistics.stdev(predictions)
        covariance_sum = sum(
            (target - target_mean) * (prediction - prediction_mean)
            for target, prediction in zip(targets, predictions)
        )
        target_ss = sum((target - target_mean) ** 2 for target in targets)
        prediction_ss = sum(
            (prediction - prediction_mean) ** 2 for prediction in predictions
        )
        pearson = (
            covariance_sum / math.sqrt(target_ss * prediction_ss)
            if target_ss and prediction_ss
            else float("nan")
        )
    else:
        target_std = prediction_std = pearson = float("nan")
    return {
        "test_mae": sum(absolute_errors) / count,
        "test_mse": mse,
        "test_rmse": math.sqrt(mse),
        "test_r2": r2,
        "test_pearson": pearson,
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "prediction_std_ratio": (
            prediction_std / target_std if target_std else float("nan")
        ),
        "test_count": float(count),
    }


def split_signature(config: Mapping[str, Any]) -> str:
    """Hash ordered train/validation/test identifiers."""

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


def _extract_saved_test_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    test = metrics.get("test")
    if not isinstance(test, dict):
        raise PipelineError("metrics.json does not contain a test object")
    result = {
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_mse": test.get("mse"),
        "test_count": test.get("num_samples"),
    }
    for source, target in (
        ("r2", "test_r2"),
        ("pearson", "test_pearson"),
    ):
        if source in test:
            result[target] = test[source]
    return result


def _source_commit(config: Mapping[str, Any]) -> str:
    for key in ("git_commit", "source_commit", "commit", "git_commit_saved"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def verify_family(
    repo_root: Path,
    spec: Mapping[str, Any],
    *,
    allow_incomplete: bool,
) -> VerificationResult:
    """Independently verify artifacts, configs, splits, and predictions."""

    errors: list[str] = []
    incomplete: list[str] = []
    verified_rows: list[dict[str, Any]] = []
    observed_config_values: dict[str, dict[str, list[str]]] = {}
    tolerance = float(spec.get("metric_tolerance", 1e-6))
    inspections = inspect_family(repo_root, spec)
    runs_by_pair = {
        (json.dumps(run["group_value"], sort_keys=True), int(run["seed"])): run
        for run in deterministic_runs(spec)
    }

    for inspection in inspections:
        key = (
            json.dumps(inspection.grouping_value, sort_keys=True),
            inspection.seed,
        )
        run = runs_by_pair[key]
        label = f"{spec['grouping_field']}={inspection.grouping_value}, seed={inspection.seed}"
        if inspection.status == "excluded":
            continue
        if inspection.status in {"missing", "partial"}:
            message = (
                f"{label}: {inspection.status} ({inspection.experiment_name}); "
                f"missing={list(inspection.missing_artifacts)}"
            )
            incomplete.append(message)
            if not allow_incomplete:
                errors.append(message)
            continue
        if inspection.compatibility != "compatible":
            for detail in inspection.compatibility_errors or ("unknown mismatch",):
                errors.append(f"{label}: {detail}")
            continue

        experiment_dir = resolve_repo_path(repo_root, run["experiment_path"])
        config = read_json(experiment_dir / "config.json")
        for field, value in config.items():
            encoded = json.dumps(value, sort_keys=True, default=str)
            observed_config_values.setdefault(field, {}).setdefault(encoded, []).append(
                label
            )
        train_ids = config.get("train_ids")
        val_ids = config.get("val_ids")
        test_ids = config.get("test_ids")
        if not all(isinstance(ids, list) for ids in (train_ids, val_ids, test_ids)):
            errors.append(f"{label}: split ID lists are missing")
            continue
        split_sets = [set(train_ids), set(val_ids), set(test_ids)]
        if (
            split_sets[0] & split_sets[1]
            or split_sets[0] & split_sets[2]
            or split_sets[1] & split_sets[2]
        ):
            errors.append(f"{label}: split overlap detected")
            continue
        expected_total = inspection.grouping_value if spec["grouping_field"] == "universes" else None
        if expected_total is not None:
            train_ratio = float(spec["fixed_scientific_settings"]["config"]["train_ratio"])
            val_ratio = float(spec["fixed_scientific_settings"]["config"]["val_ratio"])
            expected_train = int(expected_total * train_ratio)
            expected_val = int(expected_total * val_ratio)
            expected_counts = (
                expected_train,
                expected_val,
                expected_total - expected_train - expected_val,
            )
            actual_counts = (len(train_ids), len(val_ids), len(test_ids))
            if actual_counts != expected_counts:
                errors.append(
                    f"{label}: split counts={actual_counts}, expected={expected_counts}"
                )
                continue

        prediction_path = experiment_dir / str(spec["prediction_file"])
        try:
            targets, predictions = read_prediction_pairs(
                prediction_path,
                spec["target_column_aliases"],
                spec["prediction_column_aliases"],
            )
            recomputed = recompute_metrics(targets, predictions)
            saved_metrics = read_json(experiment_dir / "metrics.json")
            saved = _extract_saved_test_metrics(saved_metrics)
        except PipelineError as exc:
            errors.append(f"{label}: {exc}")
            continue
        for metric in ("test_mae", "test_rmse", "test_mse"):
            actual = saved.get(metric)
            if actual is None or not values_equal(actual, recomputed[metric], tolerance):
                errors.append(
                    f"{label}: saved {metric}={actual!r}, "
                    f"recomputed={recomputed[metric]!r}"
                )
        saved_count = saved.get("test_count")
        if saved_count is None or int(saved_count) != int(recomputed["test_count"]):
            errors.append(
                f"{label}: saved test count={saved_count!r}, "
                f"prediction rows={int(recomputed['test_count'])}"
            )
        if len(test_ids) != int(recomputed["test_count"]):
            errors.append(
                f"{label}: test split IDs={len(test_ids)}, "
                f"prediction rows={int(recomputed['test_count'])}"
            )
        best_epoch = saved_metrics.get("best_epoch")
        if not isinstance(best_epoch, (int, float)) or isinstance(best_epoch, bool):
            errors.append(f"{label}: metrics.json has no numeric best_epoch")
        for metric in ("test_r2", "test_pearson"):
            if metric in saved and not values_equal(saved[metric], recomputed[metric], tolerance):
                errors.append(
                    f"{label}: saved {metric}={saved[metric]!r}, "
                    f"recomputed={recomputed[metric]!r}"
                )
        finite_metric_names = (
            "test_mae",
            "test_rmse",
            "test_mse",
            "test_r2",
            "target_mean",
            "target_std",
            "prediction_mean",
            "prediction_std",
            "prediction_std_ratio",
            "test_pearson",
        )
        for metric in finite_metric_names:
            if not math.isfinite(float(recomputed[metric])):
                errors.append(f"{label}: non-finite recomputed {metric}")

        verified_rows.append(
            {
                "family_id": spec["family_id"],
                "grouping_field": spec["grouping_field"],
                "grouping_value": inspection.grouping_value,
                "seed": inspection.seed,
                "experiment_name": inspection.experiment_name,
                "experiment_path": inspection.experiment_path,
                "origin": inspection.origin,
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "test_count": len(test_ids),
                "best_epoch": best_epoch if best_epoch is not None else "",
                **{key: value for key, value in recomputed.items() if key != "test_count"},
                "split_signature": split_signature(config),
                "source_commit_if_available": _source_commit(config),
                "notes": inspection.notes,
            }
        )

    expected_nonexcluded = {
        (json.dumps(run["group_value"], sort_keys=True), int(run["seed"]))
        for run in deterministic_runs(spec)
        if run["action"] != "exclude"
    }
    observed = {
        (json.dumps(row["grouping_value"], sort_keys=True), int(row["seed"]))
        for row in verified_rows
    }
    if not allow_incomplete and observed != expected_nonexcluded:
        errors.append(
            "verified seed/group coverage mismatch: "
            f"missing={sorted(expected_nonexcluded - observed)}"
        )
    fixed_fields = set(spec["fixed_scientific_settings"].get("config", {}))
    allowed_fields = set(spec.get("allowed_varying_fields", ()))
    allowed_fields.update(
        {
            "seed",
            "dataset_path",
            "experiment_name",
            "output_root",
            str(spec.get("grouping_config_field", "")),
        }
    )
    for field, values in sorted(observed_config_values.items()):
        if len(values) > 1 and field not in fixed_fields and field not in allowed_fields:
            rendered = "; ".join(
                f"{encoded}: {labels}" for encoded, labels in sorted(values.items())
            )
            errors.append(
                f"config field {field!r} varies but is not explicitly allowed: {rendered}"
            )
    complete = not incomplete and observed == expected_nonexcluded
    valid = not errors and (allow_incomplete or complete)
    return VerificationResult(
        valid=valid,
        complete=complete,
        rows=tuple(verified_rows),
        errors=tuple(errors),
        incomplete_messages=tuple(incomplete),
    )


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return sample SD (n-1); zero for a single observation."""

    return statistics.stdev(values) if len(values) > 1 else 0.0


def format_terminal_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Format a compact deterministic plain-text table."""

    if not rows:
        return "(no rows)"
    widths = {
        column: max(
            len(column),
            *(len(str(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Format a GitHub-flavored Markdown table."""

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write deterministic CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with one trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_provenance(repo_root: Path) -> dict[str, str]:
    """Read lightweight Git provenance without mutating the repository."""

    head_path = repo_root / ".git" / "HEAD"
    if not head_path.is_file():
        return {"commit": "", "branch": ""}
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            ref_path = repo_root / ".git" / ref
            commit = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else ""
            return {"commit": commit, "branch": ref.rsplit("/", 1)[-1]}
        return {"commit": head, "branch": ""}
    except OSError:
        return {"commit": "", "branch": ""}


def filtered_runs(
    spec: Mapping[str, Any],
    only_group: str | None,
    only_seed: int | None,
) -> list[dict[str, Any]]:
    """Filter runs without modifying the specification."""

    runs = deterministic_runs(spec)
    if only_group is not None:
        matches = [
            value
            for value in spec["grouping_values"]
            if str(value) == str(only_group)
        ]
        if not matches:
            raise PipelineError(
                f"--only-group {only_group!r} is not in {spec['grouping_values']!r}"
            )
        selected = matches[0]
        runs = [run for run in runs if run["group_value"] == selected]
    if only_seed is not None:
        if only_seed not in spec["required_seeds"]:
            raise PipelineError(
                f"--only-seed {only_seed} is not in {spec['required_seeds']!r}"
            )
        runs = [run for run in runs if int(run["seed"]) == only_seed]
    return runs


def build_training_command(
    spec: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[str]:
    """Build the deterministic training command encoded by the specification."""

    command = [
        "python3",
        "-m",
        str(spec["training_module"]),
        "--dataset_path",
        str(run["dataset_path"]),
        "--experiment_name",
        str(run["experiment_name"]),
        "--output_root",
        str(spec["output_root"]),
        "--seed",
        str(run["seed"]),
    ]
    overrides = run.get("argument_overrides", {})
    if not isinstance(overrides, dict):
        raise PipelineError(
            f"{run['experiment_name']}: argument_overrides must be an object"
        )
    entries = spec["runner"].get("argument_order")
    if not isinstance(entries, list):
        raise PipelineError("runner.argument_order must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "flag" not in entry:
            raise PipelineError("each runner.argument_order entry needs name and flag")
        name = str(entry["name"])
        value = overrides.get(name, entry.get("value"))
        if entry.get("flag_only"):
            enabled = overrides.get(name, True)
            if enabled:
                command.append(str(entry["flag"]))
            continue
        if value is None:
            raise PipelineError(f"runner argument {name!r} has no value")
        command.extend([str(entry["flag"]), str(value)])
    return command
