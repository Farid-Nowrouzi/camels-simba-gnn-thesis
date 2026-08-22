"""Integrity gates for U1000 Top500/Top750 infrastructure; no builds or training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis_reporting import analyze_u1000_halo_count_scaling_train700 as analysis
from scripts.validation import prepare_u1000_top500_top750_train700 as splits
from scripts.validation import validate_u1000_top500_top750_sparse_dataset as datasets


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/production/run_u1000_top500_top750_sparse_build.sh"


@pytest.mark.parametrize("top_n", [500, 750])
def test_top_n_whitelist_accepts_only_authorized_values(top_n: int) -> None:
    assert datasets.require_top_n(top_n) == top_n


@pytest.mark.parametrize("top_n", [499, 501, 749, 751, 1000, 1500, 1750, 2000])
def test_top_n_whitelist_rejects_other_values(top_n: int) -> None:
    with pytest.raises(ValueError, match="exactly"):
        datasets.require_top_n(top_n)


@pytest.mark.parametrize(
    ("top_n", "expected"),
    [
        (
            500,
            "data/processed/temporal_1000u_none_top500_periodic_knn_sparse/"
            "camels_1000u_temporal_logmass_none_top500_periodic_knn_sparse.pt",
        ),
        (
            750,
            "data/processed/temporal_1000u_none_top750_periodic_knn_sparse/"
            "camels_1000u_temporal_logmass_none_top750_periodic_knn_sparse.pt",
        ),
    ],
)
def test_deterministic_dataset_path_mapping(top_n: int, expected: str) -> None:
    assert datasets.dataset_layout(top_n).dataset.as_posix() == expected


@pytest.mark.parametrize("top_n", [500, 750])
def test_builder_command_has_frozen_scientific_arguments(top_n: int) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    command_block = source.split("command=(", 1)[1].split(")\nprintf -v command_text", 1)[0]
    required_fragments = (
        '--raw_dir "$RAW_DIR"',
        '--num_universes 1000',
        '--num_snapshots 5',
        '--num_nodes "$top_n"',
        '--normalization none',
        '--graph_mode knn',
        '--k 8',
        '--periodic_boundary',
        '--box_size 25.0',
        '--graph_storage sparse_edge_index',
        '--source_manifest_policy full_sha256',
        '--targets_csv "$TARGET_FILE"',
        '--device cpu',
        '--builder_entrypoint "$BUILDER_MODULE"',
        '--build_launcher_path "$BUILD_LAUNCHER"',
    )
    for fragment in required_fragments:
        assert fragment in command_block
    assert "--overwrite" not in command_block
    assert "--allow_partial" not in command_block
    assert f"top{top_n}" in datasets.dataset_layout(top_n).dataset.as_posix()


def test_destination_collision_refuses_replacement(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.pt"
    dataset.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing replacement"):
        datasets.assert_destination_clean(dataset)


@pytest.mark.parametrize("suffix", [".metadata.json", ".complete", ".pt.lock"])
def test_destination_sidecar_collision_refuses_replacement(tmp_path: Path, suffix: str) -> None:
    dataset = tmp_path / "dataset.pt"
    if suffix == ".pt.lock":
        collision = dataset.with_suffix(dataset.suffix + ".lock")
    else:
        collision = dataset.with_suffix(suffix)
    collision.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        datasets.assert_destination_clean(dataset)


def test_destination_atomic_temporary_collision_refuses_replacement(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.pt"
    (tmp_path / ".dataset.pt.build.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError):
        datasets.assert_destination_clean(dataset)


def test_existing_dedicated_output_directory_refuses_replacement(tmp_path: Path) -> None:
    output_directory = tmp_path / "intended_dataset_directory"
    output_directory.mkdir()
    with pytest.raises(FileExistsError):
        datasets.assert_destination_clean(output_directory / "dataset.pt")


@pytest.mark.parametrize("seed", [42, 123, 2025])
def test_split_references_are_exact_ordered_matches(seed: int) -> None:
    top1000, top1500 = splits.load_reference_pair(ROOT, seed)
    for partition, expected_count in splits.EXPECTED_COUNTS.items():
        key = f"{partition}_ids"
        assert top1000[key] == top1500[key]
        assert len(top1000[key]) == expected_count


@pytest.mark.parametrize("seed", [-1, 0, 41, 43, 124, 1000])
def test_split_preparation_rejects_unsupported_seeds(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be exactly"):
        splits.reference_paths(seed)


def _mock_binding(top_n: int) -> dict[str, str]:
    layout = datasets.dataset_layout(top_n)

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    return {
        "dataset_path": layout.dataset.as_posix(),
        "dataset_sha256": digest("temporary-test-dataset"),
        "metadata_path": layout.metadata.as_posix(),
        "metadata_sha256": digest("temporary-test-metadata"),
        "completion_marker_path": layout.completion_marker.as_posix(),
        "completion_marker_sha256": digest("temporary-test-marker"),
        "dataset_schema_version": "camels_temporal_sparse_v1",
        "graph_storage_mode": "sparse_edge_index",
        "logical_dataset_id": layout.logical_dataset_id,
        "source_manifest_policy": "full_sha256",
        "source_manifest_sha256": digest("temporary-test-source-manifest"),
        "target_table_path": splits.TARGET.as_posix(),
        "target_table_sha256": splits.TARGET_SHA256,
    }


@pytest.mark.parametrize("top_n", [500, 750])
@pytest.mark.parametrize("seed", [42, 123, 2025])
def test_bound_manifest_preserves_exact_reference_lists(top_n: int, seed: int) -> None:
    top1000, top1500 = splits.load_reference_pair(ROOT, seed)
    candidate = splits.create_bound_manifest(top1000, _mock_binding(top_n), top_n)
    splits.verify_preserved_ids(candidate, top1000, top1500, seed)
    assert candidate["counts"] == {"population": 1000, "train": 700, "val": 99, "test": 201, "unused": 0}
    assert candidate["graph_protocol_summary"]["top_n"] == top_n
    assert candidate["dataset_identity"] == candidate["dataset_binding"]["dataset_sha256"]


def test_bound_manifest_rejects_missing_dataset_identity() -> None:
    source, _ = splits.load_reference_pair(ROOT, 42)
    binding = _mock_binding(500)
    del binding["dataset_sha256"]
    with pytest.raises(RuntimeError, match="dataset_sha256"):
        splits.create_bound_manifest(source, binding, 500)


def test_binding_rejects_unvalidated_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="required input missing|does not exist|missing"):
        splits.build_dataset_binding(tmp_path, 500)


def test_preflight_does_not_create_manifests() -> None:
    expected = [splits.output_manifest_path(top_n, seed) for top_n in (500, 750) for seed in splits.SUPPORTED_SEEDS]
    assert all(not (ROOT / path).exists() for path in expected)


def test_no_random_resplitting_code_path() -> None:
    source = (ROOT / "scripts/validation/prepare_u1000_top500_top750_train700.py").read_text(encoding="utf-8")
    assert "random.shuffle" not in source
    assert "split_universes" not in source
    assert "load_reference_pair" in source


@pytest.mark.parametrize("top_n", [1000, 1500])
@pytest.mark.parametrize(
    ("model_key", "expected"),
    [
        (
            "static",
            {
                "model": "StaticGCNRegressor", "hidden_dim": 32, "num_layers": 3,
                "graph_pooling": "mean", "conv_type": "gcn", "dropout": 0.2,
                "batch_size": 8, "optimizer": "AdamW", "learning_rate": 0.001,
                "weight_decay": 1e-5, "epochs": 300, "patience": 40,
                "grad_clip_norm": 1.0, "checkpoint_criterion": "minimum_validation_mse",
                "target_normalization": "none",
            },
        ),
        (
            "evolve",
            {
                "model": "EvolveGCNHRegressor", "hidden_dim": 32, "num_layers": 2,
                "graph_pooling": "mean", "temporal_pooling": "mean", "head_type": "linear",
                "activation": "relu", "dropout": 0.2, "batch_size": 4,
                "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 1e-5,
                "epochs": 300, "patience": 40, "grad_clip_norm": 1.0,
                "checkpoint_criterion": "minimum_validation_mse", "target_normalization": "none",
            },
        ),
    ],
)
def test_verified_reference_model_configuration(top_n: int, model_key: str, expected: dict[str, object]) -> None:
    if top_n == 1000:
        prefix = ROOT / "configs/pilots"
    else:
        prefix = ROOT / "configs/production/u1000_top1500_training_scaling"
    if model_key == "static":
        filename = f"static_gcn_u1000_top{top_n}_sparse_train700_seed42_none_h32_l3_mean_mlp_final.json"
    else:
        filename = f"evolvegcn_h_u1000_top{top_n}_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear.json"
    config = json.loads((prefix / filename).read_text(encoding="utf-8"))
    for key, value in expected.items():
        assert config.get(key) == value
    assert config["scheduler"] == {
        "name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
        "patience": 10, "min_lr": 1e-6,
    }
    if model_key == "static":
        assert config["snapshot_selection"] == "exact_final_snapshot_index_minus_one"
    else:
        assert config["snapshots"] == 5


def test_analysis_protocol_and_output_isolation() -> None:
    assert analysis.HALO_COUNTS == (500, 750, 1000, 1500)
    assert analysis.SEEDS == (42, 123, 2025)
    assert analysis.OUTPUT_DIR == Path("reports/analysis/u1000_halo_count_scaling_train700")
    assert set(analysis.EXPECTED_OUTPUTS) == {
        "seed_level_results.csv", "halo_count_summary.csv", "paired_seed_differences.csv",
        "plotting_data.csv", "halo_count_scaling.png", "scientific_report.md", "validation.md",
    }
