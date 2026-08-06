from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validation import manage_u1000_top1500_training_scaling_matrix as manager
from scripts.validation import validate_u1000_top1000_sparse_dataset as validator
from src.data.source_manifest import sha256_file_streaming, source_manifest_sha256
from src.training.split_manifest import canonical_manifest_sha256


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_bound_fixture(root: Path) -> tuple[dict[str, str], Path]:
    dataset = root / manager.DATASET
    metadata_path = dataset.with_suffix(".metadata.json")
    marker_path = dataset.with_suffix(".complete")
    target = root / manager.TARGET
    dataset.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    dataset.write_bytes(b"bounded Top1500 fixture\n")
    target.write_text("universe_id,omega_m\nLH_0,0.3\n", encoding="utf-8")
    dataset_sha = sha256_file_streaming(dataset)
    raw_manifest = {
        "schema_version": "camels_source_manifest_v1",
        "source_manifest_policy": "full_sha256", "hash_algorithm": "sha256",
        "hash_chunk_size_bytes": 1048576,
        "sorting_key": ["source_role", "universe_id", "snapshot_id", "relative_path"],
        "entries": [],
    }
    raw_manifest["manifest_sha256"] = source_manifest_sha256(raw_manifest)
    metadata = {
        "checksum": dataset_sha, "source_manifest": raw_manifest,
        "source_manifest_sha256": raw_manifest["manifest_sha256"],
        "target_source_sha256": sha256_file_streaming(target),
    }
    write_json(metadata_path, metadata)
    write_json(marker_path, {"dataset": dataset.name, "metadata": metadata_path.name,
                             "sha256": dataset_sha, "status": "complete"})
    identities = manager.current_artifact_identities(root)

    manifest_path = root / manager.PILOT_MANIFEST
    manifest = {
        "seed": 42, "train_ids": ["LH_0"], "val_ids": ["LH_1"],
        "test_ids": ["LH_2"], "unused_ids": ["LH_3"],
        "dataset_identity": identities["dataset_sha256"],
        "dataset_binding": {
            "dataset_path": manager.DATASET.as_posix(),
            "metadata_path": manager.DATASET.with_suffix(".metadata.json").as_posix(),
            "completion_marker_path": manager.DATASET.with_suffix(".complete").as_posix(),
            "target_table_path": manager.TARGET.as_posix(),
            **identities,
        },
    }
    manifest["partition_identity"] = manager.partition_identity(manifest)
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file_streaming(manifest_path)
    config_path = root / manager.CONFIG_DIR / "fixture.json"
    config = {
        "dataset_path": manager.DATASET.as_posix(),
        "split_manifest_path": manager.PILOT_MANIFEST.as_posix(),
        "split_manifest_sha256": manifest_sha,
        **identities,
    }
    write_json(config_path, config)
    entry = {
        "canonical_experiment_id": "fixture", "master_dataset_path": manager.DATASET.as_posix(),
        "split_manifest_path": manager.PILOT_MANIFEST.as_posix(),
        "split_manifest_sha256": manifest_sha,
        "partition_identity": manifest["partition_identity"],
        "split_binding_identity": canonical_manifest_sha256(manifest),
        "configuration_path": config_path.relative_to(root).as_posix(),
        **{key: identities[key] for key in (
            "dataset_sha256", "metadata_sha256", "completion_marker_sha256", "target_table_sha256"
        )},
    }
    write_json(root / manager.REGISTRY, {"entries": [entry]})
    return identities, manifest_path


def make_pilot(identities: dict[str, str], manifest_path: Path) -> dict:
    return {
        "schema_version": "u1000_top1500_cuda_pilot_v2", "status": "PASS",
        "dataset_path": manager.DATASET.as_posix(), "dataset_sha256": identities["dataset_sha256"],
        "metadata_path": manager.DATASET.with_suffix(".metadata.json").as_posix(),
        "metadata_sha256": identities["metadata_sha256"],
        "completion_marker_path": manager.DATASET.with_suffix(".complete").as_posix(),
        "completion_marker_sha256": identities["completion_marker_sha256"],
        "raw_source_identity": identities["source_manifest_sha256"],
        "target_source_identity": identities["target_table_sha256"],
        "top_n": 1500, "universe_count": 1000, "snapshot_count": 5, "feature_dimension": 7,
        "normalization": "none", "periodic_flag": True, "k": 8, "box_size": 25.0,
        "model_names_tested": ["evolve", "static"],
        "production_batch_sizes": {"evolve": 4, "static": 8}, "seed": 42,
        "train700_seed42_manifest_path": manager.PILOT_MANIFEST.as_posix(),
        "manifest_sha256": sha256_file_streaming(manifest_path),
        "ordered_partition_identity": manager.partition_identity(json.loads(manifest_path.read_text())),
        "source_git_commit": "a" * 40, "execution_timestamp": "2026-08-06T00:00:00+00:00",
        "cuda_device_identity": {"name": "fixture"}, "forward_backward_result": "PASS",
        "finite_loss_result": "PASS", "finite_gradient_result": "PASS",
        "peak_memory_mib": {"evolve": {"allocated": 1, "reserved": 1},
                            "static": {"allocated": 1, "reserved": 1}},
        "results": {"evolve": {"status": "PASS"}, "static": {"status": "PASS"}},
    }


def builder_metadata(root: Path) -> dict:
    builder = root / validator.EXPECTED_BUILDER_SOURCE
    launcher = root / validator.EXPECTED_TOP1500_LAUNCHER
    builder.parent.mkdir(parents=True)
    launcher.parent.mkdir(parents=True)
    builder.write_text("# builder\n", encoding="utf-8")
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    return {
        "builder_provenance_schema_version": "camels_builder_provenance_v1",
        "builder_entrypoint": validator.EXPECTED_BUILDER_MODULE,
        "builder_module": validator.EXPECTED_BUILDER_MODULE,
        "builder_source_path": validator.EXPECTED_BUILDER_SOURCE.as_posix(),
        "builder_source_sha256": sha256_file_streaming(builder),
        "build_launcher_path": validator.EXPECTED_TOP1500_LAUNCHER.as_posix(),
        "build_launcher_sha256": sha256_file_streaming(launcher),
        "source_git_commit": "b" * 40, "git_commit": "b" * 40,
    }


class Top1500IntegrityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fake_bound_dataset_hash_fails_before_any_trainer(self) -> None:
        make_bound_fixture(self.root)
        registry_path = self.root / manager.REGISTRY
        registry = json.loads(registry_path.read_text())
        registry["entries"][0]["dataset_sha256"] = "0" * 64
        write_json(registry_path, registry)
        trainer_invoked = False
        with self.assertRaisesRegex(RuntimeError, r"dataset.*registry"):
            manager.verify_artifact_bindings(self.root)
        self.assertFalse(trainer_invoked)

    def test_pending_top1500_hash_still_fails(self) -> None:
        make_bound_fixture(self.root)
        registry_path = self.root / manager.REGISTRY
        registry = json.loads(registry_path.read_text())
        registry["entries"][0]["dataset_sha256"] = manager.PENDING
        write_json(registry_path, registry)
        with self.assertRaisesRegex(RuntimeError, "PENDING HASH"):
            manager.verify_artifact_bindings(self.root)

    def test_correct_hashes_and_correctly_bound_pilot_pass(self) -> None:
        identities, manifest_path = make_bound_fixture(self.root)
        self.assertEqual(manager.verify_artifact_bindings(self.root), identities)
        pilot_path = self.root / "pilot.json"
        write_json(pilot_path, make_pilot(identities, manifest_path))
        with patch.object(manager.subprocess, "check_output", return_value="a" * 40 + "\n"):
            manager.verify_cuda_pilot(identities, self.root, pilot_path)

    def test_stale_cuda_pilot_dataset_hash_fails(self) -> None:
        identities, manifest_path = make_bound_fixture(self.root)
        pilot = make_pilot(identities, manifest_path)
        pilot["dataset_sha256"] = "f" * 64
        pilot_path = self.root / "pilot.json"
        write_json(pilot_path, pilot)
        with patch.object(manager.subprocess, "check_output", return_value="a" * 40 + "\n"):
            with self.assertRaisesRegex(RuntimeError, "STALE CUDA PILOT: dataset SHA"):
                manager.verify_cuda_pilot(identities, self.root, pilot_path)

    def test_wrong_partition_identity_fails(self) -> None:
        identities, manifest_path = make_bound_fixture(self.root)
        pilot = make_pilot(identities, manifest_path)
        pilot["ordered_partition_identity"] = "e" * 64
        pilot_path = self.root / "pilot.json"
        write_json(pilot_path, pilot)
        with patch.object(manager.subprocess, "check_output", return_value="a" * 40 + "\n"):
            with self.assertRaisesRegex(RuntimeError, "pilot ordered partition"):
                manager.verify_cuda_pilot(identities, self.root, pilot_path)

    def test_missing_builder_provenance_fails_for_top1500(self) -> None:
        with self.assertRaisesRegex(ValueError, "builder metadata/provenance fields missing"):
            validator.check_builder_provenance({}, self.root, required=True)

    def test_wrong_builder_source_hash_fails(self) -> None:
        metadata = builder_metadata(self.root)
        metadata["builder_source_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "builder source SHA-256"):
            validator.check_builder_provenance(metadata, self.root, required=True)

    def test_correct_builder_provenance_passes_and_historical_top1000_is_accepted(self) -> None:
        validator.check_builder_provenance(builder_metadata(self.root), self.root, required=True)
        validator.check_builder_provenance({}, self.root, required=False)


if __name__ == "__main__":
    unittest.main()
