from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.validation import manage_u1000_top1500_training_scaling_matrix as manager
from scripts.validation import validate_u1000_top1000_sparse_dataset as validator
from scripts.validation import validate_u1000_top1500_knn_variants as variant_validator
from src.data.source_manifest import sha256_file_streaming, source_manifest_sha256
from src.training.split_manifest import canonical_manifest_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP1500_LAUNCHER = PROJECT_ROOT / "scripts/production/run_u1000_top1500_sparse_build.sh"
ESTABLISHED_PYTHON = "/home/ml/thesis-camels/envs/camels-gnn/bin/python"


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


def committed_builder_metadata(root: Path) -> dict:
    metadata = builder_metadata(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", metadata["builder_source_path"], metadata["build_launcher_path"]],
                   cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture provenance"], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata["source_git_commit"] = commit
    metadata["git_commit"] = commit
    return metadata


def launcher_resolution(
    *arguments: str,
    launcher: Path = TOP1500_LAUNCHER,
    project_root: Path = PROJECT_ROOT,
    python_override: str | None = ESTABLISHED_PYTHON,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if python_override is None:
        environment.pop("CAMELS_PYTHON", None)
    else:
        environment["CAMELS_PYTHON"] = python_override
    return subprocess.run(
        ["bash", str(launcher), "--resolve-only", *arguments],
        cwd=project_root, text=True, capture_output=True, check=False, env=environment,
    )


def parse_resolution(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


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
        metadata = committed_builder_metadata(self.root)
        metadata["builder_source_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "recorded Git commit blob"):
            validator.check_builder_provenance(metadata, self.root, required=True)

    def test_wrong_launcher_hash_fails(self) -> None:
        metadata = committed_builder_metadata(self.root)
        metadata["build_launcher_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "recorded Git commit blob"):
            validator.check_builder_provenance(metadata, self.root, required=True)

    def test_correct_builder_provenance_passes_and_historical_top1000_is_accepted(self) -> None:
        metadata = committed_builder_metadata(self.root)
        validator.check_builder_provenance(metadata, self.root, required=True)
        (self.root / metadata["build_launcher_path"]).write_text("#!/bin/sh\n# later revision\n")
        validator.check_builder_provenance(metadata, self.root, required=True)
        validator.check_builder_provenance({}, self.root, required=False)

    def test_wrong_provenance_commit_cannot_validate(self) -> None:
        metadata = committed_builder_metadata(self.root)
        metadata["source_git_commit"] = metadata["git_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "recorded provenance blob is unavailable"):
            validator.check_builder_provenance(metadata, self.root, required=True)

    def test_graph_source_commit_identity_is_fail_closed(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Fixture"], cwd=self.root, check=True)
        graph_source = self.root / validator.EXPECTED_GRAPH_SOURCE
        graph_source.parent.mkdir(parents=True)
        graph_source.write_text("# authoritative graph mathematics\n", encoding="utf-8")
        subprocess.run(["git", "add", validator.EXPECTED_GRAPH_SOURCE.as_posix()],
                       cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "k8 graph"], cwd=self.root, check=True)
        k8_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
        ).strip()
        authoritative_hash = sha256_file_streaming(graph_source)

        note = self.root / "note.txt"
        note.write_text("variant commit without graph changes\n", encoding="utf-8")
        subprocess.run(["git", "add", "note.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "matched variant"], cwd=self.root, check=True)
        matched_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
        ).strip()
        variant_validator.require_matching_graph_source(
            self.root, matched_commit, k8_commit, authoritative_sha256=authoritative_hash,
        )

        graph_source.write_text("# changed graph mathematics\n", encoding="utf-8")
        subprocess.run(["git", "add", validator.EXPECTED_GRAPH_SOURCE.as_posix()],
                       cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "changed graph"], cwd=self.root, check=True)
        changed_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
        ).strip()
        with self.assertRaisesRegex(ValueError, "differs from the authoritative"):
            variant_validator.require_matching_graph_source(
                self.root, changed_commit, k8_commit, authoritative_sha256=authoritative_hash,
            )
        with self.assertRaisesRegex(ValueError, "recorded provenance blob is unavailable"):
            variant_validator.require_matching_graph_source(
                self.root, "0" * 40, k8_commit, authoritative_sha256=authoritative_hash,
            )

    def test_missing_graph_source_at_recorded_commit_fails(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Fixture"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-qm", "missing graph source"],
                       cwd=self.root, check=True)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
        ).strip()
        with self.assertRaisesRegex(ValueError, "recorded provenance blob is unavailable"):
            variant_validator.require_matching_graph_source(
                self.root, commit, commit, authoritative_sha256="a" * 64,
            )

    def test_cross_variant_tensor_dtype_identity_is_explicit(self) -> None:
        cases = (
            ("target", torch.tensor([1.0], dtype=torch.float64),
             torch.tensor([1.0], dtype=torch.float32)),
            ("node features", torch.ones((2, 7), dtype=torch.float64),
             torch.ones((2, 7), dtype=torch.float32)),
            ("node mask", torch.ones((2, 1), dtype=torch.float64),
             torch.ones((2, 1), dtype=torch.float32)),
        )
        for label, candidate, control in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, rf"{label}: dtype changed; expected .*actual"):
                    variant_validator.require_same_tensor(candidate, control, label=label)

    def test_k_dependent_preprocessing_version_is_exact(self) -> None:
        for k in (4, 6, 8, 12):
            expected = variant_validator.expected_preprocessing_version(k)
            self.assertIn(f"knn_k{k}_box25", expected)
            variant_validator.require_preprocessing_version(expected, k=k, label="fixture")
            with self.assertRaisesRegex(ValueError, "wrong k-dependent preprocessing version"):
                variant_validator.require_preprocessing_version(
                    expected.replace(f"knn_k{k}_", "knn_k99_"), k=k, label="fixture",
                )

    def test_snapshot_source_path_requires_full_stored_identity(self) -> None:
        variant_validator.require_same_snapshot_path(
            "data/raw/A/snapshot.txt", "data/raw/A/snapshot.txt", label="fixture",
        )
        with self.assertRaisesRegex(ValueError, "stored source catalogue path changed"):
            variant_validator.require_same_snapshot_path(
                "data/raw/B/snapshot.txt", "data/raw/A/snapshot.txt", label="fixture",
            )

    def test_launcher_k_resolution_is_distinct_and_backward_compatible(self) -> None:
        default_result = launcher_resolution()
        explicit_k8_result = launcher_resolution("8")
        self.assertEqual(default_result.returncode, 0, default_result.stderr)
        self.assertEqual(explicit_k8_result.returncode, 0, explicit_k8_result.stderr)
        default = parse_resolution(default_result.stdout)
        explicit_k8 = parse_resolution(explicit_k8_result.stdout)
        self.assertEqual(default, explicit_k8)
        self.assertEqual(default["K"], "8")
        self.assertEqual(default["VALIDATOR"],
                         "scripts/validation/validate_u1000_top1500_sparse_dataset.py")
        self.assertEqual(
            default["OUTPUT_FILE"],
            "data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/"
            "camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt",
        )

        outputs = {8: default["OUTPUT_FILE"]}
        for k in (4, 6, 12):
            result = launcher_resolution(str(k))
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved = parse_resolution(result.stdout)
            self.assertEqual(resolved["K"], str(k))
            self.assertEqual(resolved["GRAPH_MODE"], "knn")
            self.assertEqual(resolved["TOP_N"], "1500")
            self.assertIn(f"periodic_knn_k{k}_sparse", resolved["OUTPUT_FILE"])
            self.assertIn(f"knn_k{k}_box25_sparse_v1", resolved["LOGICAL_DATASET_ID"])
            self.assertIn(f"--k {k}", resolved["BUILDER_INVOCATION"])
            self.assertEqual(
                resolved["VALIDATOR"],
                "scripts/validation/validate_u1000_top1500_knn_variants.py",
            )
            outputs[k] = resolved["OUTPUT_FILE"]
        self.assertEqual(len(set(outputs.values())), 4)

        invalid = launcher_resolution("5")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("Unsupported argument or k value: 5", invalid.stderr)

    def test_launcher_interpreter_resolution_and_dirty_build_refusal(self) -> None:
        selected = launcher_resolution("4")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(parse_resolution(selected.stdout)["PYTHON"], ESTABLISHED_PYTHON)

        invalid = launcher_resolution("4", python_override=str(self.root / "not-executable"))
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("set CAMELS_PYTHON explicitly", invalid.stderr)

        fixture_launcher = self.root / "scripts/production/run_u1000_top1500_sparse_build.sh"
        fixture_launcher.parent.mkdir(parents=True)
        fixture_launcher.write_text(TOP1500_LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
        local_python = self.root / "envs/camels-gnn/bin/python"
        local_python.parent.mkdir(parents=True)
        local_python.write_text("#!/bin/sh\n", encoding="utf-8")
        local_python.chmod(0o755)
        local = launcher_resolution(
            "4", launcher=fixture_launcher, project_root=self.root, python_override=None,
        )
        self.assertEqual(local.returncode, 0, local.stderr)
        self.assertEqual(parse_resolution(local.stdout)["PYTHON"], str(local_python))

        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Fixture"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "scripts/production/run_u1000_top1500_sparse_build.sh",
                        "envs/camels-gnn/bin/python"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "clean launcher fixture"],
                       cwd=self.root, check=True)
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.root, text=True,
            ),
            "",
        )

        k4_directory = self.root / "data/processed/temporal_1000u_none_top1500_periodic_knn_k4_sparse"
        self.assertFalse(k4_directory.exists())
        environment = os.environ.copy()
        environment.pop("CAMELS_PYTHON", None)
        sentinel = self.root / f".dirty-worktree-sentinel-{uuid.uuid4().hex}"
        self.assertFalse(sentinel.exists())
        sentinel.write_text("test-owned dirty-worktree sentinel\n", encoding="utf-8")
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.root, text=True,
            )
            self.assertIn(sentinel.name, status)
            refused = subprocess.run(
                ["bash", str(fixture_launcher), "4"], cwd=self.root,
                text=True, capture_output=True, check=False, env=environment,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("production builds require a clean, reviewed Git worktree", refused.stderr)
            self.assertFalse(k4_directory.exists())
        finally:
            sentinel.unlink()
        self.assertFalse(sentinel.exists())
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.root, text=True,
            ),
            "",
        )
        self.assertFalse(k4_directory.exists())

    def test_variant_validator_names_and_supported_k_are_closed(self) -> None:
        for k in (4, 6, 12):
            path = variant_validator.variant_relative_path(k).as_posix()
            self.assertIn(f"periodic_knn_k{k}_sparse", path)
            self.assertIn(f"knn_k{k}_box25_sparse_v1", variant_validator.logical_dataset_id(k))
        with self.assertRaisesRegex(ValueError, "unsupported k=8"):
            variant_validator.variant_relative_path(8)


if __name__ == "__main__":
    unittest.main()
