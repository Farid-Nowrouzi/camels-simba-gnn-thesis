from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.data.build_temporal_sequences import build_temporal_dataset
from src.data.camels_graph_utils import GRAPH_STORAGE_SPARSE
from src.data.source_manifest import (
    HASH_CHUNK_SIZE,
    SOURCE_MANIFEST_POLICY_FULL,
    build_full_source_manifest,
    classify_source_provenance,
    sha256_file_streaming,
    source_manifest_sha256,
    verify_full_source_manifest,
)


class FullSourceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.targets = self.root / "targets"
        self.raw.mkdir()
        self.targets.mkdir()
        self.catalogues = []
        for universe, snapshot in ((1, "0.20000"), (0, "1.00000"), (0, "0.20000")):
            path = self.raw / f"LH_{universe}_hlist_{snapshot}.list"
            path.write_text(f"# fixture\n{universe} {snapshot}\n", encoding="utf-8")
            self.catalogues.append(path)
        self.target = self.targets / "omega.csv"
        self.target.write_text("universe_id,omega_m\nLH_0,0.2\nLH_1,0.3\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, paths=None, target=True):
        return build_full_source_manifest(
            self.catalogues if paths is None else paths,
            self.raw,
            self.target if target else None,
            self.targets,
            require_target=target,
        )

    @staticmethod
    def write_builder_catalogue(path: Path, snapshot_index: int) -> None:
        lines = ["# full-source manifest integration fixture"]
        for row_index in range(6):
            values = [0.0] * 23
            values[1] = row_index
            values[10] = 1000.0 - row_index
            values[17:23] = [row_index + snapshot_index * 0.1 + offset for offset in range(6)]
            lines.append(" ".join(str(value) for value in values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_stable_streaming_digest_and_canonical_identity(self) -> None:
        first = self.build()
        second = self.build()
        self.assertEqual(sha256_file_streaming(self.catalogues[0]), sha256_file_streaming(self.catalogues[0]))
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first["manifest_sha256"], source_manifest_sha256(first))
        self.assertEqual(first["hash_chunk_size_bytes"], HASH_CHUNK_SIZE)
        self.assertEqual(verify_full_source_manifest(first)["verification_result"], "verified_full_sha256")

    def test_different_input_order_has_identical_entries_and_identity(self) -> None:
        first = self.build(self.catalogues)
        second = self.build(reversed(self.catalogues))
        self.assertEqual(first["entries"], second["entries"])
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        catalogue_keys = [
            (entry["universe_id"], entry["snapshot_id"])
            for entry in first["entries"] if entry["source_role"] == "halo_catalogue"
        ]
        self.assertEqual(catalogue_keys, [("LH_0", "0.20000"), ("LH_0", "1.00000"), ("LH_1", "0.20000")])

    def test_content_and_same_size_mutations_are_rejected(self) -> None:
        manifest = self.build()
        path = self.catalogues[0]
        original = path.read_bytes()
        path.write_bytes(b"X" + original[1:])
        self.assertEqual(path.stat().st_size, len(original))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_full_source_manifest(manifest)

    def test_same_size_and_restored_mtime_mutation_is_rejected(self) -> None:
        manifest = self.build()
        path = self.catalogues[1]
        stat = path.stat()
        original = path.read_bytes()
        path.write_bytes(original[:-1] + (b"X" if original[-1:] != b"X" else b"Y"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_full_source_manifest(manifest)

    def test_target_omission_is_rejected_for_production_policy(self) -> None:
        manifest = self.build(target=False)
        self.assertEqual(manifest["target_source_count"], 0)
        with self.assertRaisesRegex(ValueError, "exactly one target table"):
            verify_full_source_manifest(manifest, require_target=True)

    def test_target_mutation_invalidates_verification_and_new_identity(self) -> None:
        manifest = self.build()
        original_identity = manifest["manifest_sha256"]
        content = self.target.read_bytes()
        self.target.write_bytes(content.replace(b"0.2", b"0.4"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_full_source_manifest(manifest)
        rebuilt = self.build()
        self.assertNotEqual(rebuilt["manifest_sha256"], original_identity)
        target_entry = next(entry for entry in rebuilt["entries"] if entry["source_role"] == "target_table")
        self.assertEqual(target_entry["target_column"], "omega_m")
        self.assertEqual(target_entry["universe_id_column"], "universe_id")

    def test_duplicate_paths_and_duplicate_catalogue_identity_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate source relative path"):
            verify_full_source_manifest(self.build(self.catalogues + [self.catalogues[0]]))
        manifest = self.build()
        duplicate = copy.deepcopy(manifest["entries"][0])
        duplicate["relative_path"] = f"duplicate/{duplicate['relative_path']}"
        (self.raw / "duplicate").mkdir()
        (self.raw / duplicate["relative_path"]).write_bytes(
            (self.raw / manifest["entries"][0]["relative_path"]).read_bytes()
        )
        manifest["entries"].insert(1, duplicate)
        manifest["entry_count"] += 1
        manifest["catalogue_count"] += 1
        manifest["manifest_sha256"] = source_manifest_sha256(manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate halo catalogue universe/snapshot"):
            verify_full_source_manifest(manifest)

    def test_invalid_role_and_top_level_hash_are_rejected(self) -> None:
        manifest = self.build()
        manifest["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Top-level canonical"):
            verify_full_source_manifest(manifest)
        manifest = self.build()
        manifest["entries"][0]["source_role"] = "unknown"
        manifest["manifest_sha256"] = source_manifest_sha256(manifest)
        with self.assertRaisesRegex((ValueError, KeyError), "unknown"):
            verify_full_source_manifest(manifest)

    def test_legacy_metadata_is_readable_but_unverified(self) -> None:
        legacy = {"source_manifest_hash": "stat-based-value", "dataset_schema_version": "legacy_dense_v1"}
        encoded = json.loads(json.dumps(legacy))
        self.assertEqual(classify_source_provenance(encoded), "legacy_unverified_stat_only")
        self.assertNotEqual(classify_source_provenance(encoded), "verified_full_sha256")
        self.assertEqual(SOURCE_MANIFEST_POLICY_FULL, "full_sha256")

    def test_sparse_builder_persists_verified_catalogue_and_target_identity(self) -> None:
        raw = self.root / "builder_raw"
        raw.mkdir()
        for index, snapshot in enumerate(("0.20000", "0.25000", "0.51209", "0.75065", "1.00000")):
            self.write_builder_catalogue(raw / f"LH_0_hlist_{snapshot}.list", index)
        target = self.root / "builder_targets.csv"
        target.write_text("universe_id,omega_m\nLH_0,0.3\n", encoding="utf-8")
        output = self.root / "builder_sparse.pt"
        build_temporal_dataset(
            raw, output, 1, num_snapshots=5, num_nodes=4, normalization="none",
            graph_mode="knn", k=2, targets_csv=target,
            graph_storage=GRAPH_STORAGE_SPARSE,
        )
        metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_manifest_policy"], "full_sha256")
        self.assertEqual(metadata["source_manifest_entry_count"], 6)
        self.assertEqual(metadata["source_manifest_catalogue_count"], 5)
        self.assertEqual(metadata["source_manifest_target_source_count"], 1)
        self.assertEqual(metadata["target_source_relative_path"], target.name)
        self.assertEqual(metadata["target_source_sha256"], sha256_file_streaming(target))
        self.assertTrue(metadata["source_manifest_verification"]["verified"])
        self.assertEqual(metadata["source_manifest_sha256"], metadata["source_manifest"]["manifest_sha256"])

    def test_new_sparse_builder_rejects_legacy_stat_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "require source_manifest_policy=full_sha256"):
            build_temporal_dataset(
                self.raw, self.root / "rejected.pt", 1, num_nodes=4,
                dummy_target=0.3, graph_storage=GRAPH_STORAGE_SPARSE,
                source_manifest_policy="legacy_stat_only",
            )
        self.assertFalse((self.root / "rejected.pt").exists())


if __name__ == "__main__":
    unittest.main()
