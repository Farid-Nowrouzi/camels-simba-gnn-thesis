from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.split_manifest import (
    canonical_manifest_sha256,
    load_split_manifest,
    ordered_id_hash,
)
from src.training.train_evolvegcn_h import create_loaders as create_evolve_loaders
from src.training.train_evolvegcn_h import train_evolvegcn_h
from src.training.train_static_gcn import convert_temporal_final_snapshot_to_static
from src.training.train_static_gcn import create_loaders as create_static_loaders
from src.training.train_static_gcn import train_static_gcn


class SplitManifestSeedBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ids = [f"LH_{index}" for index in range(6)]
        self.temporal = {}
        edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        for index, universe_id in enumerate(self.ids):
            x = torch.full((2, 2), float(index))
            mask = torch.ones(2, 1)
            self.temporal[universe_id] = {
                "graph_storage": "sparse_edge_index",
                "Nodes_list": [x for _ in range(5)],
                "edge_index_list": [edge for _ in range(5)],
                "edge_weight_list": None,
                "mask_list": [mask for _ in range(5)],
                "target": torch.tensor(0.1 + index * 0.01),
                "snapshots": [{"snapshot_value": value} for value in range(5)],
            }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, seed=42, include_seed=True) -> Path:
        path = self.root / "split.json"
        manifest = {
            "dataset_identity": "tiny-v1",
            "train_ids": ["LH_2", "LH_0"],
            "val_ids": ["LH_1"],
            "test_ids": ["LH_3"],
            "counts": {"train": 2, "val": 1, "test": 1},
            "split_hashes": {
                "train": ordered_id_hash(["LH_2", "LH_0"]),
                "val": ordered_id_hash(["LH_1"]),
                "test": ordered_id_hash(["LH_3"]),
            },
            "parent_training_subset": ["LH_2"],
            "target_summaries": {},
            "creation_metadata": {"fixture": True},
        }
        if include_seed:
            manifest["seed"] = seed
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def loader_arguments(self, path: Path):
        return dict(
            seed=42, batch_size=2, train_ratio=.5, val_ratio=.25, test_ratio=.25,
            split_manifest_path=path, dataset_identity="tiny-v1",
        )

    def test_matching_seed_passes_for_evolve_and_static(self) -> None:
        path = self.write_manifest(42)
        evolve = create_evolve_loaders(self.temporal, **self.loader_arguments(path))
        static = create_static_loaders(
            convert_temporal_final_snapshot_to_static(self.temporal), **self.loader_arguments(path)
        )
        self.assertEqual(evolve[3:6], (["LH_2", "LH_0"], ["LH_1"], ["LH_3"]))
        self.assertEqual(static[3:6], (["LH_2", "LH_0"], ["LH_1"], ["LH_3"]))

    def test_evolve_mismatch_reports_both_seeds_before_loader_creation(self) -> None:
        path = self.write_manifest(123)
        with self.assertRaisesRegex(ValueError, r"trainer seed=42, manifest seed=123.*matching trainer seed"):
            create_evolve_loaders(self.temporal, **self.loader_arguments(path))

    def test_static_mismatch_reports_both_seeds_before_loader_creation(self) -> None:
        path = self.write_manifest(123)
        with self.assertRaisesRegex(ValueError, r"trainer seed=42, manifest seed=123.*correct split manifest"):
            create_static_loaders(
                convert_temporal_final_snapshot_to_static(self.temporal), **self.loader_arguments(path)
            )

    def test_missing_and_noninteger_manifest_seed_are_rejected(self) -> None:
        missing = self.write_manifest(include_seed=False)
        with self.assertRaisesRegex(KeyError, "missing required.*seed"):
            load_split_manifest(missing, self.ids, "tiny-v1", expected_seed=42)
        malformed = self.write_manifest(seed="42")
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            load_split_manifest(malformed, self.ids, "tiny-v1", expected_seed=42)

    def test_no_manifest_preserves_historical_internal_split(self) -> None:
        arguments = dict(seed=42, batch_size=2, train_ratio=.5, val_ratio=.25, test_ratio=.25)
        first = create_evolve_loaders(self.temporal, **arguments)
        second = create_evolve_loaders(self.temporal, **arguments)
        self.assertEqual(first[3:6], second[3:6])
        static = create_static_loaders(convert_temporal_final_snapshot_to_static(self.temporal), **arguments)
        self.assertEqual(tuple(map(len, first[3:6])), tuple(map(len, static[3:6])))

    def test_rejected_mismatch_creates_no_evolve_or_static_output(self) -> None:
        path = self.write_manifest(123)
        output_root = self.root / "experiments"
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            train_evolvegcn_h(
                self.root / "does-not-need-to-exist.pt", "evolve_rejected",
                output_root=output_root, seed=42, split_manifest_path=path,
                dataset_identity="tiny-v1", epochs=1,
            )
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            train_static_gcn(
                self.root / "does-not-need-to-exist.pt", "static_rejected",
                output_root=output_root, seed=42, split_manifest_path=path,
                dataset_identity="tiny-v1", epochs=1,
            )
        self.assertFalse(output_root.exists())

    def test_production_fields_validate_unused_coverage_and_canonical_identity(self) -> None:
        path = self.write_manifest(42)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["unused_ids"] = ["LH_4", "LH_5"]
        manifest["counts"]["unused"] = 2
        manifest["split_hashes"]["unused"] = ordered_id_hash(manifest["unused_ids"])
        manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = load_split_manifest(path, self.ids, "tiny-v1", expected_seed=42)
        self.assertEqual(loaded["unused_ids"], ["LH_4", "LH_5"])

        manifest["unused_ids"] = ["LH_4", "LH_4"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            load_split_manifest(path, self.ids, "tiny-v1", expected_seed=42)

    def test_canonical_identity_ignores_creation_metadata_only(self) -> None:
        manifest = {"seed": 42, "creation_metadata": {"timestamp": "first"}, "train_ids": ["LH_0"]}
        first = canonical_manifest_sha256(manifest)
        manifest["creation_metadata"]["timestamp"] = "second"
        self.assertEqual(first, canonical_manifest_sha256(manifest))
        manifest["train_ids"].append("LH_1")
        self.assertNotEqual(first, canonical_manifest_sha256(manifest))


if __name__ == "__main__":
    unittest.main()
