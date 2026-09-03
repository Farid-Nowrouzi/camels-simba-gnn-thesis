from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.static_gcn import StaticGCNRegressor
from src.models.temporal_gcn_encoder import SharedGCNSnapshotEncoder
from src.training.sparse_batch import collate_sparse_static
from src.training.temporal_architecture_common import load_and_validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/production/u1000_top1500_temporal_architectures"


class TemporalPreparationTests(unittest.TestCase):
    def test_common_encoder_feature_dimension_and_batch_pooling(self):
        encoder = SharedGCNSnapshotEncoder(dropout=0.0).eval()
        graphs = [
            {"x": torch.zeros(2, 7), "edge_index": torch.tensor([[0, 1], [1, 0]]), "mask": torch.ones(2, 1)},
            {"x": torch.ones(5, 7), "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]), "mask": torch.ones(5, 1)},
        ]
        batch = collate_sparse_static(graphs)
        pooled = encoder(batch)
        self.assertEqual(tuple(pooled.shape), (2, 32))
        self.assertFalse(torch.equal(pooled[0], pooled[1]))
        with self.assertRaisesRegex(ValueError, "Expected sparse x"):
            encoder(dict(batch, x=torch.zeros(7, 8)))

    def test_exactly_six_frozen_configs_and_shared_manifests(self):
        paths = sorted(CONFIG_DIR.glob("*.json"))
        self.assertEqual(len(paths), 6)
        configs = [json.loads(path.read_text()) for path in paths]
        for seed in (42, 123, 2025):
            pair = [item for item in configs if item["seed"] == seed]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["split_manifest_path"], pair[1]["split_manifest_path"])
            self.assertEqual(pair[0]["split_manifest_sha256"], pair[1]["split_manifest_sha256"])
        for config, path in zip(configs, paths):
            load_and_validate_config(path, config["model"])

    def test_historical_model_imports_and_protocol_records(self):
        self.assertIsNotNone(StaticGCNRegressor)
        self.assertIsNotNone(EvolveGCNHRegressor)
        for name in (
            "u1000_top1500_gcn_gru_temporal_protocol.json",
            "u1000_top1500_gcn_temporal_transformer_protocol.json",
        ):
            record = json.loads((ROOT / "reports/experiment_registry" / name).read_text())
            self.assertFalse(record["production_training_performed"])


if __name__ == "__main__":
    unittest.main()
