from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.models.evolvegcn_o import EvolveGCNORegressor, count_parameters
from src.training.train_evolvegcn_o import FROZEN_O_VALUES, build_model
from src.training.temporal_architecture_common import load_and_validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/production/u1000_top1500_evolvegcn_o"


class EvolveGCNOPreparationTests(unittest.TestCase):
    def test_exact_three_configs(self):
        paths = sorted(CONFIG_DIR.glob("*.json"))
        self.assertEqual(len(paths), 3)
        configs = [json.loads(path.read_text()) for path in paths]
        self.assertEqual(sorted(item["seed"] for item in configs), [42, 123, 2025])
        for path, config in zip(paths, configs):
            loaded = load_and_validate_config(
                path, "EvolveGCNORegressor", frozen_values=FROZEN_O_VALUES
            )
            self.assertEqual(count_parameters(build_model(loaded)), 11_527)

    def test_h_matched_protocol_fields(self):
        control_root = Path("/home/ml/thesis-camels/experiments")
        fields = {
            "batch_size", "epochs", "patience", "learning_rate", "weight_decay",
            "hidden_dim", "num_layers", "dropout", "activation", "temporal_pooling",
            "graph_pooling", "head_type", "add_self_loops", "grad_clip_norm",
        }
        for seed in (42, 123, 2025):
            o = json.loads((CONFIG_DIR / f"evolvegcn_o_seed{seed}.json").read_text())
            h_path = control_root / (
                f"evolvegcn_h_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_"
                "mean_temporal_mean_linear/config.json"
            )
            h = json.loads(h_path.read_text())
            self.assertEqual({key: o[key] for key in fields}, {key: h[key] for key in fields})
            self.assertEqual(o["split_manifest_path"], h["split_source"])
            self.assertEqual(o["split_manifest_sha256"], h["split_manifest_sha256"])

    def test_public_model_import(self):
        self.assertIsNotNone(EvolveGCNORegressor)


if __name__ == "__main__":
    unittest.main()
