"""CNN3D metadata contract for the later registry integration."""

from __future__ import annotations

from dataclasses import replace
import unittest

from src.data.halo_voxelization import EXPERIMENT_1_SPEC, representation_fingerprint
from src.evaluation import run_modern_cnn3d as runner


class CNNRegistryMetadataTests(unittest.TestCase):
    def test_family_config_carries_unambiguous_grid_metadata(self) -> None:
        family = runner.load_family()
        voxel = family["voxelization"]
        self.assertEqual(family["experiment_family"], "cnn3d")
        self.assertEqual(family["representation_family"], "grid")
        self.assertFalse(family["uses_graph_edges"])
        self.assertEqual(voxel["grid_resolution"], 32)
        self.assertEqual(voxel["channel_count"], 5)
        self.assertEqual(voxel["deposition_method"], "cic")
        self.assertEqual(family["architecture"]["periodic_padding"], "circular")
        self.assertEqual(family["dataset"]["source_feature_order"], list(EXPERIMENT_1_SPEC.source_feature_order))
        self.assertEqual(family["architecture"]["expected_trainable_parameters"], 20_017)
        self.assertEqual(family["dataset"]["top_n"], 1500)
        self.assertEqual(family["dataset"]["snapshot"], "final")
        self.assertEqual(family["seeds"], [42, 123, 2025])

    def test_representation_fingerprint_separates_grid_definitions(self) -> None:
        family = runner.load_family()
        self.assertEqual(family["voxelization"]["representation_fingerprint"], representation_fingerprint(EXPERIMENT_1_SPEC))
        self.assertNotEqual(
            representation_fingerprint(EXPERIMENT_1_SPEC),
            representation_fingerprint(replace(EXPERIMENT_1_SPEC, resolution=16)),
        )

    def test_smoke_configuration_is_explicitly_non_scientific(self) -> None:
        family = runner.load_family()
        job = runner.expand_family_jobs(family)[0]
        config = runner.run_config(family, job, None, __import__("torch").device("cpu"), smoke=True)
        self.assertFalse(config["scientific_run"])
        self.assertIn("smoke", str(runner.SMOKE_PATH))


if __name__ == "__main__":
    unittest.main()
