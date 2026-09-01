from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.validation import prepare_u1000_top1500_knn_training as knn
from scripts.validation import prepare_u1000_top1500_radius_training as radius


class RadiusPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = {
            "dataset_path": radius.DATASET.as_posix(),
            "dataset_sha256": radius.DATASET_SHA256,
            "metadata_path": radius.DATASET.with_suffix(".metadata.json").as_posix(),
            "metadata_sha256": radius.METADATA_SHA256,
            "completion_marker_path": radius.DATASET.with_suffix(".complete").as_posix(),
            "completion_marker_sha256": radius.MARKER_SHA256,
            "logical_dataset_id": radius.LOGICAL_DATASET_ID,
            "source_manifest_sha256": knn.SOURCE_MANIFEST_SHA256,
            "target_table_sha256": knn.TARGET_SHA256,
        }

    def test_radius_splits_preserve_every_ordered_partition(self) -> None:
        for seed in radius.SEEDS:
            with self.subTest(seed=seed):
                control = knn.read_json(radius.ROOT / radius.control_split_relative_path(seed))
                candidate = radius.make_split(control, self.binding, seed=seed)
                for partition in ("train", "val", "test", "unused"):
                    self.assertEqual(candidate[f"{partition}_ids"], control[f"{partition}_ids"])
                self.assertEqual(
                    knn.differing_paths(candidate, control), radius.SPLIT_ALLOWED_DIFFERENCES,
                )

    def test_radius_configs_have_only_frozen_allowed_differences(self) -> None:
        for seed in radius.SEEDS:
            with self.subTest(seed=seed):
                control = knn.read_json(radius.ROOT / radius.control_config_relative_path(seed))
                split_path = radius.ROOT / radius.split_relative_path(seed)
                with patch.object(knn, "sha256_file", return_value="0" * 64):
                    candidate = radius.make_config(control, self.binding, split_path, seed=seed)
                expected_differences = set(radius.CONFIG_ALLOWED_DIFFERENCES)
                self.assertEqual(knn.differing_paths(candidate, control), expected_differences)
                self.assertEqual(candidate["seed"], seed)
                self.assertEqual(candidate["hidden_dim"], 32)
                self.assertEqual(candidate["num_layers"], 3)
                self.assertEqual(candidate["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
