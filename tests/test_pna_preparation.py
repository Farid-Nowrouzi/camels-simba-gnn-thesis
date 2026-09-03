from __future__ import annotations

import unittest

from src.models.static_gcn import StaticGCNRegressor, count_parameters
from src.models.static_pna import StaticPNARegressor
from scripts.validation.prepare_u1000_top1500_pna_training import (
    DATASET_SHA,
    EXPECTED_SPLIT_SHAS,
    PNA_FIELDS,
    PROTOCOL,
    ROOT,
    SEEDS,
    config_differences,
    config_path,
    control_config_path,
    experiment_name,
    histogram_content_sha256,
    histogram_path,
    ordered_id_hash,
    protocol_document,
    read_json,
    split_path,
)


class PNAPreparationTests(unittest.TestCase):
    def test_histograms_are_training_only_and_internally_complete(self) -> None:
        for seed in SEEDS:
            split = read_json(ROOT / split_path(seed))
            record = read_json(ROOT / histogram_path(seed))
            self.assertEqual(record["seed"], seed)
            self.assertEqual(record["dataset_sha256"], DATASET_SHA)
            self.assertEqual(record["split_manifest_sha256"], EXPECTED_SPLIT_SHAS[seed])
            self.assertEqual(record["training_id_order_sha256"], ordered_id_hash(split["train_ids"]))
            self.assertEqual(record["number_of_training_universes"], 700)
            self.assertEqual(sum(record["histogram"]), record["total_node_count"])
            self.assertEqual(record["total_node_count"], 1_050_000)
            self.assertEqual(record["max_degree"], len(record["histogram"]) - 1)
            self.assertEqual(record["histogram_content_sha256"],
                             histogram_content_sha256(record["histogram"]))
            self.assertEqual(record["partition_hygiene"]["validation_ids_contributed"], 0)
            self.assertEqual(record["partition_hygiene"]["test_ids_contributed"], 0)
            self.assertFalse(record["partition_hygiene"]["targets_read"])

    def test_configs_have_exactly_declared_controlled_differences(self) -> None:
        for seed in SEEDS:
            control = read_json(ROOT / control_config_path(seed))
            treatment = read_json(ROOT / config_path(seed))
            self.assertEqual(config_differences(control, treatment), PNA_FIELDS)
            self.assertEqual(treatment["experiment_name"], experiment_name(seed))
            self.assertEqual(treatment["model"], "StaticPNARegressor")
            self.assertEqual(treatment["split_manifest_path"], split_path(seed).as_posix())

    def test_protocol_is_exact_and_parameter_counts_are_frozen(self) -> None:
        records = {seed: read_json(ROOT / histogram_path(seed)) for seed in SEEDS}
        protocol = read_json(ROOT / PROTOCOL)
        self.assertEqual(protocol, protocol_document(records))
        self.assertEqual(count_parameters(StaticGCNRegressor(7, 32, 3, .2, "mean", "gcn")), 5281)
        self.assertEqual(count_parameters(StaticPNARegressor(deg=records[42]["histogram"])), 51553)
        self.assertTrue(protocol["selection"]["test_metrics_must_not_be_used_for_architecture_selection"])
        self.assertTrue(protocol["no_pna_hyperparameter_tuning_performed"])

    def test_production_run_names_are_collision_free(self) -> None:
        for seed in SEEDS:
            self.assertFalse((ROOT / "experiments" / experiment_name(seed)).exists())


if __name__ == "__main__":
    unittest.main()
