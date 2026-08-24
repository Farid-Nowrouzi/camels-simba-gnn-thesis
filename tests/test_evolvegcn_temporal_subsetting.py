"""Regression tests for explicit EvolveGCN-H temporal snapshot views."""

from __future__ import annotations

import copy
import unittest

import torch

from src.models.evolvegcn_h import EvolveGCNHRegressor, count_parameters
from src.training.sparse_batch import collate_sparse_temporal
from src.training.train_evolvegcn_h import select_temporal_snapshots


VALUES = [0.2, 0.25, 0.51209, 0.75065, 1.0]
SUBSETS = {
    1: [4],
    2: [3, 4],
    3: [2, 3, 4],
    4: [1, 2, 3, 4],
    5: [0, 1, 2, 3, 4],
}


def synthetic_sample(with_weights: bool = True) -> dict:
    nodes = [torch.full((3, 7), float(index)) for index in range(5)]
    masks = [torch.ones(3, 1) for _ in range(5)]
    edges = [torch.tensor([[0, 1, 2], [1, 2, 0]]) for _ in range(5)]
    sample = {
        "Nodes_list": nodes,
        "mask_list": masks,
        "edge_index_list": edges,
        "snapshots": [
            {"snapshot_value": value, "stored_index": index}
            for index, value in enumerate(VALUES)
        ],
        "edge_weight_list": (
            [torch.full((3,), float(index + 1)) for index in range(5)]
            if with_weights
            else None
        ),
        "target": torch.tensor(0.3),
        "universe_id": "LH_0",
        "num_snapshots": 5,
    }
    return sample


def build_model() -> EvolveGCNHRegressor:
    return EvolveGCNHRegressor(
        node_features=7,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        activation="relu",
        temporal_pooling="mean",
        graph_pooling="mean",
        head_type="linear",
        add_self_loops=True,
    )


class TemporalSubsettingTests(unittest.TestCase):
    def test_default_none_returns_original_full_sample(self) -> None:
        sample = synthetic_sample()
        result = select_temporal_snapshots(sample)
        self.assertIs(result, sample)
        self.assertEqual(len(result["snapshots"]), 5)

    def test_exact_t1_through_t5_views_and_alignment(self) -> None:
        sample = synthetic_sample()
        for timestep_count, indices in SUBSETS.items():
            with self.subTest(T=timestep_count):
                result = select_temporal_snapshots(
                    sample, indices, [VALUES[index] for index in indices]
                )
                self.assertEqual(result["num_snapshots"], timestep_count)
                self.assertEqual(
                    [item["stored_index"] for item in result["snapshots"]], indices
                )
                for position, index in enumerate(indices):
                    self.assertIs(result["Nodes_list"][position], sample["Nodes_list"][index])
                    self.assertIs(result["mask_list"][position], sample["mask_list"][index])
                    self.assertIs(
                        result["edge_index_list"][position],
                        sample["edge_index_list"][index],
                    )
                    self.assertIs(
                        result["edge_weight_list"][position],
                        sample["edge_weight_list"][index],
                    )
                self.assertIsNot(result, sample)
                self.assertEqual(len(sample["snapshots"]), 5)

    def test_edge_weights_may_be_absent(self) -> None:
        sample = synthetic_sample(with_weights=False)
        result = select_temporal_snapshots(sample, [3, 4], VALUES[3:])
        self.assertIsNone(result["edge_weight_list"])

    def test_empty_subset_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            select_temporal_snapshots(synthetic_sample(), [])

    def test_duplicate_subset_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_temporal_snapshots(synthetic_sample(), [3, 4, 4])

    def test_unsorted_subset_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            select_temporal_snapshots(synthetic_sample(), [4, 3])

    def test_out_of_range_and_negative_rejected(self) -> None:
        for indices in ([-1, 4], [4, 5]):
            with self.subTest(indices=indices), self.assertRaisesRegex(
                ValueError, "out-of-range"
            ):
                select_temporal_snapshots(synthetic_sample(), indices)

    def test_incorrect_expected_value_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match"):
            select_temporal_snapshots(synthetic_sample(), [3, 4], [0.5, 1.0])

    def test_expected_values_without_indices_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            select_temporal_snapshots(synthetic_sample(), None, VALUES)

    def test_non_final_ending_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "last stored snapshot"):
            select_temporal_snapshots(synthetic_sample(), [2, 3])

    def test_non_unit_final_value_rejected(self) -> None:
        sample = synthetic_sample()
        sample["snapshots"][-1]["snapshot_value"] = 0.9
        with self.assertRaisesRegex(ValueError, "1.00000"):
            select_temporal_snapshots(sample, [4])

    def test_non_increasing_snapshot_values_rejected(self) -> None:
        sample = synthetic_sample()
        sample["snapshots"][3]["snapshot_value"] = 1.0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_temporal_snapshots(sample, [3, 4])

    def test_temporal_field_length_mismatch_rejected(self) -> None:
        sample = synthetic_sample()
        sample["mask_list"] = sample["mask_list"][:-1]
        with self.assertRaisesRegex(ValueError, "field-length mismatch"):
            select_temporal_snapshots(sample, [4])

    def test_explicit_t5_equals_default_contents(self) -> None:
        sample = synthetic_sample()
        original = copy.deepcopy(sample)
        default = select_temporal_snapshots(sample)
        explicit = select_temporal_snapshots(sample, SUBSETS[5], VALUES)
        for field in ("Nodes_list", "mask_list", "edge_index_list", "edge_weight_list"):
            for default_tensor, explicit_tensor in zip(default[field], explicit[field]):
                self.assertTrue(torch.equal(default_tensor, explicit_tensor))
        self.assertEqual(default["snapshots"], explicit["snapshots"])
        self.assertTrue(torch.equal(default["target"], explicit["target"]))
        self.assertEqual(sample["snapshots"], original["snapshots"])

    def test_t1_sparse_forward_backward(self) -> None:
        selected = select_temporal_snapshots(synthetic_sample(), [4], [1.0])
        batch = collate_sparse_temporal([selected, selected])
        model = build_model()
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(torch.isfinite(loss))
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))

    def test_parameter_and_state_dict_schema_invariant_t1_to_t5(self) -> None:
        schemas = []
        for timestep_count in SUBSETS:
            model = build_model()
            self.assertEqual(count_parameters(model), 3_408_097, msg=f"T={timestep_count}")
            schemas.append(
                [(key, tuple(value.shape)) for key, value in model.state_dict().items()]
            )
        self.assertTrue(all(schema == schemas[0] for schema in schemas[1:]))


if __name__ == "__main__":
    unittest.main()
