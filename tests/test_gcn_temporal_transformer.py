from __future__ import annotations

import io
import unittest

import torch
import torch.nn as nn

from src.models.gcn_temporal_transformer import (
    DEFAULT_SCALE_FACTORS, GCNTemporalTransformerRegressor,
    count_parameters, sinusoidal_scale_factor_encoding,
)
from src.training.sparse_batch import collate_sparse_temporal


def sample(node_counts=(3, 4, 5, 6, 7), offset=0.0):
    nodes = [torch.full((count, 7), offset + timestep) for timestep, count in enumerate(node_counts)]
    return {
        "Nodes_list": nodes, "mask_list": [torch.ones(x.shape[0], 1) for x in nodes],
        "edge_index_list": [torch.stack((torch.arange(x.shape[0]), torch.roll(torch.arange(x.shape[0]), -1))) for x in nodes],
        "target": torch.tensor(0.3),
    }


class GCNTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = collate_sparse_temporal([sample(offset=0), sample((7, 6, 5, 4, 3), 10)])
        self.model = GCNTemporalTransformerRegressor()

    def test_encoding_is_deterministic_ordered_and_correct_shape(self):
        first = sinusoidal_scale_factor_encoding(DEFAULT_SCALE_FACTORS, 32)
        second = sinusoidal_scale_factor_encoding(DEFAULT_SCALE_FACTORS, 32)
        self.assertEqual(tuple(first.shape), (1, 5, 32))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(isinstance(self.model.temporal_encoding, nn.Parameter))
        self.assertTrue(torch.equal(self.model.scale_factors, torch.tensor(DEFAULT_SCALE_FACTORS)))
        self.assertFalse(torch.equal(first[:, 0], first[:, -1]))

    def test_shapes_final_token_forward_and_backward(self):
        self.model.eval()
        sequence = self.model.encode_snapshots(self.batch)
        self.assertEqual(tuple(sequence.shape), (2, 5, 32))
        self.assertEqual(tuple(self.model.temporal_representation(sequence).shape), (2, 32))
        transformer_input = []
        handle = self.model.temporal_module.register_forward_pre_hook(
            lambda _m, inputs: transformer_input.append(inputs[0].detach().clone())
        )
        output = self.model(self.batch)
        handle.remove()
        self.assertEqual(tuple(output.shape), (2, 1))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(count_parameters(self.model), 13825)
        self.assertTrue(torch.equal(
            transformer_input[0], sequence + self.model.temporal_encoding
        ))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters() if p.requires_grad))

    def test_final_snapshot_token_selected_after_transformer(self):
        self.model.temporal_module = nn.Identity()
        sequence = torch.randn(2, 5, 32)
        expected = sequence[:, -1] + self.model.temporal_encoding[:, -1]
        self.assertTrue(torch.equal(self.model.temporal_representation(sequence), expected))

    def test_state_dict_round_trip(self):
        self.model.eval()
        expected = self.model(self.batch)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = GCNTemporalTransformerRegressor().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(expected, restored(self.batch)))


if __name__ == "__main__":
    unittest.main()
