from __future__ import annotations

import io
import unittest

import torch

from src.models.gcn_gru import GCNGRURegressor, count_parameters
from src.training.sparse_batch import collate_sparse_temporal


def sample(node_counts=(3, 4, 5, 6, 7), offset=0.0):
    nodes = [torch.full((count, 7), offset + timestep) for timestep, count in enumerate(node_counts)]
    return {
        "Nodes_list": nodes, "mask_list": [torch.ones(x.shape[0], 1) for x in nodes],
        "edge_index_list": [torch.stack((torch.arange(x.shape[0]), torch.roll(torch.arange(x.shape[0]), -1))) for x in nodes],
        "target": torch.tensor(0.3),
    }


class GCNGRUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = collate_sparse_temporal([sample(offset=0), sample((8, 7, 6, 5, 4), 10)])
        self.model = GCNGRURegressor()

    def test_shapes_order_and_single_shared_encoder(self):
        self.model.eval()
        sequence = self.model.encode_snapshots(self.batch)
        self.assertEqual(tuple(sequence.shape), (2, 5, 32))
        self.assertEqual(tuple(self.model.temporal_representation(sequence).shape), (2, 32))
        self.assertEqual(len([m for m in self.model.modules() if m is self.model.snapshot_encoder]), 1)
        seen = []
        handle = self.model.snapshot_encoder.register_forward_hook(lambda _m, _i, output: seen.append(output.shape))
        gru_input = []
        gru_handle = self.model.temporal_module.register_forward_pre_hook(
            lambda _m, inputs: gru_input.append(inputs[0].detach().clone())
        )
        output = self.model(self.batch)
        gru_handle.remove()
        handle.remove()
        self.assertEqual(seen, [torch.Size([2, 32])] * 5)
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(gru_input[0], sequence))

    def test_forward_backward_parameter_count_and_no_node_correspondence(self):
        output = self.model(self.batch)
        output.square().mean().backward()
        self.assertEqual(count_parameters(self.model), 11617)
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

    def test_state_dict_round_trip_and_seed_determinism(self):
        self.model.eval()
        expected = self.model(self.batch)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = GCNGRURegressor().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(expected, restored(self.batch)))
        torch.manual_seed(7)
        first = GCNGRURegressor().eval()(self.batch)
        torch.manual_seed(7)
        second = GCNGRURegressor().eval()(self.batch)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
