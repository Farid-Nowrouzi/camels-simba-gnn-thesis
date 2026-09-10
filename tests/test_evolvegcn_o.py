from __future__ import annotations

import io
import unittest

import torch

from src.models.evolvegcn_o import EvolveGCNORegressor, count_parameters
from src.training.sparse_batch import collate_sparse_temporal


def sample(node_counts=(3, 4, 5, 6, 7), offset=0.0):
    nodes = [
        torch.arange(count * 7, dtype=torch.float32).reshape(count, 7) / 20 + offset + time
        for time, count in enumerate(node_counts)
    ]
    return {
        "Nodes_list": nodes,
        "mask_list": [torch.ones(x.shape[0], 1) for x in nodes],
        "edge_index_list": [
            torch.stack((torch.arange(x.shape[0]), torch.roll(torch.arange(x.shape[0]), -1)))
            for x in nodes
        ],
        "target": torch.tensor(0.3),
    }


def model():
    return EvolveGCNORegressor(
        node_features=7, hidden_dim=32, num_layers=2, dropout=0.0,
        activation="relu", temporal_pooling="mean", graph_pooling="mean",
        head_type="linear", add_self_loops=True,
    )


class EvolveGCNOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = collate_sparse_temporal([
            sample(), sample((8, 7, 6, 5, 4), offset=10.0),
        ])

    def test_import_feature_dimension_five_snapshots_and_output(self):
        network = model().eval()
        prediction = network(self.batch)
        self.assertEqual(network.node_features, 7)
        self.assertEqual(self.batch["num_timesteps"], 5)
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_finite_backward_and_parameter_count(self):
        network = model()
        loss = network(self.batch).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(count_parameters(network), 11_527)
        gradients = [p.grad for p in network.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

    def test_weight_state_evolves_across_snapshots(self):
        layer = model().layers[0]
        states = layer.evolved_weights(5)
        self.assertEqual(len(states), 5)
        self.assertFalse(torch.equal(layer.initial_weight, states[0]))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(states, states[1:])))

    def test_snapshot_order_changes_result(self):
        network = model().eval()
        forward = network(self.batch)
        reverse = dict(self.batch, snapshots=list(reversed(self.batch["snapshots"])))
        self.assertFalse(torch.allclose(forward, network(reverse)))

    def test_no_node_correspondence_and_varying_counts(self):
        # The two sequences deliberately have unrelated and reversed node counts.
        prediction = model().eval()(self.batch)
        self.assertEqual(tuple(prediction.shape), (2, 1))
        counts = [torch.diff(snapshot["ptr"]).tolist() for snapshot in self.batch["snapshots"]]
        self.assertEqual(counts, [[3, 8], [4, 7], [5, 6], [6, 5], [7, 4]])

    def test_state_dict_round_trip_and_seed_determinism(self):
        network = model().eval()
        expected = network(self.batch)
        buffer = io.BytesIO()
        torch.save(network.state_dict(), buffer)
        buffer.seek(0)
        restored = model().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(expected, restored(self.batch)))
        torch.manual_seed(7)
        first = model().eval()(self.batch)
        torch.manual_seed(7)
        second = model().eval()(self.batch)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
