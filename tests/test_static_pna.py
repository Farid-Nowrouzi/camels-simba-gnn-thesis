from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.models.static_pna import (
    CANONICAL_AGGREGATORS,
    CANONICAL_SCALERS,
    StaticPNARegressor,
    effective_pna_edge_index,
)
from src.training.sparse_batch import collate_sparse_static
from src.training.train_static_pna import histogram_content_sha256, load_degree_histogram


def graph_batch() -> dict:
    samples = []
    for nodes in (4, 3):
        edge = torch.tensor([
            [index for index in range(nodes) for _ in (0, 1)],
            [value for index in range(nodes) for value in ((index + 1) % nodes, (index - 1) % nodes)],
        ])
        samples.append({
            "graph_storage": "sparse_edge_index",
            "x": torch.randn(nodes, 7),
            "edge_index": edge,
            "edge_weight": None,
            "mask": torch.ones(nodes, 1),
        })
    return collate_sparse_static(samples)


class StaticPNATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.deg = torch.tensor([0, 0, 0, 7])

    def test_construction_architecture_pool_and_no_temporal_path(self) -> None:
        model = StaticPNARegressor(deg=self.deg)
        self.assertEqual(model.node_features, 7)
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.graph_pooling, "mean")
        self.assertEqual(model.aggregators, CANONICAL_AGGREGATORS)
        self.assertEqual(model.scalers, CANONICAL_SCALERS)
        self.assertTrue(all(isinstance(layer.layer_norm, nn.LayerNorm) for layer in model.layers))
        self.assertTrue(all(layer.use_residual for layer in model.layers))
        linear_shapes = [(m.in_features, m.out_features) for m in model.regressor if isinstance(m, nn.Linear)]
        self.assertEqual(linear_shapes, [(32, 32), (32, 16), (16, 1)])
        with self.assertRaisesRegex(ValueError, "sparse static"):
            model.encode_nodes({"snapshots": []})

    def test_output_shape_and_deterministic_eval_forward(self) -> None:
        batch = graph_batch()
        original_edge = batch["edge_index"].clone()
        model = StaticPNARegressor(deg=self.deg).eval()
        with torch.inference_mode():
            first = model(batch)
            second = model(batch)
        self.assertEqual(tuple(first.shape), (2, 1))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        torch.testing.assert_close(batch["edge_index"], original_edge, rtol=0.0, atol=0.0)

    def test_forward_uses_exact_mean_graph_pooling(self) -> None:
        batch = graph_batch()
        model = StaticPNARegressor(deg=self.deg).eval()
        nodes = torch.arange(7 * 32, dtype=torch.float32).reshape(7, 32)
        model.encode_nodes = lambda graph: nodes  # type: ignore[method-assign]
        model.regressor = nn.Identity()
        pooled = model(batch)
        torch.testing.assert_close(pooled[0], nodes[:4].mean(dim=0))
        torch.testing.assert_close(pooled[1], nodes[4:].mean(dim=0))

    def test_remaining_self_loops_are_unique_deterministic_and_nonmutating(self) -> None:
        edge = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 1]])
        original = edge.clone()
        first = effective_pna_edge_index(edge, 3)
        second = effective_pna_edge_index(edge, 3)
        torch.testing.assert_close(edge, original, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        loops = first[:, first[0] == first[1]]
        self.assertEqual(loops.shape[1], 3)
        self.assertEqual(set(map(tuple, loops.T.tolist())), {(0, 0), (1, 1), (2, 2)})

    def test_missing_and_invalid_histograms_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            StaticPNARegressor(deg=None)
        with self.assertRaisesRegex(ValueError, "positive"):
            StaticPNARegressor(deg=[0, 0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "degree.json"
            path.write_text(json.dumps({"histogram": [0, 0, 3],
                                        "total_node_count": 3,
                                        "histogram_content_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_degree_histogram(path)
            path.write_text(json.dumps({"histogram": [0, 1.5],
                                        "total_node_count": 1,
                                        "histogram_content_sha256": histogram_content_sha256([0, 1])}))
            with self.assertRaisesRegex(ValueError, "integers"):
                load_degree_histogram(path)


if __name__ == "__main__":
    unittest.main()
