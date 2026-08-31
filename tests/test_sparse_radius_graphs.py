from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.camels_graph_utils import (
    build_radius_adjacency,
    build_sparse_radius_edge_index,
)


def mask(real: int, total: int) -> np.ndarray:
    value = np.zeros((total, 1), dtype=np.float32)
    value[:real] = 1
    return value


def dense_edge_index(adjacency: np.ndarray) -> np.ndarray:
    source, target = np.nonzero(adjacency)
    if source.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    order = np.lexsort((target, source))
    return np.stack((source[order], target[order])).astype(np.int64)


class SparseRadiusGraphTests(unittest.TestCase):
    def assert_equivalent(
        self, positions: np.ndarray, real: int, radius: float, *, periodic: bool = True
    ) -> np.ndarray:
        node_mask = mask(real, len(positions))
        sparse = build_sparse_radius_edge_index(
            positions, node_mask, radius, periodic_boundary=periodic, box_size=25.0,
        )
        dense = build_radius_adjacency(
            positions, node_mask, radius, periodic_boundary=periodic, box_size=25.0,
        )
        np.testing.assert_array_equal(sparse, dense_edge_index(dense))
        return sparse

    def test_ordinary_nonperiodic_cloud_exact_dense_equivalence(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.float32)
        expected = np.array([[0, 1], [1, 0]], dtype=np.int64)
        np.testing.assert_array_equal(self.assert_equivalent(positions, 3, 1.5, periodic=False), expected)

    def test_periodic_boundary_crossing(self) -> None:
        positions = np.array([[0.1, 0, 0], [24.9, 0, 0], [12, 0, 0]], dtype=np.float32)
        edges = self.assert_equivalent(positions, 3, 0.21)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [1, 0]], dtype=np.int64))

    def test_exact_threshold_is_inclusive(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        self.assertEqual(self.assert_equivalent(positions, 2, 1.0).shape[1], 2)

    def test_infinitesimally_above_threshold_is_excluded(self) -> None:
        positions = np.array(
            [[0, 0, 0], [np.nextafter(np.float32(1.0), np.float32(2.0)), 0, 0]],
            dtype=np.float32,
        )
        self.assertEqual(self.assert_equivalent(positions, 2, 1.0).shape, (2, 0))

    def test_padding_is_excluded(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        edges = self.assert_equivalent(positions, 2, 2.0)
        self.assertTrue(edges.size == 0 or int(edges.max()) < 2)

    def test_one_real_node(self) -> None:
        positions = np.zeros((3, 3), dtype=np.float32)
        self.assertEqual(self.assert_equivalent(positions, 1, 2.0).shape, (2, 0))

    def test_two_real_nodes(self) -> None:
        positions = np.array([[0, 0, 0], [0, 0, 0.5]], dtype=np.float32)
        self.assertEqual(self.assert_equivalent(positions, 2, 0.5).shape, (2, 2))

    def test_radius_yielding_no_edges(self) -> None:
        positions = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float32)
        self.assertEqual(self.assert_equivalent(positions, 3, 0.1).shape, (2, 0))

    def test_radius_yielding_complete_graph(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.assertEqual(self.assert_equivalent(positions, 4, 2.0).shape[1], 12)

    def test_contract_properties_and_determinism(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=np.float32)
        first = self.assert_equivalent(positions, 4, 1.5)
        second = build_sparse_radius_edge_index(positions, mask(4, 4), 1.5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.int64)
        pairs = [tuple(pair) for pair in first.T.tolist()]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(source != target for source, target in pairs))
        self.assertTrue(all((target, source) in set(pairs) for source, target in pairs))
        self.assertEqual(pairs, sorted(pairs))

    def test_output_converts_to_torch_int64(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        edge_index = torch.tensor(
            build_sparse_radius_edge_index(positions, mask(2, 2), 1.0), dtype=torch.long,
        )
        self.assertEqual(edge_index.dtype, torch.int64)

    def test_invalid_radius_fails_closed(self) -> None:
        positions = np.zeros((2, 3), dtype=np.float32)
        for radius in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_sparse_radius_edge_index(positions, mask(2, 2), radius)


if __name__ == "__main__":
    unittest.main()
