from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.camels_graph_utils import build_sparse_knn_edge_index, compute_pairwise_distances
from src.evaluation.symmetry import (
    apply_periodic_translation,
    identity_rotation_id,
    proper_cube_rotations,
    transform_raw7_features,
    translation_vectors,
    validate_proper_cube_rotation,
)

BOX = 25.0


def periodic_distances(positions: torch.Tensor) -> np.ndarray:
    return compute_pairwise_distances(positions.numpy(), True, BOX)


def fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(90125)
    features = torch.randn((19, 7), generator=generator, dtype=torch.float64)
    features[:, 0] = torch.linspace(10.0, 14.0, 19, dtype=torch.float64)
    features[:, 1:4] = torch.rand((19, 3), generator=generator, dtype=torch.float64) * BOX
    features[0, 1:4] = torch.tensor([0.01, 24.98, 12.5], dtype=torch.float64)
    features[1, 1:4] = torch.tensor([24.99, 0.03, 12.7], dtype=torch.float64)
    mask = torch.ones((19, 1), dtype=torch.float64)
    mask[-3:] = 0
    return features, mask


class SymmetryTests(unittest.TestCase):
    def test_translation_contract_and_wrapping(self) -> None:
        self.assertEqual(translation_vectors(BOX), {
            "T1": (12.5, 0.0, 0.0), "T2": (0.0, 12.5, 0.0),
            "T3": (0.0, 0.0, 12.5), "T4": (6.25, 12.5, 18.75),
        })
        positions = torch.tensor([[24.0, 1.0, 12.0]], dtype=torch.float32)
        actual = apply_periodic_translation(positions, (2.0, -2.0, 20.0), BOX)
        self.assertTrue(torch.equal(actual, torch.tensor([[1.0, 24.0, 7.0]])))

    def test_rotations_are_complete_proper_and_deterministic(self) -> None:
        first = proper_cube_rotations()
        second = proper_cube_rotations()
        self.assertEqual(list(first), [f"R{i:02d}" for i in range(24)])
        self.assertEqual(list(first), list(second))
        self.assertEqual(len({matrix.tobytes() for matrix in first.values()}), 24)
        for key in first:
            matrix = first[key]
            self.assertTrue(np.array_equal(matrix, second[key]))
            self.assertTrue(np.array_equal(matrix.T @ matrix, np.eye(3, dtype=np.int8)))
            self.assertEqual(round(np.linalg.det(matrix)), 1)
            self.assertTrue(np.all(np.count_nonzero(matrix, axis=0) == 1))
            self.assertTrue(np.all(np.count_nonzero(matrix, axis=1) == 1))
        self.assertEqual(identity_rotation_id(), "R03")
        self.assertEqual(sum(np.array_equal(matrix, np.eye(3)) for matrix in first.values()), 1)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_proper_cube_rotation(np.diag([-1, 1, 1]))
        with self.assertRaises(ValueError):
            transform_raw7_features(torch.zeros(2, 6), translation=(1, 2, 3))
        with self.assertRaises(ValueError):
            transform_raw7_features(torch.zeros(2, 7), box_size=0, translation=(1, 2, 3))
        with self.assertRaises(ValueError):
            transform_raw7_features(torch.zeros(2, 7), translation=(1, 2, 3), rotation=np.eye(3))

    def test_translation_contract_and_periodic_distances(self) -> None:
        features, mask = fixture()
        source = features.clone()
        target = torch.tensor(0.31, dtype=features.dtype)
        for transform_id, delta in translation_vectors(BOX).items():
            with self.subTest(transform_id=transform_id):
                transformed = transform_raw7_features(features, translation=delta, box_size=BOX)
                self.assertTrue(torch.equal(features, source))
                self.assertEqual(transformed.dtype, features.dtype)
                self.assertEqual(transformed.device, features.device)
                self.assertTrue(torch.equal(transformed[:, 0], features[:, 0]))
                self.assertTrue(torch.equal(transformed[:, 4:7], features[:, 4:7]))
                self.assertTrue(torch.equal(mask, mask.clone()))
                self.assertTrue(torch.equal(target, target.clone()))
                np.testing.assert_allclose(periodic_distances(transformed[:, 1:4]),
                                           periodic_distances(features[:, 1:4]), rtol=0, atol=2e-6)

    def test_rotation_contract_and_periodic_distances(self) -> None:
        features, mask = fixture()
        source = features.clone()
        for rotation_id, rotation in proper_cube_rotations().items():
            with self.subTest(rotation_id=rotation_id):
                transformed = transform_raw7_features(features, rotation=rotation, box_size=BOX)
                self.assertTrue(torch.equal(features, source))
                self.assertEqual(transformed.dtype, features.dtype)
                self.assertEqual(transformed.device, features.device)
                self.assertTrue(torch.equal(transformed[:, 0], features[:, 0]))
                expected_velocity = features[:, 4:7] @ torch.as_tensor(rotation, dtype=features.dtype).T
                self.assertTrue(torch.equal(transformed[:, 4:7], expected_velocity))
                self.assertTrue(bool(((transformed[:, 1:4] >= 0) &
                                      (transformed[:, 1:4] < BOX)).all()))
                self.assertTrue(torch.equal(mask, mask.clone()))
                np.testing.assert_allclose(periodic_distances(transformed[:, 1:4]),
                                           periodic_distances(features[:, 1:4]), rtol=0, atol=2e-6)

    def test_sparse_knn_topology_all_transforms_padding_and_rebuilds(self) -> None:
        features, mask = fixture()
        positions = features[:, 1:4].numpy()
        mask_array = mask.numpy()
        baseline = build_sparse_knn_edge_index(positions, mask_array, k=8, box_size=BOX)
        repeated = build_sparse_knn_edge_index(positions, mask_array, k=8, box_size=BOX)
        self.assertTrue(np.array_equal(baseline, repeated))
        self.assertFalse(np.isin(np.arange(16, 19), baseline).any())
        transforms = [("translation", key, value) for key, value in translation_vectors(BOX).items()]
        transforms += [("rotation", key, value) for key, value in proper_cube_rotations().items()]
        for family, transform_id, value in transforms:
            with self.subTest(family=family, transform_id=transform_id):
                changed = transform_raw7_features(features, box_size=BOX, **{family: value})
                rebuilt = build_sparse_knn_edge_index(changed[:, 1:4].numpy(), mask_array,
                                                      k=8, box_size=BOX)
                self.assertTrue(np.array_equal(baseline, rebuilt))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_transform_preserves_cuda_device(self) -> None:
        features = torch.zeros((2, 7), device="cuda", dtype=torch.float32)
        translated = transform_raw7_features(features, translation=(1, 2, 3))
        rotated = transform_raw7_features(features, rotation=np.eye(3, dtype=np.int8))
        self.assertEqual(translated.device, features.device)
        self.assertEqual(rotated.device, features.device)


if __name__ == "__main__":
    unittest.main()
