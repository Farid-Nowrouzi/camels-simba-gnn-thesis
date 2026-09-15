"""Synthetic contracts for the frozen periodic CIC representation."""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from src.data.halo_voxelization import (
    EXPERIMENT_1_SPEC, RAW7_FEATURE_NAMES, periodic_cic_indices_and_weights,
    representation_fingerprint, voxelize_halo_features,
)


def halo(x: float, y: float, z: float, mass: float = 10.0, vx: float = 2.0, vy: float = 3.0, vz: float = 4.0) -> torch.Tensor:
    return torch.tensor([[mass, x, y, z, vx, vy, vz]], dtype=torch.float64)


class HaloVoxelizationTests(unittest.TestCase):
    def assert_conserved(self, values: torch.Tensor) -> None:
        grid = voxelize_halo_features(values)
        raw = grid.double() * torch.tensor([1., 10., 100., 100., 100.]).view(5, 1, 1, 1)
        expected = torch.tensor([len(values), values[:, 0].sum(), values[:, 4].sum(), values[:, 5].sum(), values[:, 6].sum()])
        torch.testing.assert_close(raw.sum(dim=(1, 2, 3)), expected, rtol=0, atol=2e-5)

    def test_shape_dtype_finite_scaling_and_empty_zero(self) -> None:
        grid = voxelize_halo_features(halo(1., 2., 3.))
        self.assertEqual(tuple(grid.shape), (5, 32, 32, 32))
        self.assertEqual(grid.dtype, torch.float32)
        self.assertTrue(torch.isfinite(grid).all())
        self.assertTrue((grid[:, 0, 0, 0] == 0).all())
        self.assertAlmostEqual(float(grid[0].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(grid[1].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(grid[2].sum()), .02, places=6)

    def test_center_midpoint_and_weights_partition_unity(self) -> None:
        delta = 25.0 / 32
        indices, weights = periodic_cic_indices_and_weights(halo(0.5 * delta, 0.5 * delta, 0.5 * delta)[:, 1:4])
        self.assertEqual(int((weights[0] > 0).sum()), 1)
        torch.testing.assert_close(weights.sum(1), torch.ones(1, dtype=torch.float64), rtol=0, atol=0)
        _, midpoint = periodic_cic_indices_and_weights(halo(delta, delta, delta)[:, 1:4])
        torch.testing.assert_close(midpoint, torch.full((1, 8), 0.125, dtype=torch.float64), rtol=0, atol=1e-15)

    def test_boundaries_and_periodic_equivalence(self) -> None:
        values = torch.cat([halo(0., 0., 0.), halo(25., 25., 25.), halo(-.1, 25.1, 50.1), halo(25. - 1e-10, 25. - 1e-10, 25. - 1e-10)])
        self.assert_conserved(values)
        torch.testing.assert_close(voxelize_halo_features(halo(0., 0., 0.)), voxelize_halo_features(halo(25., 25., 25.)), rtol=0, atol=0)
        self.assertTrue((voxelize_halo_features(halo(0., 0., 0.))[0, -1] > 0).any())

    def test_face_edge_corner_wrapping(self) -> None:
        for coordinate in ((0., 4., 6.), (0., 0., 6.), (0., 0., 0.)):
            grid = voxelize_halo_features(halo(*coordinate))
            self.assertGreater(int((grid[0] > 0).sum()), 1)
            self.assertAlmostEqual(float(grid[0].sum()), 1.0, places=6)

    def test_conservation_multiple_halos_velocity_cancellation_and_order(self) -> None:
        values = torch.cat([halo(3., 4., 5., vx=7.), halo(3., 4., 5., vx=-7.), halo(8., 9., 10., mass=11., vx=0., vy=-5.)])
        self.assert_conserved(values)
        left = voxelize_halo_features(values)
        right = voxelize_halo_features(values[torch.tensor([2, 0, 1])])
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7)
        self.assertAlmostEqual(float(left[2].sum()), 0.0, places=6)

    def test_deterministic_fingerprint_and_contract_rejections(self) -> None:
        values = halo(1., 2., 3.)
        torch.testing.assert_close(voxelize_halo_features(values), voxelize_halo_features(values), rtol=0, atol=0)
        self.assertEqual(representation_fingerprint(), representation_fingerprint())
        self.assertNotEqual(representation_fingerprint(), representation_fingerprint(replace(EXPERIMENT_1_SPEC, resolution=16)))
        with self.assertRaisesRegex(ValueError, "feature order"):
            voxelize_halo_features(values, feature_names=list(reversed(RAW7_FEATURE_NAMES)))
        with self.assertRaisesRegex(ValueError, "Expected halo features"):
            voxelize_halo_features(torch.zeros(1, 6))


if __name__ == "__main__":
    unittest.main()
