"""Architecture and protected-runner tests for the frozen CNN3D experiment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.evaluation import run_modern_cnn3d as runner
from src.models.static_cnn3d import (
    EXPECTED_PARAMETERS,
    DeterministicAvgPool3d2,
    StaticCNN3DRegressor,
    count_parameters,
)


class CNNArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        runner.seed_everything(42)
        self.model = StaticCNN3DRegressor()
        self.x = torch.randn(2, 5, 32, 32, 32)

    def test_shapes_validation_and_parameters(self) -> None:
        self.assertEqual(tuple(self.model(self.x).shape), (2, 1))
        self.assertEqual(tuple(self.model(self.x[:1]).shape), (1, 1))
        self.assertEqual(count_parameters(self.model), EXPECTED_PARAMETERS)
        with self.assertRaisesRegex(ValueError, "Expected input"):
            self.model(torch.randn(2, 4, 32, 32, 32))
        with self.assertRaisesRegex(ValueError, "Expected input"):
            self.model(torch.randn(2, 5, 32, 32))

    def test_backward_init_dropout_checkpoint_and_even_shift(self) -> None:
        loss = nn.MSELoss()(self.model(self.x), torch.randn(2, 1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        torch.optim.AdamW(self.model.parameters(), lr=.001).step()
        runner.seed_everything(123); left = StaticCNN3DRegressor().state_dict()
        runner.seed_everything(123); right = StaticCNN3DRegressor().state_dict()
        self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))
        self.model.eval()
        with torch.no_grad():
            reference = self.model(self.x)
            shifted = self.model(torch.roll(self.x, shifts=(2, 2, 2), dims=(2, 3, 4)))
        torch.testing.assert_close(reference, shifted, rtol=2e-5, atol=2e-6)
        self.assertEqual(self.model.conv1.padding_mode, "circular")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(self.model.state_dict(), path)
            restored = StaticCNN3DRegressor(); restored.load_state_dict(torch.load(path, weights_only=True)); restored.eval()
            torch.testing.assert_close(reference, restored(self.x), rtol=0, atol=0)


class DeterministicAvgPoolTests(unittest.TestCase):
    def test_equivalent_to_avgpool3d_and_parameter_free(self) -> None:
        pool = DeterministicAvgPool3d2()
        reference = nn.AvgPool3d(kernel_size=2, stride=2)
        for shape in ((1, 1, 4, 4, 4), (2, 16, 32, 32, 32)):
            values = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
            actual = pool(values)
            expected = reference(values)
            self.assertEqual(tuple(actual.shape), (shape[0], shape[1], shape[2] // 2, shape[3] // 2, shape[4] // 2))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(count_parameters(pool), 0)

    def test_rejects_incompatible_spatial_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 2"):
            DeterministicAvgPool3d2()(torch.randn(1, 1, 3, 4, 4))


class RunnerContractTests(unittest.TestCase):
    def test_scientific_binding_and_production_guards(self) -> None:
        runner.verify_scientific_freeze()
        family = runner.load_family()
        self.assertEqual(len(runner.expand_family_jobs(family)), 3)
        with self.assertRaisesRegex(ValueError, "Implementation/source freeze"):
            runner.require_implementation_freeze("not-a-freeze")
        with self.assertRaisesRegex(ValueError, "Implementation/source freeze"):
            runner.verify_all_trained("not-a-freeze")

    def test_partition_access_control(self) -> None:
        family = runner.load_family()
        job = runner.expand_family_jobs(family)[0]
        with self.assertRaisesRegex(ValueError, "cannot access test"):
            runner._partition_rows(StaticCNN3DRegressor(), {}, job["manifest"], "test", torch.device("cpu"), "train")


if __name__ == "__main__":
    unittest.main()
