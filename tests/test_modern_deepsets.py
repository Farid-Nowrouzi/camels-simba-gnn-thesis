from __future__ import annotations

import inspect
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.evaluation.run_modern_deepsets import (
    ARCHITECTURE_VERSION, FEATURE_NAMES, DeepSetsRegressor, build_run_config,
    checkpoint_payload, completion_state, expand_family_jobs, load_family,
    model_from_checkpoint, required_artifacts, run_job, seed_everything, train_model,
    validate_raw_dataset,
)
from src.models.deepsets import count_parameters, masked_mean_pool
from src.models.static_gcn import StaticGCNRegressor
from src.evaluation.baseline_common import build_prediction_rows, compute_metrics, write_prediction_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = REPO_ROOT / "configs/experiment_families/u1000_modern_deepsets.json"


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.model = DeepSetsRegressor(dropout=0.0).eval()
        self.x = torch.randn(3, 11, 7)
        self.mask = torch.tensor([[1] * 11, [1] * 8 + [0] * 3, [1] * 5 + [0] * 6], dtype=torch.bool)

    def test_input_output_and_mask_shapes(self):
        self.assertEqual(tuple(self.model(self.x, self.mask).shape), (3, 1))
        self.assertEqual(tuple(self.model.encode(self.x, self.mask).shape), (3, 11, 32))
        self.assertEqual(tuple(self.model.pool(self.model.encode(self.x, self.mask), self.mask).shape), (3, 32))
        with self.assertRaisesRegex(ValueError, "Expected x"):
            self.model(torch.randn(3, 11, 6), self.mask)
        with self.assertRaisesRegex(ValueError, "Expected mask"):
            self.model(self.x, self.mask.unsqueeze(-1))
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            self.model(self.x, self.mask.float())

    def test_masked_mean_correctness(self):
        encoded = torch.tensor([[[1., 2.], [3., 6.], [100., 200.]]])
        result = masked_mean_pool(encoded, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(result, torch.tensor([[2., 4.]]), rtol=0.0, atol=0.0)

    def test_permutation_invariance_synthetic(self):
        permutation = torch.randperm(self.x.shape[1])
        left = self.model(self.x, self.mask)
        right = self.model(self.x[:, permutation], self.mask[:, permutation])
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)

    def test_padding_invariance_multiple_lengths(self):
        reference = self.model(self.x, self.mask)
        for amount in (1, 7, 31):
            padded_x = torch.cat([self.x, torch.zeros(3, amount, 7)], dim=1)
            padded_mask = torch.cat([self.mask, torch.zeros(3, amount, dtype=torch.bool)], dim=1)
            torch.testing.assert_close(reference, self.model(padded_x, padded_mask), rtol=0.0, atol=1e-7)

    def test_all_masked_rejected(self):
        with self.assertRaisesRegex(ValueError, "all-masked"):
            self.model(torch.zeros(1, 4, 7), torch.zeros(1, 4, dtype=torch.bool))

    def test_no_graph_api_or_dependency(self):
        self.assertEqual(list(inspect.signature(self.model.forward).parameters), ["x", "mask"])
        source = inspect.getsource(inspect.getmodule(DeepSetsRegressor))
        for forbidden in ("GCNConv", "MessagePassing", "pairwise", "adjacency", "edge_index", "knn"):
            self.assertNotIn(forbidden, source)

    def test_cpu_forward_backward_and_finite_gradients(self):
        model = DeepSetsRegressor()
        loss = nn.MSELoss()(model(self.x, self.mask), torch.randn(3, 1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        model = DeepSetsRegressor().cuda()
        loss = nn.MSELoss()(model(self.x.cuda(), self.mask.cuda()), torch.randn(3, 1, device="cuda"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_parameter_counts(self):
        self.assertEqual(count_parameters(DeepSetsRegressor()), 4161)
        self.assertEqual(count_parameters(StaticGCNRegressor(node_features=7, hidden_dim=32, num_layers=3,
                                                              dropout=0.2, graph_pooling="mean")), 5281)

    def test_deterministic_seed_initialization(self):
        seed_everything(123); left = DeepSetsRegressor().state_dict()
        seed_everything(123); right = DeepSetsRegressor().state_dict()
        self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))

    def test_training_api_has_no_test_data(self):
        parameters = inspect.signature(train_model).parameters
        self.assertFalse(any("test" in name for name in parameters))
        self.assertIn("val_mse < best_mse", inspect.getsource(train_model))


class FamilyTests(unittest.TestCase):
    def test_family_and_feature_order(self):
        family = load_family(FAMILY_PATH)
        self.assertEqual(family["feature_names"], FEATURE_NAMES)
        self.assertEqual(family["representation"], "raw_halo_set")
        self.assertIs(family["uses_graph_edges"], False)
        jobs = expand_family_jobs(family, REPO_ROOT)
        self.assertEqual([(job["top_n"], job["seed"]) for job in jobs], [
            (500, 42), (500, 123), (500, 2025), (1000, 42), (1000, 123), (1000, 2025),
            (1500, 42), (1500, 123), (1500, 2025), (2000, 42), (2000, 123), (2000, 2025)])

    def test_real_manifest_counts_and_static_identity(self):
        jobs = expand_family_jobs(load_family(FAMILY_PATH), REPO_ROOT)
        for job in jobs:
            manifest = json.loads((REPO_ROOT / job["split_manifest_path"]).read_text())
            self.assertEqual([len(manifest[key]) for key in ("train_ids", "val_ids", "test_ids")], [700, 99, 201])
            static_path = REPO_ROOT / "experiments" / (
                f"static_gcn_u1000_top{job['top_n']}_sparse_train700_seed{job['seed']}_none_h32_l3_mean_mlp_final/config.json")
            self.assertTrue(static_path.is_file())
            static = json.loads(static_path.read_text())
            self.assertEqual(static["train_ids"], manifest["train_ids"])
            self.assertEqual(static["val_ids"], manifest["val_ids"])
            self.assertEqual(static["test_ids"], manifest["test_ids"])

    def test_final_snapshot_dataset_validation(self):
        samples = {}
        for index in range(1000):
            samples[f"LH_{index}"] = {"Nodes_list": [torch.zeros(2, 7)], "mask_list": [torch.ones(2, 1)],
                                      "target": torch.tensor(0.3), "snapshots": [1.0],
                                      "feature_names": FEATURE_NAMES, "normalization": "none"}
        validate_raw_dataset(samples, 2)
        samples["LH_0"]["snapshots"] = [0.75]
        with self.assertRaisesRegex(ValueError, "a=1.0"):
            validate_raw_dataset(samples, 2)


class ArtifactTests(unittest.TestCase):
    def _job(self):
        return {"model_family": "deepsets", "top_n": 500, "seed": 42,
                "experiment_name": "deepsets_synthetic", "dataset_path": "synthetic.pt",
                "dataset_sha256": "dataset", "split_manifest_path": "synthetic.json",
                "split_manifest_sha256": "manifest", "split_manifest_canonical_sha256": "canonical",
                "split_counts": {"population": 1000, "train": 700, "val": 99, "test": 201, "unused": 0}}

    def test_checkpoint_reload_prediction_reproduction(self):
        job = self._job(); seed_everything(2025); model = DeepSetsRegressor().eval()
        x, mask = torch.randn(4, 9, 7), torch.ones(4, 9, dtype=torch.bool)
        before = model(x, mask)
        restored = model_from_checkpoint(checkpoint_payload(model, 3, 0.1, job))
        torch.testing.assert_close(before, restored(x, mask), rtol=0.0, atol=0.0)

    def test_prediction_schema_and_metrics(self):
        rows = build_prediction_rows(["LH_0", "LH_1"], [0.2, 0.4], [0.25, 0.35])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_prediction_csv(rows, path)
            with path.open(newline="") as handle:
                self.assertEqual(next(csv.reader(handle)), ["universe_id", "true_omega_m", "pred_omega_m",
                                                             "absolute_error", "squared_error"])
        metrics = compute_metrics(rows)
        self.assertEqual(metrics["num_samples"], 2)
        self.assertTrue(all(np.isfinite(metrics[key]) for key in ("mse", "rmse", "mae", "r2")))

    def test_completed_partial_and_incompatible_collision(self):
        job = self._job(); config = build_run_config(job, REPO_ROOT, "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            self.assertEqual(completion_state(root, config), "missing")
            root.mkdir(); (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Partial"):
                completion_state(root, config)
            for relative in required_artifacts():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
            (root / "config.json").write_text(json.dumps(config))
            self.assertEqual(completion_state(root, config), "complete")
            incompatible = dict(config); incompatible["seed"] = 123
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                completion_state(root, incompatible)

    def test_completed_run_skips_before_loading_or_training(self):
        job = self._job(); config = build_run_config(job, REPO_ROOT, "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); destination = root / job["experiment_name"]
            for relative in required_artifacts():
                path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
            (destination / "config.json").write_text(json.dumps(config))
            self.assertEqual(run_job(job, root, REPO_ROOT, "cpu"), "skipped_complete")


if __name__ == "__main__":
    unittest.main()
