from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.evaluation.baseline_common import (
    SPLIT_FILENAMES,
    build_prediction_rows,
    compute_metrics,
    write_json,
    write_prediction_csv,
)
from src.evaluation.run_modern_summary_mlp import (
    ARCHITECTURE_VERSION,
    INPUT_DIM,
    SummaryMLP,
    build_run_config,
    checkpoint_payload,
    completion_state,
    expand_family_jobs,
    extract_dataset_summaries,
    fit_feature_scaler,
    load_checkpoint,
    load_family,
    model_from_checkpoint,
    predict,
    required_artifacts,
    run_job,
    seed_everything,
    train_model,
    transform_features,
    write_train_log,
)
from src.evaluation import summary_features
from src.evaluation.summary_features import (
    SUMMARY_DEFINITION_VERSION,
    SUMMARY_FEATURE_NAMES,
    extract_summary,
)
from src.training.split_manifest import load_split_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = REPO_ROOT / "configs/experiment_families/u1000_modern_summary_mlp.json"
VERIFIER_PATH = REPO_ROOT / "scripts/verify_u1000_modern_summary_mlp.py"


class ArchitectureAndFeatureTests(unittest.TestCase):
    def test_fixed_architecture_and_output_shape(self):
        model = SummaryMLP()
        linear = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        dropout = [layer.p for layer in model.net if isinstance(layer, nn.Dropout)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear], [(20, 64), (64, 64), (64, 1)])
        self.assertEqual(dropout, [0.2, 0.2])
        self.assertEqual(tuple(model(torch.zeros(7, INPUT_DIM)).shape), (7, 1))
        with self.assertRaises(ValueError):
            model(torch.zeros(7, 100))

    def test_shared_final_snapshot_extractor_and_feature_identity(self):
        self.assertIs(extract_dataset_summaries, summary_features.extract_dataset_summaries)
        self.assertEqual(SUMMARY_DEFINITION_VERSION, "u1000_final_summary20_v1")
        self.assertEqual(len(SUMMARY_FEATURE_NAMES), 20)
        early = torch.zeros((3, 7))
        final = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        sample = {
            "Nodes_list": [early, final],
            "mask_list": [torch.ones(3, 1), torch.tensor([[1], [1], [0]])],
            "snapshots": [{"scale_factor": 0.5}, {"scale_factor": 1.0}],
            "target": torch.tensor(0.3),
        }
        result = extract_summary(sample)
        self.assertEqual(result.shape, (20,))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0], 2)

    def test_train_only_scaling_and_constant_feature_safety(self):
        train = np.tile(np.arange(20, dtype=np.float64), (700, 1))
        train[:, 1] = np.arange(700)
        held_out = np.full((300, 20), 1e6)
        mean, scale = fit_feature_scaler(train)
        self.assertTrue(np.array_equal(mean, train.mean(axis=0)))
        expected_scale = train.std(axis=0, ddof=0)
        expected_scale[expected_scale == 0] = 1.0
        self.assertTrue(np.array_equal(scale, expected_scale))
        self.assertFalse(np.allclose(mean, np.vstack([train, held_out]).mean(axis=0)))
        self.assertTrue(np.isfinite(transform_features(train, mean, scale)).all())

    def test_cpu_forward_backward_finite(self):
        generator = seed_everything(42)
        inputs = torch.randn(16, 20, generator=generator)
        targets = torch.randn(16, 1, generator=generator)
        model = SummaryMLP()
        loss = nn.MSELoss()(model(inputs), targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_seed_setup_reproduces_initialization(self):
        seed_everything(123)
        left = SummaryMLP().state_dict()
        seed_everything(123)
        right = SummaryMLP().state_dict()
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]))

    def test_training_api_cannot_receive_test_data(self):
        parameters = inspect.signature(train_model).parameters
        self.assertNotIn("test_features", parameters)
        self.assertNotIn("test_targets", parameters)
        source = inspect.getsource(train_model)
        self.assertIn("validation_mse < best_validation_mse", source)


class FamilyAndManifestTests(unittest.TestCase):
    def test_family_is_exact_twelve_job_matrix(self):
        jobs = expand_family_jobs(load_family(FAMILY_PATH), REPO_ROOT)
        self.assertEqual(len(jobs), 12)
        self.assertEqual([(j["top_n"], j["seed"]) for j in jobs], [
            (500, 42), (500, 123), (500, 2025),
            (1000, 42), (1000, 123), (1000, 2025),
            (1500, 42), (1500, 123), (1500, 2025),
            (2000, 42), (2000, 123), (2000, 2025),
        ])
        self.assertNotIn(750, {job["top_n"] for job in jobs})

    def test_real_manifest_identity_counts_and_coverage(self):
        path = REPO_ROOT / "configs/splits/u1000_top500_none_k8_sparse/seed42_train700.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        manifest = load_split_manifest(path, [f"LH_{i}" for i in range(1000)], raw["dataset_identity"], 42)
        self.assertEqual([len(manifest[key]) for key in ("train_ids", "val_ids", "test_ids", "unused_ids")], [700, 99, 201, 0])
        combined = manifest["train_ids"] + manifest["val_ids"] + manifest["test_ids"] + manifest["unused_ids"]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(len(combined), 1000)


class ArtifactAndVerifierTests(unittest.TestCase):
    def _job_and_bound_data(self):
        ids = [f"LH_{i}" for i in range(1000)]
        rng = np.random.default_rng(7)
        features = rng.normal(size=(1000, 20))
        features[:, 0] = 500
        targets = 0.3 + 0.01 * features[:, 1]
        manifest = {
            "train_ids": ids[:700], "val_ids": ids[700:799], "test_ids": ids[799:],
            "counts": {"population": 1000, "train": 700, "val": 99, "test": 201, "unused": 0},
        }
        job = {
            "model_family": "summary_mlp", "top_n": 500, "seed": 42,
            "experiment_name": "summary_mlp_synthetic",
            "dataset_path": "synthetic.pt", "dataset_sha256": "dataset",
            "split_manifest_path": "synthetic.json", "split_manifest_sha256": "manifest",
            "split_manifest_canonical_sha256": "canonical", "split_counts": manifest["counts"],
        }
        return job, (ids, features, targets, manifest)

    def _write_synthetic_artifact(self, root: Path):
        job, bound = self._job_and_bound_data()
        ids, features, targets, manifest = bound
        mean, scale = fit_feature_scaler(features[:700])
        seed_everything(42)
        model = SummaryMLP()
        config = build_run_config(job, REPO_ROOT, "cpu")
        write_json(config, root / "config.json")
        metrics = {}
        for split, key, metric_key in (("train", "train_ids", "train"), ("val", "val_ids", "validation"), ("test", "test_ids", "test")):
            indexes = [int(item.split("_")[1]) for item in manifest[key]]
            split_x = features[indexes]
            split_y = targets[indexes]
            predictions = predict(model, transform_features(split_x, mean, scale), torch.device("cpu"))
            rows = build_prediction_rows(manifest[key], split_y, predictions)
            write_prediction_csv(rows, root / "predictions" / SPLIT_FILENAMES[split])
            metrics[metric_key] = compute_metrics(rows)
        metrics["best_epoch"] = 1
        metrics["best_validation_mse"] = metrics["validation"]["mse"]
        write_json(metrics, root / "metrics.json")
        write_train_log([{
            "epoch": 1, "train_mse": metrics["train"]["mse"],
            "validation_mse": metrics["validation"]["mse"],
            "validation_rmse": metrics["validation"]["rmse"],
            "validation_mae": metrics["validation"]["mae"],
            "learning_rate": 0.001, "improved": True, "patience_counter": 0,
        }], root / "train_log.csv")
        payload = checkpoint_payload(model, mean, scale, 1, metrics["validation"]["mse"], job)
        (root / "checkpoints").mkdir(parents=True)
        torch.save(payload, root / "checkpoints" / "best_model.pt")
        return job, bound

    def test_prediction_schema_and_metric_reproduction(self):
        rows = build_prediction_rows(["LH_2", "LH_1"], [0.2, 0.4], [0.25, 0.35])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_prediction_csv(rows, path)
            with path.open(newline="") as handle:
                self.assertEqual(next(csv.reader(handle)), [
                    "universe_id", "true_omega_m", "pred_omega_m", "absolute_error", "squared_error",
                ])
            self.assertEqual(compute_metrics(rows)["num_samples"], 2)

    def test_checkpoint_reload_reproduces_predictions(self):
        job, bound = self._job_and_bound_data()
        features = bound[1]
        mean, scale = fit_feature_scaler(features[:700])
        seed_everything(2025)
        model = SummaryMLP()
        before = predict(model, transform_features(features[:32], mean, scale), torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best_model.pt"
            torch.save(checkpoint_payload(model, mean, scale, 3, 0.1, job), path)
            checkpoint = load_checkpoint(path)
            after = predict(model_from_checkpoint(checkpoint), transform_features(features[:32], checkpoint["scaler_mean"], checkpoint["scaler_scale"]), torch.device("cpu"))
        np.testing.assert_allclose(before, after, rtol=0.0, atol=0.0)
        self.assertEqual(checkpoint["architecture_version"], ARCHITECTURE_VERSION)

    def test_verifier_accepts_complete_synthetic_artifact(self):
        spec = importlib.util.spec_from_file_location("modern_summary_mlp_verifier", VERIFIER_PATH)
        verifier = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, bound = self._write_synthetic_artifact(root)
            result = verifier.verify_run(job, root, bound_data=bound, hashes_verified=True)
        self.assertEqual(result["status"], "verified")

    def test_completed_partial_and_incompatible_collision_states(self):
        job, _ = self._job_and_bound_data()
        config = build_run_config(job, REPO_ROOT, "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            self.assertEqual(completion_state(root, config), "missing")
            root.mkdir()
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Partial"):
                completion_state(root, config)
            for relative in required_artifacts():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            write_json(config, root / "config.json")
            self.assertEqual(completion_state(root, config), "complete")
            incompatible = dict(config)
            incompatible["seed"] = 123
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                completion_state(root, incompatible)

    def test_completed_run_is_skipped_without_training(self):
        job, bound = self._job_and_bound_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            destination = output_root / job["experiment_name"]
            self._write_synthetic_artifact(destination)
            result = run_job(job, output_root, REPO_ROOT, "cpu", bound)
        self.assertEqual(result, "skipped_complete")


if __name__ == "__main__":
    unittest.main()
