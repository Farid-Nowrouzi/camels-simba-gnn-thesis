from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluation.baseline_common import (
    PREDICTION_COLUMNS,
    build_prediction_rows,
    completion_state,
    compute_metrics,
    read_prediction_csv,
    write_json,
    write_prediction_csv,
)
from src.evaluation.run_modern_summary_baselines import (
    build_model,
    expand_family_jobs,
    experiment_name,
    load_family,
    run_job,
)
from src.evaluation.summary_features import (
    SUMMARY_FEATURE_NAMES,
    arrays_for_ids,
    extract_summary,
    select_snapshot,
    summarize_nodes,
)
from src.training.split_manifest import load_split_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = REPO_ROOT / "configs/experiment_families/u1000_modern_classical_baselines.json"


def sample(nodes_by_time: list[torch.Tensor], masks_by_time: list[torch.Tensor], target: float = 0.3):
    scales = [1.0] if len(nodes_by_time) == 1 else [0.5, 1.0]
    return {
        "Nodes_list": nodes_by_time,
        "mask_list": masks_by_time,
        "snapshots": [{"scale_factor": value} for value in scales],
        "target": torch.tensor(target),
    }


class SummaryFeatureTests(unittest.TestCase):
    def test_feature_names_exact_order(self):
        self.assertEqual(len(SUMMARY_FEATURE_NAMES), 20)
        self.assertEqual(SUMMARY_FEATURE_NAMES, [
            "halo_count", "log10_Mvir_mean", "log10_Mvir_std", "log10_Mvir_min",
            "log10_Mvir_max", "log10_Mvir_median", "X_mean", "Y_mean", "Z_mean",
            "X_std", "Y_std", "Z_std", "VX_mean", "VY_mean", "VZ_mean",
            "VX_std", "VY_std", "VZ_std", "speed_mean", "speed_std",
        ])

    def test_final_snapshot_selection(self):
        early = torch.zeros((2, 7))
        final = torch.ones((2, 7))
        nodes, mask = select_snapshot(sample([early, final], [torch.ones(2, 1)] * 2))
        np.testing.assert_array_equal(nodes, final.numpy())
        self.assertEqual(mask.shape, (2, 1))
        with self.assertRaises(ValueError):
            select_snapshot(sample([early], [torch.ones(2, 1)]), "temporal")
        wrong = sample([early], [torch.ones(2, 1)])
        wrong["snapshots"][-1]["scale_factor"] = 0.75
        with self.assertRaisesRegex(ValueError, "not a=1.0"):
            select_snapshot(wrong)

    def test_mask_speed_and_population_sd(self):
        nodes = np.asarray([
            [1, 1, 2, 3, 3, 4, 0],
            [3, 3, 4, 5, 0, 0, 0],
            [999, 999, 999, 999, 999, 999, 999],
        ], dtype=np.float64)
        result = summarize_nodes(nodes, np.asarray([[1], [1], [0]]))
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 2)
        self.assertEqual(result[2], 1)  # ddof=0, not sample SD sqrt(2)
        self.assertEqual(result[18], 2.5)  # speeds 5 and 0
        self.assertEqual(result[19], 2.5)
        self.assertLess(result[4], 999)  # masked padding excluded

    def test_target_never_enters_summary(self):
        nodes = torch.arange(14, dtype=torch.float32).reshape(2, 7)
        mask = torch.ones(2, 1)
        left = extract_summary(sample([nodes], [mask], target=0.1))
        right = extract_summary(sample([nodes], [mask], target=0.9))
        np.testing.assert_array_equal(left, right)


class SplitAndModelTests(unittest.TestCase):
    def test_real_manifest_binding_and_exact_counts(self):
        path = REPO_ROOT / "configs/splits/u1000_top500_none_k8_sparse/seed42_train700.json"
        raw = json.loads(path.read_text())
        ids = [f"LH_{index}" for index in range(1000)]
        manifest = load_split_manifest(path, ids, raw["dataset_identity"], expected_seed=42)
        self.assertEqual([len(manifest[key]) for key in ("train_ids", "val_ids", "test_ids")], [700, 99, 201])
        self.assertEqual(manifest["counts"]["unused"], 0)
        ordered = manifest["train_ids"][:3]
        features = np.arange(20000).reshape(1000, 20)
        targets = np.arange(1000)
        X, y = arrays_for_ids(ordered, ids, features, targets)
        self.assertEqual(y.tolist(), [int(item.split("_")[1]) for item in ordered])
        self.assertEqual(X.shape, (3, 20))

    def test_ridge_scaler_is_fit_on_training_rows_only(self):
        X_train = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        y_train = np.asarray([0.1, 0.2, 0.3])
        X_held_out = np.asarray([[1000.0, 2000.0]])
        model = build_model("ridge", seed=42)
        model.fit(X_train, y_train)
        scaler = model.named_steps["standard_scaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, np.vstack([X_train, X_held_out]).mean(axis=0)))

    def test_sklearn_seeds_and_fixed_hyperparameters(self):
        rf = build_model("random_forest", 2025)
        gb = build_model("gradient_boosting", 123)
        self.assertEqual((rf.random_state, rf.n_estimators, rf.min_samples_leaf, rf.n_jobs), (2025, 300, 2, -1))
        self.assertIsNone(rf.max_depth)
        self.assertEqual((gb.random_state, gb.n_estimators, gb.learning_rate, gb.max_depth), (123, 300, 0.03, 3))

    def test_family_matrix_and_mean_aliases(self):
        family = load_family(FAMILY_PATH)
        jobs = expand_family_jobs(family)
        self.assertEqual(len(jobs), 39)
        self.assertEqual(sum(job["model_family"] == "mean" for job in jobs), 3)
        self.assertEqual(sum(job["model_family"] != "mean" for job in jobs), 36)
        self.assertEqual(experiment_name("mean", 42), "mean_baseline_u1000_train700_seed42_targetmean")
        aliases = family["mean_baseline"]["aliases"]
        self.assertEqual(len(set(aliases.values())), 1)


class MetricAndArtifactTests(unittest.TestCase):
    def test_metrics_match_graph_formula(self):
        rows = build_prediction_rows(["LH_2", "LH_1"], [1.0, 3.0], [2.0, 2.0])
        self.assertEqual(compute_metrics(rows), {
            "mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.0, "num_samples": 2,
        })

    def test_prediction_schema_order_and_metric_reproduction(self):
        rows = build_prediction_rows(["LH_9", "LH_3"], [0.2, 0.4], [0.25, 0.35])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_prediction_csv(rows, path)
            with path.open(newline="") as handle:
                self.assertEqual(next(csv.reader(handle)), PREDICTION_COLUMNS)
            loaded = read_prediction_csv(path)
            self.assertEqual([row["universe_id"] for row in loaded], ["LH_9", "LH_3"])
            self.assertEqual(compute_metrics(loaded), compute_metrics(rows))

    def _config(self, name="test", family="mean"):
        return {
            "experiment_name": name, "model_family": family,
            "representation": "engineered_summary", "snapshot_protocol": "final",
            "summary_definition_version": "u1000_final_summary20_v1", "top_n": None,
            "seed": 42, "dataset_sha256": "d", "split_manifest_sha256": "s",
            "target": "Omega_m", "target_normalization": "none",
            "model_hyperparameters": {"statistic": "training_target_mean"},
        }

    def test_completion_partial_and_collision_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            self.assertEqual(completion_state(root, self._config()), "missing")
            root.mkdir()
            write_json(self._config(), root / "config.json")
            with self.assertRaisesRegex(RuntimeError, "Partial"):
                completion_state(root, self._config())
            write_json({"train": {}, "val": {}, "test": {}}, root / "metrics.json")
            for filename in ("train_predictions.csv", "val_predictions.csv", "test_predictions.csv"):
                write_prediction_csv([], root / "predictions" / filename)
            self.assertEqual(completion_state(root, self._config()), "complete")
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                completion_state(root, self._config(name="different"))

    def test_synthetic_ridge_artifact_and_verifier_reload(self):
        verifier_path = REPO_ROOT / "scripts/verify_u1000_modern_classical_baselines.py"
        spec = importlib.util.spec_from_file_location("modern_verifier", verifier_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        ids = [f"LH_{index}" for index in range(1000)]
        rng = np.random.default_rng(9)
        features = rng.normal(size=(1000, 20))
        targets = 0.3 + 0.01 * features[:, 0]
        manifest = {
            "train_ids": ids[:700], "val_ids": ids[700:799], "test_ids": ids[799:],
            "counts": {"train": 700, "val": 99, "test": 201, "unused": 0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "synthetic.pt"
            manifest_path = root / "split.json"
            dataset_path.write_bytes(b"synthetic dataset identity")
            manifest_path.write_text("{}")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            job = {
                "model_family": "ridge", "top_n": 500, "seed": 42,
                "experiment_name": experiment_name("ridge", 42, 500),
                "dataset_path": str(dataset_path), "dataset_sha256": digest(dataset_path),
                "split_manifest_path": str(manifest_path), "split_manifest_sha256": digest(manifest_path),
            }
            bound = (ids, features, targets, manifest)
            self.assertEqual(run_job(job, root / "experiments", REPO_ROOT, bound), "completed")
            experiment_dir = root / "experiments" / job["experiment_name"]
            self.assertTrue((experiment_dir / "model.joblib").is_file())
            result = module.verify_run(job, experiment_dir, bound)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(run_job(job, root / "experiments", REPO_ROOT, bound), "skipped_complete")

    def test_mean_prediction_uses_training_targets_only(self):
        ids = [f"LH_{index}" for index in range(1000)]
        features = np.zeros((1000, 20))
        targets = np.concatenate([np.full(700, 0.2), np.full(300, 0.9)])
        manifest = {
            "train_ids": ids[:700], "val_ids": ids[700:799], "test_ids": ids[799:],
            "counts": {"train": 700, "val": 99, "test": 201, "unused": 0},
        }
        job = {
            "model_family": "mean", "top_n": None, "seed": 42,
            "experiment_name": experiment_name("mean", 42),
            "dataset_path": "synthetic.pt", "dataset_sha256": "synthetic",
            "split_manifest_path": "split.json", "split_manifest_sha256": "synthetic",
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_job(job, Path(temporary), REPO_ROOT, (ids, features, targets, manifest))
            prediction_path = Path(temporary) / job["experiment_name"] / "predictions/test_predictions.csv"
            predictions = [row["pred_omega_m"] for row in read_prediction_csv(prediction_path)]
            np.testing.assert_allclose(predictions, 0.2, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
