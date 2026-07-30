from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.experiment_pipeline.common import recompute_metrics, verify_family


class ExperimentFamilyVerificationTests(unittest.TestCase):
    def _fixture(
        self,
        predictions: list[float],
        *,
        prediction_ids: list[str] | None = None,
        metric_offset: float = 0.0,
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, object]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        experiment_name = "fixture_experiment"
        experiment_path = root / "experiments" / experiment_name
        prediction_path = experiment_path / "predictions" / "test_predictions.csv"
        prediction_path.parent.mkdir(parents=True)
        (experiment_path / "checkpoints").mkdir()
        (experiment_path / "checkpoints" / "best_model.pt").write_bytes(b"fixture")
        (experiment_path / "train_log.csv").write_text(
            "epoch,train_loss\n1,0.1\n", encoding="utf-8"
        )

        train_ids = ["LH_train"]
        val_ids = ["LH_val"]
        test_ids = ["LH_1", "LH_2", "LH_3"]
        targets = [0.1, 0.2, 0.4]
        config = {
            "seed": 42,
            "dataset_path": "datasets/fixture.pt",
            "experiment_name": experiment_name,
            "output_root": "experiments",
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }
        (experiment_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        dataset_path = root / "datasets" / "fixture.pt"
        dataset_path.parent.mkdir()
        dataset_path.write_bytes(b"not loaded")
        dataset_path.with_suffix(".metadata.json").write_text(
            "{}\n", encoding="utf-8"
        )

        ids = prediction_ids if prediction_ids is not None else test_ids
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("universe_id", "true_omega_m", "pred_omega_m"))
            for universe_id, target, prediction in zip(ids, targets, predictions):
                writer.writerow((universe_id, target, prediction))

        finite_pairs = [
            (target, prediction)
            for target, prediction in zip(targets, predictions)
            if math.isfinite(prediction)
        ]
        metrics = recompute_metrics(
            [pair[0] for pair in finite_pairs],
            [pair[1] for pair in finite_pairs],
        )
        saved = {
            "best_epoch": 1,
            "test": {
                "mae": metrics["test_mae"] + metric_offset,
                "rmse": metrics["test_rmse"] + metric_offset,
                "mse": metrics["test_mse"],
                "num_samples": len(predictions),
            },
        }
        (experiment_path / "metrics.json").write_text(
            json.dumps(saved), encoding="utf-8"
        )

        spec: dict[str, object] = {
            "schema_version": "1.0",
            "family_id": "fixture_family",
            "family_title": "Fixture family",
            "scientific_question": "Verifier behavior",
            "model_name": "Fixture",
            "training_module": "unused",
            "output_root": "experiments",
            "grouping_field": "case",
            "grouping_values": ["only"],
            "required_seeds": [42],
            "prediction_file": "predictions/test_predictions.csv",
            "expected_artifacts": [
                "config.json",
                "metrics.json",
                "train_log.csv",
                "predictions/test_predictions.csv",
                "checkpoints/best_model.pt",
            ],
            "fixed_scientific_settings": {
                "config": {},
                "dataset_metadata": {},
            },
            "allowed_varying_fields": [],
            "target_column_aliases": ["true_omega_m"],
            "prediction_column_aliases": ["pred_omega_m"],
            "runner": {},
            "metric_tolerance": 1e-6,
            "runs": [
                {
                    "group_value": "only",
                    "seed": 42,
                    "dataset_path": "datasets/fixture.pt",
                    "experiment_name": experiment_name,
                    "experiment_path": f"experiments/{experiment_name}",
                    "origin": "planned_new",
                    "action": "run_if_missing",
                }
            ],
        }
        return temporary, root, spec

    def test_variable_predictions_have_finite_pearson_and_pass(self) -> None:
        temporary, root, spec = self._fixture([0.12, 0.19, 0.38])
        self.addCleanup(temporary.cleanup)
        result = verify_family(root, spec, allow_incomplete=False)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.rows[0]["pearson_status"], "defined")
        self.assertTrue(math.isfinite(result.rows[0]["test_pearson"]))

    def test_constant_predictions_warn_and_pass(self) -> None:
        temporary, root, spec = self._fixture([0.3, 0.3, 0.3])
        self.addCleanup(temporary.cleanup)
        result = verify_family(root, spec, allow_incomplete=False)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(
            result.rows[0]["pearson_status"],
            "undefined_zero_prediction_variance",
        )
        self.assertTrue(math.isnan(result.rows[0]["test_pearson"]))
        self.assertTrue(result.warnings)

    def test_nan_prediction_fails(self) -> None:
        temporary, root, spec = self._fixture([0.12, float("nan"), 0.38])
        self.addCleanup(temporary.cleanup)
        result = verify_family(root, spec, allow_incomplete=False)
        self.assertFalse(result.valid)
        self.assertTrue(any("non-finite prediction" in error for error in result.errors))

    def test_missing_prediction_row_fails(self) -> None:
        temporary, root, spec = self._fixture(
            [0.12, 0.19],
            prediction_ids=["LH_1", "LH_2"],
        )
        self.addCleanup(temporary.cleanup)
        result = verify_family(root, spec, allow_incomplete=False)
        self.assertFalse(result.valid)
        self.assertTrue(
            any(
                "ordered prediction universe IDs" in error
                or "test split IDs" in error
                for error in result.errors
            )
        )

    def test_mismatched_mae_and_rmse_fail(self) -> None:
        temporary, root, spec = self._fixture(
            [0.12, 0.19, 0.38],
            metric_offset=0.1,
        )
        self.addCleanup(temporary.cleanup)
        result = verify_family(root, spec, allow_incomplete=False)
        self.assertFalse(result.valid)
        self.assertTrue(any("saved test_mae" in error for error in result.errors))
        self.assertTrue(any("saved test_rmse" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
