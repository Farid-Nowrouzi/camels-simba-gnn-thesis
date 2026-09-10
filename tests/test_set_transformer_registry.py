"""Read-only registry regression checks; no model loading or inference."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_registry import build_experiment_rows, discover_artifacts


ROOT = Path(__file__).resolve().parents[1]
RUN = "set_transformer_u1000_top1500_raw7_final_train700_seed42_none_d16_h2_m32_isab2_pma1"


class SetTransformerRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "experiments" / RUN
        self.run.mkdir(parents=True)
        for relative in ("config.json", "metrics.json", "run_metadata.json", "train_log.csv",
                         "predictions/test_predictions.csv"):
            destination = self.run / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "experiments" / RUN / relative, destination)
        config = json.loads((self.run / "config.json").read_text())
        split = self.root / config["split_manifest_path"]
        split.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / config["split_manifest_path"], split)
        (self.run / "checkpoints").mkdir()
        (self.run / "checkpoints/best_model.pt").write_bytes(b"presence-only registry fixture")

    def scan(self):
        rows, errors = build_experiment_rows(self.root, self.root / "experiments")
        self.assertEqual(len(rows), 1)
        return rows[0], errors

    def test_finalized_nested_config_and_bound_split(self):
        row, errors = self.scan()
        self.assertEqual(errors, [])
        self.assertEqual(row["status"], "completed")
        self.assertEqual([row[k + "_count"] for k in ("train", "val", "test")], [700, 99, 201])
        self.assertEqual(row["snapshot_input_protocol"], "final")
        self.assertTrue(row["final_snapshot_only"])
        self.assertEqual(row["snapshots"], 1)
        self.assertEqual(row["node_features"], 7)
        self.assertEqual(row["k"], "")
        self.assertEqual(row["trainable_parameters"], 11313)
        self.assertAlmostEqual(row["test_mae"], row["test_mae_recomputed"], places=12)

    def test_attempt_without_finalization_is_partial(self):
        path = self.run / "run_metadata.json"
        metadata = json.loads(path.read_text())
        metadata.update(state="trained", test_status="pending")
        path.write_text(json.dumps(metadata))
        row, _ = self.scan()
        self.assertEqual(row["status"], "partial")

    def test_smoke_cannot_be_matched_as_scientific_output(self):
        smoke = self.root / "outputs/smoke/notebook17_set_transformer"
        smoke.mkdir(parents=True)
        (smoke / "config.json").write_text(json.dumps({"experiment_name": RUN}))
        self.assertEqual(discover_artifacts(self.root, [RUN]), [])

    def test_changed_split_and_missing_checkpoint_are_rejected(self):
        config = json.loads((self.run / "config.json").read_text())
        split = self.root / config["split_manifest_path"]
        split.write_text(split.read_text() + "\n")
        row, errors = self.scan()
        self.assertEqual(row["status"], "partial")
        self.assertTrue(any("SHA256 mismatch" in error for error in errors))
        shutil.copyfile(ROOT / config["split_manifest_path"], split)
        (self.run / "checkpoints/best_model.pt").unlink()
        row, errors = self.scan()
        self.assertEqual(errors, [])
        self.assertEqual(row["status"], "partial")
        self.assertIn("checkpoints/best_model.pt", row["missing_required_artifacts"])


if __name__ == "__main__":
    unittest.main()
