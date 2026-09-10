from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

import scripts.validation.prepare_u1000_top1500_evolvegcn_o as auditor
import src.training.temporal_architecture_common as common


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/production/run_u1000_top1500_evolvegcn_o_recovery_123_2025.sh"


class TinyRegressor(nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.regressor = nn.Linear(1, 1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticLifecycle:
    def __init__(self, root: Path, seed: int):
        self.root = root
        self.seed = seed
        self.dataset = root / f"dataset-{seed}.bin"
        self.manifest = root / f"manifest-{seed}.json"
        self.dataset.write_bytes(b"synthetic-dataset")
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.name = f"tiny-evolvegcn-o-seed{seed}"
        self.config_path = root / f"seed{seed}.json"
        self.config = {
            "model": "TinyEvolveGCNO", "experiment_name": self.name,
            "output_root": str(root / "experiments"),
            "dataset_path": str(self.dataset), "dataset_sha256": digest(self.dataset),
            "split_manifest_path": str(self.manifest),
            "split_manifest_sha256": digest(self.manifest), "seed": seed,
            "scale_factors": [0.2, 0.25, 0.51209, 0.75065, 1.0],
            "device": "cpu", "epochs": 5, "patience": 2,
            "batch_size": 2,
            "learning_rate": 0.001, "weight_decay": 1e-5,
            "scheduler_factor": 0.5, "scheduler_patience": 1,
            "scheduler_min_lr": 1e-6, "grad_clip_norm": 1.0,
            "parameter_count": 2, "hidden_dim": 1, "num_layers": 1,
            "graph_pooling": "mean", "temporal_pooling": "mean", "head_type": "linear",
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.loaders = ([object()], [object()], [object()])
        self.ids = {
            id(self.loaders[0]): ["train-0", "train-1"],
            id(self.loaders[1]): ["val-0", "val-1"],
            id(self.loaders[2]): ["test-0", "test-1"],
        }

    def predictions(self, _model, loader, _device):
        return [
            {"universe_id": universe_id, "target": float(index), "prediction": float(index) + 0.1}
            for index, universe_id in enumerate(self.ids[id(loader)])
        ]

    def patches(self):
        return mock.patch.multiple(
            common,
            load_temporal_dataset=mock.DEFAULT,
            create_loaders=mock.DEFAULT,
            unpack_batch=mock.DEFAULT,
            validate_example_batch=mock.DEFAULT,
            run_one_epoch=mock.DEFAULT,
            evaluate_loss=mock.DEFAULT,
            collect_predictions=mock.DEFAULT,
            load_dataset_provenance=mock.DEFAULT,
            current_repository_commit=mock.DEFAULT,
        )

    def configure(self, patched):
        patched["load_temporal_dataset"].return_value = {"synthetic": {}}
        patched["create_loaders"].return_value = (
            *self.loaders,
            self.ids[id(self.loaders[0])], self.ids[id(self.loaders[1])],
            self.ids[id(self.loaders[2])], None, None,
        )
        patched["unpack_batch"].return_value = (
            ["train-0"], torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), None
        )
        patched["run_one_epoch"].return_value = 1.0
        patched["evaluate_loss"].side_effect = [3.0, 2.0, 1.0, 1.5, 2.0]
        patched["collect_predictions"].side_effect = self.predictions
        patched["load_dataset_provenance"].return_value = {"synthetic": True}
        patched["current_repository_commit"].return_value = "synthetic-head"

    def train(self):
        with self.patches() as patched:
            self.configure(patched)
            return common.train_from_config(
                self.config_path, "TinyEvolveGCNO", TinyRegressor,
                frozen_values={"epochs": 5, "patience": 2},
            )


class EvolveGCNORecoveryTests(unittest.TestCase):
    def test_seed42_and_seed123_full_lifecycle_checkpoint_reload_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (42, 123):
                with self.subTest(seed=seed):
                    fixture = SyntheticLifecycle(root, seed)
                    fixture.train()
                    run = root / "experiments" / fixture.name
                    self.assertTrue((run / "checkpoints/best_model.pt").is_file())
                    checkpoint = torch.load(
                        run / "checkpoints/best_model.pt", map_location="cpu", weights_only=False
                    )
                    restored = TinyRegressor(fixture.config)
                    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    with (run / "train_log.csv").open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 5)
                    self.assertEqual(int(rows[-1]["epoch"]) - int(rows[-1]["best_epoch"]), 2)
                    for relative in (
                        "metrics.json", "train_log.csv", "predictions/val_predictions.csv",
                        "predictions/test_predictions.csv",
                    ):
                        self.assertTrue((run / relative).is_file(), relative)

    def test_inference_only_finalization_preserves_training_evidence_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticLifecycle(Path(temporary), 123)
            fixture.train()
            run = Path(temporary) / "experiments" / fixture.name
            for path in (run / "metrics.json", *sorted((run / "predictions").glob("*.csv"))):
                path.unlink()
            protected = {
                relative: digest(run / relative)
                for relative in (
                    "config.json", "run_metadata.json", "train_log.csv", "checkpoints/best_model.pt"
                )
            }
            with fixture.patches() as patched:
                fixture.configure(patched)
                patched["evaluate_loss"].side_effect = None
                with mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")):
                    common.finalize_existing_run(
                        fixture.config_path, "TinyEvolveGCNO", TinyRegressor,
                        expected_checkpoint_sha256=protected["checkpoints/best_model.pt"],
                        frozen_values={"epochs": 5, "patience": 2},
                    )
            self.assertEqual(
                protected,
                {relative: digest(run / relative) for relative in protected},
            )
            recovery = json.loads((run / "recovery_metadata.json").read_text())
            self.assertTrue(recovery["no_additional_optimization_performed"])
            self.assertTrue(recovery["checkpoint_unchanged"])
            self.assertTrue(recovery["checkpoint_selection_unchanged"])
            self.assertEqual(recovery["selected_checkpoint_epoch"], 3)

    def test_post_training_failure_propagates_without_partial_csv_or_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticLifecycle(Path(temporary), 123)
            fixture.train()
            run = Path(temporary) / "experiments" / fixture.name
            for path in (run / "metrics.json", *sorted((run / "predictions").glob("*.csv"))):
                path.unlink()
            checkpoint_sha = digest(run / "checkpoints/best_model.pt")
            with fixture.patches() as patched:
                fixture.configure(patched)
                patched["collect_predictions"].side_effect = RuntimeError("injected inference failure")
                with self.assertRaisesRegex(RuntimeError, "injected inference failure"):
                    common.finalize_existing_run(
                        fixture.config_path, "TinyEvolveGCNO", TinyRegressor,
                        expected_checkpoint_sha256=checkpoint_sha,
                        frozen_values={"epochs": 5, "patience": 2},
                    )
            self.assertFalse((run / "metrics.json").exists())
            self.assertEqual(list((run / "predictions").glob("*.csv")), [])
            self.assertEqual(digest(run / "checkpoints/best_model.pt"), checkpoint_sha)

    def test_incomplete_training_and_wrong_checkpoint_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticLifecycle(Path(temporary), 123)
            fixture.train()
            run = Path(temporary) / "experiments" / fixture.name
            log_path = run / "train_log.csv"
            with log_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:4]
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "no normal stop condition"):
                common.validate_finalizable_run(
                    fixture.config_path, "TinyEvolveGCNO", TinyRegressor,
                    frozen_values={"epochs": 5, "patience": 2},
                )

            fixture2 = SyntheticLifecycle(Path(temporary), 2025)
            fixture2.train()
            with fixture2.patches() as patched:
                fixture2.configure(patched)
                with self.assertRaisesRegex(RuntimeError, "forensic recovery binding"):
                    common.finalize_existing_run(
                        fixture2.config_path, "TinyEvolveGCNO", TinyRegressor,
                        expected_checkpoint_sha256="0" * 64,
                        frozen_values={"epochs": 5, "patience": 2},
                    )

    def test_completed_collision_rejected_and_archived_partial_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticLifecycle(root, 42)
            fixture.train()
            with fixture.patches() as patched:
                fixture.configure(patched)
                with self.assertRaises(FileExistsError):
                    common.train_from_config(
                        fixture.config_path, "TinyEvolveGCNO", TinyRegressor,
                        frozen_values={"epochs": 5, "patience": 2},
                    )
            run = root / "experiments" / fixture.name
            archive = run.with_name(run.name + "_INTERRUPTED_20260903T090136Z")
            run.rename(archive)
            fixture.train()
            self.assertTrue(archive.is_dir())
            self.assertTrue(run.is_dir())

    def test_official_auditor_accepts_exact_manifest_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            (run / "checkpoints").mkdir(parents=True)
            (run / "predictions").mkdir()
            for relative, content in (
                ("config.json", "{}\n"), ("metrics.json", "{}\n"),
                ("train_log.csv", "epoch\n1\n"), ("run_metadata.json", "{}\n"),
                ("checkpoints/best_model.pt", "checkpoint"),
            ):
                (run / relative).write_text(content, encoding="utf-8")
            manifest = root / "manifest.json"
            val_ids = [f"v-{index}" for index in range(99)]
            test_ids = [f"t-{index}" for index in range(201)]
            manifest.write_text(json.dumps({"val_ids": val_ids, "test_ids": test_ids}))
            for split, ids in (("val", val_ids), ("test", test_ids)):
                with (run / f"predictions/{split}_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["universe_id", "target", "prediction"])
                    writer.writeheader()
                    writer.writerows(
                        {"universe_id": item, "target": 0.3, "prediction": 0.4} for item in ids
                    )
            config = {
                "output_root": str(root), "experiment_name": "run",
                "split_manifest_path": str(manifest),
            }
            with mock.patch.object(auditor, "load_and_validate_config", return_value=config), \
                 mock.patch.object(auditor, "validate_finalizable_run"):
                auditor.validate_run(root / "config.json")

    def test_recovery_runner_is_fail_closed_and_never_trains_seed42(self):
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("trap on_err ERR", text)
        self.assertIn("seed=123 auditor_PASS", text)
        self.assertLess(text.index("seed=123 auditor_PASS"), text.index('current_seed="2025"'))
        self.assertNotIn("train_evolvegcn_o --config \"$CONFIG_DIR/evolvegcn_o_seed42.json\"", text)
        self.assertIn("verify_seed42", text)
        self.assertNotIn("|| true", text)


if __name__ == "__main__":
    unittest.main()
