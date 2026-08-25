"""Focused controls for the T2/T5 EvolveGCN-H mean-vs-last diagnostic."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.models.evolvegcn_h import EvolveGCNHRegressor, count_parameters
from src.training.sparse_batch import collate_sparse_temporal
from src.training.train_evolvegcn_h import select_temporal_snapshots


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "configs/experiment_families/u1000_top1500_temporal_pooling_diagnostic_evolvegcn_h.json"
VALUES = [0.2, 0.25, 0.51209, 0.75065, 1.0]
SUBSETS = {2: [3, 4], 5: [0, 1, 2, 3, 4]}


def sample(offset: float = 0.0) -> dict:
    nodes = [torch.arange(21, dtype=torch.float32).reshape(3, 7) / 20 + i + offset for i in range(5)]
    return {
        "Nodes_list": nodes,
        "mask_list": [torch.ones(3, 1) for _ in range(5)],
        "edge_index_list": [torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]) for _ in range(5)],
        "edge_weight_list": [torch.ones(4) for _ in range(5)],
        "snapshots": [{"snapshot_value": value, "stored_index": i} for i, value in enumerate(VALUES)],
        "target": torch.tensor(0.3 + offset / 100),
        "universe_id": f"LH_{int(offset)}",
        "num_snapshots": 5,
    }


def model(pooling: str) -> EvolveGCNHRegressor:
    return EvolveGCNHRegressor(
        node_features=7, hidden_dim=32, num_layers=2, dropout=0.0,
        activation="relu", temporal_pooling=pooling, graph_pooling="mean",
        head_type="linear", add_self_loops=True,
    )


def schema(module: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [(key, tuple(value.shape)) for key, value in module.state_dict().items()]


class TemporalPoolingDiagnosticTests(unittest.TestCase):
    def test_temporal_pooling_exactly_mean_or_final_representation(self) -> None:
        sequence = torch.tensor([[[1.0, 2.0], [3.0, 8.0], [9.0, 4.0]]])
        self.assertTrue(torch.equal(model("last").temporal_pool(sequence), sequence[:, -1, :]))
        self.assertTrue(torch.equal(model("mean").temporal_pool(sequence), sequence.mean(dim=1)))

    def test_t2_t5_mean_last_forward_backward_shape_and_invariance(self) -> None:
        for T, indices in SUBSETS.items():
            selected = [
                select_temporal_snapshots(sample(offset), indices, [VALUES[i] for i in indices])
                for offset in (0.0, 1.0)
            ]
            batch = collate_sparse_temporal(selected)
            reference = model("mean")
            state = reference.state_dict()
            schemas = []
            for pooling in ("mean", "last"):
                with self.subTest(T=T, pooling=pooling):
                    candidate = model(pooling)
                    candidate.load_state_dict(state)
                    prediction = candidate(batch)
                    loss = torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))
                    loss.backward()
                    gradients = [parameter.grad for parameter in candidate.parameters()]
                    self.assertEqual(tuple(prediction.shape), (2, 1))
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(gradient is not None for gradient in gradients))
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                    self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))
                    self.assertEqual(count_parameters(candidate), 3_408_097)
                    schemas.append(schema(candidate))
            self.assertEqual(schemas[0], schemas[1])

    def test_t5_last_runs_every_snapshot_through_every_recurrent_layer(self) -> None:
        batch = collate_sparse_temporal([sample(0.0), sample(1.0)])
        candidate = model("last")
        calls = [0 for _ in candidate.layers]
        hooks = []
        for index, layer in enumerate(candidate.layers):
            hooks.append(layer.weight_evolver.register_forward_hook(
                lambda _module, _args, _output, i=index: calls.__setitem__(i, calls[i] + 1)
            ))
        try:
            prediction = candidate(batch)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertEqual(calls, [5, 5])

    def test_family_matrix_and_only_pooling_differs_within_matched_cell(self) -> None:
        spec = json.loads(FAMILY.read_text(encoding="utf-8"))
        self.assertEqual(spec["grouping_values"], ["T2_mean", "T2_last", "T5_mean", "T5_last"])
        self.assertEqual(len(spec["runs"]), 12)
        self.assertEqual(sum(run["action"] == "reuse" for run in spec["runs"]), 6)
        self.assertEqual(sum(run["action"] == "run_if_missing" for run in spec["runs"]), 6)
        runs = {(run["T"], run["pooling"], run["seed"]): run for run in spec["runs"]}
        for T, indices in SUBSETS.items():
            for seed in (42, 123, 2025):
                mean = runs[T, "mean", seed]
                last = runs[T, "last", seed]
                self.assertEqual(mean["selected_snapshot_indices"], indices)
                self.assertEqual(last["selected_snapshot_indices"], indices)
                self.assertEqual(mean["split_manifest_sha256"], last["split_manifest_sha256"])
                mean_args = dict(mean["argument_overrides"])
                last_args = dict(last["argument_overrides"])
                self.assertEqual(mean_args.pop("temporal_pooling"), "mean")
                self.assertEqual(last_args.pop("temporal_pooling"), "last")
                self.assertEqual(mean_args, last_args)

    def test_existing_mean_configs_remain_backward_compatible(self) -> None:
        t2 = ROOT / "experiments/evolvegcn_h_u1000_top1500_t2_ctxsuffix_sparse_train700_seed42_none_h32_l2_gmean_tmean_linear/config.json"
        t5 = ROOT / "experiments/evolvegcn_h_u1000_top1500_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear/config.json"
        t2_config = json.loads(t2.read_text(encoding="utf-8"))
        t5_config = json.loads(t5.read_text(encoding="utf-8"))
        self.assertEqual(t2_config["selected_snapshot_indices"], [3, 4])
        self.assertEqual(t2_config["temporal_pooling"], "mean")
        self.assertEqual(t5_config["num_snapshots"], 5)
        self.assertEqual(t5_config["temporal_pooling"], "mean")


if __name__ == "__main__":
    unittest.main()
