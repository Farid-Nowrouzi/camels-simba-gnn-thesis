from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.atomic_dataset import atomic_write_sparse_dataset, sha256_file
from src.data.build_temporal_sequences import build_temporal_dataset
from src.data.camels_graph_utils import (
    GRAPH_STORAGE_SPARSE,
    build_knn_adjacency,
    build_sparse_knn_edge_index,
    clean_halo_dataframe,
    preprocessing_version_for_config,
    select_top_halos,
    selection_provenance,
)
from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.static_gcn import StaticGCNRegressor, normalize_adjacency, normalize_sparse_edges
from src.training.sparse_batch import collate_sparse_static, collate_sparse_temporal
from src.training.split_manifest import load_split_manifest, ordered_id_hash
from src.training.train_static_gcn import convert_temporal_final_snapshot_to_static
from src.training.train_static_gcn import create_loaders as create_static_loaders
from src.training.train_evolvegcn_h import create_loaders as create_evolve_loaders


ATOL = 1e-6
RTOL = 1e-5


def edge_set(edge_index: np.ndarray | torch.Tensor) -> set[tuple[int, int]]:
    values = edge_index.tolist() if isinstance(edge_index, torch.Tensor) else edge_index.tolist()
    return set(zip(values[0], values[1]))


def dense_to_standard_edges(adjacency: torch.Tensor) -> torch.Tensor:
    # Dense A[row target, col source], sparse convention [source,target].
    return adjacency.nonzero(as_tuple=False)[:, [1, 0]].T.contiguous()


def sparse_sample(x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> dict:
    return {
        "graph_storage": GRAPH_STORAGE_SPARSE,
        "x": x,
        "edge_index": dense_to_standard_edges(adjacency),
        "edge_weight": None,
        "mask": mask,
    }


class TopNAndKNNTests(unittest.TestCase):
    def test_preprocessing_label_is_generated_from_effective_configuration(self) -> None:
        production = preprocessing_version_for_config(
            num_nodes=1000, normalization="none", graph_mode="knn", k=8,
            radius=None, periodic_boundary=True, box_size=25.0,
            graph_storage="sparse_edge_index",
        )
        ablation = preprocessing_version_for_config(
            num_nodes=500, normalization="zscore", graph_mode="knn", k=4,
            radius=None, periodic_boundary=False, box_size=50.0,
            graph_storage="dense_adjacency",
        )
        self.assertEqual(
            production,
            "v3_logmass_none_top1000_periodic_knn_k8_box25_sparse_edge_index",
        )
        self.assertEqual(
            ablation,
            "v3_logmass_zscore_top500_nonperiodic_knn_k4_box50_dense_adjacency",
        )
        self.assertNotIn("minmax_top100", production)

    def frame(self) -> pd.DataFrame:
        rows = []
        masses = [8.0, 10.0, np.nan, 10.0, -1.0, 9.0]
        ids = [8, 4, 2, 3, 1, 7]
        for index, (mass, halo_id) in enumerate(zip(masses, ids)):
            row = {f"col_{column}": float(column + index) for column in range(23)}
            row["col_1"] = halo_id
            row["col_10"] = mass
            rows.append(row)
        return pd.DataFrame(rows)

    def test_stable_topn_ties_filter_repeatability_and_nesting(self) -> None:
        cleaned = clean_halo_dataframe(self.frame())
        top2a = select_top_halos(cleaned, 2)
        top2b = select_top_halos(cleaned, 2)
        top3 = select_top_halos(cleaned, 3)
        self.assertEqual(top2a["col_10"].tolist(), [10.0, 10.0])
        self.assertEqual(top2a["col_1"].tolist(), [3, 4])
        self.assertEqual(top2a["col_1"].tolist(), top2b["col_1"].tolist())
        self.assertEqual(top2a["col_1"].tolist(), top3["col_1"].tolist()[:2])
        self.assertEqual(selection_provenance(top2a)["selection_hash_sha256"],
                         selection_provenance(top2b)["selection_hash_sha256"])
        self.assertTrue((cleaned["col_10"] > 0).all())

    def test_sparse_periodic_knn_invariants_and_dense_equivalence(self) -> None:
        positions = np.array([
            [0.1, 0.0, 0.0], [24.9, 0.0, 0.0], [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        mask = np.array([[1], [1], [1], [1], [0]], dtype=np.float32)
        sparse = build_sparse_knn_edge_index(positions, mask, k=1, box_size=25.0)
        pairs = edge_set(sparse)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertFalse(any(4 in pair for pair in pairs))
        self.assertFalse(any(a == b for a, b in pairs))
        self.assertEqual(len(pairs), sparse.shape[1])
        self.assertEqual(list(zip(*sparse.tolist())), sorted(pairs))
        self.assertTrue(all((b, a) in pairs for a, b in pairs))

        dense = build_knn_adjacency(positions, mask, k=1, box_size=25.0)
        dense_pairs = set(map(tuple, np.argwhere(dense > 0).tolist()))
        self.assertEqual(pairs, dense_pairs)  # symmetric orientation is immaterial

    def test_distance_ties_and_fewer_than_k_plus_one_are_deterministic(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [24, 0, 0]], dtype=np.float32)
        mask = np.ones((3, 1), dtype=np.float32)
        first = build_sparse_knn_edge_index(positions, mask, k=8, tie_keys=np.array([0, 1, 2]))
        second = build_sparse_knn_edge_index(positions, mask, k=8, tie_keys=np.array([0, 1, 2]))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape[1], 6)


class ModelAndBatchEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch_size, self.timesteps, self.nodes, self.features = 2, 5, 6, 3
        adjacency = torch.zeros(self.batch_size, self.nodes, self.nodes)
        for batch in range(self.batch_size):
            for node in range(self.nodes):
                other = (node + 1) % self.nodes
                adjacency[batch, node, other] = adjacency[batch, other, node] = 1
        self.A = adjacency
        self.X = torch.randn(self.batch_size, self.nodes, self.features)
        self.mask = torch.ones(self.batch_size, self.nodes, 1)

    def static_sparse_batch(self) -> dict:
        return collate_sparse_static([
            sparse_sample(self.X[b], self.A[b], self.mask[b]) for b in range(self.batch_size)
        ])

    def temporal_sparse_batch(self) -> dict:
        samples = []
        for batch in range(self.batch_size):
            sample = sparse_sample(self.X[batch], self.A[batch], self.mask[batch])
            samples.append({
                "graph_storage": GRAPH_STORAGE_SPARSE,
                "Nodes_list": [self.X[batch] + timestep * 0.01 for timestep in range(self.timesteps)],
                "edge_index_list": [sample["edge_index"] for _ in range(self.timesteps)],
                "mask_list": [self.mask[batch] for _ in range(self.timesteps)],
            })
        return collate_sparse_temporal(samples)

    def test_normalized_edge_weights_equal_dense(self) -> None:
        dense = normalize_adjacency(self.A)
        batch = self.static_sparse_batch()
        edge_index, weights = normalize_sparse_edges(batch["edge_index"], batch["x"].shape[0])
        reconstructed = torch.zeros(self.batch_size * self.nodes, self.batch_size * self.nodes)
        reconstructed[edge_index[1], edge_index[0]] = weights
        expected = torch.block_diag(*list(dense))
        torch.testing.assert_close(reconstructed, expected, atol=ATOL, rtol=RTOL)

    def test_legacy_dense_defaults_and_forward_remain_available(self) -> None:
        signature = inspect.signature(build_temporal_dataset)
        self.assertEqual(signature.parameters["graph_storage"].default, "dense_adjacency")
        model = StaticGCNRegressor(self.features, 4, 1, 0.0)
        output = model(self.A, self.X, self.mask)
        self.assertEqual(tuple(output.shape), (self.batch_size, 1))

    def test_static_forward_layer_pool_and_gradient_equivalence(self) -> None:
        dense_model = StaticGCNRegressor(self.features, 8, 3, 0.0, "mean", "gcn")
        sparse_model = StaticGCNRegressor(self.features, 8, 3, 0.0, "mean", "gcn")
        sparse_model.load_state_dict(dense_model.state_dict())
        dense_nodes = dense_model.encode_nodes(self.A, self.X, self.mask)
        sparse_nodes = sparse_model.encode_nodes(self.static_sparse_batch())
        torch.testing.assert_close(sparse_nodes.reshape_as(dense_nodes), dense_nodes, atol=ATOL, rtol=RTOL)
        dense_out = dense_model(self.A, self.X, self.mask)
        sparse_out = sparse_model(self.static_sparse_batch())
        torch.testing.assert_close(sparse_out, dense_out, atol=ATOL, rtol=RTOL)
        dense_out.square().mean().backward()
        sparse_out.square().mean().backward()
        for dense_parameter, sparse_parameter in zip(dense_model.parameters(), sparse_model.parameters()):
            self.assertTrue(torch.isfinite(dense_parameter.grad).all())
            self.assertTrue(torch.isfinite(sparse_parameter.grad).all())
            torch.testing.assert_close(sparse_parameter.grad, dense_parameter.grad, atol=2e-6, rtol=2e-5)

    def test_evolve_layer_pool_forward_and_gradient_equivalence(self) -> None:
        X_seq = torch.stack([self.X + timestep * 0.01 for timestep in range(self.timesteps)], dim=1)
        A_seq = self.A[:, None].repeat(1, self.timesteps, 1, 1)
        mask_seq = self.mask[:, None].repeat(1, self.timesteps, 1, 1)
        dense_model = EvolveGCNHRegressor(self.features, 7, 2, 0.0, head_type="linear")
        sparse_model = EvolveGCNHRegressor(self.features, 7, 2, 0.0, head_type="linear")
        sparse_model.load_state_dict(dense_model.state_dict())
        sparse_batch = self.temporal_sparse_batch()

        dense_layer = dense_model.layers[0](A_seq, X_seq, mask_seq)
        sparse_layer = sparse_model.layers[0].forward_sparse(sparse_batch["snapshots"])
        sparse_layer_tensor = torch.stack(
            [value.reshape(self.batch_size, self.nodes, -1) for value in sparse_layer], dim=1
        )
        torch.testing.assert_close(sparse_layer_tensor, dense_layer, atol=ATOL, rtol=RTOL)

        dense_out = dense_model(A_seq, X_seq, mask_seq)
        sparse_out = sparse_model(sparse_batch)
        torch.testing.assert_close(sparse_out, dense_out, atol=ATOL, rtol=RTOL)
        dense_out.square().mean().backward()
        sparse_out.square().mean().backward()
        for dense_parameter, sparse_parameter in zip(dense_model.parameters(), sparse_model.parameters()):
            self.assertTrue(torch.isfinite(dense_parameter.grad).all())
            self.assertTrue(torch.isfinite(sparse_parameter.grad).all())
            torch.testing.assert_close(sparse_parameter.grad, dense_parameter.grad, atol=3e-6, rtol=3e-5)

    def test_variable_node_five_snapshot_batch_and_target_alignment(self) -> None:
        samples = []
        targets = []
        for sample_index, count in enumerate((3, 5)):
            x = torch.randn(count + 2, self.features)
            mask = torch.cat([torch.ones(count, 1), torch.zeros(2, 1)])
            edge = torch.tensor([[i, (i + 1) % count] for i in range(count)] +
                                [[(i + 1) % count, i] for i in range(count)]).T
            samples.append({
                "graph_storage": GRAPH_STORAGE_SPARSE,
                "Nodes_list": [x for _ in range(5)],
                "edge_index_list": [edge for _ in range(5)],
                "mask_list": [mask for _ in range(5)],
                "target": torch.tensor(0.1 + sample_index),
            })
            targets.append(float(samples[-1]["target"]))
        batch = collate_sparse_temporal(samples)
        self.assertEqual(batch["num_timesteps"], 5)
        self.assertEqual(batch["num_graphs"], 2)
        self.assertTrue(all(graph["x"].shape[0] == 8 for graph in batch["snapshots"]))
        self.assertEqual(targets, [float(samples[0]["target"]), float(samples[1]["target"])])

    def test_static_final_snapshot_is_same_sparse_record(self) -> None:
        edge_list = [torch.tensor([[0, 1], [1, 0]]) for _ in range(5)]
        x_list = [torch.randn(2, 3) for _ in range(5)]
        masks = [torch.ones(2, 1) for _ in range(5)]
        target = torch.tensor(0.3)
        temporal = {"LH_7": {
            "graph_storage": GRAPH_STORAGE_SPARSE, "edge_index_list": edge_list,
            "edge_weight_list": None, "Nodes_list": x_list, "mask_list": masks,
            "target": target, "snapshots": [{"snapshot_value": value} for value in range(5)],
        }}
        static = convert_temporal_final_snapshot_to_static(temporal)["LH_7"]
        self.assertIs(static["edge_index"], edge_list[-1])
        self.assertIs(static["X"], x_list[-1])
        self.assertIs(static["mask"], masks[-1])
        self.assertIs(static["target"], target)
        self.assertEqual(static["universe_id"], "LH_7")
        self.assertEqual(static["snapshot"]["snapshot_value"], 4)


class ManifestAtomicAndSmokeTests(unittest.TestCase):
    def test_manifest_exact_order_hashes_identity_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            value = {
                "dataset_identity": "tiny-v1", "seed": 42,
                "train_ids": ["LH_2", "LH_0"], "val_ids": ["LH_1"], "test_ids": ["LH_3"],
                "counts": {"train": 2, "val": 1, "test": 1},
                "split_hashes": {
                    "train": ordered_id_hash(["LH_2", "LH_0"]),
                    "val": ordered_id_hash(["LH_1"]), "test": ordered_id_hash(["LH_3"]),
                },
                "parent_training_subset": ["LH_2"], "target_summaries": {},
                "creation_metadata": {"created": "fixture"},
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            loaded = load_split_manifest(path, ["LH_0", "LH_1", "LH_2", "LH_3"], "tiny-v1")
            self.assertEqual(loaded["train_ids"], ["LH_2", "LH_0"])
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_split_manifest(path, ["LH_0", "LH_1", "LH_2", "LH_3"], "wrong")
            value["train_ids"] = ["LH_2", "LH_2"]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_split_manifest(path, ["LH_0", "LH_1", "LH_2", "LH_3"], "tiny-v1")

    def test_both_trainers_consume_manifest_order_without_resplitting(self) -> None:
        ids = ["LH_0", "LH_1", "LH_2", "LH_3"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            manifest = {
                "dataset_identity": "tiny-v1", "seed": 42,
                "train_ids": ["LH_2", "LH_0"], "val_ids": ["LH_1"], "test_ids": ["LH_3"],
                "counts": {"train": 2, "val": 1, "test": 1},
                "split_hashes": {"train": ordered_id_hash(["LH_2", "LH_0"]),
                                 "val": ordered_id_hash(["LH_1"]),
                                 "test": ordered_id_hash(["LH_3"])},
                "parent_training_subset": ["LH_2"], "target_summaries": {},
                "creation_metadata": {"created": "fixture"},
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            temporal = {}
            for index, universe_id in enumerate(ids):
                x = torch.randn(3, 2)
                mask = torch.ones(3, 1)
                edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
                temporal[universe_id] = {
                    "graph_storage": GRAPH_STORAGE_SPARSE,
                    "Nodes_list": [x for _ in range(5)],
                    "edge_index_list": [edge for _ in range(5)],
                    "edge_weight_list": None,
                    "mask_list": [mask for _ in range(5)],
                    "target": torch.tensor(0.1 + index * 0.01),
                    "snapshots": [{"snapshot_value": t} for t in range(5)],
                }
            evolve_result = create_evolve_loaders(
                temporal, seed=42, batch_size=2, train_ratio=.5, val_ratio=.25,
                test_ratio=.25, split_manifest_path=path, dataset_identity="tiny-v1",
            )
            self.assertEqual(evolve_result[3:6], (["LH_2", "LH_0"], ["LH_1"], ["LH_3"]))
            self.assertEqual(evolve_result[0].dataset.universe_ids, ["LH_2", "LH_0"])

            static_data = convert_temporal_final_snapshot_to_static(temporal)
            static_result = create_static_loaders(
                static_data, seed=42, batch_size=2, train_ratio=.5, val_ratio=.25,
                test_ratio=.25, split_manifest_path=path, dataset_identity="tiny-v1",
            )
            self.assertEqual(static_result[3:6], (["LH_2", "LH_0"], ["LH_1"], ["LH_3"]))
            self.assertEqual(static_result[0].dataset.universe_ids, ["LH_2", "LH_0"])

    def test_atomic_output_checksum_collision_partial_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tiny.pt"
            metadata = atomic_write_sparse_dataset(
                {"LH_0": {"target": torch.tensor(0.2)}}, output,
                {"builder_config_hash": "a", "source_manifest_hash": "b", "git_commit": "c"},
                validate=lambda dataset: self.assertIn("LH_0", dataset),
            )
            self.assertEqual(metadata["checksum"], sha256_file(output))
            self.assertTrue(output.with_suffix(".metadata.json").exists())
            self.assertTrue(output.with_suffix(".complete").exists())
            self.assertFalse(output.with_suffix(".pt.lock").exists())
            self.assertFalse(list(output.parent.glob("*.tmp")))
            with self.assertRaises(FileExistsError):
                atomic_write_sparse_dataset({}, output, {})

    @staticmethod
    def write_catalog(path: Path, universe: int, timestep: int, rows: int = 18) -> None:
        lines = ["# tiny bounded CAMELS-like fixture"]
        for row_index in range(rows):
            values = [0.0] * 23
            values[1] = float(row_index)
            values[10] = float(1000 - row_index)
            values[17] = (row_index * 1.3 + universe) % 25
            values[18] = (row_index * 2.1 + timestep) % 25
            values[19] = (row_index * 0.7) % 25
            values[20:23] = [row_index * 0.1, row_index * 0.2, row_index * 0.3]
            lines.append(" ".join(str(value) for value in values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_tiny_two_universe_five_snapshot_sparse_smoke_build(self) -> None:
        snapshots = ["0.20000", "0.25000", "0.51209", "0.75065", "1.00000"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for universe in range(2):
                for timestep, snapshot in enumerate(snapshots):
                    self.write_catalog(raw / f"LH_{universe}_hlist_{snapshot}.list", universe, timestep)
            output = root / "tiny_sparse.pt"
            dataset = build_temporal_dataset(
                raw, output, num_universes=2, num_snapshots=5, num_nodes=16,
                normalization="none", graph_mode="knn", k=3, periodic_boundary=True,
                box_size=25.0, dummy_target=0.3, graph_storage=GRAPH_STORAGE_SPARSE,
            )
            self.assertEqual(list(dataset), ["LH_0", "LH_1"])
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".complete").exists())
            loaded = torch.load(output, map_location="cpu", weights_only=False)
            expected_version = (
                "v3_logmass_none_top16_periodic_knn_k3_box25_sparse_edge_index"
            )
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["preprocessing_version"], expected_version)
            self.assertTrue(all(
                sample["preprocessing_version"] == expected_version
                for sample in loaded.values()
            ))
            self.assertTrue(all(
                snapshot["preprocessing_version"] == expected_version
                for sample in loaded.values() for snapshot in sample["snapshots"]
            ))
            temporal_batch = collate_sparse_temporal(list(loaded.values()))
            final_static = convert_temporal_final_snapshot_to_static(loaded)
            static_batch = collate_sparse_static([
                {"graph_storage": GRAPH_STORAGE_SPARSE, "x": sample["X"],
                 "edge_index": sample["edge_index"], "edge_weight": sample["edge_weight"],
                 "mask": sample["mask"]} for sample in final_static.values()
            ])
            evolve = EvolveGCNHRegressor(7, 8, 2, 0.0, head_type="linear")
            static = StaticGCNRegressor(7, 8, 3, 0.0)
            evolve_loss = evolve(temporal_batch).square().mean()
            static_loss = static(static_batch).square().mean()
            evolve_loss.backward()
            static_loss.backward()
            self.assertTrue(torch.isfinite(evolve_loss))
            self.assertTrue(torch.isfinite(static_loss))
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in evolve.parameters()))
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in static.parameters()))


if __name__ == "__main__":
    unittest.main()
