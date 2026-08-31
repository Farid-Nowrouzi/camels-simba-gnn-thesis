from __future__ import annotations

import unittest
from copy import deepcopy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.validation.calibrate_u1000_top1500_radius import (
    count_at_radius,
    periodic_tree,
    select_float32_radius,
)
from scripts.validation.validate_u1000_top1500_radius_dataset import compare_dataset_pair
from src.data.camels_graph_utils import build_sparse_radius_edge_index
from src.data.build_temporal_sequences import build_temporal_dataset


def sample(graph_mode: str, radius: float = 1.0) -> dict:
    features = torch.tensor(
        [[12.0, 0.0, 0.0, 0.0, 1, 2, 3],
         [11.0, 0.5, 0.0, 0.0, 4, 5, 6],
         [10.0, 3.0, 0.0, 0.0, 7, 8, 9]], dtype=torch.float32,
    )
    node_mask = torch.ones(3, 1)
    positions = features[:, 1:4].numpy()
    radius_edge = torch.from_numpy(build_sparse_radius_edge_index(
        positions, node_mask.numpy(), radius, periodic_boundary=True, box_size=25.0,
    ))
    control_edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    edges = radius_edge if graph_mode == "radius" else control_edge
    metadata = {
        "path": "/raw/LH_0_hlist_1.00000.list", "snapshot_value": 1.0,
        "feature_names": ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"],
        "mass_feature": "log10_Mvir", "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "normalization": "none", "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "original_num_halos": 3, "valid_num_halos": 3,
        "selected_num_halos_before_padding": 3, "periodic_boundary": True,
        "box_size": 25.0, "num_real_nodes": 3, "selection_hash_sha256": "a" * 64,
        "tie_breaking_policy": "authoritative_halo_id_ascending",
        "selected_halo_keys": ["10", "11", "12"], "graph_mode": graph_mode,
        "k": 8 if graph_mode == "knn" else None,
        "radius": radius if graph_mode == "radius" else None,
    }
    return {
        "feature_names": metadata["feature_names"], "feature_columns": ["col_10"],
        "raw_feature_columns": ["col_10"], "position_columns": ["col_17", "col_18", "col_19"],
        "velocity_columns": ["col_20", "col_21", "col_22"], "mass_column": "col_10",
        "mass_feature": "log10_Mvir", "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "normalization": "none", "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "num_nodes": 3, "num_snapshots": 5, "periodic_boundary": True, "box_size": 25.0,
        "graph_storage": "sparse_edge_index", "schema_version": "camels_temporal_sparse_v1",
        "graph_mode": graph_mode, "target": torch.tensor(0.3, dtype=torch.float32),
        "Nodes_list": [features.clone() for _ in range(5)],
        "mask_list": [node_mask.clone() for _ in range(5)],
        "edge_index_list": [edges.clone() for _ in range(5)],
        "snapshots": [deepcopy(metadata) for _ in range(5)],
    }


class RadiusCalibrationTests(unittest.TestCase):
    def test_exact_count_uses_periodic_float32_contract(self) -> None:
        positions = np.array([[0.1, 0, 0], [24.9, 0, 0], [4, 0, 0]], dtype=np.float32)
        trees = [periodic_tree(positions)]
        self.assertEqual(count_at_radius([positions], trees, 0.21), 1)
        self.assertEqual(count_at_radius([positions], trees, 0.19), 0)

    def test_float32_search_matches_target_and_is_deterministic(self) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        first, table = select_float32_radius([positions], 2)
        second, _ = select_float32_radius([positions], 2)
        self.assertEqual(first, second)
        self.assertEqual(first, 1.0)
        self.assertEqual(len(table), 5)
        self.assertEqual(table[2]["total_unordered_edges"], 2)


class RadiusValidatorFixtureTests(unittest.TestCase):
    def test_exact_pair_and_reconstruction_pass(self) -> None:
        compare_dataset_pair({"LH_0": sample("knn")}, {"LH_0": sample("radius")}, 1.0)

    def test_changed_feature_fails(self) -> None:
        control, treatment = {"LH_0": sample("knn")}, {"LH_0": sample("radius")}
        treatment["LH_0"]["Nodes_list"][0][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "node features"):
            compare_dataset_pair(control, treatment, 1.0)


class RadiusBuilderIntegrationTests(unittest.TestCase):
    @staticmethod
    def write_catalog(path: Path, timestep: int) -> None:
        lines = ["# radius fixture"]
        for index in range(6):
            values = [0.0] * 23
            values[1] = index
            values[10] = 1000 - index
            values[17:20] = [(index * 0.4) % 25, (timestep * 0.1) % 25, 0.0]
            values[20:23] = [index, index + 1, index + 2]
            lines.append(" ".join(str(value) for value in values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_sparse_radius_builds_all_five_snapshots_with_provenance(self) -> None:
        snapshots = ["0.20000", "0.25000", "0.51209", "0.75065", "1.00000"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for timestep, snapshot_id in enumerate(snapshots):
                self.write_catalog(raw / f"LH_0_hlist_{snapshot_id}.list", timestep)
            output = root / "radius.pt"
            dataset = build_temporal_dataset(
                raw, output, num_universes=1, num_snapshots=5, num_nodes=8,
                normalization="none", graph_mode="radius", radius=0.5,
                periodic_boundary=True, box_size=25.0, dummy_target=0.3,
                graph_storage="sparse_edge_index",
            )
            sequence = dataset["LH_0"]
            self.assertEqual(len(sequence["edge_index_list"]), 5)
            self.assertTrue(all(edge.dtype == torch.int64 for edge in sequence["edge_index_list"]))
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["threshold_rule"], "distance <= radius")
            self.assertEqual(metadata["radius_units"], "h^-1 Mpc")
            self.assertEqual(metadata["edge_ordering_policy"], "lexicographic_source_then_target")

    def test_noncanonical_or_wrong_topology_fails(self) -> None:
        control, treatment = {"LH_0": sample("knn")}, {"LH_0": sample("radius")}
        treatment["LH_0"]["edge_index_list"][0] = treatment["LH_0"]["edge_index_list"][0].flip(1)
        with self.assertRaisesRegex(ValueError, "lexicographically"):
            compare_dataset_pair(control, treatment, 1.0)


if __name__ == "__main__":
    unittest.main()
