import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_18_bound_manifests_preserve_ordered_partitions():
    source = ROOT / "configs/splits/u1000_top1000_none_k8_sparse"
    bound = ROOT / "configs/splits/u1000_top1500_none_k8_sparse"
    source_paths = sorted(source.glob("*.json"))
    bound_paths = sorted(bound.glob("*.json"))
    assert len(source_paths) == len(bound_paths) == 18
    for source_path, bound_path in zip(source_paths, bound_paths):
        assert source_path.name == bound_path.name
        source_manifest, bound_manifest = load(source_path), load(bound_path)
        for partition in ("train", "val", "test", "unused"):
            assert bound_manifest[f"{partition}_ids"] == source_manifest[f"{partition}_ids"]
        assert bound_manifest["seed"] == source_manifest["seed"]
        assert bound_manifest["counts"] == source_manifest["counts"]


def test_matrix_has_exactly_36_unique_planned_cells():
    registry = load(ROOT / "configs/experiment_registry/u1000_top1500_training_scaling_matrix.json")
    entries = registry["entries"]
    assert len(entries) == 36
    assert len({entry["canonical_experiment_id"] for entry in entries}) == 36
    assert len({(entry["model_key"], entry["training_universe_count"], entry["seed"])
                for entry in entries}) == 36
    assert sum(entry["model_key"] == "evolve" for entry in entries) == 18
    assert sum(entry["model_key"] == "static" for entry in entries) == 18
    assert {entry["status"] for entry in entries} == {"planned"}


def test_raw_audit_covers_all_catalogues_and_matches_padding_total():
    path = ROOT / "reports/experiment_registry/u1000_top1500_raw_halo_count_distribution.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5000
    assert {row["status"] for row in rows} == {"PASS"}
    assert sum(int(row["padded_slots_top1500"]) for row in rows) == 66313
    assert sum(int(row["at_least_1500"] == "False") for row in rows) == 111
