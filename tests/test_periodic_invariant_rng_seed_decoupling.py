import json

import pytest
import torch
from torch.utils.data import RandomSampler

import src.training.train_periodic_invariant_gnn as trainer
from src.models.periodic_invariant_gnn import PeriodicInvariantGNNRegressor, count_parameters


def _state(seed):
    trainer.set_seed(seed)
    return {name: value.detach().clone() for name, value in PeriodicInvariantGNNRegressor().state_dict().items()}


def _sample(target):
    return {
        "Nodes_list": [torch.tensor([[10., 1., 1., 1., 0., 0., 0.], [11., 2., 1., 1., 1., 0., 0.], [12., 3., 1., 1., 2., 0., 0.]])],
        "edge_index_list": [torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])],
        "mask_list": [torch.ones(3)],
        "target": target,
    }


def test_seed_role_defaults_and_explicit_override():
    assert trainer.resolve_seed_roles(42) == (42, 42)
    assert trainer.resolve_seed_roles(42, 123) == (42, 123)


def test_model_initialization_is_controlled_by_training_rng_seed():
    same_left, same_right = _state(123), _state(123)
    different = _state(2025)
    assert all(torch.equal(same_left[name], same_right[name]) for name in same_left)
    assert any(not torch.equal(same_left[name], different[name]) for name in same_left)
    assert count_parameters(PeriodicInvariantGNNRegressor()) == 5270


def test_training_shuffle_generator_is_seeded_by_training_rng_seed():
    first = list(RandomSampler(range(16), generator=trainer.training_shuffle_generator(123)))
    same = list(RandomSampler(range(16), generator=trainer.training_shuffle_generator(123)))
    different = list(RandomSampler(range(16), generator=trainer.training_shuffle_generator(2025)))
    assert first == same
    assert first != different


def test_split_identity_stays_on_seed_and_metadata_records_both_roles(tmp_path, monkeypatch):
    dataset, protocol, split = [tmp_path / name for name in ("dataset.pt", "protocol.json", "split42.json")]
    dataset.write_bytes(b"synthetic"); protocol.write_text("{}")
    split.write_text("{}")
    samples = {"train_a": _sample(0.2), "train_b": _sample(0.3), "val_a": _sample(0.4), "protected_test_only": _sample(0.5)}
    observed = {}

    monkeypatch.setattr(trainer, "_load", lambda _: samples)
    monkeypatch.setattr(trainer, "load_dataset_provenance", lambda _: {})
    original_dataset = trainer._Dataset

    class TrackingDataset(original_dataset):
        def __init__(self, loaded_samples, ids):
            observed.setdefault("loader_ids", set()).update(ids)
            super().__init__(loaded_samples, ids)

    monkeypatch.setattr(trainer, "_Dataset", TrackingDataset)

    def fake_manifest(path, dataset_ids, dataset_identity, expected_seed=None):
        observed["expected_seed"] = expected_seed
        observed["dataset_ids"] = set(dataset_ids)
        if expected_seed != 42:
            raise ValueError("Split-manifest seed mismatch")
        return {"train_ids": ["train_a", "train_b"], "val_ids": ["val_a"]}

    monkeypatch.setattr(trainer, "load_split_manifest", fake_manifest)
    output = tmp_path / "accepted"
    payload = trainer.train(dataset_path=dataset, split_manifest_path=split, seed=42, training_rng_seed=123, output_dir=output, protocol_freeze_path=protocol, epochs=1, batch_size=2, patience=40)
    metadata = json.loads((output / "metadata.json").read_text())
    assert observed["expected_seed"] == 42
    assert "protected_test_only" in observed["dataset_ids"]
    assert "protected_test_only" not in observed["loader_ids"]
    assert payload["seed"] == metadata["seed"] == 42
    assert payload["split_seed"] == metadata["split_seed"] == 42
    assert payload["training_rng_seed"] == metadata["training_rng_seed"] == 123
    assert metadata["parameter_count"] == 5270
    assert metadata["test_status"] == "pending / forbidden"
    with pytest.raises(ValueError, match="Split-manifest seed mismatch"):
        trainer.train(dataset_path=dataset, split_manifest_path=split, seed=123, training_rng_seed=42, output_dir=tmp_path / "rejected", protocol_freeze_path=protocol, epochs=1, batch_size=2, patience=40)
