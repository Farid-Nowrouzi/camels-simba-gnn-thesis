import hashlib
import json

import pytest

from src.evaluation.finalize_periodic_invariant_gnn import canonical_final_output_paths, finalize_with_callback


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_kwargs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol, implementation, checkpoint, split = [tmp_path / name for name in ("protocol", "implementation", "checkpoint", "split.json")]
    implementation.write_text(implementation.name)
    checkpoint.write_text(checkpoint.name)
    ids = [f"synthetic_{index:03d}" for index in range(201)]
    split.write_text(json.dumps({"test_ids": ids}))
    protocol.write_text(json.dumps({"split_manifests": {"42": {"path": str(split), "sha256": sha(split)}}}))
    return dict(protocol_freeze=protocol, implementation_freeze=implementation, expected_protocol_sha256=sha(protocol), expected_implementation_sha256=sha(implementation), checkpoint_paths=[checkpoint], checkpoint_hashes={str(checkpoint): sha(checkpoint)}, split_paths=[split], split_hashes={str(split): sha(split)}, selected_seed=42, attempt_marker=tmp_path / "test_evaluation_started.json", test_ids=ids)


def assert_refuses(kwargs):
    called = False
    def callback():
        nonlocal called
        called = True
    with pytest.raises(RuntimeError): finalize_with_callback(callback=callback, **kwargs)
    assert not called


@pytest.mark.parametrize("bad_ids", [None, [], [f"id_{i}" for i in range(200)], [f"id_{i}" for i in range(202)]], ids=["none", "empty", "200", "202"])
def test_missing_or_wrong_count_test_ids_refuse_before_callback(tmp_path, bad_ids):
    kwargs = protected_kwargs(tmp_path); kwargs["test_ids"] = bad_ids; assert_refuses(kwargs)


def test_duplicate_wrong_identity_and_wrong_order_refuse_before_callback(tmp_path):
    for name, mutate in {"duplicate": lambda ids: ids[:-1] + [ids[-2]], "wrong_identity": lambda ids: ids[:-1] + ["not_in_manifest"], "wrong_order": lambda ids: list(reversed(ids))}.items():
        kwargs = protected_kwargs(tmp_path / name); kwargs["test_ids"] = mutate(kwargs["test_ids"]); assert_refuses(kwargs)


def test_bad_split_hash_and_existing_marker_refuse_before_callback(tmp_path):
    kwargs = protected_kwargs(tmp_path / "hash"); kwargs["split_hashes"] = {str(kwargs["split_paths"][0]): "bad"}; assert_refuses(kwargs)
    kwargs = protected_kwargs(tmp_path / "marker"); kwargs["attempt_marker"].write_text("previous"); assert_refuses(kwargs)


def test_protocol_split_binding_mismatch_refuses_before_callback(tmp_path):
    kwargs = protected_kwargs(tmp_path)
    other = tmp_path / "other_split.json"
    other.write_text(json.dumps({"test_ids": kwargs["test_ids"]}))
    kwargs["split_paths"] = [other]
    kwargs["split_hashes"] = {str(other): sha(other)}
    assert_refuses(kwargs)


def test_canonical_output_set_is_owned_by_finalizer(tmp_path):
    names = {path.name for path in canonical_final_output_paths(tmp_path)}
    assert names == {"test_predictions.csv", "test_metrics.json", "finalization_report.json"}
    extra = tmp_path / "extra.json"
    assert set(canonical_final_output_paths(tmp_path)) | {extra} == {*canonical_final_output_paths(tmp_path), extra}


@pytest.mark.parametrize("name, extras", [
    ("test_predictions.csv", None),
    ("test_predictions.csv", ["test_metrics.json"]),
    ("finalization_report.json", ["extra.json"]),
    ("test_metrics.json", []),
])
def test_canonical_output_omission_cannot_bypass_collision_protection(tmp_path, name, extras):
    kwargs = protected_kwargs(tmp_path / name)
    (kwargs["attempt_marker"].parent / name).write_text("existing")
    if extras is not None: kwargs["protected_output_paths"] = [kwargs["attempt_marker"].parent / extra for extra in extras]
    assert_refuses(kwargs)


def test_existing_extra_protected_output_refuses_before_callback(tmp_path):
    kwargs = protected_kwargs(tmp_path)
    extra = tmp_path / "extra.json"; extra.write_text("existing")
    kwargs["protected_output_paths"] = [extra]
    assert_refuses(kwargs)


def test_valid_identity_marker_precedes_callback_and_success_outputs_once(tmp_path):
    kwargs = protected_kwargs(tmp_path)
    def callback():
        assert kwargs["attempt_marker"].exists()
        for path in canonical_final_output_paths(kwargs["attempt_marker"].parent): path.write_text("created")
        return "done"
    assert finalize_with_callback(callback=callback, **kwargs) == "done"
    assert all(path.read_text() == "created" for path in canonical_final_output_paths(kwargs["attempt_marker"].parent))
    assert_refuses(kwargs)


def test_failed_callback_leaves_marker_and_second_attempt_refuses(tmp_path):
    kwargs = protected_kwargs(tmp_path)
    with pytest.raises(ValueError): finalize_with_callback(callback=lambda: (_ for _ in ()).throw(ValueError("synthetic failure")), **kwargs)
    assert kwargs["attempt_marker"].exists()
    assert_refuses(kwargs)
