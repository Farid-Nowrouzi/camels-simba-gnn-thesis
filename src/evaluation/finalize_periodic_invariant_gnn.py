"""One-time protected finalization guard. Never invoked during implementation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Callable, Iterable

def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_test_ids(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text())
        ids = payload["test_ids"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("protected split manifest does not provide test_ids") from exc
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RuntimeError("protected split manifest test_ids are invalid")
    return ids


def _protocol_split_contract(protocol_freeze: Path, selected_seed: int) -> tuple[Path, str]:
    try:
        contract = json.loads(protocol_freeze.read_text())["split_manifests"][str(selected_seed)]
        return Path(contract["path"]), str(contract["sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("protocol freeze does not bind the selected split manifest") from exc


def canonical_final_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    """Return the mandatory protected outputs for one finalization directory."""
    return tuple(output_dir / name for name in (
        "test_predictions.csv", "test_metrics.json", "finalization_report.json",
    ))


def require_finalization_ready(*, protocol_freeze: Path, implementation_freeze: Path, expected_protocol_sha256: str, expected_implementation_sha256: str, checkpoint_paths: list[Path], checkpoint_hashes: dict[str,str], split_paths: list[Path], split_hashes: dict[str,str], selected_seed: int, attempt_marker: Path, test_ids: Iterable[str] | None, protected_output_paths: Iterable[Path] | None = None) -> None:
    """Validate immutable inputs and persist a marker before any test callback."""
    if not protocol_freeze.exists() or not implementation_freeze.exists(): raise RuntimeError("both freeze files are required")
    if sha256(protocol_freeze) != expected_protocol_sha256 or sha256(implementation_freeze) != expected_implementation_sha256: raise RuntimeError("freeze hash mismatch")
    for path in checkpoint_paths:
        if not path.exists() or sha256(path) != checkpoint_hashes.get(str(path)): raise RuntimeError("missing or changed production checkpoint")
    for path in split_paths:
        if not path.exists() or sha256(path) != split_hashes.get(str(path)): raise RuntimeError("split manifest hash mismatch")
    if len(split_paths) != 1:
        raise RuntimeError("exactly one selected protected split manifest is required")
    expected_split_path, expected_split_sha256 = _protocol_split_contract(protocol_freeze, selected_seed)
    if split_paths[0].resolve() != expected_split_path.resolve() or split_hashes.get(str(split_paths[0])) != expected_split_sha256:
        raise RuntimeError("selected split manifest does not match the protocol freeze")
    if test_ids is None:
        raise RuntimeError("explicit protected test_ids are required")
    ids = list(test_ids)
    expected_ids = _manifest_test_ids(split_paths[0])
    if len(ids) != 201 or len(expected_ids) != 201:
        raise RuntimeError("test partition has the wrong protected count")
    if len(set(ids)) != len(ids):
        raise RuntimeError("test partition contains duplicate IDs")
    if ids != expected_ids:
        raise RuntimeError("test partition does not match protected manifest identity/order")
    if attempt_marker.exists(): raise RuntimeError("test finalization already attempted; refusing rerun")
    output_paths = list(canonical_final_output_paths(attempt_marker.parent))
    output_paths.extend(Path(path) for path in (protected_output_paths or ()))
    if attempt_marker in output_paths:
        raise RuntimeError("attempt marker cannot be a final output path")
    collision = next((path for path in output_paths if path.exists()), None)
    if collision is not None:
        raise RuntimeError(f"protected final output already exists: {collision}")
    attempt_marker.parent.mkdir(parents=True,exist_ok=True); attempt_marker.write_text(json.dumps({"status":"started","protected":"one-time"},sort_keys=True)+"\n")

def finalize_with_callback(*, callback: Callable[[], object], **kwargs):
    """For synthetic tests or later authorized finalization: marker precedes callback."""
    require_finalization_ready(**kwargs)
    return callback()
