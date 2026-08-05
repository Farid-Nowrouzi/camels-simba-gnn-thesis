"""Strict immutable split-manifest loading and validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ordered_id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in ids).encode("utf-8")).hexdigest()


def canonical_manifest_sha256(manifest: Dict[str, Any]) -> str:
    """Hash scientific manifest content while excluding volatile creation metadata."""
    payload = {
        key: value for key, value in manifest.items()
        if key not in {"canonical_manifest_sha256", "creation_metadata", "manifest_sha256"}
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_seed(manifest: Dict[str, Any], expected_seed: int | None, path: Path) -> None:
    if "seed" not in manifest:
        raise KeyError(f"Split manifest is missing required integer seed: {path}")
    manifest_seed = manifest["seed"]
    if type(manifest_seed) is not int:
        raise TypeError(
            f"Split manifest seed must be an integer, got {manifest_seed!r} "
            f"({type(manifest_seed).__name__}) in {path}"
        )
    if expected_seed is not None:
        if type(expected_seed) is not int:
            raise TypeError(f"Trainer invocation seed must be an integer, got {expected_seed!r}")
        if manifest_seed != expected_seed:
            raise ValueError(
                "Split-manifest seed mismatch: "
                f"trainer seed={expected_seed}, manifest seed={manifest_seed}, manifest={path}. "
                "Use the matching trainer seed or the correct split manifest."
            )


def validate_split_manifest_seed(path: str | Path, expected_seed: int) -> Dict[str, Any]:
    """Fail before output creation when trainer and manifest seeds disagree."""
    manifest_path = Path(path)
    manifest = _read_manifest(manifest_path)
    _validate_seed(manifest, expected_seed, manifest_path)
    return {
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split_manifest_seed": manifest["seed"],
        "split_hashes": dict(manifest.get("split_hashes", {})),
        "split_dataset_identity": manifest.get("dataset_identity"),
    }


def load_dataset_provenance(dataset_path: str | Path) -> Dict[str, Any]:
    """Expose already-published dataset/source identities for experiment config output."""
    path = Path(dataset_path)
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        return {"metadata_status": "legacy_or_missing", "metadata_path": str(metadata_path)}
    metadata = _read_manifest(metadata_path)
    return {
        "metadata_status": "available",
        "metadata_path": str(metadata_path),
        "dataset_sha256": metadata.get("checksum"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256", metadata.get("source_manifest_hash")),
        "target_source_sha256": metadata.get("target_source_sha256"),
        "source_manifest_policy": metadata.get("source_manifest_policy", "legacy_unverified"),
        "graph_storage": metadata.get("graph_storage"),
        "dataset_schema_version": metadata.get("dataset_schema_version"),
        "builder_config_hash": metadata.get("builder_config_hash"),
        "builder_git_commit": metadata.get("git_commit"),
    }


def current_repository_commit() -> str:
    """Return the training-code commit recorded with new experiment identities."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_split_manifest(
    path: str | Path,
    dataset_ids: List[str],
    dataset_identity: str,
    expected_seed: int | None = None,
) -> Dict[str, Any]:
    path = Path(path)
    manifest = _read_manifest(path)

    required = [
        "dataset_identity", "seed", "train_ids", "val_ids", "test_ids",
        "counts", "split_hashes", "parent_training_subset", "target_summaries",
        "creation_metadata",
    ]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise KeyError(f"Split manifest missing required keys: {missing}")
    _validate_seed(manifest, expected_seed, path)
    if manifest["dataset_identity"] != dataset_identity:
        raise ValueError(
            "Split manifest dataset identity mismatch: "
            f"manifest={manifest['dataset_identity']!r}, dataset={dataset_identity!r}"
        )

    dataset_set = set(dataset_ids)
    sets: Dict[str, set[str]] = {}
    partitions = [("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")]
    if "unused_ids" in manifest:
        partitions.append(("unused", "unused_ids"))
    for name, key in partitions:
        ids = manifest[key]
        if not isinstance(ids, list) or (name != "unused" and not ids):
            raise ValueError(f"Manifest {key} must be a list (non-empty except unused_ids).")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Manifest {key} contains duplicate IDs.")
        missing_ids = [item for item in ids if item not in dataset_set]
        if missing_ids:
            raise ValueError(f"Manifest {key} contains missing dataset IDs: {missing_ids[:20]}")
        if int(manifest["counts"].get(name, -1)) != len(ids):
            raise ValueError(f"Manifest {name} count does not match ordered IDs.")
        actual_hash = ordered_id_hash(ids)
        expected_hash = manifest["split_hashes"].get(name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Manifest {name} hash mismatch: expected {expected_hash}, calculated {actual_hash}"
            )
        sets[name] = set(ids)

    split_names = list(sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(sets[left] & sets[right])
            if overlap:
                raise ValueError(f"Manifest splits {left}/{right} overlap: {overlap[:20]}")

    if "unused" in sets:
        covered = set().union(*sets.values())
        if covered != dataset_set:
            missing_partition_ids = sorted(dataset_set - covered)
            raise ValueError(
                "Manifest partitions do not cover the dataset exactly: "
                f"missing={missing_partition_ids[:20]}"
            )

    expected_canonical_hash = manifest.get("canonical_manifest_sha256")
    if expected_canonical_hash is not None:
        actual_canonical_hash = canonical_manifest_sha256(manifest)
        if expected_canonical_hash != actual_canonical_hash:
            raise ValueError(
                "Canonical split-manifest hash mismatch: "
                f"expected {expected_canonical_hash}, calculated {actual_canonical_hash}"
            )

    parent = manifest["parent_training_subset"]
    if parent is not None:
        if not isinstance(parent, list):
            raise TypeError("parent_training_subset must be null or an ordered ID list.")
        if manifest["train_ids"][:len(parent)] != parent:
            raise ValueError("Parent training subset is not an ordered prefix of train_ids.")

    manifest["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest
