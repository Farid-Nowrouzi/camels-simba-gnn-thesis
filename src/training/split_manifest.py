"""Strict immutable split-manifest loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ordered_id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in ids).encode("utf-8")).hexdigest()


def load_split_manifest(
    path: str | Path,
    dataset_ids: List[str],
    dataset_identity: str,
) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    required = [
        "dataset_identity", "seed", "train_ids", "val_ids", "test_ids",
        "counts", "split_hashes", "parent_training_subset", "target_summaries",
        "creation_metadata",
    ]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise KeyError(f"Split manifest missing required keys: {missing}")
    if manifest["dataset_identity"] != dataset_identity:
        raise ValueError(
            "Split manifest dataset identity mismatch: "
            f"manifest={manifest['dataset_identity']!r}, dataset={dataset_identity!r}"
        )

    dataset_set = set(dataset_ids)
    sets: Dict[str, set[str]] = {}
    for name, key in (("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")):
        ids = manifest[key]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Manifest {key} must be a non-empty list.")
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

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(sets[left] & sets[right])
        if overlap:
            raise ValueError(f"Manifest splits {left}/{right} overlap: {overlap[:20]}")

    parent = manifest["parent_training_subset"]
    if parent is not None:
        if not isinstance(parent, list):
            raise TypeError("parent_training_subset must be null or an ordered ID list.")
        if manifest["train_ids"][:len(parent)] != parent:
            raise ValueError("Parent training subset is not an ordered prefix of train_ids.")

    manifest["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest
