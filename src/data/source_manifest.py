"""Deterministic full-content provenance for sparse CAMELS dataset builds."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


SOURCE_MANIFEST_SCHEMA_VERSION = "camels_source_manifest_v1"
SOURCE_MANIFEST_POLICY_FULL = "full_sha256"
SOURCE_MANIFEST_POLICY_LEGACY = "legacy_stat_only"
HASH_CHUNK_SIZE = 1024 * 1024
VALID_SOURCE_ROLES = {"halo_catalogue", "target_table", "other"}
ROLE_ORDER = {"halo_catalogue": 0, "target_table": 1, "other": 2}
CATALOGUE_PATTERN = re.compile(r"^LH_(\d+)_hlist_([0-9]+(?:\.[0-9]+)?)\.list$")

UNIVERSE_COLUMNS = ("universe_id", "Universe", "universe", "Universe_ID", "lh_id", "LH")
TARGET_COLUMNS = ("omega_m", "Omega_m", "Omega_M", "target", "Target", "omega_m_value")


def sha256_file_streaming(path: str | Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Hash a file using bounded reads; never load the complete source into memory."""
    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("SHA-256 chunk_size must be positive.")
    try:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise OSError(f"Unable to hash source file {source}: {exc}") from exc


def _first_present(values: Iterable[str], candidates: Iterable[str], label: str) -> str:
    available = set(values)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"Target table has no supported {label} column; columns={sorted(available)}")


def inspect_target_table(path: str | Path) -> Dict[str, Any]:
    """Return inexpensive, content-derived target-table structure metadata."""
    source = Path(path)
    try:
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Target table has no header: {source}")
            universe_column = _first_present(reader.fieldnames, UNIVERSE_COLUMNS, "universe-ID")
            target_column = _first_present(reader.fieldnames, TARGET_COLUMNS, "target")
            row_count = sum(1 for _ in reader)
    except OSError as exc:
        raise OSError(f"Unable to inspect target source {source}: {exc}") from exc
    return {
        "row_count": row_count,
        "universe_id_column": universe_column,
        "target_column": target_column,
    }


def _catalogue_identity(path: Path) -> tuple[str, str]:
    match = CATALOGUE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Halo catalogue filename does not encode universe/snapshot identity: {path}")
    return f"LH_{int(match.group(1))}", f"{float(match.group(2)):.5f}"


def _entry_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    universe = entry.get("universe_id")
    universe_index = int(str(universe).split("_", 1)[1]) if universe is not None else -1
    snapshot = float(entry["snapshot_id"]) if entry.get("snapshot_id") is not None else -1.0
    return (ROLE_ORDER[entry["source_role"]], universe_index, snapshot, entry["relative_path"])


def canonical_manifest_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact portable fields covered by the top-level digest."""
    return {
        "schema_version": manifest["schema_version"],
        "source_manifest_policy": manifest["source_manifest_policy"],
        "hash_algorithm": manifest["hash_algorithm"],
        "hash_chunk_size_bytes": manifest["hash_chunk_size_bytes"],
        "sorting_key": manifest["sorting_key"],
        "entries": manifest["entries"],
    }


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(
        canonical_manifest_payload(manifest), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def source_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def build_full_source_manifest(
    catalogue_paths: Iterable[str | Path],
    raw_root: str | Path,
    target_path: Optional[str | Path],
    target_root: Optional[str | Path] = None,
    require_target: bool = True,
) -> Dict[str, Any]:
    """Create a canonical ordered manifest for exactly the supplied sources."""
    raw_root_path = Path(raw_root).resolve()
    entries = []
    for value in catalogue_paths:
        path = Path(value).resolve()
        try:
            relative_path = path.relative_to(raw_root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"Catalogue {path} is outside raw root {raw_root_path}") from exc
        universe_id, snapshot_id = _catalogue_identity(path)
        stat = path.stat()
        entries.append({
            "relative_path": relative_path,
            "source_role": "halo_catalogue",
            "universe_id": universe_id,
            "snapshot_id": snapshot_id,
            "size_bytes": stat.st_size,
            "sha256": sha256_file_streaming(path),
        })

    roots = {"halo_catalogue": str(raw_root_path)}
    if target_path is not None:
        target = Path(target_path).resolve()
        target_root_path = Path(target_root).resolve() if target_root is not None else target.parent
        try:
            relative_target = target.relative_to(target_root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"Target source {target} is outside target root {target_root_path}") from exc
        stat = target.stat()
        entries.append({
            "relative_path": relative_target,
            "source_role": "target_table",
            "universe_id": None,
            "snapshot_id": None,
            "size_bytes": stat.st_size,
            "sha256": sha256_file_streaming(target),
            **inspect_target_table(target),
        })
        roots["target_table"] = str(target_root_path)
    elif require_target:
        raise ValueError("Full-SHA256 production source manifest requires exactly one target table.")

    entries.sort(key=_entry_sort_key)
    manifest: Dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_manifest_policy": SOURCE_MANIFEST_POLICY_FULL,
        "hash_algorithm": "sha256",
        "hash_chunk_size_bytes": HASH_CHUNK_SIZE,
        "sorting_key": ["source_role", "universe_id_numeric", "snapshot_id_numeric", "relative_path"],
        "entries": entries,
        "entry_count": len(entries),
        "catalogue_count": sum(entry["source_role"] == "halo_catalogue" for entry in entries),
        "target_source_count": sum(entry["source_role"] == "target_table" for entry in entries),
        "source_roots": roots,
    }
    manifest["source_root_identity"] = hashlib.sha256(json.dumps(
        {role: Path(root).name for role, root in sorted(roots.items())},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    manifest["manifest_sha256"] = source_manifest_sha256(manifest)
    return manifest


def verify_full_source_manifest(
    manifest: Mapping[str, Any],
    source_roots: Optional[Mapping[str, str | Path]] = None,
    require_target: bool = True,
) -> Dict[str, Any]:
    """Verify schema, uniqueness, canonical identity, file size, and full content."""
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("A new sparse production build requires full-SHA256 source-manifest schema v1.")
    if manifest.get("source_manifest_policy") != SOURCE_MANIFEST_POLICY_FULL:
        raise ValueError("A new sparse production build rejects legacy/stat-only source provenance.")
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("Source manifest hash_algorithm must be sha256.")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise TypeError("Source manifest entries must be a list.")
    if entries != sorted(entries, key=_entry_sort_key):
        raise ValueError("Source manifest entries are not in canonical deterministic order.")
    if manifest.get("entry_count") != len(entries):
        raise ValueError("Source manifest entry_count does not match entries.")
    expected_identity = source_manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != expected_identity:
        raise ValueError("Top-level canonical source-manifest SHA-256 mismatch.")

    relative_paths = set()
    catalogue_keys = set()
    target_entries = []
    roots = source_roots or manifest.get("source_roots", {})
    for entry in entries:
        role = entry.get("source_role")
        if role not in VALID_SOURCE_ROLES:
            raise ValueError(f"Invalid source role: {role!r}")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Source relative_path must be a non-empty relative path: {relative!r}")
        if relative in relative_paths:
            raise ValueError(f"Duplicate source relative path: {relative}")
        relative_paths.add(relative)
        if role not in roots:
            raise ValueError(f"No verification root supplied for source role {role!r}.")
        path = Path(roots[role]) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Source-manifest file is missing: {path}")
        if path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Source size mismatch for {path}")
        actual_digest = sha256_file_streaming(path, int(manifest["hash_chunk_size_bytes"]))
        if actual_digest != entry.get("sha256"):
            raise ValueError(f"Source SHA-256 mismatch for {path}")
        if role == "halo_catalogue":
            universe_id, snapshot_id = _catalogue_identity(path)
            if entry.get("universe_id") != universe_id or entry.get("snapshot_id") != snapshot_id:
                raise ValueError(f"Catalogue universe/snapshot metadata mismatch for {path}")
            key = (universe_id, snapshot_id)
            if key in catalogue_keys:
                raise ValueError(f"Duplicate halo catalogue universe/snapshot: {key}")
            catalogue_keys.add(key)
        elif role == "target_table":
            target_entries.append(entry)
            structure = inspect_target_table(path)
            if any(entry.get(key) != value for key, value in structure.items()):
                raise ValueError(f"Target-table structure metadata mismatch for {path}")

    if require_target and len(target_entries) != 1:
        raise ValueError(
            f"Full-SHA256 production source manifest requires exactly one target table; found {len(target_entries)}."
        )
    if not require_target and len(target_entries) > 1:
        raise ValueError("Source manifest supports at most one authoritative target table.")
    if manifest.get("catalogue_count") != len(catalogue_keys):
        raise ValueError("Source manifest catalogue_count does not match verified catalogues.")
    if manifest.get("target_source_count") != len(target_entries):
        raise ValueError("Source manifest target_source_count does not match verified targets.")
    return {
        "verified": True,
        "verification_result": "verified_full_sha256",
        "entry_count": len(entries),
        "catalogue_count": len(catalogue_keys),
        "target_source_count": len(target_entries),
        "manifest_sha256": expected_identity,
    }


def classify_source_provenance(metadata: Mapping[str, Any]) -> str:
    """Classify historical metadata without preventing legacy dataset loading."""
    if metadata.get("source_manifest_policy") == SOURCE_MANIFEST_POLICY_FULL:
        verification = metadata.get("source_manifest_verification", {})
        return "verified_full_sha256" if verification.get("verified") is True else "unverified_full_sha256"
    if "source_manifest_hash" in metadata or "source_manifest_sha256" in metadata:
        return "legacy_unverified_stat_only"
    return "legacy_unverified_missing"
