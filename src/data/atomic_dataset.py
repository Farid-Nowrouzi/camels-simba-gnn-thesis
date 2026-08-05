"""Atomic, checksummed writer used by the explicit sparse dataset path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import torch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json_fsync(path: Path, value: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_sparse_dataset(
    dataset: Dict[str, Any],
    output_path: str | Path,
    metadata: Dict[str, Any],
    validate: Optional[Callable[[Dict[str, Any]], None]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write data/temp metadata, validate/checksum, atomically publish, then mark complete."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".metadata.json")
    complete_path = output.with_suffix(".complete")
    lock_path = output.with_suffix(output.suffix + ".lock")

    if complete_path.exists() and not overwrite:
        raise FileExistsError(f"Completed dataset already exists: {output}")
    existing = [path for path in (output, metadata_path, complete_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Partial or existing dataset paths require --overwrite: {existing}")

    lock_fd = None
    token = uuid4().hex
    temp_output = output.with_name(f".{output.name}.{token}.tmp")
    temp_metadata = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(lock_fd, f"pid={os.getpid()} token={token}\n".encode("utf-8"))
        os.fsync(lock_fd)

        if overwrite:
            for path in (complete_path, metadata_path, output):
                if path.exists():
                    path.unlink()

        torch.save(dataset, temp_output)
        _fsync_file(temp_output)
        if validate is not None:
            validate(dataset)
        checksum = sha256_file(temp_output)
        final_metadata = dict(metadata)
        final_metadata.update({
            "checksum_algorithm": "sha256",
            "checksum": checksum,
            "completion_status": "complete",
        })
        _write_json_fsync(temp_metadata, final_metadata)
        os.replace(temp_output, output)
        os.replace(temp_metadata, metadata_path)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _write_json_fsync(complete_path, {
            "dataset": output.name,
            "metadata": metadata_path.name,
            "sha256": checksum,
            "status": "complete",
        })
        return final_metadata
    finally:
        for path in (temp_output, temp_metadata):
            if path.exists():
                path.unlink()
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()
