"""
src/utils/fingerprint_utils.py

Content-based fingerprint utilities.

This module is responsible ONLY for calculating fingerprints of
physical files and directories.

It does not know anything about:

    Data Pipeline
    ML Pipeline
    model training
    feature engineering
    MLflow
    pipeline state

PipelineStateManager uses these utilities indirectly through the
orchestrators.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


DEFAULT_CHUNK_SIZE = 1024 * 1024


# ============================================================================
# File Fingerprinting
# ============================================================================


def hash_file(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Calculate SHA-256 fingerprint of a file's contents.

    Parameters
    ----------
    file_path:
        File to hash.

    chunk_size:
        Number of bytes read at a time.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.

    Example
    -------
    hash_file("data/raw/postings.csv")
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot fingerprint missing file: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but received: {path}"
        )

    sha256 = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ============================================================================
# Directory Fingerprinting
# ============================================================================


def hash_directory(
    directory: str | Path,
) -> str:
    """
    Calculate a deterministic fingerprint for an entire directory.

    The fingerprint depends on:

        relative file paths
        file contents

    Therefore the fingerprint changes when:

        a file is added
        a file is removed
        a file is modified
        a file is renamed

    File ordering is deterministic.
    """

    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"Cannot fingerprint missing directory: {root}"
        )

    if not root.is_dir():
        raise ValueError(
            f"Expected a directory but received: {root}"
        )

    file_hashes: list[
        tuple[str, str]
    ] = []

    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
        ),
        key=lambda path: str(
            path.relative_to(root)
        ).replace("\\", "/"),
    )

    for path in files:

        relative_path = str(
            path.relative_to(root)
        ).replace(
            "\\",
            "/",
        )

        content_hash = hash_file(
            path
        )

        file_hashes.append(
            (
                relative_path,
                content_hash,
            )
        )

    canonical_representation = repr(
        file_hashes
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical_representation
    ).hexdigest()


# ============================================================================
# Multiple File Fingerprinting
# ============================================================================


def hash_files(
    file_paths: Iterable[
        str | Path
    ],
) -> str:
    """
    Calculate one deterministic fingerprint from multiple files.

    Useful when a stage depends on specific files from different
    locations.
    """

    entries: list[
        tuple[str, str]
    ] = []

    for file_path in file_paths:

        path = Path(file_path)

        content_hash = hash_file(
            path
        )

        entries.append(
            (
                str(path).replace(
                    "\\",
                    "/",
                ),
                content_hash,
            )
        )

    entries.sort(
        key=lambda item: item[0]
    )

    canonical_representation = repr(
        entries
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical_representation
    ).hexdigest()