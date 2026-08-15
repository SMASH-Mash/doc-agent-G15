"""Content-addressed corpus versioning."""

from __future__ import annotations

import hashlib
from pathlib import Path


def snapshot(corpus_dir: str) -> str:
    """Return a deterministic SHA-256 identifier for all files in ``corpus_dir``."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Corpus directory is empty: {corpus_dir}")

    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")

    return digest.hexdigest()
