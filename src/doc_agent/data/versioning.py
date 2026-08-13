"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

import hashlib
from pathlib import Path


def snapshot(corpus_dir: str) -> str:
    """Hash the corpus directory's file list + sizes (mtimes ignored, so a re-run over identical
    bytes is stable) into a short version id. A1's commitment: 'corpus is versioned by a manifest
    checksum'; used as the re-ingest trigger if a source is silently edited."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"corpus dir {corpus_dir} does not exist")
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(str(path.stat().st_size).encode("utf-8"))
    return h.hexdigest()[:16]
