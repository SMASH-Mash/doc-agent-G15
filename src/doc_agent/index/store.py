"""Stage 4 — FAISS vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ..contracts import *  # noqa: F401,F403


def _paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    index_cfg = cfg.get("index", {})

    index_dir = Path(index_cfg.get("path", "data/index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    return (
        index_dir / "index.faiss",
        index_dir / "chunks.json",
    )


def build(chunks, vectors, cfg: dict) -> None:
    """Build and persist the configured FAISS index plus chunk metadata."""
    index_cfg = cfg.get("index", {})
    index_type = str(index_cfg.get("type", "faiss:flat_ip"))

    if index_type != "faiss:flat_ip":
        raise ValueError(f"Unsupported index type: {index_type}")

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {matrix.shape}")

    if len(chunks) != matrix.shape[0]:
        raise ValueError(
            f"chunk/vector count mismatch: " f"{len(chunks)} chunks vs {matrix.shape[0]} vectors"
        )

    if matrix.shape[0] == 0:
        raise ValueError("Cannot build an index from zero chunks")

    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")

    # The embedding stage normalizes vectors. Inner product therefore
    # corresponds to cosine similarity.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_path, metadata_path = _paths(cfg)

    faiss.write_index(index, str(index_path))

    records = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "page_ids": list(chunk.page_ids),
                "text": chunk.text,
            }
        )

    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(cfg: dict):
    """Load the persisted FAISS index and its chunk metadata."""
    index_path, metadata_path = _paths(cfg)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")

    index = faiss.read_index(str(index_path))

    records = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Chunk metadata must contain a JSON list")

    if index.ntotal != len(records):
        raise ValueError(
            f"Index/metadata mismatch: " f"{index.ntotal} vectors vs {len(records)} records"
        )

    return index, records
