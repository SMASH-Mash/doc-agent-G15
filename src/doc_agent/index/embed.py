"""Stage 4B — embed chunks with a sentence-transformers model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..contracts import Chunk
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)


def _resolve_device(requested: str) -> str:
    """Resolve the configured device, falling back to CPU when CUDA is unavailable."""
    requested = requested.strip().lower()

    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested for embeddings but is unavailable; falling back to CPU")
        return "cpu"

    return requested


def _texts(chunks: Sequence[Chunk]) -> list[str]:
    """Extract and validate chunk text."""
    texts = [str(chunk.text).strip() for chunk in chunks]

    if any(not text for text in texts):
        raise ValueError("Embedding input contains an empty chunk")

    return texts


def encode(chunks: list[Chunk], cfg: dict[str, Any]) -> np.ndarray:
    """Embed chunks and return a float32 matrix shaped (n_chunks, dim)."""
    embed_cfg = cfg.get("embed", {})

    expected_dim = int(embed_cfg.get("dim", 0))

    if not chunks:
        if expected_dim <= 0:
            raise ValueError("embed.dim must be positive when encoding an empty chunk list")

        return np.empty((0, expected_dim), dtype=np.float32)

    model_name = str(embed_cfg.get("model", "BAAI/bge-m3"))
    model_revision = str(
        embed_cfg.get(
            "revision",
            "142964af7e05de16511657561de8e8750fc153a0",
        )
    )
    normalize = bool(embed_cfg.get("normalize", True))
    batch_size = int(embed_cfg.get("batch_size", 2))

    if batch_size <= 0:
        raise ValueError("embed.batch_size must be positive")

    device = _resolve_device(str(embed_cfg.get("device", cfg.get("device", "cpu"))))

    texts = _texts(chunks)

    LOGGER.info(
        "loading embedding model=%s device=%s batch_size=%d normalize=%s",
        model_name,
        device,
        batch_size,
        normalize,
    )

    model = SentenceTransformer(
        model_name,
        device=device,
        revision=model_revision,
    )

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding output must have shape " f"({len(chunks)}, dim), got {matrix.shape}"
        )

    if expected_dim > 0 and matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: " f"expected {expected_dim}, got {matrix.shape[1]}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Embedding output contains non-finite values")

    return matrix
