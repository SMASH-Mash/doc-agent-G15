"""Stage 4 — embed chunks"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    if model_name not in _model_cache:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
        logger.info(f"embed: loaded {model_name} on {device}")
    return _model_cache[model_name]


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed with cfg['embed']['model']. L2-normalized so index/store.py's cosine-via-inner-product
    HNSW index is correct. IMPLEMENT."""
    model = _get_model(cfg["embed"]["model"])
    if not chunks:
        return np.zeros((0, cfg["embed"]["dim"]), dtype="float32")
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=cfg["embed"].get("batch_size", 32),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")
