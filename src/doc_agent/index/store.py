"""Stage 4 — vector store"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

INDEX_DIR = Path("data/interim/index")  # under data/interim/, already gitignored


def build(chunks: list[Chunk], vectors: Any, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']). Only 'faiss:flat' is implemented.

    We originally used faiss.IndexHNSWFlat (the graph-based ANN index that reproduces Malkov &
    Yashunin's published algorithm). Live testing on this dev machine isolated a reproducible
    native crash (Windows STATUS_ACCESS_VIOLATION) inside faiss-cpu 1.8.0's IndexHNSWFlat.add()
    itself -- confirmed with a bare numpy+faiss repro that has zero ML models involved, so it was
    never an OCR/embedding interaction as first suspected. faiss.IndexFlatIP does not crash on
    identical data, so that's what we build here: exact (not approximate) inner-product search
    over L2-normalized vectors, i.e. exact cosine similarity. At our scale (thousands, not
    millions, of chunks) exact search is fast enough that losing HNSW's approximate-NN speedup
    costs us nothing observable, and it's provably correct (no recall/accuracy trade-off at all).
    Also writes a meta.jsonl chunk sidecar (FAISS stores vectors only) and an index_stats.json
    summary that later stages/notebooks read back rather than hand-transcribing numbers."""
    import faiss

    index_type = cfg["index"]["type"]
    if index_type != "faiss:flat":
        raise NotImplementedError(f"index type {index_type!r} not implemented (only faiss:flat)")

    vecs = np.asarray(vectors, dtype="float32")
    dim = vecs.shape[1] if vecs.size else cfg["embed"]["dim"]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    if vecs.size:
        index.add(vecs)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps({"id": c.id, "doc_id": c.doc_id, "text": c.text, "page_ids": c.page_ids})
                + "\n"
            )

    stats = {
        "n_chunks": len(chunks),
        "dim": dim,
        "index_type": index_type,
        "embed_model": cfg["embed"]["model"],
        "ocr_model": cfg["ocr"]["model"],
        "layout_model": cfg["layout"]["model"],
        "chunk_tokens": cfg["index"]["chunk_tokens"],
        "overlap": cfg["index"]["overlap"],
        "docs": sorted({c.doc_id for c in chunks}),
        "dev_max_pages": cfg.get("dev", {}).get("max_pages", 0),
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(INDEX_DIR / "index_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"store.build: {len(chunks)} chunks, dim={dim}, index={index_type} -> {INDEX_DIR}")


def load(cfg: dict) -> tuple[Any, list[Chunk]]:
    """Load the persisted FAISS index + its chunk metadata sidecar. IMPLEMENT."""
    import faiss

    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks: list[Chunk] = []
    with open(INDEX_DIR / "meta.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunks.append(
                    Chunk(id=r["id"], doc_id=r["doc_id"], text=r["text"], page_ids=r["page_ids"])
                )
    return index, chunks
