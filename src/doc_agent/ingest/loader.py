"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import *  # noqa
from ..data.validate import validate

DATA_RAW = Path("data/raw")
MANIFEST_PATH = DATA_RAW / "manifest.jsonl"


def _split_page_id(page_id: str) -> tuple[str, int]:
    """page_id scheme: f'{doc_id}_p{page_num:04d}' (doc_ids never contain '_p')."""
    doc_id, sep, num_part = page_id.rpartition("_p")
    if not sep:
        raise ValueError(f"page_id {page_id!r} does not match the '<doc_id>_p<NNNN>' scheme")
    return doc_id, int(num_part)


def doc_id_for(page_id: str) -> str:
    """page_id -> the document/book it belongs to. contracts.Region only carries page_id (fixed
    contract, no doc_id field), so vision/ocr.py uses this to build Chunk.doc_id."""
    doc_id, _ = _split_page_id(page_id)
    return doc_id


def page_num_for(page_id: str) -> int:
    """page_id -> its 1-indexed page number within its document (used by vision/layout.py's
    PyMuPDF fallback to index into the original source PDF)."""
    _, page_num = _split_page_id(page_id)
    return page_num


def image_path_for(page_id: str) -> str:
    """Deterministic page_id -> rasterised PNG path, matching scripts/get_data.sh's on-disk layout
    (data/raw/<doc_id>/<page_num:04d>.png). contracts.Region only carries page_id + bbox (fixed
    contract, no image path field), so vision/layout.py and vision/ocr.py both resolve the actual
    image file through this single shared helper rather than duplicating the naming convention."""
    doc_id, page_num = _split_page_id(page_id)
    return str(DATA_RAW / doc_id / f"{page_num:04d}.png")


def _stratified_sample(pages: list[Page], max_pages: int) -> list[Page]:
    """Even sample across all documents (every Nth page within each book, not just the first N
    pages, which would be front matter/table of contents) so a small dev.max_pages smoke-test run
    still exercises every book's content, not just the alphabetically-first one."""
    if max_pages <= 0 or len(pages) <= max_pages:
        return pages
    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)
    per_doc = max(1, max_pages // len(by_doc))
    sampled: list[Page] = []
    for doc_pages in by_doc.values():
        step = max(1, len(doc_pages) // per_doc)
        sampled.extend(doc_pages[::step][:per_doc])
    sampled_ids = {p.id for p in sampled}
    return [p for p in pages if p.id in sampled_ids][:max_pages]


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/manifest.jsonl (written by scripts/get_data.sh) -> list[Page]. Validates the
    FULL downloaded corpus against the >=300 pages / >=60k words floor before applying
    cfg['dev']['max_pages'] dev-mode sampling, so the floor check always reflects the real corpus,
    not a small test-mode subset (0 = no limit = the full corpus, the actual A2 deliverable run)."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found -- run `bash scripts/get_data.sh` first to fetch the corpus."
        )
    records = [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_pages = [
        Page(
            id=f"{r['book_id']}_p{r['page_num']:04d}",
            image_path=r["image_path"],
            doc_id=r["book_id"],
        )
        for r in records
    ]
    validate(all_pages)

    max_pages = cfg.get("dev", {}).get("max_pages", 0)
    return _stratified_sample(all_pages, max_pages)
