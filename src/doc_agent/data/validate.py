"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pymupdf

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

MIN_PAGES = 300
MIN_WORDS = 60_000

# Split assignment for our 4-book corpus (see data/provenance.md), checked here for cross-split
# leakage: train = siyavula_gr11 + openstax_calc1, test = siyavula_gr12 + openstax_calc2.
TRAIN_DOCS = {"siyavula_gr11", "openstax_calc1"}
TEST_DOCS = {"siyavula_gr12", "openstax_calc2"}


def validate(pages: list[Page]) -> None:
    """Assert >=300 pages, >=60k words, and that no document is assigned to both splits
    (leakage rule). Raises AssertionError with the specific failure, so callers get an actionable
    message rather than a silent pass."""
    if len(pages) < MIN_PAGES:
        raise AssertionError(f"corpus has {len(pages)} pages, need >= {MIN_PAGES}")

    by_doc = Counter(p.doc_id for p in pages)
    leaked = set(by_doc) & TRAIN_DOCS & TEST_DOCS
    if leaked:
        raise AssertionError(
            f"document(s) {leaked} assigned to both train and test splits -- leakage"
        )

    words = _estimate_word_count(by_doc.keys())
    if words < MIN_WORDS:
        raise AssertionError(f"corpus has an estimated {words} words, need >= {MIN_WORDS}")

    logger.info(
        f"data.validate: {len(pages)} pages, ~{words} words across {len(by_doc)} documents -- OK"
    )


def _estimate_word_count(doc_ids: Iterable[str]) -> int:
    """QC-only word count via each source PDF's embedded text layer. This is NEVER used as the
    OCR result for the knowledge base -- the pipeline deliberately discards that layer and reads
    pixels via Nougat instead (A1_form.md Section 5's stated trade-off, kept for A2). Here it is
    only a corpus-scale sanity check, to catch a truncated/partial download before wasting compute
    on the real OCR stage. If a source PDF isn't present (e.g. only a dev-mode page subset was
    fetched), that document is skipped rather than failing the whole check."""
    total = 0
    for doc_id in doc_ids:
        pdf_path = Path("data/raw/_source_pdfs") / f"{doc_id}.pdf"
        if not pdf_path.exists():
            logger.info(f"data.validate: {pdf_path} not found, skipping word estimate for {doc_id}")
            continue
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                total += len(page.get_text("text").split())
    return total
