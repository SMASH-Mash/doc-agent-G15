"""Data quality checks for the ingested page manifest."""

from __future__ import annotations

from pathlib import Path

from ..contracts import Page


def validate(pages: list[Page]) -> None:
    """Validate page identity, paths, and document/page uniqueness.

    Corpus size and split-level leakage checks require corpus metadata that is not
    present on the fixed ``Page`` contract, so those checks belong at the manifest
    or dataset level rather than being inferred here.
    """
    if not pages:
        raise ValueError("Corpus validation failed: no pages were provided")

    page_ids: set[str] = set()
    doc_page_pairs: set[tuple[str, str]] = set()

    for page in pages:
        if not page.id.strip():
            raise ValueError("Corpus validation failed: page id is empty")
        if not page.doc_id.strip():
            raise ValueError(f"Corpus validation failed: {page.id} has no doc_id")

        image_path = Path(page.image_path)
        if not image_path.is_file():
            raise ValueError(
                f"Corpus validation failed: missing image for {page.id}: {page.image_path}"
            )
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            raise ValueError(
                f"Corpus validation failed: unsupported image format for {page.id}: "
                f"{image_path.suffix or '<none>'}"
            )

        if page.id in page_ids:
            raise ValueError(f"Corpus validation failed: duplicate page id: {page.id}")
        page_ids.add(page.id)

        pair = (page.doc_id, page.id)
        if pair in doc_page_pairs:
            raise ValueError(f"Corpus validation failed: duplicate document/page pair: {pair}")
        doc_page_pairs.add(pair)
