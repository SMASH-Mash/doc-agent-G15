"""Stage 1 — load scanned page-images."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageOps

from ..contracts import Page
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def _slug(value: str) -> str:
    """Return a deterministic identifier safe for file and page IDs."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "document"


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path while still supporting absolute config paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_id(path: Path, raw_dir: Path) -> str:
    """Create a stable document ID from the source's path below data/raw."""
    try:
        relative = path.relative_to(raw_dir)
    except ValueError:
        relative = Path(path.name)
    without_suffix = relative.with_suffix("")
    return _slug("__".join(without_suffix.parts))


def _save_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _render_pdf(
    source_path: Path,
    raw_dir: Path,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> tuple[list[Page], list[dict[str, Any]]]:
    doc_id = _document_id(source_path, raw_dir)
    pages: list[Page] = []
    rows: list[dict[str, Any]] = []

    with fitz.open(source_path) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF contains no pages: {source_path}")

        for page_index in range(document.page_count):
            page_number = page_index + 1
            page_id = f"{doc_id}_p{page_number:04d}"
            output_path = output_dir / doc_id / f"{page_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if overwrite or not output_path.exists():
                pixmap = document.load_page(page_index).get_pixmap(dpi=dpi, alpha=False)
                pixmap.save(output_path)

            with Image.open(output_path) as image:
                width, height = image.size

            portable_output = _portable_path(output_path)
            pages.append(Page(id=page_id, image_path=portable_output, doc_id=doc_id))
            rows.append(
                {
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "source_path": _portable_path(source_path),
                    "source_page": page_number,
                    "image_path": portable_output,
                    "width": width,
                    "height": height,
                    "dpi": dpi,
                    "sha256": _sha256(output_path),
                }
            )

    return pages, rows


def _convert_image(
    source_path: Path,
    raw_dir: Path,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> tuple[Page, dict[str, Any]]:
    doc_id = _document_id(source_path, raw_dir)
    page_id = f"{doc_id}_p0001"
    output_path = output_dir / doc_id / f"{page_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not output_path.exists():
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(output_path, format="PNG", dpi=(dpi, dpi))

    with Image.open(output_path) as image:
        width, height = image.size

    portable_output = _portable_path(output_path)
    page = Page(id=page_id, image_path=portable_output, doc_id=doc_id)
    row = {
        "doc_id": doc_id,
        "page_id": page_id,
        "source_path": _portable_path(source_path),
        "source_page": 1,
        "image_path": portable_output,
        "width": width,
        "height": height,
        "dpi": dpi,
        "sha256": _sha256(output_path),
    }
    return page, row


def load_pages(cfg: dict) -> list[Page]:
    """Rasterise PDFs and normalise image files into deterministic ``Page`` objects.

    Configuration is read from ``cfg['ingest']``. The defaults are deliberately safe for
    the ATLAS corpus: 300-dpi RGB pages, recursive discovery below ``data/raw``, and a JSONL
    manifest containing one row per generated page.
    """
    ingest_cfg = cfg.get("ingest", {})
    raw_dir = Path(ingest_cfg.get("raw_dir", "data/raw"))
    output_dir = Path(ingest_cfg.get("pages_dir", "data/interim/pages"))
    manifest_path = Path(ingest_cfg.get("manifest_path", "data/interim/pages.jsonl"))
    dpi = int(ingest_cfg.get("dpi", 300))
    overwrite = bool(ingest_cfg.get("overwrite", False))

    if dpi < 72:
        raise ValueError(f"ingest.dpi must be at least 72, got {dpi}")
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {raw_dir}. Run scripts/get_data.sh first."
        )

    source_paths = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not source_paths:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(
            f"No supported corpus files found below {raw_dir}. Supported: {supported}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    manifest_rows: list[dict[str, Any]] = []

    for source_path in source_paths:
        if source_path.suffix.lower() == ".pdf":
            source_pages, rows = _render_pdf(
                source_path=source_path,
                raw_dir=raw_dir,
                output_dir=output_dir,
                dpi=dpi,
                overwrite=overwrite,
            )
            pages.extend(source_pages)
            manifest_rows.extend(rows)
        else:
            page, row = _convert_image(
                source_path=source_path,
                raw_dir=raw_dir,
                output_dir=output_dir,
                dpi=dpi,
                overwrite=overwrite,
            )
            pages.append(page)
            manifest_rows.append(row)

    page_ids = [page.id for page in pages]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Generated duplicate page IDs; check for colliding corpus filenames")

    _save_manifest(manifest_rows, manifest_path)
    LOGGER.info(
        "ingest_complete pages=%d documents=%d manifest=%s",
        len(pages),
        len({page.doc_id for page in pages}),
        manifest_path,
    )
    return pages
