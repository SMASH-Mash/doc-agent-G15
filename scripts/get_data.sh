#!/usr/bin/env bash
# A1 — fetch or recreate the scanned corpus into data/raw/
#
# Corpus (see data/provenance.md): Siyavula Grade 11 & 12 Mathematics (CC BY) + OpenStax Calculus
# Volume 1 & 2 (CC BY-NC-SA 4.0). Sources are each publisher's own canonical PDF download link
# (verified reachable directly, no auth/quota needed) rather than the team's Drive mirror, per the
# handbook's preference for stable public sources. Each PDF is saved directly to
# data/raw/<book_id>.pdf (no pre-rasterization here -- ingest/loader.py rasterises PDFs itself,
# at cfg['ingest']['dpi'], into data/interim/pages/; doing it twice would duplicate every page
# under mismatched doc_ids). data/raw/manifest.jsonl records one row per book (book_id, title,
# source_url, license, pdf_path, pdf_sha256, page_count) for reproducibility (data/versioning.py
# hashes this directory).
#
# Usage: bash scripts/get_data.sh            # fetch all 4 books
#        bash scripts/get_data.sh gr11       # fetch just one book (siyavula_gr11 |
#                                             # siyavula_gr12 | openstax_calc1 | openstax_calc2)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw   # gitignored; PDFs land here directly, bare -- this is what loader.py expects

PYBIN="${PYTHON:-python}"
command -v uv >/dev/null 2>&1 && PYBIN="uv run python"

$PYBIN - "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pymupdf

DATA_RAW = Path("data/raw")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) doc-agent-G15/A2 (educational corpus fetch)"

BOOKS = {
    "siyavula_gr11": {
        "url": "https://www.siyavula.com/downloads/books/maths/Gr11_Mathematics_Learner_Eng.pdf",
        "title": "Siyavula Grade 11 Mathematics",
        "license": "CC BY",
    },
    "siyavula_gr12": {
        "url": "https://www.siyavula.com/downloads/books/maths/Gr12_Mathematics_Learner_Eng.pdf",
        "title": "Siyavula Grade 12 Mathematics",
        "license": "CC BY",
    },
    "openstax_calc1": {
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/calculus-volume-1_-_WEB.pdf",
        "title": "OpenStax Calculus Volume 1",
        "license": "CC BY-NC-SA 4.0",
    },
    "openstax_calc2": {
        "url": "https://assets.openstax.org/oscms-prodcms/media/documents/calculus-volume-2_-_WEB.pdf",
        "title": "OpenStax Calculus Volume 2",
        "license": "CC BY-NC-SA 4.0",
    },
}


def download(book_id: str, url: str) -> Path:
    import urllib.request

    dest = DATA_RAW / f"{book_id}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{book_id}] already downloaded ({dest.stat().st_size:,} bytes), skipping fetch")
        return dest
    print(f"[{book_id}] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"[{book_id}] saved {dest.stat().st_size:,} bytes")
    return dest


def probe(book_id: str, pdf_path: Path) -> dict:
    """Verify the PDF opens and record its page count -- no rendering/saving of images.
    Rasterization happens once, in ingest/loader.py, at build time."""
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
    print(f"[{book_id}] verified {page_count} pages, sha256={sha256[:12]}...")
    return {
        "book_id": book_id,
        "title": BOOKS[book_id]["title"],
        "source_url": BOOKS[book_id]["url"],
        "license": BOOKS[book_id]["license"],
        "pdf_path": str(pdf_path).replace("\\", "/"),
        "pdf_sha256": sha256,
        "page_count": page_count,
    }


def _write_manifest(manifest_path: Path, manifest: list[dict]) -> None:
    # Rewritten after EVERY book (cheap -- a handful of short JSON lines) so a crash or network
    # blip partway through a run never loses a prior book's already-recorded manifest row, unlike
    # a single write-at-the-end which silently drops everything on an unhandled exception.
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str]) -> None:
    requested = argv[1:] or None  # e.g. ["gr11"] -> only (re)fetch books matching this token
    manifest_path = DATA_RAW / "manifest.jsonl"
    manifest: list[dict] = []

    # download() is idempotent (skips an already-downloaded PDF), so it's always safe and cheap
    # to reprocess every book on every run -- this also self-heals a manifest left incomplete by
    # an earlier interrupted run.
    for book_id, meta in BOOKS.items():
        if requested and not any(tok in book_id for tok in requested):
            pdf_path = DATA_RAW / f"{book_id}.pdf"
            if pdf_path.exists():
                # not requested this run, but keep its already-downloaded book in the manifest
                manifest.append(probe(book_id, pdf_path))
            continue
        pdf_path = download(book_id, meta["url"])
        manifest.append(probe(book_id, pdf_path))
        _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    by_book = sorted({r["book_id"] for r in manifest})
    total_pages = sum(r["page_count"] for r in manifest)
    print(f"\nWrote {manifest_path}: {len(manifest)} books, {total_pages} total pages: {by_book}")


if __name__ == "__main__":
    main(sys.argv)
PY
