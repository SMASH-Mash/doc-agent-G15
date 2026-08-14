#!/usr/bin/env bash
# A1 — fetch or recreate the scanned corpus into data/raw/
#
# Corpus (see data/provenance.md): Siyavula Grade 11 & 12 Mathematics (CC BY) + OpenStax Calculus
# Volume 1 & 2 (CC BY-NC-SA 4.0). Sources are each publisher's own canonical PDF download link
# (verified reachable directly, no auth/quota needed) rather than the team's Drive mirror, per the
# handbook's preference for stable public sources. Every page is rasterised at 300dpi RGB into
# data/raw/<book_id>/<page_num:04d>.png, plus data/raw/manifest.jsonl recording book_id/page_num/
# source_url/pdf_sha256 for reproducibility (data/versioning.py hashes this directory).
#
# Usage: bash scripts/get_data.sh            # fetch + rasterize all 4 books
#        bash scripts/get_data.sh gr11       # fetch + rasterize just one book (siyavula_gr11 |
#                                             # siyavula_gr12 | openstax_calc1 | openstax_calc2)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/_source_pdfs   # under data/raw/, already gitignored; PDFs are the download cache

PYBIN="${PYTHON:-python}"
command -v uv >/dev/null 2>&1 && PYBIN="uv run python"

$PYBIN - "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pymupdf

DATA_RAW = Path("data/raw")
PDF_DIR = DATA_RAW / "_source_pdfs"
DPI = 300
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

    dest = PDF_DIR / f"{book_id}.pdf"
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


def rasterize(book_id: str, pdf_path: Path) -> list[dict]:
    out_dir = DATA_RAW / book_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    records = []
    with pymupdf.open(pdf_path) as doc:
        zoom = DPI / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        n = doc.page_count
        for i, page in enumerate(doc, start=1):
            out_path = out_dir / f"{i:04d}.png"
            if not out_path.exists():
                pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB, alpha=False)
                pix.save(out_path)
            records.append(
                {
                    "book_id": book_id,
                    "page_num": i,
                    "image_path": str(out_path).replace("\\", "/"),
                    "source_url": BOOKS[book_id]["url"],
                    "pdf_sha256": sha256,
                }
            )
            if i % 100 == 0 or i == n:
                print(f"[{book_id}] rasterized {i}/{n} pages", flush=True)
    return records


def _write_manifest(manifest_path: Path, manifest: list[dict]) -> None:
    # Rewritten after EVERY book (cheap -- a few thousand short JSON lines) so a crash or network
    # blip partway through a run never loses a prior book's already-rasterized pages, unlike a
    # single write-at-the-end which silently drops everything on an unhandled exception.
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")


def main(argv: list[str]) -> None:
    requested = argv[1:] or None  # e.g. ["gr11"] -> only (re)fetch books matching this token
    manifest_path = DATA_RAW / "manifest.jsonl"
    manifest: list[dict] = []

    # download()/rasterize() are both idempotent (skip already-downloaded PDFs / already-rendered
    # pages), so it's always safe and cheap to reprocess every book on every run -- this also
    # self-heals a manifest left incomplete by an earlier interrupted run, since the actual page
    # images from that run are already on disk and just need to be re-listed, not re-rendered.
    for book_id, meta in BOOKS.items():
        if requested and not any(tok in book_id for tok in requested):
            if (DATA_RAW / book_id).exists():
                # not requested this run, but keep its already-known pages in the manifest
                manifest.extend(rasterize(book_id, PDF_DIR / f"{book_id}.pdf"))
            continue
        pdf_path = download(book_id, meta["url"])
        manifest.extend(rasterize(book_id, pdf_path))
        _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    by_book = sorted({r["book_id"] for r in manifest})
    print(f"\nWrote {manifest_path} with {len(manifest)} page records across {len(by_book)} books: {by_book}")


if __name__ == "__main__":
    main(sys.argv)
PY
