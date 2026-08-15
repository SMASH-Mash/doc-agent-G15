"""Unit tests for deterministic corpus ingestion and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from PIL import Image

from doc_agent.ingest import loader, preprocess


def _make_two_page_pdf(path: Path) -> None:
    document = pymupdf.open()
    for label in ("Page one: x^2", "Page two: integral"):
        page = document.new_page(width=240, height=320)
        page.insert_text((30, 50), label)
    document.save(path)
    document.close()


def test_load_pages_rasterises_pdf_and_writes_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_two_page_pdf(raw_dir / "Calculus Vol 1.pdf")

    cfg = {
        "ingest": {
            "raw_dir": str(raw_dir),
            "pages_dir": str(tmp_path / "pages"),
            "manifest_path": str(tmp_path / "pages.jsonl"),
            "dpi": 144,
        }
    }

    pages = loader.load_pages(cfg)

    assert [page.id for page in pages] == ["calculus_vol_1_p0001", "calculus_vol_1_p0002"]
    assert all(Path(page.image_path).exists() for page in pages)
    assert {page.doc_id for page in pages} == {"calculus_vol_1"}

    rows = [json.loads(line) for line in (tmp_path / "pages.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["source_page"] == 1
    assert rows[1]["source_page"] == 2
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_load_pages_accepts_standalone_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    nested = raw_dir / "siyavula"
    nested.mkdir(parents=True)
    Image.new("RGB", (80, 60), "white").save(nested / "exercise.jpg")

    cfg = {
        "ingest": {
            "raw_dir": str(raw_dir),
            "pages_dir": str(tmp_path / "pages"),
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "dpi": 300,
        }
    }

    pages = loader.load_pages(cfg)

    assert len(pages) == 1
    assert pages[0].id == "siyavula_exercise_p0001"
    assert Path(pages[0].image_path).suffix == ".png"


def test_preprocess_preserves_clean_rgb_pages_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 90), (250, 249, 248)).save(source)
    page = loader.Page(id="book_p0001", image_path=str(source), doc_id="book")
    cfg = {
        "preprocess": {
            "enabled": True,
            "output_dir": str(tmp_path / "preprocessed"),
            "deskew": False,
            "denoise": False,
            "autocontrast": False,
            "grayscale": False,
            "binarize": False,
        }
    }

    result = preprocess.run([page], cfg)

    assert len(result) == 1
    output_path = Path(result[0].image_path)
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.size == (120, 90)
