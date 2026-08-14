"""Unit tests for data validation and corpus versioning."""

from __future__ import annotations

import pytest

from doc_agent.contracts import Page
from doc_agent.data.validate import validate
from doc_agent.data.versioning import snapshot


def test_validate_accepts_valid_pages(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")

    pages = [Page(id="book_p001", image_path=str(image), doc_id="book")]
    validate(pages)


def test_validate_rejects_empty_pages():
    with pytest.raises(ValueError, match="no pages"):
        validate([])


def test_validate_rejects_duplicate_page_ids(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    page = Page(id="book_p001", image_path=str(image), doc_id="book")

    with pytest.raises(ValueError, match="duplicate page id"):
        validate([page, page])


def test_validate_rejects_missing_images(tmp_path):
    page = Page(id="book_p001", image_path=str(tmp_path / "missing.png"), doc_id="book")

    with pytest.raises(ValueError, match="missing image"):
        validate([page])


def test_validate_rejects_unsupported_image_format(tmp_path):
    image = tmp_path / "page.pdf"
    image.write_bytes(b"pdf")
    page = Page(id="book_p001", image_path=str(image), doc_id="book")

    with pytest.raises(ValueError, match="unsupported image format"):
        validate([page])


def test_snapshot_is_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("beta")

    first = snapshot(str(tmp_path))
    second = snapshot(str(tmp_path))

    assert first == second
    assert len(first) == 64


def test_snapshot_changes_when_corpus_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("alpha")
    first = snapshot(str(tmp_path))

    path.write_text("beta")
    second = snapshot(str(tmp_path))

    assert first != second


def test_snapshot_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        snapshot(str(tmp_path / "missing"))


def test_snapshot_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        snapshot(str(tmp_path))
