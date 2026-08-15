"""Unit tests for Stage 4 chunking and retrieval preparation."""

from __future__ import annotations

import json
from pathlib import Path

from doc_agent.index import chunk


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cfg(tmp_path: Path, *, chunk_tokens: int = 36, overlap: int = 6) -> dict:
    return {
        "ocr": {"manifest_path": str(tmp_path / "ocr.jsonl")},
        "chunk": {
            "strategy": "structure_aware",
            "chunk_tokens": chunk_tokens,
            "overlap": overlap,
            "heading_context_tokens": 12,
            "heading_labels": ["heading", "section_header", "title"],
            "atomic_labels": ["formula", "table", "code"],
            "manifest_path": str(tmp_path / "chunks.jsonl"),
        },
    }


def test_load_ocr_chunks_ignores_skipped_and_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_jsonl(
        Path(cfg["ocr"]["manifest_path"]),
        [
            {
                "chunk_id": "book_p0001_r0000",
                "doc_id": "book",
                "page_id": "book_p0001",
                "status": "ok",
                "text": "Functions",
            },
            {
                "chunk_id": "book_p0001_r0001",
                "doc_id": "book",
                "page_id": "book_p0001",
                "status": "skipped",
                "text": "",
            },
            {
                "chunk_id": "book_p0001_r0002",
                "doc_id": "book",
                "page_id": "book_p0001",
                "status": "rejected_repetition_or_empty",
                "text": "",
            },
        ],
    )

    loaded = chunk.load_ocr_chunks(cfg)

    assert [item.id for item in loaded] == ["book_p0001_r0000"]


def test_structure_aware_chunking_preserves_heading_formula_and_traceability(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, chunk_tokens=48, overlap=8)
    rows = [
        {
            "chunk_id": "book_p0001_r0000",
            "doc_id": "book",
            "page_id": "book_p0001",
            "kind": "heading",
            "source_label": "section_header",
            "status": "ok",
            "text": "1.1 Review of Functions",
        },
        {
            "chunk_id": "book_p0001_r0001",
            "doc_id": "book",
            "page_id": "book_p0001",
            "kind": "text",
            "source_label": "text",
            "status": "ok",
            "text": (
                "A function maps each input to exactly one output. "
                "Its domain contains all permitted input values."
            ),
        },
        {
            "chunk_id": "book_p0002_r0000",
            "doc_id": "book",
            "page_id": "book_p0002",
            "kind": "text",
            "source_label": "formula",
            "status": "ok",
            "text": r"\[f(x)=x^2\]",
        },
        {
            "chunk_id": "book_p0002_r0001",
            "doc_id": "book",
            "page_id": "book_p0002",
            "kind": "text",
            "source_label": "text",
            "status": "ok",
            "text": "The range contains the corresponding output values.",
        },
    ]
    _write_jsonl(Path(cfg["ocr"]["manifest_path"]), rows)

    built = chunk.build_from_manifest(cfg)
    first_manifest = Path(cfg["chunk"]["manifest_path"]).read_text(encoding="utf-8")
    repeated = chunk.build_from_manifest(cfg)
    second_manifest = Path(cfg["chunk"]["manifest_path"]).read_text(encoding="utf-8")

    assert [item.id for item in built] == [item.id for item in repeated]
    assert first_manifest == second_manifest
    assert all("1.1 Review of Functions" in item.text for item in built)
    assert any(r"\[f(x)=x^2\]" in item.text for item in built)

    manifest_rows = [json.loads(line) for line in first_manifest.splitlines()]
    assert manifest_rows[0]["source_chunk_ids"]
    assert set(manifest_rows[0]["page_ids"]).issubset({"book_p0001", "book_p0002"})
    assert all(row["estimated_tokens"] <= 48 for row in manifest_rows)


def test_structure_aware_chunking_adds_overlap_without_splitting_atomic_math(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, chunk_tokens=24, overlap=5)
    rows = [
        {
            "chunk_id": "book_p0001_r0000",
            "doc_id": "book",
            "page_id": "book_p0001",
            "kind": "heading",
            "source_label": "section_header",
            "status": "ok",
            "text": "Functions",
        },
        {
            "chunk_id": "book_p0001_r0001",
            "doc_id": "book",
            "page_id": "book_p0001",
            "kind": "text",
            "source_label": "text",
            "status": "ok",
            "text": (
                "First sentence introduces the domain. "
                "Second sentence explains the range. "
                "Third sentence gives another example."
            ),
        },
        {
            "chunk_id": "book_p0001_r0002",
            "doc_id": "book",
            "page_id": "book_p0001",
            "kind": "text",
            "source_label": "formula",
            "status": "ok",
            "text": r"\[D=\{x\mid x\geq 0\}\]",
        },
    ]
    _write_jsonl(Path(cfg["ocr"]["manifest_path"]), rows)

    built = chunk.build_from_manifest(cfg)
    manifest_rows = [
        json.loads(line) for line in Path(cfg["chunk"]["manifest_path"]).read_text().splitlines()
    ]

    assert len(built) >= 2
    assert any(row["has_overlap"] for row in manifest_rows[1:])
    formula_chunks = [item for item in built if r"\[D=\{x\mid x\geq 0\}\]" in item.text]
    assert len(formula_chunks) == 1
