"""Stage 4A — deterministic, structure-aware chunking."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contracts import Chunk
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r".*?(?:[.!?](?=\s|$)|\n+|$)", re.DOTALL)
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class _Unit:
    chunk_id: str
    doc_id: str
    text: str
    page_ids: tuple[str, ...]
    source_label: str
    kind: str


@dataclass(frozen=True)
class _Piece:
    text: str
    source_chunk_ids: tuple[str, ...]
    page_ids: tuple[str, ...]
    source_labels: tuple[str, ...]
    atomic: bool = False
    overlap: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required OCR manifest does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _normalise_text(text: str) -> str:
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _TOKEN_RE.finditer(text)]


def estimate_tokens(text: str) -> int:
    """Return a deterministic token estimate without loading the embedding model."""
    return len(_token_spans(text))


def _slice_by_tokens(text: str, start: int, stop: int) -> str:
    spans = _token_spans(text)
    if not spans or start >= len(spans) or start >= stop:
        return ""
    stop = min(stop, len(spans))
    return text[spans[start][0] : spans[stop - 1][1]].strip()


def _tail_by_tokens(text: str, limit: int) -> str:
    count = estimate_tokens(text)
    if limit <= 0 or count == 0:
        return ""
    return _slice_by_tokens(text, max(0, count - limit), count)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def load_ocr_chunks(cfg: dict[str, Any]) -> list[Chunk]:
    """Load successful region OCR records in canonical reading order."""
    ocr_cfg = cfg.get("ocr", {})
    path = Path(ocr_cfg.get("manifest_path", "data/interim/ocr/chunks.jsonl"))
    chunks: list[Chunk] = []
    for row in _read_jsonl(path):
        if row.get("status") != "ok" or not str(row.get("text", "")).strip():
            continue
        chunks.append(
            Chunk(
                id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                page_ids=[str(row["page_id"])],
            )
        )
    return chunks


def _metadata_by_id(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(cfg.get("ocr", {}).get("manifest_path", "data/interim/ocr/chunks.jsonl"))
    if not path.exists():
        return {}
    return {str(row["chunk_id"]): row for row in _read_jsonl(path) if row.get("chunk_id")}


def _units(chunks: list[Chunk], cfg: dict[str, Any]) -> list[_Unit]:
    metadata = _metadata_by_id(cfg)
    units: list[_Unit] = []
    for chunk in chunks:
        text = _normalise_text(chunk.text)
        if not text:
            continue
        row = metadata.get(chunk.id, {})
        units.append(
            _Unit(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                text=text,
                page_ids=tuple(chunk.page_ids),
                source_label=str(row.get("source_label", row.get("kind", "text"))),
                kind=str(row.get("kind", "text")),
            )
        )
    return units


def _sentence_segments(text: str) -> list[str]:
    segments = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    return [segment for segment in segments if segment]


def _split_non_atomic(unit: _Unit, limit: int) -> list[_Piece]:
    if estimate_tokens(unit.text) <= limit:
        return [
            _Piece(
                text=unit.text,
                source_chunk_ids=(unit.chunk_id,),
                page_ids=unit.page_ids,
                source_labels=(unit.source_label,),
            )
        ]

    segments = _sentence_segments(unit.text)
    if not segments:
        segments = [unit.text]

    pieces: list[_Piece] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def emit_buffer() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        pieces.append(
            _Piece(
                text=" ".join(buffer).strip(),
                source_chunk_ids=(unit.chunk_id,),
                page_ids=unit.page_ids,
                source_labels=(unit.source_label,),
            )
        )
        buffer = []
        buffer_tokens = 0

    for segment in segments:
        segment_tokens = estimate_tokens(segment)
        if segment_tokens > limit:
            emit_buffer()
            start = 0
            while start < segment_tokens:
                text = _slice_by_tokens(segment, start, start + limit)
                if text:
                    pieces.append(
                        _Piece(
                            text=text,
                            source_chunk_ids=(unit.chunk_id,),
                            page_ids=unit.page_ids,
                            source_labels=(unit.source_label,),
                        )
                    )
                start += limit
            continue

        if buffer and buffer_tokens + segment_tokens > limit:
            emit_buffer()
        buffer.append(segment)
        buffer_tokens += segment_tokens

    emit_buffer()
    return pieces


def _piece_for_atomic(unit: _Unit) -> _Piece:
    return _Piece(
        text=unit.text,
        source_chunk_ids=(unit.chunk_id,),
        page_ids=unit.page_ids,
        source_labels=(unit.source_label,),
        atomic=True,
    )


def _overlap_tail(pieces: list[_Piece], limit: int) -> list[_Piece]:
    if limit <= 0:
        return []

    selected: list[_Piece] = []
    remaining = limit
    for piece in reversed(pieces):
        if piece.atomic or piece.overlap:
            continue
        count = estimate_tokens(piece.text)
        if count <= remaining:
            selected.append(replace(piece, overlap=True))
            remaining -= count
        else:
            tail = _tail_by_tokens(piece.text, remaining)
            if tail:
                selected.append(replace(piece, text=tail, overlap=True))
            remaining = 0
        if remaining <= 0:
            break
    return list(reversed(selected))


def _heading_text(headings: list[_Unit], max_tokens: int) -> str:
    text = "\n".join(unit.text for unit in headings).strip()
    if estimate_tokens(text) <= max_tokens:
        return text
    return _tail_by_tokens(text, max_tokens)


def _chunk_id(
    doc_id: str,
    sequence: int,
    text: str,
    source_chunk_ids: tuple[str, ...],
    chunk_cfg: dict[str, Any],
) -> tuple[str, str]:
    payload = json.dumps(
        {
            "doc_id": doc_id,
            "source_chunk_ids": source_chunk_ids,
            "strategy": chunk_cfg.get("strategy", "structure_aware"),
            "chunk_tokens": int(chunk_cfg.get("chunk_tokens", 384)),
            "overlap": int(chunk_cfg.get("overlap", 48)),
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{doc_id}_c{sequence:05d}_{digest[:12]}", digest


def _emit_chunk(
    *,
    doc_id: str,
    sequence: int,
    heading: str,
    heading_units: list[_Unit],
    pieces: list[_Piece],
    chunk_cfg: dict[str, Any],
) -> tuple[Chunk, dict[str, Any]]:
    texts = [heading] if heading else []
    texts.extend(piece.text for piece in pieces if piece.text)
    text = "\n\n".join(texts).strip()
    source_ids = _unique(
        [unit.chunk_id for unit in heading_units]
        + [source_id for piece in pieces for source_id in piece.source_chunk_ids]
    )
    page_ids = _unique(
        [page_id for unit in heading_units for page_id in unit.page_ids]
        + [page_id for piece in pieces for page_id in piece.page_ids]
    )
    labels = _unique(
        [unit.source_label for unit in heading_units]
        + [label for piece in pieces for label in piece.source_labels]
    )
    chunk_id, input_hash = _chunk_id(doc_id, sequence, text, source_ids, chunk_cfg)
    chunk = Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        page_ids=list(page_ids),
    )
    row = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "estimated_tokens": estimate_tokens(text),
        "has_overlap": any(piece.overlap for piece in pieces),
        "input_hash": input_hash,
        "page_ids": list(page_ids),
        "section_heading": heading,
        "source_chunk_ids": list(source_ids),
        "source_labels": list(labels),
        "strategy": str(chunk_cfg.get("strategy", "structure_aware")),
        "text": text,
    }
    return chunk, row


def _sections(
    units: list[_Unit], heading_labels: set[str]
) -> list[tuple[list[_Unit], list[_Unit]]]:
    sections: list[tuple[list[_Unit], list[_Unit]]] = []
    headings: list[_Unit] = []
    content: list[_Unit] = []

    def flush() -> None:
        nonlocal headings, content
        if headings or content:
            sections.append((headings, content))
        headings = []
        content = []

    for unit in units:
        is_heading = unit.kind == "heading" or unit.source_label in heading_labels
        if is_heading:
            if content:
                flush()
            headings.append(unit)
        else:
            content.append(unit)
    flush()
    return sections


def _split_document(
    units: list[_Unit],
    cfg: dict[str, Any],
    start_sequence: int,
) -> tuple[list[Chunk], list[dict[str, Any]], int]:
    chunk_cfg = cfg.get("chunk", {})
    chunk_tokens = max(32, int(chunk_cfg.get("chunk_tokens", 384)))
    overlap_tokens = max(0, int(chunk_cfg.get("overlap", 48)))
    heading_limit = max(0, int(chunk_cfg.get("heading_context_tokens", 64)))
    heading_labels = {
        str(value)
        for value in chunk_cfg.get("heading_labels", ["heading", "section_header", "title"])
    }
    atomic_labels = {
        str(value) for value in chunk_cfg.get("atomic_labels", ["formula", "table", "code"])
    }

    output: list[Chunk] = []
    rows: list[dict[str, Any]] = []
    sequence = start_sequence
    doc_id = units[0].doc_id

    for headings, content in _sections(units, heading_labels):
        heading = _heading_text(headings, heading_limit)
        heading_tokens = estimate_tokens(heading)
        content_limit = max(16, chunk_tokens - heading_tokens)
        pieces: list[_Piece] = []
        for unit in content:
            if unit.source_label in atomic_labels:
                pieces.append(_piece_for_atomic(unit))
            else:
                pieces.extend(_split_non_atomic(unit, content_limit))

        if not pieces and heading:
            heading_pieces = [
                _Piece(
                    text="",
                    source_chunk_ids=tuple(unit.chunk_id for unit in headings),
                    page_ids=_unique(page_id for unit in headings for page_id in unit.page_ids),
                    source_labels=_unique(unit.source_label for unit in headings),
                )
            ]
            chunk, row = _emit_chunk(
                doc_id=doc_id,
                sequence=sequence,
                heading=heading,
                heading_units=headings,
                pieces=heading_pieces,
                chunk_cfg=chunk_cfg,
            )
            output.append(chunk)
            rows.append(row)
            sequence += 1
            continue

        current: list[_Piece] = []
        current_tokens = 0
        last_new_pieces: list[_Piece] = []

        def emit_current(
            section_heading: str = heading,
            section_heading_units: tuple[_Unit, ...] = tuple(headings),
        ) -> None:
            nonlocal current, current_tokens, last_new_pieces, sequence
            if not current:
                return
            chunk, row = _emit_chunk(
                doc_id=doc_id,
                sequence=sequence,
                heading=section_heading,
                heading_units=list(section_heading_units),
                pieces=current,
                chunk_cfg=chunk_cfg,
            )
            output.append(chunk)
            rows.append(row)
            last_new_pieces = [piece for piece in current if not piece.overlap]
            sequence += 1
            current = []
            current_tokens = 0

        for piece in pieces:
            piece_tokens = estimate_tokens(piece.text)
            if piece.atomic and piece_tokens > content_limit:
                emit_current()
                current = [piece]
                current_tokens = piece_tokens
                emit_current()
                continue

            if not current:
                allowed_overlap = min(
                    overlap_tokens,
                    max(0, content_limit - piece_tokens),
                )
                current = _overlap_tail(last_new_pieces, allowed_overlap)
                current_tokens = sum(estimate_tokens(item.text) for item in current)

            if current and current_tokens + piece_tokens > content_limit:
                emit_current()
                allowed_overlap = min(
                    overlap_tokens,
                    max(0, content_limit - piece_tokens),
                )
                current = _overlap_tail(last_new_pieces, allowed_overlap)
                current_tokens = sum(estimate_tokens(item.text) for item in current)

            current.append(piece)
            current_tokens += piece_tokens

        emit_current()

    return output, rows, sequence


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Group OCR regions into traceable, heading-aware passages with overlap."""
    if not chunks:
        return []

    units = _units(chunks, cfg)
    if not units:
        return []

    output: list[Chunk] = []
    rows: list[dict[str, Any]] = []
    sequence_by_doc: dict[str, int] = {}

    current_doc = units[0].doc_id
    document_units: list[_Unit] = []

    def flush_document() -> None:
        nonlocal document_units
        if not document_units:
            return
        doc_id = document_units[0].doc_id
        doc_chunks, doc_rows, next_sequence = _split_document(
            document_units,
            cfg,
            sequence_by_doc.get(doc_id, 0),
        )
        output.extend(doc_chunks)
        rows.extend(doc_rows)
        sequence_by_doc[doc_id] = next_sequence
        document_units = []

    for unit in units:
        if unit.doc_id != current_doc:
            flush_document()
            current_doc = unit.doc_id
        document_units.append(unit)
    flush_document()

    manifest_path = Path(
        cfg.get("chunk", {}).get("manifest_path", "data/interim/chunks/chunks.jsonl")
    )
    _write_manifest(manifest_path, rows)
    LOGGER.info(
        "chunk_complete input_regions=%d output_chunks=%d strategy=%s manifest=%s",
        len(units),
        len(output),
        cfg.get("chunk", {}).get("strategy", "structure_aware"),
        manifest_path,
    )
    return output


def build_from_manifest(cfg: dict[str, Any]) -> list[Chunk]:
    """Build structure-aware chunks from the existing OCR manifest."""
    return split(load_ocr_chunks(cfg), cfg)


def load_chunks(cfg: dict[str, Any]) -> list[Chunk]:
    """Load the chunk manifest already written by ``split()``, without recomputing anything.

    Lets a later stage (e.g. scripts/run_index.py) rebuild embeddings/index from a prior
    ingest pass's output, instead of re-running layout/OCR/chunking to get there again."""
    path = Path(cfg.get("chunk", {}).get("manifest_path", "data/interim/chunks/chunks.jsonl"))
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk manifest does not exist: {path}. Run scripts/run_ingest.py first."
        )
    return [
        Chunk(
            id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            text=str(row["text"]),
            page_ids=[str(page_id) for page_id in row.get("page_ids", [])],
        )
        for row in _read_jsonl(path)
        if str(row.get("text", "")).strip()
    ]
