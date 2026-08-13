"""Stage 4 — chunk text"""

from __future__ import annotations

import re

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Display-math delimiters our OCR (Nougat, markdown+LaTeX output) actually emits. A chunk boundary
# landing inside one of these would split a formula across two chunks and break retrieval/citation
# on it (A1's own "math-aware chunking" commitment: chunk at theorem/problem boundaries, don't
# split formulas). _MATH_TOKEN_RE finds delimiter tokens so split() can track open/closed state per
# word and push a boundary forward past any word where we're still inside a formula.
_MATH_TOKEN_RE = re.compile(r"\\\[|\\\]|\$\$")


def _word_in_math_delta(word: str) -> int:
    """Net open(+1)/close(-1) contribution of one word's math delimiters, so callers can track a
    running in-math depth across a word sequence without look-ahead."""
    delta = 0
    for tok in _MATH_TOKEN_RE.findall(word):
        if tok == r"\[":
            delta += 1
        elif tok == r"\]":
            delta -= 1
        else:  # "$$" toggles a display-math block; approximate as alternating open/close
            delta += 1 if delta >= 0 else -1
    return delta


def _flat_words(doc_chunks: list[Chunk]) -> tuple[list[str], list[str]]:
    """One doc's raw OCR chunks (already page/region ordered by ocr.transcribe) -> a flat
    (word, source_page_id) sequence."""
    words: list[str] = []
    page_ids: list[str] = []
    for c in doc_chunks:
        page_id = c.page_ids[0] if c.page_ids else ""
        for w in c.text.split():
            words.append(w)
            page_ids.append(page_id)
    return words, page_ids


def _token_counts(words: list[str], tokenizer_name: str, revision: str | None) -> list[int]:
    """Per-word subword token count, via the SAME tokenizer embed.encode() will use, so chunk sizes
    are measured in the unit that actually matters (embedding-model tokens, not raw words).
    revision: pinned HF commit SHA (configs/config.yaml: embed.revision) -- avoids an unpinned
    from_pretrained() download (bandit B615)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)
    if not words:
        return []
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    counts = [0] * len(words)
    for wid in word_ids:
        if wid is not None:
            counts[wid] += 1
    return [max(1, c) for c in counts]  # every word costs >=1 token (avoids zero-width words)


def _windows(
    n_words: int, tok_counts: list[int], chunk_tokens: int, overlap: int
) -> list[tuple[int, int]]:
    """Word-index [start, end) windows whose token budget is ~chunk_tokens, advancing by
    (chunk_tokens - overlap) tokens each step."""
    stride = max(1, chunk_tokens - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_words:
        budget = 0
        end = start
        while end < n_words and budget < chunk_tokens:
            budget += tok_counts[end]
            end += 1
        spans.append((start, end))
        if end >= n_words:
            break
        # advance start by `stride` tokens' worth of words, not `stride` words themselves
        advanced = 0
        new_start = start
        while new_start < end and advanced < stride:
            advanced += tok_counts[new_start]
            new_start += 1
        start = max(new_start, start + 1)
    return spans


def _extend_past_open_math(words: list[str], end: int) -> int:
    """If the window [.., end) ends mid-formula, push `end` forward to the word that closes it
    (or to the end of the doc if it never closes -- a mis-detected delimiter shouldn't hang)."""
    depth = sum(_word_in_math_delta(w) for w in words[:end])
    i = end
    while depth > 0 and i < len(words):
        depth += _word_in_math_delta(words[i])
        i += 1
    return i


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk the raw per-region OCR chunks to cfg['index']['chunk_tokens']/'overlap', grouped by
    document, with a boundary guard so a window is never cut inside an open LaTeX display-math
    block."""
    chunk_tokens = cfg["index"]["chunk_tokens"]
    overlap = cfg["index"]["overlap"]
    tokenizer_name = cfg["embed"]["model"]
    tokenizer_revision = cfg["embed"].get("revision")

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    out: list[Chunk] = []
    for doc_id, doc_chunks in by_doc.items():
        words, page_ids = _flat_words(doc_chunks)
        if not words:
            continue
        tok_counts = _token_counts(words, tokenizer_name, tokenizer_revision)
        spans = _windows(len(words), tok_counts, chunk_tokens, overlap)
        for i, (start, end) in enumerate(spans):
            end = min(len(words), _extend_past_open_math(words, end))
            if end <= start:
                continue
            text = " ".join(words[start:end])
            pages_in_window = sorted(dict.fromkeys(p for p in page_ids[start:end] if p))
            out.append(
                Chunk(
                    id=f"{doc_id}_c{i:04d}",
                    doc_id=doc_id,
                    text=text,
                    page_ids=pages_in_window,
                )
            )
    logger.info(f"chunk.split: {len(chunks)} raw region-chunks -> {len(out)} final chunks")
    return out
