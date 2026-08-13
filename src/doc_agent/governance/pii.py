"""Governance — PII detection + redaction (mandatory)"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ID_NUMBER_RE = re.compile(r"\b\d{6}[-\s]?\d{4}[-\s]?\d{3}\b")  # e.g. SA 13-digit ID numbers
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")

# Order matters: match the more specific/structured patterns (email, ID number) before the loose
# phone-number pattern, which would otherwise also match parts of them.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL_RE, "email"),
    (_ID_NUMBER_RE, "id_number"),
    (_PHONE_RE, "phone"),
]


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans. Our corpus is a math textbook -- A1's own risk analysis
    found PII risk 'low but not zero' (invented word-problem names like 'Thabo buys 3 apples', real
    author names confined to front matter, which is excluded from the retrievable corpus entirely).
    This baseline covers structured, high-confidence PII (emails, ID numbers, phone numbers). It
    deliberately does not flag ordinary personal names in isolation: that would flag fictional
    word-problem names as false positives, with no reliable way to tell them apart from a real
    person without much heavier NER machinery than an ingest-time regex pass warrants."""
    spans: set[tuple[int, int, str]] = set()
    claimed: list[tuple[int, int]] = []
    for pattern, kind in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in claimed):
                continue  # already covered by a more specific pattern
            spans.add((m.start(), m.end(), kind))
            claimed.append((m.start(), m.end()))
    return sorted(spans)


def redact(text: str) -> str:
    """Replace each detected PII span with a typed placeholder, right-to-left so earlier spans'
    character offsets stay valid as later ones are substituted."""
    for start, end, kind in sorted(detect(text), reverse=True):
        text = text[:start] + f"[REDACTED:{kind.upper()}]" + text[end:]
    return text


def _scrub_ctx(ctx: dict) -> dict:
    """Redact whatever text this seam's ctx carries. Duck-typed across the three seams it's wired
    to (AFTER_OCR gives {'chunks': list[Chunk]}, BEFORE_ANSWER/ON_LOG shapes are A3 territory) so
    this handles all of them without erroring on a shape it doesn't recognise."""
    if "chunks" in ctx:
        for c in ctx["chunks"]:
            if hasattr(c, "text"):
                c.text = redact(c.text)
    answer = ctx.get("answer")
    if answer is not None and hasattr(answer, "text"):
        answer.text = redact(answer.text)
    if isinstance(ctx.get("message"), str):
        ctx["message"] = redact(ctx["message"])
    return ctx


def register(hooks: Any) -> None:
    """Wire PII redaction into the pipeline: scrub extracted text before indexing, the outgoing
    answer, and logs -- mandatory per STRUCTURE.md regardless of primary NFR."""
    hooks.register(hooks.AFTER_OCR, _scrub_ctx)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub_ctx)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub_ctx)  # scrub logs
