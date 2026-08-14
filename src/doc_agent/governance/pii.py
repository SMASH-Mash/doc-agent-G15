"""Governance — PII detection + redaction (mandatory)."""

from __future__ import annotations

import re

from ..contracts import *  # noqa: F401,F403

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# Conservative phone-number pattern. Requires separators so that ordinary
# mathematical numbers, years, page numbers, etc. are not treated as PII.
_PHONE_RE = re.compile(
    r"(?<!\d)" r"(?:\+?\d{1,3}[-.\s])?" r"(?:\(?\d{2,4}\)?[-.\s])" r"\d{3,4}[-.\s]\d{3,4}" r"(?!\d)"
)


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) spans for high-confidence PII."""
    spans: list[tuple[int, int, str]] = []

    for match in _EMAIL_RE.finditer(text):
        spans.append((match.start(), match.end(), "email"))

    for match in _SSN_RE.finditer(text):
        spans.append((match.start(), match.end(), "ssn"))

    for match in _PHONE_RE.finditer(text):
        spans.append((match.start(), match.end(), "phone"))

    # Keep deterministic ordering and remove overlaps.
    spans.sort(key=lambda item: (item[0], item[1], item[2]))

    result: list[tuple[int, int, str]] = []
    last_end = -1

    for start, end, pii_type in spans:
        if start >= last_end:
            result.append((start, end, pii_type))
            last_end = end

    return result


def redact(text: str) -> str:
    """Replace detected PII spans with explicit redaction markers."""
    spans = detect(text)

    if not spans:
        return text

    replacements = {
        "email": "[REDACTED_EMAIL]",
        "ssn": "[REDACTED_SSN]",
        "phone": "[REDACTED_PHONE]",
    }

    # Replace from right to left so original offsets remain valid.
    result = text

    for start, end, pii_type in reversed(spans):
        result = result[:start] + replacements[pii_type] + result[end:]

    return result


def register(hooks) -> None:
    """Wire PII redaction into the pipeline hooks."""

    def _scrub(ctx: dict) -> dict:
        if not isinstance(ctx, dict):
            return ctx

        # Common scalar text fields.
        for key in ("text", "answer", "log"):
            value = ctx.get(key)

            if isinstance(value, str):
                ctx[key] = redact(value)

            elif isinstance(value, list):
                ctx[key] = [redact(item) if isinstance(item, str) else item for item in value]

        # OCR/chunk collections.
        chunks = ctx.get("chunks")

        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    value = chunk.get("text")
                    if isinstance(value, str):
                        chunk["text"] = redact(value)

                elif hasattr(chunk, "text") and isinstance(chunk.text, str):
                    chunk.text = redact(chunk.text)

        hooks_context = ctx.get("context")
        if isinstance(hooks_context, str):
            ctx["context"] = redact(hooks_context)

        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)
    hooks.register(hooks.BEFORE_ANSWER, _scrub)
    hooks.register(hooks.ON_LOG, _scrub)
