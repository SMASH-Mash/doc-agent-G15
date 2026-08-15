"""Per-task verifier. FIXED signature."""

from __future__ import annotations


def check(task: dict, answer: dict) -> bool:
    """Return True if `answer` satisfies `task`.

    Exact for fact tasks; judge for open. IMPLEMENT.
    """
    raise NotImplementedError("verify a task result")
