"""Conservative text normalization for loaded document segments."""

from __future__ import annotations

import re


_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_SPACES_AROUND_NEWLINES = re.compile(r" *\n *")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)
    normalized = _SPACES_AROUND_NEWLINES.sub("\n", normalized)
    normalized = _EXCESS_NEWLINES.sub("\n\n", normalized)
    return normalized.strip()
