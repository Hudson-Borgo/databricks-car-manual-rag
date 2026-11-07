"""Deterministic cleaning rules for parsed car-manual elements."""

from __future__ import annotations

import html
import re
import unicodedata

CLEANING_VERSION = "v1"

_LINE_BREAK_HYPHENATION = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
_SLASH_LINE_BREAK = re.compile(r"(?<=/)\s*\n\s*(?=\w)")
_WHITESPACE = re.compile(r"\s+")

_CONTENT_ROLES = {
    "page_header": "page_furniture",
    "page_footer": "page_furniture",
    "page_number": "page_furniture",
    "title": "structure",
    "section_header": "structure",
    "text": "body",
    "table": "body",
    "caption": "body",
    "footnote": "body",
    "figure": "visual",
    "signature": "visual",
}

_TEXT_RETRIEVAL_ROLES = {"structure", "body"}


def clean_extracted_text(value: str | None) -> str | None:
    """Normalize parser text without changing the preserved raw value.

    The rules decode HTML entities, normalize Unicode, remove discretionary
    hyphenation at line breaks, repair slash-separated paths split across lines,
    and collapse remaining whitespace. Inline hyphens are preserved.
    """
    if value is None:
        return None

    cleaned = html.unescape(value)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.replace("\u00ad", "").replace("\u200b", "")
    cleaned = _LINE_BREAK_HYPHENATION.sub("", cleaned)
    cleaned = _SLASH_LINE_BREAK.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    return cleaned or None


def classify_content_role(element_type: str | None) -> str:
    """Map a parser element type to its downstream content role."""
    if element_type is None:
        return "other"
    return _CONTENT_ROLES.get(element_type, "other")


def is_text_retrieval_eligible(
    element_type: str | None,
    cleaned_content: str | None,
) -> bool:
    """Return whether an element can contribute to text retrieval chunks."""
    role = classify_content_role(element_type)
    return role in _TEXT_RETRIEVAL_ROLES and bool(cleaned_content)
