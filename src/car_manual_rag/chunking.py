"""Section-aware chunk construction for cleaned document elements."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping

CHUNKING_STRATEGY = "section_aware_elements_v1"

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def build_section_aware_chunks(
    document_id: str,
    elements: Iterable[Mapping[str, object]],
    *,
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> list[dict[str, object]]:
    """Group adjacent body elements while carrying title and section context."""
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and less than max_chars")

    chunks: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    current_title: str | None = None
    current_section: str | None = None

    def prefix() -> str:
        parts = []
        if current_title:
            parts.append(f"Document title: {current_title}")
        if current_section:
            parts.append(f"Section: {current_section}")
        return "\n".join(parts)

    def render(items: list[dict[str, object]]) -> str:
        body = "\n\n".join(str(item["text"]) for item in items)
        context = prefix()
        return f"{context}\n\n{body}" if context else body

    def append_chunk(items: list[dict[str, object]]) -> None:
        if not items:
            return
        chunk_text = render(items).strip()
        if not chunk_text:
            return

        element_ids = sorted({int(item["element_id"]) for item in items})
        pdf_page_numbers = sorted(
            {
                int(page)
                for item in items
                for page in (item.get("pdf_page_numbers") or [])
            }
        )
        chunk_index = len(chunks)
        digest = hashlib.sha256(
            f"{document_id}|{chunk_index}|{chunk_text}".encode("utf-8")
        ).hexdigest()[:20]

        chunks.append(
            {
                "chunk_id": f"{document_id}_{digest}",
                "chunk_index": chunk_index,
                "section_title": current_section,
                "document_title": current_title,
                "start_element_id": min(element_ids),
                "end_element_id": max(element_ids),
                "element_ids": element_ids,
                "pdf_page_numbers": pdf_page_numbers,
                "chunk_text": chunk_text,
                "character_count": len(chunk_text),
                "estimated_token_count": max(1, math.ceil(len(chunk_text) / 4)),
                "chunking_strategy": CHUNKING_STRATEGY,
            }
        )

    def overlap_tail(items: list[dict[str, object]]) -> list[dict[str, object]]:
        carried: list[dict[str, object]] = []
        carried_chars = 0
        for item in reversed(items):
            item_chars = len(str(item["text"]))
            if carried_chars + item_chars > overlap_chars:
                break
            carried.insert(0, item)
            carried_chars += item_chars
        return carried

    for raw_element in elements:
        element_type = str(raw_element.get("element_type") or "")
        content = _element_text(element_type, raw_element.get("cleaned_content"))
        if not content:
            continue

        if element_type == "title":
            append_chunk(buffer)
            buffer = []
            if current_title is None:
                current_title = content
            else:
                current_section = content
            continue

        if element_type == "section_header":
            append_chunk(buffer)
            buffer = []
            current_section = content
            continue

        element = {
            "element_id": int(raw_element["element_id"]),
            "pdf_page_numbers": list(raw_element.get("pdf_page_numbers") or []),
            "text": content,
        }

        available_chars = max(200, max_chars - len(prefix()) - 2)
        if len(content) > available_chars:
            append_chunk(buffer)
            buffer = []
            for segment in split_text(
                content,
                max_chars=available_chars,
                overlap_chars=min(overlap_chars, available_chars // 4),
            ):
                split_element = dict(element)
                split_element["text"] = segment
                append_chunk([split_element])
            continue

        candidate = buffer + [element]
        if buffer and len(render(candidate)) > max_chars:
            append_chunk(buffer)
            buffer = overlap_tail(buffer)
            candidate = buffer + [element]
            if buffer and len(render(candidate)) > max_chars:
                buffer = []

        buffer.append(element)

    append_chunk(buffer)
    return chunks


def split_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split long text near whitespace with deterministic character overlap."""
    if len(text) <= max_chars:
        return [text]

    segments = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary

        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap_chars)
        while next_start < text_length and text[next_start].isspace():
            next_start += 1
        start = next_start

    return segments


def _element_text(element_type: str, content: object) -> str | None:
    if content is None:
        return None
    text = str(content)
    if element_type == "table":
        text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None
