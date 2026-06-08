"""Text chunking and snippet generation."""

from __future__ import annotations

import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize line endings and collapse excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[tuple[int, int, str]]:
    """Split text into overlapping chunks, breaking on natural boundaries.

    Returns list of (start_pos, end_pos, chunk_text).
    """
    text = clean_text(text)
    if not text:
        return []
    chunks: list[tuple[int, int, str]] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            split = max(candidates)
            if split > start + chunk_size // 2:
                end = split + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= text_len:
            break
        next_start = max(end - overlap, 0)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def make_snippet(text: str, limit: int = 300) -> str:
    """Collapse whitespace and truncate to *limit* chars."""
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def make_embed_input(rel_path: str, chunk: str) -> str:
    """Build the text sent to the embedding model for one chunk."""
    filename = Path(rel_path).name
    return f"File: {rel_path}\nName: {filename}\n\n{chunk}"
