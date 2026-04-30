"""Smart document chunking — split documents into meaningful pieces."""

from __future__ import annotations

import re
from typing import Generator


def chunk_markdown(text: str, max_chunk: int = 1500) -> list[dict]:
    """Split markdown by headers, then by paragraphs if too long."""
    sections = re.split(r'\n(#{1,4}\s+.+)', text)
    chunks = []
    current_title = ""
    
    i = 0
    while i < len(sections):
        part = sections[i].strip()
        if re.match(r'^#{1,4}\s+', part):
            current_title = part.lstrip('#').strip()
            i += 1
            continue
        if not part:
            i += 1
            continue
        # Split long sections by paragraph
        paragraphs = [p.strip() for p in part.split('\n\n') if p.strip()]
        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) > max_chunk and buffer:
                chunks.append({"title": current_title, "content": buffer.strip()})
                buffer = para
            else:
                buffer = f"{buffer}\n\n{para}" if buffer else para
        if buffer:
            chunks.append({"title": current_title, "content": buffer.strip()})
        i += 1

    # If no chunks produced, treat whole text as one chunk
    if not chunks and text.strip():
        for i in range(0, len(text), max_chunk):
            chunks.append({"title": "", "content": text[i:i+max_chunk].strip()})
    return [c for c in chunks if len(c["content"]) >= 30]


def chunk_text(text: str, max_chunk: int = 1500) -> list[dict]:
    """Split plain text by paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) > max_chunk and buffer:
            chunks.append({"title": "", "content": buffer.strip()})
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer:
        chunks.append({"title": "", "content": buffer.strip()})
    if not chunks and text.strip():
        for i in range(0, len(text), max_chunk):
            chunks.append({"title": "", "content": text[i:i+max_chunk].strip()})
    return [c for c in chunks if len(c["content"]) >= 30]


def chunk_code(text: str, max_chunk: int = 2000) -> list[dict]:
    """Split code by function/class definitions."""
    # Try to split by def/class
    parts = re.split(r'\n(?=(?:def |class |async def ))', text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract function/class name
        m = re.match(r'(?:async )?(?:def|class)\s+(\w+)', part)
        title = m.group(1) if m else ""
        if len(part) > max_chunk:
            # Split long functions
            for i in range(0, len(part), max_chunk):
                chunks.append({"title": title, "content": part[i:i+max_chunk]})
        else:
            chunks.append({"title": title, "content": part})
    if not chunks and text.strip():
        chunks = chunk_text(text, max_chunk)
    return [c for c in chunks if len(c["content"]) >= 30]


def chunk_document(text: str, ext: str, max_chunk: int = 1500) -> list[dict]:
    """Route to appropriate chunker based on file extension."""
    if ext in ('.md', '.rst'):
        return chunk_markdown(text, max_chunk)
    elif ext in ('.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp', '.rb'):
        return chunk_code(text, max_chunk)
    else:
        return chunk_text(text, max_chunk)
