"""Smart document chunking — split documents into meaningful pieces."""

from __future__ import annotations

import re
from typing import Final, TypedDict

__all__ = ["Chunk", "chunk_markdown", "chunk_text", "chunk_code", "chunk_document"]

# Chunks shorter than this many characters are discarded.
_MIN_CONTENT_LEN: Final[int] = 30

_HEADER_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"\n(#{1,4}\s+.+)")
_HEADER_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^#{1,4}\s+")
_DEF_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"\n(?=(?:def |class |async def ))"
)
_DEF_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:async )?(?:def|class)\s+(\w+)"
)

#: File extensions routed to :func:`chunk_code`.
_CODE_EXTS: Final[frozenset[str]] = frozenset(
    {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".rb"}
)
#: File extensions routed to :func:`chunk_markdown`.
_MARKUP_EXTS: Final[frozenset[str]] = frozenset({".md", ".rst"})


class Chunk(TypedDict):
    """A contiguous block of text extracted from a document.

    Attributes:
        title: Nearest section heading or top-level symbol name, stripped of
            leading ``#`` characters.  Empty string when no heading is in scope.
        content: Trimmed text body.  Always at least :data:`_MIN_CONTENT_LEN`
            characters long after filtering.
    """

    title: str
    content: str


def _split_paragraphs(text: str) -> list[str]:
    """Return non-empty, stripped paragraphs from *text* split on ``\\n\\n``."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _pack_paragraphs(
    paragraphs: list[str],
    title: str,
    max_chunk: int,
) -> list[Chunk]:
    """Greedily pack *paragraphs* into :class:`Chunk` objects under *max_chunk* chars.

    A new chunk is started whenever the next paragraph would push the running
    buffer past *max_chunk*.  A paragraph larger than *max_chunk* on its own is
    placed in a chunk by itself (no hard split inside a paragraph).

    Args:
        paragraphs: Non-empty strings to pack.
        title: Heading label applied to every resulting chunk.
        max_chunk: Soft character limit per chunk.

    Returns:
        List of :class:`Chunk` objects.  May be empty if *paragraphs* is empty.
    """
    chunks: list[Chunk] = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > max_chunk:
            chunks.append({"title": title, "content": buffer.strip()})
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer:
        chunks.append({"title": title, "content": buffer.strip()})
    return chunks


def chunk_markdown(text: str, max_chunk: int = 1500) -> list[Chunk]:
    """Split Markdown into chunks delimited by ATX headings (``#``–``####``).

    Each heading starts a new chunk.  If a section's body exceeds *max_chunk*
    characters it is subdivided greedily at paragraph (``\\n\\n``) boundaries.
    Content that is too large to subdivide further is kept as a single
    oversized chunk rather than split mid-paragraph.

    When the document contains no headings the entire text is treated as a
    single section with an empty title and split by paragraph.  If that still
    produces no chunks (e.g. the text has no double-newlines) a hard fallback
    slices the raw text into *max_chunk*-character windows.

    Args:
        text: Raw Markdown source.
        max_chunk: Soft character limit per chunk.  Defaults to ``1500``.

    Returns:
        Ordered list of :class:`Chunk` objects.  Chunks whose ``content`` is
        shorter than 30 characters are omitted.
    """
    sections = _HEADER_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    current_title = ""

    i = 0
    while i < len(sections):
        part = sections[i].strip()
        if _HEADER_PREFIX_RE.match(part):
            current_title = _HEADER_PREFIX_RE.sub("", part).strip()
            i += 1
            continue
        if part:
            chunks.extend(
                _pack_paragraphs(_split_paragraphs(part), current_title, max_chunk)
            )
        i += 1

    if not chunks and text.strip():
        for i in range(0, len(text), max_chunk):
            chunks.append({"title": "", "content": text[i : i + max_chunk].strip()})

    return [c for c in chunks if len(c["content"]) >= _MIN_CONTENT_LEN]


def chunk_text(text: str, max_chunk: int = 1500) -> list[Chunk]:
    """Split plain text into chunks by double-newline paragraph boundaries.

    Paragraphs are packed greedily: a new chunk is opened when the next
    paragraph would push the current buffer past *max_chunk*.  If no paragraph
    boundaries exist a hard fallback slices the raw text into *max_chunk*-
    character windows.

    Args:
        text: Plain text to split.
        max_chunk: Soft character limit per chunk.  Defaults to ``1500``.

    Returns:
        Ordered list of :class:`Chunk` objects with empty ``title`` fields.
        Chunks whose ``content`` is shorter than 30 characters are omitted.
    """
    chunks = _pack_paragraphs(_split_paragraphs(text), "", max_chunk)

    if not chunks and text.strip():
        for i in range(0, len(text), max_chunk):
            chunks.append({"title": "", "content": text[i : i + max_chunk].strip()})

    return [c for c in chunks if len(c["content"]) >= _MIN_CONTENT_LEN]


def chunk_code(text: str, max_chunk: int = 2000) -> list[Chunk]:
    """Split source code at top-level ``def``, ``async def``, and ``class`` boundaries.

    Each top-level definition becomes its own chunk.  If a definition exceeds
    *max_chunk* characters it is hard-split at byte boundaries as a fallback.
    Preamble code that appears before the first definition is captured as a
    chunk with an empty title.  If no definitions are found, :func:`chunk_text`
    is used as a fallback.

    Args:
        text: Source code text (any language whose top-level definitions start
            with ``def``, ``async def``, or ``class``).
        max_chunk: Hard character limit per chunk.  Defaults to ``2000``.

    Returns:
        Ordered list of :class:`Chunk` objects.  The ``title`` of each chunk is
        the bare function or class name, or an empty string for preamble blocks.
        Chunks whose ``content`` is shorter than 30 characters are omitted.
    """
    parts = _DEF_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _DEF_NAME_RE.match(part)
        title = m.group(1) if m else ""
        if len(part) > max_chunk:
            for i in range(0, len(part), max_chunk):
                chunks.append({"title": title, "content": part[i : i + max_chunk]})
        else:
            chunks.append({"title": title, "content": part})

    if not chunks and text.strip():
        chunks = chunk_text(text, max_chunk)

    return [c for c in chunks if len(c["content"]) >= _MIN_CONTENT_LEN]


def chunk_document(text: str, ext: str, max_chunk: int = 1500) -> list[Chunk]:
    """Route *text* to the appropriate chunker based on file extension *ext*.

    Dispatch table:

    * ``'.md'``, ``'.rst'`` → :func:`chunk_markdown`
    * ``'.py'``, ``'.js'``, ``'.ts'``, ``'.java'``, ``'.go'``, ``'.rs'``,
      ``'.c'``, ``'.cpp'``, ``'.rb'`` → :func:`chunk_code`
    * anything else → :func:`chunk_text`

    Args:
        text: Document text to chunk.
        ext: File extension including the leading dot, e.g. ``'.md'`` or
            ``'.py'``.  Case-sensitive.
        max_chunk: Soft character limit passed through to the selected chunker.
            Defaults to ``1500``; :func:`chunk_code` uses ``2000`` by default
            but respects this override when called via :func:`chunk_document`.

    Returns:
        Ordered list of :class:`Chunk` objects produced by the dispatched
        chunker.
    """
    if ext in _MARKUP_EXTS:
        return chunk_markdown(text, max_chunk)
    if ext in _CODE_EXTS:
        return chunk_code(text, max_chunk)
    return chunk_text(text, max_chunk)
