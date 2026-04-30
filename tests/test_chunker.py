"""Basic tests for deepbrain.chunker."""

import pytest

from deepbrain.chunker import chunk_code, chunk_document, chunk_markdown, chunk_text


def test_chunk_markdown_splits_by_header():
    md = "# Title\n\nSome top level overview text that is long enough.\n\n## Introduction\n\nHello world, this is the intro section with enough content to pass the minimum filter.\n\n## Details\n\nMore detail here with additional text so it passes the thirty character minimum length requirement."
    chunks = chunk_markdown(md)
    assert len(chunks) >= 2
    titles = [c["title"] for c in chunks]
    assert "Introduction" in titles
    assert "Details" in titles


def test_chunk_text_respects_max_chunk():
    # Two paragraphs that together exceed 50 chars should produce 2 chunks
    para_a = "A" * 40
    para_b = "B" * 40
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text, max_chunk=50)
    assert len(chunks) == 2
    assert chunks[0]["content"] == para_a
    assert chunks[1]["content"] == para_b


def test_chunk_document_routes_by_extension():
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    plain = "Just some plain text with enough characters to survive the filter."

    code_chunks = chunk_document(code, ".py")
    text_chunks = chunk_document(plain, ".txt")

    assert all(c["title"] in ("foo", "bar") for c in code_chunks)
    assert all(c["title"] == "" for c in text_chunks)
