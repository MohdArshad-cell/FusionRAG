import pytest
from app.core.chunker import RecursiveCharacterChunker


def test_chunker_basic():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.chunk(text, doc_id="test_doc", source_file="test.txt")

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == "test_doc"
        assert chunk.source_file == "test.txt"
        assert len(chunk.text) <= 50 + 10  # allow reasonable margin
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start


def test_chunker_empty_string():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk("", doc_id="empty_doc")
    assert len(chunks) == 0


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=60)


def test_chunker_long_word_fallback():
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    long_text = "A" * 100
    chunks = chunker.chunk(long_text, doc_id="long_doc")
    assert len(chunks) >= 3

