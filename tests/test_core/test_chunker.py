"""Test text chunking."""

import pytest
from src.core.chunker import TextChunker


def test_chunker_no_split_needed():
    """Test that short text is not chunked."""
    chunker = TextChunker(max_chars=100)
    text = "Short text"
    
    chunks, stats = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert chunks[0] == text
    assert stats.chunks_count == 1


def test_chunker_split_by_paragraphs():
    """Test chunking by paragraphs."""
    chunker = TextChunker(max_chars=50)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    
    chunks, stats = chunker.chunk(text)
    
    assert len(chunks) > 1
    assert stats.chunks_count == len(chunks)


def test_chunker_preserves_structure():
    """Test that chunking preserves paragraph structure."""
    chunker = TextChunker(max_chars=100)
    text = "Para 1.\n\nPara 2.\n\nPara 3."
    
    chunks, stats = chunker.chunk(text)
    joined = chunker.join_chunks(chunks)
    
    # Should preserve double newlines
    assert "\n\n" in joined or len(chunks) == 1


def test_chunker_stats():
    """Test chunking statistics."""
    chunker = TextChunker(max_chars=50)
    text = "A" * 150  # Long text
    
    chunks, stats = chunker.chunk(text)
    
    assert stats.chunks_count > 1
    assert stats.max_chunk_len <= 50
    assert stats.avg_chunk_len > 0


def test_chunker_join():
    """Test joining chunks."""
    chunker = TextChunker(max_chars=50)
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    
    joined = chunker.join_chunks(chunks)
    
    assert "Chunk 1" in joined
    assert "Chunk 2" in joined
    assert "Chunk 3" in joined
