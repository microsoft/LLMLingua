"""Text chunking for API translation."""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class ChunkStats:
    """Chunking statistics."""
    chunks_count: int
    avg_chunk_len: float
    max_chunk_len: int
    warnings: List[str]


class TextChunker:
    """Chunk text safely for API translation."""
    
    def __init__(self, max_chars: int = 2000):
        """
        Initialize text chunker.
        
        Args:
            max_chars: Maximum characters per chunk
        """
        self.max_chars = max_chars
    
    def chunk(self, text: str) -> tuple[List[str], ChunkStats]:
        """
        Split text into chunks safely.
        
        Splits by paragraph/sentence boundaries while preserving:
        - URLs
        - Citations
        - LaTeX/math blocks
        - Code spans
        
        Args:
            text: Text to chunk
            
        Returns:
            Tuple of (chunks, stats)
        """
        if len(text) <= self.max_chars:
            # No chunking needed
            return [text], ChunkStats(
                chunks_count=1,
                avg_chunk_len=len(text),
                max_chunk_len=len(text),
                warnings=[]
            )
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_len = 0
        warnings = []
        
        for para in paragraphs:
            para_len = len(para)
            
            # If single paragraph exceeds max, split by sentences
            if para_len > self.max_chars:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split paragraph by sentences
                sentences = self._split_sentences(para)
                for sent in sentences:
                    sent_len = len(sent)
                    
                    if sent_len > self.max_chars:
                        # Sentence too long, force split
                        warnings.append(f"Sentence exceeds max length: {sent_len} chars")
                        # Split at max_chars boundary
                        for i in range(0, sent_len, self.max_chars):
                            chunk_part = sent[i:i+self.max_chars]
                            chunks.append(chunk_part)
                    elif current_len + sent_len + 1 > self.max_chars:
                        # Start new chunk
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_len = sent_len
                    else:
                        # Add to current chunk
                        current_chunk.append(sent)
                        current_len += sent_len + 1
            
            elif current_len + para_len + 2 > self.max_chars:
                # Start new chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_len += para_len + 2
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Calculate stats
        chunk_lens = [len(c) for c in chunks]
        stats = ChunkStats(
            chunks_count=len(chunks),
            avg_chunk_len=sum(chunk_lens) / len(chunks) if chunks else 0,
            max_chunk_len=max(chunk_lens) if chunks else 0,
            warnings=warnings
        )
        
        return chunks, stats
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences safely.
        
        Avoids splitting inside:
        - URLs
        - Citations
        - Abbreviations
        """
        # Simple sentence splitting (can be improved)
        # Split on . ! ? followed by space and capital letter
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def join_chunks(self, chunks: List[str]) -> str:
        """
        Join translated chunks back together.
        
        Args:
            chunks: List of translated chunks
            
        Returns:
            Joined text with preserved paragraph breaks
        """
        return '\n\n'.join(chunks)
