"""PDF parsing with PyMuPDF - extract text, tables, images."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class SpanData:
    """Text span with formatting."""
    text: str
    font: str = ""
    size: float = 12.0
    flags: int = 0
    color: int = 0


@dataclass
class LineData:
    """Text line with spans."""
    spans: List[SpanData] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    
    @property
    def text(self) -> str:
        """Get combined text from all spans."""
        return "".join(span.text for span in self.spans)


@dataclass
class ContentBlock:
    """Content block (text, table, image)."""
    type: str  # 'text', 'table', 'image'
    content: any
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    page_num: int = 0


@dataclass
class PageData:
    """PDF page data."""
    page_num: int
    width: float
    height: float
    blocks: List[ContentBlock] = field(default_factory=list)
    lines: List[LineData] = field(default_factory=list)  # For backward compatibility


@dataclass
class PDFData:
    """Complete PDF document data."""
    pages: List[PageData] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """Parse PDF documents."""
    
    def __init__(self):
        """Initialize PDF parser."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF parsing. Install with: pip install PyMuPDF")
    
    def parse(self, pdf_path: Path) -> PDFData:
        """
        Parse PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFData with extracted content
        """
        doc = fitz.open(pdf_path)
        pdf_data = PDFData(metadata=doc.metadata)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = self._parse_page(page, page_num)
            pdf_data.pages.append(page_data)
        
        doc.close()
        return pdf_data
    
    def _parse_page(self, page, page_num: int) -> PageData:
        """Parse a single PDF page."""
        page_data = PageData(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height
        )
        
        # Extract text blocks
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # Text block
                content_block = self._parse_text_block(block, page_num)
                page_data.blocks.append(content_block)
                # Also populate lines for backward compatibility
                for line in block.get("lines", []):
                    line_data = LineData(
                        spans=[SpanData(
                            text=span.get("text", ""),
                            font=span.get("font", ""),
                            size=span.get("size", 12.0),
                            flags=span.get("flags", 0),
                            color=span.get("color", 0)
                        ) for span in line.get("spans", [])],
                        bbox=tuple(line.get("bbox", (0, 0, 0, 0)))
                    )
                    page_data.lines.append(line_data)
        
        return page_data
    
    def _parse_text_block(self, block: dict, page_num: int) -> ContentBlock:
        """Parse a text block."""
        lines = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            lines.append(line_text)
        
        content = "\n".join(lines)
        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        
        return ContentBlock(
            type="text",
            content=content,
            bbox=bbox,
            page_num=page_num
        )
