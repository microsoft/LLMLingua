"""PDF smoke and integration tests."""

import pytest
import tempfile
from pathlib import Path

from src.formats.pdf.parser import PDFData, PageData, ContentBlock, LineData, SpanData


def test_pdf_data_structures_importable():
    """Test that all PDF data structures can be imported."""
    # This test ensures backward compatibility
    assert PDFData is not None
    assert PageData is not None
    assert ContentBlock is not None
    assert LineData is not None
    assert SpanData is not None


def test_pdf_data_creation():
    """Test creating PDF data structures."""
    # Create a simple PDF data structure
    span = SpanData(text="Hello", font="Arial", size=12.0)
    line = LineData(spans=[span], bbox=(0, 0, 100, 20))
    
    assert line.text == "Hello"
    assert len(line.spans) == 1
    
    block = ContentBlock(
        type="text",
        content="Hello world",
        bbox=(0, 0, 100, 50),
        page_num=0
    )
    
    assert block.type == "text"
    assert block.content == "Hello world"
    
    page = PageData(
        page_num=0,
        width=612,
        height=792,
        blocks=[block],
        lines=[line]
    )
    
    assert page.page_num == 0
    assert len(page.blocks) == 1
    assert len(page.lines) == 1
    
    pdf_data = PDFData(pages=[page])
    assert len(pdf_data.pages) == 1


def test_qa_report_includes_tables_and_images():
    """Test that QA report includes tables and images sections."""
    from src.core.qa_report import QAReport
    
    report = QAReport(
        input_file="test.pdf",
        output_file="test_out.pdf",
        format="pdf",
        direction="en_to_ar",
        mode="bilingual"
    )
    
    # Check that tables section exists
    assert "tables" in report.to_dict()
    assert "detected" in report.tables
    assert "translated" in report.tables
    assert "method" in report.tables
    assert "warnings" in report.tables
    
    # Check that images section exists
    assert "images" in report.to_dict()
    assert "detected" in report.images
    assert "captions_added" in report.images
    assert "resized_count" in report.images
    assert "warnings" in report.images
    
    # Check other required sections
    assert "chunking" in report.to_dict()
    assert "cache" in report.to_dict()
    assert "glossary" in report.to_dict()
    assert "retries" in report.to_dict()
    assert "warnings" in report.to_dict()
    assert "fallbacks_used" in report.to_dict()
    assert "conversion_warnings" in report.to_dict()
