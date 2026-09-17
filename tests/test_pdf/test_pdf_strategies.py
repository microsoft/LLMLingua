"""Test PDF translation strategies."""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from src.core.translator import TranslationDirection, TranslationMode
from src.core.translators.mock_translator import MockTranslator
from src.formats.pdf.tables import TableData, TableCell, TableTranslator
from src.formats.pdf.images import ImageMasker


class TestPDFTables:
    """Test table handling."""
    
    def test_table_structure_preserved(self):
        """Test that table structure is preserved after translation."""
        translator = MockTranslator()
        table_translator = TableTranslator(translator)
        
        # Create a simple 2x2 table
        table = TableData(
            rows=2,
            cols=2,
            cells=[
                TableCell(content="Header 1", row=0, col=0),
                TableCell(content="Header 2", row=0, col=1),
                TableCell(content="Data 1", row=1, col=0),
                TableCell(content="Data 2", row=1, col=1),
            ]
        )
        
        translated = table_translator.translate_table(
            table,
            TranslationDirection.EN_TO_AR,
            TranslationMode.TARGET_ONLY
        )
        
        # Structure should be preserved
        assert translated.rows == 2
        assert translated.cols == 2
        assert len(translated.cells) == 4
    
    def test_table_invariants_preserved(self):
        """Test that numbers in table cells are preserved."""
        translator = MockTranslator()
        table_translator = TableTranslator(translator)
        
        # Table with numbers
        table = TableData(
            rows=1,
            cols=1,
            cells=[
                TableCell(content="25", row=0, col=0),
            ]
        )
        
        translated = table_translator.translate_table(
            table,
            TranslationDirection.EN_TO_AR,
            TranslationMode.TARGET_ONLY
        )
        
        # Number should be preserved exactly
        assert translated.cells[0].content == "25"
    
    def test_table_to_markdown(self):
        """Test table markdown export."""
        table = TableData(
            rows=2,
            cols=2,
            cells=[
                TableCell(content="A", row=0, col=0),
                TableCell(content="B", row=0, col=1),
                TableCell(content="C", row=1, col=0),
                TableCell(content="D", row=1, col=1),
            ]
        )
        
        markdown = table.to_markdown()
        
        # Should contain pipe separators
        assert "|" in markdown
        # Should contain all cells
        assert "A" in markdown
        assert "B" in markdown
        assert "C" in markdown
        assert "D" in markdown


class TestPDFImages:
    """Test image handling."""
    
    def test_image_masking(self):
        """Test that masking whites out bbox region."""
        masker = ImageMasker()
        
        # Create a test image (100x100, all black)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mask a region
        bbox = (10, 10, 50, 50)
        masked_image, mask = masker.make_mask(image, bbox)
        
        # Check that bbox region is white (255)
        region = masked_image[10:50, 10:50]
        assert np.all(region == 255)
        
        # Check mask
        assert mask[25, 25] == 255  # Inside bbox
        assert mask[0, 0] == 0  # Outside bbox
    
    def test_image_masking_bbox_clipping(self):
        """Test that bbox is clipped to image bounds."""
        masker = ImageMasker()
        
        # Small image
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        # Bbox exceeds image bounds
        bbox = (40, 40, 100, 100)
        masked_image, mask = masker.make_mask(image, bbox)
        
        # Should not crash and should clip to image bounds
        assert masked_image.shape == image.shape
        assert mask.shape == (50, 50)
