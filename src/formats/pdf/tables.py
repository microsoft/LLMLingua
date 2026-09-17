"""PDF table extraction and translation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from enum import Enum

from src.core.translator import Translator, TranslationDirection, TranslationMode
from src.core.invariants import InvariantProtector


class TableExtractionMethod(Enum):
    """Table extraction methods."""
    NONE = "none"
    AUTO = "auto"
    DOCLING = "docling"


@dataclass
class TableCell:
    """Table cell data."""
    content: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


@dataclass
class TableData:
    """Structured table representation."""
    rows: int
    cols: int
    cells: List[TableCell] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    page_num: int = 0
    
    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at position."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None
    
    def to_markdown(self) -> str:
        """Export table as Markdown for debugging."""
        if not self.cells:
            return ""
        
        lines = []
        for row in range(self.rows):
            row_cells = []
            for col in range(self.cols):
                cell = self.get_cell(row, col)
                content = cell.content if cell else ""
                row_cells.append(content)
            lines.append("| " + " | ".join(row_cells) + " |")
            
            # Add separator after header
            if row == 0:
                lines.append("|" + "|".join(["---"] * self.cols) + "|")
        
        return "\n".join(lines)


class TableExtractor:
    """Extract tables from PDF pages."""
    
    def __init__(self, method: TableExtractionMethod = TableExtractionMethod.AUTO):
        """
        Initialize table extractor.
        
        Args:
            method: Extraction method to use
        """
        self.method = method
        self.docling_available = False
        
        if method in (TableExtractionMethod.AUTO, TableExtractionMethod.DOCLING):
            try:
                import docling
                self.docling_available = True
            except ImportError:
                self.docling_available = False
    
    def extract_tables(self, pdf_path: Path, page_num: Optional[int] = None) -> List[TableData]:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Optional specific page number
            
        Returns:
            List of extracted tables
        """
        if self.method == TableExtractionMethod.NONE:
            return []
        
        if self.method == TableExtractionMethod.DOCLING and not self.docling_available:
            return []
        
        if self.method == TableExtractionMethod.AUTO and not self.docling_available:
            return []
        
        # If docling is available, use it
        if self.docling_available:
            return self._extract_with_docling(pdf_path, page_num)
        
        return []
    
    def _extract_with_docling(self, pdf_path: Path, page_num: Optional[int]) -> List[TableData]:
        """Extract tables using docling (placeholder for now)."""
        # Docling integration would go here
        # For now, return empty list
        return []


class TableTranslator:
    """Translate table cells."""
    
    def __init__(self, translator: Translator):
        """Initialize table translator."""
        self.translator = translator
        self.protector = InvariantProtector()
    
    def translate_table(
        self,
        table: TableData,
        direction: TranslationDirection,
        mode: TranslationMode
    ) -> TableData:
        """
        Translate table cell-by-cell.
        
        Args:
            table: Table to translate
            direction: Translation direction
            mode: Translation mode
            
        Returns:
            Translated table
        """
        translated_cells = []
        
        for cell in table.cells:
            # Check if cell is invariant-only
            if self.protector.is_invariant_only(cell.content):
                # Don't translate invariant-only cells
                translated_content = cell.content
            else:
                # Translate cell content
                translated_text, _ = self.translator.translate(
                    cell.content,
                    direction,
                    mode,
                    context={"is_table_cell": True}
                )
                translated_content = translated_text
            
            translated_cell = TableCell(
                content=translated_content,
                row=cell.row,
                col=cell.col,
                rowspan=cell.rowspan,
                colspan=cell.colspan
            )
            translated_cells.append(translated_cell)
        
        return TableData(
            rows=table.rows,
            cols=table.cols,
            cells=translated_cells,
            bbox=table.bbox,
            page_num=table.page_num
        )
