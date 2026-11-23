"""PDF writer with SAFE strategy (page-after-page)."""

from pathlib import Path
from typing import Optional
from enum import Enum

from src.core.translator import Translator, TranslationDirection, TranslationMode
from src.core.qa_report import QAReport
from src.formats.pdf.parser import PDFParser, PDFData
from src.formats.pdf.tables import TableExtractor, TableTranslator, TableExtractionMethod
from src.formats.pdf.images import ImageDetector, ImageMode

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFStrategy(Enum):
    """PDF translation strategies."""
    SAFE = "safe"  # Page-after-page: source page, then translated page


class PDFWriter:
    """Write translated PDF documents."""
    
    def __init__(
        self,
        translator: Translator,
        strategy: PDFStrategy = PDFStrategy.SAFE,
        table_method: TableExtractionMethod = TableExtractionMethod.AUTO,
        image_mode: ImageMode = ImageMode.CAPTION
    ):
        """
        Initialize PDF writer.
        
        Args:
            translator: Translator instance
            strategy: Translation strategy
            table_method: Table extraction method
            image_mode: Image handling mode
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF writing")
        
        self.translator = translator
        self.strategy = strategy
        self.parser = PDFParser()
        self.table_extractor = TableExtractor(table_method)
        self.table_translator = TableTranslator(translator)
        self.image_detector = ImageDetector()
        self.table_method = table_method
        self.image_mode = image_mode
    
    def translate_pdf(
        self,
        input_path: Path,
        output_path: Path,
        direction: TranslationDirection,
        mode: TranslationMode,
        qa_report: Optional[QAReport] = None
    ):
        """
        Translate PDF document.
        
        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            direction: Translation direction
            mode: Translation mode
            qa_report: Optional QA report to update
        """
        # Parse input PDF
        pdf_data = self.parser.parse(input_path)
        
        # Detect tables
        tables = self.table_extractor.extract_tables(input_path)
        
        # Detect images
        images = self.image_detector.detect_images(input_path)
        
        # Update QA report
        if qa_report:
            qa_report.pages_count = len(pdf_data.pages)
            qa_report.tables["detected"] = len(tables)
            qa_report.tables["method"] = self.table_method.value
            qa_report.images["detected"] = len(images)
            
            if self.table_method != TableExtractionMethod.NONE and not self.table_extractor.docling_available:
                qa_report.add_warning("docling not installed; table extraction skipped")
                qa_report.tables["warnings"].append("docling not available")
        
        # Translate based on strategy
        if self.strategy == PDFStrategy.SAFE:
            self._translate_safe(
                input_path,
                output_path,
                pdf_data,
                tables,
                images,
                direction,
                mode,
                qa_report
            )
    
    def _translate_safe(
        self,
        input_path: Path,
        output_path: Path,
        pdf_data: PDFData,
        tables: list,
        images: list,
        direction: TranslationDirection,
        mode: TranslationMode,
        qa_report: Optional[QAReport]
    ):
        """
        SAFE strategy: source page, then translated page.
        
        Creates a new PDF with alternating source and translated pages.
        """
        # Open source PDF
        src_doc = fitz.open(input_path)
        
        # Create output PDF
        out_doc = fitz.open()
        
        blocks_translated = 0
        tables_translated = 0
        captions_added = 0
        
        for page_num in range(len(src_doc)):
            src_page = src_doc[page_num]
            
            # Copy source page
            out_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            
            # Create translated page
            trans_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
            
            # Translate text blocks
            page_data = pdf_data.pages[page_num]
            y_offset = 50
            
            for block in page_data.blocks:
                if block.type == "text" and block.content.strip():
                    translated_text, metadata = self.translator.translate(
                        block.content,
                        direction,
                        mode
                    )
                    
                    # Write translated text
                    trans_page.insert_text(
                        (50, y_offset),
                        translated_text,
                        fontsize=11,
                        fontname="helv"
                    )
                    y_offset += 50
                    blocks_translated += 1
            
            # Handle tables on this page
            page_tables = [t for t in tables if t.page_num == page_num]
            for table in page_tables:
                translated_table = self.table_translator.translate_table(table, direction, mode)
                # Render table as markdown block
                table_md = translated_table.to_markdown()
                trans_page.insert_text(
                    (50, y_offset),
                    table_md,
                    fontsize=9,
                    fontname="cour"
                )
                y_offset += 100
                tables_translated += 1
            
            # Handle images on this page
            page_images = [img for img in images if img.page_num == page_num]
            for img in page_images:
                if self.image_mode == ImageMode.CAPTION:
                    # Add caption below image placeholder
                    caption = f"[Image {img.image_index}]"
                    trans_page.insert_text(
                        (50, y_offset),
                        caption,
                        fontsize=10,
                        fontname="helv"
                    )
                    y_offset += 30
                    captions_added += 1
        
        # Update QA report
        if qa_report:
            qa_report.blocks_translated = blocks_translated
            qa_report.tables["translated"] = tables_translated
            qa_report.images["captions_added"] = captions_added
        
        # Save output
        out_doc.save(output_path)
        out_doc.close()
        src_doc.close()
