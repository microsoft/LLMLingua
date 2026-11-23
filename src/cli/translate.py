"""CLI for document translation."""

import argparse
import sys
from pathlib import Path

from src.core.translator import get_translator, TranslationDirection, TranslationMode
from src.core.qa_report import QAReportManager
from src.formats.pdf.writer import PDFWriter, PDFStrategy
from src.formats.pdf.tables import TableExtractionMethod
from src.formats.pdf.images import ImageMode
from config.settings import get_settings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PROTranslate - Professional document translation"
    )
    
    # Input/output
    parser.add_argument("input", type=str, help="Input file path")
    parser.add_argument("output", type=str, help="Output file path")
    
    # Translation settings
    parser.add_argument(
        "--direction",
        choices=["en_to_ar", "ar_to_en"],
        default="en_to_ar",
        help="Translation direction"
    )
    parser.add_argument(
        "--mode",
        choices=["bilingual", "target_only"],
        default="bilingual",
        help="Output mode"
    )
    
    # Translator backend
    parser.add_argument(
        "--translator",
        choices=["mock", "api"],
        default="mock",
        help="Translator backend"
    )
    parser.add_argument(
        "--provider",
        default="gemini_openai_compat",
        help="API provider (for api mode)"
    )
    parser.add_argument(
        "--base-url",
        help="API base URL (optional override)"
    )
    parser.add_argument(
        "--model",
        help="Model name (optional override)"
    )
    
    # PDF-specific options
    parser.add_argument(
        "--pdf-tables",
        choices=["auto", "docling", "none"],
        default="auto",
        help="PDF table extraction method"
    )
    parser.add_argument(
        "--pdf-images",
        choices=["none", "caption", "mask"],
        default="caption",
        help="PDF image handling mode"
    )
    
    # Cache and glossary
    parser.add_argument(
        "--cache",
        choices=["on", "off"],
        default="on",
        help="Enable/disable caching"
    )
    parser.add_argument(
        "--glossary",
        type=str,
        help="Path to glossary file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine format
    ext = input_path.suffix.lower()
    if ext not in [".pdf", ".pptx", ".docx"]:
        print(f"Error: Unsupported file format: {ext}", file=sys.stderr)
        print("Supported formats: .pdf, .pptx, .docx", file=sys.stderr)
        sys.exit(1)
    
    # Override settings from CLI
    settings = get_settings()
    if args.translator:
        settings.TRANSLATOR_MODE = args.translator
    if args.base_url:
        settings.API_BASE_URL = args.base_url
    if args.model:
        settings.MODEL = args.model
    if args.cache:
        settings.CACHE_ENABLED = args.cache == "on"
    if args.glossary:
        settings.GLOSSARY_PATH = args.glossary
    
    # Get translator
    try:
        translator = get_translator(backend=settings.TRANSLATOR_MODE)
    except Exception as e:
        print(f"Error initializing translator: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse direction and mode
    direction = TranslationDirection(args.direction)
    mode = TranslationMode(args.mode)
    
    # Create QA report manager
    qa_manager = QAReportManager()
    qa_report = qa_manager.create_report(
        input_file=str(input_path),
        output_file=args.output,
        format=ext[1:],  # Remove dot
        direction=args.direction,
        mode=args.mode,
        translator_backend=settings.TRANSLATOR_MODE,
        provider=settings.API_PROVIDER if settings.TRANSLATOR_MODE == "api" else None,
        model=settings.MODEL if settings.TRANSLATOR_MODE == "api" else None,
        prompt_version=settings.PROMPT_VERSION if settings.TRANSLATOR_MODE == "api" else None
    )
    
    # Translate based on format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if ext == ".pdf":
            # PDF translation
            table_method = TableExtractionMethod(args.pdf_tables)
            image_mode = ImageMode(args.pdf_images)
            
            writer = PDFWriter(
                translator=translator,
                strategy=PDFStrategy.SAFE,
                table_method=table_method,
                image_mode=image_mode
            )
            
            writer.translate_pdf(
                input_path=input_path,
                output_path=output_path,
                direction=direction,
                mode=mode,
                qa_report=qa_report
            )
            
            print(f"✓ PDF translated successfully: {output_path}")
        
        elif ext == ".pptx":
            print("Error: PPTX translation not implemented yet", file=sys.stderr)
            sys.exit(1)
        
        elif ext == ".docx":
            print("Error: DOCX translation not implemented yet", file=sys.stderr)
            sys.exit(1)
        
        # Update cache stats in QA report
        if settings.TRANSLATOR_MODE == "api":
            cache_stats = translator.cache.get_stats()
            qa_report.cache["enabled"] = True
            qa_report.cache["hits"] = cache_stats.hits
            qa_report.cache["misses"] = cache_stats.misses
            qa_report.cache["hit_rate"] = cache_stats.hit_rate
            qa_report.cache["cache_size"] = cache_stats.cache_size
            
            # Glossary stats
            glossary_stats = translator.glossary_processor.get_stats()
            qa_report.glossary["enabled"] = True
            qa_report.glossary["terms_matched_count"] = glossary_stats.terms_matched_count
            qa_report.glossary["protected_terms_count"] = glossary_stats.protected_terms_count
            qa_report.glossary["mapping_terms_count"] = glossary_stats.mapping_terms_count
        
        # Save QA report
        qa_manager.save_report(qa_report)
        print(f"✓ QA report saved: {qa_manager.get_report_path()}")
    
    except Exception as e:
        print(f"Error during translation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
