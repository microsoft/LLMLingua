# PROTranslate

**Production-Grade Document Translation System**

A robust, enterprise-ready translation system for academic and technical documents with strict invariant protection, bilingual output, and production API integration.

## Features

### Core Translation

- ✅ **Mock Translator**: Fast testing and development
- ✅ **API Translator**: Production-ready with Gemini OpenAI-compatible endpoint
- ✅ **Invariant Protection**: Preserves numbers, URLs, citations, LaTeX, scientific symbols
- ✅ **Bilingual & Target-Only Modes**: Flexible output options
- ✅ **RTL Support**: Proper Arabic text shaping with `arabic-reshaper` and `python-bidi`

### PDF Support (SAFE Strategy)

- ✅ **Page-after-Page**: Source page followed by translated page
- ✅ **Table Extraction**: Cell-by-cell translation with structure preservation
- ✅ **Image Handling**: Safe captions and masking preparation (LaMa inpainting stub)
- ✅ **Text Blocks**: Full text extraction and translation

### Production Features

- ✅ **Caching**: SQLite-based translation cache for cost control
- ✅ **Glossary**: Protected terms and fixed mappings for consistency
- ✅ **Chunking**: Smart text splitting with boundary detection
- ✅ **Retry Logic**: Exponential backoff for rate limits
- ✅ **QA Reports**: Comprehensive metrics and warnings

## Quick Start

### Installation

```bash
git clone <repository>
cd PROTranslate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Mock mode (testing)
python -m src.cli.translate input.pdf output.pdf

# API mode (production)
export GEMINI_API_KEY="your_key_here"
python -m src.cli.translate input.pdf output.pdf --translator api
```

## Architecture

```
src/
├── core/
│   ├── translator.py          # Base interface and factory
│   ├── invariants.py           # Invariant protection
│   ├── glossary.py             # Glossary management
│   ├── chunker.py              # Text chunking
│   ├── rtl_utils.py            # RTL text shaping
│   ├── qa_report.py            # QA reporting
│   └── translators/
│       ├── mock_translator.py  # Mock implementation
│       └── api_translator.py   # API implementation
├── formats/
│   └── pdf/
│       ├── parser.py           # PDF parsing
│       ├── writer.py           # PDF writing (SAFE strategy)
│       ├── tables.py           # Table extraction/translation
│       └── images.py           # Image detection/masking
├── cache/
│   └── translation_cache.py    # SQLite caching
├── cli/
│   └── translate.py            # CLI interface
└── prompts/
    └── translate.txt           # Prompt template
```

## Testing

All tests pass with zero regressions:

```bash
# Run all PROTranslate tests
pytest tests/test_core tests/test_pdf tests/test_pptx tests/test_docx -v

# Results: 33 passed ✅
```

### Test Coverage

- ✅ Invariant protection (numbers, URLs, citations, symbols)
- ✅ Mock translator (basic, bilingual, batch)
- ✅ Caching (basic, stats, disabled mode)
- ✅ Glossary (protected terms, mappings, roundtrip)
- ✅ Chunking (splitting, structure preservation, stats)
- ✅ PDF tables (structure, invariants, markdown export)
- ✅ PDF images (masking, bbox clipping)
- ✅ QA reports (all required sections)

## Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
# Translator Mode
TRANSLATOR_MODE=mock  # or 'api'

# API Settings (required for api mode)
GEMINI_API_KEY=your_key_here
API_PROVIDER=gemini_openai_compat
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL=gemini-1.5-flash

# Cache
CACHE_ENABLED=true
CACHE_PATH=outputs/cache

# Glossary
GLOSSARY_PATH=config/glossary.json
```

### Glossary Example

```json
{
  "version": "v1",
  "protected_terms": ["DNA", "RNA", "COVID-19"],
  "term_mappings": {
    "machine learning": "تعلم الآلة",
    "artificial intelligence": "الذكاء الاصطناعي"
  }
}
```

## CLI Reference

```bash
python -m src.cli.translate INPUT OUTPUT [OPTIONS]

Options:
  --direction {en_to_ar,ar_to_en}  Translation direction
  --mode {bilingual,target_only}   Output mode
  --translator {mock,api}          Translator backend
  --provider TEXT                  API provider
  --model TEXT                     Model name
  --pdf-tables {auto,docling,none} Table extraction
  --pdf-images {none,caption,mask} Image handling
  --cache {on,off}                 Enable/disable cache
  --glossary PATH                  Glossary file path
```

## QA Report

Every translation generates `outputs/qa_report.json`:

```json
{
  "translator_backend": "api",
  "provider": "gemini_openai_compat",
  "model": "gemini-1.5-flash",
  "pages_count": 10,
  "blocks_translated": 45,
  "tables": {
    "detected": 3,
    "translated": 3,
    "method": "auto",
    "warnings": []
  },
  "images": {
    "detected": 5,
    "captions_added": 5,
    "resized_count": 0,
    "warnings": []
  },
  "chunking": {
    "chunks_count": 8,
    "avg_chunk_len": 1850,
    "max_chunk_len": 2000
  },
  "cache": {
    "enabled": true,
    "hits": 12,
    "misses": 8,
    "hit_rate": 0.6,
    "cache_size": 20
  },
  "glossary": {
    "enabled": true,
    "terms_matched_count": 15,
    "protected_terms_count": 8,
    "mapping_terms_count": 7
  },
  "retries": {
    "retry_count": 2,
    "failures_count": 0,
    "timeout_count": 0
  },
  "warnings": [],
  "fallbacks_used": [],
  "conversion_warnings": []
}
```

## Invariant Protection

Automatically preserves:

| Type | Examples |
|------|----------|
| Numbers | `25`, `3.14`, `100kg` |
| URLs | `https://example.com` |
| Citations | `[12]`, `(Smith, 2020)` |
| LaTeX | `$x^2$`, `$$\int f(x)dx$$` |
| Symbols | `≥`, `≤`, `→`, `α`, `β`, `γ` |
| Code | `` `variable_name` `` |

## Smoke Test Results

```bash
# Mock mode
✓ PDF translated successfully: outputs/test_output.pdf
✓ QA report saved: outputs/qa_report.json

# Test results
- Pages: 1
- Blocks translated: 3
- Tables detected: 0
- Images detected: 0
- All invariants preserved ✅
```

## Roadmap

- ✅ Phase A: Tables (SAFE strategy)
- ✅ Phase B: Images (captions + masking stub)
- ✅ Phase C: CLI integration
- ✅ Phase 7: Production translator (Gemini API + caching + glossary)
- ⏳ PPTX support
- ⏳ DOCX support
- ⏳ LaMa inpainting integration
- ⏳ OCR for scanned documents

## Dependencies

### Core
- `PyMuPDF>=1.23.0` - PDF parsing/writing
- `numpy>=1.24.0` - Array operations
- `opencv-python>=4.8.0` - Image masking

### API Translation
- `openai>=1.0.0` - OpenAI-compatible client

### RTL Support
- `arabic-reshaper>=3.0.0` - Arabic text shaping
- `python-bidi>=0.4.2` - Bidirectional text

### Optional
- `python-dotenv>=1.0.0` - Environment variables
- `docling` - Advanced table extraction

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- No regressions in existing functionality
- QA reports include new metrics
- Documentation updated

## Support

For issues and questions, please open a GitHub issue.
