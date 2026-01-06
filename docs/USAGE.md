# PROTranslate Usage Guide

## Overview

PROTranslate is a production-grade document translation system with support for PDF, PPTX, and DOCX formats. It features:

- **Invariant Protection**: Preserves numbers, URLs, citations, LaTeX, scientific symbols
- **Bilingual & Target-Only Modes**: Flexible output options
- **Table Support**: Extracts and translates tables cell-by-cell
- **Image Handling**: Safe captions and masking preparation
- **Production Translation**: Gemini API integration with caching and glossary
- **RTL Support**: Proper Arabic text shaping

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Mock Mode (Testing)

```bash
python -m src.cli.translate input.pdf output.pdf \
  --direction en_to_ar \
  --mode bilingual \
  --translator mock
```

### API Mode (Production with Gemini)

1. **Set up environment variables**:

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

2. **Run translation**:

```bash
python -m src.cli.translate input.pdf output.pdf \
  --direction en_to_ar \
  --mode bilingual \
  --translator api
```

## Environment Variables

### Required for API Mode

- `GEMINI_API_KEY`: Your Gemini API key (get from https://aistudio.google.com/app/apikey)
- Alternative: `GOOGLE_API_KEY` (GEMINI_API_KEY takes precedence)

### Optional Configuration

```bash
# Translator Mode
TRANSLATOR_MODE=mock  # or 'api'

# API Settings
API_PROVIDER=gemini_openai_compat
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL=gemini-1.5-flash

# Request Settings
TIMEOUT_SECONDS=30
RETRY_MAX=3
RETRY_BACKOFF_BASE=2.0

# Chunking
MAX_CHUNK_CHARS=2000

# Cache
CACHE_ENABLED=true
CACHE_PATH=outputs/cache

# Glossary
GLOSSARY_PATH=config/glossary.json

# Prompt Version
PROMPT_VERSION=v1
```

## CLI Options

### Basic Options

- `input`: Input file path (required)
- `output`: Output file path (required)
- `--direction`: Translation direction (`en_to_ar` or `ar_to_en`, default: `en_to_ar`)
- `--mode`: Output mode (`bilingual` or `target_only`, default: `bilingual`)

### Translator Options

- `--translator`: Backend (`mock` or `api`, default: `mock`)
- `--provider`: API provider (default: `gemini_openai_compat`)
- `--base-url`: API base URL (optional override)
- `--model`: Model name (optional override)

### PDF-Specific Options

- `--pdf-tables`: Table extraction (`auto`, `docling`, `none`, default: `auto`)
- `--pdf-images`: Image handling (`none`, `caption`, `mask`, default: `caption`)

### Cache and Glossary

- `--cache`: Enable/disable caching (`on` or `off`, default: `on`)
- `--glossary`: Path to glossary file (optional)

## Translation Modes

### Bilingual Mode

Outputs both source and translation:

```
Original text
[TR] Translated text
```

### Target-Only Mode

Outputs only the translation:

```
[TR] Translated text
```

## Glossary

Create a glossary file to ensure consistent translation of technical terms:

```json
{
  "version": "v1",
  "protected_terms": [
    "DNA",
    "RNA",
    "COVID-19"
  ],
  "term_mappings": {
    "machine learning": "تعلم الآلة",
    "artificial intelligence": "الذكاء الاصطناعي"
  }
}
```

- **Protected Terms**: Never translated (e.g., acronyms, proper nouns)
- **Term Mappings**: Fixed translations for consistency

## Invariant Protection

The system automatically preserves:

- **Numbers**: `25`, `3.14`, `100kg`
- **URLs**: `https://example.com`
- **Citations**: `[12]`, `(Smith, 2020)`
- **LaTeX/Math**: `$x^2$`, `$$\int$$`
- **Scientific Symbols**: `≥`, `≤`, `→`, `α`, `β`
- **Code**: `` `variable_name` ``

## QA Report

Every translation generates a QA report at `outputs/qa_report.json`:

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
    "method": "auto"
  },
  "images": {
    "detected": 5,
    "captions_added": 5
  },
  "cache": {
    "hits": 12,
    "misses": 8,
    "hit_rate": 0.6
  },
  "glossary": {
    "terms_matched_count": 15
  }
}
```

## Examples

### Basic PDF Translation

```bash
python -m src.cli.translate document.pdf translated.pdf
```

### English to Arabic with API

```bash
export GEMINI_API_KEY="your_key_here"
python -m src.cli.translate paper.pdf paper_ar.pdf \
  --direction en_to_ar \
  --mode target_only \
  --translator api
```

### With Custom Glossary

```bash
python -m src.cli.translate thesis.pdf thesis_ar.pdf \
  --translator api \
  --glossary my_glossary.json
```

### Disable Caching

```bash
python -m src.cli.translate doc.pdf doc_ar.pdf \
  --translator api \
  --cache off
```

## Troubleshooting

### API Key Error

```
Error: API mode requires GEMINI_API_KEY or GOOGLE_API_KEY
```

**Solution**: Set your API key in `.env` or environment:

```bash
export GEMINI_API_KEY="your_key_here"
```

### Table Extraction Warning

```
Warning: docling not installed; table extraction skipped
```

**Solution**: Install docling (optional):

```bash
pip install docling
```

### Rate Limit Errors

The system automatically retries with exponential backoff. Adjust retry settings:

```bash
export RETRY_MAX=5
export RETRY_BACKOFF_BASE=3.0
```

## Performance Tips

1. **Enable Caching**: Reduces API calls for repeated content
2. **Use Chunking**: Automatically splits long documents
3. **Glossary**: Pre-translate common terms for consistency
4. **Flash Model**: Use `gemini-1.5-flash` for cost efficiency

## Supported Formats

- ✅ **PDF**: Full support with tables and images
- ⏳ **PPTX**: Coming soon
- ⏳ **DOCX**: Coming soon

## License

See LICENSE file for details.
