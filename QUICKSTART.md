# PROTranslate Quick Start

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Test the System

```bash
# Run all tests (should see 33 passed)
pytest tests/test_core tests/test_pdf tests/test_pptx tests/test_docx -v
```

## Create a Test PDF

```bash
python3 -c "
import fitz
doc = fitz.open()
page = doc.new_page(width=612, height=792)
page.insert_text((50, 50), 'Sample Document', fontsize=16)
page.insert_text((50, 100), 'This is a test with number 25 and URL https://example.com', fontsize=12)
page.insert_text((50, 150), 'Scientific notation: α ≥ 0.05', fontsize=12)
doc.save('test_input.pdf')
doc.close()
print('✓ Test PDF created: test_input.pdf')
"
```

## Run Translation (Mock Mode)

```bash
python -m src.cli.translate test_input.pdf outputs/test_output.pdf \
  --direction en_to_ar \
  --mode bilingual \
  --translator mock \
  --pdf-tables auto \
  --pdf-images caption
```

**Expected Output:**
```
✓ PDF translated successfully: outputs/test_output.pdf
✓ QA report saved: outputs/qa_report.json
```

## Check QA Report

```bash
cat outputs/qa_report.json
```

**Should show:**
- `translator_backend: "mock"`
- `pages_count: 1`
- `blocks_translated: 3`
- Tables and images sections
- All required QA fields

## Run with Gemini API (Production)

1. **Get API Key**: https://aistudio.google.com/app/apikey

2. **Set Environment Variable**:
```bash
export GEMINI_API_KEY="your_key_here"
```

3. **Run Translation**:
```bash
python -m src.cli.translate test_input.pdf outputs/test_output_api.pdf \
  --direction en_to_ar \
  --mode target_only \
  --translator api \
  --cache on
```

## Verify Invariants Preserved

```bash
# Check that numbers, URLs, and symbols are preserved
python3 -c "
import fitz
doc = fitz.open('outputs/test_output.pdf')
text = doc[1].get_text()  # Translated page
print('Checking invariants in translated page:')
print('✓ Number 25 preserved:', '25' in text)
print('✓ URL preserved:', 'https://example.com' in text)
print('✓ Symbol preserved:', '≥' in text or 'α' in text)
doc.close()
"
```

## Common Commands

### Mock Translation (Fast Testing)
```bash
python -m src.cli.translate input.pdf output.pdf
```

### API Translation with Caching
```bash
python -m src.cli.translate input.pdf output.pdf --translator api --cache on
```

### Target-Only Mode (No Source)
```bash
python -m src.cli.translate input.pdf output.pdf --mode target_only
```

### With Custom Glossary
```bash
python -m src.cli.translate input.pdf output.pdf --glossary my_glossary.json
```

## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
# Set your API key
export GEMINI_API_KEY="your_key_here"
```

### "docling not installed"
This is expected and not an error. Table extraction will be skipped gracefully.

To enable advanced table extraction:
```bash
pip install docling
```

### Run Tests to Verify Installation
```bash
pytest tests/test_core tests/test_pdf -v
# Should see: 33 passed
```

## Next Steps

- Read `docs/USAGE.md` for complete documentation
- Read `docs/README_PROTRANSLATE.md` for architecture details
- Check `IMPLEMENTATION_SUMMARY.md` for implementation details
- Customize `config/glossary.json` for your domain

## Support

For issues, check:
1. All dependencies installed: `pip install -r requirements.txt`
2. Tests passing: `pytest tests/test_core tests/test_pdf -v`
3. API key set (for API mode): `echo $GEMINI_API_KEY`
