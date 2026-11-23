"""Test mock translator."""

import pytest
from src.core.translator import TranslationDirection, TranslationMode
from src.core.translators.mock_translator import MockTranslator


def test_mock_translator_basic():
    """Test basic mock translation."""
    translator = MockTranslator()
    
    text = "Hello world"
    result, metadata = translator.translate(
        text,
        TranslationDirection.EN_TO_AR,
        TranslationMode.TARGET_ONLY
    )
    
    assert "[TR]" in result
    assert metadata.backend == "mock"
    assert metadata.chunks_count == 1


def test_mock_translator_bilingual():
    """Test bilingual mode."""
    translator = MockTranslator()
    
    text = "Hello world"
    result, metadata = translator.translate(
        text,
        TranslationDirection.EN_TO_AR,
        TranslationMode.BILINGUAL
    )
    
    # Should contain both source and translation
    assert "Hello world" in result
    assert "[TR]" in result
    assert "\n" in result  # Separated by newline


def test_mock_translator_preserves_numbers():
    """Test that numbers are preserved."""
    translator = MockTranslator()
    
    text = "The value is 25"
    result, metadata = translator.translate(
        text,
        TranslationDirection.EN_TO_AR,
        TranslationMode.TARGET_ONLY
    )
    
    # Number should be preserved exactly
    assert "25" in result


def test_mock_translator_invariant_only():
    """Test that invariant-only text is not translated."""
    translator = MockTranslator()
    
    text = "25"
    result, metadata = translator.translate(
        text,
        TranslationDirection.EN_TO_AR,
        TranslationMode.TARGET_ONLY
    )
    
    # Should return exactly "25", not "[TR] 25"
    assert result == "25"


def test_mock_translator_batch():
    """Test batch translation."""
    translator = MockTranslator()
    
    texts = ["Hello", "World"]
    results = translator.translate_batch(
        texts,
        TranslationDirection.EN_TO_AR,
        TranslationMode.TARGET_ONLY
    )
    
    assert len(results) == 2
    assert all("[TR]" in r[0] for r in results)
