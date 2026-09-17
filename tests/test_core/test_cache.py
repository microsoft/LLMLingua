"""Test translation caching."""

import pytest
import tempfile
from pathlib import Path
from src.cache.translation_cache import TranslationCache


def test_cache_basic():
    """Test basic cache operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TranslationCache(Path(tmpdir), enabled=True)
        
        # Cache miss
        result = cache.get(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello"
        )
        assert result is None
        
        # Set cache
        cache.set(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello",
            translation="مرحبا"
        )
        
        # Cache hit
        result = cache.get(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello"
        )
        assert result is not None
        translation, metadata = result
        assert translation == "مرحبا"


def test_cache_stats():
    """Test cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TranslationCache(Path(tmpdir), enabled=True)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        
        # Miss
        cache.get(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello"
        )
        
        stats = cache.get_stats()
        assert stats.misses == 1
        
        # Set and hit
        cache.set(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello",
            translation="مرحبا"
        )
        
        cache.get(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello"
        )
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.hit_rate == 0.5  # 1 hit, 1 miss


def test_cache_disabled():
    """Test that disabled cache doesn't store anything."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TranslationCache(Path(tmpdir), enabled=False)
        
        cache.set(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello",
            translation="مرحبا"
        )
        
        result = cache.get(
            provider="test",
            base_url="http://test",
            model="test-model",
            direction="en_to_ar",
            mode="target_only",
            prompt_version="v1",
            glossary_version="v1",
            text="Hello"
        )
        
        assert result is None
