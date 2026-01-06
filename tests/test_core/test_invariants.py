"""Test invariant protection."""

import pytest
from src.core.invariants import InvariantProtector


def test_protect_numbers():
    """Test that numbers are protected."""
    protector = InvariantProtector()
    text = "The value is 25 and the ratio is 3.14"
    protected = protector.protect(text)
    
    # Numbers should be replaced with placeholders
    assert "25" not in protected
    assert "3.14" not in protected
    assert "__INVARIANT_" in protected
    
    # Restore should bring back original numbers
    restored = protector.restore(protected)
    assert restored == text


def test_protect_urls():
    """Test that URLs are protected."""
    protector = InvariantProtector()
    text = "Visit https://example.com for more info"
    protected = protector.protect(text)
    
    assert "https://example.com" not in protected
    assert "__INVARIANT_" in protected
    
    restored = protector.restore(protected)
    assert restored == text


def test_protect_citations():
    """Test that citations are protected."""
    protector = InvariantProtector()
    text = "According to research [12] and (Smith, 2020)"
    protected = protector.protect(text)
    
    assert "[12]" not in protected
    assert "(Smith, 2020)" not in protected
    
    restored = protector.restore(protected)
    assert restored == text


def test_protect_scientific_symbols():
    """Test that scientific symbols are protected."""
    protector = InvariantProtector()
    text = "The inequality is x ≥ 5 and α → β"
    protected = protector.protect(text)
    
    assert "≥" not in protected
    assert "→" not in protected
    assert "α" not in protected
    assert "β" not in protected
    
    restored = protector.restore(protected)
    assert restored == text


def test_is_invariant_only():
    """Test detection of invariant-only text."""
    protector = InvariantProtector()
    
    # Pure number
    assert protector.is_invariant_only("25")
    
    # URL only
    assert protector.is_invariant_only("https://example.com")
    
    # Mixed content
    assert not protector.is_invariant_only("The value is 25")
    
    # Regular text
    assert not protector.is_invariant_only("Hello world")


def test_roundtrip_complex_text():
    """Test protect/restore roundtrip with complex text."""
    protector = InvariantProtector()
    text = "Study [12] shows that 95% of samples at https://data.org have α ≥ 0.05"
    
    protected = protector.protect(text)
    restored = protector.restore(protected)
    
    assert restored == text
