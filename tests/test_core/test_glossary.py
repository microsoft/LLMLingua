"""Test glossary functionality."""

import pytest
import tempfile
from pathlib import Path
from src.core.glossary import Glossary, GlossaryProcessor


def test_glossary_load_save():
    """Test glossary loading and saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "glossary.json"
        
        # Create and save
        glossary = Glossary(
            protected_terms=["DNA", "RNA"],
            term_mappings={"machine learning": "تعلم الآلة"},
            version="v1"
        )
        glossary.save(path)
        
        # Load
        loaded = Glossary.load(path)
        assert loaded.protected_terms == ["DNA", "RNA"]
        assert loaded.term_mappings == {"machine learning": "تعلم الآلة"}
        assert loaded.version == "v1"


def test_glossary_protect_protected_terms():
    """Test protection of protected terms."""
    glossary = Glossary(protected_terms=["DNA", "RNA"])
    processor = GlossaryProcessor(glossary)
    
    text = "The DNA sequence contains RNA"
    protected = processor.protect(text)
    
    # Terms should be replaced with placeholders
    assert "DNA" not in protected
    assert "RNA" not in protected
    assert "__GLOSSARY_PROTECTED_" in protected
    
    # Restore should bring back original terms
    restored = processor.restore(protected, apply_mappings=False)
    assert restored == text


def test_glossary_term_mappings():
    """Test term mappings."""
    glossary = Glossary(
        term_mappings={"machine learning": "تعلم الآلة"}
    )
    processor = GlossaryProcessor(glossary)
    
    text = "Study of machine learning"
    protected = processor.protect(text)
    
    # Source term should be replaced
    assert "machine learning" not in protected
    assert "__GLOSSARY_MAPPING_" in protected
    
    # Restore with mappings should apply translation
    restored = processor.restore(protected, apply_mappings=True)
    assert "تعلم الآلة" in restored
    assert "machine learning" not in restored


def test_glossary_stats():
    """Test glossary statistics."""
    glossary = Glossary(
        protected_terms=["DNA"],
        term_mappings={"machine learning": "تعلم الآلة"}
    )
    processor = GlossaryProcessor(glossary)
    
    text = "DNA and machine learning"
    processor.protect(text)
    
    stats = processor.get_stats()
    assert stats.protected_terms_count == 1
    assert stats.mapping_terms_count == 1
    assert stats.terms_matched_count == 2


def test_glossary_roundtrip():
    """Test protect/restore roundtrip."""
    glossary = Glossary(
        protected_terms=["DNA"],
        term_mappings={"ML": "تعلم الآلة"}
    )
    processor = GlossaryProcessor(glossary)
    
    text = "DNA research in ML"
    protected = processor.protect(text)
    restored = processor.restore(protected, apply_mappings=True)
    
    # DNA should be preserved, ML should be mapped
    assert "DNA" in restored
    assert "تعلم الآلة" in restored
    assert "ML" not in restored
