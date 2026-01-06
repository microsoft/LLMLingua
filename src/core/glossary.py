"""Glossary management for consistent academic translation."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class GlossaryStats:
    """Glossary usage statistics."""
    terms_matched_count: int = 0
    protected_terms_count: int = 0
    mapping_terms_count: int = 0


@dataclass
class Glossary:
    """Glossary with protected terms and mappings."""
    protected_terms: List[str] = field(default_factory=list)
    term_mappings: Dict[str, str] = field(default_factory=dict)
    version: str = "v1"
    
    @classmethod
    def load(cls, path: Path) -> 'Glossary':
        """Load glossary from JSON file."""
        if not path.exists():
            return cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            protected_terms=data.get("protected_terms", []),
            term_mappings=data.get("term_mappings", {}),
            version=data.get("version", "v1")
        )
    
    def save(self, path: Path):
        """Save glossary to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "protected_terms": self.protected_terms,
            "term_mappings": self.term_mappings,
            "version": self.version
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class GlossaryProcessor:
    """Process text with glossary protection."""
    
    def __init__(self, glossary: Glossary):
        """Initialize glossary processor."""
        self.glossary = glossary
        self.placeholder_map: Dict[str, str] = {}
        self.counter = 0
        self.stats = GlossaryStats()
    
    def protect(self, text: str) -> str:
        """
        Replace glossary terms with collision-safe placeholders.
        
        Args:
            text: Original text
            
        Returns:
            Text with glossary terms replaced by placeholders
        """
        self.placeholder_map = {}
        self.counter = 0
        self.stats = GlossaryStats()
        
        protected_text = text
        
        # Protect protected terms (never translate)
        for term in self.glossary.protected_terms:
            if term in protected_text:
                placeholder = f"__GLOSSARY_PROTECTED_{self.counter}__"
                self.placeholder_map[placeholder] = term
                protected_text = protected_text.replace(term, placeholder)
                self.counter += 1
                self.stats.protected_terms_count += 1
                self.stats.terms_matched_count += 1
        
        # Protect source terms that have mappings
        for source_term in self.glossary.term_mappings.keys():
            if source_term in protected_text:
                placeholder = f"__GLOSSARY_MAPPING_{self.counter}__"
                self.placeholder_map[placeholder] = source_term
                protected_text = protected_text.replace(source_term, placeholder)
                self.counter += 1
                self.stats.mapping_terms_count += 1
                self.stats.terms_matched_count += 1
        
        return protected_text
    
    def restore(self, text: str, apply_mappings: bool = True) -> str:
        """
        Restore glossary terms from placeholders.
        
        Args:
            text: Text with placeholders
            apply_mappings: Whether to apply term mappings
            
        Returns:
            Text with glossary terms restored
        """
        restored_text = text
        
        for placeholder, original_term in self.placeholder_map.items():
            if placeholder.startswith("__GLOSSARY_PROTECTED_"):
                # Restore protected term as-is
                restored_text = restored_text.replace(placeholder, original_term)
            elif placeholder.startswith("__GLOSSARY_MAPPING_"):
                # Apply mapping if available
                if apply_mappings and original_term in self.glossary.term_mappings:
                    target_term = self.glossary.term_mappings[original_term]
                    restored_text = restored_text.replace(placeholder, target_term)
                else:
                    restored_text = restored_text.replace(placeholder, original_term)
        
        return restored_text
    
    def get_stats(self) -> GlossaryStats:
        """Get glossary usage statistics."""
        return self.stats
