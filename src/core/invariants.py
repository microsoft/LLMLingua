"""Invariant protection for translation - preserve numbers, URLs, citations, symbols."""

import re
from typing import Dict, Tuple


class InvariantProtector:
    """Protects invariants (numbers, URLs, citations, symbols) during translation."""
    
    # Patterns for invariants that should never be translated
    PATTERNS = {
        'url': re.compile(r'https?://[^\s]+|www\.[^\s]+'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'number': re.compile(r'\b\d+\.?\d*\b'),
        'citation_bracket': re.compile(r'\[\d+\]|\[\d+,\s*\d+\]'),
        'citation_paren': re.compile(r'\([A-Z][a-z]+,?\s+\d{4}\)'),
        'latex_inline': re.compile(r'\$[^$]+\$'),
        'latex_display': re.compile(r'\$\$[^$]+\$\$'),
        'scientific_symbol': re.compile(r'[≥≤≈≠±×÷∞∑∏∫∂∇√∈∉⊂⊃∪∩→←↔αβγδεζηθικλμνξοπρστυφχψω]'),
        'code_inline': re.compile(r'`[^`]+`'),
        'variable': re.compile(r'\b[a-z_][a-z0-9_]*\b(?=\s*[=\(])'),  # Simple variable detection
    }
    
    def __init__(self):
        self.placeholder_map: Dict[str, str] = {}
        self.counter = 0
    
    def protect(self, text: str) -> str:
        """
        Replace invariants with placeholders.
        
        Args:
            text: Original text
            
        Returns:
            Text with invariants replaced by placeholders
        """
        self.placeholder_map = {}
        self.counter = 0
        protected_text = text
        
        # Protect each pattern type
        for pattern_name, pattern in self.PATTERNS.items():
            protected_text = self._protect_pattern(protected_text, pattern, pattern_name)
        
        return protected_text
    
    def _protect_pattern(self, text: str, pattern: re.Pattern, pattern_name: str) -> str:
        """Protect a specific pattern with placeholders."""
        def replacer(match):
            original = match.group(0)
            placeholder = f"__INVARIANT_{self.counter}__"
            self.placeholder_map[placeholder] = original
            self.counter += 1
            return placeholder
        
        return pattern.sub(replacer, text)
    
    def restore(self, text: str) -> str:
        """
        Restore invariants from placeholders.
        
        Args:
            text: Text with placeholders
            
        Returns:
            Text with original invariants restored
        """
        restored_text = text
        for placeholder, original in self.placeholder_map.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text
    
    def is_invariant_only(self, text: str) -> bool:
        """
        Check if text contains only invariants (no translatable content).
        
        Args:
            text: Text to check
            
        Returns:
            True if text is invariant-only
        """
        protected = self.protect(text.strip())
        # Remove all placeholders
        for placeholder in self.placeholder_map.keys():
            protected = protected.replace(placeholder, '')
        # Check if anything meaningful remains
        remaining = protected.strip()
        return len(remaining) == 0 or remaining.replace(' ', '') == ''
