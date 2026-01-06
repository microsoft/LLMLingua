"""Mock translator for testing and development."""

from typing import Optional, Dict, Any
from src.core.translator import Translator, TranslationDirection, TranslationMode, TranslationMetadata
from src.core.invariants import InvariantProtector


class MockTranslator(Translator):
    """Mock translator that prefixes text with [TR] for testing."""
    
    def __init__(self, **kwargs):
        """Initialize mock translator."""
        self.protector = InvariantProtector()
    
    def translate(
        self,
        text: str,
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, TranslationMetadata]:
        """
        Mock translate by prefixing with [TR].
        
        Preserves invariants (numbers, URLs, citations, symbols).
        """
        # Check if text is invariant-only
        if self.protector.is_invariant_only(text):
            # Don't translate invariant-only content
            result = text
        else:
            # Protect invariants
            protected = self.protector.protect(text)
            # Mock translation: prefix with [TR]
            translated = f"[TR] {protected}"
            # Restore invariants
            result = self.protector.restore(translated)
        
        # Apply mode
        if mode == TranslationMode.BILINGUAL:
            output = f"{text}\n{result}"
        else:
            output = result
        
        metadata = TranslationMetadata(
            backend="mock",
            chunks_count=1,
            cache_hit=False
        )
        
        return output, metadata
    
    def translate_batch(
        self,
        texts: list[str],
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> list[tuple[str, TranslationMetadata]]:
        """Translate multiple texts."""
        return [self.translate(text, direction, mode, context) for text in texts]
