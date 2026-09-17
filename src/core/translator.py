"""Base translator interface and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class TranslationDirection(Enum):
    """Translation direction."""
    EN_TO_AR = "en_to_ar"
    AR_TO_EN = "ar_to_en"


class TranslationMode(Enum):
    """Translation output mode."""
    BILINGUAL = "bilingual"  # Source + translation
    TARGET_ONLY = "target_only"  # Translation only


@dataclass
class TranslationMetadata:
    """Metadata about a translation operation."""
    backend: str
    provider: Optional[str] = None
    model: Optional[str] = None
    chunks_count: int = 1
    cache_hit: bool = False
    retry_count: int = 0
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class Translator(ABC):
    """Abstract base translator interface."""
    
    @abstractmethod
    def translate(
        self,
        text: str,
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, TranslationMetadata]:
        """
        Translate text.
        
        Args:
            text: Source text to translate
            direction: Translation direction
            mode: Output mode (bilingual or target-only)
            context: Optional context for translation
            
        Returns:
            Tuple of (translated_text, metadata)
        """
        pass
    
    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> list[tuple[str, TranslationMetadata]]:
        """Translate multiple texts efficiently."""
        pass


def get_translator(backend: str = "mock", **kwargs) -> Translator:
    """
    Factory function to get translator instance.
    
    Args:
        backend: Translator backend ('mock' or 'api')
        **kwargs: Additional configuration for the translator
        
    Returns:
        Translator instance
    """
    if backend == "mock":
        from src.core.translators.mock_translator import MockTranslator
        return MockTranslator(**kwargs)
    elif backend == "api":
        from src.core.translators.api_translator import APITranslator
        return APITranslator(**kwargs)
    else:
        raise ValueError(f"Unknown translator backend: {backend}")
