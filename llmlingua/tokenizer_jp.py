"""
Japanese tokenizer for LLMLingua using fugashi + unidic-lite.
Provides tokenization that preserves sentence structure and punctuation.
"""

import re
from typing import List, Tuple

try:
    import fugashi
    import unidic_lite  # noqa: F401 - required by fugashi
except ImportError:
    raise ImportError(
        "Japanese tokenization requires fugashi and unidic-lite. "
        "Install with: pip install fugashi unidic-lite"
    )


class JapaneseTokenizer:
    """Japanese text tokenizer using fugashi with unidic-lite dictionary."""

    def __init__(self):
        """Initialize the Japanese tokenizer."""
        self.tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Japanese text into words.

        Args:
            text: Input Japanese text

        Returns:
            List of tokenized words
        """
        if not text or not text.strip():
            return []

        # Parse with fugashi
        words = self.tagger(text)

        # Extract surface forms and filter empty tokens
        tokens = [word.surface for word in words if word.surface.strip()]

        return tokens

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize Japanese text with part-of-speech information.

        Args:
            text: Input Japanese text

        Returns:
            List of (token, pos) tuples
        """
        if not text or not text.strip():
            return []

        words = self.tagger(text)
        tokens_with_pos = [
            (word.surface, word.pos) for word in words if word.surface.strip()
        ]

        return tokens_with_pos


def tokenize_jp(text: str, preserve_punctuation: bool = True) -> str:
    """
    Tokenize Japanese text and return space-separated string.

    Args:
        text: Input Japanese text
        preserve_punctuation: Whether to preserve punctuation marks

    Returns:
        Space-separated tokenized text
    """
    tokenizer = JapaneseTokenizer()
    tokens = tokenizer.tokenize(text)

    if not preserve_punctuation:
        # Remove punctuation tokens
        tokens = [token for token in tokens if not re.match(r"^[^\w\s]+$", token)]

    return " ".join(tokens)


def is_japanese_text(text: str, threshold: float = 0.3) -> bool:
    """
    Detect if text contains Japanese characters.

    Args:
        text: Input text to check
        threshold: Minimum ratio of Japanese characters to consider as Japanese

    Returns:
        True if text is likely Japanese
    """
    if not text:
        return False

    # Japanese character ranges
    hiragana = "\u3040-\u309f"
    katakana = "\u30a0-\u30ff"
    kanji = "\u4e00-\u9faf"
    jp_chars = f"[{hiragana}{katakana}{kanji}]"

    # Count Japanese characters
    jp_char_count = len(re.findall(jp_chars, text))
    total_chars = len(text.strip())

    if total_chars == 0:
        return False

    ratio = jp_char_count / total_chars
    return ratio >= threshold
