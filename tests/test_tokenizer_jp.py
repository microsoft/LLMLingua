"""
Tests for Japanese tokenizer functionality.
"""

import pytest
from llmlingua.tokenizer_jp import JapaneseTokenizer, tokenize_jp, is_japanese_text


class TestJapaneseTokenizer:
    """Test cases for JapaneseTokenizer class."""

    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = JapaneseTokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, "tagger")

    def test_tokenize_basic_japanese(self):
        """Test basic Japanese tokenization."""
        tokenizer = JapaneseTokenizer()
        text = "私は学生です。"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)

    def test_tokenize_with_pos(self):
        """Test tokenization with part-of-speech information."""
        tokenizer = JapaneseTokenizer()
        text = "美しい花が咲いています。"
        tokens_with_pos = tokenizer.tokenize_with_pos(text)

        assert isinstance(tokens_with_pos, list)
        assert len(tokens_with_pos) > 0
        assert all(
            isinstance(item, tuple) and len(item) == 2 for item in tokens_with_pos
        )

    def test_empty_text(self):
        """Test handling of empty text."""
        tokenizer = JapaneseTokenizer()

        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []
        assert tokenizer.tokenize_with_pos("") == []
        assert tokenizer.tokenize_with_pos("   ") == []


class TestTokenizeJp:
    """Test cases for tokenize_jp function."""

    def test_basic_tokenization(self):
        """Test basic tokenize_jp functionality."""
        text = "今日は良い天気ですね。"
        result = tokenize_jp(text)

        assert isinstance(result, str)
        assert len(result) > 0
        assert " " in result  # Should be space-separated

    def test_preserve_punctuation(self):
        """Test punctuation preservation."""
        text = "こんにちは！元気ですか？"

        # With punctuation
        result_with = tokenize_jp(text, preserve_punctuation=True)
        assert "！" in result_with or "?" in result_with

        # Without punctuation
        result_without = tokenize_jp(text, preserve_punctuation=False)
        assert "！" not in result_without and "?" not in result_without

    def test_mixed_text(self):
        """Test mixed Japanese and English text."""
        text = "Hello 世界！This is a test."
        result = tokenize_jp(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_input(self):
        """Test empty input handling."""
        assert tokenize_jp("") == ""
        assert tokenize_jp("   ") == ""


class TestIsJapaneseText:
    """Test cases for is_japanese_text function."""

    def test_pure_japanese(self):
        """Test pure Japanese text detection."""
        text = "日本語のテキストです。"
        assert is_japanese_text(text) is True

    def test_mixed_text(self):
        """Test mixed text detection."""
        text = "Hello 世界！This is a test."
        # This text has 2 Japanese chars out of 24 total = 0.083 ratio
        # With default threshold 0.3, this should be False
        assert is_japanese_text(text) is False

    def test_english_only(self):
        """Test English-only text."""
        text = "This is English text only."
        assert is_japanese_text(text) is False

    def test_empty_text(self):
        """Test empty text handling."""
        assert is_japanese_text("") is False
        assert is_japanese_text("   ") is False

    def test_custom_threshold(self):
        """Test custom threshold setting."""
        text = "Hello 世界"  # 2 Japanese chars, 8 total chars = 0.25 ratio

        # Default threshold (0.3) should return False
        assert is_japanese_text(text) is False

        # Lower threshold should return True
        assert is_japanese_text(text, threshold=0.2) is True

    def test_hiragana_katakana_kanji(self):
        """Test different Japanese character types."""
        hiragana = "あいうえお"
        katakana = "アイウエオ"
        kanji = "漢字"

        assert is_japanese_text(hiragana) is True
        assert is_japanese_text(katakana) is True
        assert is_japanese_text(kanji) is True


@pytest.mark.integration
class TestTokenizerIntegration:
    """Integration tests for tokenizer."""

    def test_long_text(self):
        """Test tokenization of longer text."""
        text = """
        自然言語処理（しぜんげんごしょり、英語: natural language processing、略称: NLP）は、
        人間が日常的に使っている自然言語をコンピュータに処理させる一連の技術であり、
        人工知能と言語学の一分野である。
        """

        result = tokenize_jp(text)
        assert isinstance(result, str)
        assert len(result) > 0

        # Should preserve sentence structure
        tokens = result.split()
        assert len(tokens) > 10  # Should have multiple tokens

    def test_special_characters(self):
        """Test handling of special characters."""
        text = "「引用」や（括弧）など、様々な記号を含むテキスト。"
        result = tokenize_jp(text)

        assert isinstance(result, str)
        assert len(result) > 0
