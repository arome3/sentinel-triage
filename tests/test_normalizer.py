"""Tests for app.normalizer module."""

import pytest

from app.normalizer import normalize_text


class TestNormalizeText:
    """Test normalize_text() transformations."""

    def test_smart_quotes_replaced(self):
        """Curly single and double quotes become ASCII equivalents."""
        assert normalize_text("‘hello’") == "'hello'"
        assert normalize_text("“hi”") == '\"hi\"'

    def test_dashes_replaced(self):
        """En-dash and em-dash become ASCII hyphens."""
        assert normalize_text("a–b") == "a-b"
        assert normalize_text("a—b") == "a-b"

    def test_ellipsis_replaced(self):
        """Horizontal ellipsis becomes three dots."""
        assert normalize_text("wait…") == "wait..."

    def test_excessive_newlines_collapsed(self):
        """Three or more consecutive newlines collapse to two."""
        assert normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_double_newlines_preserved(self):
        """Exactly two newlines (paragraph break) are kept."""
        assert normalize_text("a\n\nb") == "a\n\nb"

    def test_whitespace_stripped(self):
        """Leading and trailing whitespace is removed."""
        assert normalize_text("  hello  ") == "hello"

    def test_unicode_nfc(self):
        """Decomposed characters are composed (NFC)."""
        decomposed = "é"
        result = normalize_text(decomposed)
        assert result == "é"

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert normalize_text("") == ""

    def test_plain_ascii_unchanged(self):
        """Plain ASCII text passes through unmodified."""
        text = "Hello, world\! This is a test."
        assert normalize_text(text) == text

    def test_combined_transformations(self):
        """Multiple normalizations apply in a single pass."""
        text = "“Hello” — wait…"
        assert normalize_text(text) == '\"Hello\" - wait...'
