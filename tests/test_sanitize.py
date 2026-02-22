"""Tests for app.sanitize module."""

import unicodedata

import pytest
from app.sanitize import DEFAULT_MAX_LENGTH, sanitize_content


class TestSanitizeContent:
    """Tests for sanitize_content()."""

    def test_plain_text_unchanged(self):
        """Normal text passes through without modification."""
        text = "Hello, world! This is a normal sentence."
        assert sanitize_content(text) == text

    def test_empty_input(self):
        """Empty string returns empty string."""
        assert sanitize_content("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input returns empty string after strip."""
        assert sanitize_content("   ") == ""

    def test_strips_null_bytes(self):
        """Null bytes are removed."""
        assert sanitize_content("hello\x00world") == "helloworld"

    def test_strips_control_characters(self):
        """ASCII control chars 0x01-0x08, 0x0B, 0x0C, 0x0E-0x1F are removed."""
        text = "a\x01b\x02c\x07d\x0be\x0cf\x1fg"
        assert sanitize_content(text) == "abcdefg"

    def test_preserves_tab(self):
        """Tab characters (0x09) are preserved."""
        assert sanitize_content("a\tb") == "a\tb"

    def test_preserves_newline(self):
        """Newline characters (0x0A) are preserved."""
        assert sanitize_content("line1\nline2") == "line1\nline2"

    def test_strips_ansi_color(self):
        """ANSI color escape sequences are removed."""
        text = "\x1b[31mERROR\x1b[0m: something failed"
        assert sanitize_content(text) == "ERROR: something failed"

    def test_strips_ansi_cursor(self):
        """ANSI cursor movement sequences are removed."""
        text = "before\x1b[2Aafter"
        assert sanitize_content(text) == "beforeafter"

    def test_strips_ansi_complex(self):
        """Complex ANSI sequences with multiple parameters are removed."""
        text = "\x1b[1;32;40mBOLD GREEN\x1b[0m normal"
        assert sanitize_content(text) == "BOLD GREEN normal"

    def test_unicode_nfc_normalization(self):
        """Decomposed Unicode is normalized to NFC."""
        # e + combining acute accent -> e-acute (NFC)
        decomposed = "e\u0301"  # NFD form
        composed = "\u00e9"  # NFC form
        assert sanitize_content(decomposed) == composed

    def test_unicode_already_nfc(self):
        """Already-NFC text is unchanged."""
        text = "caf\u00e9"
        result = sanitize_content(text)
        assert result == text
        assert unicodedata.is_normalized("NFC", result)

    def test_collapses_excessive_spaces(self):
        """Three or more spaces collapse to one."""
        assert sanitize_content("a     b") == "a b"

    def test_two_spaces_preserved(self):
        """Two spaces are NOT collapsed (threshold is 3+)."""
        assert sanitize_content("a  b") == "a  b"

    def test_collapses_excessive_tabs(self):
        """Three or more tabs collapse to one space."""
        assert sanitize_content("a\t\t\tb") == "a b"

    def test_collapses_mixed_whitespace(self):
        """Mixed spaces and tabs collapse when total >= 3."""
        assert sanitize_content("a \t b") == "a b"

    def test_preserves_newlines_in_collapse(self):
        """Newlines are not part of the horizontal whitespace collapse."""
        text = "line1\n\n\nline2"
        assert sanitize_content(text) == "line1\n\n\nline2"

    def test_combined_sanitization(self):
        """All transformations applied together."""
        text = "\x1b[31m\x00Hello\x07     e\u0301\x1b[0m"
        result = sanitize_content(text)
        assert result == "Hello \u00e9"
        assert "\x00" not in result
        assert "\x07" not in result
        assert "\x1b" not in result

    def test_strips_leading_trailing_whitespace(self):
        """Leading and trailing whitespace is stripped."""
        assert sanitize_content("  hello  ") == "hello"

    @pytest.mark.parametrize(
        "input_text",
        [
            "Simple ASCII",
            "Unicode: \u4f60\u597d\u4e16\u754c",
            "Emoji: \U0001f600\U0001f680",
            "Math: \u222b f(x) dx",
        ],
    )
    def test_international_text_preserved(self, input_text):
        """Non-ASCII Unicode text passes through correctly."""
        result = sanitize_content(input_text)
        assert result == input_text

    def test_is_pure_function(self):
        """Calling twice with same input yields same output."""
        text = "\x1b[31m\x00test\x07   data"
        assert sanitize_content(text) == sanitize_content(text)


class TestMaxLength:
    """Tests for the max_length truncation parameter."""

    def test_default_max_length_constant(self):
        """DEFAULT_MAX_LENGTH is 10_000."""
        assert DEFAULT_MAX_LENGTH == 10_000

    def test_short_input_not_truncated(self):
        """Input shorter than max_length passes through fully."""
        text = "short"
        assert sanitize_content(text) == text

    def test_exact_max_length_not_truncated(self):
        """Input exactly at max_length is not truncated."""
        text = "a" * 100
        assert sanitize_content(text, max_length=100) == text

    def test_oversized_input_truncated(self):
        """Input exceeding max_length is truncated."""
        text = "a" * 200
        result = sanitize_content(text, max_length=50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_truncation_before_sanitization(self):
        """Truncation happens before other processing steps."""
        # 5 chars of content + control char at position 6
        text = "hello\x00world"
        result = sanitize_content(text, max_length=5)
        assert result == "hello"

    def test_max_length_zero_disables_truncation(self):
        """max_length=0 disables truncation entirely."""
        text = "a" * 20_000
        result = sanitize_content(text, max_length=0)
        assert len(result) == 20_000

    def test_negative_max_length_raises(self):
        """Negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be >= 0"):
            sanitize_content("test", max_length=-1)

    def test_default_truncates_at_10000(self):
        """Default max_length truncates at 10_000 characters."""
        text = "x" * 15_000
        result = sanitize_content(text)
        assert len(result) == 10_000
