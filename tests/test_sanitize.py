"""Tests for app.sanitize — content sanitization utilities."""

from __future__ import annotations

import unicodedata

from app.sanitize import sanitize_content


class TestSanitizeContent:
    """Unit tests for sanitize_content()."""

    # ── Empty / trivial input ───────────────────────────────────────

    def test_empty_string(self):
        assert sanitize_content("") == ""

    def test_whitespace_only(self):
        assert sanitize_content("   ") == ""

    def test_plain_text_unchanged(self):
        assert sanitize_content("hello world") == "hello world"

    # ── Control character stripping ─────────────────────────────────

    def test_null_byte_stripped(self):
        assert sanitize_content("hello\x00world") == "helloworld"

    def test_bell_and_backspace_stripped(self):
        assert sanitize_content("a\x07b\x08c") == "abc"

    def test_del_character_stripped(self):
        assert sanitize_content("before\x7fafter") == "beforeafter"

    def test_tab_preserved(self):
        result = sanitize_content("col1\tcol2")
        assert "col1" in result
        assert "col2" in result

    def test_newline_preserved(self):
        result = sanitize_content("line1\nline2")
        assert "line1" in result
        assert "line2" in result
        assert "\n" in result

    def test_carriage_return_preserved(self):
        # CR is \x0d — should NOT be stripped
        result = sanitize_content("line1\r\nline2")
        assert "line1" in result
        assert "line2" in result

    # ── NFC normalization ───────────────────────────────────────────

    def test_nfc_normalization_composed(self):
        # e + combining acute accent -> single composed char
        decomposed = "e\u0301"  # NFD form
        result = sanitize_content(decomposed)
        assert result == "\u00e9"  # NFC composed form
        assert unicodedata.is_normalized("NFC", result)

    def test_already_nfc_unchanged(self):
        composed = "\u00e9"  # already NFC
        assert sanitize_content(composed) == composed

    def test_hangul_normalization(self):
        # Hangul syllable composition
        decomposed = "\u1100\u1161"  # ᄀ + ᅡ
        result = sanitize_content(decomposed)
        assert unicodedata.is_normalized("NFC", result)

    # ── Whitespace collapsing ───────────────────────────────────────

    def test_multiple_spaces_collapsed(self):
        assert sanitize_content("too   many   spaces") == "too many spaces"

    def test_leading_trailing_stripped(self):
        assert sanitize_content("  hello  ") == "hello"

    def test_tabs_collapsed_to_single_space(self):
        assert sanitize_content("a\t\tb") == "a b"

    def test_mixed_whitespace_collapsed(self):
        assert sanitize_content("a \t  b") == "a b"

    def test_double_newline_preserved(self):
        result = sanitize_content("para1\n\npara2")
        assert "para1\n\npara2" == result

    def test_triple_newline_collapsed(self):
        result = sanitize_content("para1\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_many_newlines_collapsed(self):
        result = sanitize_content("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    # ── Combined transformations ────────────────────────────────────

    def test_control_chars_and_whitespace(self):
        result = sanitize_content("  \x00hello\x07   world\x7f  ")
        assert result == "hello world"

    def test_full_pipeline(self):
        # Control chars + decomposed unicode + extra whitespace
        text = "  \x00he\u0301llo\x07   world\x7f  "
        result = sanitize_content(text)
        assert "h\u00e9llo" in result
        assert "world" in result
        assert "\x00" not in result
        assert "\x07" not in result

    def test_multiline_document(self):
        doc = """Title\n\n\n\n\nBody   with   spaces\n\nFooter"""
        result = sanitize_content(doc)
        assert result == "Title\n\nBody with spaces\n\nFooter"

    # ── Edge cases ──────────────────────────────────────────────────

    def test_only_control_characters(self):
        assert sanitize_content("\x00\x01\x02\x03") == ""

    def test_unicode_emoji_preserved(self):
        assert sanitize_content("hello 🌍 world") == "hello 🌍 world"

    def test_cjk_text_preserved(self):
        assert sanitize_content("你好世界") == "你好世界"

    def test_rtl_text_preserved(self):
        assert sanitize_content("مرحبا بالعالم") == "مرحبا بالعالم"

    def test_very_long_input(self):
        text = "word " * 10000
        result = sanitize_content(text)
        assert len(result) > 0
        assert "  " not in result
