"""Tests for zero-width character stripping in the sanitizer.

Zero-width characters are a known evasion technique against content
moderation systems. These tests verify that invisible characters are
stripped before content reaches the routing and dispatch layers.
"""

from app.sanitize import sanitize_content


class TestZeroWidthStripping:
    """Verify that zero-width characters are removed from input."""

    def test_zwsp_between_letters(self):
        """ZWSP (U+200B) injected between characters should be stripped."""
        # "k\u200bi\u200bl\u200bl" renders as "kill" visually
        evasion = "k\u200bi\u200bl\u200bl"
        assert sanitize_content(evasion) == "kill"

    def test_zwnj_in_harmful_word(self):
        """ZWNJ (U+200C) should be stripped."""
        assert sanitize_content("ha\u200cte") == "hate"

    def test_zwj_in_harmful_word(self):
        """ZWJ (U+200D) should be stripped."""
        assert sanitize_content("th\u200dreat") == "threat"

    def test_bom_stripped(self):
        """BOM / ZWNBSP (U+FEFF) at start or middle should be stripped."""
        assert sanitize_content("\ufeffhello") == "hello"
        assert sanitize_content("he\ufeffllo") == "hello"

    def test_soft_hyphen_stripped(self):
        """Soft hyphen (U+00AD) should be stripped."""
        assert sanitize_content("dan\u00adger") == "danger"

    def test_word_joiner_stripped(self):
        """Word joiner (U+2060) should be stripped."""
        assert sanitize_content("sp\u2060am") == "spam"

    def test_bidi_override_stripped(self):
        """Bidirectional override characters should be stripped."""
        # RTL override can visually reorder text to disguise content
        text = "\u202ehello\u202c"
        assert sanitize_content(text) == "hello"

    def test_multiple_zero_width_types(self):
        """Multiple different zero-width chars in one string."""
        evasion = "\ufeffk\u200bi\u200cl\u200dl\u2060!"
        assert sanitize_content(evasion) == "kill!"

    def test_normal_text_unchanged(self):
        """Normal text without zero-width chars passes through."""
        normal = "This is perfectly normal text."
        assert sanitize_content(normal) == normal

    def test_empty_after_stripping(self):
        """String of only zero-width characters becomes empty."""
        invisible = "\u200b\u200c\u200d\ufeff"
        assert sanitize_content(invisible) == ""

    def test_ltr_rtl_marks_stripped(self):
        """LTR/RTL marks (U+200E, U+200F) should be stripped."""
        assert sanitize_content("test\u200etext") == "testtext"
        assert sanitize_content("test\u200ftext") == "testtext"

    def test_directional_isolates_stripped(self):
        """Directional isolate characters (U+2066-U+2069) should be stripped."""
        text = "\u2066hidden\u2069"
        assert sanitize_content(text) == "hidden"

    def test_mixed_with_normal_whitespace(self):
        """Zero-width chars mixed with normal spaces are handled correctly."""
        text = "hello \u200b world"
        assert sanitize_content(text) == "hello  world"
