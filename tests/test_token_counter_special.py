"""Tests for special-character-aware token estimation.

Verifies that the token counter correctly accounts for punctuation, URLs,
and code snippets that inflate actual token counts beyond the naive
word-based heuristic.
"""

from app.token_counter import estimate_tokens, _count_special_chars


class TestSpecialCharCounting:
    """Verify special character detection."""

    def test_plain_text_no_special(self):
        assert _count_special_chars("hello world") == 0

    def test_punctuation_counted(self):
        assert _count_special_chars("hello, world!") == 2

    def test_url_special_chars(self):
        url = "https://example.com/path?q=1&lang=en"
        # :, /, /, ., /, ?, =, &, =
        count = _count_special_chars(url)
        assert count >= 8

    def test_code_snippet(self):
        code = "def foo(x): return x['bar']"
        count = _count_special_chars(code)
        assert count >= 6  # (, ), :, [, ', ', ]


class TestTokenEstimationWithSpecialChars:
    """Verify token estimates account for special characters."""

    def test_url_tokens_higher_than_word_count(self):
        """A URL is 1 word but should estimate many more tokens."""
        result = estimate_tokens("https://example.com/path?query=value&other=1")
        assert result.word_count == 1
        assert result.estimated_tokens > 3  # way more than 1 word * 1.3

    def test_code_tokens_higher_than_word_count(self):
        """Code with brackets and operators needs more tokens."""
        code = "if (x > 0) { return arr[i]; }"
        result = estimate_tokens(code)
        assert result.estimated_tokens > result.word_count

    def test_plain_text_minimally_affected(self):
        """Plain English text should see minimal change."""
        result = estimate_tokens("The quick brown fox jumps over the lazy dog")
        # 9 words * 1.3 = 11.7 -> 12 tokens (no special chars)
        assert result.estimated_tokens == 12

    def test_empty_string(self):
        result = estimate_tokens("")
        assert result.estimated_tokens == 0
        assert result.cost_tier == "short"

    def test_json_payload(self):
        """JSON payloads have many special chars."""
        json_text = '{"name": "test", "values": [1, 2, 3]}'
        result = estimate_tokens(json_text)
        assert result.estimated_tokens > 5
