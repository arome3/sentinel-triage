"""Tests for the smart text truncation utility."""

import pytest

from app.text_truncator import TruncationResult, truncate_batch, truncate_text


class TestTruncateText:
    """Unit tests for truncate_text()."""

    def test_short_text_unchanged(self):
        """Text within max_length is returned as-is."""
        result = truncate_text("hello world", max_length=50)
        assert result.text == "hello world"
        assert result.was_truncated is False

    def test_exact_length_unchanged(self):
        """Text exactly at max_length is not truncated."""
        text = "a" * 200
        result = truncate_text(text, max_length=200)
        assert result.was_truncated is False
        assert result.text == text

    def test_truncation_at_word_boundary(self):
        """Long text is cut at the last word boundary."""
        text = "The quick brown fox jumps over the lazy dog and more words follow"
        result = truncate_text(text, max_length=30)
        assert result.was_truncated is True
        assert len(result.text) <= 30
        assert result.text.endswith("...")
        assert not result.text[: -len("...")].endswith(" ")

    def test_sentence_boundary_preferred(self):
        """Truncation prefers sentence boundaries when available."""
        text = "First sentence. Second sentence. Third sentence is much longer here."
        result = truncate_text(text, max_length=40)
        assert result.was_truncated is True
        assert result.text.endswith("...")
        # Should cut at a sentence boundary
        body = result.text[: -len("...")]
        assert body.endswith(".") or body.endswith("!") or body.endswith("?")

    def test_suffix_appended(self):
        """Truncated text ends with the configured suffix."""
        text = "a " * 200
        result = truncate_text(text, max_length=20, suffix=">>")
        assert result.text.endswith(">>")
        assert result.suffix_used == ">>"

    def test_custom_suffix(self):
        """Custom suffix is used instead of default."""
        text = "word " * 100
        result = truncate_text(text, max_length=20, suffix=" [more]")
        assert result.text.endswith(" [more]")

    def test_empty_string(self):
        """Empty input returns empty result."""
        result = truncate_text("")
        assert result.text == ""
        assert result.was_truncated is False
        assert result.original_length == 0

    def test_single_long_word(self):
        """A single word longer than max_length is hard-cut."""
        text = "a" * 300
        result = truncate_text(text, max_length=50)
        assert result.was_truncated is True
        assert len(result.text) == 50

    def test_original_length_populated(self):
        """original_length reflects the input character count."""
        text = "hello world"
        result = truncate_text(text, max_length=5)
        assert result.original_length == 11

    def test_return_type(self):
        """truncate_text returns a TruncationResult."""
        result = truncate_text("test")
        assert isinstance(result, TruncationResult)

    def test_immutable(self):
        """TruncationResult is frozen."""
        result = truncate_text("test")
        with pytest.raises(AttributeError):
            result.text = "changed"

    def test_max_length_too_small_raises(self):
        """max_length smaller than suffix + 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_length"):
            truncate_text("hello", max_length=3, suffix="...")

    def test_output_never_exceeds_max_length(self):
        """Output text never exceeds max_length."""
        for length in [10, 20, 50, 100]:
            text = "word " * 200
            result = truncate_text(text, max_length=length)
            assert len(result.text) <= length


class TestTruncateBatch:
    """Unit tests for truncate_batch()."""

    def test_batch_matches_individual(self):
        """Batch results match individual calls."""
        texts = ["short", "a " * 200, "medium length text here"]
        batch = truncate_batch(texts, max_length=20)
        individual = [truncate_text(t, max_length=20) for t in texts]
        assert batch == individual

    def test_empty_batch(self):
        """Empty list returns empty list."""
        assert truncate_batch([]) == []

    def test_batch_preserves_order(self):
        """Results are in the same order as inputs."""
        texts = ["first", "second", "third"]
        result = truncate_batch(texts)
        assert result[0].text == "first"
        assert result[1].text == "second"
        assert result[2].text == "third"
