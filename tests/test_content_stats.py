"""Tests for the content statistics utility."""

from app.content_stats import ContentStats, compute_stats


class TestComputeStats:
    """Unit tests for compute_stats()."""

    def test_basic_sentence(self):
        """Computes correct metrics for a simple sentence."""
        stats = compute_stats("hello world")
        assert stats.char_count == 11
        assert stats.word_count == 2
        assert stats.line_count == 1
        assert stats.avg_word_length == 5.0

    def test_multiline(self):
        """Line count reflects newlines."""
        stats = compute_stats("line one\nline two\nline three")
        assert stats.line_count == 3
        assert stats.word_count == 6

    def test_empty_string(self):
        """Empty input returns zeroed stats."""
        stats = compute_stats("")
        assert stats.char_count == 0
        assert stats.word_count == 0
        assert stats.line_count == 0
        assert stats.avg_word_length == 0.0

    def test_single_word(self):
        """Single word has avg_word_length equal to its length."""
        stats = compute_stats("python")
        assert stats.word_count == 1
        assert stats.avg_word_length == 6.0

    def test_whitespace_only(self):
        """Whitespace-only input has zero words."""
        stats = compute_stats("   \n  \t  ")
        assert stats.word_count == 0
        assert stats.avg_word_length == 0.0
        assert stats.char_count == 9

    def test_avg_word_length_rounding(self):
        """Average word length is rounded to 2 decimal places."""
        stats = compute_stats("hi hey hello")
        # lengths: 2, 3, 5 -> avg = 10/3 = 3.33
        assert stats.avg_word_length == 3.33

    def test_frozen_dataclass(self):
        """ContentStats is immutable."""
        stats = compute_stats("test")
        try:
            stats.char_count = 999
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_return_type(self):
        """compute_stats returns a ContentStats instance."""
        stats = compute_stats("hello")
        assert isinstance(stats, ContentStats)
