"""Tests for the token estimation utility."""

from app.token_counter import (
    TIER_LONG,
    TIER_MEDIUM,
    TIER_SHORT,
    TOKENS_PER_WORD,
    TokenEstimate,
    estimate_batch_cost,
    estimate_tokens,
)


class TestEstimateTokens:
    """Unit tests for estimate_tokens()."""

    def test_basic_estimation(self):
        """Word count is multiplied by tokens-per-word ratio."""
        result = estimate_tokens("hello world foo bar")
        assert result.word_count == 4
        assert result.estimated_tokens == int(4 * TOKENS_PER_WORD + 0.5)

    def test_empty_string(self):
        """Empty input yields zero tokens."""
        result = estimate_tokens("")
        assert result.word_count == 0
        assert result.estimated_tokens == 0
        assert result.cost_tier == "short"

    def test_single_word(self):
        """Single word estimation is correct."""
        result = estimate_tokens("hello")
        assert result.word_count == 1
        assert result.estimated_tokens == int(1 * TOKENS_PER_WORD + 0.5)

    def test_text_length_populated(self):
        """text_length reflects the raw character count."""
        text = "hello world"
        result = estimate_tokens(text)
        assert result.text_length == len(text)

    def test_short_tier(self):
        """Text with few tokens is classified as short."""
        result = estimate_tokens("hello world")
        assert result.cost_tier == "short"

    def test_medium_tier(self):
        """Text crossing the short threshold is medium."""
        words = " ".join(["word"] * 100)
        result = estimate_tokens(words)
        assert result.cost_tier == "medium"

    def test_long_tier(self):
        """Text crossing the medium threshold is long."""
        words = " ".join(["word"] * 500)
        result = estimate_tokens(words)
        assert result.cost_tier == "long"

    def test_very_long_tier(self):
        """Text crossing the long threshold is very_long."""
        words = " ".join(["word"] * 2000)
        result = estimate_tokens(words)
        assert result.cost_tier == "very_long"

    def test_return_type(self):
        """estimate_tokens returns a TokenEstimate instance."""
        result = estimate_tokens("test")
        assert isinstance(result, TokenEstimate)

    def test_immutable(self):
        """TokenEstimate is frozen."""
        result = estimate_tokens("test")
        try:
            result.estimated_tokens = 999
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_whitespace_only(self):
        """Whitespace-only text has zero words."""
        result = estimate_tokens("   \n\t  ")
        assert result.word_count == 0
        assert result.estimated_tokens == 0

    def test_constants_are_ordered(self):
        """Tier thresholds are in ascending order."""
        assert TIER_SHORT < TIER_MEDIUM < TIER_LONG


class TestEstimateBatchCost:
    """Unit tests for estimate_batch_cost()."""

    def test_empty_batch(self):
        """Empty list returns zeroed summary."""
        result = estimate_batch_cost([])
        assert result["total_tokens"] == 0
        assert result["count"] == 0

    def test_single_text(self):
        """Single-item batch has matching total and average."""
        result = estimate_batch_cost(["hello world"])
        assert result["count"] == 1
        assert result["total_tokens"] == result["avg_tokens"]

    def test_tier_breakdown(self):
        """Batch cost includes per-tier counts."""
        texts = ["hi", " ".join(["word"] * 200)]
        result = estimate_batch_cost(texts)
        assert (
            result["short"] + result["medium"] + result["long"] + result["very_long"]
            == 2
        )

    def test_total_is_sum(self):
        """Total tokens equals sum of individual estimates."""
        texts = ["one two three", "four five six seven"]
        result = estimate_batch_cost(texts)
        individual_sum = sum(estimate_tokens(t).estimated_tokens for t in texts)
        assert result["total_tokens"] == individual_sum
