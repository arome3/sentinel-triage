"""Content statistics utility for routing decisions.

Computes lightweight text metrics (word count, line count, character count,
average word length) that the semantic router can use to select appropriate
models and cost tiers without invoking embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContentStats:
    """Immutable container for content metrics."""

    char_count: int
    word_count: int
    line_count: int
    avg_word_length: float


def compute_stats(text: str) -> ContentStats:
    """Compute basic statistics for *text*.

    Args:
        text: Input text (may be empty).

    Returns:
        A :class:`ContentStats` instance with computed metrics.
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    line_count = text.count("\n") + 1 if text else 0
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0.0
    return ContentStats(
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        avg_word_length=round(avg_word_length, 2),
    )
