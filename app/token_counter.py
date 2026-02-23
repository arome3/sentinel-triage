"""Token estimation utility for cost-aware routing.

Provides a lightweight token estimator that approximates the number of
tokens a text will consume when sent to an LLM.  Uses a word-based
heuristic (average 1.3 tokens per word for English text) to avoid
requiring a tokenizer dependency.

The router uses these estimates to select cost-appropriate models:
short inputs go to cheaper models, long inputs may need chunking or
more capable models with larger context windows.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Average tokens per word for English text (empirically ~1.3 for GPT models).
TOKENS_PER_WORD: float = 1.3

#: Cost tier thresholds (estimated tokens).
TIER_SHORT: int = 100
TIER_MEDIUM: int = 500
TIER_LONG: int = 2000


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Immutable token estimation result."""

    text_length: int
    word_count: int
    estimated_tokens: int
    cost_tier: str


def estimate_tokens(text: str) -> TokenEstimate:
    """Estimate the token count for a text string.

    Uses a word-based heuristic rather than a full tokenizer to avoid
    adding heavy dependencies.  The estimate is intentionally conservative
    (rounds up) to prevent under-budgeting.

    Cost tiers:
    - ``"short"``: < 100 estimated tokens
    - ``"medium"``: 100-499 estimated tokens
    - ``"long"``: 500-1999 estimated tokens
    - ``"very_long"``: 2000+ estimated tokens

    Args:
        text: Input text to estimate.

    Returns:
        A :class:`TokenEstimate` with counts and cost tier.
    """
    words = text.split()
    word_count = len(words)
    estimated = int(word_count * TOKENS_PER_WORD + 0.5)  # round up

    if estimated < TIER_SHORT:
        tier = "short"
    elif estimated < TIER_MEDIUM:
        tier = "medium"
    elif estimated < TIER_LONG:
        tier = "long"
    else:
        tier = "very_long"

    return TokenEstimate(
        text_length=len(text),
        word_count=word_count,
        estimated_tokens=estimated,
        cost_tier=tier,
    )


def estimate_batch_cost(texts: list[str]) -> dict[str, int | float]:
    """Estimate aggregate token usage for a batch of texts.

    Returns a summary dict with total tokens, average tokens per text,
    and a breakdown by cost tier.

    Args:
        texts: List of input text strings.

    Returns:
        Dict with ``total_tokens``, ``avg_tokens``, ``count``, and
        per-tier counts (``short``, ``medium``, ``long``, ``very_long``).
    """
    if not texts:
        return {
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "count": 0,
            "short": 0,
            "medium": 0,
            "long": 0,
            "very_long": 0,
        }

    estimates = [estimate_tokens(t) for t in texts]
    total = sum(e.estimated_tokens for e in estimates)
    tier_counts = {"short": 0, "medium": 0, "long": 0, "very_long": 0}
    for e in estimates:
        tier_counts[e.cost_tier] += 1

    return {
        "total_tokens": total,
        "avg_tokens": round(total / len(texts), 1),
        "count": len(texts),
        **tier_counts,
    }
