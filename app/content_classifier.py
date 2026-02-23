"""Content classification utility for routing decisions.

Provides rule-based content classification using keyword matching and
text statistics.  Produces a classification label and confidence score
that the semantic router can use to select the appropriate dispatch
handler without invoking an embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from app.content_stats import compute_stats
from app.sanitize import sanitize_content

# Pattern sets for classification rules
_URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(urgent|emergency|critical|asap|immediately)\b", re.IGNORECASE),
    re.compile(r"\b(help me now|right away|time.?sensitive)\b", re.IGNORECASE),
]

_HARMFUL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(threat|harm|abuse|attack|violence)\b", re.IGNORECASE),
    re.compile(r"\b(illegal|exploit|hack|malicious)\b", re.IGNORECASE),
]

_SPAM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(buy now|click here|free money|act now)\b", re.IGNORECASE),
    re.compile(r"\b(limited offer|congratulations.*won|earn \$)\b", re.IGNORECASE),
    re.compile(r"(.)\1{10,}"),  # repeated characters (aaaaaaaaaa...)
]


@dataclass(frozen=True, slots=True)
class Classification:
    """Immutable classification result."""

    label: str
    confidence: float
    matched_category: str
    word_count: int

    #: Known classification labels.
    LABELS: ClassVar[tuple[str, ...]] = (
        "urgent",
        "harmful",
        "spam",
        "standard",
    )


def _count_pattern_matches(text: str, patterns: list[re.Pattern[str]]) -> int:
    """Count how many patterns match anywhere in text."""
    return sum(1 for p in patterns if p.search(text))


def classify_content(text: str) -> Classification:
    """Classify content using rule-based pattern matching.

    Applies sanitization first, then checks patterns in priority order:
    urgent > harmful > spam > standard.

    Confidence is based on the ratio of matched patterns to total patterns
    in the winning category, with a minimum floor of 0.5 for any match.

    Args:
        text: Raw user content.

    Returns:
        A :class:`Classification` with label, confidence, and metadata.
    """
    cleaned = sanitize_content(text)
    stats = compute_stats(cleaned)

    # Check categories in priority order
    categories: list[tuple[str, list[re.Pattern[str]]]] = [
        ("urgent", _URGENCY_PATTERNS),
        ("harmful", _HARMFUL_PATTERNS),
        ("spam", _SPAM_PATTERNS),
    ]

    for label, patterns in categories:
        matches = _count_pattern_matches(cleaned, patterns)
        if matches > 0:
            raw_confidence = matches / len(patterns)
            confidence = max(0.5, min(1.0, raw_confidence))
            return Classification(
                label=label,
                confidence=round(confidence, 2),
                matched_category=label,
                word_count=stats.word_count,
            )

    # Default: standard content
    return Classification(
        label="standard",
        confidence=1.0,
        matched_category="none",
        word_count=stats.word_count,
    )
