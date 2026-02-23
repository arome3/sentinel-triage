"""Smart text truncation for display contexts.

Provides word-boundary-aware truncation that avoids cutting text mid-word.
Prefers sentence boundaries when possible and returns metadata about the
truncation result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """Immutable truncation result with metadata."""

    text: str
    original_length: int
    was_truncated: bool
    suffix_used: str


def truncate_text(
    text: str,
    max_length: int = 200,
    suffix: str = "...",
) -> TruncationResult:
    """Truncate text at word boundaries with optional sentence preference.

    If the text is already within ``max_length``, it is returned unchanged.
    Otherwise the text is cut at the last word boundary that fits within
    ``max_length - len(suffix)`` and the suffix is appended.

    When a sentence boundary (period, question mark, or exclamation mark
    followed by a space) exists within the allowed range, the text is cut
    there instead to produce more natural output.

    Args:
        text: Input text to truncate.
        max_length: Maximum length of the output (including suffix).
        suffix: String to append when truncation occurs.

    Returns:
        A :class:`TruncationResult` with the output and metadata.

    Raises:
        ValueError: If ``max_length`` is less than ``len(suffix) + 1``.
    """
    if max_length < len(suffix) + 1:
        raise ValueError(
            f"max_length ({max_length}) must be at least len(suffix) + 1 ({len(suffix) + 1})"
        )

    original_length = len(text)

    if original_length <= max_length:
        return TruncationResult(
            text=text,
            original_length=original_length,
            was_truncated=False,
            suffix_used="",
        )

    budget = max_length - len(suffix)
    candidate = text[:budget]

    # Try sentence boundary first
    for punct in ".!?":
        idx = candidate.rfind(punct)
        if idx > 0 and idx < budget - 1:
            # Check that it's followed by a space in the original text
            next_pos = idx + 1
            if next_pos < original_length and text[next_pos] == " ":
                return TruncationResult(
                    text=text[: idx + 1] + suffix,
                    original_length=original_length,
                    was_truncated=True,
                    suffix_used=suffix,
                )

    # Fall back to word boundary
    space_idx = candidate.rfind(" ")
    if space_idx > 0:
        return TruncationResult(
            text=text[:space_idx] + suffix,
            original_length=original_length,
            was_truncated=True,
            suffix_used=suffix,
        )

    # No word boundary found — hard cut
    return TruncationResult(
        text=candidate + suffix,
        original_length=original_length,
        was_truncated=True,
        suffix_used=suffix,
    )


def truncate_batch(
    texts: list[str],
    max_length: int = 200,
    suffix: str = "...",
) -> list[TruncationResult]:
    """Truncate a batch of texts with the same settings.

    Args:
        texts: List of input texts.
        max_length: Maximum length per text.
        suffix: Suffix for truncated texts.

    Returns:
        List of :class:`TruncationResult` in the same order as inputs.
    """
    return [truncate_text(t, max_length=max_length, suffix=suffix) for t in texts]
