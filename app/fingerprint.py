"""Content fingerprinting utility for deduplication.

Chains sanitization and normalization to produce a deterministic SHA-256
fingerprint for any input text.  Semantically equivalent content (differing
only in whitespace, casing, smart quotes, or control characters) maps to
the same fingerprint.
"""

from __future__ import annotations

import hashlib

from app.normalizer import normalize_text
from app.sanitize import sanitize_content


def content_fingerprint(text: str) -> str:
    """Produce a deterministic SHA-256 hex fingerprint for *text*.

    Processing pipeline:
    1. ``sanitize_content()`` -- strip control chars, ANSI escapes, truncate
    2. ``normalize_text()`` -- smart quotes, dashes, ellipsis, newlines
    3. Lowercase -- case-insensitive matching
    4. SHA-256 hex digest

    Args:
        text: Raw user content.

    Returns:
        64-character lowercase hex string (SHA-256 digest).
    """
    cleaned = sanitize_content(text)
    normalized = normalize_text(cleaned)
    lowered = normalized.lower()
    return hashlib.sha256(lowered.encode("utf-8")).hexdigest()


def batch_fingerprint(texts: list[str]) -> list[str]:
    """Produce fingerprints for multiple texts in one call.

    Applies :func:`content_fingerprint` to each element and returns
    fingerprints in the same order as the input list.

    Args:
        texts: List of raw user content strings.

    Returns:
        List of 64-character lowercase hex strings (SHA-256 digests).
    """
    return [content_fingerprint(t) for t in texts]
