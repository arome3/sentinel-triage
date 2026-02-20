"""Content sanitization utilities for input preprocessing.

Provides pure functions to normalise and clean user-supplied text before
it enters the moderation pipeline.  All functions are side-effect free
and perform no I/O.
"""

from __future__ import annotations

import re
import unicodedata

# ASCII control characters to strip (C0 range), keeping common whitespace.
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
)

# Two or more consecutive whitespace characters (space, tab, etc.).
_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")

# Three or more consecutive newlines.
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def sanitize_content(text: str) -> str:
    """Sanitize user-supplied text for safe downstream processing.

    The function applies three transformations in order:

    1. **Strip control characters** — removes ASCII C0 controls (``\x00``
       through ``\x1f``) except horizontal tab (``\t``), newline (``\n``),
       and carriage return (``\r``).  Also removes ``\x7f`` (DEL).
    2. **NFC normalization** — converts Unicode to Canonical Composition
       form so that visually identical characters compare equal.
    3. **Collapse whitespace** — replaces runs of spaces/tabs with a
       single space, and runs of 3+ newlines with two newlines.  Leading
       and trailing whitespace is stripped.

    Parameters
    ----------
    text:
        Raw user input.  May contain arbitrary Unicode.

    Returns
    -------
    str
        Cleaned text ready for the moderation pipeline.

    Examples
    --------
    >>> sanitize_content("hello\x00world")
    'helloworld'
    >>> sanitize_content("  too   many   spaces  ")
    'too many spaces'
    """
    if not text:
        return text

    # 1. Strip control characters
    result = _CONTROL_CHAR_RE.sub("", text)

    # 2. NFC normalization
    result = unicodedata.normalize("NFC", result)

    # 3. Collapse whitespace
    result = _MULTI_SPACE_RE.sub(" ", result)
    result = _MULTI_NEWLINE_RE.sub("\n\n", result)

    return result.strip()
