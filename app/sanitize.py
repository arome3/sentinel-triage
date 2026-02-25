"""Content sanitization utility for input preprocessing.

Provides a pure function to normalize and clean user-submitted content
before it enters the routing and dispatch layers. This ensures consistent
hashing, deduplication, and safe LLM inference.
"""

import re
import unicodedata

# ASCII control characters to strip (0x00-0x1F), keeping \t (0x09) and \n (0x0A)
_CONTROL_CHAR_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f]",
)

# ANSI escape sequences: ESC[ ... final_byte
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;]*[A-Za-z]",
)

# Zero-width and invisible formatting characters that can be injected
# between graphemes to evade pattern-matching without changing visual output.
_ZERO_WIDTH_RE = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u2060"  # WORD JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u00ad"  # SOFT HYPHEN
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
    "]",
)

# Three or more whitespace characters in a row
_EXCESSIVE_WHITESPACE_RE = re.compile(
    r"[ \t]{3,}",
)

#: Default maximum input length (characters).  Inputs longer than this are
#: truncated *before* any regex processing to bound worst-case CPU time.
DEFAULT_MAX_LENGTH: int = 10_000


def sanitize_content(text: str, *, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Sanitize user content for safe downstream processing.

    Applies the following transformations in order:

    1. Truncate to *max_length* characters (prevents regex DoS on huge input)
    2. Strip ANSI escape sequences
    3. Strip zero-width and invisible formatting characters
    4. Strip ASCII control characters (except tab and newline)
    5. Apply Unicode NFC normalization
    6. Collapse runs of 3+ horizontal whitespace to a single space
    7. Strip leading/trailing whitespace

    This function is pure: no side effects, no I/O.

    Args:
        text: Raw user content string.
        max_length: Maximum number of characters to retain.  Defaults to
            :data:`DEFAULT_MAX_LENGTH` (10 000).  Pass ``0`` to disable
            truncation.

    Returns:
        Sanitized content string.

    Raises:
        ValueError: If *max_length* is negative.
    """
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")

    # 0. Truncate oversized input
    if max_length and len(text) > max_length:
        text = text[:max_length]

    # 1. Remove ANSI escape sequences
    text = _ANSI_ESCAPE_RE.sub("", text)

    # 2. Remove zero-width and invisible formatting characters
    text = _ZERO_WIDTH_RE.sub("", text)

    # 3. Remove control characters (keep \t and \n)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 4. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 5. Collapse excessive horizontal whitespace
    text = _EXCESSIVE_WHITESPACE_RE.sub(" ", text)

    # 6. Strip leading/trailing whitespace
    return text.strip()
