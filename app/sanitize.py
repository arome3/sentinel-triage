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

# Three or more whitespace characters in a row
_EXCESSIVE_WHITESPACE_RE = re.compile(
    r"[ \t]{3,}",
)


def sanitize_content(text: str) -> str:
    """Sanitize user content for safe downstream processing.

    Applies the following transformations in order:

    1. Strip ANSI escape sequences
    2. Strip ASCII control characters (except tab and newline)
    3. Apply Unicode NFC normalization
    4. Collapse runs of 3+ horizontal whitespace to a single space
    5. Strip leading/trailing whitespace

    This function is pure: no side effects, no I/O.

    Args:
        text: Raw user content string.

    Returns:
        Sanitized content string.
    """
    # 1. Remove ANSI escape sequences
    text = _ANSI_ESCAPE_RE.sub("", text)

    # 2. Remove control characters (keep \t and \n)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 3. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 4. Collapse excessive horizontal whitespace
    text = _EXCESSIVE_WHITESPACE_RE.sub(" ", text)

    # 5. Strip leading/trailing whitespace
    return text.strip()
