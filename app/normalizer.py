"""Text normalization utility for the preprocessing pipeline.

Provides deterministic text normalization that improves semantic router
matching consistency by standardizing common textual variations.
Unlike ``sanitize_content`` (which strips harmful characters), this module
focuses on *semantic equivalence*: mapping variant spellings and encodings
to a single canonical form.
"""

import re
import unicodedata

# Curly/smart quotes -> straight ASCII quotes
_SMART_QUOTES = str.maketrans(
    "‘’“”",  # ‘ ’ “ ”
    "''\"\"",  # ' ' " "
)

# En/em dashes -> ASCII hyphen
_DASHES = str.maketrans(
    "–—",  # – —
    "--",
)

# Horizontal ellipsis -> three dots
_ELLIPSIS_RE = re.compile("…")

# Multiple consecutive newlines -> double newline
_MULTI_NEWLINE_RE = re.compile(r"
{3,}")


def normalize_text(text: str) -> str:
    """Normalize text for consistent semantic matching.

    Applies the following transformations in order:

    1. Unicode NFC normalization
    2. Replace smart/curly quotes with ASCII equivalents
    3. Replace en-dash and em-dash with ASCII hyphen
    4. Replace horizontal ellipsis (U+2026) with three dots
    5. Collapse runs of 3+ newlines to a double newline
    6. Strip leading/trailing whitespace

    This function is pure: no side effects, no I/O.

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Smart quotes -> ASCII
    text = text.translate(_SMART_QUOTES)

    # 3. Dashes -> ASCII hyphen
    text = text.translate(_DASHES)

    # 4. Ellipsis -> three dots
    text = _ELLIPSIS_RE.sub("...", text)

    # 5. Collapse excessive newlines
    text = _MULTI_NEWLINE_RE.sub("

", text)

    # 6. Strip outer whitespace
    return text.strip()
