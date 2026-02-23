"""Tests for the content fingerprinting utility."""

import hashlib

from app.fingerprint import content_fingerprint


class TestContentFingerprint:
    """Unit tests for content_fingerprint()."""

    def test_basic_hashing(self):
        """Fingerprint returns a 64-char hex SHA-256 digest."""
        result = content_fingerprint("hello world")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        """Same input always produces the same fingerprint."""
        fp1 = content_fingerprint("test content")
        fp2 = content_fingerprint("test content")
        assert fp1 == fp2

    def test_case_insensitive(self):
        """Fingerprints are case-insensitive."""
        assert content_fingerprint("Hello World") == content_fingerprint("hello world")

    def test_whitespace_equivalence(self):
        """Excessive whitespace is collapsed before hashing."""
        assert content_fingerprint("hello     world") == content_fingerprint(
            "hello world"
        )

    def test_smart_quote_equivalence(self):
        """Smart quotes normalize to ASCII before hashing."""
        assert content_fingerprint("\u201chello\u201d") == content_fingerprint(
            '"hello"'
        )

    def test_dash_equivalence(self):
        """En-dash and em-dash normalize to ASCII hyphen."""
        assert content_fingerprint("a\u2013b") == content_fingerprint("a-b")
        assert content_fingerprint("a\u2014b") == content_fingerprint("a-b")

    def test_control_char_stripped(self):
        """Control characters are stripped before hashing."""
        assert content_fingerprint("hello\x00world") == content_fingerprint(
            "helloworld"
        )

    def test_empty_input(self):
        """Empty string produces a valid fingerprint (SHA-256 of empty)."""
        result = content_fingerprint("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_unicode_stability(self):
        """NFC normalization ensures Unicode stability."""
        # e + combining acute vs precomposed e-acute
        decomposed = "caf\u0065\u0301"
        precomposed = "caf\u00e9"
        assert content_fingerprint(decomposed) == content_fingerprint(precomposed)

    def test_different_content_different_fingerprint(self):
        """Different content produces different fingerprints."""
        assert content_fingerprint("alpha") != content_fingerprint("beta")
