"""Tests for package version metadata."""

from app import __version__, version_info


def test_version_string_format():
    """Version string follows semver major.minor.patch format."""
    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected 3 semver parts, got {len(parts)}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version component: {part!r}"


def test_version_info_tuple():
    """version_info is a 3-tuple of ints matching __version__."""
    assert isinstance(version_info, tuple)
    assert len(version_info) == 3
    assert all(isinstance(v, int) for v in version_info)
    assert version_info == (1, 0, 0)


def test_version_consistency():
    """__version__ and version_info agree."""
    reconstructed = ".".join(str(v) for v in version_info)
    assert reconstructed == __version__
