"""
Sentinel-Triage: Semantic Router for Content Moderation

A sentiment-driven content moderation pipeline that uses semantic routing
to intelligently direct user-generated content to the most appropriate
AI model based on intent, risk level, and language.
"""

__version__ = "1.0.0"

version_info: tuple[int, int, int] = tuple(int(p) for p in __version__.split("."))  # type: ignore[assignment]

__all__ = ["__version__", "version_info"]
