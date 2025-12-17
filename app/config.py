"""
Sentinel-Triage Configuration Module

This module manages application settings and environment variables using
pydantic-settings for type-safe configuration management.

Environment variables are loaded from .env file or system environment.
All sensitive values use SecretStr to prevent accidental logging.

See: docs/01-configuration.md for detailed documentation.
"""

from functools import lru_cache
from typing import Literal
import logging
import sys

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic Settings automatically reads from:
    1. Environment variables
    2. .env file in working directory
    3. Default values defined here

    API keys use SecretStr to prevent accidental exposure in logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars
    )

    openai_api_key: SecretStr = Field(
        ..., description="OpenAI API key for embeddings"  # Required
    )

    groq_api_key: SecretStr = Field(
        ..., description="Groq API key for Llama inference"  # Required
    )

    deepseek_api_key: SecretStr | None = Field(
        default=None, description="DeepSeek API key for Tier 2 reasoning (optional)"
    )

    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for route matching (0.0-1.0)",
    )

    default_route: str = Field(
        default="obvious_safe",
        description="Fallback route when no confident match is found",
    )

    fallback_model: str = Field(
        default="llama-3.1-8b", description="Model to use when routing fails entirely"
    )

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model for local semantic routing (ONNX Runtime)",
    )

    embedding_dimensions: int = Field(
        default=384,  # bge-small-en-v1.5 produces 384-dimensional vectors
        description="Embedding vector dimensions",
    )

    embedding_cache_dir: str | None = Field(
        default=None,
        description="Directory to cache embedding model (default: fastembed cache)",
    )

    embedding_threads: int | None = Field(
        default=None, description="CPU threads for embedding (default: auto-detect)"
    )

    track_costs: bool = Field(
        default=True, description="Enable cost calculation and logging"
    )

    gpt4o_cost_per_1m: float = Field(
        default=5.00, description="GPT-4o cost per 1M tokens (baseline for comparison)"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Application log level"
    )

    host: str = Field(default="0.0.0.0", description="Server bind host")

    port: int = Field(default=8000, description="Server bind port")

    debug: bool = Field(default=False, description="Enable debug mode")

    @field_validator("default_route")
    @classmethod
    def validate_default_route(cls, v: str) -> str:
        """Ensure default_route is one of the valid semantic routes."""
        valid_routes = {
            "obvious_harm",
            "obvious_safe",
            "ambiguous_risk",
            "system_attack",
            "non_english",
        }
        if v not in valid_routes:
            raise ValueError(f"default_route must be one of {valid_routes}")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Returns cached settings instance.

    Using lru_cache ensures settings are loaded once and reused,
    avoiding repeated file reads and environment parsing.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()


def configure_logging(settings: Settings) -> None:
    """
    Configure application logging based on settings.

    Sets up structured logging with timestamps and reduces noise
    from third-party HTTP libraries.

    Args:
        settings: The application settings instance.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)
