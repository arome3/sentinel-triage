"""
Registry module: Model pool configuration and metadata.

This module contains:
- models.py: 4-tier model registry with cost, latency, and capability metadata

Public API:
- ModelTier: Enum for model tier classification
- ModelProvider: Enum for inference providers
- ModelCapability: Enum for model capabilities
- ModelMetadata: Pydantic model for model configuration
- ModelRegistry: Central registry class
- get_model_registry: Singleton accessor function
"""

from app.registry.models import (
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelRegistry,
    ModelTier,
    get_model_registry,
)

__all__ = [
    "ModelTier",
    "ModelProvider",
    "ModelCapability",
    "ModelMetadata",
    "ModelRegistry",
    "get_model_registry",
]
