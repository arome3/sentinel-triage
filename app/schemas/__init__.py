"""
Schemas module: Pydantic request/response models.

This module provides validated data models for the Sentinel-Triage API:
- Request/response models for /moderate endpoint
- Error response models for consistent error handling
- Metrics and health check response models

All schemas follow Pydantic v2 patterns with comprehensive validation,
field descriptions, and OpenAPI documentation support.

Example usage:
    from app.schemas import ModerationRequest, ModerationResponse

    # Validate incoming request
    request = ModerationRequest(content="Hello world")

    # Build response from internal components
    from app.schemas import build_moderation_response
    response = build_moderation_response(dispatch_result, route_choice, model_meta)
"""

from app.schemas.moderation import (
    # Enums
    RouteName,
    Verdict,
    # Request models
    ModerationRequest,
    RequestMetadata,
    # Response models
    ModerationResponse,
    RoutingInfo,
    TokenUsage,
    # Error models
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    # Metrics models
    MetricsResponse,
    ModelMetrics,
    RouteMetrics,
    # Health models
    ComponentHealth,
    HealthResponse,
    # Conversion utilities
    build_moderation_response,
    routing_info_from_route_choice,
    token_usage_from_dispatch,
)

# Re-export ModelTier from registry for convenience
from app.registry.models import ModelTier

__all__ = [
    # Enums
    "Verdict",
    "RouteName",
    "ModelTier",
    # Request models
    "RequestMetadata",
    "ModerationRequest",
    # Response models
    "TokenUsage",
    "RoutingInfo",
    "ModerationResponse",
    # Error models
    "ErrorCodes",
    "ErrorDetail",
    "ErrorResponse",
    # Metrics models
    "RouteMetrics",
    "ModelMetrics",
    "MetricsResponse",
    # Health models
    "ComponentHealth",
    "HealthResponse",
    # Conversion utilities
    "routing_info_from_route_choice",
    "token_usage_from_dispatch",
    "build_moderation_response",
]
