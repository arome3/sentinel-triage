"""
Pydantic Schemas for Moderation API

This module defines the request and response models for the Sentinel-Triage API:
- ModerationRequest: Input content with optional metadata
- ModerationResponse: Verdict, confidence, reasoning, route, model, metrics
- Error responses, metrics, and health check schemas

All schemas follow Pydantic v2 patterns with comprehensive validation,
field descriptions, and OpenAPI documentation support.
"""

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.registry.models import ModelTier

if TYPE_CHECKING:
    from app.dispatcher.handlers import DispatchResult
    from app.dispatcher.handlers import TokenUsage as DispatcherTokenUsage
    from app.registry.models import ModelMetadata
    from app.router.engine import RouteChoice


# =============================================================================
# ENUMERATIONS
# =============================================================================


class Verdict(str, Enum):
    """
    Content moderation verdict classification.

    SAFE: Content passes moderation, no issues detected
    FLAGGED: Content violates guidelines, should be blocked/reviewed
    REQUIRES_REVIEW: Content is ambiguous, needs human review
    """

    SAFE = "safe"
    FLAGGED = "flagged"
    REQUIRES_REVIEW = "requires_review"


class RouteName(str, Enum):
    """
    Available semantic routes for content classification.

    Maps to the route definitions in router/routes.py. Each route
    corresponds to a specific model tier optimized for that content type.
    """

    OBVIOUS_HARM = "obvious_harm"
    OBVIOUS_SAFE = "obvious_safe"
    AMBIGUOUS_RISK = "ambiguous_risk"
    SYSTEM_ATTACK = "system_attack"
    NON_ENGLISH = "non_english"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RequestMetadata(BaseModel):
    """
    Optional metadata accompanying a moderation request.

    Metadata provides additional context that may influence
    routing decisions or enable better tracking and analytics.
    """

    source: str | None = Field(
        default=None,
        max_length=100,
        description="Source of the content (e.g., 'comments', 'posts', 'chat')",
    )

    language_hint: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description="Expected language ISO code (e.g., 'en', 'es', 'fr-CA')",
    )

    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="Anonymous user identifier for tracking (not stored)",
    )

    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Processing priority hint",
    )

    model_config = ConfigDict(extra="ignore")


class ModerationRequest(BaseModel):
    """
    Request body for the /moderate endpoint.

    Contains the text content to be moderated along with optional
    metadata for routing hints and tracking purposes.

    Example:
        {
            "content": "Great article, thanks for sharing!",
            "metadata": {
                "source": "comments",
                "language_hint": "en"
            }
        }
    """

    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text content to moderate",
    )

    metadata: RequestMetadata | None = Field(
        default=None,
        description="Optional request metadata for routing and tracking",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_whitespace(cls, v: str) -> str:
        """Ensure content is not empty or whitespace only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Content cannot be empty or whitespace only")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "content": "This is a great product, highly recommend!",
                    "metadata": {"source": "reviews"},
                },
                {
                    "content": "I'm going to kill this presentation tomorrow",
                    "metadata": {"source": "messages", "priority": "high"},
                },
            ]
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class TokenUsage(BaseModel):
    """
    Token consumption metrics for cost tracking.

    Tracks input and output tokens separately to enable accurate
    cost calculation based on provider pricing models.
    """

    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of input tokens processed",
    )

    output_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of output tokens generated",
    )

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens


class RoutingInfo(BaseModel):
    """
    Information about the semantic routing decision.

    Captures details about which route was selected, the confidence
    of the match, and timing information for the routing step.
    """

    route_selected: str = Field(
        ...,
        description="Name of the semantic route that matched",
    )

    route_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of route match (0.0-1.0)",
    )

    routing_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken for routing decision in milliseconds",
    )

    fallback_used: bool = Field(
        default=False,
        description="Whether the default fallback route was used due to low confidence",
    )


class ModerationResponse(BaseModel):
    """
    Response from the /moderate endpoint.

    Contains the moderation verdict along with comprehensive metadata
    about the routing decision, model used, performance metrics, and
    cost tracking information.

    Example:
        {
            "verdict": "safe",
            "confidence": 0.92,
            "reasoning": null,
            "routing": {
                "route_selected": "obvious_safe",
                "route_confidence": 0.87,
                "routing_latency_ms": 38.5,
                "fallback_used": false
            },
            "model_used": "llama-3.1-8b",
            "model_tier": "tier1",
            "inference_latency_ms": 145.2,
            "tokens": {
                "input_tokens": 128,
                "output_tokens": 45
            },
            "estimated_cost_usd": 0.000008
        }
    """

    verdict: Verdict = Field(
        ...,
        description="Moderation decision (safe, flagged, or requires_review)",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the verdict (0.0-1.0)",
    )

    reasoning: str | None = Field(
        default=None,
        description="Explanation of verdict (populated for Tier 2 reasoning models)",
    )

    routing: RoutingInfo = Field(
        ...,
        description="Details about the routing decision",
    )

    model_used: str = Field(
        ...,
        description="Model ID that processed the content",
    )

    model_tier: ModelTier = Field(
        ...,
        description="Tier classification of the model used",
    )

    inference_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time for model inference in milliseconds",
    )

    tokens: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token consumption metrics",
    )

    estimated_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated cost of this request in USD",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "verdict": "safe",
                    "confidence": 0.92,
                    "reasoning": None,
                    "routing": {
                        "route_selected": "obvious_safe",
                        "route_confidence": 0.87,
                        "routing_latency_ms": 38.5,
                        "fallback_used": False,
                    },
                    "model_used": "llama-3.1-8b",
                    "model_tier": "tier1",
                    "inference_latency_ms": 145.2,
                    "tokens": {"input_tokens": 128, "output_tokens": 45},
                    "estimated_cost_usd": 0.000008,
                }
            ]
        }
    )


# =============================================================================
# ERROR MODELS
# =============================================================================


class ErrorCodes:
    """
    Standard error codes for API responses.

    These codes enable programmatic error handling by clients
    without parsing human-readable messages.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"
    CONTENT_EMPTY = "CONTENT_EMPTY"
    ROUTING_ERROR = "ROUTING_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ErrorDetail(BaseModel):
    """
    Detailed error information for API error responses.

    Provides machine-readable error codes, human-readable messages,
    and optional field information for validation errors.
    """

    code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling",
    )

    message: str = Field(
        ...,
        description="Human-readable error message",
    )

    field: str | None = Field(
        default=None,
        description="Field that caused the error (for validation errors)",
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format for all API errors.

    Provides consistent error structure across all endpoints
    for predictable client-side error handling.

    Example:
        {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Content cannot be empty",
                "field": "content"
            },
            "request_id": "req_abc123"
        }
    """

    error: ErrorDetail = Field(
        ...,
        description="Error details",
    )

    request_id: str | None = Field(
        default=None,
        description="Request ID for tracking and support",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Content cannot be empty",
                        "field": "content",
                    },
                    "request_id": "req_xyz789",
                }
            ]
        }
    )


# =============================================================================
# METRICS MODELS
# =============================================================================


class RouteMetrics(BaseModel):
    """
    Aggregated metrics for a specific semantic route.

    Tracks request volume, average confidence, and latency
    for each route to monitor routing effectiveness.
    """

    route_name: str = Field(
        ...,
        description="Semantic route identifier",
    )

    request_count: int = Field(
        default=0,
        ge=0,
        description="Total requests classified to this route",
    )

    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average routing confidence score",
    )

    avg_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average routing latency in milliseconds",
    )


class ModelMetrics(BaseModel):
    """
    Aggregated metrics for a specific model in the pool.

    Tracks usage, token consumption, costs, and performance
    for each model to enable cost optimization analysis.
    """

    model_id: str = Field(
        ...,
        description="Model identifier",
    )

    request_count: int = Field(
        default=0,
        ge=0,
        description="Total requests processed by this model",
    )

    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (input + output)",
    )

    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated cost in USD",
    )

    avg_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average inference latency in milliseconds",
    )


class MetricsResponse(BaseModel):
    """
    Response from the /metrics endpoint.

    Provides aggregated statistics for monitoring system performance,
    cost efficiency, and routing effectiveness. The cost savings
    calculation compares actual costs against a hypothetical scenario
    where all requests used GPT-4o.

    Example:
        {
            "total_requests": 1000,
            "requests_by_route": {...},
            "requests_by_model": {...},
            "total_cost_usd": 0.15,
            "hypothetical_cost_usd": 5.00,
            "cost_savings_percent": 97.0,
            "avg_routing_latency_ms": 35.2,
            "avg_inference_latency_ms": 180.5
        }
    """

    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total moderation requests processed",
    )

    requests_by_route: dict[str, RouteMetrics] = Field(
        default_factory=dict,
        description="Metrics breakdown by semantic route",
    )

    requests_by_model: dict[str, ModelMetrics] = Field(
        default_factory=dict,
        description="Metrics breakdown by model",
    )

    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Actual total cost with semantic routing",
    )

    hypothetical_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Hypothetical cost if all requests used GPT-4o",
    )

    cost_savings_percent: float = Field(
        default=0.0,
        description="Percentage saved compared to GPT-4o baseline",
    )

    avg_routing_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average routing decision time in milliseconds",
    )

    avg_inference_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average model inference time in milliseconds",
    )


# =============================================================================
# HEALTH MODELS
# =============================================================================


class ComponentHealth(BaseModel):
    """
    Health status of an individual system component.

    Used to report status of router, model providers, and
    other dependencies in health check responses.
    """

    name: str = Field(
        ...,
        description="Component name (e.g., 'router', 'groq', 'openai')",
    )

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Component health status",
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Last known latency for this component",
    )

    message: str | None = Field(
        default=None,
        description="Additional status information or error details",
    )


class HealthResponse(BaseModel):
    """
    Response from the /health endpoint.

    Provides overall system health status along with individual
    component health for detailed diagnostics.

    Example:
        {
            "status": "healthy",
            "service": "sentinel-triage",
            "version": "0.1.0",
            "components": [
                {"name": "router", "status": "healthy", "latency_ms": 2.1},
                {"name": "groq", "status": "healthy"}
            ],
            "uptime_seconds": 3600.5
        }
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall service health status",
    )

    service: str = Field(
        default="sentinel-triage",
        description="Service identifier",
    )

    version: str = Field(
        ...,
        description="Application version",
    )

    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Health status of individual components",
    )

    uptime_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Time since service start in seconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "service": "sentinel-triage",
                    "version": "0.1.0",
                    "components": [
                        {"name": "router", "status": "healthy", "latency_ms": 25.3},
                        {"name": "groq", "status": "healthy", "latency_ms": 120.0},
                        {"name": "openai", "status": "healthy", "latency_ms": 350.0},
                    ],
                    "uptime_seconds": 3600.5,
                }
            ]
        }
    )


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================


def routing_info_from_route_choice(choice: "RouteChoice") -> RoutingInfo:
    """
    Convert a RouteChoice dataclass to a RoutingInfo Pydantic model.

    This bridges the internal router representation with the API
    response format for consistent serialization.

    Args:
        choice: RouteChoice dataclass from router/engine.py

    Returns:
        RoutingInfo Pydantic model for API response
    """
    return RoutingInfo(
        route_selected=choice.route_name,
        route_confidence=choice.confidence,
        routing_latency_ms=choice.latency_ms,
        fallback_used=choice.fallback_used,
    )


def token_usage_from_dispatch(tokens: "DispatcherTokenUsage") -> TokenUsage:
    """
    Convert dispatcher TokenUsage dataclass to Pydantic model.

    Args:
        tokens: TokenUsage dataclass from dispatcher/handlers.py

    Returns:
        TokenUsage Pydantic model for API response
    """
    return TokenUsage(
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
    )


def build_moderation_response(
    dispatch_result: "DispatchResult",
    route_choice: "RouteChoice",
    model_metadata: "ModelMetadata",
) -> ModerationResponse:
    """
    Build a complete ModerationResponse from internal components.

    Aggregates data from the dispatcher result, routing decision, and
    model metadata into a unified API response. Calculates the estimated
    cost based on token usage and model pricing.

    Args:
        dispatch_result: Result from dispatcher with verdict and tokens
        route_choice: Routing decision from semantic router
        model_metadata: Metadata for the model that processed the request

    Returns:
        Complete ModerationResponse ready for API serialization
    """
    # Calculate estimated cost based on token usage and model pricing
    cost_usd = (
        dispatch_result.tokens.input_tokens
        * model_metadata.cost_per_1m_input_tokens
        / 1_000_000
    ) + (
        dispatch_result.tokens.output_tokens
        * model_metadata.cost_per_1m_output_tokens
        / 1_000_000
    )

    # Parse verdict string to enum (handle case variations)
    verdict_str = dispatch_result.verdict.lower()
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        # Default to requires_review for unexpected values
        verdict = Verdict.REQUIRES_REVIEW

    return ModerationResponse(
        verdict=verdict,
        confidence=dispatch_result.confidence,
        reasoning=dispatch_result.reasoning,
        routing=routing_info_from_route_choice(route_choice),
        model_used=dispatch_result.model_used,
        model_tier=model_metadata.tier,
        inference_latency_ms=dispatch_result.latency_ms,
        tokens=token_usage_from_dispatch(dispatch_result.tokens),
        estimated_cost_usd=round(cost_usd, 10),
    )
