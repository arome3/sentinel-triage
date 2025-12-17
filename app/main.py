"""
Sentinel-Triage: FastAPI Application Entry Point

This module initializes the FastAPI application and defines the core endpoints:
- /health: Health check endpoint
- /config: Non-sensitive configuration values
- /models: Model registry information
- /moderate: Content moderation endpoint
- /metrics: Routing statistics endpoint

The application uses a lifespan context manager to:
1. Load and validate configuration at startup
2. Configure logging based on settings
3. Validate that required API keys are present
4. Pre-warm the semantic router
"""

from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Depends, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.config import Settings, get_settings, configure_logging
from app.registry import get_model_registry
from app.router.engine import ensure_router_initialized, get_router_engine
from app.dispatcher.handlers import dispatch_by_route
from app.schemas.moderation import (
    ModerationRequest,
    ModerationResponse,
    ErrorResponse,
    ErrorDetail,
    ErrorCodes,
    MetricsResponse,
    HealthResponse,
    ComponentHealth,
    build_moderation_response,
)

logger = logging.getLogger(__name__)

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup/shutdown events.

    On startup:
    - Loads configuration from environment
    - Configures logging
    - Validates required API keys are present

    On shutdown:
    - Logs shutdown message
    """
    settings = get_settings()
    configure_logging(settings)

    logger.info("=" * 60)
    logger.info("Sentinel-Triage starting up...")
    logger.info("=" * 60)
    logger.info(f"Similarity threshold: {settings.similarity_threshold}")
    logger.info(f"Default route: {settings.default_route}")
    logger.info(f"Fallback model: {settings.fallback_model}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Cost tracking: {'enabled' if settings.track_costs else 'disabled'}")
    logger.info(f"Debug mode: {'enabled' if settings.debug else 'disabled'}")

    # Validate API keys are present (not empty)
    if not settings.openai_api_key.get_secret_value():
        raise ValueError("OPENAI_API_KEY is required but not set")
    if not settings.groq_api_key.get_secret_value():
        raise ValueError("GROQ_API_KEY is required but not set")

    logger.info("OpenAI API key: configured")
    logger.info("Groq API key: configured")
    if settings.deepseek_api_key:
        logger.info("DeepSeek API key: configured")
    else:
        logger.info("DeepSeek API key: not configured (using OpenAI fallback for Tier 2)")

    logger.info("Configuration validated successfully")

    logger.info("Pre-warming router engine...")
    await ensure_router_initialized()

    engine = await get_router_engine()
    routes_info = engine.get_routes_info()
    logger.info(f"Router ready with {routes_info['num_routes']} routes:")
    for route in routes_info['routes']:
        logger.info(f"  - {route['name']}: {route['num_utterances']} utterances")
    logger.info(f"Router initialization took {routes_info['init_latency_ms']:.2f}ms")

    # Record start time for uptime tracking
    global _start_time
    _start_time = time.time()

    logger.info("=" * 60)
    logger.info("Sentinel-Triage ready to accept requests")

    yield  # Application runs here

    logger.info("Sentinel-Triage shutting down...")


app = FastAPI(
    title="Sentinel-Triage",
    description="Semantic Router for Intelligent Content Moderation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Sentinel-Triage",
        "description": "Semantic Router for Content Moderation",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "config": "/config"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check system health and component status.",
)
async def health_check():
    """
    Health check endpoint for monitoring and orchestration.

    Checks:
    - Router initialization status
    - Registry availability
    - Provider connectivity (optional)
    - System uptime

    Used for Kubernetes readiness/liveness probes and load balancer health checks.
    """
    components = []
    overall_status = "healthy"

    try:
        engine = await get_router_engine()
        if engine.is_initialized:
            routes_info = engine.get_routes_info()
            components.append(
                ComponentHealth(
                    name="router",
                    status="healthy",
                    latency_ms=routes_info.get("init_latency_ms"),
                    message=f"Router initialized with {routes_info['num_routes']} routes",
                )
            )
        else:
            components.append(
                ComponentHealth(
                    name="router",
                    status="unhealthy",
                    message="Router not initialized",
                )
            )
            overall_status = "unhealthy"
    except Exception as e:
        components.append(
            ComponentHealth(
                name="router",
                status="unhealthy",
                message=str(e),
            )
        )
        overall_status = "unhealthy"

    try:
        registry = get_model_registry()
        model_count = len(registry.list_models())
        components.append(
            ComponentHealth(
                name="registry",
                status="healthy",
                message=f"{model_count} models registered",
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="registry",
                status="unhealthy",
                message=str(e),
            )
        )
        overall_status = "degraded"

    uptime = time.time() - _start_time if _start_time > 0 else 0.0

    return HealthResponse(
        status=overall_status,
        service="sentinel-triage",
        version="0.1.0",
        components=components,
        uptime_seconds=uptime,
    )


@app.get("/config")
async def show_config(settings: Settings = Depends(get_settings)):
    """
    Returns non-sensitive configuration values.

    API keys are SecretStr and are NOT exposed in this endpoint.
    This is safe to call for debugging configuration issues.
    """
    return {
        "router": {
            "similarity_threshold": settings.similarity_threshold,
            "default_route": settings.default_route,
            "fallback_model": settings.fallback_model
        },
        "embedding": {
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions
        },
        "cost_tracking": {
            "enabled": settings.track_costs,
            "gpt4o_baseline_per_1m": settings.gpt4o_cost_per_1m
        },
        "server": {
            "host": settings.host,
            "port": settings.port,
            "debug": settings.debug
        },
        "logging": {
            "level": settings.log_level
        },
        "api_keys_configured": {
            "openai": bool(settings.openai_api_key.get_secret_value()),
            "groq": bool(settings.groq_api_key.get_secret_value()),
            "deepseek": bool(settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else False)
        }
    }


@app.get("/models")
async def list_models():
    """
    List all registered models with their metadata.

    Returns the complete model registry including:
    - Model identifiers and display names
    - Tier classification (tier1, tier2, specialist)
    - Provider information
    - Cost metadata (per 1M tokens)
    - Latency targets
    - Capabilities
    - Route-to-model mapping
    """
    registry = get_model_registry()

    return {
        "models": [
            {
                "model_id": model.model_id,
                "display_name": model.display_name,
                "tier": model.tier.value,
                "provider": model.provider.value,
                "api_model_name": model.api_model_name,
                "cost_per_1m_input": model.cost_per_1m_input_tokens,
                "cost_per_1m_output": model.cost_per_1m_output_tokens,
                "latency_target_ms": model.latency_target_ms,
                "capabilities": [cap.value for cap in model.capabilities],
                "max_tokens": model.max_tokens,
                "temperature": model.temperature,
            }
            for model in registry.list_models()
        ],
        "route_mapping": registry.get_route_mapping(),
        "total_models": len(registry.list_models()),
    }


@app.get("/routes")
async def list_routes():
    """
    List all configured semantic routes.

    Returns information about each route including:
    - Route name
    - Description
    - Number of utterances (example phrases)
    - Router configuration (encoder, threshold)
    """
    engine = await get_router_engine()
    return engine.get_routes_info()


@app.post("/route")
async def test_route(
    content: str = Query(
        ...,
        description="The text content to classify",
        min_length=1,
        max_length=10000,
        examples=["You are an idiot", "Great post, thanks!", "Bonjour, comment ca va?"]
    )
):
    """
    Test routing without full moderation.

    This endpoint classifies content into one of the semantic routes:
    - obvious_harm: Clear violations (spam, harassment)
    - obvious_safe: Benign engagement
    - ambiguous_risk: Nuanced content requiring reasoning
    - system_attack: Jailbreak/PII attempts
    - non_english: Foreign language content

    Useful for debugging route classification before full moderation.

    Returns:
        - content_preview: First 100 chars of input
        - route: Selected route name
        - confidence: Similarity score (0.0-1.0)
        - latency_ms: Routing decision time
        - fallback_used: Whether default route was used
    """
    engine = await get_router_engine()
    result = await engine.route(content)

    return {
        "content_preview": content[:100] + "..." if len(content) > 100 else content,
        "route": result.route_name,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 2),
        "fallback_used": result.fallback_used
    }


@app.post(
    "/moderate",
    response_model=ModerationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Moderate content",
    description="Classify and moderate user-generated content using semantic routing.",
)
async def moderate_content(request: ModerationRequest):
    """
    Main content moderation endpoint.

    Accepts user-generated text and returns a moderation verdict with:
    - Verdict (safe/flagged/requires_review)
    - Confidence score
    - Reasoning (for Tier 2 models)
    - Routing metadata
    - Model information
    - Cost tracking

    Flow:
    1. Route content to appropriate category
    2. Dispatch to selected model
    3. Calculate cost
    4. Record metrics
    5. Return verdict with metadata
    """
    settings = get_settings()

    try:
        # Step 1: Route the content
        engine = await get_router_engine()
        route_choice = await engine.route(request.content)

        # Step 2: Dispatch to model
        dispatch_result = await dispatch_by_route(route_choice.route_name, request.content)

        if not dispatch_result.success:
            logger.error(f"Dispatch failed: {dispatch_result.error}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.INFERENCE_ERROR,
                        message=f"Model inference failed: {dispatch_result.error}",
                    )
                ).model_dump(),
            )

        registry = get_model_registry()
        model = registry.get_model(dispatch_result.model_used)

        if not model:
            logger.error(f"Model not found in registry: {dispatch_result.model_used}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.INTERNAL_ERROR,
                        message=f"Model not found: {dispatch_result.model_used}",
                    )
                ).model_dump(),
            )

        response = build_moderation_response(
            dispatch_result=dispatch_result,
            route_choice=route_choice,
            model_metadata=model,
        )

        if settings.track_costs:
            from app.metrics import get_metrics_store, get_cost_calculator, RequestMetric

            cost_breakdown = get_cost_calculator().calculate_by_model_id(
                model_id=dispatch_result.model_used,
                input_tokens=dispatch_result.tokens.input_tokens,
                output_tokens=dispatch_result.tokens.output_tokens
            )

            if cost_breakdown:
                get_metrics_store().record(RequestMetric(
                    timestamp=time.time(),
                    route_name=route_choice.route_name,
                    model_id=dispatch_result.model_used,
                    route_confidence=route_choice.confidence,
                    routing_latency_ms=route_choice.latency_ms,
                    inference_latency_ms=dispatch_result.latency_ms,
                    input_tokens=dispatch_result.tokens.input_tokens,
                    output_tokens=dispatch_result.tokens.output_tokens,
                    actual_cost_usd=cost_breakdown.actual_cost_usd,
                    hypothetical_cost_usd=cost_breakdown.hypothetical_cost_usd,
                    verdict=dispatch_result.verdict
                ))

        return response

    except Exception as e:
        logger.exception("Moderation failed")
        raise HTTPException(
            status_code=500,
            detail={"code": ErrorCodes.INTERNAL_ERROR, "message": str(e)},
        )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get metrics",
    description="Retrieve aggregated routing and cost metrics.",
)
async def get_metrics():
    """
    Return aggregated metrics for monitoring and cost analysis.

    Uses the MetricsReporter to generate a complete report from
    the thread-safe MetricsStore.

    Includes:
    - Request counts by route and model
    - Cost tracking with savings calculation
    - Latency statistics
    - Percentage savings compared to GPT-4o baseline
    """
    from app.metrics import MetricsReporter

    reporter = MetricsReporter()
    return reporter.generate_report()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Returns a consistent error response format with the first validation
    error's details for client-side error handling.
    """
    errors = exc.errors()
    first_error = errors[0] if errors else {}

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCodes.VALIDATION_ERROR,
                "message": first_error.get("msg", "Validation failed"),
                "field": ".".join(str(loc) for loc in first_error.get("loc", [])),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """
    Handle HTTP exceptions with consistent format.

    Ensures all HTTP errors return a consistent error response structure
    for predictable client-side error handling.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "HTTP_ERROR", "message": str(detail)}},
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Logs the full exception for debugging and returns a generic error
    response to avoid leaking implementation details.
    """
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": ErrorCodes.INTERNAL_ERROR,
                "message": "An unexpected error occurred",
            }
        },
    )
