"""
Router Engine - Core routing logic for content moderation.

This module implements the semantic routing decision layer that:
1. Embeds incoming content using FastEmbed's local ONNX inference
2. Compares against pre-computed route embeddings
3. Returns the best matching route with confidence score

The engine uses local CPU inference to meet the strict <50ms latency
constraint, ensuring the triage step doesn't become a bottleneck.

Performance Requirements:
    - Routing Decision Latency: < 50ms (guaranteed with local inference)
    - Embedding Inference: ~15-25ms (local CPU, ONNX Runtime)
    - Route Comparison: < 5ms

See: docs/04-router-engine.md for detailed documentation.
"""

import re
import time
import logging
from dataclasses import dataclass

from semantic_router import SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

from app.config import get_settings
from app.router.routes import create_routes

logger = logging.getLogger(__name__)


@dataclass
class RouteChoice:
    """
    Result of a routing decision.

    Attributes:
        route_name: Name of the selected route, or None if no match found
                   (in which case fallback will be applied before returning)
        confidence: Similarity score between content and route (0.0-1.0)
        latency_ms: Time taken for routing decision in milliseconds
        fallback_used: Whether the default route was used due to low confidence
                      or no match
    """
    route_name: str | None
    confidence: float
    latency_ms: float
    fallback_used: bool = False

    def __post_init__(self):
        """Ensure confidence is within valid bounds."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "route_name": self.route_name,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "fallback_used": self.fallback_used
        }


class RouterEngine:
    """
    Semantic Router Engine for content classification.

    The engine initializes once at startup, embedding all route utterances.
    Subsequent routing calls only need to embed the incoming content and
    perform vector comparison - no re-embedding of routes.

    The engine uses FastEmbed's local ONNX inference (BAAI/bge-small-en-v1.5)
    for fast, efficient embeddings (<50ms) and the semantic-router library
    for route matching.

    Usage:
        engine = RouterEngine()
        await engine.initialize()
        result = await engine.route("content to classify")

    Attributes:
        is_initialized: Whether the engine has been initialized
    """

    def __init__(self):
        """
        Initialize the router engine.

        Note: This does NOT initialize the encoder or routes. Call
        initialize() separately to allow for async initialization.
        """
        self._settings = get_settings()
        self._encoder: FastEmbedEncoder | None = None
        self._router: SemanticRouter | None = None
        self._initialized = False
        self._init_latency_ms: float = 0.0

    async def initialize(self) -> None:
        """
        Initialize the encoder and route layer.

        This is separated from __init__ to support async initialization
        and to allow lazy loading if needed. Route utterances are embedded
        during this step, which may take several seconds.

        Raises:
            RuntimeError: If initialization fails (e.g., invalid API key)
        """
        if self._initialized:
            logger.debug("Router already initialized, skipping")
            return

        logger.info("Initializing Router Engine...")
        start_time = time.perf_counter()

        try:
            self._encoder = FastEmbedEncoder(
                name=self._settings.embedding_model,
                score_threshold=self._settings.similarity_threshold,
                cache_dir=self._settings.embedding_cache_dir,
                threads=self._settings.embedding_threads,
            )
            logger.debug(f"Created FastEmbed encoder: {self._settings.embedding_model}")

            routes = create_routes()
            logger.info(f"Created {len(routes)} semantic routes")
            for route in routes:
                logger.debug(f"  - {route.name}: {len(route.utterances)} utterances")

            self._router = SemanticRouter(
                encoder=self._encoder,
                routes=routes,
                auto_sync="local"  # Use local index for in-memory routing
            )

            self._init_latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Router Engine initialized in {self._init_latency_ms:.2f}ms")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Router Engine: {e}")
            raise RuntimeError(f"Router initialization failed: {e}") from e

    def _ensure_initialized(self) -> None:
        """
        Raise error if engine not initialized.

        Raises:
            RuntimeError: If engine has not been initialized
        """
        if not self._initialized or self._router is None:
            raise RuntimeError(
                "RouterEngine not initialized. Call initialize() first."
            )

    def _check_pattern_filters(self, content: str) -> RouteChoice | None:
        """
        Check for specific sarcasm patterns that embedding models miss.

        These patterns catch structural indicators of sarcasm that confuse
        bag-of-words embeddings:
        - Negation-based: "nice person NOT"
        - Compound: "thanks for ruining"
        - Conditional: "sure, because that worked"

        Args:
            content: The text content to check

        Returns:
            RouteChoice if a pattern matches, None otherwise
        """
        content_lower = content.lower()

        # Pattern 1: Negation-based sarcasm (positive word + NOT)
        if re.search(r'\b(nice|great|perfect|wonderful|excellent|brilliant|genius)\b.*\bNOT\b', content, re.IGNORECASE):
            logger.debug(f"Pattern filter matched: negation sarcasm")
            return RouteChoice(
                route_name="ambiguous_risk",
                confidence=0.85,
                latency_ms=0.1,
                fallback_used=False
            )

        # Pattern 2: Compound sarcasm (thanks/great + negative action)
        if re.search(r'\b(thanks|thank you|great job|perfect|wonderful)\b.*\b(ruining|breaking|messing|destroying|wasting)\b', content, re.IGNORECASE):
            logger.debug(f"Pattern filter matched: compound sarcasm")
            return RouteChoice(
                route_name="ambiguous_risk",
                confidence=0.82,
                latency_ms=0.1,
                fallback_used=False
            )

        # Pattern 3: Conditional sarcasm (sure/yeah + because + past tense)
        if re.search(r'\b(sure|yeah|right|oh yeah|oh sure)\b[,.]?\s*(because|since)\b.*\b(worked|works|helped|helps|went)\b', content, re.IGNORECASE):
            logger.debug(f"Pattern filter matched: conditional sarcasm")
            return RouteChoice(
                route_name="ambiguous_risk",
                confidence=0.80,
                latency_ms=0.1,
                fallback_used=False
            )

        return None

    async def route(self, content: str) -> RouteChoice:
        """
        Route content to the appropriate model category.

        This method:
        1. Embeds the incoming content using local FastEmbed inference
        2. Compares against pre-computed route embeddings
        3. Returns the best matching route with confidence

        If no route matches with sufficient confidence, the default route
        from settings is used (fallback).

        Args:
            content: The text content to classify

        Returns:
            RouteChoice with route name, confidence, timing, and fallback flag

        Raises:
            RuntimeError: If engine not initialized
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Check pattern filters first (fast regex, ~0.1ms)
        pattern_result = self._check_pattern_filters(content)
        if pattern_result is not None:
            pattern_result.latency_ms = (time.perf_counter() - start_time) * 1000
            return pattern_result

        try:
            # Perform routing (embeds content and compares to routes)
            # The SemanticRouter returns a RouteChoice or similar object
            route_result = self._router(content)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Handle no match case - route_result.name is None
            if route_result is None or route_result.name is None:
                logger.debug(
                    f"No route matched for content, using fallback '{self._settings.default_route}'. "
                    f"Latency: {latency_ms:.2f}ms"
                )
                return RouteChoice(
                    route_name=self._settings.default_route,
                    confidence=0.0,
                    latency_ms=latency_ms,
                    fallback_used=True
                )

            # Extract confidence (similarity score) if available
            # The semantic-router library uses 'similarity_score' as the attribute name
            confidence = 0.0
            if hasattr(route_result, 'similarity_score') and route_result.similarity_score is not None:
                confidence = float(route_result.similarity_score)
            elif hasattr(route_result, 'similarity') and route_result.similarity is not None:
                confidence = float(route_result.similarity)
            elif hasattr(route_result, 'score') and route_result.score is not None:
                confidence = float(route_result.score)

            logger.debug(
                f"Routed to '{route_result.name}' with confidence {confidence:.3f}. "
                f"Latency: {latency_ms:.2f}ms"
            )

            return RouteChoice(
                route_name=route_result.name,
                confidence=confidence,
                latency_ms=latency_ms,
                fallback_used=False
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Routing failed: {e}. Using fallback route.")

            # Graceful degradation - return fallback route on error
            return RouteChoice(
                route_name=self._settings.default_route,
                confidence=0.0,
                latency_ms=latency_ms,
                fallback_used=True
            )

    async def route_batch(self, contents: list[str]) -> list[RouteChoice]:
        """
        Route multiple content items.

        Currently processes items sequentially. Future optimization could
        batch embedding calls for better throughput.

        Args:
            contents: List of text content to classify

        Returns:
            List of RouteChoice results in same order as input
        """
        self._ensure_initialized()

        results = []
        for content in contents:
            result = await self.route(content)
            results.append(result)

        return results

    @property
    def is_initialized(self) -> bool:
        """Check if engine is ready for routing."""
        return self._initialized

    @property
    def initialization_latency_ms(self) -> float:
        """Return the time taken to initialize the engine."""
        return self._init_latency_ms

    def get_routes_info(self) -> dict:
        """
        Return information about configured routes.

        Returns:
            Dictionary with route configuration details including:
            - num_routes: Total number of routes
            - routes: List of route info (name, description, utterance count)
            - encoder: Embedding model name
            - threshold: Similarity threshold for matching
        """
        self._ensure_initialized()

        return {
            "num_routes": len(self._router.routes),
            "routes": [
                {
                    "name": r.name,
                    "description": getattr(r, 'description', ''),
                    "num_utterances": len(r.utterances) if hasattr(r, 'utterances') else 0
                }
                for r in self._router.routes
            ],
            "encoder": self._settings.embedding_model,
            "threshold": self._settings.similarity_threshold,
            "default_route": self._settings.default_route,
            "init_latency_ms": round(self._init_latency_ms, 2)
        }


_engine_instance: RouterEngine | None = None


async def get_router_engine() -> RouterEngine:
    """
    Get the global router engine instance.

    Creates and initializes the engine on first call.
    Subsequent calls return the same instance.

    This pattern ensures:
    - Routes are only embedded once (at first request or startup)
    - Memory is shared across all requests
    - Thread-safe for async operations

    Returns:
        The initialized RouterEngine singleton
    """
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = RouterEngine()
        await _engine_instance.initialize()

    return _engine_instance


async def ensure_router_initialized() -> None:
    """
    Ensure the router is initialized.

    Call this at application startup to pre-warm the router
    rather than initializing on first request. This avoids
    latency spikes for the first user request.

    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await ensure_router_initialized()
            yield
    """
    await get_router_engine()


def reset_router_engine() -> None:
    """
    Reset the global router engine instance.

    This is primarily useful for testing to ensure a fresh
    engine is created between test runs.
    """
    global _engine_instance
    _engine_instance = None
