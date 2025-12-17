"""
Dispatcher Handlers - Provider-specific inference execution.

This module handles the actual API calls to model providers (Groq, OpenAI, DeepSeek),
abstracting away provider differences behind a unified dispatch interface.

Key components:
- TokenUsage: Tracks token consumption for cost calculation
- DispatchResult: Standardized response from any provider
- ProviderClients: Lazy-initialized async SDK clients
- dispatch(): Unified interface routing to correct provider
- dispatch_with_retry(): Adds exponential backoff retry logic

See: docs/05-dispatcher.md for detailed documentation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from groq import AsyncGroq
from openai import AsyncOpenAI

from app.config import get_settings
from app.registry.models import ModelMetadata, ModelProvider, get_model_registry

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """
    Token usage from model inference.

    Used for cost calculation based on ModelMetadata pricing.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class DispatchResult:
    """
    Result from model inference.

    Contains the parsed moderation verdict along with metadata
    about the inference call for monitoring and cost tracking.

    Attributes:
        verdict: Classification result - "safe", "flagged", or "requires_review"
        confidence: Model's confidence score (0.0-1.0)
        reasoning: Explanation (populated for Tier 2 models)
        model_used: Model ID from registry
        provider: Provider that executed inference
        latency_ms: Inference time in milliseconds
        tokens: Token usage for cost calculation
        raw_response: Full parsed response for debugging
        error: Error message if inference failed
    """

    verdict: str  # "safe" | "flagged" | "requires_review"
    confidence: float  # 0.0-1.0
    reasoning: str | None  # Populated for Tier 2 only
    model_used: str  # Model ID
    provider: str  # Provider name
    latency_ms: float  # Inference time
    tokens: TokenUsage = field(default_factory=TokenUsage)
    raw_response: dict | None = None  # Full response for debugging
    error: str | None = None  # Error message if failed

    @property
    def success(self) -> bool:
        """Check if dispatch completed without errors."""
        return self.error is None


class ProviderClients:
    """
    Lazy-initialized provider SDK clients.

    Clients are created on first use to avoid initialization errors
    when API keys are not configured for unused providers.

    Uses async clients (AsyncGroq, AsyncOpenAI) for better concurrency
    when handling multiple moderation requests.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._groq: AsyncGroq | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def groq(self) -> AsyncGroq:
        """
        Get Groq client (lazy initialization).

        Returns:
            AsyncGroq client configured with API key from settings.

        Raises:
            ValueError: If GROQ_API_KEY is not configured.
        """
        if self._groq is None:
            api_key = self._settings.groq_api_key.get_secret_value()
            self._groq = AsyncGroq(api_key=api_key)
            logger.debug("Initialized Groq client")
        return self._groq

    @property
    def openai(self) -> AsyncOpenAI:
        """
        Get OpenAI client (lazy initialization).

        Returns:
            AsyncOpenAI client configured with API key from settings.

        Raises:
            ValueError: If OPENAI_API_KEY is not configured.
        """
        if self._openai is None:
            api_key = self._settings.openai_api_key.get_secret_value()
            self._openai = AsyncOpenAI(api_key=api_key)
            logger.debug("Initialized OpenAI client")
        return self._openai


# Global client instance (singleton pattern)
_clients: ProviderClients | None = None


def get_clients() -> ProviderClients:
    """
    Get the global provider clients instance.

    Uses lazy initialization to create clients only when first needed.

    Returns:
        The singleton ProviderClients instance.
    """
    global _clients
    if _clients is None:
        _clients = ProviderClients()
    return _clients


def _parse_model_response(response_text: str | None) -> dict:
    """
    Parse model response into structured format.

    Models are prompted to return JSON, but we handle cases where they
    don't comply with multiple fallback strategies:
    1. Direct JSON parse
    2. Extract from ```json code blocks
    3. Extract from ``` code blocks
    4. Infer verdict from keywords

    Args:
        response_text: Raw text response from the model.

    Returns:
        Dictionary with at least 'verdict' and 'confidence' keys.
    """
    if not response_text:
        return {"verdict": "requires_review", "confidence": 0.0}

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json code block
    if "```json" in response_text:
        try:
            json_start = response_text.index("```json") + 7
            json_end = response_text.index("```", json_start)
            json_str = response_text[json_start:json_end].strip()
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

    # Strategy 3: Extract from any ``` code block
    if "```" in response_text:
        try:
            json_start = response_text.index("```") + 3
            # Skip language identifier if present (e.g., ```python)
            newline_pos = response_text.find("\n", json_start)
            if newline_pos != -1 and newline_pos < json_start + 20:
                json_start = newline_pos + 1
            json_end = response_text.index("```", json_start)
            json_str = response_text[json_start:json_end].strip()
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

    # Strategy 4: Infer verdict from keywords
    response_lower = response_text.lower()

    if "safe" in response_lower and "unsafe" not in response_lower:
        verdict = "safe"
    elif any(
        word in response_lower
        for word in ["flagged", "unsafe", "violation", "harmful", "blocked"]
    ):
        verdict = "flagged"
    else:
        verdict = "requires_review"

    return {
        "verdict": verdict,
        "confidence": 0.5,  # Low confidence for inferred verdicts
        "reasoning": response_text[:500],  # Include raw text as reasoning
    }


async def dispatch_groq(model: ModelMetadata, content: str) -> DispatchResult:
    """
    Dispatch inference to Groq.

    Groq provides fast inference for open-source models like
    Llama 3.1, Llama Guard, and Llama 4 Maverick.

    Args:
        model: Model metadata containing API model name and prompts.
        content: Content to moderate.

    Returns:
        DispatchResult with verdict and metadata.
    """
    clients = get_clients()
    start_time = time.perf_counter()

    try:
        response = await clients.groq.chat.completions.create(
            model=model.api_model_name,
            messages=[
                {"role": "system", "content": model.system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            response_format={"type": "json_object"},
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        response_text = response.choices[0].message.content
        parsed = _parse_model_response(response_text)

        logger.info(
            f"Groq dispatch completed: model={model.model_id}, "
            f"latency={latency_ms:.0f}ms, verdict={parsed.get('verdict')}"
        )

        return DispatchResult(
            verdict=parsed.get("verdict", "requires_review"),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning"),
            model_used=model.model_id,
            provider="groq",
            latency_ms=latency_ms,
            tokens=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            raw_response=parsed,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Groq dispatch failed: {e}")
        return DispatchResult(
            verdict="requires_review",
            confidence=0.0,
            reasoning=None,
            model_used=model.model_id,
            provider="groq",
            latency_ms=latency_ms,
            error=str(e),
        )


async def dispatch_openai(model: ModelMetadata, content: str) -> DispatchResult:
    """
    Dispatch inference to OpenAI.

    Used for Tier 2 reasoning with GPT-4o, which provides
    superior chain-of-thought analysis for ambiguous content.

    Args:
        model: Model metadata containing API model name and prompts.
        content: Content to moderate.

    Returns:
        DispatchResult with verdict and metadata.
    """
    clients = get_clients()
    start_time = time.perf_counter()

    try:
        response = await clients.openai.chat.completions.create(
            model=model.api_model_name,
            messages=[
                {"role": "system", "content": model.system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=model.max_tokens,
            temperature=model.temperature,
            response_format={"type": "json_object"},
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        response_text = response.choices[0].message.content
        parsed = _parse_model_response(response_text)

        logger.info(
            f"OpenAI dispatch completed: model={model.model_id}, "
            f"latency={latency_ms:.0f}ms, verdict={parsed.get('verdict')}"
        )

        return DispatchResult(
            verdict=parsed.get("verdict", "requires_review"),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning"),
            model_used=model.model_id,
            provider="openai",
            latency_ms=latency_ms,
            tokens=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            raw_response=parsed,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"OpenAI dispatch failed: {e}")
        return DispatchResult(
            verdict="requires_review",
            confidence=0.0,
            reasoning=None,
            model_used=model.model_id,
            provider="openai",
            latency_ms=latency_ms,
            error=str(e),
        )


async def dispatch_deepseek(model: ModelMetadata, content: str) -> DispatchResult:
    """
    Dispatch inference to DeepSeek (STUB).

    DeepSeek-R1 support is not currently implemented.
    This stub returns an error indicating the provider is not configured.

    Args:
        model: Model metadata (unused in stub).
        content: Content to moderate (unused in stub).

    Returns:
        DispatchResult with error indicating DeepSeek is not configured.
    """
    logger.warning("DeepSeek dispatch requested but not configured")

    return DispatchResult(
        verdict="requires_review",
        confidence=0.0,
        reasoning=None,
        model_used=model.model_id,
        provider="deepseek",
        latency_ms=0.0,
        error="DeepSeek provider is not configured. Use OpenAI (GPT-4o) for Tier 2 reasoning.",
    )


async def dispatch(model: ModelMetadata, content: str) -> DispatchResult:
    """
    Dispatch content to the specified model for inference.

    This is the main entry point for the dispatcher. It routes
    to the appropriate provider based on the model's configuration.

    Args:
        model: Model metadata from registry.
        content: Content to moderate.

    Returns:
        DispatchResult with verdict and metadata.
    """
    logger.info(f"Dispatching to {model.model_id} via {model.provider.value}")

    match model.provider:
        case ModelProvider.GROQ:
            return await dispatch_groq(model, content)
        case ModelProvider.OPENAI:
            return await dispatch_openai(model, content)
        case ModelProvider.DEEPSEEK:
            return await dispatch_deepseek(model, content)
        case _:
            logger.error(f"Unknown provider: {model.provider}")
            return DispatchResult(
                verdict="requires_review",
                confidence=0.0,
                reasoning=None,
                model_used=model.model_id,
                provider="unknown",
                latency_ms=0.0,
                error=f"Unknown provider: {model.provider}",
            )


async def dispatch_by_route(route_name: str, content: str) -> DispatchResult:
    """
    Dispatch content based on route name.

    Convenience function that looks up the model from registry
    and dispatches to it. Falls back to fallback_model if route not found.

    Args:
        route_name: Name of the selected route (e.g., "obvious_harm").
        content: Content to moderate.

    Returns:
        DispatchResult with verdict and metadata.
    """
    registry = get_model_registry()
    model = registry.get_model_for_route(route_name)

    if model is None:
        settings = get_settings()
        logger.warning(
            f"No model found for route '{route_name}', "
            f"using fallback: {settings.fallback_model}"
        )
        model = registry.get_model(settings.fallback_model)

        if model is None:
            return DispatchResult(
                verdict="requires_review",
                confidence=0.0,
                reasoning=None,
                model_used="unknown",
                provider="none",
                latency_ms=0.0,
                error=f"No model found for route: {route_name}",
            )

    return await dispatch(model, content)


async def dispatch_with_retry(
    model: ModelMetadata,
    content: str,
    max_retries: int = 3,
) -> DispatchResult:
    """
    Dispatch with automatic retry on transient failures.

    Uses exponential backoff (1s, 2s, 4s) between retry attempts.
    Useful for handling rate limits and temporary API errors.

    Args:
        model: Model metadata from registry.
        content: Content to moderate.
        max_retries: Maximum number of attempts (default: 3).

    Returns:
        DispatchResult from successful attempt or last failed attempt.
    """
    for attempt in range(max_retries):
        result = await dispatch(model, content)

        if result.success:
            return result

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            logger.warning(
                f"Dispatch failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait_time}s: {result.error}"
            )
            await asyncio.sleep(wait_time)

    logger.error(f"Dispatch failed after {max_retries} attempts: {result.error}")
    return result
