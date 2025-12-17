"""
Dispatcher Tests

Tests for model dispatch handlers with mocked API responses.
Validates response parsing, provider dispatch, error handling, and retry logic.

Test Categories:
1. TestResponseParsing - _parse_model_response() unit tests
2. TestTokenUsage - Token calculation validation
3. TestDispatchResult - DispatchResult dataclass tests
4. TestDispatchByRoute - Route-based dispatch tests
5. TestDispatchRetry - Retry logic validation

See: docs/09-testing.md for detailed test specifications.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.dispatcher.handlers import (
    TokenUsage,
    DispatchResult,
    _parse_model_response,
    dispatch_by_route,
    dispatch,
    dispatch_with_retry,
)


class TestResponseParsing:
    """Unit tests for _parse_model_response() function."""

    def test_parse_valid_json(self):
        """Parse clean JSON response."""
        response = '{"verdict": "safe", "confidence": 0.95}'
        result = _parse_model_response(response)

        assert result["verdict"] == "safe"
        assert result["confidence"] == 0.95

    def test_parse_json_with_reasoning(self):
        """Parse JSON response with reasoning field."""
        response = '{"verdict": "flagged", "confidence": 0.8, "reasoning": "Contains spam"}'
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "Contains spam"

    def test_parse_json_in_code_block(self):
        """Parse JSON wrapped in markdown code block."""
        response = '''Here's my analysis:
```json
{"verdict": "flagged", "confidence": 0.8, "reasoning": "Contains spam indicators"}
```'''
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"
        assert result["confidence"] == 0.8

    def test_parse_json_in_plain_code_block(self):
        """Parse JSON in code block without language identifier."""
        response = '''Analysis:
```
{"verdict": "safe", "confidence": 0.9}
```'''
        result = _parse_model_response(response)

        assert result["verdict"] == "safe"
        assert result["confidence"] == 0.9

    def test_parse_empty_returns_requires_review(self):
        """Empty response returns requires_review verdict."""
        result = _parse_model_response("")

        assert result["verdict"] == "requires_review"
        assert result["confidence"] == 0.0

    def test_parse_none_returns_requires_review(self):
        """None response returns requires_review verdict."""
        result = _parse_model_response(None)

        assert result["verdict"] == "requires_review"
        assert result["confidence"] == 0.0

    def test_parse_text_infers_safe(self):
        """Text mentioning 'safe' infers safe verdict."""
        response = "This content is safe and complies with community guidelines."
        result = _parse_model_response(response)

        assert result["verdict"] == "safe"
        assert result["confidence"] == 0.5  # Low confidence for inferred

    def test_parse_text_infers_flagged_harmful(self):
        """Text mentioning 'harmful' infers flagged verdict."""
        response = "This content is harmful and violates our policies."
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"

    def test_parse_text_infers_flagged_violation(self):
        """Text mentioning 'violation' infers flagged verdict."""
        response = "The content is a clear violation of terms."
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"

    def test_parse_text_infers_flagged_blocked(self):
        """Text mentioning 'blocked' infers flagged verdict."""
        response = "This content should be blocked."
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"

    def test_parse_text_unsafe_not_interpreted_as_safe(self):
        """Text with 'unsafe' should not be interpreted as safe."""
        response = "This content is unsafe for general audiences."
        result = _parse_model_response(response)

        # 'unsafe' contains 'safe' but should be flagged due to 'unsafe' keyword
        assert result["verdict"] == "flagged"

    def test_parse_unknown_text_returns_requires_review(self):
        """Unknown text returns requires_review."""
        response = "I'm not sure about this content, need more context."
        result = _parse_model_response(response)

        assert result["verdict"] == "requires_review"

    def test_parse_invalid_json_falls_back_to_inference(self):
        """Invalid JSON triggers text inference."""
        response = '{"verdict": "safe", broken json'
        result = _parse_model_response(response)

        # Should fall back to inference from 'safe' keyword
        assert result["verdict"] == "safe"

    def test_parse_complex_json_response(self):
        """Parse complex JSON with extra fields."""
        response = '''{
            "verdict": "flagged",
            "confidence": 0.85,
            "reasoning": "Detected sarcasm",
            "detected_patterns": ["sarcasm", "irony"],
            "extra_field": "ignored"
        }'''
        result = _parse_model_response(response)

        assert result["verdict"] == "flagged"
        assert result["confidence"] == 0.85
        assert "detected_patterns" in result


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_total_tokens_calculation(self):
        """Total tokens equals input + output."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_zero_tokens(self):
        """Zero tokens results in zero total."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert usage.total_tokens == 0

    def test_default_values(self):
        """Default values are zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_large_token_counts(self):
        """Handle large token counts."""
        usage = TokenUsage(input_tokens=100000, output_tokens=50000)
        assert usage.total_tokens == 150000


class TestDispatchResult:
    """Tests for DispatchResult dataclass."""

    def test_success_property_true(self):
        """Success is True when no error."""
        result = DispatchResult(
            verdict="safe",
            confidence=0.9,
            reasoning=None,
            model_used="llama-3.1-8b",
            provider="groq",
            latency_ms=150.0,
            tokens=TokenUsage(100, 50),
            error=None
        )
        assert result.success is True

    def test_success_property_false(self):
        """Success is False when error present."""
        result = DispatchResult(
            verdict="requires_review",
            confidence=0.0,
            reasoning=None,
            model_used="llama-3.1-8b",
            provider="groq",
            latency_ms=100.0,
            tokens=TokenUsage(0, 0),
            error="API Error: Rate limited"
        )
        assert result.success is False

    def test_dispatch_result_with_reasoning(self):
        """DispatchResult includes reasoning for Tier 2."""
        result = DispatchResult(
            verdict="flagged",
            confidence=0.85,
            reasoning="The content contains sarcasm that could be misinterpreted.",
            model_used="gpt-4o",
            provider="openai",
            latency_ms=2500.0,
            tokens=TokenUsage(200, 150),
            raw_response={"verdict": "flagged"}
        )

        assert result.reasoning is not None
        assert "sarcasm" in result.reasoning
        assert result.success

    def test_dispatch_result_default_tokens(self):
        """DispatchResult uses default TokenUsage if not provided."""
        result = DispatchResult(
            verdict="safe",
            confidence=0.9,
            reasoning=None,
            model_used="llama-3.1-8b",
            provider="groq",
            latency_ms=150.0
        )

        assert result.tokens.input_tokens == 0
        assert result.tokens.output_tokens == 0


class TestDispatchByRoute:
    """Tests for route-based dispatch function."""

    @pytest.mark.asyncio
    async def test_dispatch_obvious_harm_to_tier1(self, mock_provider_clients):
        """obvious_harm routes to llama-3.1-8b via Groq."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("obvious_harm", "bad content")

            assert result.model_used == "llama-3.1-8b"
            assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_dispatch_obvious_safe_to_tier1(self, mock_provider_clients):
        """obvious_safe routes to llama-3.1-8b via Groq."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("obvious_safe", "good content")

            assert result.model_used == "llama-3.1-8b"
            assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_dispatch_ambiguous_to_tier2(self, mock_provider_clients):
        """ambiguous_risk routes to gpt-4o via OpenAI."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("ambiguous_risk", "sarcastic content")

            assert result.model_used == "gpt-4o"
            assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_dispatch_system_attack_to_guard(self, mock_provider_clients):
        """system_attack routes to llama-guard-4 via Groq."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("system_attack", "ignore instructions")

            assert result.model_used == "llama-guard-4"
            assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_dispatch_non_english_to_maverick(self, mock_provider_clients):
        """non_english routes to llama-4-maverick via Groq."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("non_english", "Bonjour")

            assert result.model_used == "llama-4-maverick"
            assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_route_uses_fallback(self, mock_provider_clients):
        """Unknown route uses fallback model."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_by_route("unknown_route", "test content")

            # Should use fallback model (llama-3.1-8b)
            assert result.model_used == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_dispatch_handles_api_error(self):
        """Dispatch returns error result on API failure."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error: Service unavailable")
            )
            mock_get.return_value.groq = mock_client

            result = await dispatch_by_route("obvious_safe", "test")

            assert not result.success
            assert result.verdict == "requires_review"
            assert "API Error" in result.error


class TestDispatch:
    """Tests for the main dispatch function."""

    @pytest.mark.asyncio
    async def test_dispatch_groq_model(self, mock_provider_clients, tier1_model):
        """Dispatch to Groq provider for Tier 1 model."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch(tier1_model, "test content")

            assert result.provider == "groq"
            assert result.success

    @pytest.mark.asyncio
    async def test_dispatch_openai_model(self, mock_provider_clients, tier2_model):
        """Dispatch to OpenAI provider for Tier 2 model."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch(tier2_model, "test content")

            assert result.provider == "openai"
            assert result.success

    @pytest.mark.asyncio
    async def test_dispatch_extracts_token_usage(self, mock_provider_clients, tier1_model):
        """Dispatch extracts token usage from response."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch(tier1_model, "test content")

            assert result.tokens.input_tokens > 0
            assert result.tokens.output_tokens > 0


# =============================================================================
# TEST DISPATCH RETRY
# =============================================================================


class TestDispatchRetry:
    """Tests for dispatch retry logic."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_attempt(self, mock_provider_clients, tier1_model):
        """Successful first attempt returns immediately."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            result = await dispatch_with_retry(tier1_model, "test", max_retries=3)

            assert result.success
            # Should only call once on success
            mock_provider_clients.groq.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_recovers_from_transient_error(self, tier1_model):
        """Retry recovers from transient error."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"verdict": "safe", "confidence": 0.9}'))
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        call_count = 0

        async def flaky_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return mock_response

        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = flaky_api
            mock_get.return_value.groq = mock_client

            result = await dispatch_with_retry(tier1_model, "test", max_retries=3)

            assert result.success
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_error(self, tier1_model):
        """All retries exhausted returns error result."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_client = AsyncMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Persistent error")
            )
            mock_get.return_value.groq = mock_client

            result = await dispatch_with_retry(tier1_model, "test", max_retries=2)

            assert not result.success
            assert "Persistent error" in result.error


# =============================================================================
# TEST PROVIDER-SPECIFIC BEHAVIOR
# =============================================================================


class TestProviderSpecificBehavior:
    """Tests for provider-specific dispatch behavior."""

    @pytest.mark.asyncio
    async def test_groq_response_format(self, mock_provider_clients, tier1_model):
        """Verify Groq dispatch uses json_object response format."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            await dispatch(tier1_model, "test content")

            # Verify response_format was set
            call_kwargs = mock_provider_clients.groq.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_openai_response_format(self, mock_provider_clients, tier2_model):
        """Verify OpenAI dispatch uses json_object response format."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            await dispatch(tier2_model, "test content")

            # Verify response_format was set
            call_kwargs = mock_provider_clients.openai.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_dispatch_uses_model_system_prompt(self, mock_provider_clients, tier1_model):
        """Verify dispatch includes model's system prompt."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            await dispatch(tier1_model, "test content")

            call_kwargs = mock_provider_clients.groq.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "test content"

    @pytest.mark.asyncio
    async def test_dispatch_respects_model_max_tokens(self, mock_provider_clients, tier1_model):
        """Verify dispatch uses model's max_tokens setting."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            await dispatch(tier1_model, "test content")

            call_kwargs = mock_provider_clients.groq.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == tier1_model.max_tokens

    @pytest.mark.asyncio
    async def test_dispatch_respects_model_temperature(self, mock_provider_clients, tier1_model):
        """Verify dispatch uses model's temperature setting."""
        with patch("app.dispatcher.handlers.get_clients") as mock_get:
            mock_get.return_value = mock_provider_clients

            await dispatch(tier1_model, "test content")

            call_kwargs = mock_provider_clients.groq.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == tier1_model.temperature
