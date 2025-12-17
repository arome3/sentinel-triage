"""
Dispatcher module: Provider-specific API handlers for model inference.

This module provides a unified interface for dispatching content to AI models
across multiple providers (Groq, OpenAI). It handles provider-specific API calls,
response parsing, and error handling.

Key exports:
- TokenUsage: Token consumption tracking for cost calculation
- DispatchResult: Standardized response from any provider
- ProviderClients: Lazy-initialized async SDK clients
- get_clients(): Get the global provider clients instance
- dispatch(): Main dispatch function routing to correct provider
- dispatch_by_route(): Dispatch based on semantic route name
- dispatch_with_retry(): Dispatch with exponential backoff retry

See: docs/05-dispatcher.md for detailed documentation.
"""

from app.dispatcher.handlers import (
    # Data classes
    TokenUsage,
    DispatchResult,
    # Provider clients
    ProviderClients,
    get_clients,
    # Core dispatch functions
    dispatch,
    dispatch_by_route,
    dispatch_with_retry,
    # Provider-specific (for testing/advanced use)
    dispatch_groq,
    dispatch_openai,
    dispatch_deepseek,
)

__all__ = [
    # Data classes
    "TokenUsage",
    "DispatchResult",
    # Provider clients
    "ProviderClients",
    "get_clients",
    # Core dispatch functions
    "dispatch",
    "dispatch_by_route",
    "dispatch_with_retry",
    # Provider-specific
    "dispatch_groq",
    "dispatch_openai",
    "dispatch_deepseek",
]
