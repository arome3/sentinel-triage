"""Reusable retry decorator for transient network failures.

Provides an async-compatible ``@retry`` decorator with exponential backoff,
configurable max attempts, and an allow-list of retryable exception types.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_JITTER = 0.25  # fraction of delay added as jitter


class RetryExhaustedError(Exception):
    """All retry attempts were exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Exhausted {attempts} attempts. Last error: {last_exception!r}"
        )


def retry(
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    retryable: tuple[type[BaseException], ...] = (OSError, TimeoutError),
) -> Callable[[F], F]:
    """Decorator that retries an async function on transient failures.

    Uses exponential backoff with jitter. Only exceptions matching the
    ``retryable`` tuple are caught; all others propagate immediately.

    Args:
        max_retries: Maximum number of retry attempts (0 means no retries).
        base_delay: Initial delay between retries in seconds.
        max_delay: Upper bound on delay between retries.
        jitter: Random jitter as a fraction of the computed delay.
        retryable: Tuple of exception types that trigger a retry.

    Returns:
        A decorator that wraps an async callable with retry logic.

    Raises:
        RetryExhaustedError: When all attempts are exhausted.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 2):  # +2: initial + retries
                try:
                    return await func(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_retries + 1:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter > 0:
                        delay += delay * jitter * random.random()  # noqa: S311
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt,
                        max_retries,
                        func.__qualname__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            if last_exc is not None:
                raise RetryExhaustedError(max_retries + 1, last_exc)

        return wrapper  # type: ignore[return-value]

    return decorator


def compute_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> float:
    """Compute the exponential backoff delay for a given attempt number.

    Args:
        attempt: The current attempt number (1-based).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.

    Returns:
        The delay in seconds (without jitter).
    """
    return min(base_delay * (2 ** (attempt - 1)), max_delay)
