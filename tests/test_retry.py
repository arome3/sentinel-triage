"""Tests for the retry decorator utility."""

import pytest

from app.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    RetryExhaustedError,
    compute_delay,
    retry,
)


class TestRetryDecorator:
    """Unit tests for @retry."""

    async def test_success_no_retry(self):
        """Successful call is not retried."""
        call_count = 0

        @retry(max_retries=3)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_transient_error(self):
        """Transient errors trigger retries up to max_retries."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.01, retryable=(OSError,))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("connection reset")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    async def test_raises_retry_exhausted(self):
        """RetryExhaustedError is raised when all attempts fail."""

        @retry(max_retries=2, base_delay=0.01, retryable=(OSError,))
        async def always_fail():
            raise OSError("always broken")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fail()
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, OSError)

    async def test_non_retryable_propagates(self):
        """Non-retryable exceptions propagate immediately."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01, retryable=(OSError,))
        async def bad_input():
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid argument")

        with pytest.raises(ValueError, match="invalid argument"):
            await bad_input()
        assert call_count == 1

    async def test_zero_retries(self):
        """max_retries=0 means no retries, just the initial attempt."""

        @retry(max_retries=0, base_delay=0.01, retryable=(OSError,))
        async def fail_once():
            raise OSError("fail")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await fail_once()
        assert exc_info.value.attempts == 1

    async def test_preserves_function_name(self):
        """Decorated function preserves its name via functools.wraps."""

        @retry()
        async def my_function():
            pass

        assert my_function.__name__ == "my_function"

    async def test_custom_retryable_exceptions(self):
        """Custom exception types can be specified as retryable."""
        call_count = 0

        @retry(max_retries=1, base_delay=0.01, retryable=(TimeoutError,))
        async def timeout_prone():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            return "done"

        result = await timeout_prone()
        assert result == "done"
        assert call_count == 2

    async def test_multiple_retryable_types(self):
        """Multiple exception types can trigger retries."""
        errors = [OSError("conn"), TimeoutError("slow"), ConnectionError("reset")]
        call_count = 0

        @retry(
            max_retries=3,
            base_delay=0.01,
            retryable=(OSError, TimeoutError, ConnectionError),
        )
        async def multi_error():
            nonlocal call_count
            if call_count < len(errors):
                exc = errors[call_count]
                call_count += 1
                raise exc
            call_count += 1
            return "ok"

        result = await multi_error()
        assert result == "ok"
        assert call_count == 4

    async def test_passes_args_and_kwargs(self):
        """Arguments are forwarded correctly to the wrapped function."""

        @retry(max_retries=1, base_delay=0.01)
        async def add(a, b, extra=0):
            return a + b + extra

        result = await add(1, 2, extra=10)
        assert result == 13

    async def test_return_value_preserved(self):
        """Return value from the wrapped function is preserved."""

        @retry()
        async def returns_dict():
            return {"key": "value", "count": 42}

        result = await returns_dict()
        assert result == {"key": "value", "count": 42}


class TestComputeDelay:
    """Unit tests for compute_delay()."""

    def test_first_attempt(self):
        """First attempt uses base_delay."""
        assert compute_delay(1, base_delay=1.0) == 1.0

    def test_exponential_growth(self):
        """Delay doubles with each attempt."""
        assert compute_delay(1, base_delay=0.5) == 0.5
        assert compute_delay(2, base_delay=0.5) == 1.0
        assert compute_delay(3, base_delay=0.5) == 2.0
        assert compute_delay(4, base_delay=0.5) == 4.0

    def test_capped_at_max_delay(self):
        """Delay never exceeds max_delay."""
        delay = compute_delay(100, base_delay=1.0, max_delay=30.0)
        assert delay == 30.0

    def test_default_values(self):
        """Defaults produce expected results."""
        delay = compute_delay(1)
        assert delay == DEFAULT_BASE_DELAY

    def test_large_attempt_bounded(self):
        """Very large attempt numbers stay within max_delay."""
        delay = compute_delay(1000, base_delay=0.1, max_delay=60.0)
        assert delay == 60.0
