"""
Metrics Store for Request Tracking

Aggregates per-request metrics for analysis and reporting.
Uses in-memory storage for demonstration; production systems
should use Redis, Prometheus, or a time-series database.

The store is thread-safe using threading.Lock to handle
concurrent requests in FastAPI's async environment.
"""

import threading
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RequestMetric:
    """
    Individual request metric record.

    Captures all relevant data from a single moderation request
    for later aggregation and analysis.

    Attributes:
        timestamp: Unix timestamp when request was processed
        route_name: Semantic route selected (e.g., 'obvious_safe')
        model_id: Model that processed the request
        route_confidence: Confidence score of route match (0.0-1.0)
        routing_latency_ms: Time for routing decision in milliseconds
        inference_latency_ms: Time for model inference in milliseconds
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        actual_cost_usd: Actual cost with semantic routing
        hypothetical_cost_usd: Hypothetical cost if GPT-4o was used
        verdict: Moderation verdict ('safe', 'flagged', 'requires_review')
    """

    timestamp: float
    route_name: str
    model_id: str
    route_confidence: float
    routing_latency_ms: float
    inference_latency_ms: float
    input_tokens: int
    output_tokens: int
    actual_cost_usd: float
    hypothetical_cost_usd: float
    verdict: str

    @property
    def total_tokens(self) -> int:
        """Total tokens processed (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def total_latency_ms(self) -> float:
        """Total latency (routing + inference)."""
        return self.routing_latency_ms + self.inference_latency_ms


@dataclass
class _RouteAggregate:
    """Internal aggregate for per-route metrics."""

    count: int = 0
    confidences: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)


@dataclass
class _ModelAggregate:
    """Internal aggregate for per-model metrics."""

    count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics snapshot for reporting.

    Provides pre-computed aggregates for efficient API responses.
    All fields are snapshots captured at a specific moment.

    Attributes:
        total_requests: Total number of requests processed
        total_actual_cost: Cumulative actual cost in USD
        total_hypothetical_cost: Cumulative hypothetical cost in USD
        total_input_tokens: Total input tokens across all requests
        total_output_tokens: Total output tokens across all requests
        requests_by_route: Request count and metrics per route
        requests_by_model: Request count and metrics per model
        routing_latencies: List of routing latencies for percentile calculation
        inference_latencies: List of inference latencies for percentile calculation
        verdicts: Count of each verdict type
    """

    total_requests: int = 0
    total_actual_cost: float = 0.0
    total_hypothetical_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # By-dimension aggregates
    requests_by_route: dict[str, _RouteAggregate] = field(
        default_factory=lambda: defaultdict(_RouteAggregate)
    )
    requests_by_model: dict[str, _ModelAggregate] = field(
        default_factory=lambda: defaultdict(_ModelAggregate)
    )

    # Latency lists for percentile calculations
    routing_latencies: list[float] = field(default_factory=list)
    inference_latencies: list[float] = field(default_factory=list)

    # Verdict distribution
    verdicts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class MetricsStore:
    """
    Thread-safe in-memory metrics storage.

    Stores individual request metrics and provides aggregation
    for reporting. Uses threading.Lock for safe concurrent access
    in FastAPI's async environment.

    Designed for single-process deployment. For multi-process or
    distributed deployments, consider Redis or a time-series database.

    Example:
        store = MetricsStore()
        store.record(RequestMetric(
            timestamp=time.time(),
            route_name="obvious_safe",
            model_id="llama-3.1-8b",
            ...
        ))
        aggregated = store.get_aggregated()
        print(f"Total requests: {aggregated.total_requests}")
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize the metrics store.

        Args:
            max_history: Maximum individual metrics to retain.
                         Older metrics are discarded when limit is reached.
                         Aggregates are preserved regardless of this limit.
        """
        self._lock = threading.Lock()
        self._metrics: list[RequestMetric] = []
        self._max_history = max_history

        self._total_requests: int = 0
        self._total_actual_cost: float = 0.0
        self._total_hypothetical_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        self._by_route: dict[str, _RouteAggregate] = defaultdict(_RouteAggregate)
        self._by_model: dict[str, _ModelAggregate] = defaultdict(_ModelAggregate)

        self._routing_latencies: list[float] = []
        self._inference_latencies: list[float] = []

        self._verdicts: dict[str, int] = defaultdict(int)

    def record(self, metric: RequestMetric) -> None:
        """
        Record a new request metric.

        Thread-safe. Updates both raw history and pre-computed aggregates.

        Args:
            metric: The request metric to record
        """
        with self._lock:
            self._metrics.append(metric)
            if len(self._metrics) > self._max_history:
                self._metrics = self._metrics[-self._max_history :]

            self._total_requests += 1
            self._total_actual_cost += metric.actual_cost_usd
            self._total_hypothetical_cost += metric.hypothetical_cost_usd
            self._total_input_tokens += metric.input_tokens
            self._total_output_tokens += metric.output_tokens

            route_agg = self._by_route[metric.route_name]
            route_agg.count += 1
            route_agg.confidences.append(metric.route_confidence)
            route_agg.latencies.append(metric.routing_latency_ms)

            model_agg = self._by_model[metric.model_id]
            model_agg.count += 1
            model_agg.total_tokens += metric.total_tokens
            model_agg.total_cost += metric.actual_cost_usd
            model_agg.latencies.append(metric.inference_latency_ms)

            self._routing_latencies.append(metric.routing_latency_ms)
            self._inference_latencies.append(metric.inference_latency_ms)

            if len(self._routing_latencies) > self._max_history:
                self._routing_latencies = self._routing_latencies[-self._max_history :]
            if len(self._inference_latencies) > self._max_history:
                self._inference_latencies = self._inference_latencies[
                    -self._max_history :
                ]

            self._verdicts[metric.verdict] += 1

    def get_aggregated(self) -> AggregatedMetrics:
        """
        Get current aggregated metrics.

        Thread-safe. Returns a snapshot of current state.
        The returned object is a copy and safe to use outside the lock.

        Returns:
            AggregatedMetrics snapshot
        """
        with self._lock:
            by_route_copy = {
                route: _RouteAggregate(
                    count=agg.count,
                    confidences=list(agg.confidences),
                    latencies=list(agg.latencies),
                )
                for route, agg in self._by_route.items()
            }

            by_model_copy = {
                model: _ModelAggregate(
                    count=agg.count,
                    total_tokens=agg.total_tokens,
                    total_cost=agg.total_cost,
                    latencies=list(agg.latencies),
                )
                for model, agg in self._by_model.items()
            }

            return AggregatedMetrics(
                total_requests=self._total_requests,
                total_actual_cost=self._total_actual_cost,
                total_hypothetical_cost=self._total_hypothetical_cost,
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                requests_by_route=by_route_copy,
                requests_by_model=by_model_copy,
                routing_latencies=list(self._routing_latencies),
                inference_latencies=list(self._inference_latencies),
                verdicts=dict(self._verdicts),
            )

    def get_recent(self, count: int = 100) -> list[RequestMetric]:
        """
        Get most recent request metrics.

        Thread-safe. Returns copies of the most recent metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent RequestMetric objects
        """
        with self._lock:
            return list(self._metrics[-count:])

    def reset(self) -> None:
        """
        Reset all metrics.

        Thread-safe. Clears all stored data and aggregates.
        Primarily used for testing.
        """
        with self._lock:
            self._metrics.clear()
            self._total_requests = 0
            self._total_actual_cost = 0.0
            self._total_hypothetical_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._by_route.clear()
            self._by_model.clear()
            self._routing_latencies.clear()
            self._inference_latencies.clear()
            self._verdicts.clear()


_store: MetricsStore | None = None


def get_metrics_store() -> MetricsStore:
    """
    Get the global metrics store instance.

    Returns:
        Singleton MetricsStore instance
    """
    global _store
    if _store is None:
        _store = MetricsStore()
    return _store
