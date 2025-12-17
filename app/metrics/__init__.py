"""
Metrics Module: Cost Tracking, Storage, and Reporting

This module provides comprehensive cost tracking for the Sentinel-Triage
content moderation pipeline. It enables measurement of the core value
proposition: demonstrating >60% cost savings through semantic routing.

Components:
    CostCalculator: Calculate per-request costs using model registry pricing
    CostBreakdown: Detailed cost information for a single request
    MetricsStore: Thread-safe in-memory metrics aggregation
    RequestMetric: Individual request metric record
    AggregatedMetrics: Pre-computed aggregates for reporting
    MetricsReporter: Generate MetricsResponse for API endpoints

Usage:
    from app.metrics import get_metrics_store, get_cost_calculator, RequestMetric

    # Calculate cost for a request
    calculator = get_cost_calculator()
    cost = calculator.calculate_by_model_id(
        model_id="llama-3.1-8b",
        input_tokens=150,
        output_tokens=50
    )

    # Record the metric
    store = get_metrics_store()
    store.record(RequestMetric(
        timestamp=time.time(),
        route_name="obvious_safe",
        model_id="llama-3.1-8b",
        route_confidence=0.85,
        routing_latency_ms=25.0,
        inference_latency_ms=150.0,
        input_tokens=150,
        output_tokens=50,
        actual_cost_usd=cost.actual_cost_usd,
        hypothetical_cost_usd=cost.hypothetical_cost_usd,
        verdict="safe"
    ))

    # Generate report for API
    from app.metrics import MetricsReporter
    reporter = MetricsReporter()
    response = reporter.generate_report()  # Returns MetricsResponse

Singleton Access:
    get_cost_calculator(): Returns global CostCalculator instance
    get_metrics_store(): Returns global MetricsStore instance
"""

# Cost calculation
from app.metrics.cost import (
    CostCalculator,
    CostBreakdown,
    get_cost_calculator,
)

# Storage
from app.metrics.store import (
    MetricsStore,
    RequestMetric,
    AggregatedMetrics,
    get_metrics_store,
)

# Reporting
from app.metrics.reporter import (
    MetricsReporter,
    get_reporter,
)


__all__ = [
    # Cost calculation
    "CostCalculator",
    "CostBreakdown",
    "get_cost_calculator",
    # Storage
    "MetricsStore",
    "RequestMetric",
    "AggregatedMetrics",
    "get_metrics_store",
    # Reporting
    "MetricsReporter",
    "get_reporter",
]
