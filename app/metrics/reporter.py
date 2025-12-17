"""
Metrics Reporter for API Responses

Transforms raw aggregated metrics into structured API responses
with computed fields like savings percentage and averages.

The reporter bridges the internal metrics representation to the
Pydantic schemas used by the REST API.
"""

from app.metrics.store import get_metrics_store, MetricsStore
from app.schemas.moderation import MetricsResponse, RouteMetrics, ModelMetrics


class MetricsReporter:
    """
    Generate metrics reports from aggregated data.

    Transforms internal metrics representation into API-compatible
    MetricsResponse objects with computed statistics.

    Example:
        reporter = MetricsReporter()
        response = reporter.generate_report()
        return response  # Ready for JSON serialization
    """

    def __init__(self, store: MetricsStore | None = None):
        """
        Initialize the reporter.

        Args:
            store: MetricsStore instance to report from.
                   If None, uses the global singleton.
        """
        self._store = store or get_metrics_store()

    def generate_report(self) -> MetricsResponse:
        """
        Generate a complete metrics report.

        Retrieves current aggregated metrics and transforms them
        into a MetricsResponse with computed statistics.

        Returns:
            MetricsResponse ready for API serialization
        """
        agg = self._store.get_aggregated()

        routes: dict[str, RouteMetrics] = {}
        for route_name, route_data in agg.requests_by_route.items():
            confidences = route_data.confidences
            latencies = route_data.latencies

            routes[route_name] = RouteMetrics(
                route_name=route_name,
                request_count=route_data.count,
                avg_confidence=(
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                avg_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
            )

        models: dict[str, ModelMetrics] = {}
        for model_id, model_data in agg.requests_by_model.items():
            latencies = model_data.latencies

            models[model_id] = ModelMetrics(
                model_id=model_id,
                request_count=model_data.count,
                total_tokens=model_data.total_tokens,
                total_cost_usd=model_data.total_cost,
                avg_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
            )

        if agg.total_hypothetical_cost > 0:
            savings_percent = (
                (agg.total_hypothetical_cost - agg.total_actual_cost)
                / agg.total_hypothetical_cost
                * 100
            )
        else:
            savings_percent = 0.0

        avg_routing = (
            sum(agg.routing_latencies) / len(agg.routing_latencies)
            if agg.routing_latencies
            else 0.0
        )
        avg_inference = (
            sum(agg.inference_latencies) / len(agg.inference_latencies)
            if agg.inference_latencies
            else 0.0
        )

        return MetricsResponse(
            total_requests=agg.total_requests,
            requests_by_route=routes,
            requests_by_model=models,
            total_cost_usd=round(agg.total_actual_cost, 6),
            hypothetical_cost_usd=round(agg.total_hypothetical_cost, 6),
            cost_savings_percent=round(savings_percent, 2),
            avg_routing_latency_ms=round(avg_routing, 2),
            avg_inference_latency_ms=round(avg_inference, 2),
        )

    def get_tier_distribution(self) -> dict[str, float]:
        """
        Get traffic distribution by model tier.

        Calculates the percentage of requests handled by each
        model tier (tier1, tier2, specialist).

        Returns:
            Dictionary mapping tier names to percentage of total traffic
        """
        agg = self._store.get_aggregated()

        tier_counts: dict[str, int] = {"tier1": 0, "tier2": 0, "specialist": 0}

        tier_map = {
            "llama-3.1-8b": "tier1",
            "gpt-4o": "tier2",
            "llama-guard-4": "specialist",
            "llama-4-maverick": "specialist",
        }

        for model_id, model_data in agg.requests_by_model.items():
            tier = tier_map.get(model_id, "tier1")
            tier_counts[tier] += model_data.count

        total = sum(tier_counts.values())
        if total == 0:
            return {"tier1": 0.0, "tier2": 0.0, "specialist": 0.0}

        return {
            tier: round(count / total * 100, 1) for tier, count in tier_counts.items()
        }

    def get_verdict_distribution(self) -> dict[str, int]:
        """
        Get distribution of moderation verdicts.

        Returns:
            Dictionary mapping verdict names to counts
        """
        agg = self._store.get_aggregated()
        return dict(agg.verdicts)


def get_reporter(store: MetricsStore | None = None) -> MetricsReporter:
    """
    Get a metrics reporter instance.

    Args:
        store: Optional MetricsStore to use. Defaults to global singleton.

    Returns:
        MetricsReporter instance
    """
    return MetricsReporter(store)
