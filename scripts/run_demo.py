#!/usr/bin/env python3
"""
Demo Runner Script

Processes the demo dataset through the Sentinel-Triage moderation pipeline
and reports results including routing accuracy and cost savings.

This script:
1. Loads the demo dataset using DatasetLoader
2. Processes each sample through the RouterEngine
3. Tracks routing accuracy per route
4. Calculates cost metrics (actual vs hypothetical)
5. Generates a summary report

Usage:
    python scripts/run_demo.py                    # Run all samples
    python scripts/run_demo.py --route obvious_harm  # Filter by route
    python scripts/run_demo.py --difficulty hard     # Filter by difficulty
    python scripts/run_demo.py --verbose             # Show each sample result
    python scripts/run_demo.py --dry-run             # Validate dataset only
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.data.loader import DatasetLoader, Sample, VALID_ROUTES, VALID_DIFFICULTIES

# Lazy imports for router components (only needed when actually routing)
RouterEngine = None
RouteChoice = None
get_model_registry = None


def _import_router_components():
    """Import router components on demand."""
    global RouterEngine, RouteChoice, get_model_registry
    if RouterEngine is None:
        from app.router.engine import RouterEngine as _RouterEngine, RouteChoice as _RouteChoice
        from app.registry.models import get_model_registry as _get_model_registry
        RouterEngine = _RouterEngine
        RouteChoice = _RouteChoice
        get_model_registry = _get_model_registry


@dataclass
class SampleResult:
    """Result of processing a single sample."""

    sample: Sample
    route_choice: RouteChoice
    is_correct: bool
    expected_route: str
    actual_route: str


@dataclass
class RouteStats:
    """Statistics for a single route."""

    correct: int = 0
    total: int = 0
    total_latency_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0.0


@dataclass
class DemoResults:
    """Aggregate results from demo run."""

    total_samples: int = 0
    correct_routes: int = 0
    incorrect_routes: int = 0
    total_latency_ms: float = 0.0
    by_route: dict[str, RouteStats] = field(default_factory=dict)
    mismatches: list[SampleResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_routes / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_samples if self.total_samples > 0 else 0.0


# Estimated tokens per moderation request (average)
ESTIMATED_INPUT_TOKENS = 100
ESTIMATED_OUTPUT_TOKENS = 50

# GPT-4o baseline pricing per 1M tokens
GPT4O_INPUT_PER_1M = 5.00
GPT4O_OUTPUT_PER_1M = 15.00


def estimate_costs(results: DemoResults) -> dict:
    """
    Estimate cost savings based on routing distribution.

    Uses model pricing from registry to calculate actual costs vs
    hypothetical GPT-4o-only costs.
    """
    _import_router_components()
    registry = get_model_registry()

    # Calculate hypothetical cost (all through GPT-4o)
    total_input_tokens = results.total_samples * ESTIMATED_INPUT_TOKENS
    total_output_tokens = results.total_samples * ESTIMATED_OUTPUT_TOKENS

    hypothetical_cost = (
        (total_input_tokens / 1_000_000) * GPT4O_INPUT_PER_1M +
        (total_output_tokens / 1_000_000) * GPT4O_OUTPUT_PER_1M
    )

    # Calculate actual cost based on routing
    actual_cost = 0.0
    cost_by_model: dict[str, float] = {}

    for route, stats in results.by_route.items():
        model = registry.get_model_for_route(route)
        if model:
            route_input_tokens = stats.total * ESTIMATED_INPUT_TOKENS
            route_output_tokens = stats.total * ESTIMATED_OUTPUT_TOKENS
            route_cost = (
                (route_input_tokens / 1_000_000) * model.cost_per_1m_input_tokens +
                (route_output_tokens / 1_000_000) * model.cost_per_1m_output_tokens
            )
            actual_cost += route_cost
            cost_by_model[model.model_id] = cost_by_model.get(model.model_id, 0) + route_cost

    savings = hypothetical_cost - actual_cost
    savings_percent = (savings / hypothetical_cost * 100) if hypothetical_cost > 0 else 0

    return {
        "hypothetical_cost_usd": hypothetical_cost,
        "actual_cost_usd": actual_cost,
        "savings_usd": savings,
        "savings_percent": savings_percent,
        "cost_by_model": cost_by_model,
    }


async def run_demo(
    samples: list[Sample],
    verbose: bool = False
) -> DemoResults:
    """
    Run the demo dataset through the router.

    Args:
        samples: List of samples to process
        verbose: Whether to print each sample result

    Returns:
        DemoResults with accuracy and timing statistics
    """
    _import_router_components()

    print("\nInitializing Router Engine...")
    engine = RouterEngine()
    await engine.initialize()
    print(f"Router initialized in {engine.initialization_latency_ms:.0f}ms")

    results = DemoResults(total_samples=len(samples))
    start_time = time.time()

    print(f"\nProcessing {len(samples)} samples...")
    print("-" * 60)

    for i, sample in enumerate(samples, 1):
        # Route the content
        route_choice = await engine.route(sample.content)

        # Check correctness
        is_correct = route_choice.route_name == sample.expected_route

        # Create result
        sample_result = SampleResult(
            sample=sample,
            route_choice=route_choice,
            is_correct=is_correct,
            expected_route=sample.expected_route,
            actual_route=route_choice.route_name or "none",
        )

        # Update stats
        if is_correct:
            results.correct_routes += 1
        else:
            results.incorrect_routes += 1
            results.mismatches.append(sample_result)

        results.total_latency_ms += route_choice.latency_ms

        # Update per-route stats
        route = sample.expected_route
        if route not in results.by_route:
            results.by_route[route] = RouteStats()
        results.by_route[route].total += 1
        results.by_route[route].total_latency_ms += route_choice.latency_ms
        if is_correct:
            results.by_route[route].correct += 1

        # Verbose output
        if verbose:
            status = "OK" if is_correct else "MISMATCH"
            print(
                f"[{i:3d}/{len(samples)}] {status:8s} | "
                f"Expected: {sample.expected_route:15s} | "
                f"Got: {route_choice.route_name or 'none':15s} | "
                f"Conf: {route_choice.confidence:.2f} | "
                f"{route_choice.latency_ms:.1f}ms"
            )

        # Progress indicator (non-verbose)
        if not verbose and i % 10 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")

    results.elapsed_seconds = time.time() - start_time

    return results


def print_report(results: DemoResults, show_mismatches: bool = True) -> None:
    """Print a formatted report of demo results."""

    print("\n" + "=" * 60)
    print("SENTINEL-TRIAGE DEMO RESULTS")
    print("=" * 60)

    # Overall accuracy
    print(f"\nRouting Accuracy: {results.accuracy:.1%}")
    print(f"  Correct:   {results.correct_routes}")
    print(f"  Incorrect: {results.incorrect_routes}")

    # Timing
    print(f"\nTiming:")
    print(f"  Total time:      {results.elapsed_seconds:.2f}s")
    print(f"  Avg per sample:  {results.avg_latency_ms:.1f}ms")

    # Per-route breakdown
    print("\nBy Route:")
    print(f"  {'Route':<18} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Avg ms':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")

    for route in sorted(results.by_route.keys()):
        stats = results.by_route[route]
        print(
            f"  {route:<18} {stats.accuracy:>9.1%} "
            f"{stats.correct:>8} {stats.total:>6} "
            f"{stats.avg_latency_ms:>7.1f}"
        )

    # Cost analysis
    if results.total_samples > 0:
        costs = estimate_costs(results)
        print("\nCost Analysis (estimated):")
        print(f"  Hypothetical (GPT-4o only): ${costs['hypothetical_cost_usd']:.6f}")
        print(f"  Actual (routed):            ${costs['actual_cost_usd']:.6f}")
        print(f"  Savings:                    ${costs['savings_usd']:.6f} ({costs['savings_percent']:.1f}%)")

        if costs['cost_by_model']:
            print("\n  Cost by Model:")
            for model_id, cost in sorted(costs['cost_by_model'].items()):
                print(f"    {model_id:<20} ${cost:.6f}")

    # Mismatches
    if show_mismatches and results.mismatches:
        print(f"\nMismatches ({len(results.mismatches)}):")
        for m in results.mismatches[:10]:  # Show first 10
            print(f"\n  [{m.sample.id}] {m.sample.category}")
            print(f"    Content:  \"{m.sample.content[:60]}{'...' if len(m.sample.content) > 60 else ''}\"")
            print(f"    Expected: {m.expected_route}")
            print(f"    Got:      {m.actual_route} (conf: {m.route_choice.confidence:.2f})")

        if len(results.mismatches) > 10:
            print(f"\n  ... and {len(results.mismatches) - 10} more mismatches")

    print("\n" + "=" * 60)


def main():
    """Main entry point for the demo runner."""

    parser = argparse.ArgumentParser(
        description="Run Sentinel-Triage demo dataset through the router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demo.py                     Run all samples
  python scripts/run_demo.py --route obvious_harm   Filter by route
  python scripts/run_demo.py --difficulty hard      Filter by difficulty
  python scripts/run_demo.py --verbose              Show each result
  python scripts/run_demo.py --dry-run              Validate only
        """
    )

    parser.add_argument(
        "--route",
        choices=sorted(VALID_ROUTES),
        help="Filter samples by route"
    )
    parser.add_argument(
        "--difficulty",
        choices=sorted(VALID_DIFFICULTIES),
        help="Filter samples by difficulty"
    )
    parser.add_argument(
        "--category",
        help="Filter samples by category"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show each sample result"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset without routing"
    )
    parser.add_argument(
        "--dataset",
        default="data/sample_inputs.json",
        help="Path to dataset file (default: data/sample_inputs.json)"
    )
    parser.add_argument(
        "--no-mismatches",
        action="store_true",
        help="Don't show mismatch details in report"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sentinel-Triage Demo Runner")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    loader = DatasetLoader(path=args.dataset)

    try:
        samples = loader.load()
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {args.dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")

    # Validate dataset
    validation = loader.validate()
    if not validation.is_valid:
        print("\nDataset validation FAILED:")
        for error in validation.errors:
            print(f"  - {error}")
        if not args.dry_run:
            print("\nFix validation errors before running demo.")
            sys.exit(1)

    if validation.warnings:
        print("\nDataset warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    # Show distribution
    print("\nDataset distribution:")
    dist = loader.get_distribution()
    for route, count in sorted(dist.items()):
        pct = count / len(samples) * 100
        print(f"  {route:<18} {count:>3} ({pct:>5.1f}%)")

    # Dry run - just validate
    if args.dry_run:
        print("\n--dry-run specified, skipping routing.")
        if validation.is_valid:
            print("Dataset validation PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    # Apply filters
    filtered_samples = list(samples)

    if args.route:
        filtered_samples = [s for s in filtered_samples if s.expected_route == args.route]
        print(f"\nFiltered to route '{args.route}': {len(filtered_samples)} samples")

    if args.difficulty:
        filtered_samples = [s for s in filtered_samples if s.difficulty == args.difficulty]
        print(f"Filtered to difficulty '{args.difficulty}': {len(filtered_samples)} samples")

    if args.category:
        filtered_samples = [s for s in filtered_samples if s.category == args.category]
        print(f"Filtered to category '{args.category}': {len(filtered_samples)} samples")

    if not filtered_samples:
        print("\nNo samples match the specified filters.")
        sys.exit(1)

    # Run demo
    results = asyncio.run(run_demo(filtered_samples, verbose=args.verbose))

    # Print report
    print_report(results, show_mismatches=not args.no_mismatches)

    # Exit code based on accuracy
    if results.accuracy < 0.80:
        print("\nWARNING: Routing accuracy below 80% threshold")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
