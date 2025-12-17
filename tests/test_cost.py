"""
Cost Calculation Tests

Validates cost calculation accuracy and savings computation.
Ensures the router achieves the >60% savings target with mixed traffic.

Test Categories:
1. TestCostBreakdown - CostBreakdown dataclass validation
2. TestCostCalculator - Per-request cost calculation
3. TestSavingsValidation - 60% savings target validation

See: docs/09-testing.md for detailed test specifications.
"""

import pytest

from app.metrics.cost import CostCalculator, CostBreakdown, get_cost_calculator
from app.registry.models import get_model_registry


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_total_tokens_property(self):
        """Verify total_tokens property calculates correctly."""
        breakdown = CostBreakdown(
            input_tokens=100,
            output_tokens=50,
            input_cost_usd=0.0001,
            output_cost_usd=0.00005,
            actual_cost_usd=0.00015,
            hypothetical_cost_usd=0.01,
            savings_usd=0.00985,
            savings_percent=98.5,
            model_used="llama-3.1-8b",
        )

        assert breakdown.total_tokens == 150

    def test_cost_breakdown_fields(self):
        """Verify all fields are accessible."""
        breakdown = CostBreakdown(
            input_tokens=200,
            output_tokens=100,
            input_cost_usd=0.00001,
            output_cost_usd=0.000008,
            actual_cost_usd=0.000018,
            hypothetical_cost_usd=0.0025,
            savings_usd=0.002482,
            savings_percent=99.28,
            model_used="llama-3.1-8b",
        )

        assert breakdown.input_tokens == 200
        assert breakdown.output_tokens == 100
        assert breakdown.input_cost_usd > 0
        assert breakdown.output_cost_usd > 0
        assert breakdown.actual_cost_usd > 0
        assert breakdown.hypothetical_cost_usd > breakdown.actual_cost_usd
        assert breakdown.savings_usd > 0
        assert breakdown.savings_percent > 0


class TestCostCalculator:
    """Tests for CostCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Get a fresh CostCalculator instance."""
        return CostCalculator()

    def test_tier1_cost_calculation(self, calculator, tier1_model):
        """Tier 1 model cost is calculated correctly."""
        breakdown = calculator.calculate(
            model=tier1_model, input_tokens=1000, output_tokens=500
        )

        # Expected: (1000/1M * 0.05) + (500/1M * 0.08) = 0.00009
        expected_input = 1000 / 1_000_000 * 0.05
        expected_output = 500 / 1_000_000 * 0.08
        expected_total = expected_input + expected_output

        assert breakdown.actual_cost_usd == pytest.approx(expected_total, rel=0.01)
        assert breakdown.input_cost_usd == pytest.approx(expected_input, rel=0.01)
        assert breakdown.output_cost_usd == pytest.approx(expected_output, rel=0.01)

    def test_tier2_cost_calculation(self, calculator, tier2_model):
        """Tier 2 model cost is calculated correctly."""
        breakdown = calculator.calculate(
            model=tier2_model, input_tokens=1000, output_tokens=500
        )

        # Expected: (1000/1M * 5.00) + (500/1M * 15.00) = 0.0125
        expected_input = 1000 / 1_000_000 * 5.00
        expected_output = 500 / 1_000_000 * 15.00
        expected_total = expected_input + expected_output

        assert breakdown.actual_cost_usd == pytest.approx(expected_total, rel=0.01)

    def test_guard_model_cost_calculation(self, calculator, guard_model):
        """Specialist guard model cost is calculated correctly."""
        breakdown = calculator.calculate(
            model=guard_model, input_tokens=1000, output_tokens=500
        )

        # Expected: (1000/1M * 0.20) + (500/1M * 0.20) = 0.0003
        expected = (1000 + 500) / 1_000_000 * 0.20

        assert breakdown.actual_cost_usd == pytest.approx(expected, rel=0.01)

    def test_multilingual_model_cost_calculation(self, calculator, multilingual_model):
        """Multilingual model cost is calculated correctly."""
        breakdown = calculator.calculate(
            model=multilingual_model, input_tokens=1000, output_tokens=500
        )

        # Expected: (1000/1M * 0.20) + (500/1M * 0.60) = 0.0005
        expected_input = 1000 / 1_000_000 * 0.20
        expected_output = 500 / 1_000_000 * 0.60
        expected_total = expected_input + expected_output

        assert breakdown.actual_cost_usd == pytest.approx(expected_total, rel=0.01)

    def test_zero_tokens_returns_zero_cost(self, calculator, tier1_model):
        """Zero tokens results in zero cost."""
        breakdown = calculator.calculate(
            model=tier1_model, input_tokens=0, output_tokens=0
        )

        assert breakdown.actual_cost_usd == 0.0
        assert breakdown.input_cost_usd == 0.0
        assert breakdown.output_cost_usd == 0.0
        assert breakdown.total_tokens == 0

    def test_hypothetical_cost_uses_gpt4o_baseline(self, calculator, tier1_model):
        """Hypothetical cost uses GPT-4o pricing."""
        breakdown = calculator.calculate(
            model=tier1_model, input_tokens=1000, output_tokens=500
        )

        # Hypothetical: (1000/1M * 5.00) + (500/1M * 15.00) = 0.0125
        expected_hypothetical = (1000 / 1_000_000 * 5.00) + (500 / 1_000_000 * 15.00)

        assert breakdown.hypothetical_cost_usd == pytest.approx(
            expected_hypothetical, rel=0.01
        )

    def test_savings_calculation_tier1_vs_baseline(self, calculator, tier1_model):
        """Savings percentage for Tier 1 vs GPT-4o baseline."""
        breakdown = calculator.calculate(
            model=tier1_model, input_tokens=1000, output_tokens=500
        )

        # Tier 1 should have >99% savings vs GPT-4o baseline
        # Actual: ~0.00009, Hypothetical: ~0.0125
        # Savings: (0.0125 - 0.00009) / 0.0125 = 99.28%
        assert breakdown.savings_percent > 99.0
        assert breakdown.savings_usd > 0

    def test_tier2_has_zero_savings(self, calculator, tier2_model):
        """Tier 2 (GPT-4o) has 0% savings vs baseline."""
        breakdown = calculator.calculate(
            model=tier2_model, input_tokens=1000, output_tokens=500
        )

        # GPT-4o is the baseline, so savings should be ~0%
        assert breakdown.savings_percent == pytest.approx(0.0, abs=0.1)
        assert breakdown.actual_cost_usd == pytest.approx(
            breakdown.hypothetical_cost_usd, rel=0.01
        )

    def test_calculate_by_model_id(self, calculator):
        """Calculate cost using model ID lookup."""
        breakdown = calculator.calculate_by_model_id(
            model_id="llama-3.1-8b", input_tokens=1000, output_tokens=500
        )

        assert breakdown is not None
        assert breakdown.model_used == "llama-3.1-8b"
        assert breakdown.actual_cost_usd > 0

    def test_calculate_by_model_id_not_found(self, calculator):
        """Returns None for unknown model ID."""
        breakdown = calculator.calculate_by_model_id(
            model_id="nonexistent-model", input_tokens=1000, output_tokens=500
        )

        assert breakdown is None


class TestSavingsValidation:
    """Validate the 60%+ savings target with mixed traffic."""

    def test_mixed_traffic_achieves_60_percent_savings(self):
        """
        Simulate 70/20/10 traffic mix and verify >60% savings.

        Traffic distribution:
        - 70% Tier 1 (obvious_harm + obvious_safe): llama-3.1-8b
        - 20% Tier 2 (ambiguous_risk): gpt-4o
        - 10% Specialist (system_attack + non_english): llama-guard-4
        """
        calculator = CostCalculator()
        registry = get_model_registry()

        total_actual = 0.0
        total_hypothetical = 0.0

        # Average tokens per request
        avg_input = 200
        avg_output = 100

        # 70 Tier 1 requests
        tier1 = registry.get_model("llama-3.1-8b")
        for _ in range(70):
            breakdown = calculator.calculate(tier1, avg_input, avg_output)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        # 20 Tier 2 requests
        tier2 = registry.get_model("gpt-4o")
        for _ in range(20):
            breakdown = calculator.calculate(tier2, avg_input, avg_output)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        # 10 Specialist requests (using guard model)
        specialist = registry.get_model("llama-guard-4")
        for _ in range(10):
            breakdown = calculator.calculate(specialist, avg_input, avg_output)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        savings_percent = (total_hypothetical - total_actual) / total_hypothetical * 100

        assert savings_percent > 60.0, (
            f"Savings {savings_percent:.1f}% is below 60% target. "
            f"Total actual: ${total_actual:.6f}, "
            f"Total hypothetical: ${total_hypothetical:.6f}"
        )

    def test_80_20_traffic_mix(self):
        """
        Test 80/20 traffic mix (target scenario from docs).

        80% cheap models, 20% expensive = should achieve 60%+ savings.
        """
        calculator = CostCalculator()
        registry = get_model_registry()

        total_actual = 0.0
        total_hypothetical = 0.0

        avg_input = 200
        avg_output = 100

        # 80% Tier 1
        tier1 = registry.get_model("llama-3.1-8b")
        for _ in range(80):
            breakdown = calculator.calculate(tier1, avg_input, avg_output)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        # 20% Tier 2
        tier2 = registry.get_model("gpt-4o")
        for _ in range(20):
            breakdown = calculator.calculate(tier2, avg_input, avg_output)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        savings_percent = (total_hypothetical - total_actual) / total_hypothetical * 100

        assert (
            savings_percent > 60.0
        ), f"80/20 mix achieved only {savings_percent:.1f}% savings"

    def test_worst_case_all_tier2(self):
        """100% Tier 2 traffic has 0% savings."""
        calculator = CostCalculator()
        registry = get_model_registry()

        tier2 = registry.get_model("gpt-4o")

        total_actual = 0.0
        total_hypothetical = 0.0

        for _ in range(100):
            breakdown = calculator.calculate(tier2, 200, 100)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        savings_percent = (total_hypothetical - total_actual) / total_hypothetical * 100

        # 100% GPT-4o = 0% savings (it IS the baseline)
        assert savings_percent < 1.0

    def test_best_case_all_tier1(self):
        """100% Tier 1 traffic achieves maximum savings."""
        calculator = CostCalculator()
        registry = get_model_registry()

        tier1 = registry.get_model("llama-3.1-8b")

        total_actual = 0.0
        total_hypothetical = 0.0

        for _ in range(100):
            breakdown = calculator.calculate(tier1, 200, 100)
            total_actual += breakdown.actual_cost_usd
            total_hypothetical += breakdown.hypothetical_cost_usd

        savings_percent = (total_hypothetical - total_actual) / total_hypothetical * 100

        # 100% Tier 1 should achieve >99% savings
        assert savings_percent > 99.0


class TestCostCalculatorSingleton:
    """Tests for the cost calculator singleton."""

    def test_get_cost_calculator_returns_instance(self):
        """get_cost_calculator returns a CostCalculator instance."""
        calculator = get_cost_calculator()
        assert isinstance(calculator, CostCalculator)

    def test_get_cost_calculator_returns_same_instance(self):
        """get_cost_calculator returns the same instance."""
        calc1 = get_cost_calculator()
        calc2 = get_cost_calculator()

        # Note: Due to reset_singletons fixture, this may create new instances
        # but within a single test they should be the same
        assert calc1 is calc2
