"""
Cost Calculator for Model Inference

Calculates actual and hypothetical costs for moderation requests,
enabling savings analysis and cost optimization decisions.

The CostCalculator uses model pricing from the registry and compares
actual costs against a GPT-4o baseline to demonstrate routing savings.
"""

from dataclasses import dataclass

from app.registry.models import ModelMetadata, get_model_registry
from app.config import get_settings


@dataclass
class CostBreakdown:
    """
    Detailed cost breakdown for a single moderation request.

    Captures both actual costs (using the routed model) and hypothetical
    costs (if GPT-4o processed everything) for savings analysis.

    Attributes:
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        input_cost_usd: Cost for input tokens in USD
        output_cost_usd: Cost for output tokens in USD
        actual_cost_usd: Total actual cost (input + output)
        hypothetical_cost_usd: Cost if GPT-4o processed this request
        savings_usd: Dollar amount saved by routing
        savings_percent: Percentage savings achieved
        model_used: ID of the model that processed the request
    """

    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    actual_cost_usd: float
    hypothetical_cost_usd: float
    savings_usd: float
    savings_percent: float
    model_used: str

    @property
    def total_tokens(self) -> int:
        """Total tokens processed (input + output)."""
        return self.input_tokens + self.output_tokens


class CostCalculator:
    """
    Calculate inference costs and savings.

    Uses model metadata from the registry for actual pricing and
    GPT-4o baseline pricing for hypothetical cost comparison.

    The calculator is thread-safe as it only performs read operations
    on the model registry and configuration.

    Example:
        calculator = CostCalculator()
        cost = calculator.calculate_by_model_id(
            model_id="llama-3.1-8b",
            input_tokens=150,
            output_tokens=50
        )
        print(f"Saved ${cost.savings_usd:.6f} ({cost.savings_percent:.1f}%)")
    """

    def __init__(self):
        """
        Initialize the cost calculator.

        Loads baseline GPT-4o pricing from settings for hypothetical
        cost calculations.
        """
        self._settings = get_settings()
        self._baseline_input_per_1m = 5.00  # $5/1M input tokens
        self._baseline_output_per_1m = 15.00  # $15/1M output tokens

    def calculate(
        self, model: ModelMetadata, input_tokens: int, output_tokens: int
    ) -> CostBreakdown:
        """
        Calculate cost breakdown for a request.

        Args:
            model: Model metadata with pricing information
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated

        Returns:
            Complete cost breakdown with savings calculation
        """
        input_cost = (input_tokens / 1_000_000) * model.cost_per_1m_input_tokens
        output_cost = (output_tokens / 1_000_000) * model.cost_per_1m_output_tokens
        actual_cost = input_cost + output_cost

        hypothetical_input = (input_tokens / 1_000_000) * self._baseline_input_per_1m
        hypothetical_output = (output_tokens / 1_000_000) * self._baseline_output_per_1m
        hypothetical_cost = hypothetical_input + hypothetical_output

        savings_usd = hypothetical_cost - actual_cost
        savings_percent = (
            (savings_usd / hypothetical_cost * 100) if hypothetical_cost > 0 else 0.0
        )

        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            actual_cost_usd=actual_cost,
            hypothetical_cost_usd=hypothetical_cost,
            savings_usd=savings_usd,
            savings_percent=savings_percent,
            model_used=model.model_id,
        )

    def calculate_by_model_id(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> CostBreakdown | None:
        """
        Calculate cost using model ID lookup.

        Convenience method that looks up the model in the registry
        and calculates the cost breakdown.

        Args:
            model_id: ID of the model in the registry
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated

        Returns:
            CostBreakdown if model found, None otherwise
        """
        registry = get_model_registry()
        model = registry.get_model(model_id)
        if model is None:
            return None
        return self.calculate(model, input_tokens, output_tokens)


_calculator: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    """
    Get the global cost calculator instance.

    Returns:
        Singleton CostCalculator instance
    """
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator()
    return _calculator
