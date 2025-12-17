"""
Demo Dataset Loader

Utilities for loading and validating the demo dataset for Sentinel-Triage.
The dataset provides curated samples for:
- Routing validation: Verify content routes correctly to expected models
- Cost analysis: Calculate actual vs hypothetical costs
- Demo scripting: Reproducible demonstrations
- Benchmark testing: Performance measurement baseline

Dataset Structure:
{
    "metadata": { version, created, total_samples, description },
    "samples": [
        { id, content, expected_route, expected_verdict, category, difficulty, notes }
    ]
}

Target Distribution (100 samples):
- obvious_safe: 40% (positive feedback, questions, neutral)
- obvious_harm: 25% (spam, harassment, profanity, threats)
- ambiguous_risk: 20% (sarcasm, metaphor, dark humor)
- system_attack: 10% (prompt injection, PII exposure)
- non_english: 5% (French, German, Spanish, Chinese, Japanese)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Sample:
    """
    Individual sample from the demo dataset.

    Attributes:
        id: Unique identifier (e.g., 'safe-001', 'harm-015')
        content: The text content to be moderated
        expected_route: Expected semantic route classification
        expected_verdict: Expected moderation verdict (safe/flagged/requires_review)
        category: Subcategory within the route (e.g., 'sarcasm', 'spam')
        difficulty: Routing difficulty level (easy/medium/hard)
        notes: Optional notes about the sample
    """

    id: str
    content: str
    expected_route: str
    expected_verdict: str
    category: str
    difficulty: str
    notes: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "expected_route": self.expected_route,
            "expected_verdict": self.expected_verdict,
            "category": self.category,
            "difficulty": self.difficulty,
            "notes": self.notes,
        }


@dataclass
class DatasetMetadata:
    """
    Metadata about the demo dataset.

    Attributes:
        version: Dataset version string
        created: Creation date (ISO format)
        total_samples: Total number of samples
        description: Brief description of the dataset
    """

    version: str
    created: str
    total_samples: int
    description: str


@dataclass
class ValidationResult:
    """
    Result of dataset validation.

    Attributes:
        is_valid: Whether the dataset passed all validation checks
        errors: List of validation error messages
        warnings: List of validation warning messages
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════


VALID_ROUTES = frozenset(
    {
        "obvious_safe",
        "obvious_harm",
        "ambiguous_risk",
        "system_attack",
        "non_english",
    }
)

VALID_VERDICTS = frozenset(
    {
        "safe",
        "flagged",
        "requires_review",
    }
)

VALID_DIFFICULTIES = frozenset(
    {
        "easy",
        "medium",
        "hard",
    }
)

# Target distribution percentages
TARGET_DISTRIBUTION = {
    "obvious_safe": 0.40,
    "obvious_harm": 0.25,
    "ambiguous_risk": 0.20,
    "system_attack": 0.10,
    "non_english": 0.05,
}

MINIMUM_SAMPLES = 50


class DatasetLoader:
    """
    Load and manage the demo dataset for Sentinel-Triage.

    Provides lazy loading with caching, filtering by route/difficulty/category,
    and comprehensive validation checks.

    Example:
        >>> loader = DatasetLoader()
        >>> samples = loader.load()
        >>> harm_samples = loader.get_by_route("obvious_harm")
        >>> hard_samples = loader.get_by_difficulty("hard")
        >>> dist = loader.get_distribution()
    """

    def __init__(self, path: str | Path = "data/sample_inputs.json"):
        """
        Initialize the dataset loader.

        Args:
            path: Path to the JSON dataset file (relative to project root or absolute)
        """
        self._path = Path(path)
        self._samples: list[Sample] = []
        self._metadata: DatasetMetadata | None = None
        self._loaded = False

    @property
    def path(self) -> Path:
        """Get the dataset file path."""
        return self._path

    @property
    def metadata(self) -> DatasetMetadata | None:
        """Get dataset metadata (None if not yet loaded)."""
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        """Check if dataset has been loaded."""
        return self._loaded

    def load(self) -> list[Sample]:
        """
        Load samples from the JSON dataset file.

        Returns cached samples if already loaded. Handles both the new
        metadata wrapper format and legacy flat array format.

        Returns:
            List of Sample objects

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If sample data is malformed
        """
        if self._loaded:
            return self._samples

        logger.info(f"Loading dataset from {self._path}")

        if not self._path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both wrapper format and legacy flat array format
        if isinstance(data, list):
            # Legacy format: flat array of samples
            logger.debug("Loading legacy flat array format")
            samples_data = data
            self._metadata = DatasetMetadata(
                version="1.0.0",
                created="unknown",
                total_samples=len(data),
                description="Legacy format dataset",
            )
        elif isinstance(data, dict):
            # New format: metadata wrapper
            logger.debug("Loading metadata wrapper format")
            if "metadata" in data:
                meta = data["metadata"]
                self._metadata = DatasetMetadata(
                    version=meta.get("version", "1.0.0"),
                    created=meta.get("created", "unknown"),
                    total_samples=meta.get("total_samples", 0),
                    description=meta.get("description", ""),
                )
            samples_data = data.get("samples", [])
        else:
            raise ValueError(
                f"Invalid dataset format: expected list or dict, got {type(data)}"
            )

        self._samples = []
        for i, sample_data in enumerate(samples_data):
            try:
                sample = Sample(
                    id=str(sample_data.get("id", f"sample-{i:03d}")),
                    content=sample_data["content"],
                    expected_route=sample_data["expected_route"],
                    expected_verdict=sample_data.get(
                        "expected_verdict", "requires_review"
                    ),
                    category=sample_data.get("category", "unknown"),
                    difficulty=sample_data.get("difficulty", "medium"),
                    notes=sample_data.get("notes"),
                )
                self._samples.append(sample)
            except KeyError as e:
                raise ValueError(f"Sample {i} missing required field: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._samples)} samples")

        return self._samples

    def reload(self) -> list[Sample]:
        """
        Force reload the dataset from file.

        Returns:
            Fresh list of Sample objects
        """
        self._loaded = False
        self._samples = []
        self._metadata = None
        return self.load()

    def get_by_route(self, route_name: str) -> list[Sample]:
        """
        Get samples for a specific semantic route.

        Args:
            route_name: Name of the route (e.g., 'obvious_harm')

        Returns:
            List of samples matching the route
        """
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.expected_route == route_name]

    def get_by_difficulty(self, difficulty: str) -> list[Sample]:
        """
        Get samples by difficulty level.

        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')

        Returns:
            List of samples matching the difficulty
        """
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.difficulty == difficulty]

    def get_by_category(self, category: str) -> list[Sample]:
        """
        Get samples by subcategory.

        Args:
            category: Category name (e.g., 'sarcasm', 'spam')

        Returns:
            List of samples matching the category
        """
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.category == category]

    def get_by_verdict(self, verdict: str) -> list[Sample]:
        """
        Get samples by expected verdict.

        Args:
            verdict: Expected verdict ('safe', 'flagged', 'requires_review')

        Returns:
            List of samples matching the verdict
        """
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.expected_verdict == verdict]

    def get_distribution(self) -> dict[str, int]:
        """
        Get sample count by route.

        Returns:
            Dictionary mapping route names to sample counts
        """
        if not self._loaded:
            self.load()

        distribution: dict[str, int] = {}
        for sample in self._samples:
            route = sample.expected_route
            distribution[route] = distribution.get(route, 0) + 1

        return distribution

    def get_categories(self) -> dict[str, list[str]]:
        """
        Get all categories grouped by route.

        Returns:
            Dictionary mapping route names to lists of categories
        """
        if not self._loaded:
            self.load()

        categories: dict[str, set[str]] = {}
        for sample in self._samples:
            route = sample.expected_route
            if route not in categories:
                categories[route] = set()
            categories[route].add(sample.category)

        return {route: sorted(cats) for route, cats in categories.items()}

    def validate(self) -> ValidationResult:
        """
        Validate dataset integrity.

        Checks:
        - Minimum sample count (50+)
        - All 5 routes represented
        - No duplicate sample IDs
        - Valid route names
        - Valid verdict values
        - Valid difficulty levels
        - Reasonable distribution (warnings only)

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not self._loaded:
            try:
                self.load()
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to load dataset: {e}"],
                )

        if len(self._samples) < MINIMUM_SAMPLES:
            errors.append(
                f"Dataset has {len(self._samples)} samples, "
                f"minimum required is {MINIMUM_SAMPLES}"
            )

        routes_present = {s.expected_route for s in self._samples}
        missing_routes = VALID_ROUTES - routes_present
        if missing_routes:
            errors.append(f"Missing routes: {sorted(missing_routes)}")

        invalid_routes = routes_present - VALID_ROUTES
        if invalid_routes:
            errors.append(f"Invalid routes found: {sorted(invalid_routes)}")

        # Check for duplicate IDs
        ids = [s.id for s in self._samples]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            errors.append(f"Duplicate sample IDs: {sorted(set(duplicates))}")

        invalid_verdicts = {
            s.expected_verdict
            for s in self._samples
            if s.expected_verdict not in VALID_VERDICTS
        }
        if invalid_verdicts:
            errors.append(f"Invalid verdicts found: {sorted(invalid_verdicts)}")

        invalid_difficulties = {
            s.difficulty
            for s in self._samples
            if s.difficulty not in VALID_DIFFICULTIES
        }
        if invalid_difficulties:
            errors.append(f"Invalid difficulties found: {sorted(invalid_difficulties)}")

        empty_content = [s.id for s in self._samples if not s.content.strip()]
        if empty_content:
            errors.append(f"Samples with empty content: {empty_content}")

        if len(self._samples) >= MINIMUM_SAMPLES:
            distribution = self.get_distribution()
            for route, target_pct in TARGET_DISTRIBUTION.items():
                actual_count = distribution.get(route, 0)
                actual_pct = actual_count / len(self._samples)
                # Warn if off by more than 10 percentage points
                if abs(actual_pct - target_pct) > 0.10:
                    warnings.append(
                        f"Route '{route}' has {actual_pct:.0%} of samples "
                        f"(target: {target_pct:.0%})"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def __len__(self) -> int:
        """Get total number of samples."""
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self):
        """Iterate over samples."""
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def __getitem__(self, index: int) -> Sample:
        """Get sample by index."""
        if not self._loaded:
            self.load()
        return self._samples[index]


@lru_cache(maxsize=1)
def get_dataset_loader(path: str = "data/sample_inputs.json") -> DatasetLoader:
    """
    Get the global DatasetLoader instance (singleton).

    Args:
        path: Path to the dataset file

    Returns:
        Cached DatasetLoader instance
    """
    return DatasetLoader(path=path)
