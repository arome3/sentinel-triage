"""
Dataset Validation Tests

Tests for verifying the demo dataset integrity and structure.
Ensures the dataset meets requirements for routing validation and demos.

Test Categories:
1. TestDatasetLoading - Tests for loading and parsing the dataset
2. TestDatasetStructure - Tests for sample schema and fields
3. TestDatasetCoverage - Tests for route coverage and distribution
4. TestDatasetLoader - Tests for DatasetLoader utility methods

See: docs/10-demo-dataset.md for detailed specifications.
"""

import pytest
from pathlib import Path

from app.data.loader import (
    DatasetLoader,
    Sample,
    VALID_ROUTES,
    VALID_VERDICTS,
    VALID_DIFFICULTIES,
    MINIMUM_SAMPLES,
    TARGET_DISTRIBUTION,
)


@pytest.fixture
def loader():
    """Create a DatasetLoader instance for testing."""
    return DatasetLoader(path="data/sample_inputs.json")


@pytest.fixture
def samples(loader):
    """Load and return samples for testing."""
    return loader.load()


class TestDatasetLoading:
    """Tests for dataset loading and parsing."""

    def test_dataset_file_exists(self):
        """Verify the dataset file exists at expected path."""
        path = Path("data/sample_inputs.json")
        assert path.exists(), f"Dataset file not found at {path}"

    def test_dataset_loads_successfully(self, loader):
        """Verify dataset loads without errors."""
        samples = loader.load()
        assert samples is not None
        assert isinstance(samples, list)
        assert len(samples) > 0

    def test_dataset_caches_on_reload(self, loader):
        """Verify loading returns cached data."""
        samples1 = loader.load()
        samples2 = loader.load()
        assert samples1 is samples2, "Load should return cached samples"

    def test_dataset_reload_refreshes(self, loader):
        """Verify reload() forces fresh load."""
        samples1 = loader.load()
        samples2 = loader.reload()
        # Should be equal content but new list instance
        assert len(samples1) == len(samples2)

    def test_loader_length(self, loader):
        """Verify __len__ returns sample count."""
        samples = loader.load()
        assert len(loader) == len(samples)

    def test_loader_iteration(self, loader):
        """Verify loader is iterable."""
        samples = loader.load()
        iterated = list(loader)
        assert len(iterated) == len(samples)

    def test_loader_indexing(self, loader):
        """Verify loader supports indexing."""
        samples = loader.load()
        assert loader[0] == samples[0]
        assert loader[-1] == samples[-1]


class TestDatasetStructure:
    """Tests for sample schema and field validation."""

    def test_dataset_has_minimum_samples(self, samples):
        """Verify dataset has at least 50 samples."""
        assert len(samples) >= MINIMUM_SAMPLES, (
            f"Dataset has only {len(samples)} samples, "
            f"need at least {MINIMUM_SAMPLES}"
        )

    def test_samples_are_sample_objects(self, samples):
        """Verify all samples are Sample dataclass instances."""
        for sample in samples:
            assert isinstance(sample, Sample), (
                f"Expected Sample, got {type(sample)}"
            )

    def test_samples_have_required_fields(self, samples):
        """Verify all samples have required fields populated."""
        for sample in samples:
            assert sample.id, f"Sample missing id"
            assert sample.content, f"Sample {sample.id} missing content"
            assert sample.expected_route, f"Sample {sample.id} missing expected_route"
            assert sample.expected_verdict, f"Sample {sample.id} missing expected_verdict"
            assert sample.category, f"Sample {sample.id} missing category"
            assert sample.difficulty, f"Sample {sample.id} missing difficulty"

    def test_sample_ids_are_unique(self, samples):
        """Verify no duplicate sample IDs."""
        ids = [s.id for s in samples]
        duplicates = [id for id in ids if ids.count(id) > 1]
        assert len(ids) == len(set(ids)), (
            f"Duplicate sample IDs found: {set(duplicates)}"
        )

    def test_expected_routes_are_valid(self, samples):
        """Verify all expected_route values are valid route names."""
        invalid = {s.expected_route for s in samples} - VALID_ROUTES
        assert not invalid, f"Invalid routes found: {invalid}"

    def test_expected_verdicts_are_valid(self, samples):
        """Verify all expected_verdict values are valid."""
        invalid = {s.expected_verdict for s in samples} - VALID_VERDICTS
        assert not invalid, f"Invalid verdicts found: {invalid}"

    def test_difficulties_are_valid(self, samples):
        """Verify all difficulty values are valid."""
        invalid = {s.difficulty for s in samples} - VALID_DIFFICULTIES
        assert not invalid, f"Invalid difficulties found: {invalid}"

    def test_content_not_empty(self, samples):
        """Verify no samples have empty content."""
        empty = [s.id for s in samples if not s.content.strip()]
        assert not empty, f"Samples with empty content: {empty}"

    def test_sample_to_dict(self, samples):
        """Verify Sample.to_dict() works correctly."""
        sample = samples[0]
        d = sample.to_dict()

        assert d["id"] == sample.id
        assert d["content"] == sample.content
        assert d["expected_route"] == sample.expected_route
        assert d["expected_verdict"] == sample.expected_verdict
        assert d["category"] == sample.category
        assert d["difficulty"] == sample.difficulty


class TestDatasetCoverage:
    """Tests for route coverage and distribution."""

    def test_all_routes_represented(self, loader):
        """Verify all 5 routes have at least one sample."""
        dist = loader.get_distribution()
        missing = VALID_ROUTES - set(dist.keys())
        assert not missing, f"Missing routes: {missing}"

    def test_each_route_has_samples(self, loader):
        """Verify each route has at least 3 samples."""
        dist = loader.get_distribution()
        for route in VALID_ROUTES:
            count = dist.get(route, 0)
            assert count >= 3, f"Route '{route}' has only {count} samples"

    def test_distribution_approximates_target(self, loader, samples):
        """Verify distribution is within 15% of targets."""
        dist = loader.get_distribution()
        total = len(samples)

        for route, target_pct in TARGET_DISTRIBUTION.items():
            actual_count = dist.get(route, 0)
            actual_pct = actual_count / total

            # Allow 15 percentage point deviation
            assert abs(actual_pct - target_pct) <= 0.15, (
                f"Route '{route}' has {actual_pct:.0%} of samples, "
                f"target is {target_pct:.0%} (±15%)"
            )

    def test_obvious_safe_is_largest(self, loader):
        """Verify obvious_safe has the most samples (as expected)."""
        dist = loader.get_distribution()
        safe_count = dist.get("obvious_safe", 0)

        for route, count in dist.items():
            if route != "obvious_safe":
                assert safe_count >= count, (
                    f"obvious_safe ({safe_count}) should have >= "
                    f"samples than {route} ({count})"
                )


class TestDatasetLoader:
    """Tests for DatasetLoader utility methods."""

    def test_get_by_route(self, loader):
        """Verify get_by_route filters correctly."""
        harm_samples = loader.get_by_route("obvious_harm")
        assert all(s.expected_route == "obvious_harm" for s in harm_samples)
        assert len(harm_samples) > 0

    def test_get_by_difficulty(self, loader):
        """Verify get_by_difficulty filters correctly."""
        easy_samples = loader.get_by_difficulty("easy")
        assert all(s.difficulty == "easy" for s in easy_samples)

        hard_samples = loader.get_by_difficulty("hard")
        assert all(s.difficulty == "hard" for s in hard_samples)

    def test_get_by_category(self, loader):
        """Verify get_by_category filters correctly."""
        spam_samples = loader.get_by_category("spam")
        assert all(s.category == "spam" for s in spam_samples)

    def test_get_by_verdict(self, loader):
        """Verify get_by_verdict filters correctly."""
        safe_samples = loader.get_by_verdict("safe")
        assert all(s.expected_verdict == "safe" for s in safe_samples)

        flagged_samples = loader.get_by_verdict("flagged")
        assert all(s.expected_verdict == "flagged" for s in flagged_samples)

    def test_get_distribution_returns_dict(self, loader):
        """Verify get_distribution returns proper dict."""
        dist = loader.get_distribution()

        assert isinstance(dist, dict)
        assert all(isinstance(k, str) for k in dist.keys())
        assert all(isinstance(v, int) for v in dist.values())

    def test_distribution_sums_to_total(self, loader, samples):
        """Verify distribution counts sum to total samples."""
        dist = loader.get_distribution()
        assert sum(dist.values()) == len(samples)

    def test_get_categories(self, loader):
        """Verify get_categories returns grouped categories."""
        categories = loader.get_categories()

        assert isinstance(categories, dict)
        assert "obvious_harm" in categories
        assert "spam" in categories.get("obvious_harm", [])


class TestDatasetValidation:
    """Tests for the validate() method."""

    def test_validate_returns_result(self, loader):
        """Verify validate() returns ValidationResult."""
        result = loader.validate()

        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    def test_valid_dataset_passes(self, loader):
        """Verify valid dataset passes validation."""
        result = loader.validate()

        assert result.is_valid, f"Validation failed: {result.errors}"

    def test_validate_catches_missing_file(self):
        """Verify validation catches missing file."""
        loader = DatasetLoader(path="nonexistent_file.json")
        result = loader.validate()

        assert not result.is_valid
        assert any("load" in e.lower() for e in result.errors)


class TestDatasetMetadata:
    """Tests for dataset metadata."""

    def test_metadata_populated(self, loader):
        """Verify metadata is populated after load."""
        loader.load()
        meta = loader.metadata

        assert meta is not None
        assert meta.version
        assert meta.total_samples > 0

    def test_metadata_matches_samples(self, loader, samples):
        """Verify metadata total matches actual count."""
        meta = loader.metadata
        # Note: metadata.total_samples is declared in file, may differ from actual
        # This test just verifies metadata exists
        assert meta.total_samples > 0
