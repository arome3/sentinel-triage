"""Tests for the content classification utility."""

from app.content_classifier import Classification, classify_content


class TestClassifyContent:
    """Unit tests for classify_content()."""

    def test_standard_content(self):
        """Normal text is classified as standard."""
        result = classify_content("The weather is nice today.")
        assert result.label == "standard"
        assert result.confidence == 1.0
        assert result.matched_category == "none"

    def test_urgent_content(self):
        """Urgent keywords trigger urgent classification."""
        result = classify_content("This is an emergency, please help immediately!")
        assert result.label == "urgent"
        assert result.confidence >= 0.5

    def test_harmful_content(self):
        """Harmful keywords trigger harmful classification."""
        result = classify_content("There was an attack with threats of violence.")
        assert result.label == "harmful"
        assert result.confidence >= 0.5

    def test_spam_content(self):
        """Spam phrases trigger spam classification."""
        result = classify_content("Buy now! Click here for free money!")
        assert result.label == "spam"
        assert result.confidence >= 0.5

    def test_repeated_chars_spam(self):
        """Excessive character repetition is flagged as spam."""
        result = classify_content("aaaaaaaaaaaaaaaaaaaa check this out")
        assert result.label == "spam"

    def test_priority_urgent_over_harmful(self):
        """Urgent takes priority over harmful when both match."""
        result = classify_content("Emergency! There is a threat, act immediately!")
        assert result.label == "urgent"

    def test_priority_harmful_over_spam(self):
        """Harmful takes priority over spam when both match."""
        result = classify_content("This is a malicious attack, buy now!")
        assert result.label == "harmful"

    def test_case_insensitive(self):
        """Pattern matching is case-insensitive."""
        result = classify_content("URGENT EMERGENCY")
        assert result.label == "urgent"

    def test_sanitization_applied(self):
        """Control characters are stripped before classification."""
        result = classify_content("normal \x00 content \x01 here")
        assert result.label == "standard"

    def test_word_count_populated(self):
        """Word count is included in the classification."""
        result = classify_content("one two three four five")
        assert result.word_count == 5

    def test_confidence_bounded(self):
        """Confidence is always between 0.0 and 1.0."""
        for text in [
            "urgent emergency critical immediately asap",
            "hello world",
            "buy now click here free money act now",
        ]:
            result = classify_content(text)
            assert 0.0 <= result.confidence <= 1.0

    def test_classification_immutable(self):
        """Classification is a frozen dataclass."""
        result = classify_content("test")
        try:
            result.label = "changed"
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_empty_input(self):
        """Empty string is classified as standard."""
        result = classify_content("")
        assert result.label == "standard"

    def test_return_type(self):
        """classify_content returns a Classification instance."""
        result = classify_content("hello")
        assert isinstance(result, Classification)

    def test_known_labels(self):
        """All returned labels are in the known set."""
        texts = [
            "hello",
            "urgent help",
            "threat of violence",
            "buy now free money",
        ]
        for text in texts:
            result = classify_content(text)
            assert result.label in Classification.LABELS

    def test_multiple_urgent_keywords_high_confidence(self):
        """Multiple urgent keywords yield full confidence."""
        result = classify_content(
            "This is urgent, an emergency requiring critical and immediate action"
        )
        assert result.label == "urgent"
        assert result.confidence == 1.0
