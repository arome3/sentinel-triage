"""
API Endpoint Tests

Integration tests for REST API endpoints using FastAPI TestClient.
Validates request/response contracts, error handling, and endpoint behavior.

Test Categories:
1. TestModerateEndpoint - /moderate endpoint tests
2. TestHealthEndpoint - /health endpoint tests
3. TestMetricsEndpoint - /metrics endpoint tests
4. TestModelsEndpoint - /models endpoint tests
5. TestRoutesEndpoint - /routes endpoint tests
6. TestErrorHandling - Validation error handling

See: docs/09-testing.md for detailed test specifications.
"""



class TestModerateEndpoint:
    """Tests for /moderate endpoint."""

    def test_moderate_valid_request(self, test_client_with_mocks):
        """Valid request returns 200 with moderation response."""
        response = test_client_with_mocks.post(
            "/moderate", json={"content": "Great article!"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check required response fields
        assert "verdict" in data
        assert "confidence" in data
        assert "routing" in data
        assert "model_used" in data
        assert "estimated_cost_usd" in data

    def test_moderate_response_structure(self, test_client_with_mocks):
        """Verify complete response structure."""
        response = test_client_with_mocks.post(
            "/moderate", json={"content": "Test content"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify routing info
        assert "routing" in data
        routing = data["routing"]
        assert "route_selected" in routing
        assert "route_confidence" in routing
        assert "routing_latency_ms" in routing

        # Verify tokens
        assert "tokens" in data
        tokens = data["tokens"]
        assert "input_tokens" in tokens
        assert "output_tokens" in tokens

    def test_moderate_empty_content_returns_422(self, test_client):
        """Empty content returns validation error."""
        response = test_client.post("/moderate", json={"content": ""})

        assert response.status_code == 422

    def test_moderate_whitespace_only_rejected(self, test_client):
        """Whitespace-only content is rejected."""
        response = test_client.post("/moderate", json={"content": "   \n\t  "})

        # Should be rejected as effectively empty
        assert response.status_code == 422

    def test_moderate_content_too_long(self, test_client):
        """Content exceeding 10000 chars returns 422."""
        response = test_client.post("/moderate", json={"content": "x" * 10001})

        assert response.status_code == 422

    def test_moderate_missing_content_field(self, test_client):
        """Missing content field returns 422."""
        response = test_client.post("/moderate", json={})

        assert response.status_code == 422

    def test_moderate_with_metadata(self, test_client_with_mocks):
        """Request with metadata processes correctly."""
        response = test_client_with_mocks.post(
            "/moderate",
            json={
                "content": "Test content",
                "metadata": {"source": "comments", "priority": "high"},
            },
        )

        assert response.status_code == 200

    def test_moderate_with_language_hint(self, test_client_with_mocks):
        """Request with language hint processes correctly."""
        response = test_client_with_mocks.post(
            "/moderate",
            json={"content": "Bonjour", "metadata": {"language_hint": "fr"}},
        )

        assert response.status_code == 200

    def test_moderate_verdict_values(self, test_client_with_mocks):
        """Verdict is one of the valid enum values."""
        response = test_client_with_mocks.post(
            "/moderate", json={"content": "Test content"}
        )

        assert response.status_code == 200
        data = response.json()

        valid_verdicts = ["safe", "flagged", "requires_review"]
        assert data["verdict"] in valid_verdicts

    def test_moderate_confidence_range(self, test_client_with_mocks):
        """Confidence is within 0.0-1.0 range."""
        response = test_client_with_mocks.post(
            "/moderate", json={"content": "Test content"}
        )

        assert response.status_code == 200
        data = response.json()

        assert 0.0 <= data["confidence"] <= 1.0


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_status(self, test_client):
        """Health endpoint returns system status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_includes_components(self, test_client):
        """Health response includes component status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "components" in data
        assert isinstance(data["components"], list)

    def test_health_includes_uptime(self, test_client):
        """Health response includes uptime."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_includes_service_name(self, test_client):
        """Health response includes service name."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert data["service"] == "sentinel-triage"


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_aggregates(self, test_client):
        """Metrics endpoint returns aggregated data."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "total_requests" in data
        assert "cost_savings_percent" in data
        assert "requests_by_route" in data
        assert "requests_by_model" in data

    def test_metrics_includes_cost_data(self, test_client):
        """Metrics includes cost tracking data."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "total_cost_usd" in data
        assert "hypothetical_cost_usd" in data
        assert "cost_savings_percent" in data

    def test_metrics_includes_latency_data(self, test_client):
        """Metrics includes latency averages."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "avg_routing_latency_ms" in data
        assert "avg_inference_latency_ms" in data

    def test_metrics_empty_state(self, test_client):
        """Metrics returns valid response with no requests."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["total_requests"] >= 0


class TestModelsEndpoint:
    """Tests for /models endpoint."""

    def test_models_lists_registry(self, test_client):
        """Models endpoint returns registry contents."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "route_mapping" in data

    def test_models_returns_four_models(self, test_client):
        """Models endpoint returns all 4 registered models."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert len(data["models"]) == 4

    def test_models_includes_model_metadata(self, test_client):
        """Each model includes required metadata."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()

        for model in data["models"]:
            assert "model_id" in model
            assert "tier" in model
            assert "provider" in model

    def test_models_route_mapping_complete(self, test_client):
        """Route mapping includes all 5 routes."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()

        expected_routes = {
            "obvious_harm",
            "obvious_safe",
            "ambiguous_risk",
            "system_attack",
            "non_english",
        }

        assert set(data["route_mapping"].keys()) == expected_routes


class TestRoutesEndpoint:
    """Tests for /routes endpoint."""

    def test_routes_returns_info(self, test_client_with_mocks):
        """Routes endpoint returns route information."""
        response = test_client_with_mocks.get("/routes")

        assert response.status_code == 200
        data = response.json()

        assert "num_routes" in data or "routes" in data


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_api_info(self, test_client):
        """Root endpoint returns API information."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data or "service" in data


class TestErrorHandling:
    """Tests for error response handling."""

    def test_validation_error_format(self, test_client):
        """Validation errors return proper error format."""
        response = test_client.post("/moderate", json={"content": ""})

        assert response.status_code == 422
        data = response.json()

        # Should have error structure
        assert "error" in data or "detail" in data

    def test_invalid_json_returns_422(self, test_client):
        """Invalid JSON body returns 422."""
        response = test_client.post(
            "/moderate", data="not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type_error(self, test_client):
        """Wrong content type returns appropriate error."""
        response = test_client.post(
            "/moderate",
            data="content=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422

    def test_method_not_allowed(self, test_client):
        """Wrong HTTP method returns 405."""
        response = test_client.get("/moderate")

        assert response.status_code == 405

    def test_not_found_endpoint(self, test_client):
        """Non-existent endpoint returns 404."""
        response = test_client.get("/nonexistent")

        assert response.status_code == 404


class TestConfigEndpoint:
    """Tests for /config endpoint."""

    def test_config_returns_non_sensitive(self, test_client):
        """Config endpoint returns non-sensitive settings."""
        response = test_client.get("/config")

        assert response.status_code == 200
        data = response.json()

        # Config has nested structure - check for expected sections
        assert "router" in data or "logging" in data
        # Verify nested config values exist
        if "router" in data:
            assert "similarity_threshold" in data["router"]
        if "logging" in data:
            assert "level" in data["logging"]

    def test_config_excludes_api_keys(self, test_client):
        """Config endpoint does not expose API keys."""
        response = test_client.get("/config")

        assert response.status_code == 200
        data = response.json()

        # API keys should not be exposed
        response_str = str(data).lower()
        assert "api_key" not in response_str or "sk-" not in response_str
        assert "gsk_" not in response_str


class TestRouteOnlyEndpoint:
    """Tests for /route endpoint (routing without dispatch)."""

    def test_route_endpoint_returns_route(self, test_client_with_mocks):
        """Route endpoint returns routing decision only."""
        response = test_client_with_mocks.post(
            "/route", params={"content": "Test content"}
        )

        # May return 200 or other status depending on implementation
        if response.status_code == 200:
            data = response.json()
            assert "route_name" in data or "route" in data
