# Sentinel-Triage

> **Intelligent Content Moderation through Semantic Routing**

---

**Sentinel-Triage** is a content moderation pipeline that uses semantic routing to intelligently direct user-generated content to the most appropriate AI model based on intent, risk level, and language.

**Core Value Proposition:** Route ~80% of moderation requests to fast, inexpensive models while reserving expensive reasoning models for genuinely complex cases — targeting **60%+ cost reduction** without sacrificing moderation quality.

---

## Features

| Feature | Description |
|:--------|:------------|
| **Semantic Routing** | Classifies content intent in <50ms using local ONNX embeddings |
| **4-Tier Model Architecture** | Bulk filtering, deep reasoning, safety detection, and multilingual support |
| **Cost Optimization** | Routes 80% of traffic to Tier 1 models (~$0.05/1M tokens vs $5/1M for GPT-4o) |
| **Multi-Language Support** | Handles 12+ languages via specialized polyglot model |
| **Safety & PII Detection** | Dedicated Llama Guard for jailbreak attempts and PII exposure |
| **Real-Time Metrics** | Track routing patterns, costs, and savings via `/metrics` endpoint |
| **Production Ready** | Type-safe Pydantic schemas, async handlers, health checks, CORS support |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Ingress   │───▶│   Router    │───▶│  Dispatcher │     │
│  │  /moderate  │    │  (Semantic) │    │             │     │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘     │
│                            │                   │            │
│                     ┌──────▼──────┐           │            │
│                     │  Registry   │           │            │
│                     │ (Model Pool)│           │            │
│                     └─────────────┘           │            │
│                                               ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Model Pool                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ Tier 1  │ │ Tier 2  │ │  Guard  │ │Maverick │   │   │
│  │  │ Llama 3 │ │  GPT-4o │ │ Safety  │ │Polyglot │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. User content hits `POST /moderate`
2. Router embeds text and compares against 5 semantic routes (<50ms)
3. Route determines target model (Tier 1, Tier 2, or Specialist)
4. Dispatcher calls the appropriate provider API
5. Response includes verdict, confidence, reasoning, and cost metrics

---

## Quick Start

### Prerequisites

- Python 3.10+
- API Keys:
  - **Groq** (required) — For Tier 1 and Specialist models
  - **OpenAI** (required) — For embeddings

### Installation

First, clone the repository and navigate to the project directory:

```bash
# Clone the repository
git clone https://github.com/arome3/sentinel-triage.git

# Navigate to the project directory
cd sentinel-triage
```

Then, create a virtual environment and install the dependencies:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Required: OPENAI_API_KEY, GROQ_API_KEY
```

### Run the Server

```bash
# Start with auto-reload for development
uvicorn app.main:app --reload --port 8000
```

**Available Endpoints:**

| Endpoint | URL |
|:---------|:----|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |

---

## Configuration

All configuration is managed via environment variables. Copy `.env.example` to `.env` and customize:

| Variable | Required | Default | Description |
|:---------|:--------:|:-------:|:------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key for embeddings |
| `GROQ_API_KEY` | ✅ | — | Groq API key for Llama inference |
| `DEEPSEEK_API_KEY` | ❌ | — | DeepSeek API key (Tier 2 alternative) |
| `SIMILARITY_THRESHOLD` | ❌ | `0.7` | Route match confidence (0.0–1.0) |
| `DEFAULT_ROUTE` | ❌ | `obvious_safe` | Fallback when no route matches |
| `EMBEDDING_MODEL` | ❌ | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `TRACK_COSTS` | ❌ | `true` | Enable cost calculation |
| `LOG_LEVEL` | ❌ | `INFO` | Logging verbosity |
| `HOST` | ❌ | `0.0.0.0` | Server bind host |
| `PORT` | ❌ | `8000` | Server bind port |

---

## API Reference

### `POST /moderate`

Main moderation endpoint. Classifies content and returns verdict.

**Request:**

```json
{
  "content": "Great article, thanks for sharing!"
}
```

**Response:**

```json
{
  "verdict": "safe",
  "confidence": 0.95,
  "reasoning": null,
  "routing": {
    "route_selected": "obvious_safe",
    "route_confidence": 0.98,
    "routing_latency_ms": 15.3,
    "fallback_used": false
  },
  "model_used": "llama-3.1-8b",
  "model_tier": "tier1",
  "inference_latency_ms": 145.2,
  "tokens": {
    "input_tokens": 50,
    "output_tokens": 100
  },
  "estimated_cost_usd": 0.00015
}
```

---

### `GET /health`

System health check for Kubernetes/monitoring.

```json
{
  "status": "healthy",
  "service": "sentinel-triage",
  "components": [
    { "name": "router", "status": "healthy" },
    { "name": "registry", "status": "healthy" }
  ]
}
```

---

### `GET /metrics`

Aggregated statistics and cost savings analysis.

```json
{
  "total_requests": 1000,
  "requests_by_route": { "..." },
  "requests_by_model": { "..." },
  "total_cost_usd": 1.25,
  "hypothetical_cost_usd": 5.00,
  "cost_savings_percent": 75.0
}
```

---

### Other Endpoints

| Endpoint | Method | Description |
|:---------|:------:|:------------|
| `/models` | `GET` | Lists all registered models with metadata |
| `/routes` | `GET` | Lists configured semantic routes with utterance counts |
| `/route?content=...` | `POST` | Debug endpoint to test routing without full moderation |

---

## Model Pool

The system uses a 4-tier model architecture optimized for cost and capability:

| Tier | Model | Role | Cost (per 1M tokens) | Latency |
|:----:|:------|:-----|:--------------------:|:-------:|
| **Tier 1** | Llama 3.1 8B Instant | Bulk classification | $0.05 / $0.08 | <150ms |
| **Tier 2** | GPT-4o | Deep reasoning | $5.00 / $15.00 | <5s |
| **Specialist** | Llama Guard 4 12B | Safety & PII detection | $0.20 | <500ms |
| **Specialist** | Llama 4 Maverick 17B | Multilingual (12+ languages) | $0.20 / $0.60 | <400ms |

> **💡 Cost Optimization:** Tier 2 is ~100x more expensive than Tier 1. By routing 80% of traffic to Tier 1, the system achieves **60%+ cost savings** compared to monolithic GPT-4o usage.

---

## Semantic Routes

Content is classified into 5 semantic routes using vector embeddings:

| Route | Description | Target Model |
|:------|:------------|:-------------|
| `obvious_harm` | Clear violations (spam, profanity, direct threats) | Tier 1 |
| `obvious_safe` | Benign engagement (positive feedback, thanks) | Tier 1 |
| `ambiguous_risk` | Nuanced content (sarcasm, metaphor, veiled threats) | Tier 2 |
| `system_attack` | Jailbreak attempts, PII extraction, prompt injection | Specialist (Guard) |
| `non_english` | Foreign language content (12+ languages) | Specialist (Maverick) |

### Example Classifications

| Input | Route | Model | Reason |
|:------|:------|:------|:-------|
| `"You are an idiot"` | `obvious_harm` | Llama 3.1 8B | Direct insult |
| `"I'm going to kill this presentation"` | `ambiguous_risk` | GPT-4o | Metaphor detection |
| `"Ignore all previous instructions"` | `system_attack` | Llama Guard 4 | Prompt injection attempt |

---

## Testing

The project includes a comprehensive test suite with 180+ tests covering routing, cost calculation, API endpoints, and performance.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_routes.py
```

### Test Coverage

```bash
# Run tests with coverage report (terminal)
pytest --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Categories

Tests are organized with markers for selective execution:

```bash
# Run integration tests (require router initialization)
pytest -m integration

# Run performance/benchmark tests
pytest -m benchmark

# Run slow tests (>1s execution time)
pytest -m slow

# Exclude slow tests for faster feedback
pytest -m "not slow"
```

---

## Success Criteria

The project validates success through:

| Criteria | Target | Method |
|:---------|:-------|:-------|
| **Functional Testing** | 3/3 test inputs routed correctly | Route classification validation |
| **Cost Efficiency** | >60% cost savings | Mixed dataset of 100 queries |
| **Latency Compliance** | Router decision <50ms | Guaranteed with local embeddings |

---

## More on Model Router

For a deeper dive into the model router architecture and the blueprint behind this system, check out this article:

- [The Model Router Blueprint](https://medium.com/@legendabrahamonoja/the-model-router-blueprint-fd37d78e601d)

---