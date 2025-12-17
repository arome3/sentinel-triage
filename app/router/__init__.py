"""
Router module: Semantic routing logic and route definitions.

This module contains:
- routes.py: Semantic route definitions (obvious_harm, obvious_safe, etc.)
- engine.py: Router initialization and decision logic

Public API:
- create_routes(): Factory function to create Route objects
- get_route_names(): Returns list of all route names
- ROUTE_CHARACTERISTICS: Route metadata for tuning
- RouterEngine: Core routing engine class
- RouteChoice: Result dataclass for routing decisions
- get_router_engine(): Get the singleton router engine
- ensure_router_initialized(): Pre-warm router at startup
"""

from app.router.routes import (
    create_routes,
    get_route_names,
    ROUTE_CHARACTERISTICS,
    OBVIOUS_HARM_UTTERANCES,
    OBVIOUS_SAFE_UTTERANCES,
    AMBIGUOUS_RISK_UTTERANCES,
    SYSTEM_ATTACK_UTTERANCES,
    NON_ENGLISH_UTTERANCES,
)

from app.router.engine import (
    RouterEngine,
    RouteChoice,
    get_router_engine,
    ensure_router_initialized,
    reset_router_engine,
)

__all__ = [
    # Factory functions
    "create_routes",
    "get_route_names",
    # Route metadata
    "ROUTE_CHARACTERISTICS",
    # Utterance constants (for testing/extension)
    "OBVIOUS_HARM_UTTERANCES",
    "OBVIOUS_SAFE_UTTERANCES",
    "AMBIGUOUS_RISK_UTTERANCES",
    "SYSTEM_ATTACK_UTTERANCES",
    "NON_ENGLISH_UTTERANCES",
    # Router engine
    "RouterEngine",
    "RouteChoice",
    "get_router_engine",
    "ensure_router_initialized",
    "reset_router_engine",
]
