"""Planner-Scheduler-Critic multi-agent scheduler package.

Public API (from NetLogo):
    init_agent()             - call once in setup
    schedule_service(...)    - call every time a service needs placement
    last_decision_summary()  - optional, for monitoring
    last_decision_dict()     - optional, structured decision payload
    hybrid_stats()           - hybrid path / global-risk telemetry
    hybrid_stats_summary()   - human-readable telemetry summary
"""

from .scheduler import (
    hybrid_stats,
    hybrid_stats_summary,
    init_agent,
    last_decision_dict,
    last_decision_summary,
    schedule_service,
)

__all__ = [
    "init_agent",
    "schedule_service",
    "last_decision_summary",
    "last_decision_dict",
    "hybrid_stats",
    "hybrid_stats_summary",
]
