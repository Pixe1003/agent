"""
Phase 1 scheduler agent package.

Public API (from NetLogo):
    init_agent()           - call once in setup
    schedule_service(...)  - call every time a service needs placement
    last_decision_summary() - optional, for monitoring
"""
from .scheduler import (
    init_agent,
    schedule_service,
    last_decision_summary,
    last_decision_dict,
)

__all__ = [
    "init_agent",
    "schedule_service",
    "last_decision_summary",
    "last_decision_dict",
]
