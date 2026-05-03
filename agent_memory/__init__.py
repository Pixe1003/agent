"""Memory-augmented scheduler package.

Wraps multi_agent with WorkingMemory + EpisodicMemory and exposes the
NetLogo-facing API.
"""

from .scheduler import (
    agent_usage_stats,
    agent_usage_summary,
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
    "agent_usage_stats",
    "agent_usage_summary",
]
