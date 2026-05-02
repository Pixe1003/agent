"""AIOps monitoring and policy recommendation agent.

The module exposes a Python-first control-plane API. It observes scheduler
signals and produces advisory insights; it never selects a server or mutates
runtime scheduling policy.
"""

from .monitor import (
    aiops_stats,
    aiops_stats_summary,
    analyze_ops_state,
    current_alerts,
    init_agent,
    last_insight_dict,
    last_insight_summary,
    observe_ops_state,
)

__all__ = [
    "init_agent",
    "analyze_ops_state",
    "observe_ops_state",
    "current_alerts",
    "aiops_stats",
    "aiops_stats_summary",
    "last_insight_dict",
    "last_insight_summary",
]
