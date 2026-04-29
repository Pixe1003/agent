"""
Phase 1 scheduler agent package.

Public API (from NetLogo):
    init_agent()            - call once in setup
    schedule_service(...)   - call every time a service needs placement
    last_decision_summary() - optional, for monitoring

The public functions are imported lazily so data-only modules such as
``agent_phase1.schemas`` can be reused without requiring LangChain/Ollama.
"""

__all__ = [
    "init_agent",
    "schedule_service",
    "last_decision_summary",
    "last_decision_dict",
]


def __getattr__(name):
    if name in __all__:
        from . import scheduler

        return getattr(scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
