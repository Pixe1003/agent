"""SFT/LoRA fine-tuned scheduler agent.

Same NetLogo-facing API as multi_agent / agent_memory:
    init_agent(model_path=..., n_threads=..., enable_tracing=...)
    schedule_service(servers, service, global_state=None, memory=None, aiops_insight=None) -> int
    last_decision_summary() / last_decision_dict() / sft_stats()

Loading the GGUF model is lazy and only happens on init_agent. If llama-cpp-python
is not installed or the model file is missing, init_agent raises a clear error.
"""

from .scheduler import (
    init_agent,
    schedule_service,
    last_decision_dict,
    last_decision_summary,
    sft_stats,
    sft_stats_summary,
)

__all__ = [
    "init_agent",
    "schedule_service",
    "last_decision_summary",
    "last_decision_dict",
    "sft_stats",
    "sft_stats_summary",
]
