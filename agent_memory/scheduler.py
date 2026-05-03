from __future__ import annotations

import uuid
from typing import Any

import multi_agent
from agent_common.schemas import SchedulingDecision
from .memory import Episode, EpisodicMemory, WorkingMemory, summarize_context


_WORKING_MEMORY = WorkingMemory(max_items=5)
_EPISODIC_MEMORY = EpisodicMemory()
_LAST_DECISION: dict[str, Any] = {}
_RUN_ID = f"agent_memory-{uuid.uuid4().hex[:10]}"
_TICK = 0


def init_agent(
    model_name: str = "heuristic",
    trace_dir: str = "traces",
    run_id: str | None = None,
    memory_path: str = "traces/episodes.jsonl",
    enable_tracing: bool = True,
    persist_episodes: bool = True,
    flush_every: int = 16,
    **kwargs: Any,
) -> None:
    """Initialize agent_memory scheduler.

    persist_episodes=False skips JSONL writes entirely (recommended for benchmarks
    or short demos where cross-run persistence is not needed). Episodes still live
    in process memory and are retrievable within the same session.
    flush_every controls batched flushing when persist_episodes=True.
    """
    global _WORKING_MEMORY, _EPISODIC_MEMORY, _LAST_DECISION, _RUN_ID, _TICK
    _RUN_ID = run_id or f"agent_memory-{uuid.uuid4().hex[:10]}"
    _TICK = 0
    _LAST_DECISION = {}
    _WORKING_MEMORY = WorkingMemory(max_items=5)
    # 关掉旧 memory 的文件句柄，防止泄漏
    try:
        _EPISODIC_MEMORY.close()
    except Exception:
        pass
    _EPISODIC_MEMORY = EpisodicMemory(memory_path, persist=persist_episodes, flush_every=flush_every)
    multi_agent.init_agent(
        model_name=model_name,
        trace_dir=trace_dir,
        run_id=_RUN_ID,
        enable_tracing=enable_tracing,
        **kwargs,
    )


def schedule_service(
    servers_raw: list,
    service_req_raw: list,
    global_state_raw: Any | None = None,
    aiops_insight_raw: Any | None = None,
) -> int:
    global _LAST_DECISION, _TICK
    _TICK += 1
    summary, features = summarize_context(servers_raw, service_req_raw)
    retrieved = _EPISODIC_MEMORY.retrieve(summary, features, top_k=3)
    memory_context = {
        "working": _WORKING_MEMORY.render(),
        "episodic": [
            {
                "episode_id": episode.episode_id,
                "action_server_id": episode.action_server_id,
                "reasoning_trace": episode.reasoning_trace,
                "reward": episode.reward,
            }
            for episode in retrieved
        ],
    }

    sid = multi_agent.schedule_service(
        servers_raw,
        service_req_raw,
        global_state_raw,
        memory_context,
        aiops_insight_raw,
    )
    decision = multi_agent.last_decision_dict()
    decision["phase"] = "agent_memory"
    decision["memory_context"] = memory_context
    _LAST_DECISION = decision

    if decision.get("action") == "select" and sid >= 0:
        scheduling_decision = SchedulingDecision(
            action="select",
            server_id=sid,
            reasoning=decision.get("reasoning", ""),
            latency_ms=float(decision.get("latency_ms", 0.0)),
            tool_call_succeeded=bool(decision.get("tool_call_succeeded", True)),
        )
        _WORKING_MEMORY.add(scheduling_decision)
        _EPISODIC_MEMORY.add(
            Episode(
                run_id=_RUN_ID,
                tick=_TICK,
                state_summary_text=summary,
                state_features=features,
                service_request={
                    "cpu_pct": service_req_raw[0],
                    "ram_pct": service_req_raw[1],
                    "net_pct": service_req_raw[2],
                },
                action_server_id=sid,
                reasoning_trace=decision.get("reasoning", ""),
                reward=1.0,
            )
        )
    return sid


def last_decision_summary() -> str:
    if not _LAST_DECISION:
        return "no decision yet"
    return (
        f"[{_LAST_DECISION['action']}] server={_LAST_DECISION.get('server_id')} "
        f"memory={len(_LAST_DECISION.get('memory_context', {}).get('episodic', []))} "
        f"{_LAST_DECISION.get('latency_ms', 0):.0f}ms | {_LAST_DECISION.get('reasoning', '')[:80]}"
    )


def agent_usage_stats() -> dict[str, Any]:
    return multi_agent.hybrid_stats()


def agent_usage_summary() -> str:
    stats = agent_usage_stats()
    total = stats.get("total_decisions", 0)
    if total == 0:
        return "agent_memory total=0"
    return (
        f"agent_memory total={total} "
        f"fast={stats.get('fast_path_decisions', 0)} "
        f"escalate={stats.get('agent_escalation_needed', 0)} "
        f"({stats.get('escalation_ratio', 0.0) * 100:.1f}%) "
        f"agent_sync={stats.get('agent_sync_calls', 0)} "
        f"({stats.get('hybrid_agent_call_ratio', 0.0) * 100:.1f}%) "
        f"memory={stats.get('memory_used_decisions', 0)} "
        f"({stats.get('memory_usage_ratio', 0.0) * 100:.1f}%) "
        f"avg_retrieved={stats.get('avg_retrieved_episode_count', 0.0):.2f} "
        f"avg_latency={stats.get('avg_latency_ms', 0.0):.3f}ms "
        f"fallback={stats.get('fallback_decisions', 0)}"
    )


def last_decision_dict() -> dict[str, Any]:
    return dict(_LAST_DECISION)
