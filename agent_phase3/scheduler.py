from __future__ import annotations

import uuid
from typing import Any

import agent_phase2
from agent_phase1.schemas import SchedulingDecision
from .memory import Episode, EpisodicMemory, WorkingMemory, summarize_context


_WORKING_MEMORY = WorkingMemory(max_items=5)
_EPISODIC_MEMORY = EpisodicMemory()
_LAST_DECISION: dict[str, Any] = {}
_RUN_ID = f"phase3-{uuid.uuid4().hex[:10]}"
_TICK = 0


def init_agent(
    model_name: str = "heuristic",
    trace_dir: str = "traces",
    run_id: str | None = None,
    memory_path: str = "traces/episodes.jsonl",
    enable_tracing: bool = True,
    **kwargs: Any,
) -> None:
    global _WORKING_MEMORY, _EPISODIC_MEMORY, _LAST_DECISION, _RUN_ID, _TICK
    _RUN_ID = run_id or f"phase3-{uuid.uuid4().hex[:10]}"
    _TICK = 0
    _LAST_DECISION = {}
    _WORKING_MEMORY = WorkingMemory(max_items=5)
    _EPISODIC_MEMORY = EpisodicMemory(memory_path)
    agent_phase2.init_agent(
        model_name=model_name,
        trace_dir=trace_dir,
        run_id=_RUN_ID,
        enable_tracing=enable_tracing,
        **kwargs,
    )


def schedule_service(servers_raw: list, service_req_raw: list, global_state_raw: Any | None = None) -> int:
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

    sid = agent_phase2.schedule_service(servers_raw, service_req_raw, global_state_raw, memory_context)
    decision = agent_phase2.last_decision_dict()
    decision["phase"] = "phase3"
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


def last_decision_dict() -> dict[str, Any]:
    return dict(_LAST_DECISION)
