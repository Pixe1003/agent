from __future__ import annotations

import statistics
import time
from typing import Any, Literal, Protocol, TypedDict

from agent_common.tracing import TraceLogger
from agent_phase1.prompts import render_cluster_state
from agent_phase1.schemas import (
    SchedulingContext,
    SchedulingDecision,
    ServerSnapshot,
    ServiceRequest,
)
from pydantic import BaseModel, Field, ValidationError


_MODEL_NAME = "heuristic"
_BACKEND_MODE: Literal["heuristic", "structured", "hybrid"] = "heuristic"
_HYBRID_AGENT_MODE: Literal["record", "sync"] = "record"
_COMPLEXITY_THRESHOLD = 0.5
_TRACE_LOGGER: TraceLogger | None = None
_STRUCTURED_BACKEND: StructuredBackend | None = None
_GRAPH: Any = None
_LAST_DECISION: dict[str, Any] = {}
_HYBRID_STATS: dict[str, Any] = {}
_MAX_REVISIONS = 2


class PlannerOutput(BaseModel):
    strategy_tag: str = Field(..., min_length=3, max_length=80)
    strategy_reasoning: str = Field(..., min_length=10, max_length=500)


class SchedulerProposal(BaseModel):
    action: Literal["select", "reject"]
    server_id: int | None = Field(default=None, ge=0)
    reasoning: str = Field(..., min_length=10, max_length=500)


class RiskSnapshot(BaseModel):
    active_cpu_util: float = Field(default=0.0, ge=0.0)
    active_mem_util: float = Field(default=0.0, ge=0.0)
    active_net_util: float = Field(default=0.0, ge=0.0)
    active_servers: int = Field(default=0, ge=0)
    current_auto_migrations: int = Field(default=0, ge=0)
    current_consolidation_migrations: int = Field(default=0, ge=0)
    rescheduled_services: int = Field(default=0, ge=0)
    ops_sla_violations: float = Field(default=0.0, ge=0.0)
    mem_sla_violations: float = Field(default=0.0, ge=0.0)
    net_sla_violations: float = Field(default=0.0, ge=0.0)


class StructuredBackend(Protocol):
    def plan(
        self,
        ctx: SchedulingContext,
        risk_analysis: dict[str, Any] | None = None,
    ) -> PlannerOutput | dict[str, Any]:
        ...

    def propose(
        self,
        ctx: SchedulingContext,
        *,
        strategy_tag: str,
        strategy_reasoning: str,
        excluded_server_ids: set[int],
        critic_feedback: str | None,
        risk_analysis: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> SchedulerProposal | dict[str, Any]:
        ...


class AgentState(TypedDict, total=False):
    ctx: SchedulingContext
    excluded_server_ids: list[int]
    strategy_tag: str | None
    strategy_reasoning: str | None
    planner_succeeded: bool
    proposed_action: str | None
    proposed_server_id: int | None
    proposed_reasoning: str | None
    structured_output_succeeded: bool
    critic_verdict: str | None
    critic_reasoning: str | None
    critic_feedback: str | None
    revise_count: int
    scheduler_error: str | None
    fallback_reason: str | None
    risk_analysis: dict[str, Any] | None
    memory_context: dict[str, Any] | None


class OllamaStructuredBackend:
    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        num_predict: int,
        base_url: str | None = None,
    ) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "LangChain/Ollama dependencies are missing. "
                "Run: pip install -r agent_phase1/requirements.txt"
            ) from e

        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "num_predict": num_predict,
            "reasoning": False,
        }
        if base_url:
            kwargs["base_url"] = base_url
        llm = ChatOllama(**kwargs)
        self._planner = llm.with_structured_output(PlannerOutput)
        self._scheduler = llm.with_structured_output(SchedulerProposal)

    def plan(
        self,
        ctx: SchedulingContext,
        risk_analysis: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        return _coerce_planner_output(self._planner.invoke(_planner_prompt(ctx, risk_analysis)))

    def propose(
        self,
        ctx: SchedulingContext,
        *,
        strategy_tag: str,
        strategy_reasoning: str,
        excluded_server_ids: set[int],
        critic_feedback: str | None,
        risk_analysis: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> SchedulerProposal:
        prompt = _scheduler_prompt(
            ctx,
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            excluded_server_ids=excluded_server_ids,
            critic_feedback=critic_feedback,
            risk_analysis=risk_analysis,
            memory_context=memory_context,
        )
        return _coerce_scheduler_proposal(self._scheduler.invoke(prompt))


class _InlineGraph:
    """Small fallback for environments that have tests but lack langgraph."""

    def invoke(self, state: AgentState) -> AgentState:
        state.update(_planner_node(state))
        while True:
            state.update(_scheduler_node(state))
            state.update(_critic_node(state))
            if _route_after_critic(state) != "scheduler":
                return state


def init_agent(
    model_name: str = "heuristic",
    temperature: float = 0.0,
    num_predict: int = 512,
    trace_dir: str = "traces",
    run_id: str | None = None,
    enable_tracing: bool = True,
    backend: Literal["auto", "heuristic", "structured", "hybrid"] = "auto",
    hybrid_agent_mode: Literal["record", "sync"] = "record",
    complexity_threshold: float = 0.5,
    structured_backend: StructuredBackend | None = None,
    base_url: str | None = None,
    **_: Any,
) -> None:
    """Initialize Phase 2.

    The NetLogo-facing API stays stable. backend="auto" uses the hybrid path:
    fast deterministic scheduling for normal cases, with complexity metadata
    and optional structured-agent escalation for hard cases.
    """
    global _MODEL_NAME, _BACKEND_MODE, _HYBRID_AGENT_MODE, _COMPLEXITY_THRESHOLD
    global _TRACE_LOGGER, _STRUCTURED_BACKEND, _GRAPH, _HYBRID_STATS
    _MODEL_NAME = model_name
    _BACKEND_MODE = _resolve_backend(model_name, backend)
    _HYBRID_AGENT_MODE = hybrid_agent_mode
    _COMPLEXITY_THRESHOLD = complexity_threshold
    _HYBRID_STATS = _empty_hybrid_stats()
    if _BACKEND_MODE == "structured" or (_BACKEND_MODE == "hybrid" and _HYBRID_AGENT_MODE == "sync"):
        _STRUCTURED_BACKEND = structured_backend or OllamaStructuredBackend(
            model_name=model_name,
            temperature=temperature,
            num_predict=num_predict,
            base_url=base_url,
        )
        _GRAPH = _build_graph()
    else:
        _STRUCTURED_BACKEND = None
        _GRAPH = None
    _TRACE_LOGGER = TraceLogger(
        trace_dir=trace_dir,
        run_id=run_id,
        phase="phase2",
        model=model_name,
        enabled=enable_tracing,
    )


def schedule_service(
    servers_raw: list,
    service_req_raw: list,
    global_state_raw: Any | None = None,
    memory_context_raw: Any | None = None,
) -> int:
    global _LAST_DECISION
    t0 = time.perf_counter()

    try:
        ctx = _parse_context(servers_raw, service_req_raw)
        risk = _parse_risk_snapshot(global_state_raw)
        memory_context = _parse_memory_context(memory_context_raw)
    except (ValidationError, ValueError, IndexError, TypeError) as e:
        decision = SchedulingDecision(
            action="fallback",
            reasoning=f"Input validation error: {e}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        _LAST_DECISION = _record_decision(
            decision,
            t0=t0,
            messages=[],
            tool_calls=[],
            strategy_tag=None,
            critic_verdict="fallback",
            structured_output_succeeded=False,
            fallback_reason=decision.reasoning,
        )
        _record_hybrid_stats(_LAST_DECISION)
        return -1

    if _BACKEND_MODE == "structured":
        final_state = _run_structured_graph(ctx, _analyze_global_risk(risk), memory_context)
        _LAST_DECISION, result = _decision_from_structured_state(final_state, t0)
        _record_hybrid_stats(_LAST_DECISION)
        return result

    if _BACKEND_MODE == "hybrid":
        _LAST_DECISION, result = _run_hybrid(ctx, risk, memory_context, t0)
        _record_hybrid_stats(_LAST_DECISION)
        return result

    _LAST_DECISION, result = _run_heuristic(ctx, t0)
    _record_hybrid_stats(_LAST_DECISION)
    return result


def _run_heuristic(ctx: SchedulingContext, t0: float, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    strategy_tag, strategy_reasoning = _plan(ctx)
    excluded: set[int] = set()
    revise_count = 0
    risk_policy = (extra or {}).get("risk_policy")

    while revise_count <= 2:
        proposal = _propose(ctx, excluded, risk_policy=risk_policy)
        if proposal is None:
            decision = SchedulingDecision(
                action="reject",
                reasoning="No server can host the service without exceeding resource headroom.",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            data = _record_decision(
                decision,
                t0=t0,
                messages=_messages(ctx, strategy_tag, strategy_reasoning),
                tool_calls=[{"name": "reject_service", "args": {"reason": decision.reasoning}}],
                strategy_tag=strategy_tag,
                strategy_reasoning=strategy_reasoning,
                critic_verdict="approve",
                structured_output_succeeded=False,
                revise_count=revise_count,
                extra=extra,
            )
            return data, -2

        verdict, critic_reasoning = _critic(ctx, proposal)
        if verdict == "approve":
            decision = SchedulingDecision(
                action="select",
                server_id=proposal.server_id,
                reasoning=f"{strategy_tag}: approved server {proposal.server_id}; {critic_reasoning}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            data = _record_decision(
                decision,
                t0=t0,
                messages=_messages(ctx, strategy_tag, strategy_reasoning),
                tool_calls=[
                    {
                        "name": "select_server",
                        "args": {"server_id": proposal.server_id, "reasoning": decision.reasoning},
                    }
                ],
                strategy_tag=strategy_tag,
                strategy_reasoning=strategy_reasoning,
                critic_verdict=verdict,
                critic_reasoning=critic_reasoning,
                structured_output_succeeded=False,
                revise_count=revise_count,
                extra=extra,
            )
            return data, proposal.server_id

        excluded.add(proposal.server_id)
        revise_count += 1

    decision = SchedulingDecision(
        action="fallback",
        reasoning="Critic requested too many revisions; falling back to Balanced-Fit.",
        latency_ms=(time.perf_counter() - t0) * 1000,
        tool_call_succeeded=False,
    )
    data = _record_decision(
        decision,
        t0=t0,
        messages=_messages(ctx, strategy_tag, strategy_reasoning),
        tool_calls=[],
        strategy_tag=strategy_tag,
        strategy_reasoning=strategy_reasoning,
        critic_verdict="fallback",
        structured_output_succeeded=False,
        revise_count=revise_count,
        fallback_reason=decision.reasoning,
        extra=extra,
    )
    return data, -1


def last_decision_summary() -> str:
    if not _LAST_DECISION:
        return "no decision yet"
    return (
        f"[{_LAST_DECISION['action']}] server={_LAST_DECISION.get('server_id')} "
        f"critic={_LAST_DECISION.get('critic_verdict')} "
        f"{_LAST_DECISION.get('latency_ms', 0):.0f}ms | "
        f"{_LAST_DECISION.get('reasoning', '')[:90]}"
    )


def last_decision_dict() -> dict[str, Any]:
    return dict(_LAST_DECISION)


def hybrid_stats() -> dict[str, Any]:
    stats = _stats_with_ratios(_HYBRID_STATS)
    stats["complexity_reason_counts"] = dict(_HYBRID_STATS.get("complexity_reason_counts", {}))
    stats["risk_tag_counts"] = dict(_HYBRID_STATS.get("risk_tag_counts", {}))
    return stats


def hybrid_stats_summary() -> str:
    stats = hybrid_stats()
    if stats["total_decisions"] == 0:
        return "hybrid total=0"
    reasons = stats["complexity_reason_counts"]
    top_reasons = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
    reasons_text = ", ".join(f"{name}:{count}" for name, count in top_reasons) or "none"
    return (
        f"hybrid total={stats['total_decisions']} fast={stats['fast_path_decisions']} "
        f"escalate={stats['agent_escalation_needed']} ({stats['escalation_ratio'] * 100:.1f}%) "
        f"risk={stats['global_risk_agent_triggers']} ({stats['global_risk_trigger_ratio'] * 100:.1f}%) "
        f"agent_sync={stats['agent_sync_calls']} ({stats['hybrid_agent_call_ratio'] * 100:.1f}%) "
        f"memory={stats['memory_used_decisions']} avg_latency={stats['avg_latency_ms']:.3f}ms "
        f"fallback={stats['fallback_decisions']} reasons={reasons_text}"
    )


def _resolve_backend(model_name: str, backend: str) -> Literal["heuristic", "structured", "hybrid"]:
    if backend == "auto":
        return "hybrid"
    if backend in {"heuristic", "structured", "hybrid"}:
        return backend  # type: ignore[return-value]
    raise ValueError("backend must be one of: auto, heuristic, structured, hybrid")


def _run_hybrid(
    ctx: SchedulingContext,
    risk: RiskSnapshot,
    memory_context: dict[str, Any],
    t0: float,
) -> tuple[dict[str, Any], int]:
    risk_analysis = _analyze_global_risk(risk)
    analysis = _analyze_complexity(ctx, risk_analysis, memory_context)
    metadata = _hybrid_metadata(analysis, fast_path_used=True)
    if analysis["agent_escalation_needed"] and _HYBRID_AGENT_MODE == "sync":
        final_state = _run_structured_graph(ctx, risk_analysis, memory_context)
        return _decision_from_structured_state(
            final_state,
            t0,
            extra=_hybrid_metadata(analysis, fast_path_used=False),
        )
    return _run_heuristic(ctx, t0, extra=metadata)


def _hybrid_metadata(analysis: dict[str, Any], *, fast_path_used: bool) -> dict[str, Any]:
    return {
        "fast_path_used": fast_path_used,
        "hybrid_agent_mode": _HYBRID_AGENT_MODE,
        "complexity_score": analysis["complexity_score"],
        "complexity_reasons": analysis["complexity_reasons"],
        "agent_escalation_needed": analysis["agent_escalation_needed"],
        "valid_candidate_count": analysis["valid_candidate_count"],
        "local_complexity_score": analysis["local_complexity_score"],
        "global_risk_score": analysis["global_risk_score"],
        "global_risk_level": analysis["global_risk_level"],
        "global_risk_tags": analysis["global_risk_tags"],
        "global_risk_agent_triggered": analysis["global_risk_agent_triggered"],
        "risk_policy": analysis["risk_policy"],
        "risk_aware_fast_path": bool(analysis["global_risk_tags"]),
        "retrieved_episode_count": analysis["retrieved_episode_count"],
        "memory_used": analysis["memory_used"],
        "memory_confidence": analysis["memory_confidence"],
    }


def _analyze_complexity(
    ctx: SchedulingContext,
    risk_analysis: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = _valid_candidates(ctx)
    reasons: list[str] = []
    score = 0.0
    risk_analysis = risk_analysis or _analyze_global_risk(RiskSnapshot())
    memory_context = memory_context or _parse_memory_context(None)

    if not candidates:
        return {
            "complexity_score": 0.0,
            "local_complexity_score": 0.0,
            "complexity_reasons": ["no-valid-candidates"],
            "agent_escalation_needed": False,
            "valid_candidate_count": 0,
            "global_risk_score": risk_analysis["global_risk_score"],
            "global_risk_level": risk_analysis["global_risk_level"],
            "global_risk_tags": risk_analysis["global_risk_tags"],
            "global_risk_agent_triggered": False,
            "risk_policy": risk_analysis["risk_policy"],
            "retrieved_episode_count": memory_context["retrieved_episode_count"],
            "memory_used": memory_context["memory_used"],
            "memory_confidence": memory_context["memory_confidence"],
        }

    if len(candidates) <= 2:
        score += 0.35
        reasons.append("few-valid-candidates")

    max_demand = max(ctx.service.cpu_pct, ctx.service.ram_pct, ctx.service.net_pct)
    if max_demand >= 35:
        score += 0.25
        reasons.append("large-service-request")

    best = min(candidates, key=lambda server: _candidate_score(ctx, server))
    residual = [
        best.cpu_free_pct - ctx.service.cpu_pct,
        best.ram_free_pct - ctx.service.ram_pct,
        best.net_free_pct - ctx.service.net_pct,
    ]
    min_residual = min(residual)
    if min_residual < 10:
        score += 0.25
        reasons.append("low-post-placement-headroom")
    elif min_residual < 20:
        score += 0.15
        reasons.append("moderate-post-placement-headroom")

    if _cluster_fragmentation(ctx) > 25:
        score += 0.15
        reasons.append("fragmented-cluster-state")

    if memory_context["memory_used"]:
        reasons.append("retrieved-memory-context")
        if memory_context["memory_confidence"] >= 0.75:
            score = max(0.0, score - 0.1)

    local_score = min(score, 1.0)
    global_score = risk_analysis["global_risk_score"]
    combined_score = min(1.0, local_score + global_score)
    global_triggered = (
        global_score > 0
        and combined_score >= _COMPLEXITY_THRESHOLD
        and (local_score < _COMPLEXITY_THRESHOLD or global_score >= _COMPLEXITY_THRESHOLD)
    )
    global_reasons = [f"global-{tag}" for tag in risk_analysis["global_risk_tags"]]
    return {
        "complexity_score": round(combined_score, 3),
        "local_complexity_score": round(local_score, 3),
        "complexity_reasons": reasons + global_reasons,
        "agent_escalation_needed": combined_score >= _COMPLEXITY_THRESHOLD,
        "valid_candidate_count": len(candidates),
        "global_risk_score": global_score,
        "global_risk_level": risk_analysis["global_risk_level"],
        "global_risk_tags": risk_analysis["global_risk_tags"],
        "global_risk_agent_triggered": global_triggered,
        "risk_policy": risk_analysis["risk_policy"],
        "retrieved_episode_count": memory_context["retrieved_episode_count"],
        "memory_used": memory_context["memory_used"],
        "memory_confidence": memory_context["memory_confidence"],
    }


def _analyze_global_risk(risk: RiskSnapshot) -> dict[str, Any]:
    score = 0.0
    tags: list[str] = []

    cpu_util = _normalize_utilization(risk.active_cpu_util)
    mem_util = _normalize_utilization(risk.active_mem_util)
    net_util = _normalize_utilization(risk.active_net_util)

    if net_util >= 0.9:
        score += 0.35
        tags.append("network-pressure")
    elif net_util >= 0.75:
        score += 0.2
        tags.append("network-watch")

    if cpu_util >= 0.9:
        score += 0.2
        tags.append("cpu-pressure")
    if mem_util >= 0.9:
        score += 0.2
        tags.append("memory-pressure")

    if risk.current_auto_migrations >= 5:
        score += 0.25
        tags.append("migration-pressure")
    elif risk.current_auto_migrations > 0:
        score += 0.1
        tags.append("migration-watch")

    if risk.current_consolidation_migrations >= 5:
        score += 0.15
        tags.append("consolidation-pressure")

    if risk.net_sla_violations > 0:
        score += 0.35
        if "network-pressure" not in tags:
            tags.append("network-pressure")
        tags.append("sla-risk")
    elif risk.ops_sla_violations > 0 or risk.mem_sla_violations > 0:
        score += 0.2
        tags.append("sla-risk")

    if risk.rescheduled_services >= 5:
        score += 0.15
        tags.append("reschedule-pressure")
    elif risk.rescheduled_services > 0:
        score += 0.1
        tags.append("reschedule-watch")

    score = min(score, 1.0)
    tags = _unique(tags)
    return {
        "global_risk_score": round(score, 3),
        "global_risk_level": _risk_level(score),
        "global_risk_tags": tags,
        "risk_policy": _risk_policy(tags),
    }


def _risk_policy(tags: list[str]) -> dict[str, Any]:
    weights = {"cpu": 1.0, "mem": 1.0, "net": 1.0}
    placement_style = "balanced-fit"

    if "network-pressure" in tags or "network-watch" in tags:
        weights["net"] = 3.0 if "network-pressure" in tags else 2.0
        placement_style = "protect-network-headroom"
    if "cpu-pressure" in tags:
        weights["cpu"] = max(weights["cpu"], 2.0)
        placement_style = "protect-hot-resource-headroom"
    if "memory-pressure" in tags:
        weights["mem"] = max(weights["mem"], 2.0)
        placement_style = "protect-hot-resource-headroom"
    if "migration-pressure" in tags:
        placement_style = "stabilize-placement"

    return {
        "placement_style": placement_style,
        "resource_weights": weights,
        "avoid_tight_headroom": bool(tags),
    }


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _normalize_utilization(value: float) -> float:
    return value / 100 if value > 1 else value


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _build_graph() -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _InlineGraph()

    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner_node)
    graph.add_node("scheduler", _scheduler_node)
    graph.add_node("critic", _critic_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "scheduler")
    graph.add_edge("scheduler", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {"scheduler": "scheduler", "end": END})
    return graph.compile()


def _run_structured_graph(
    ctx: SchedulingContext,
    risk_analysis: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> AgentState:
    graph = _GRAPH or _build_graph()
    return graph.invoke(
        {
            "ctx": ctx,
            "risk_analysis": risk_analysis or _analyze_global_risk(RiskSnapshot()),
            "memory_context": memory_context or _parse_memory_context(None),
            "excluded_server_ids": [],
            "revise_count": 0,
            "structured_output_succeeded": False,
            "critic_verdict": None,
            "critic_feedback": None,
        }
    )


def _planner_node(state: AgentState) -> AgentState:
    if state.get("critic_verdict") == "fallback":
        return {}
    try:
        backend = _require_structured_backend()
        planner = _coerce_planner_output(_call_planner(backend, state["ctx"], state.get("risk_analysis")))
    except Exception as e:
        return {
            "planner_succeeded": False,
            "structured_output_succeeded": False,
            "critic_verdict": "fallback",
            "fallback_reason": f"Planner structured output failed: {type(e).__name__}: {e}",
        }
    return {
        "strategy_tag": planner.strategy_tag,
        "strategy_reasoning": planner.strategy_reasoning,
        "planner_succeeded": True,
    }


def _scheduler_node(state: AgentState) -> AgentState:
    if state.get("critic_verdict") == "fallback":
        return {}
    try:
        backend = _require_structured_backend()
        proposal = _coerce_scheduler_proposal(
            _call_scheduler(
                backend,
                state["ctx"],
                strategy_tag=state.get("strategy_tag") or "balanced",
                strategy_reasoning=state.get("strategy_reasoning") or "No planner reasoning recorded.",
                excluded_server_ids=set(state.get("excluded_server_ids") or []),
                critic_feedback=state.get("critic_feedback"),
                risk_analysis=state.get("risk_analysis"),
                memory_context=state.get("memory_context"),
            )
        )
    except Exception as e:
        return {
            "scheduler_error": f"{type(e).__name__}: {e}",
            "structured_output_succeeded": False,
        }
    return {
        "proposed_action": proposal.action,
        "proposed_server_id": proposal.server_id,
        "proposed_reasoning": proposal.reasoning,
        "structured_output_succeeded": True,
        "scheduler_error": None,
    }


def _critic_node(state: AgentState) -> AgentState:
    if state.get("fallback_reason"):
        return {"critic_verdict": "fallback", "critic_reasoning": state.get("fallback_reason")}
    if state.get("scheduler_error"):
        reason = f"Scheduler structured output failed: {state['scheduler_error']}"
        return {"critic_verdict": "fallback", "critic_reasoning": reason, "fallback_reason": reason}

    ctx = state["ctx"]
    action = state.get("proposed_action")
    excluded = set(state.get("excluded_server_ids") or [])
    revise_count = int(state.get("revise_count") or 0)
    proposal_id = state.get("proposed_server_id")

    if action == "reject":
        if not _valid_candidates(ctx, excluded):
            return {"critic_verdict": "approve", "critic_reasoning": "No valid candidate remains; rejection approved."}
        return _revise_or_fallback(
            state,
            revise_count=revise_count,
            excluded=excluded,
            reason="Scheduler rejected the service even though a valid candidate exists.",
        )

    if action != "select":
        return _revise_or_fallback(
            state,
            revise_count=revise_count,
            excluded=excluded,
            reason=f"Scheduler returned unsupported action: {action}",
        )

    proposal = next((server for server in ctx.servers if server.server_id == proposal_id), None)
    if proposal is None:
        if isinstance(proposal_id, int):
            excluded.add(proposal_id)
        return _revise_or_fallback(
            state,
            revise_count=revise_count,
            excluded=excluded,
            reason=f"Scheduler selected unknown server_id={proposal_id}.",
        )

    if proposal.server_id in excluded:
        return _revise_or_fallback(
            state,
            revise_count=revise_count,
            excluded=excluded,
            reason=f"Scheduler repeated excluded server_id={proposal.server_id}.",
        )

    verdict, critic_reasoning = _critic(ctx, proposal)
    if verdict == "approve":
        return {"critic_verdict": "approve", "critic_reasoning": critic_reasoning}

    excluded.add(proposal.server_id)
    return _revise_or_fallback(state, revise_count=revise_count, excluded=excluded, reason=critic_reasoning)


def _route_after_critic(state: AgentState) -> str:
    return "scheduler" if state.get("critic_verdict") == "revise" else "end"


def _revise_or_fallback(
    state: AgentState,
    *,
    revise_count: int,
    excluded: set[int],
    reason: str,
) -> AgentState:
    next_count = revise_count + 1
    if next_count >= _MAX_REVISIONS:
        fallback_reason = f"{reason} Maximum critic revisions reached; falling back to Balanced-Fit."
        return {
            "critic_verdict": "fallback",
            "critic_reasoning": fallback_reason,
            "fallback_reason": fallback_reason,
            "revise_count": next_count,
            "excluded_server_ids": sorted(excluded),
        }
    return {
        "critic_verdict": "revise",
        "critic_reasoning": reason,
        "critic_feedback": reason,
        "revise_count": next_count,
        "excluded_server_ids": sorted(excluded),
    }


def _decision_from_structured_state(
    state: AgentState,
    t0: float,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    extra = _merge_memory_metadata(extra, state.get("memory_context"))
    strategy_tag = state.get("strategy_tag")
    strategy_reasoning = state.get("strategy_reasoning")
    critic_verdict = state.get("critic_verdict")
    critic_reasoning = state.get("critic_reasoning")
    structured_ok = bool(state.get("structured_output_succeeded")) and critic_verdict == "approve"

    if critic_verdict != "approve":
        fallback_reason = state.get("fallback_reason") or critic_reasoning or "Structured graph failed to approve a decision."
        decision = SchedulingDecision(
            action="fallback",
            reasoning=fallback_reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        data = _record_decision(
            decision,
            t0=t0,
            messages=_messages(state["ctx"], strategy_tag, strategy_reasoning),
            tool_calls=[],
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            critic_verdict="fallback",
            critic_reasoning=critic_reasoning,
            structured_output_succeeded=False,
            revise_count=int(state.get("revise_count") or 0),
            fallback_reason=fallback_reason,
            extra=extra,
        )
        return data, -1

    if state.get("proposed_action") == "reject":
        reasoning = state.get("proposed_reasoning") or "Structured scheduler rejected the service."
        decision = SchedulingDecision(
            action="reject",
            reasoning=reasoning,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=True,
        )
        data = _record_decision(
            decision,
            t0=t0,
            messages=_messages(state["ctx"], strategy_tag, strategy_reasoning),
            tool_calls=[{"name": "reject_service", "args": {"reason": reasoning}}],
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            critic_verdict=critic_verdict,
            critic_reasoning=critic_reasoning,
            structured_output_succeeded=structured_ok,
            revise_count=int(state.get("revise_count") or 0),
            extra=extra,
        )
        return data, -2

    server_id = int(state.get("proposed_server_id"))
    reasoning = state.get("proposed_reasoning") or f"Structured scheduler selected server {server_id}."
    decision = SchedulingDecision(
        action="select",
        server_id=server_id,
        reasoning=reasoning,
        latency_ms=(time.perf_counter() - t0) * 1000,
        tool_call_succeeded=True,
    )
    data = _record_decision(
        decision,
        t0=t0,
        messages=_messages(state["ctx"], strategy_tag, strategy_reasoning),
        tool_calls=[{"name": "select_server", "args": {"server_id": server_id, "reasoning": reasoning}}],
        strategy_tag=strategy_tag,
        strategy_reasoning=strategy_reasoning,
        critic_verdict=critic_verdict,
        critic_reasoning=critic_reasoning,
        structured_output_succeeded=structured_ok,
        revise_count=int(state.get("revise_count") or 0),
        extra=extra,
    )
    return data, server_id


def _require_structured_backend() -> StructuredBackend:
    if _STRUCTURED_BACKEND is None:
        raise RuntimeError("Structured backend is not initialized.")
    return _STRUCTURED_BACKEND


def _call_planner(
    backend: StructuredBackend,
    ctx: SchedulingContext,
    risk_analysis: dict[str, Any] | None,
) -> PlannerOutput | dict[str, Any]:
    try:
        return backend.plan(ctx, risk_analysis=risk_analysis)
    except TypeError:
        return backend.plan(ctx)


def _call_scheduler(
    backend: StructuredBackend,
    ctx: SchedulingContext,
    *,
    strategy_tag: str,
    strategy_reasoning: str,
    excluded_server_ids: set[int],
    critic_feedback: str | None,
    risk_analysis: dict[str, Any] | None,
    memory_context: dict[str, Any] | None,
) -> SchedulerProposal | dict[str, Any]:
    try:
        return backend.propose(
            ctx,
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            excluded_server_ids=excluded_server_ids,
            critic_feedback=critic_feedback,
            risk_analysis=risk_analysis,
            memory_context=memory_context,
        )
    except TypeError:
        return backend.propose(
            ctx,
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            excluded_server_ids=excluded_server_ids,
            critic_feedback=critic_feedback,
        )


def _coerce_planner_output(raw: PlannerOutput | dict[str, Any] | Any) -> PlannerOutput:
    if isinstance(raw, PlannerOutput):
        return raw
    if isinstance(raw, BaseModel):
        return PlannerOutput.model_validate(raw.model_dump())
    return PlannerOutput.model_validate(raw)


def _coerce_scheduler_proposal(raw: SchedulerProposal | dict[str, Any] | Any) -> SchedulerProposal:
    if isinstance(raw, SchedulerProposal):
        return raw
    if isinstance(raw, BaseModel):
        return SchedulerProposal.model_validate(raw.model_dump())
    return SchedulerProposal.model_validate(raw)


def _parse_context(servers_raw: list, service_req_raw: list) -> SchedulingContext:
    return SchedulingContext(
        servers=[
            ServerSnapshot(
                server_id=int(s[0]),
                cpu_free_pct=float(s[1]),
                ram_free_pct=float(s[2]),
                net_free_pct=float(s[3]),
            )
            for s in servers_raw
        ],
        service=ServiceRequest(
            cpu_pct=float(service_req_raw[0]),
            ram_pct=float(service_req_raw[1]),
            net_pct=float(service_req_raw[2]),
        ),
    )


def _parse_risk_snapshot(global_state_raw: Any | None) -> RiskSnapshot:
    if global_state_raw is None:
        return RiskSnapshot()
    if isinstance(global_state_raw, RiskSnapshot):
        return global_state_raw
    if isinstance(global_state_raw, dict):
        return RiskSnapshot.model_validate(global_state_raw)
    if isinstance(global_state_raw, (list, tuple)):
        data: dict[str, Any] = {}
        for item in global_state_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                data[str(item[0])] = item[1]
        return RiskSnapshot.model_validate(data)
    raise TypeError(f"Unsupported global risk state: {type(global_state_raw).__name__}")


def _parse_memory_context(memory_context_raw: Any | None) -> dict[str, Any]:
    working = ""
    episodic_raw: list[Any] = []
    if isinstance(memory_context_raw, dict):
        working = str(memory_context_raw.get("working") or "")
        value = memory_context_raw.get("episodic") or []
        episodic_raw = list(value) if isinstance(value, (list, tuple)) else []
    elif isinstance(memory_context_raw, (list, tuple)):
        episodic_raw = list(memory_context_raw)

    episodes: list[dict[str, Any]] = []
    for item in episodic_raw[:3]:
        if isinstance(item, BaseModel):
            episode = item.model_dump()
        elif isinstance(item, dict):
            episode = dict(item)
        else:
            episode = {"reasoning_trace": str(item)}
        episodes.append(episode)

    return {
        "working": working,
        "episodic": episodes,
        "retrieved_episode_count": len(episodes),
        "memory_used": bool(episodes),
        "memory_confidence": _memory_confidence(episodes),
    }


def _memory_confidence(episodes: list[dict[str, Any]]) -> float:
    if not episodes:
        return 0.0
    rewards: list[float] = []
    for episode in episodes:
        try:
            rewards.append(float(episode.get("reward", 0.0)))
        except (TypeError, ValueError):
            continue
    if rewards:
        return round(max(0.0, min(1.0, max(rewards))), 3)
    return round(min(1.0, len(episodes) / 3), 3)


def _merge_memory_metadata(
    extra: dict[str, Any] | None,
    memory_context: dict[str, Any] | None,
) -> dict[str, Any]:
    data = dict(extra or {})
    metadata = _parse_memory_context(memory_context)
    for key in ("retrieved_episode_count", "memory_used", "memory_confidence"):
        data.setdefault(key, metadata[key])
    return data


def _plan(ctx: SchedulingContext) -> tuple[str, str]:
    demand = {
        "cpu-pressure": ctx.service.cpu_pct,
        "memory-pressure": ctx.service.ram_pct,
        "network-pressure": ctx.service.net_pct,
    }
    tag = max(demand, key=demand.get)
    if max(demand.values()) - min(demand.values()) < 8:
        tag = "balanced"
    return tag, f"Selected {tag} because service demand is {demand}."


def _valid_candidates(ctx: SchedulingContext, excluded: set[int] | None = None) -> list[ServerSnapshot]:
    excluded = excluded or set()
    return [
        server
        for server in ctx.servers
        if server.server_id not in excluded
        and server.cpu_free_pct >= ctx.service.cpu_pct
        and server.ram_free_pct >= ctx.service.ram_pct
        and server.net_free_pct >= ctx.service.net_pct
    ]


def _propose(
    ctx: SchedulingContext,
    excluded: set[int],
    risk_policy: dict[str, Any] | None = None,
) -> ServerSnapshot | None:
    candidates = _valid_candidates(ctx, excluded)
    if not candidates:
        return None

    return min(candidates, key=lambda server: _candidate_score(ctx, server, risk_policy=risk_policy))


def _candidate_score(
    ctx: SchedulingContext,
    server: ServerSnapshot,
    risk_policy: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    residual = [
        server.cpu_free_pct - ctx.service.cpu_pct,
        server.ram_free_pct - ctx.service.ram_pct,
        server.net_free_pct - ctx.service.net_pct,
    ]
    spread = statistics.pstdev(residual)
    if risk_policy and risk_policy.get("avoid_tight_headroom"):
        weights = risk_policy.get("resource_weights") or {}
        weighted_headroom = (
            residual[0] * float(weights.get("cpu", 1.0))
            + residual[1] * float(weights.get("mem", 1.0))
            + residual[2] * float(weights.get("net", 1.0))
        )
        return (-weighted_headroom, spread, sum(residual), server.server_id)
    return (spread, sum(residual), server.server_id)


def _cluster_fragmentation(ctx: SchedulingContext) -> float:
    if not ctx.servers:
        return 0.0
    values = []
    for server in ctx.servers:
        values.extend([server.cpu_free_pct, server.ram_free_pct, server.net_free_pct])
    return statistics.pstdev(values)


def _critic(ctx: SchedulingContext, proposal: ServerSnapshot) -> tuple[str, str]:
    residual = {
        "CPU": proposal.cpu_free_pct - ctx.service.cpu_pct,
        "RAM": proposal.ram_free_pct - ctx.service.ram_pct,
        "NET": proposal.net_free_pct - ctx.service.net_pct,
    }
    if min(residual.values()) < 0:
        return "revise", f"server {proposal.server_id} would exceed headroom: {residual}"
    return "approve", f"residual headroom after placement is {residual}"


def _messages(ctx: SchedulingContext, strategy_tag: str, strategy_reasoning: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Planner-Scheduler-Critic cloud scheduler."},
        {
            "role": "user",
            "content": f"Strategy: {strategy_tag}. {strategy_reasoning}\n\n{render_cluster_state(ctx)}",
        },
    ]


def _record_decision(
    decision: SchedulingDecision,
    *,
    t0: float,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    strategy_tag: str | None,
    strategy_reasoning: str | None = None,
    critic_verdict: str | None = None,
    critic_reasoning: str | None = None,
    structured_output_succeeded: bool = False,
    revise_count: int = 0,
    fallback_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = decision.model_dump()
    data.update(
        {
            "phase": "phase2",
            "model": _MODEL_NAME,
            "backend": _BACKEND_MODE,
            "strategy_tag": strategy_tag,
            "strategy_reasoning": strategy_reasoning,
            "planner_policy_active": strategy_tag is not None,
            "structured_output_succeeded": structured_output_succeeded,
            "critic_verdict": critic_verdict,
            "critic_reasoning": critic_reasoning,
            "revise_count": revise_count,
            "fallback_reason": fallback_reason,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    )
    if extra:
        data.update(extra)
    if _TRACE_LOGGER is not None:
        _TRACE_LOGGER.write(
            messages=messages,
            tool_calls=tool_calls,
            decision=data,
            latency_ms=data["latency_ms"],
            fallback_reason=fallback_reason,
        )
    return data


def _empty_hybrid_stats() -> dict[str, Any]:
    return {
        "total_decisions": 0,
        "hybrid_decisions": 0,
        "fast_path_decisions": 0,
        "agent_escalation_needed": 0,
        "agent_sync_calls": 0,
        "structured_backend_calls": 0,
        "agent_call_decisions": 0,
        "global_risk_agent_triggers": 0,
        "fallback_decisions": 0,
        "planner_policy_active": 0,
        "memory_used_decisions": 0,
        "retrieved_episode_count": 0,
        "total_latency_ms": 0.0,
        "complexity_reason_counts": {},
        "risk_tag_counts": {},
    }


def _record_hybrid_stats(decision: dict[str, Any]) -> None:
    if not _HYBRID_STATS:
        _HYBRID_STATS.update(_empty_hybrid_stats())

    backend = decision.get("backend")
    fast_path_used = decision.get("fast_path_used")
    is_hybrid = backend == "hybrid"
    is_structured = backend == "structured"
    is_hybrid_agent_call = is_hybrid and fast_path_used is False
    is_structured_agent_call = is_structured and not _is_input_validation_fallback(decision)
    is_agent_call = is_hybrid_agent_call or is_structured_agent_call

    _HYBRID_STATS["total_decisions"] += 1
    if is_hybrid:
        _HYBRID_STATS["hybrid_decisions"] += 1
    if fast_path_used is True:
        _HYBRID_STATS["fast_path_decisions"] += 1
    if decision.get("agent_escalation_needed") is True:
        _HYBRID_STATS["agent_escalation_needed"] += 1
    if decision.get("global_risk_agent_triggered") is True:
        _HYBRID_STATS["global_risk_agent_triggers"] += 1
    if is_hybrid_agent_call:
        _HYBRID_STATS["agent_sync_calls"] += 1
    if is_structured_agent_call:
        _HYBRID_STATS["structured_backend_calls"] += 1
    if is_agent_call:
        _HYBRID_STATS["agent_call_decisions"] += 1
    if decision.get("action") == "fallback":
        _HYBRID_STATS["fallback_decisions"] += 1
    if decision.get("planner_policy_active") is True:
        _HYBRID_STATS["planner_policy_active"] += 1
    if decision.get("memory_used") is True:
        _HYBRID_STATS["memory_used_decisions"] += 1
    _HYBRID_STATS["retrieved_episode_count"] += int(decision.get("retrieved_episode_count") or 0)
    _HYBRID_STATS["total_latency_ms"] += float(decision.get("latency_ms") or 0.0)

    reason_counts = _HYBRID_STATS["complexity_reason_counts"]
    for reason in decision.get("complexity_reasons") or []:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    risk_tag_counts = _HYBRID_STATS["risk_tag_counts"]
    for tag in decision.get("global_risk_tags") or []:
        risk_tag_counts[tag] = risk_tag_counts.get(tag, 0) + 1


def _stats_with_ratios(stats: dict[str, Any]) -> dict[str, Any]:
    total = stats.get("total_decisions", 0)
    hybrid_total = stats.get("hybrid_decisions", 0)
    data = dict(stats or _empty_hybrid_stats())
    data["risk_tag_counts"] = dict(data.get("risk_tag_counts", {}))
    data.update(
        {
            "fast_path_ratio": _safe_ratio(data["fast_path_decisions"], hybrid_total),
            "escalation_ratio": _safe_ratio(data["agent_escalation_needed"], hybrid_total),
            "global_risk_trigger_ratio": _safe_ratio(data["global_risk_agent_triggers"], hybrid_total),
            "hybrid_agent_call_ratio": _safe_ratio(data["agent_sync_calls"], hybrid_total),
            "agent_call_ratio": _safe_ratio(data["agent_call_decisions"], total),
            "structured_backend_ratio": _safe_ratio(data["structured_backend_calls"], total),
            "fallback_ratio": _safe_ratio(data["fallback_decisions"], total),
            "planner_policy_ratio": _safe_ratio(data["planner_policy_active"], total),
            "memory_usage_ratio": _safe_ratio(data["memory_used_decisions"], total),
            "avg_retrieved_episode_count": _safe_ratio(data["retrieved_episode_count"], total),
            "avg_latency_ms": _safe_ratio(data["total_latency_ms"], total),
        }
    )
    return data


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _is_input_validation_fallback(decision: dict[str, Any]) -> bool:
    fallback_reason = str(decision.get("fallback_reason") or decision.get("reasoning") or "")
    return decision.get("action") == "fallback" and fallback_reason.startswith("Input validation error:")


def _planner_prompt(ctx: SchedulingContext, risk_analysis: dict[str, Any] | None = None) -> str:
    return (
        "/no_think\n"
        "You are the Planner agent in a cloud scheduling system. "
        "Classify the incoming request and cluster state into one strategy_tag. "
        "Use concise reasoning. Valid examples include balanced, cpu-pressure, "
        "memory-pressure, network-pressure, and bursty.\n\n"
        f"{_risk_context_text(risk_analysis)}"
        f"{render_cluster_state(ctx)}"
    )


def _scheduler_prompt(
    ctx: SchedulingContext,
    *,
    strategy_tag: str,
    strategy_reasoning: str,
    excluded_server_ids: set[int],
    critic_feedback: str | None,
    risk_analysis: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> str:
    excluded = ", ".join(str(item) for item in sorted(excluded_server_ids)) or "none"
    feedback = critic_feedback or "none"
    return (
        "/no_think\n"
        "You are the Scheduler agent in a Planner-Scheduler-Critic cloud scheduling graph. "
        "Return a structured action only: select a valid server_id or reject the service. "
        "A server is valid only if CPU, RAM, and NET free percentages are all at least the "
        "incoming service demand. Do not select excluded servers.\n\n"
        f"Strategy: {strategy_tag}\n"
        f"Planner reasoning: {strategy_reasoning}\n"
        f"Excluded server IDs: {excluded}\n"
        f"Critic feedback from previous attempt: {feedback}\n\n"
        f"{_risk_context_text(risk_analysis)}"
        f"{_memory_context_text(memory_context)}"
        f"{render_cluster_state(ctx)}"
    )


def _risk_context_text(risk_analysis: dict[str, Any] | None) -> str:
    if not risk_analysis:
        return ""
    return (
        "## Global risk context\n"
        f"- Risk score: {risk_analysis.get('global_risk_score', 0)}\n"
        f"- Risk level: {risk_analysis.get('global_risk_level', 'low')}\n"
        f"- Risk tags: {risk_analysis.get('global_risk_tags', [])}\n"
        f"- Policy: {risk_analysis.get('risk_policy', {})}\n\n"
    )


def _memory_context_text(memory_context: dict[str, Any] | None) -> str:
    memory = _parse_memory_context(memory_context)
    if not memory["memory_used"]:
        return ""
    lines = [
        "## Retrieved scheduling memory",
        f"- Retrieved episodes: {memory['retrieved_episode_count']}",
        f"- Retrieval confidence: {memory['memory_confidence']}",
    ]
    if memory["working"]:
        lines.append(f"- Working memory: {memory['working']}")
    for index, episode in enumerate(memory["episodic"], start=1):
        action = episode.get("action_server_id", "unknown")
        reasoning = episode.get("reasoning_trace") or episode.get("reasoning") or "No reasoning recorded."
        lines.append(f"- Episode {index}: previous action={action}; reasoning={reasoning}")
    return "\n".join(lines) + "\n\n"
