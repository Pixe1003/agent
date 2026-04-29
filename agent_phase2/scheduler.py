from __future__ import annotations

import statistics
import time
from typing import Any

from agent_common.tracing import TraceLogger
from agent_phase1.prompts import render_cluster_state
from agent_phase1.schemas import (
    SchedulingContext,
    SchedulingDecision,
    ServerSnapshot,
    ServiceRequest,
)
from pydantic import ValidationError


_MODEL_NAME = "heuristic"
_TRACE_LOGGER: TraceLogger | None = None
_LAST_DECISION: dict[str, Any] = {}


def init_agent(
    model_name: str = "heuristic",
    temperature: float = 0.0,
    num_predict: int = 128,
    trace_dir: str = "traces",
    run_id: str | None = None,
    enable_tracing: bool = True,
    **_: Any,
) -> None:
    """Initialize Phase 2.

    The first implementation is a deterministic Planner-Scheduler-Critic
    skeleton. It keeps the NetLogo API stable while the LLM/LangGraph backend
    is integrated later.
    """
    global _MODEL_NAME, _TRACE_LOGGER
    _MODEL_NAME = model_name
    _TRACE_LOGGER = TraceLogger(
        trace_dir=trace_dir,
        run_id=run_id,
        phase="phase2",
        model=model_name,
        enabled=enable_tracing,
    )


def schedule_service(servers_raw: list, service_req_raw: list) -> int:
    global _LAST_DECISION
    t0 = time.perf_counter()

    try:
        ctx = _parse_context(servers_raw, service_req_raw)
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
            fallback_reason=decision.reasoning,
        )
        return -1

    strategy_tag, strategy_reasoning = _plan(ctx)
    excluded: set[int] = set()
    revise_count = 0

    while revise_count <= 2:
        proposal = _propose(ctx, excluded)
        if proposal is None:
            decision = SchedulingDecision(
                action="reject",
                reasoning="No server can host the service without exceeding resource headroom.",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            _LAST_DECISION = _record_decision(
                decision,
                t0=t0,
                messages=_messages(ctx, strategy_tag, strategy_reasoning),
                tool_calls=[{"name": "reject_service", "args": {"reason": decision.reasoning}}],
                strategy_tag=strategy_tag,
                strategy_reasoning=strategy_reasoning,
                critic_verdict="approve",
                revise_count=revise_count,
            )
            return -2

        verdict, critic_reasoning = _critic(ctx, proposal)
        if verdict == "approve":
            decision = SchedulingDecision(
                action="select",
                server_id=proposal.server_id,
                reasoning=f"{strategy_tag}: approved server {proposal.server_id}; {critic_reasoning}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            _LAST_DECISION = _record_decision(
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
                revise_count=revise_count,
            )
            return proposal.server_id

        excluded.add(proposal.server_id)
        revise_count += 1

    decision = SchedulingDecision(
        action="fallback",
        reasoning="Critic requested too many revisions; falling back to Balanced-Fit.",
        latency_ms=(time.perf_counter() - t0) * 1000,
        tool_call_succeeded=False,
    )
    _LAST_DECISION = _record_decision(
        decision,
        t0=t0,
        messages=_messages(ctx, strategy_tag, strategy_reasoning),
        tool_calls=[],
        strategy_tag=strategy_tag,
        strategy_reasoning=strategy_reasoning,
        critic_verdict="fallback",
        revise_count=revise_count,
        fallback_reason=decision.reasoning,
    )
    return -1


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


def _propose(ctx: SchedulingContext, excluded: set[int]) -> ServerSnapshot | None:
    candidates = _valid_candidates(ctx, excluded)
    if not candidates:
        return None

    def score(server: ServerSnapshot) -> tuple[float, float, int]:
        residual = [
            server.cpu_free_pct - ctx.service.cpu_pct,
            server.ram_free_pct - ctx.service.ram_pct,
            server.net_free_pct - ctx.service.net_pct,
        ]
        spread = statistics.pstdev(residual)
        return (spread, sum(residual), server.server_id)

    return min(candidates, key=score)


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
    revise_count: int = 0,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    data = decision.model_dump()
    data.update(
        {
            "phase": "phase2",
            "model": _MODEL_NAME,
            "strategy_tag": strategy_tag,
            "strategy_reasoning": strategy_reasoning,
            "critic_verdict": critic_verdict,
            "critic_reasoning": critic_reasoning,
            "revise_count": revise_count,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    )
    if _TRACE_LOGGER is not None:
        _TRACE_LOGGER.write(
            messages=messages,
            tool_calls=tool_calls,
            decision=data,
            latency_ms=data["latency_ms"],
            fallback_reason=fallback_reason,
        )
    return data

