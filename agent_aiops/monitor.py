from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from agent_common.tracing import TraceLogger
from pydantic import BaseModel, Field, ValidationError


_MODEL_NAME = "heuristic"
_BACKEND: Literal["rule", "structured"] = "rule"
_TRACE_LOGGER: TraceLogger | None = None
_LAST_INSIGHT: dict[str, Any] = {}
_WINDOW_SIZE = 5
_RECOMMENDATION_COOLDOWN = 10
_OBSERVED_EVENTS = 0
_WINDOW: list[dict[str, Any]] = []
_ACTIVE_ALERTS: list[dict[str, Any]] = []
_LAST_RECOMMENDATION_TICK: int | None = None


class OpsSnapshot(BaseModel):
    service_placement_algorithm: str = "unknown"
    active_cpu_util: float = Field(default=0.0, ge=0.0)
    active_mem_util: float = Field(default=0.0, ge=0.0)
    active_net_util: float = Field(default=0.0, ge=0.0)
    active_servers: int = Field(default=0, ge=0)
    current_auto_migrations: int = Field(default=0, ge=0)
    current_consolidation_migrations: int = Field(default=0, ge=0)
    rescheduled_services: int = Field(default=0, ge=0)
    rejected_services: int = Field(default=0, ge=0)
    ops_sla_violations: float = Field(default=0.0, ge=0.0)
    mem_sla_violations: float = Field(default=0.0, ge=0.0)
    net_sla_violations: float = Field(default=0.0, ge=0.0)
    energy_regression_pct: float = Field(default=0.0, ge=0.0)


def init_agent(
    model_name: str = "heuristic",
    backend: Literal["rule", "structured"] = "rule",
    enable_tracing: bool = True,
    trace_dir: str = "traces",
    run_id: str | None = None,
    window_size: int = 5,
    recommendation_cooldown: int = 10,
) -> None:
    global _MODEL_NAME, _BACKEND, _TRACE_LOGGER, _LAST_INSIGHT
    global _WINDOW_SIZE, _RECOMMENDATION_COOLDOWN, _OBSERVED_EVENTS, _WINDOW, _ACTIVE_ALERTS, _LAST_RECOMMENDATION_TICK
    _MODEL_NAME = model_name
    _BACKEND = backend
    _LAST_INSIGHT = {}
    _WINDOW_SIZE = max(1, int(window_size))
    _RECOMMENDATION_COOLDOWN = max(0, int(recommendation_cooldown))
    _OBSERVED_EVENTS = 0
    _WINDOW = []
    _ACTIVE_ALERTS = []
    _LAST_RECOMMENDATION_TICK = None
    _TRACE_LOGGER = TraceLogger(
        trace_dir=trace_dir,
        run_id=run_id or f"aiops-{uuid.uuid4().hex[:10]}",
        phase="aiops",
        model=model_name,
        enabled=enable_tracing,
    )


def analyze_ops_state(
    global_state_raw: Any,
    scheduler_stats_raw: Any | None = None,
    recent_decisions_raw: Any | None = None,
    memory_context_raw: Any | None = None,
    server_snapshots_raw: Any | None = None,
) -> dict[str, Any]:
    global _LAST_INSIGHT
    t0 = time.perf_counter()
    try:
        snapshot = _parse_ops_snapshot(global_state_raw)
    except (ValidationError, ValueError, TypeError) as e:
        insight = _invalid_insight(e, scheduler_stats_raw, recent_decisions_raw, memory_context_raw, server_snapshots_raw)
        _LAST_INSIGHT = _record_insight(insight, t0, fallback_reason=str(e))
        return dict(_LAST_INSIGHT)

    risk = _analyze_risk(snapshot)
    evidence = _build_evidence(snapshot, scheduler_stats_raw, recent_decisions_raw, memory_context_raw, server_snapshots_raw)
    recommendations = _recommend(risk["risk_tags"], snapshot)
    insight = {
        "status": "ok",
        "phase": "aiops",
        "model": _MODEL_NAME,
        "backend": _BACKEND,
        "agent_topology": "logical-multi-role",
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_tags": risk["risk_tags"],
        "root_cause_summary": _root_cause_summary(risk["risk_tags"]),
        "recommendations": recommendations,
        "guardrails": _guardrails(risk["risk_tags"]),
        "evidence": evidence,
        "roles": [
            "risk-analyzer",
            "sla-migration-analyzer",
            "memory-context",
            "policy-advisor",
            "harness-guard",
        ],
    }
    _LAST_INSIGHT = _record_insight(insight, t0)
    return dict(_LAST_INSIGHT)


def observe_ops_state(
    global_state_raw: Any,
    scheduler_stats_raw: Any | None = None,
    recent_decisions_raw: Any | None = None,
    memory_context_raw: Any | None = None,
    server_snapshots_raw: Any | None = None,
    tick: int | None = None,
) -> dict[str, Any]:
    global _LAST_INSIGHT, _OBSERVED_EVENTS, _WINDOW, _ACTIVE_ALERTS, _LAST_RECOMMENDATION_TICK
    t0 = time.perf_counter()
    _OBSERVED_EVENTS += 1
    observed_tick = int(tick) if tick is not None else _OBSERVED_EVENTS

    try:
        snapshot = _parse_ops_snapshot(global_state_raw)
    except (ValidationError, ValueError, TypeError) as e:
        insight = _invalid_insight(e, scheduler_stats_raw, recent_decisions_raw, memory_context_raw, server_snapshots_raw)
        insight["mode"] = "realtime"
        insight["tick"] = observed_tick
        _append_window(insight)
        _ACTIVE_ALERTS = _compute_alerts(_WINDOW)
        insight["active_alerts"] = list(_ACTIVE_ALERTS)
        _LAST_INSIGHT = _record_insight(insight, t0, fallback_reason=str(e))
        return dict(_LAST_INSIGHT)

    risk = _analyze_risk(snapshot)
    evidence = _build_evidence(snapshot, scheduler_stats_raw, recent_decisions_raw, memory_context_raw, server_snapshots_raw)
    recommendations = _recommend(risk["risk_tags"], snapshot)
    suppressed = False
    if recommendations and _LAST_RECOMMENDATION_TICK is not None:
        suppressed = observed_tick - _LAST_RECOMMENDATION_TICK < _RECOMMENDATION_COOLDOWN
    if suppressed:
        recommendations = []
    elif recommendations:
        _LAST_RECOMMENDATION_TICK = observed_tick

    insight = {
        "status": "ok",
        "phase": "aiops",
        "mode": "realtime",
        "tick": observed_tick,
        "model": _MODEL_NAME,
        "backend": _BACKEND,
        "agent_topology": "logical-multi-role",
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_tags": risk["risk_tags"],
        "root_cause_summary": _root_cause_summary(risk["risk_tags"]),
        "recommendations": recommendations,
        "guardrails": _guardrails(
            risk["risk_tags"],
            recommendation_suppressed_by_cooldown=suppressed,
        ),
        "evidence": evidence,
        "roles": [
            "risk-analyzer",
            "sla-migration-analyzer",
            "memory-context",
            "policy-advisor",
            "harness-guard",
        ],
    }
    _append_window(insight)
    _ACTIVE_ALERTS = _compute_alerts(_WINDOW)
    insight["active_alerts"] = list(_ACTIVE_ALERTS)
    _LAST_INSIGHT = _record_insight(insight, t0)
    return dict(_LAST_INSIGHT)


def current_alerts() -> list[dict[str, Any]]:
    return [dict(alert) for alert in _ACTIVE_ALERTS]


def aiops_stats() -> dict[str, Any]:
    risk_levels: dict[str, int] = {}
    for item in _WINDOW:
        level = str(item.get("risk_level", "low"))
        risk_levels[level] = risk_levels.get(level, 0) + 1
    return {
        "observed_events": _OBSERVED_EVENTS,
        "window_size": _WINDOW_SIZE,
        "window_count": len(_WINDOW),
        "active_alert_count": len(_ACTIVE_ALERTS),
        "active_alert_tags": [alert["tag"] for alert in _ACTIVE_ALERTS],
        "risk_level_counts": risk_levels,
        "last_risk_level": _LAST_INSIGHT.get("risk_level"),
        "last_risk_score": _LAST_INSIGHT.get("risk_score"),
    }


def aiops_stats_summary() -> str:
    stats = aiops_stats()
    tags = ",".join(stats["active_alert_tags"]) or "none"
    return (
        f"aiops observed={stats['observed_events']} "
        f"window={stats['window_count']}/{stats['window_size']} "
        f"alerts={stats['active_alert_count']} tags={tags}"
    )


def last_insight_dict() -> dict[str, Any]:
    return dict(_LAST_INSIGHT)


def last_insight_summary() -> str:
    if not _LAST_INSIGHT:
        return "aiops no insight yet"
    tags = ",".join(_LAST_INSIGHT.get("risk_tags") or []) or "none"
    return (
        f"aiops risk={_LAST_INSIGHT.get('risk_level')} "
        f"score={_LAST_INSIGHT.get('risk_score', 0.0):.3f} "
        f"tags={tags} "
        f"recommendations={len(_LAST_INSIGHT.get('recommendations') or [])}"
    )


def _parse_ops_snapshot(global_state_raw: Any) -> OpsSnapshot:
    if global_state_raw is None:
        return OpsSnapshot()
    if isinstance(global_state_raw, OpsSnapshot):
        return global_state_raw
    if isinstance(global_state_raw, dict):
        return OpsSnapshot.model_validate(global_state_raw)
    if isinstance(global_state_raw, (list, tuple)):
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in global_state_raw):
            data = {str(key): value for key, value in global_state_raw}
        else:
            keys = list(OpsSnapshot.model_fields)
            data = {key: value for key, value in zip(keys, global_state_raw)}
        return OpsSnapshot.model_validate(data)
    raise TypeError(f"Unsupported AIOps state: {type(global_state_raw).__name__}")


def _analyze_risk(snapshot: OpsSnapshot) -> dict[str, Any]:
    score = 0.0
    tags: list[str] = []
    cpu_util = _normalize_utilization(snapshot.active_cpu_util)
    mem_util = _normalize_utilization(snapshot.active_mem_util)
    net_util = _normalize_utilization(snapshot.active_net_util)

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

    if snapshot.current_auto_migrations >= 5:
        score += 0.25
        tags.append("migration-pressure")
    elif snapshot.current_auto_migrations > 0:
        score += 0.1
        tags.append("migration-watch")

    if snapshot.current_consolidation_migrations >= 5:
        score += 0.15
        tags.append("consolidation-pressure")

    if snapshot.net_sla_violations > 0:
        score += 0.35
        if "network-pressure" not in tags:
            tags.append("network-pressure")
        tags.append("sla-risk")
    elif snapshot.ops_sla_violations > 0 or snapshot.mem_sla_violations > 0:
        score += 0.2
        tags.append("sla-risk")

    if snapshot.rescheduled_services >= 5:
        score += 0.15
        tags.append("reschedule-pressure")
    elif snapshot.rescheduled_services > 0:
        score += 0.1
        tags.append("reschedule-watch")

    if snapshot.rejected_services > 0:
        score += 0.2
        tags.append("capacity-risk")

    if snapshot.energy_regression_pct >= 0.1:
        score += 0.15
        tags.append("energy-regression")

    score = min(score, 1.0)
    return {
        "risk_score": round(score, 3),
        "risk_level": _risk_level(score),
        "risk_tags": _unique(tags),
    }


def _recommend(tags: list[str], snapshot: OpsSnapshot) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    if "network-pressure" in tags or "network-watch" in tags:
        recommendations.append(
            _recommendation(
                action="enable-network-headroom-protection",
                reason="Network utilization or network SLA violations indicate the scheduler is entering a network-risk zone.",
                expected_effect="Preserve network headroom and reduce network-focused SLA violations in the next control window.",
            )
        )
    if "migration-pressure" in tags or "migration-watch" in tags:
        recommendations.append(
            _recommendation(
                action="increase-migration-cooldown",
                reason=f"Auto migrations are elevated at {snapshot.current_auto_migrations}.",
                expected_effect="Stabilize placements and reduce migration churn before the next policy change.",
            )
        )
    if "sla-risk" in tags and "energy-regression" in tags:
        recommendations.append(
            _recommendation(
                action="run-balanced-fit-regression-check",
                reason="SLA risk is rising while energy has regressed, so the current policy may be trading power for little reliability gain.",
                expected_effect="Identify whether reverting to balanced-fit or retuning thresholds improves the SLA/energy trade-off.",
            )
        )
    if "capacity-risk" in tags or "reschedule-pressure" in tags or "reschedule-watch" in tags:
        recommendations.append(
            _recommendation(
                action="review-capacity-and-admission-thresholds",
                reason="Rejected or rescheduled services indicate the cluster may be near an admission or capacity boundary.",
                expected_effect="Reduce repeated rescheduling and make capacity pressure visible before SLA impact grows.",
            )
        )
    return recommendations


def _recommendation(action: str, reason: str, expected_effect: str) -> dict[str, Any]:
    return {
        "action": action,
        "reason": reason,
        "expected_effect": expected_effect,
        "requires_human_approval": True,
    }


def _guardrails(tags: list[str], *, recommendation_suppressed_by_cooldown: bool = False) -> dict[str, Any]:
    return {
        "do_not_auto_apply": True,
        "recommendation_suppressed_by_cooldown": recommendation_suppressed_by_cooldown,
        "rollback_condition": "Rollback any approved policy change if SLA violations or migration rate worsens in the next evaluation window.",
        "cooldown_hint": "Apply at most one policy change per health-check window and re-run AIOps analysis before another change.",
        "approval_reason": "AIOps v1 is advisory only; realtime scheduling remains deterministic.",
        "active_tags": list(tags),
    }


def _build_evidence(
    snapshot: OpsSnapshot,
    scheduler_stats_raw: Any | None,
    recent_decisions_raw: Any | None,
    memory_context_raw: Any | None,
    server_snapshots_raw: Any | None,
) -> dict[str, Any]:
    return {
        "metrics": snapshot.model_dump(),
        "scheduler_stats": _dict_or_empty(scheduler_stats_raw),
        "recent_decisions": _summarize_recent_decisions(recent_decisions_raw),
        "memory": _summarize_memory(memory_context_raw),
        "server_snapshots": _summarize_servers(server_snapshots_raw),
    }


def _summarize_recent_decisions(raw: Any | None) -> dict[str, Any]:
    if raw is None:
        return {"count": 0, "items": []}
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    normalized = []
    for item in items[-5:]:
        if isinstance(item, dict):
            normalized.append(
                {
                    "action": item.get("action"),
                    "server_id": item.get("server_id"),
                    "risk_level": item.get("global_risk_level"),
                    "risk_tags": item.get("global_risk_tags") or [],
                }
            )
        else:
            normalized.append({"summary": str(item)})
    return {"count": len(items), "items": normalized}


def _summarize_memory(raw: Any | None) -> dict[str, Any]:
    if raw is None:
        episodes: list[Any] = []
    elif isinstance(raw, dict):
        episodes = list(raw.get("episodic") or [])
    elif isinstance(raw, (list, tuple)):
        episodes = list(raw)
    else:
        episodes = [{"reasoning_trace": str(raw)}]

    normalized = []
    for episode in episodes[:3]:
        if isinstance(episode, dict):
            normalized.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "action_server_id": episode.get("action_server_id"),
                    "reward": episode.get("reward"),
                    "reasoning_trace": episode.get("reasoning_trace") or episode.get("reasoning"),
                }
            )
        else:
            normalized.append({"reasoning_trace": str(episode)})
    return {
        "memory_used": bool(normalized),
        "retrieved_episode_count": len(normalized),
        "episodes": normalized,
    }


def _summarize_servers(raw: Any | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    servers = []
    for item in items:
        if isinstance(item, dict):
            cpu = _coerce_percent(item.get("cpu", item.get("cpu_util", item.get("cpu_utilization", 0.0))))
            mem = _coerce_percent(item.get("mem", item.get("mem_util", item.get("mem_utilization", 0.0))))
            net = _coerce_percent(item.get("net", item.get("net_util", item.get("net_utilization", 0.0))))
            server_id = item.get("id", item.get("server_id"))
            status = item.get("status") or _server_status(cpu, mem, net)
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            server_id = item[0]
            cpu = round(100.0 - float(item[1]), 3)
            mem = round(100.0 - float(item[2]), 3)
            net = round(100.0 - float(item[3]), 3)
            status = _server_status(cpu, mem, net)
        else:
            continue
        servers.append(
            {
                "id": int(server_id),
                "status": str(status),
                "cpu": cpu,
                "mem": mem,
                "net": net,
            }
        )
    return servers


def _invalid_insight(
    error: Exception,
    scheduler_stats_raw: Any | None,
    recent_decisions_raw: Any | None,
    memory_context_raw: Any | None,
    server_snapshots_raw: Any | None = None,
) -> dict[str, Any]:
    return {
        "status": "invalid",
        "phase": "aiops",
        "model": _MODEL_NAME,
        "backend": _BACKEND,
        "agent_topology": "logical-multi-role",
        "risk_level": "low",
        "risk_score": 0.0,
        "risk_tags": ["input-validation-error"],
        "root_cause_summary": f"Input validation error prevented AIOps analysis: {error}",
        "recommendations": [],
        "guardrails": _guardrails(["input-validation-error"]),
        "evidence": {
            "metrics": {},
            "scheduler_stats": _dict_or_empty(scheduler_stats_raw),
            "recent_decisions": _summarize_recent_decisions(recent_decisions_raw),
            "memory": _summarize_memory(memory_context_raw),
            "server_snapshots": _summarize_servers(server_snapshots_raw),
        },
        "roles": [
            "risk-analyzer",
            "sla-migration-analyzer",
            "memory-context",
            "policy-advisor",
            "harness-guard",
        ],
    }


def _record_insight(insight: dict[str, Any], t0: float, fallback_reason: str | None = None) -> dict[str, Any]:
    data = dict(insight)
    data["latency_ms"] = (time.perf_counter() - t0) * 1000
    if _TRACE_LOGGER is not None:
        _TRACE_LOGGER.write(
            decision=data,
            latency_ms=data["latency_ms"],
            fallback_reason=fallback_reason,
            extra={"insight_status": data.get("status")},
        )
    return data


def _append_window(insight: dict[str, Any]) -> None:
    _WINDOW.append(
        {
            "tick": insight.get("tick"),
            "risk_level": insight.get("risk_level"),
            "risk_score": insight.get("risk_score", 0.0),
            "risk_tags": list(insight.get("risk_tags") or []),
        }
    )
    del _WINDOW[:-_WINDOW_SIZE]


def _compute_alerts(window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not window:
        return []
    counts: dict[str, int] = {}
    first_tick: dict[str, int | None] = {}
    last_tick: dict[str, int | None] = {}
    latest = window[-1]
    for item in window:
        for tag in item.get("risk_tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
            first_tick.setdefault(tag, item.get("tick"))
            last_tick[tag] = item.get("tick")
    alerts = []
    for tag in latest.get("risk_tags") or []:
        alerts.append(
            {
                "tag": tag,
                "risk_level": latest.get("risk_level"),
                "risk_score": latest.get("risk_score", 0.0),
                "occurrence_count": counts.get(tag, 0),
                "window_count": len(window),
                "first_tick": first_tick.get(tag),
                "last_tick": last_tick.get(tag),
            }
        )
    return alerts


def _root_cause_summary(tags: list[str]) -> str:
    if not tags:
        return "No elevated AIOps risk signals detected."
    if "network-pressure" in tags and "sla-risk" in tags:
        return "Network pressure is the dominant SLA risk signal."
    if "migration-pressure" in tags:
        return "Migration churn is elevated and may destabilize placement quality."
    if "capacity-risk" in tags:
        return "Rejected services indicate possible capacity or admission threshold pressure."
    if "energy-regression" in tags:
        return "Energy usage regressed and should be compared against scheduler quality changes."
    return f"Elevated AIOps signals detected: {', '.join(tags)}."


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


def _coerce_percent(value: Any) -> float:
    number = float(value or 0.0)
    return round(number * 100 if 0 <= number <= 1 else number, 3)


def _server_status(cpu: float, mem: float, net: float) -> str:
    peak = max(cpu, mem, net)
    if peak >= 85:
        return "overload"
    if peak >= 72:
        return "warning"
    return "normal"


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _dict_or_empty(value: Any | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
