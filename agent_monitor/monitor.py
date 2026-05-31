from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MonitorState(IntEnum):
    NORMAL = 0
    WATCH = 1
    RISK = 2
    PROTECTED = 3
    CRITICAL = 4


STATE_LABELS = {
    MonitorState.NORMAL: "Normal",
    MonitorState.WATCH: "Watch",
    MonitorState.RISK: "Risk",
    MonitorState.PROTECTED: "Protected",
    MonitorState.CRITICAL: "Critical",
}

STATE_PROBABILITIES = {
    MonitorState.NORMAL: (0.02, 0.01),
    MonitorState.WATCH: (0.12, 0.05),
    MonitorState.RISK: (0.42, 0.22),
    MonitorState.PROTECTED: (0.72, 0.38),
    MonitorState.CRITICAL: (0.92, 0.68),
}

_TAG_TO_EVENT = {
    "cpu-pressure": "cpu_pressure",
    "memory-pressure": "memory_pressure",
    "network-pressure": "network_pressure",
    "network-watch": "network_pressure",
    "sla-risk": "sla_risk",
    "migration-pressure": "migration_churn",
    "migration-watch": "migration_churn",
    "consolidation-pressure": "migration_churn",
    "capacity-risk": "capacity_risk",
}

_EVENT_ORDER = [
    "cpu_pressure",
    "memory_pressure",
    "network_pressure",
    "sla_risk",
    "migration_churn",
    "capacity_risk",
    "critic_revise",
    "fallback",
    "reject",
    "select_success",
    "memory_low_confidence",
]


def extract_monitor_events(
    aiops_insight_raw: Any | None = None,
    recent_decision_raw: Any | None = None,
) -> list[str]:
    """Map AIOps and scheduler metadata into the monitor event vocabulary."""
    insight = _coerce_dict(aiops_insight_raw)
    decision = _coerce_dict(recent_decision_raw)
    events: list[str] = []

    tags = insight.get("risk_tags") or decision.get("aiops_risk_tags") or decision.get("global_risk_tags") or []
    if isinstance(tags, (list, tuple, set)):
        for tag in tags:
            event = _TAG_TO_EVENT.get(str(tag))
            if event:
                events.append(event)

    if decision.get("aiops_critic_triggered") is True or _safe_int(decision.get("aiops_critic_revisions")) > 0:
        events.append("critic_revise")

    action = str(decision.get("action") or "")
    if action == "fallback":
        events.append("fallback")
    elif action == "reject":
        events.append("reject")
    elif action == "select":
        events.append("select_success")

    if decision.get("memory_used") is True:
        try:
            confidence = float(decision.get("memory_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.35:
            events.append("memory_low_confidence")

    return _unique_in_order(events)


@dataclass
class CalibratedRiskMonitor:
    persistent_alert_threshold: int = 3
    repeated_pressure_window: int = 3
    state: MonitorState = MonitorState.NORMAL
    history: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        aiops_insight_raw: Any | None = None,
        recent_decision_raw: Any | None = None,
        *,
        tick: int | None = None,
    ) -> dict[str, Any]:
        """Advance the calibrated FSM and return shadow-only monitor state."""
        t0 = time.perf_counter()
        events = extract_monitor_events(aiops_insight_raw, recent_decision_raw)
        previous = self.state
        insight = _coerce_dict(aiops_insight_raw)
        next_state, reason = self._transition(previous, events, insight)
        self.state = next_state

        snapshot = {
            "tick": tick,
            "risk_state_id": int(next_state),
            "risk_state_label": STATE_LABELS[next_state],
            "fallback_probability": STATE_PROBABILITIES[next_state][0],
            "escalation_probability": STATE_PROBABILITIES[next_state][1],
            "state_confidence": self._confidence(next_state, events, insight),
            "events": events,
            "transition_reason": reason,
            "previous_state_label": STATE_LABELS[previous],
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "shadow_only": True,
        }
        self.history.append(snapshot)
        return dict(snapshot)

    def _transition(
        self,
        previous: MonitorState,
        events: list[str],
        insight: dict[str, Any],
    ) -> tuple[MonitorState, str]:
        if self._has_persistent_alert(insight):
            target = MonitorState.CRITICAL
            return target, "persistent active alert reached critical threshold"

        if self._has_protection_event(events):
            if previous >= MonitorState.RISK:
                return MonitorState.PROTECTED, "fallback/reject/revise moved monitor into protected mode"
            return MonitorState.RISK, "protection event raised monitor from low-risk state"

        if self._has_high_risk_event(events):
            if self._has_multi_signal_risk(events, insight):
                return max(previous, MonitorState.RISK), "multiple or high-score risk signals raised risk state"
            if previous >= MonitorState.WATCH:
                return max(previous, MonitorState.RISK), "SLA/network/capacity event raised risk state"
            return MonitorState.WATCH, "single high-risk event started watch state"

        if self._has_repeated_pressure(events):
            return max(previous, MonitorState.RISK), "repeated pressure events raised risk state"

        if self._has_mild_pressure(events):
            return max(previous, MonitorState.WATCH), "mild pressure started watch state"

        if not events:
            if previous > MonitorState.NORMAL:
                return MonitorState(previous - 1), "no risk events; stepwise recovery"
            return MonitorState.NORMAL, "no risk events"

        return previous, "events did not change monitor state"

    def _has_persistent_alert(self, insight: dict[str, Any]) -> bool:
        alerts = insight.get("active_alerts") or []
        if not isinstance(alerts, (list, tuple)):
            return False
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            try:
                occurrence_count = int(alert.get("occurrence_count") or 0)
            except (TypeError, ValueError):
                occurrence_count = 0
            if occurrence_count >= self.persistent_alert_threshold:
                return True
        return False

    def _has_repeated_pressure(self, events: list[str]) -> bool:
        if not self.history:
            return False
        pressure = {"cpu_pressure", "memory_pressure", "network_pressure", "migration_churn"}
        current = pressure.intersection(events)
        if not current:
            return False
        recent = self.history[-self.repeated_pressure_window + 1 :]
        for item in recent:
            if current.intersection(item.get("events") or []):
                return True
        return False

    @staticmethod
    def _has_high_risk_event(events: list[str]) -> bool:
        return bool({"network_pressure", "sla_risk", "capacity_risk"} & set(events))

    @staticmethod
    def _has_multi_signal_risk(events: list[str], insight: dict[str, Any]) -> bool:
        high_risk_events = {"network_pressure", "sla_risk", "capacity_risk"} & set(events)
        if len(high_risk_events) >= 2:
            return True
        try:
            risk_score = float(insight.get("risk_score") or 0.0)
        except (TypeError, ValueError):
            risk_score = 0.0
        return risk_score >= 0.5

    @staticmethod
    def _has_mild_pressure(events: list[str]) -> bool:
        return bool({"cpu_pressure", "memory_pressure", "migration_churn"} & set(events))

    @staticmethod
    def _has_protection_event(events: list[str]) -> bool:
        return bool({"fallback", "reject", "critic_revise"} & set(events))

    @staticmethod
    def _confidence(state: MonitorState, events: list[str], insight: dict[str, Any]) -> float:
        if state == MonitorState.NORMAL and not events:
            return 1.0
        try:
            score = float(insight.get("risk_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        base = 0.55 + 0.08 * int(state)
        if events:
            base += min(0.2, len(events) * 0.04)
        return round(max(0.0, min(1.0, base + min(score, 1.0) * 0.1)), 3)


def monitor_state_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    decision = record.get("decision") or {}
    if isinstance(decision, dict):
        value = decision.get("monitor_shadow") or decision.get("monitor")
        if isinstance(value, dict):
            return dict(value)
    value = record.get("monitor_shadow") or record.get("monitor")
    return dict(value) if isinstance(value, dict) else None


def _coerce_dict(value: Any | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in value
    ):
        return {str(k): v for k, v in value}
    return {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _unique_in_order(events: list[str]) -> list[str]:
    ordered = []
    seen = set()
    order = {name: index for index, name in enumerate(_EVENT_ORDER)}
    for event in sorted(events, key=lambda name: order.get(name, len(order))):
        if event not in seen:
            ordered.append(event)
            seen.add(event)
    return ordered
