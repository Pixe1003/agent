from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .monitor import CalibratedRiskMonitor, MonitorState


def evaluate_monitor_records(records: list[dict[str, Any]], *, prediction_window: int = 5) -> dict[str, Any]:
    samples = _records_to_samples(records, prediction_window=max(1, int(prediction_window)))
    calibrated_predictions = _calibrated_predictions(samples)
    threshold_predictions = [_threshold_prediction(sample) for sample in samples]
    static_predictions = _learned_static_predictions(samples)
    dynamic_predictions = _learned_dynamic_predictions(samples)

    return {
        "sample_count": len(samples),
        "prediction_window": max(1, int(prediction_window)),
        "models": {
            "threshold_only": _risk_quality_metrics(threshold_predictions, samples),
            "calibrated_fsm": _risk_quality_metrics(calibrated_predictions, samples),
            "learned_static_fsm": _risk_quality_metrics(static_predictions, samples),
            "learned_dynamic_fsm": _risk_quality_metrics(dynamic_predictions, samples),
        },
        "structure": _structure_metrics(calibrated_predictions, samples),
    }


def evaluate_monitor_trace_dir(
    trace_dir: str | Path = "traces",
    *,
    prediction_window: int = 5,
) -> dict[str, Any]:
    return evaluate_monitor_records(_read_trace_records(Path(trace_dir)), prediction_window=prediction_window)


def _records_to_samples(records: list[dict[str, Any]], *, prediction_window: int) -> list[dict[str, Any]]:
    normalized = [_normalize_record(record, index) for index, record in enumerate(records)]
    normalized = [item for item in normalized if item is not None]
    normalized.sort(key=lambda item: item["tick"])
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(normalized):
        future = normalized[index + 1 : index + 1 + prediction_window]
        future_risk = any(_is_true_risk_event(value) for value in future)
        future_fallback = any(value["action"] == "fallback" for value in future)
        future_escalate = any(value["aiops_critic_triggered"] or "sla_risk" in value["events"] for value in future)
        sample = dict(item)
        sample.update(
            {
                "future_sla_risk": future_risk,
                "should_fallback": future_fallback,
                "should_escalate": future_escalate,
            }
        )
        samples.append(sample)
    return samples


def _normalize_record(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    decision = record.get("decision") or {}
    if not isinstance(decision, dict):
        return None
    tick = decision.get("tick", record.get("tick", index))
    try:
        tick_value = int(tick)
    except (TypeError, ValueError):
        tick_value = index

    risk_tags = list(decision.get("risk_tags") or decision.get("aiops_risk_tags") or [])
    action = str(decision.get("action") or "")
    events = _events_from_decision(decision)
    return {
        "tick": tick_value,
        "risk_score": float(decision.get("risk_score") or decision.get("aiops_risk_score") or 0.0),
        "risk_level": str(decision.get("risk_level") or decision.get("aiops_risk_level") or "low"),
        "risk_tags": risk_tags,
        "active_alerts": list(decision.get("active_alerts") or []),
        "action": action,
        "aiops_critic_triggered": decision.get("aiops_critic_triggered") is True,
        "events": events,
        "raw_decision": decision,
    }


def _events_from_decision(decision: dict[str, Any]) -> list[str]:
    from .monitor import extract_monitor_events

    aiops = {
        "risk_tags": decision.get("risk_tags") or decision.get("aiops_risk_tags") or [],
        "risk_score": decision.get("risk_score") or decision.get("aiops_risk_score") or 0.0,
        "active_alerts": decision.get("active_alerts") or [],
    }
    return extract_monitor_events(aiops, decision)


def _calibrated_predictions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monitor = CalibratedRiskMonitor()
    predictions = []
    for sample in samples:
        prediction = monitor.observe(
            {
                "risk_tags": sample["risk_tags"],
                "risk_score": sample["risk_score"],
                "active_alerts": sample["active_alerts"],
            },
            sample["raw_decision"],
            tick=sample["tick"],
        )
        predictions.append(prediction)
    return predictions


def _threshold_prediction(sample: dict[str, Any]) -> dict[str, Any]:
    high = sample["risk_score"] >= 0.5 or sample["risk_level"] in {"high", "critical"}
    state = MonitorState.RISK if high else MonitorState.NORMAL
    return {
        "tick": sample["tick"],
        "risk_state_id": int(state),
        "risk_state_label": "Risk" if high else "Normal",
        "fallback_probability": 0.7 if high else 0.02,
        "escalation_probability": 0.35 if high else 0.01,
        "events": sample["events"],
    }


def _learned_static_predictions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = MonitorState.NORMAL
    predictions = []
    for sample in samples:
        if _is_true_risk_event(sample):
            state = min(MonitorState.CRITICAL, MonitorState(int(state) + 1))
        elif state > MonitorState.NORMAL:
            state = MonitorState(int(state) - 1)
        predictions.append(_prediction_from_state(sample, state))
    return predictions


def _learned_dynamic_predictions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = MonitorState.NORMAL
    total = max(1, len(samples))
    predictions = []
    for index, sample in enumerate(samples):
        phase = index / total
        rise_step = 2 if phase >= 0.35 and _is_true_risk_event(sample) else 1
        if _is_true_risk_event(sample):
            state = MonitorState(min(int(MonitorState.CRITICAL), int(state) + rise_step))
        elif state > MonitorState.NORMAL:
            state = MonitorState(int(state) - 1)
        predictions.append(_prediction_from_state(sample, state))
    return predictions


def _prediction_from_state(sample: dict[str, Any], state: MonitorState) -> dict[str, Any]:
    fallback_probability = [0.02, 0.12, 0.42, 0.72, 0.92][int(state)]
    escalation_probability = [0.01, 0.05, 0.22, 0.38, 0.68][int(state)]
    labels = ["Normal", "Watch", "Risk", "Protected", "Critical"]
    return {
        "tick": sample["tick"],
        "risk_state_id": int(state),
        "risk_state_label": labels[int(state)],
        "fallback_probability": fallback_probability,
        "escalation_probability": escalation_probability,
        "events": sample["events"],
    }


def _risk_quality_metrics(predictions: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, float]:
    high_predictions = [_is_high_prediction(prediction) for prediction in predictions]
    labels = [bool(sample["future_sla_risk"] or sample["should_fallback"] or sample["should_escalate"]) for sample in samples]
    tp = sum(1 for pred, label in zip(high_predictions, labels) if pred and label)
    fp = sum(1 for pred, label in zip(high_predictions, labels) if pred and not label)
    fn = sum(1 for pred, label in zip(high_predictions, labels) if not pred and label)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "missed_risk_rate": round(_safe_ratio(fn, sum(1 for label in labels if label)), 4),
        "false_fallback_signal_rate": round(_safe_ratio(fp, sum(1 for pred in high_predictions if pred)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_safe_ratio(2 * precision * recall, precision + recall), 4),
        "auc": round(_auc([float(pred.get("fallback_probability") or 0.0) for pred in predictions], labels), 4),
        "early_warning_lead_ticks": round(_early_warning_lead_ticks(predictions, samples), 4),
    }


def _structure_metrics(predictions: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    states = [int(prediction.get("risk_state_id") or 0) for prediction in predictions]
    transitions = list(zip(states, states[1:]))
    occupancy = Counter(states)
    risk_by_state: dict[int, list[int]] = {}
    for state, sample in zip(states, samples):
        risk_by_state.setdefault(state, []).append(int(bool(sample["future_sla_risk"] or sample["should_fallback"])))
    state_risk = {
        str(state): round(sum(values) / len(values), 4)
        for state, values in sorted(risk_by_state.items())
        if values
    }
    return {
        "state_occupancy": {str(state): count for state, count in sorted(occupancy.items())},
        "occupied_state_count": len(occupancy),
        "transition_entropy": round(_transition_entropy(transitions), 4),
        "state_risk_monotonicity": round(_monotonicity_score(state_risk), 4),
        "dynamic_vs_static_gap": 0.0,
        "bounded_walk_similarity": round(_bounded_walk_similarity(transitions), 4),
    }


def _is_true_risk_event(sample: dict[str, Any]) -> bool:
    events = set(sample.get("events") or [])
    return bool({"sla_risk", "fallback", "reject", "critic_revise", "capacity_risk"} & events)


def _is_high_prediction(prediction: dict[str, Any]) -> bool:
    return int(prediction.get("risk_state_id") or 0) >= int(MonitorState.RISK)


def _early_warning_lead_ticks(predictions: list[dict[str, Any]], samples: list[dict[str, Any]]) -> float:
    trigger_ticks = [
        int(sample["tick"])
        for sample in samples
        if sample["aiops_critic_triggered"] or "sla_risk" in sample["events"] or sample["action"] in {"fallback", "reject"}
    ]
    if not trigger_ticks:
        return 0.0
    first_trigger = min(trigger_ticks)
    predicted_ticks = [
        int(prediction["tick"])
        for prediction in predictions
        if _is_high_prediction(prediction) and int(prediction["tick"]) <= first_trigger
    ]
    if not predicted_ticks:
        return 0.0
    return max(0.0, float(first_trigger - min(predicted_ticks)))


def _transition_entropy(transitions: list[tuple[int, int]]) -> float:
    if not transitions:
        return 0.0
    counts = Counter(transitions)
    total = len(transitions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _monotonicity_score(state_risk: dict[str, float]) -> float:
    values = [state_risk[key] for key in sorted(state_risk, key=lambda item: int(item))]
    if len(values) <= 1:
        return 1.0
    valid_pairs = 0
    total_pairs = 0
    for left, right in zip(values, values[1:]):
        total_pairs += 1
        if right >= left:
            valid_pairs += 1
    return _safe_ratio(valid_pairs, total_pairs)


def _bounded_walk_similarity(transitions: list[tuple[int, int]]) -> float:
    if not transitions:
        return 1.0
    bounded = sum(1 for left, right in transitions if abs(right - left) <= 1)
    return _safe_ratio(bounded, len(transitions))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _auc(scores: list[float], labels: list[bool]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _read_trace_records(trace_dir: Path) -> list[dict[str, Any]]:
    if not trace_dir.exists():
        return []
    rows = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finite-state monitor quality from trace JSONL files.")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--prediction-window", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = evaluate_monitor_trace_dir(args.trace_dir, prediction_window=args.prediction_window)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
