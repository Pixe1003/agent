from __future__ import annotations

import argparse
from collections import deque
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def export_aiops_stream(
    trace_dir: str | Path = "traces",
    output_path: str | Path = "dashboard/aiops-stream.json",
    algorithm: str = "auto",
) -> int:
    data = build_aiops_stream(trace_dir=trace_dir, algorithm=algorithm, latest_only=False, limit=None)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(data["event_count"])


def build_aiops_stream(
    trace_dir: str | Path = "traces",
    algorithm: str = "auto",
    *,
    latest_only: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    trace_path = Path(trace_dir)
    source_files = _aiops_trace_files(trace_path, latest_only=latest_only)
    rows = _read_aiops_trace_rows(source_files, limit=limit)
    events = [_to_dashboard_event(row) for row in rows]
    return {
        "algorithm": _resolve_algorithm(algorithm, rows),
        "source": str(trace_path),
        "source_file": str(source_files[0]) if len(source_files) == 1 else None,
        "event_count": len(events),
        "generated_at": datetime.now(UTC).isoformat(),
        "events": events,
    }


def _aiops_trace_files(trace_dir: Path, *, latest_only: bool) -> list[Path]:
    if not trace_dir.exists():
        return []
    paths = sorted(trace_dir.glob("aiops-*.jsonl") if latest_only else trace_dir.glob("*.jsonl"))
    if latest_only and paths:
        return [max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))]
    return paths


def _read_aiops_trace_rows(paths: list[Path], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] | deque[dict[str, Any]]
    rows = deque(maxlen=max(0, limit)) if limit is not None else []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = record.get("decision") or {}
            if record.get("phase") == "aiops" or decision.get("phase") == "aiops":
                rows.append(record)
    return list(rows)


def _to_dashboard_event(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("decision") or {}
    evidence = decision.get("evidence") or {}
    metrics = evidence.get("metrics") or {}
    tags = list(decision.get("risk_tags") or [])
    active_alerts = list(decision.get("active_alerts") or [])
    return {
        "tick": int(decision.get("tick") or record.get("tick") or len(tags)),
        "global_state": {
            "service_placement_algorithm": metrics.get("service_placement_algorithm", "unknown"),
            "active_cpu_util": _normalize_ratio(metrics.get("active_cpu_util")),
            "active_mem_util": _normalize_ratio(metrics.get("active_mem_util")),
            "active_net_util": _normalize_ratio(metrics.get("active_net_util")),
            "current_auto_migrations": int(metrics.get("current_auto_migrations") or 0),
            "net_sla_violations": float(metrics.get("net_sla_violations") or 0.0),
        },
        "aiops": {
            "risk_level": decision.get("risk_level", "low"),
            "risk_score": float(decision.get("risk_score") or 0.0),
            "risk_tags": tags,
            "active_alerts": active_alerts,
            "root_cause_summary": decision.get("root_cause_summary", "No AIOps summary recorded."),
            "recommendations": _recommendations(decision.get("recommendations") or []),
            "guardrails": _guardrails(decision.get("guardrails") or {}),
        },
        "servers": _servers(evidence, metrics),
        "events": _event_lines(decision, tags, active_alerts),
    }


def _resolve_algorithm(algorithm: str | None, rows: list[dict[str, Any]]) -> str:
    if algorithm and algorithm.lower() != "auto":
        return algorithm
    for record in reversed(rows):
        decision = record.get("decision") or {}
        evidence = decision.get("evidence") or {}
        metrics = evidence.get("metrics") or {}
        value = (
            decision.get("service_placement_algorithm")
            or decision.get("algorithm")
            or metrics.get("service_placement_algorithm")
            or metrics.get("algorithm")
        )
        if value:
            return str(value)
    return "AI-phase2"


def _recommendations(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "action": str(item.get("action", "recommendation")),
            "reason": str(item.get("reason", "")),
            "expected_effect": str(item.get("expected_effect", "")),
        }
        for item in raw
        if isinstance(item, dict)
    ]


def _guardrails(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        return [key for key, value in raw.items() if value is True or isinstance(value, str)]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _servers(evidence: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = evidence.get("server_snapshots") or []
    if snapshots:
        return [
            {
                "id": int(item.get("id", index)),
                "status": str(item.get("status", "normal")),
                "cpu": float(item.get("cpu") or 0.0),
                "mem": float(item.get("mem") or 0.0),
                "net": float(item.get("net") or 0.0),
            }
            for index, item in enumerate(snapshots)
            if isinstance(item, dict)
        ]
    return [
        {
            "id": 0,
            "status": _server_status(
                _normalize_percent(metrics.get("active_cpu_util")),
                _normalize_percent(metrics.get("active_mem_util")),
                _normalize_percent(metrics.get("active_net_util")),
            ),
            "cpu": _normalize_percent(metrics.get("active_cpu_util")),
            "mem": _normalize_percent(metrics.get("active_mem_util")),
            "net": _normalize_percent(metrics.get("active_net_util")),
        }
    ]


def _event_lines(decision: dict[str, Any], tags: list[str], active_alerts: list[dict[str, Any]]) -> list[str]:
    lines = []
    if tags:
        lines.append(f"AIOps tags active: {', '.join(tags)}.")
    if active_alerts:
        lines.append(f"Active alerts: {', '.join(str(alert.get('tag')) for alert in active_alerts)}.")
    if decision.get("recommendations"):
        lines.append("Recommendation emitted with approval guardrail.")
    if not lines:
        lines.append("No elevated AIOps risk signals detected.")
    return lines


def _normalize_ratio(value: Any) -> float:
    number = float(value or 0.0)
    return number / 100 if number > 1 else number


def _normalize_percent(value: Any) -> float:
    return round(_normalize_ratio(value) * 100, 3)


def _server_status(cpu: float, mem: float, net: float) -> str:
    peak = max(cpu, mem, net)
    if peak >= 85:
        return "overload"
    if peak >= 72:
        return "warning"
    return "normal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AIOps traces to dashboard stream JSON.")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--output", default="dashboard/aiops-stream.json")
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--latest-only", action="store_true", help="Only export the newest traces/aiops-*.jsonl run.")
    parser.add_argument("--limit", type=int, default=None, help="Keep only the most recent N AIOps events.")
    args = parser.parse_args()
    data = build_aiops_stream(
        trace_dir=args.trace_dir,
        algorithm=args.algorithm,
        latest_only=args.latest_only,
        limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    count = int(data["event_count"])
    print(f"exported {count} AIOps events to {args.output}")


if __name__ == "__main__":
    main()
