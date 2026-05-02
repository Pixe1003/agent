import json
import os
from pathlib import Path

from dashboard.export_aiops_stream import build_aiops_stream, export_aiops_stream


def test_export_aiops_stream_converts_aiops_trace_to_dashboard_json(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_record = {
        "phase": "aiops",
        "decision": {
            "tick": 12,
            "risk_level": "critical",
            "risk_score": 0.95,
            "risk_tags": ["network-pressure", "sla-risk"],
            "active_alerts": [{"tag": "network-pressure", "occurrence_count": 2}],
            "root_cause_summary": "Network pressure is the dominant SLA risk signal.",
            "recommendations": [
                {
                    "action": "enable-network-headroom-protection",
                    "reason": "Network utilization is high.",
                    "expected_effect": "Preserve network headroom.",
                }
            ],
            "guardrails": {"do_not_auto_apply": True, "cooldown_hint": "wait"},
            "evidence": {
                "metrics": {
                    "active_cpu_util": 0.6,
                    "active_mem_util": 0.55,
                    "active_net_util": 0.93,
                },
                "server_snapshots": [
                    {"id": 0, "status": "overload", "cpu": 80.0, "mem": 70.0, "net": 92.0}
                ],
            },
        },
    }
    (trace_dir / "run.jsonl").write_text(json.dumps(trace_record) + "\n", encoding="utf-8")
    output = tmp_path / "aiops-stream.json"

    count = export_aiops_stream(trace_dir=trace_dir, output_path=output, algorithm="AI-phase2")

    data = json.loads(output.read_text(encoding="utf-8"))
    assert count == 1
    assert data["algorithm"] == "AI-phase2"
    assert data["events"][0]["tick"] == 12
    assert data["events"][0]["aiops"]["risk_level"] == "critical"
    assert data["events"][0]["global_state"]["active_net_util"] == 0.93
    assert data["events"][0]["servers"][0]["status"] == "overload"
    assert "network-pressure" in data["events"][0]["events"][0]


def test_export_aiops_stream_ignores_non_aiops_trace_rows(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "run.jsonl").write_text(
        json.dumps({"phase": "phase2", "decision": {"risk_level": "critical"}}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "aiops-stream.json"
    count = export_aiops_stream(trace_dir=trace_dir, output_path=output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert count == 0
    assert data["events"] == []


def _aiops_record(tick: int, risk_level: str = "low", algorithm: str | None = None) -> dict:
    metrics = {
        "active_cpu_util": 0.2,
        "active_mem_util": 0.3,
        "active_net_util": 0.4,
    }
    if algorithm is not None:
        metrics["service_placement_algorithm"] = algorithm
    return {
        "phase": "aiops",
        "decision": {
            "tick": tick,
            "risk_level": risk_level,
            "risk_score": 0.1,
            "risk_tags": [],
            "active_alerts": [],
            "evidence": {
                "metrics": metrics
            },
        },
    }


def test_build_aiops_stream_uses_latest_aiops_trace_file_only(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    old_trace = trace_dir / "aiops-old.jsonl"
    latest_trace = trace_dir / "aiops-latest.jsonl"
    old_trace.write_text(json.dumps(_aiops_record(10, "critical")) + "\n", encoding="utf-8")
    latest_trace.write_text(json.dumps(_aiops_record(20, "medium")) + "\n", encoding="utf-8")
    os.utime(old_trace, (1, 1))
    os.utime(latest_trace, (2, 2))

    stream = build_aiops_stream(trace_dir=trace_dir, latest_only=True)

    assert stream["source"] == str(trace_dir)
    assert stream["source_file"] == str(latest_trace)
    assert stream["event_count"] == 1
    assert [event["tick"] for event in stream["events"]] == [20]
    assert stream["events"][0]["aiops"]["risk_level"] == "medium"


def test_build_aiops_stream_limit_keeps_recent_events_from_latest_trace(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    latest_trace = trace_dir / "aiops-latest.jsonl"
    latest_trace.write_text(
        "\n".join(json.dumps(_aiops_record(tick)) for tick in (1, 2, 3)) + "\n",
        encoding="utf-8",
    )

    stream = build_aiops_stream(trace_dir=trace_dir, latest_only=True, limit=2)

    assert stream["event_count"] == 2
    assert [event["tick"] for event in stream["events"]] == [2, 3]


def test_build_aiops_stream_without_traces_returns_empty_stream(tmp_path):
    stream = build_aiops_stream(trace_dir=tmp_path / "missing")

    assert stream["event_count"] == 0
    assert stream["source_file"] is None
    assert stream["events"] == []
    assert "generated_at" in stream


def test_build_aiops_stream_skips_malformed_trace_lines(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    latest_trace = trace_dir / "aiops-latest.jsonl"
    latest_trace.write_text(
        json.dumps(_aiops_record(7)) + "\n" + '{"phase": "aiops",',
        encoding="utf-8",
    )

    stream = build_aiops_stream(trace_dir=trace_dir, latest_only=True)

    assert stream["event_count"] == 1
    assert stream["events"][0]["tick"] == 7


def test_build_aiops_stream_infers_algorithm_from_latest_trace_metrics(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    latest_trace = trace_dir / "aiops-latest.jsonl"
    latest_trace.write_text(json.dumps(_aiops_record(9, algorithm="AI-phase3")) + "\n", encoding="utf-8")

    stream = build_aiops_stream(trace_dir=trace_dir, latest_only=True, algorithm="auto")

    assert stream["algorithm"] == "AI-phase3"
    assert stream["events"][0]["global_state"]["service_placement_algorithm"] == "AI-phase3"


def test_export_aiops_stream_defaults_to_trace_algorithm(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "aiops-phase3.jsonl").write_text(
        json.dumps(_aiops_record(11, algorithm="AI-phase3")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "stream.json"

    count = export_aiops_stream(trace_dir=trace_dir, output_path=output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert count == 1
    assert data["algorithm"] == "AI-phase3"
