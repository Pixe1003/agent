import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

from dashboard.live_server import create_handler


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


def test_live_server_serves_limited_latest_aiops_stream(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    old_trace = trace_dir / "aiops-old.jsonl"
    latest_trace = trace_dir / "aiops-latest.jsonl"
    old_trace.write_text(json.dumps(_aiops_record(10, "critical")) + "\n", encoding="utf-8")
    latest_trace.write_text(
        "\n".join(json.dumps(_aiops_record(tick, "medium")) for tick in (20, 21)) + "\n",
        encoding="utf-8",
    )
    os.utime(old_trace, (1, 1))
    os.utime(latest_trace, (2, 2))

    handler = create_handler(trace_dir=trace_dir, dashboard_dir=Path("dashboard"), limit=500)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/api/aiops-stream.json?limit=1", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert data["source_file"] == str(latest_trace)
    assert data["event_count"] == 1
    assert [event["tick"] for event in data["events"]] == [21]


def test_live_server_infers_algorithm_from_latest_trace_without_cli_override(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    latest_trace = trace_dir / "aiops-latest.jsonl"
    latest_trace.write_text(json.dumps(_aiops_record(30, "medium", algorithm="AI-phase3")) + "\n", encoding="utf-8")

    handler = create_handler(trace_dir=trace_dir, dashboard_dir=Path("dashboard"), limit=500)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/api/aiops-stream.json?limit=1", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert data["algorithm"] == "AI-phase3"
