import json

from agent_common.tracing import TraceLogger


def test_trace_logger_writes_jsonl_records(tmp_path):
    logger = TraceLogger(trace_dir=tmp_path, run_id="run-1", phase="phase2", model="heuristic")

    logger.write(
        tick=7,
        decision={"action": "select", "server_id": 2},
        latency_ms=12.5,
        messages=[{"role": "user", "content": "state"}],
        tool_calls=[{"name": "select_server", "args": {"server_id": 2}}],
    )

    trace_file = tmp_path / "run-1.jsonl"
    rows = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["phase"] == "phase2"
    assert rows[0]["decision"]["server_id"] == 2
    assert rows[0]["tool_calls"][0]["name"] == "select_server"

