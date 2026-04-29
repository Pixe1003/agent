import json

from benchmark.runner import run_benchmark
from dataset.build_sft_dataset import build_sft_dataset


def test_benchmark_runner_outputs_metrics_for_algorithms(tmp_path):
    output = tmp_path / "metrics.csv"

    rows = run_benchmark(seeds=[1], distributions=["mixed"], algorithms=["first-fit", "balanced-fit", "AI-phase2"], output_path=output)

    assert output.exists()
    assert {row["algorithm"] for row in rows} == {"first-fit", "balanced-fit", "AI-phase2"}
    assert all("fallback_rate" in row for row in rows)
    assert all("avg_latency_ms" in row for row in rows)


def test_sft_dataset_builder_filters_successful_trace_rows(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "run.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "state"}],
                        "tool_calls": [{"name": "select_server", "args": {"server_id": 2}}],
                        "decision": {"action": "select", "server_id": 2, "reasoning": "good"},
                        "outcome": {"sla_violated": False, "energy": 12},
                    }
                ),
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "bad"}],
                        "tool_calls": [],
                        "decision": {"action": "fallback"},
                        "outcome": {"sla_violated": True, "energy": 99},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sft.jsonl"

    count = build_sft_dataset(trace_dir=trace_dir, output_path=output, max_samples=10)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert rows[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "select_server"
