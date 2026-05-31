import json

from agent_aiops import init_agent as init_aiops, observe_ops_state
from agent_monitor.evaluation import evaluate_monitor_records
from benchmark.runner import run_benchmark
from dashboard.export_aiops_stream import build_aiops_stream
from multi_agent import init_agent as init_scheduler, last_decision_dict, schedule_service


def test_monitor_shadow_does_not_change_scheduler_decision():
    init_scheduler(model_name="heuristic", enable_tracing=False)
    init_aiops(model_name="heuristic", backend="rule", enable_tracing=False, recommendation_cooldown=0)
    insight = observe_ops_state({"active_net_util": 0.95, "net_sla_violations": 1})

    sid_without_shadow = schedule_service(
        [[0, 20.0, 20.0, 25.0], [1, 70.0, 70.0, 70.0]],
        [10.0, 10.0, 20.0],
        None,
        None,
        insight,
    )
    decision_without_shadow = last_decision_dict()

    sid_with_shadow = schedule_service(
        [[0, 20.0, 20.0, 25.0], [1, 70.0, 70.0, 70.0]],
        [10.0, 10.0, 20.0],
        None,
        None,
        {**insight, "monitor_shadow": {"risk_state_label": "Critical", "fallback_probability": 0.99}},
    )
    decision_with_shadow = last_decision_dict()

    assert sid_with_shadow == sid_without_shadow
    assert decision_with_shadow["action"] == decision_without_shadow["action"]
    assert decision_with_shadow.get("server_id") == decision_without_shadow.get("server_id")


def test_benchmark_reports_monitor_shadow_metrics_without_new_algorithm(tmp_path):
    rows = run_benchmark(
        seeds=[1],
        distributions=["mixed-burst"],
        algorithms=["AI-phase2-aiops"],
        output_path=tmp_path / "metrics.csv",
        requests_per_scenario=8,
    )

    row = rows[0]
    assert row["algorithm"] == "AI-phase2-aiops"
    assert "monitor_shadow_count" in row
    assert "monitor_high_risk_rate" in row
    assert "monitor_avg_latency_ms" in row
    assert row["monitor_shadow_count"] == row["requests"]


def test_dashboard_stream_carries_monitor_state_but_accepts_old_aiops_trace_shape(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_record = {
        "phase": "aiops",
        "decision": {
            "tick": 12,
            "risk_level": "high",
            "risk_score": 0.7,
            "risk_tags": ["network-pressure", "sla-risk"],
            "active_alerts": [{"tag": "network-pressure", "occurrence_count": 2}],
            "evidence": {
                "metrics": {
                    "active_cpu_util": 0.6,
                    "active_mem_util": 0.55,
                    "active_net_util": 0.93,
                }
            },
        },
    }
    (trace_dir / "aiops-run.jsonl").write_text(json.dumps(trace_record) + "\n", encoding="utf-8")

    stream = build_aiops_stream(trace_dir=trace_dir, latest_only=True)

    event = stream["events"][0]
    assert "monitor" in event
    assert event["monitor"]["risk_state_label"] in {"Risk", "Protected", "Critical"}
    assert "Monitor shadow state" in " ".join(event["events"])


def test_offline_evaluation_reports_risk_quality_and_structure_metrics():
    records = [
        {
            "phase": "aiops",
            "decision": {
                "tick": 1,
                "risk_score": 0.2,
                "risk_tags": ["network-watch"],
                "active_alerts": [],
            },
        },
        {
            "phase": "aiops",
            "decision": {
                "tick": 2,
                "risk_score": 0.35,
                "risk_tags": ["network-pressure"],
                "active_alerts": [],
            },
        },
        {
            "phase": "multi_agent",
            "decision": {
                "tick": 3,
                "action": "fallback",
                "aiops_critic_triggered": True,
                "aiops_risk_tags": ["network-pressure", "sla-risk"],
            },
        },
    ]

    report = evaluate_monitor_records(records, prediction_window=2)

    assert "calibrated_fsm" in report["models"]
    assert "threshold_only" in report["models"]
    assert "learned_static_fsm" in report["models"]
    assert "learned_dynamic_fsm" in report["models"]
    assert report["models"]["calibrated_fsm"]["missed_risk_rate"] <= report["models"]["threshold_only"]["missed_risk_rate"]
    assert report["structure"]["occupied_state_count"] >= 2
    assert report["structure"]["bounded_walk_similarity"] > 0
