import json

from agent_aiops import (
    aiops_stats,
    aiops_stats_summary,
    analyze_ops_state,
    current_alerts,
    init_agent,
    last_insight_dict,
    last_insight_summary,
    observe_ops_state,
)


def test_aiops_low_risk_returns_quiet_insight():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        {
            "active_cpu_util": 0.35,
            "active_mem_util": 0.4,
            "active_net_util": 0.3,
            "current_auto_migrations": 0,
            "current_consolidation_migrations": 0,
            "rescheduled_services": 0,
            "ops_sla_violations": 0,
            "mem_sla_violations": 0,
            "net_sla_violations": 0,
        }
    )

    assert insight["risk_level"] == "low"
    assert insight["risk_score"] == 0.0
    assert insight["risk_tags"] == []
    assert insight["recommendations"] == []
    assert insight["guardrails"]["do_not_auto_apply"] is True
    assert "low" in last_insight_summary()


def test_aiops_network_sla_risk_recommends_network_protection():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        {
            "active_cpu_util": 0.45,
            "active_mem_util": 0.5,
            "active_net_util": 0.94,
            "current_auto_migrations": 1,
            "net_sla_violations": 2,
        },
        scheduler_stats_raw={"agent_escalation_needed": 4, "hybrid_decisions": 10},
    )

    assert insight["risk_level"] in {"high", "critical"}
    assert "network-pressure" in insight["risk_tags"]
    assert "sla-risk" in insight["risk_tags"]
    assert any("network" in item["action"] for item in insight["recommendations"])
    assert all(item["requires_human_approval"] is True for item in insight["recommendations"])
    assert insight["evidence"]["scheduler_stats"]["agent_escalation_needed"] == 4


def test_aiops_migration_pressure_recommends_cooldown():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        {
            "active_cpu_util": 0.5,
            "active_mem_util": 0.5,
            "active_net_util": 0.55,
            "current_auto_migrations": 8,
            "current_consolidation_migrations": 2,
            "rescheduled_services": 0,
        }
    )

    assert "migration-pressure" in insight["risk_tags"]
    assert any("cooldown" in item["action"] for item in insight["recommendations"])
    assert "migration" in insight["root_cause_summary"].lower()


def test_aiops_memory_context_is_evidence_not_autonomous_action():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        {"active_net_util": 0.91, "net_sla_violations": 1},
        memory_context_raw={
            "episodic": [
                {
                    "episode_id": "net-hotspot-1",
                    "reasoning_trace": "Similar network hotspot improved after network headroom policy.",
                    "reward": 0.8,
                }
            ]
        },
    )

    assert insight["evidence"]["memory"]["retrieved_episode_count"] == 1
    assert insight["evidence"]["memory"]["episodes"][0]["episode_id"] == "net-hotspot-1"
    assert insight["guardrails"]["do_not_auto_apply"] is True
    assert all(item["requires_human_approval"] is True for item in insight["recommendations"])


def test_aiops_malformed_input_returns_invalid_insight_without_throwing():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state({"active_net_util": "bad"})

    assert insight["status"] == "invalid"
    assert insight["risk_level"] == "low"
    assert "input-validation-error" in insight["risk_tags"]
    assert insight["recommendations"] == []
    assert "validation" in insight["root_cause_summary"].lower()


def test_aiops_tracing_writes_aiops_jsonl(tmp_path):
    init_agent(trace_dir=tmp_path, run_id="aiops-test")

    insight = analyze_ops_state({"active_net_util": 0.92, "net_sla_violations": 1})

    rows = [json.loads(line) for line in (tmp_path / "aiops-test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["phase"] == "aiops"
    assert rows[0]["decision"]["risk_level"] == insight["risk_level"]
    assert rows[0]["decision"]["guardrails"]["do_not_auto_apply"] is True
    assert rows[0]["decision"]["evidence"]["metrics"]["active_net_util"] == 0.92
    assert last_insight_dict()["risk_level"] == insight["risk_level"]


def test_aiops_accepts_netlogo_key_value_state_pairs():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        [
            ["active_cpu_util", 0.4],
            ["active_mem_util", 0.5],
            ["active_net_util", 0.93],
            ["current_auto_migrations", 6],
            ["net_sla_violations", 1],
        ]
    )

    assert insight["risk_level"] in {"high", "critical"}
    assert "network-pressure" in insight["risk_tags"]
    assert "migration-pressure" in insight["risk_tags"]


def test_aiops_records_netlogo_service_placement_algorithm_in_evidence():
    init_agent(enable_tracing=False)

    insight = analyze_ops_state(
        [
            ["service_placement_algorithm", "AI-phase3"],
            ["active_cpu_util", 0.4],
            ["active_mem_util", 0.5],
            ["active_net_util", 0.6],
        ]
    )

    assert insight["evidence"]["metrics"]["service_placement_algorithm"] == "AI-phase3"


def test_observe_ops_state_tracks_realtime_alerts_and_stats():
    init_agent(enable_tracing=False, window_size=3)

    observe_ops_state({"active_net_util": 0.2}, tick=1)
    observe_ops_state({"active_net_util": 0.91, "net_sla_violations": 1}, tick=2)
    insight = observe_ops_state({"active_net_util": 0.94, "net_sla_violations": 2}, tick=3)

    alerts = current_alerts()
    stats = aiops_stats()
    assert insight["mode"] == "realtime"
    assert stats["observed_events"] == 3
    assert stats["window_size"] == 3
    assert stats["active_alert_count"] >= 1
    assert any(alert["tag"] == "network-pressure" for alert in alerts)
    assert "observed=3" in aiops_stats_summary()


def test_observe_ops_state_applies_recommendation_cooldown():
    init_agent(enable_tracing=False, recommendation_cooldown=5)

    first = observe_ops_state({"active_net_util": 0.94, "net_sla_violations": 1}, tick=10)
    second = observe_ops_state({"active_net_util": 0.95, "net_sla_violations": 2}, tick=12)
    third = observe_ops_state({"active_net_util": 0.96, "net_sla_violations": 3}, tick=16)

    assert first["recommendations"]
    assert second["recommendations"] == []
    assert second["guardrails"]["recommendation_suppressed_by_cooldown"] is True
    assert third["recommendations"]


def test_aiops_records_server_snapshots_as_dashboard_evidence():
    init_agent(enable_tracing=False)

    insight = observe_ops_state(
        {"active_cpu_util": 0.6, "active_mem_util": 0.5, "active_net_util": 0.8},
        server_snapshots_raw=[
            [0, 20.0, 30.0, 15.0],
            [1, 70.0, 65.0, 80.0],
        ],
        tick=7,
    )

    servers = insight["evidence"]["server_snapshots"]
    assert servers == [
        {"id": 0, "status": "overload", "cpu": 80.0, "mem": 70.0, "net": 85.0},
        {"id": 1, "status": "normal", "cpu": 30.0, "mem": 35.0, "net": 20.0},
    ]
