from agent_monitor import CalibratedRiskMonitor, extract_monitor_events


def test_monitor_safe_default_outputs_normal_low_risk_state():
    monitor = CalibratedRiskMonitor()

    state = monitor.observe(None, None, tick=1)

    assert state["risk_state_id"] == 0
    assert state["risk_state_label"] == "Normal"
    assert state["fallback_probability"] == 0.02
    assert state["escalation_probability"] == 0.01
    assert state["state_confidence"] == 1.0
    assert state["events"] == []


def test_extract_monitor_events_maps_aiops_tags_and_decision_signals():
    insight = {
        "risk_tags": [
            "network-pressure",
            "sla-risk",
            "migration-watch",
            "capacity-risk",
        ],
        "active_alerts": [{"tag": "network-pressure", "occurrence_count": 2}],
    }
    decision = {
        "action": "fallback",
        "aiops_critic_triggered": True,
        "memory_used": True,
        "memory_confidence": 0.2,
    }

    events = extract_monitor_events(insight, decision)

    assert events == [
        "network_pressure",
        "sla_risk",
        "migration_churn",
        "capacity_risk",
        "critic_revise",
        "fallback",
        "memory_low_confidence",
    ]


def test_extract_monitor_events_tolerates_invalid_optional_fields():
    events = extract_monitor_events(
        {"risk_tags": "not-a-list"},
        {"aiops_critic_revisions": "bad", "memory_used": True, "memory_confidence": "bad"},
    )

    assert events == ["memory_low_confidence"]


def test_calibrated_fsm_escalates_pressure_to_risk_then_protected_then_critical():
    monitor = CalibratedRiskMonitor(persistent_alert_threshold=3)

    first = monitor.observe({"risk_tags": ["network-watch"], "risk_score": 0.2}, None, tick=1)
    second = monitor.observe({"risk_tags": ["network-pressure", "sla-risk"], "risk_score": 0.55}, None, tick=2)
    third = monitor.observe(
        {"risk_tags": ["network-pressure"], "risk_score": 0.7},
        {"action": "fallback"},
        tick=3,
    )
    fourth = monitor.observe(
        {
            "risk_tags": ["network-pressure"],
            "risk_score": 0.9,
            "active_alerts": [{"tag": "network-pressure", "occurrence_count": 3}],
        },
        None,
        tick=4,
    )

    assert first["risk_state_label"] == "Watch"
    assert second["risk_state_label"] == "Risk"
    assert third["risk_state_label"] == "Protected"
    assert fourth["risk_state_label"] == "Critical"
    assert fourth["fallback_probability"] > third["fallback_probability"]


def test_calibrated_fsm_recovers_one_state_at_a_time():
    monitor = CalibratedRiskMonitor(persistent_alert_threshold=2)
    monitor.observe(
        {
            "risk_tags": ["network-pressure", "sla-risk"],
            "active_alerts": [{"tag": "network-pressure", "occurrence_count": 2}],
        },
        {"action": "fallback"},
        tick=1,
    )

    first_recovery = monitor.observe({"risk_tags": [], "risk_score": 0.0}, None, tick=2)
    second_recovery = monitor.observe({"risk_tags": [], "risk_score": 0.0}, None, tick=3)
    third_recovery = monitor.observe({"risk_tags": [], "risk_score": 0.0}, None, tick=4)
    fourth_recovery = monitor.observe({"risk_tags": [], "risk_score": 0.0}, None, tick=5)

    assert first_recovery["risk_state_label"] == "Protected"
    assert second_recovery["risk_state_label"] == "Risk"
    assert third_recovery["risk_state_label"] == "Watch"
    assert fourth_recovery["risk_state_label"] == "Normal"
