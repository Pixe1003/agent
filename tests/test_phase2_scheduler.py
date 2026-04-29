from agent_phase2 import (
    init_agent,
    last_decision_dict,
    last_decision_summary,
    schedule_service,
)
from agent_phase2.graph import render_mermaid


def test_phase2_selects_valid_balanced_candidate():
    init_agent(model_name="heuristic")
    sid = schedule_service(
        [
            [0, 20.0, 30.0, 80.0],
            [1, 70.0, 80.0, 75.0],
            [2, 95.0, 10.0, 50.0],
        ],
        [25.0, 20.0, 10.0],
    )

    decision = last_decision_dict()
    assert sid == 1
    assert decision["action"] == "select"
    assert decision["server_id"] == 1
    assert "approved" in last_decision_summary().lower()


def test_phase2_rejects_when_no_server_fits():
    init_agent(model_name="heuristic")
    sid = schedule_service([[0, 10.0, 8.0, 12.0], [1, 9.0, 9.0, 9.0]], [80.0, 70.0, 60.0])

    assert sid == -2
    assert last_decision_dict()["action"] == "reject"


def test_phase2_fallbacks_on_malformed_input():
    init_agent(model_name="heuristic")
    sid = schedule_service([[0, "bad", 50.0, 50.0]], [10.0, 10.0, 10.0])

    assert sid == -1
    assert last_decision_dict()["action"] == "fallback"


def test_phase2_mermaid_graph_mentions_all_agents():
    graph = render_mermaid()

    assert "Planner" in graph
    assert "Scheduler" in graph
    assert "Critic" in graph

