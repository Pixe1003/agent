from agent_phase1.schemas import SchedulingDecision
from agent_phase3.memory import Episode, EpisodicMemory, WorkingMemory, summarize_context
from agent_phase3 import agent_usage_stats, agent_usage_summary, init_agent, last_decision_dict, schedule_service


def test_working_memory_keeps_recent_successful_decisions():
    memory = WorkingMemory(max_items=2)
    memory.add(SchedulingDecision(action="select", server_id=1, reasoning="first good placement", latency_ms=1, tool_call_succeeded=True))
    memory.add(SchedulingDecision(action="fallback", reasoning="bad placement", latency_ms=1, tool_call_succeeded=False))
    memory.add(SchedulingDecision(action="select", server_id=2, reasoning="second good placement", latency_ms=1, tool_call_succeeded=True))
    memory.add(SchedulingDecision(action="select", server_id=3, reasoning="third good placement", latency_ms=1, tool_call_succeeded=True))

    rendered = memory.render()
    assert "server 2" in rendered
    assert "server 3" in rendered
    assert "server 1" not in rendered
    assert "fallback" not in rendered


def test_episodic_memory_retrieves_by_text_and_features(tmp_path):
    store = EpisodicMemory(path=tmp_path / "episodes.jsonl")
    store.add(
        Episode(
            episode_id="cpu",
            run_id="run",
            tick=1,
            state_summary_text="CPU pressure with moderate memory",
            state_features=[0.9, 0.4, 0.2],
            service_request={"cpu_pct": 30},
            action_server_id=2,
            reasoning_trace="picked CPU headroom",
            reward=0.8,
        )
    )
    store.add(
        Episode(
            episode_id="net",
            run_id="run",
            tick=2,
            state_summary_text="Network pressure",
            state_features=[0.1, 0.2, 0.9],
            service_request={"net_pct": 30},
            action_server_id=3,
            reasoning_trace="picked network headroom",
            reward=0.7,
        )
    )

    matches = store.retrieve("CPU pressure", [0.85, 0.45, 0.2], top_k=1)
    assert matches[0].episode_id == "cpu"


def test_phase3_scheduler_records_memory_context():
    init_agent(model_name="heuristic")
    sid = schedule_service([[0, 90.0, 90.0, 90.0]], [10.0, 10.0, 10.0])

    assert sid == 0
    assert "memory_context" in last_decision_dict()


def test_phase3_passes_retrieved_episodes_into_phase2_metadata(tmp_path):
    memory_path = tmp_path / "episodes.jsonl"
    store = EpisodicMemory(path=memory_path)
    store.add(
        Episode(
            episode_id="similar-balanced",
            run_id="seed",
            tick=1,
            state_summary_text=(
                "Cluster has 1 active servers. Mean free resources are CPU 90.0%, "
                "RAM 90.0%, NET 90.0%. Incoming service needs CPU 10.0%, RAM 10.0%, NET 10.0%."
            ),
            state_features=[0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
            service_request={"cpu_pct": 10.0, "ram_pct": 10.0, "net_pct": 10.0},
            action_server_id=0,
            reasoning_trace="A matching request selected server 0.",
            reward=1.0,
        )
    )

    init_agent(model_name="heuristic", backend="hybrid", enable_tracing=False, memory_path=memory_path)
    sid = schedule_service([[0, 90.0, 90.0, 90.0]], [10.0, 10.0, 10.0])

    decision = last_decision_dict()
    assert sid == 0
    assert decision["memory_used"] is True
    assert decision["retrieved_episode_count"] == 1
    assert decision["memory_context"]["episodic"][0]["episode_id"] == "similar-balanced"


def test_phase3_agent_usage_summary_reports_participation_metrics(tmp_path):
    memory_path = tmp_path / "episodes.jsonl"
    store = EpisodicMemory(path=memory_path)
    store.add(
        Episode(
            episode_id="similar-balanced",
            run_id="seed",
            tick=1,
            state_summary_text=(
                "Cluster has 1 active servers. Mean free resources are CPU 90.0%, "
                "RAM 90.0%, NET 90.0%. Incoming service needs CPU 10.0%, RAM 10.0%, NET 10.0%."
            ),
            state_features=[0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
            service_request={"cpu_pct": 10.0, "ram_pct": 10.0, "net_pct": 10.0},
            action_server_id=0,
            reasoning_trace="A matching request selected server 0.",
            reward=1.0,
        )
    )

    init_agent(model_name="heuristic", backend="hybrid", enable_tracing=False, memory_path=memory_path)
    schedule_service([[0, 90.0, 90.0, 90.0]], [10.0, 10.0, 10.0])

    stats = agent_usage_stats()
    summary = agent_usage_summary()
    assert stats["total_decisions"] == 1
    assert stats["memory_used_decisions"] == 1
    assert "phase3 total=1" in summary
    assert "memory=1" in summary
    assert "avg_retrieved=1.00" in summary
    assert "agent_sync=0" in summary
    assert "avg_latency=" in summary


def test_summarize_context_uses_natural_language():
    summary, features = summarize_context([[0, 80.0, 70.0, 20.0]], [10.0, 20.0, 30.0])

    assert "Cluster has" in summary
    assert "Incoming service" in summary
    assert len(features) == 6
