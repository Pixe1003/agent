from agent_phase2 import (
    hybrid_stats,
    hybrid_stats_summary,
    init_agent,
    last_decision_dict,
    last_decision_summary,
    schedule_service,
)
from agent_phase2.graph import render_mermaid


class FakeStructuredBackend:
    def __init__(self, planner_output, scheduler_outputs):
        self.planner_output = planner_output
        self.scheduler_outputs = list(scheduler_outputs)
        self.propose_calls = []

    def plan(self, ctx):
        return self.planner_output

    def propose(self, ctx, *, strategy_tag, strategy_reasoning, excluded_server_ids, critic_feedback):
        self.propose_calls.append(
            {
                "strategy_tag": strategy_tag,
                "excluded_server_ids": set(excluded_server_ids),
                "critic_feedback": critic_feedback,
            }
        )
        output = self.scheduler_outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class MemoryAwareStructuredBackend(FakeStructuredBackend):
    def __init__(self, planner_output, scheduler_outputs):
        super().__init__(planner_output, scheduler_outputs)
        self.memory_contexts = []

    def propose(
        self,
        ctx,
        *,
        strategy_tag,
        strategy_reasoning,
        excluded_server_ids,
        critic_feedback,
        memory_context=None,
        **kwargs,
    ):
        self.memory_contexts.append(memory_context)
        return super().propose(
            ctx,
            strategy_tag=strategy_tag,
            strategy_reasoning=strategy_reasoning,
            excluded_server_ids=excluded_server_ids,
            critic_feedback=critic_feedback,
        )


class FailingStructuredBackend:
    def plan(self, ctx):
        raise AssertionError("auto backend should not call structured planner")

    def propose(self, ctx, *, strategy_tag, strategy_reasoning, excluded_server_ids, critic_feedback):
        raise AssertionError("auto backend should not call structured scheduler")


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


def test_phase2_fast_path_matches_netlogo_balanced_fit_distance():
    init_agent(model_name="heuristic", backend="heuristic", enable_tracing=False)
    sid = schedule_service(
        [
            [0, 15.0, 15.0, 55.0],
            [1, 15.0, 35.0, 60.0],
        ],
        [10.0, 10.0, 10.0],
    )

    decision = last_decision_dict()
    assert sid == 0
    assert decision["action"] == "select"
    assert decision["server_id"] == 0


def test_auto_backend_uses_fast_hybrid_even_with_llm_model_name():
    init_agent(
        model_name="qwen3:8b",
        backend="auto",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 80.0, 80.0, 80.0],
            [1, 70.0, 80.0, 75.0],
            [2, 95.0, 90.0, 90.0],
        ],
        [25.0, 20.0, 10.0],
    )

    decision = last_decision_dict()
    assert sid >= 0
    assert decision["backend"] == "hybrid"
    assert decision["model"] == "qwen3:8b"
    assert decision["fast_path_used"] is True
    assert decision["agent_escalation_needed"] is False
    assert decision["structured_output_succeeded"] is False


def test_hybrid_records_complex_escalation_without_sync_agent():
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="record",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
    )

    decision = last_decision_dict()
    assert sid == 0
    assert decision["backend"] == "hybrid"
    assert decision["fast_path_used"] is True
    assert decision["agent_escalation_needed"] is True
    assert decision["hybrid_agent_mode"] == "record"
    assert "few-valid-candidates" in decision["complexity_reasons"]


def test_hybrid_records_retrieved_memory_context_in_fast_path_stats():
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="record",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
        None,
        {
            "episodic": [
                {
                    "episode_id": "similar-high-pressure",
                    "action_server_id": 0,
                    "reasoning_trace": "A similar large balanced request selected server 0.",
                    "reward": 0.9,
                }
            ]
        },
    )

    decision = last_decision_dict()
    stats = hybrid_stats()
    assert sid == 0
    assert decision["memory_used"] is True
    assert decision["retrieved_episode_count"] == 1
    assert decision["memory_confidence"] == 0.9
    assert "retrieved-memory-context" in decision["complexity_reasons"]
    assert decision["agent_escalation_needed"] is True
    assert stats["memory_used_decisions"] == 1
    assert stats["retrieved_episode_count"] == 1
    assert stats["avg_latency_ms"] >= 0
    assert "memory=1" in hybrid_stats_summary()


def test_hybrid_stats_track_escalation_without_agent_call():
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="record",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    simple_sid = schedule_service(
        [
            [0, 80.0, 80.0, 80.0],
            [1, 70.0, 80.0, 75.0],
            [2, 95.0, 90.0, 90.0],
        ],
        [25.0, 20.0, 10.0],
    )
    complex_sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
    )

    stats = hybrid_stats()
    assert simple_sid >= 0
    assert complex_sid == 0
    assert stats["total_decisions"] == 2
    assert stats["hybrid_decisions"] == 2
    assert stats["fast_path_decisions"] == 2
    assert stats["agent_escalation_needed"] == 1
    assert stats["agent_sync_calls"] == 0
    assert stats["agent_call_decisions"] == 0
    assert stats["planner_policy_active"] == 2
    assert stats["escalation_ratio"] == 0.5
    assert stats["hybrid_agent_call_ratio"] == 0.0
    assert stats["avg_latency_ms"] >= 0
    assert stats["complexity_reason_counts"]["few-valid-candidates"] == 1
    assert "agent_sync=0 (0.0%)" in hybrid_stats_summary()


def test_hybrid_global_risk_records_escalation_without_sync_agent():
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="record",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 90.0, 90.0, 90.0],
            [1, 85.0, 85.0, 85.0],
            [2, 80.0, 80.0, 80.0],
            [3, 75.0, 75.0, 75.0],
        ],
        [10.0, 10.0, 10.0],
        {
            "active_net_util": 0.94,
            "current_auto_migrations": 8,
            "net_sla_violations": 1,
            "rescheduled_services": 4,
        },
    )

    decision = last_decision_dict()
    stats = hybrid_stats()
    assert sid >= 0
    assert decision["backend"] == "hybrid"
    assert decision["fast_path_used"] is True
    assert decision["agent_escalation_needed"] is True
    assert decision["global_risk_agent_triggered"] is True
    assert decision["global_risk_score"] >= 0.5
    assert decision["global_risk_level"] in {"high", "critical"}
    assert "network-pressure" in decision["global_risk_tags"]
    assert stats["global_risk_agent_triggers"] == 1


def test_hybrid_network_risk_fast_path_keeps_balanced_fit_primary():
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="record",
        structured_backend=FailingStructuredBackend(),
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 60.0, 60.0, 25.0],
            [1, 25.0, 25.0, 80.0],
        ],
        [20.0, 20.0, 20.0],
        {
            "active_net_util": 0.95,
            "current_auto_migrations": 4,
        },
    )

    decision = last_decision_dict()
    assert sid == 0
    assert decision["risk_aware_fast_path"] is True
    assert decision["risk_policy"]["resource_weights"]["net"] > decision["risk_policy"]["resource_weights"]["cpu"]
    assert "network-pressure" in decision["global_risk_tags"]


def test_hybrid_complex_case_can_sync_escalate_to_structured_agent():
    backend = FakeStructuredBackend(
        planner_output={
            "strategy_tag": "high-pressure",
            "strategy_reasoning": "Only one candidate can host this large request.",
        },
        scheduler_outputs=[
            {
                "action": "select",
                "server_id": 0,
                "reasoning": "Server 0 is the only server with sufficient resources.",
            }
        ],
    )
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="sync",
        structured_backend=backend,
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
    )

    decision = last_decision_dict()
    assert sid == 0
    assert decision["backend"] == "hybrid"
    assert decision["fast_path_used"] is False
    assert decision["agent_escalation_needed"] is True
    assert decision["structured_output_succeeded"] is True
    assert backend.propose_calls


def test_hybrid_sync_passes_retrieved_memory_context_to_structured_agent():
    backend = MemoryAwareStructuredBackend(
        planner_output={
            "strategy_tag": "high-pressure",
            "strategy_reasoning": "Only one candidate can host this large request.",
        },
        scheduler_outputs=[
            {
                "action": "select",
                "server_id": 0,
                "reasoning": "Server 0 matches the retrieved successful episode.",
            }
        ],
    )
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="sync",
        structured_backend=backend,
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
        None,
        {
            "episodic": [
                {
                    "episode_id": "similar-high-pressure",
                    "action_server_id": 0,
                    "reasoning_trace": "A similar large balanced request selected server 0.",
                    "reward": 0.9,
                }
            ]
        },
    )

    decision = last_decision_dict()
    assert sid == 0
    assert decision["fast_path_used"] is False
    assert decision["retrieved_episode_count"] == 1
    assert backend.memory_contexts[0]["retrieved_episode_count"] == 1
    assert backend.memory_contexts[0]["episodic"][0]["episode_id"] == "similar-high-pressure"


def test_hybrid_stats_count_sync_agent_calls():
    backend = FakeStructuredBackend(
        planner_output={
            "strategy_tag": "high-pressure",
            "strategy_reasoning": "Only one candidate can host this large request.",
        },
        scheduler_outputs=[
            {
                "action": "select",
                "server_id": 0,
                "reasoning": "Server 0 is the only server with sufficient resources.",
            }
        ],
    )
    init_agent(
        model_name="qwen3:8b",
        backend="hybrid",
        hybrid_agent_mode="sync",
        structured_backend=backend,
        enable_tracing=False,
    )

    sid = schedule_service(
        [
            [0, 42.0, 42.0, 42.0],
            [1, 60.0, 8.0, 60.0],
            [2, 5.0, 80.0, 80.0],
        ],
        [38.0, 38.0, 38.0],
    )

    stats = hybrid_stats()
    assert sid == 0
    assert stats["hybrid_decisions"] == 1
    assert stats["fast_path_decisions"] == 0
    assert stats["agent_escalation_needed"] == 1
    assert stats["agent_sync_calls"] == 1
    assert stats["agent_call_decisions"] == 1
    assert stats["hybrid_agent_call_ratio"] == 1.0


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


def test_structured_backend_selects_and_records_trace(tmp_path):
    backend = FakeStructuredBackend(
        planner_output={
            "strategy_tag": "balanced",
            "strategy_reasoning": "The request is balanced across resources.",
        },
        scheduler_outputs=[
            {
                "action": "select",
                "server_id": 0,
                "reasoning": "Server 0 has enough headroom on every resource.",
            }
        ],
    )
    init_agent(
        model_name="qwen3:8b",
        backend="structured",
        structured_backend=backend,
        trace_dir=tmp_path,
        run_id="structured-test",
    )

    sid = schedule_service([[0, 90.0, 90.0, 90.0], [1, 15.0, 15.0, 15.0]], [10.0, 10.0, 10.0])

    decision = last_decision_dict()
    assert sid == 0
    assert decision["backend"] == "structured"
    assert decision["strategy_tag"] == "balanced"
    assert decision["structured_output_succeeded"] is True
    assert decision["critic_verdict"] == "approve"
    rows = (tmp_path / "structured-test.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"name": "select_server"' in rows[0]


def test_structured_backend_revises_invalid_selection():
    backend = FakeStructuredBackend(
        planner_output={
            "strategy_tag": "balanced",
            "strategy_reasoning": "The request is balanced across resources.",
        },
        scheduler_outputs=[
            {
                "action": "select",
                "server_id": 2,
                "reasoning": "Try the low-headroom server first.",
            },
            {
                "action": "select",
                "server_id": 0,
                "reasoning": "Server 0 is the valid revised placement.",
            },
        ],
    )
    init_agent(model_name="qwen3:8b", backend="structured", structured_backend=backend, enable_tracing=False)

    sid = schedule_service([[0, 90.0, 90.0, 90.0], [2, 5.0, 5.0, 5.0]], [10.0, 10.0, 10.0])

    decision = last_decision_dict()
    assert sid == 0
    assert decision["critic_verdict"] == "approve"
    assert decision["revise_count"] == 1
    assert backend.propose_calls[1]["excluded_server_ids"] == {2}


def test_structured_backend_fallbacks_when_output_fails():
    backend = FakeStructuredBackend(
        planner_output={
            "strategy_tag": "balanced",
            "strategy_reasoning": "The request is balanced across resources.",
        },
        scheduler_outputs=[RuntimeError("structured parser failed")],
    )
    init_agent(model_name="qwen3:8b", backend="structured", structured_backend=backend, enable_tracing=False)

    sid = schedule_service([[0, 90.0, 90.0, 90.0]], [10.0, 10.0, 10.0])

    decision = last_decision_dict()
    assert sid == -1
    assert decision["action"] == "fallback"
    assert decision["structured_output_succeeded"] is False
    assert "structured parser failed" in decision["fallback_reason"]
