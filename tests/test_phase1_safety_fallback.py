from types import SimpleNamespace

from agent_phase1 import scheduler


class FreeTextLLM:
    def invoke(self, messages):
        return SimpleNamespace(tool_calls=[], content="")


def test_phase1_uses_safe_deterministic_choice_when_model_returns_no_tool_call():
    scheduler._LLM = FreeTextLLM()

    sid = scheduler.schedule_service(
        [
            [0, 20.0, 30.0, 80.0],
            [1, 70.0, 80.0, 75.0],
            [2, 95.0, 10.0, 50.0],
        ],
        [25.0, 20.0, 10.0],
    )

    decision = scheduler.last_decision_dict()
    assert sid == 1
    assert decision["action"] == "select"
    assert decision["tool_call_succeeded"] is False
    assert "deterministic safety fallback" in decision["reasoning"]


def test_phase1_safety_fallback_rejects_when_no_server_fits():
    scheduler._LLM = FreeTextLLM()

    sid = scheduler.schedule_service([[0, 10.0, 8.0, 12.0]], [80.0, 70.0, 60.0])

    decision = scheduler.last_decision_dict()
    assert sid == -2
    assert decision["action"] == "reject"
    assert decision["tool_call_succeeded"] is False
