from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NLOGO = ROOT / "2143512_Jiale Miao_2025_Supplementary.nlogo"


def test_netlogo_uses_phase1_agent_entrypoint_and_keeps_legacy_reporter():
    text = NLOGO.read_text(encoding="utf-8")

    assert "from agent_phase1 import init_agent, schedule_service, last_decision_summary" in text
    assert "from agent_phase2 import init_agent as init_agent_phase2" in text
    assert "from agent_phase3 import init_agent as init_agent_phase3" in text
    assert "init_agent(model_name='qwen3:8b', temperature=0.1)" in text
    assert "to-report find-AI-phase2-server [ the-server-set the-service ]" in text
    assert "to-report find-AI-phase3-server [ the-server-set the-service ]" in text
    assert "to-report find-AI-server [ the-server-set the-service ]" in text
    assert "to-report find-AI-server-legacy [ the-server-set the-service ]" in text
    assert "schedule_service(servers_raw, service_raw)" in text
    assert '"schedule_service_phase2"' in text
    assert '"schedule_service_phase3"' in text
    assert 'py:runresult (word python-scheduler-name "(servers_raw, service_raw)")' in text
    assert '"AI-phase2"' in text
    assert '"AI-phase3"' in text
    assert "OllamaLLM(model=\"llama3.2:latest\")" not in text
