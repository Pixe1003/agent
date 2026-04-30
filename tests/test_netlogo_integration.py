from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NLOGO = ROOT / "cloud_scheduler_agent.nlogo"


def test_netlogo_uses_phase1_agent_entrypoint_and_keeps_legacy_reporter():
    text = NLOGO.read_text(encoding="utf-8")

    assert 'py:setup ".\\\\.venv\\\\Scripts\\\\python.exe"' in text
    assert "from agent_phase1 import init_agent, schedule_service, last_decision_summary" in text
    assert "from agent_phase2 import init_agent as init_agent_phase2" in text
    assert "hybrid_stats_summary as hybrid_stats_summary_phase2" in text
    assert 'py:runresult "hybrid_stats_summary_phase2()"' in text
    assert "Phase 2 Hybrid Agent Usage" in text
    assert "global_state_raw" in text
    assert '"active_net_util"' in text
    assert 'schedule_service_phase2(servers_raw, service_raw, global_state_raw)' in text
    assert "from agent_phase3 import init_agent as init_agent_phase3" in text
    assert "init_agent(model_name='qwen3:8b', temperature=0.1)" in text
    assert "init_agent_phase2(model_name='qwen3:8b', backend='auto')" in text
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


def test_netlogo_ai_python_scheduler_uses_candidate_server_set():
    text = NLOGO.read_text(encoding="utf-8")

    assert "let svrIDs [who] of the-server-set" in text
    assert "if not any? the-server-set [ report nobody ]" in text


def test_netlogo_consolidation_guards_against_nobody_candidate():
    text = NLOGO.read_text(encoding="utf-8")
    consolidation_start = text.index("to consolidate-underutilized-servers")
    consolidation_end = text.index("to-report calc-migr-metrics", consolidation_start)
    consolidation = text[consolidation_start:consolidation_end]

    assert "let candidate find-server the-server-set self" in consolidation
    assert "ifelse candidate != nobody" in consolidation
    assert "set migr-list lput (list who ([who] of candidate)) migr-list" in consolidation
