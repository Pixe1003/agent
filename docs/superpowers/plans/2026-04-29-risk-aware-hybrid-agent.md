# Risk-Aware Hybrid Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Phase 2 consider global NetLogo risk signals while preserving millisecond fast-path scheduling.

**Architecture:** Add an optional `global_state_raw` argument to `agent_phase2.schedule_service`. Phase 2 computes local placement complexity plus global risk, records both in traces/statistics, and uses a risk-aware deterministic scorer for fast decisions. Structured Agent calls remain optional and budgeted by `hybrid_agent_mode="sync"`.

**Tech Stack:** Python 3.13, Pydantic, LangGraph optional fallback, NetLogo `py:` bridge, pytest.

---

### Task 1: Global Risk Model and Complexity Tests

**Files:**
- Modify: `tests/test_phase2_scheduler.py`
- Modify: `agent_phase2/scheduler.py`

- [ ] **Step 1: Write failing tests**

Add tests proving that a high global network/migration risk raises escalation metadata and that risk-aware fast path prefers higher NET headroom when network is hot.

- [ ] **Step 2: Run red tests**

Run: `py -3.13 -m pytest tests/test_phase2_scheduler.py -q`

Expected: tests fail because `schedule_service` does not accept global state and decision metadata lacks risk fields.

- [ ] **Step 3: Implement minimal model**

Add `RiskSnapshot`, optional `global_state_raw`, `_parse_risk_snapshot`, `_analyze_global_risk`, and merge global risk into `_analyze_complexity`.

- [ ] **Step 4: Run green tests**

Run: `py -3.13 -m pytest tests/test_phase2_scheduler.py -q`

Expected: all Phase 2 scheduler tests pass.

### Task 2: Risk-Aware Fast Path Scoring

**Files:**
- Modify: `tests/test_phase2_scheduler.py`
- Modify: `agent_phase2/scheduler.py`

- [ ] **Step 1: Write failing test**

Add a test where balanced fit chooses one server under low risk, but high NET risk chooses the candidate with larger NET residual.

- [ ] **Step 2: Implement scorer**

Pass risk metadata into `_run_heuristic` and `_propose`. Adjust `_candidate_score` with resource weights from risk tags. Keep deterministic fallback for missing risk.

- [ ] **Step 3: Verify**

Run: `py -3.13 -m pytest tests/test_phase2_scheduler.py -q`

Expected: tests pass and old heuristic behavior remains unchanged when no global risk is supplied.

### Task 3: NetLogo Phase 2 Global State Bridge

**Files:**
- Modify: `2143512_Jiale Miao_2025_Supplementary.nlogo`
- Modify: `tests/test_netlogo_integration.py`

- [ ] **Step 1: Write failing integration assertions**

Assert the NetLogo model builds `global_state_raw` from active utilization, migration counts, SLA counters, and reschedule counters, then passes it only to `schedule_service_phase2`.

- [ ] **Step 2: Implement bridge**

In `find-AI-python-server`, keep Phase 1/3 two-argument calls intact. When `python-scheduler-name = "schedule_service_phase2"`, set `global_state_raw` and call `schedule_service_phase2(servers_raw, service_raw, global_state_raw)`.

- [ ] **Step 3: Verify**

Run: `py -3.13 -m pytest tests/test_netlogo_integration.py -q`

Expected: integration assertions pass.

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `agent_phase2/README.md`
- Modify: `docs/development_log_phase2_latency.md`

- [ ] **Step 1: Document risk fields**

Explain `global_risk_score`, `risk_tags`, `risk_level`, `global_risk_agent_triggered`, and how to read `hybrid_stats_summary`.

- [ ] **Step 2: Full test suite**

Run:

```powershell
py -3.13 -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 3: NetLogo smoke**

Run:

```powershell
& 'D:\NETLOGO\netlogo-headless.bat' --model 'D:\Users\12057\Desktop\agent\2143512_Jiale Miao_2025_Supplementary.nlogo' --setup-file 'D:\Users\12057\Desktop\agent\benchmark\netlogo_100tick_smoke.xml' --experiment 'agent-100tick' --table -
```

Expected: `AI-phase2` completes 100 ticks and Summary includes Phase 2 Hybrid Agent Usage.
