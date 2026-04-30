# Risk-Aware Hybrid Agent Implementation Plan / 风险感知 Hybrid Agent 实施计划

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Let Phase 2 consider global NetLogo risk signals while preserving millisecond fast-path scheduling.

**Architecture:** Add optional `global_state_raw` to `agent_phase2.schedule_service`. Phase 2 combines local placement complexity with global risk, records both in traces/statistics, and uses risk-aware deterministic scoring for fast decisions. Structured Agent calls remain optional through `hybrid_agent_mode="sync"` or `backend="structured"`.

**Tech Stack:** Python 3.13, Pydantic, optional LangGraph structured backend, NetLogo `py:` bridge, pytest.

---

## Task 1 / Global Risk Model

**Files:**

- Modify: `tests/test_phase2_scheduler.py`
- Modify: `agent_phase2/scheduler.py`

- [x] Add tests proving high global network/migration risk raises escalation metadata.
- [x] Add tests proving risk-aware fast path can prefer higher NET headroom when network pressure is high.
- [x] Add `RiskSnapshot`, optional `global_state_raw`, `_parse_risk_snapshot`, `_analyze_global_risk`, and complexity metadata.
- [x] Verify with `py -3.13 -m pytest tests/test_phase2_scheduler.py -q`.

## Task 2 / Risk-Aware Fast Path

**Files:**

- Modify: `tests/test_phase2_scheduler.py`
- Modify: `agent_phase2/scheduler.py`

- [x] Add deterministic tests for low-risk and high-NET-risk candidate selection.
- [x] Pass risk metadata into fast-path scoring.
- [x] Keep legacy heuristic behavior unchanged when no global risk snapshot is supplied.

## Task 3 / NetLogo Global State Bridge

**Files:**

- Modify: `cloud_scheduler_agent.nlogo`
- Modify: `tests/test_netlogo_integration.py`

- [x] Assert the NetLogo model builds `global_state_raw` from active utilization, migration counts, SLA counters, and reschedule counters.
- [x] Pass `global_state_raw` only to `schedule_service_phase2(servers_raw, service_raw, global_state_raw)`.
- [x] Keep Phase 1 and Phase 3 two-argument NetLogo calls intact.

## Task 4 / Documentation and Verification

**Files:**

- Modify: `agent_phase2/README.md`
- Modify: `docs/development_log_phase2_latency.md`

- [x] Document `global_risk_score`, `global_risk_level`, `global_risk_tags`, `risk_policy`, and `hybrid_stats_summary()`.
- [x] Verify with `py -3.13 -m pytest tests -q`.
- [x] Optional NetLogo smoke command:

```powershell
& "$env:NETLOGO_HOME\netlogo-headless.bat" `
  --model ".\cloud_scheduler_agent.nlogo" `
  --setup-file ".\benchmark\netlogo_100tick_smoke.xml" `
  --experiment "agent-100tick" `
  --table -
```
