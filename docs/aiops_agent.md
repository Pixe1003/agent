# AIOps Agent Design

`agent_aiops` implements the low-frequency monitoring and policy recommendation layer for the scheduler project. It does not choose a server, call `schedule_service`, or mutate NetLogo parameters. Realtime placement remains deterministic; the AIOps layer observes risk signals and produces advisory insights.

## Public API

```python
from agent_aiops import (
    init_agent,
    analyze_ops_state,
    observe_ops_state,
    current_alerts,
    aiops_stats,
    last_insight_dict,
    last_insight_summary,
)

init_agent(model_name="heuristic", backend="rule", enable_tracing=True)
insight = analyze_ops_state(
    global_state_raw,
    scheduler_stats_raw=None,
    recent_decisions_raw=None,
    memory_context_raw=None,
)
```

`analyze_ops_state(...)` is the single-shot analysis API. `observe_ops_state(...)` is the realtime monitoring API: call it every tick or after each scheduling event. It updates a rolling window, active alerts, and AIOps stats:

```python
observe_ops_state(global_state_raw, scheduler_stats_raw, recent_decisions_raw, tick=tick)
print(current_alerts())
print(aiops_stats())
```

The default `backend="rule"` path is deterministic and does not call an LLM. `backend="structured"` is reserved for later low-frequency explanation or report generation, but v1 keeps the production-facing behavior rule based.

## Architecture

The module uses a single orchestrator with logical roles:

- Risk Analyzer: scores CPU, memory, network, SLA, migration, reschedule, rejection, and energy signals.
- SLA/Migration Analyzer: highlights network SLA concentration and migration churn.
- Memory Context: includes Phase 3 episodic memory as evidence only.
- Policy Advisor: emits recommendations such as network headroom protection or migration cooldown.
- Harness Guard: adds approval, rollback, and cooldown guardrails to every insight.

This follows the useful part of harness engineering: keep the model inside a structured environment with explicit inputs, fixed output shape, trace capture, and safety gates. v1 does not introduce a separate multi-agent runtime because the current bottleneck is observability and control-loop safety, not parallel reasoning throughput.

## Realtime Data Flow

NetLogo already computes the realtime state used by the monitor:

- active CPU/MEM/NET utilization
- active server count
- current auto and consolidation migrations
- rescheduled services
- current CPU/MEM/NET SLA violations

The NetLogo Python bridge now sends that `global_state_raw` to `observe_ops_state(...)` after Phase 2 or Phase 3 scheduling. It also passes scheduler stats and recent decisions from Phase 2/3 so AIOps can distinguish infrastructure risk from Agent-control-plane risk.

During allocation, NetLogo stores the latest AIOps process state in these reporters:

- `aiops-current-risk-level`
- `aiops-current-risk-score`
- `aiops-current-alert-count`
- `aiops-monitor-summary`
- `aiops-monitor-stats`

The interface also includes an `AIOps Realtime Risk` plot that updates from `aiops-risk-score`, so the monitoring process is visible while services are being assigned.

The dashboard can consume real AIOps traces by exporting them to `dashboard/aiops-stream.json`:

```powershell
py -3.13 -m dashboard.export_aiops_stream --trace-dir traces --output dashboard/aiops-stream.json --algorithm AI-phase2
```

When `server_snapshots` are present in the AIOps evidence, the dashboard renders real candidate-server utilization from the scheduling event.

## Output Contract

Every insight includes:

- `risk_level`: `low`, `medium`, `high`, or `critical`
- `risk_score`: normalized `0.0` to `1.0`
- `risk_tags`: stable diagnostic tags such as `network-pressure`, `migration-pressure`, `sla-risk`, `energy-regression`, and `capacity-risk`
- `root_cause_summary`: short human-readable diagnosis
- `recommendations`: advisory actions with `requires_human_approval=True`
- `guardrails`: `do_not_auto_apply`, rollback condition, and cooldown hint
- `evidence`: metrics, scheduler stats, recent decision summary, and memory summary

Trace records use `TraceLogger` with `phase="aiops"` and write the full insight as the decision payload.

`observe_ops_state(...)` adds `mode="realtime"`, `tick`, and `active_alerts` to the insight. It suppresses repeated recommendations inside the configured cooldown window while keeping alerts active.
