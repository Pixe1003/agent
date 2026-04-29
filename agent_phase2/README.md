# Phase 2 - Planner/Scheduler/Critic

This package keeps the NetLogo-facing API identical to `agent_phase1` while
adding a Planner-Scheduler-Critic pipeline. The first implementation is a
deterministic skeleton so tests, traces, memory, and benchmark tooling can run
without depending on a live local LLM.

Use:

```python
from agent_phase2 import init_agent, schedule_service

init_agent(model_name="heuristic")
server_id = schedule_service([[0, 80, 80, 80]], [10, 10, 10])
```

