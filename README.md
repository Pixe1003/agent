# Multi-Agent Cloud Scheduling

This project extends a NetLogo data-center scheduler into a staged
LLM-agent scheduling system.

## Current Milestones

- Phase 1: `agent_phase1` keeps the NetLogo API stable and calls a local
  Ollama model with Pydantic tool schemas. If the model does not return a tool
  call, a deterministic safety fallback selects or rejects safely and records
  `tool_call_succeeded=False`.
- Phase 2: `agent_phase2` adds a testable Planner-Scheduler-Critic skeleton
  with trace logging.
- Phase 3: `agent_phase3` adds working memory and episodic retrieval over
  prior scheduling decisions.
- Benchmark/SFT: `benchmark.runner` emits metrics CSV files, and
  `dataset.build_sft_dataset` turns trace rows into OpenAI-style tool-call
  SFT samples.

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r agent_phase1\requirements.txt
ollama list
```

`qwen3:8b` should be available locally. The current machine has verified that
the model exists, but tool calling is not fully reliable.

## Verification Commands

```powershell
py -3.13 -m pytest tests -q
.\.venv\Scripts\python -m agent_phase1.test_scheduler
py -3.13 -m benchmark.runner
py -3.13 -m dataset.build_sft_dataset
```

## NetLogo Integration

The model file now imports `agent_phase1` during `setup` and calls
`schedule_service(servers_raw, service_raw)` inside `find-AI-server`. The old
free-text LLaMA implementation is preserved as `find-AI-server-legacy` for
later benchmark comparisons.

The current local NetLogo install is at `D:\NETLOGO`. Run the 100-tick headless
smoke test from the project root:

```powershell
& 'D:\NETLOGO\netlogo-headless.bat' `
  --model 'D:\Users\12057\Desktop\agent\2143512_Jiale Miao_2025_Supplementary.nlogo' `
  --setup-file 'D:\Users\12057\Desktop\agent\benchmark\netlogo_100tick_smoke.xml' `
  --experiment 'agent-100tick' `
  --table -
```

The model uses `py:setup ".\\.venv\\Scripts\\python.exe"`, so create and install
the local virtual environment before launching NetLogo.

## Known Limitations

- NetLogo headless has been verified from `D:\NETLOGO` with the 100-tick
  `AI-phase2` smoke experiment.
- Phase 2/3 currently use deterministic scheduling logic. This is intentional:
  the graph, memory, trace, benchmark, and SFT data flow are now testable before
  replacing the scheduler internals with live LangGraph/ReAct LLM calls.
