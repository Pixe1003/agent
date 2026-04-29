# Multi-Agent Cloud Scheduling Report Draft

## Background

The project extends a NetLogo data-center scheduler into an agentic scheduling
system with tool calling, Planner-Scheduler-Critic coordination, memory, traces,
evaluation, and an SFT data loop.

## Method

The implementation keeps deterministic baselines and LLM-agent phases side by
side. Phase 1 introduces structured tool calling. Phase 2 adds a
Planner-Scheduler-Critic pipeline. Phase 3 adds working and episodic memory.

## Experiment Plan

Compare First-Fit, Balanced-Fit, Legacy AI, AI-phase1, AI-phase2, and AI-phase3
on SLA violation rate, energy proxy, rejection rate, fallback rate, and average
decision latency.

## Current Limitations

The first Phase 2/3 implementation is a deterministic skeleton. It is designed
to make the data flow testable before replacing the scheduler step with a live
LangGraph/ReAct LLM backend.

