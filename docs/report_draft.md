# 报告草稿 / Technical Report Draft

## 背景 / Background

本项目将 NetLogo 数据中心调度仿真扩展为一个 Agent-assisted scheduling system。系统包含结构化 tool calling、Planner-Scheduler-Critic 控制层、运行时 memory、trace、benchmark 和 SFT 数据构建流程。

核心设计是 control plane / fast path 分离：高频 placement 由确定性 fast path 完成，复杂请求、策略解释和历史案例复用由 Agent 模块参与。

## 方法 / Method

Phase 1 将自由文本 LLM 调度改为 Pydantic schema + tool calling，并保留 `select`、`reject`、`fallback` 三类明确语义。

Phase 2 引入 hybrid Planner-Scheduler-Critic control plane。默认 `backend="auto"` 只记录 Agent escalation 信号并使用 fast path；显式开启 `structured` 或 `hybrid_agent_mode="sync"` 时，复杂 case 可交给 LangGraph + Qwen3 structured Agent。

Phase 3 增加 working memory 与 episodic memory。检索到的历史调度案例会作为 `memory_context_raw` 传给 Phase 2，并进入 trace、复杂度分析和 structured Scheduler prompt。

## 实验计划 / Evaluation Plan

对比以下调度器：

- First-Fit
- Balanced-Fit
- Legacy AI
- AI-phase1
- AI-phase2
- AI-phase3

观察指标：

- SLA violation rate
- energy proxy
- service rejection rate
- fallback rate
- average decision latency
- Agent escalation rate
- memory usage rate

## 当前限制 / Limitations

本地同步 LLM 决策延迟较高，因此高频仿真默认使用 hybrid fast path。structured LLM 后端用于短 demo、复杂 case 分析、离线 trace 和数据飞轮。Phase 3 的 RAG 是调度案例检索，不是通用文档问答。
