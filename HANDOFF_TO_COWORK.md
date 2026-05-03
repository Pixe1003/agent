# 技术路线移交 / Technical Handoff

本文是开源仓库中的技术路线说明，帮助新的维护者理解系统目标、当前架构和后续扩展方向。文档不依赖历史对话，也不包含个人背景信息。

## 项目背景 / Project Background

项目以 NetLogo 数据中心服务调度仿真为基础。仿真中服务持续到达，scheduler 需要把服务放置到合适服务器上，并在资源利用率、SLA 风险、迁移成本和能耗代理指标之间做权衡。

仓库新增 Python Agent 层，用于把原有启发式调度扩展为可观测、可评测、可生成 trace 的 Agent-assisted scheduling system。

## 当前交付 / Current State

- `cloud_scheduler_agent.nlogo`：NetLogo 仿真模型，包含 baseline 算法、legacy AI 路径和 `AI-phase2` / `AI-phase3` UI 标签；`AI` / `AI-phase1` 入口目前统一委托 `multi_agent`。
- `agent_common`：共享 Pydantic schema、prompt 渲染和 JSONL tracing，作为各 Agent 模块的公共契约层。
- `multi_agent`：hybrid Planner-Scheduler-Critic control plane，支持 `global_state_raw`、`memory_context_raw` 和 `aiops_insight_raw`。
- `agent_memory`：working memory 与 episodic retrieval，检索相似历史调度案例并委托 `multi_agent`。
- `agent_aiops`：realtime ops monitor + anomaly/risk analysis，输出 risk tags、active alerts 和 guardrail 建议；闭环时作为 `multi_agent` critic 的额外安全信号。
- `benchmark`：批量评测与 metrics 输出。
- `demo` / `dashboard` / `scripts`：分别提供 AIOps 闭环 A/B demo、实时 dashboard 数据导出/服务和 Pareto 图生成。
- `dataset`：将 trace 转换为 tool-call SFT 样本。
- `tests`：覆盖 Python API、NetLogo 集成片段、trace、benchmark 和 memory 行为。

## 架构原则 / Architecture Principles

1. **NetLogo API 稳定**：Python 返回 `>=0`、`-1`、`-2` 三类整数，NetLogo 决定 select、fallback 或 reject。
2. **候选集合正确**：Python 只接收 `the-server-set` 中的候选服务器，不能接收全局 `servers`。
3. **高频路径快速**：NetLogo 实跑默认使用 `multi_agent` hybrid fast path，不每次同步调用本地 LLM。
4. **Agent 负责控制层**：Agent escalation、risk policy、AIOps insight、structured trace 和 memory context 用于解释、分析和少量复杂 case 接管。
5. **可观测性优先**：关键决策字段写入 trace，并通过 `last_decision_dict()`、`hybrid_stats()`、`aiops_stats()` 等接口暴露。

## Phase 1 / Retired Single-Agent Path

旧的 Phase 1 单 Agent tool-calling 包已经退役，相关能力被拆入当前模块：

- `agent_common.schemas` 保留 Pydantic schema 与 `valid_ids` 校验。
- `multi_agent` 保留结构化后端路径，用于短 demo 或离线 trace。
- NetLogo 中 `find-AI-server` / `AI-phase1` 兼容入口目前委托 `schedule_service_phase2(...)`，避免 UI 标签和旧实验脚本断裂。

维护时不要再新增旧的 phase 包；需要结构化 tool-calling 能力时在 `multi_agent` 的 `backend="structured"` 路径内演进。

## Phase 2 / Hybrid Control Plane

Phase 2 包含四条路径：

- `backend="auto"`：默认 hybrid fast path，适合 NetLogo 实跑。
- `backend="heuristic"`：纯确定性路径，适合无 LLM 环境。
- `backend="structured"`：LangGraph + Qwen3 structured Agent，适合短 demo 和离线 trace。
- `backend="hybrid"`：显式 hybrid，可通过 `hybrid_agent_mode="record"` / `"sync"` 控制是否同步调用 structured Agent。

`multi_agent.schedule_service` 当前签名：

```python
schedule_service(
    servers_raw,
    service_req_raw,
    global_state_raw=None,
    memory_context_raw=None,
    aiops_insight_raw=None,
) -> int
```

`global_state_raw` 来自 NetLogo 的全局运行态，包括 active utilization、migration、reschedule 和 SLA violation 信号。`multi_agent` 会据此计算 `global_risk_score`、`global_risk_level`、`global_risk_tags` 和 `risk_policy`。

`aiops_insight_raw` 来自 `agent_aiops.observe_ops_state(...)`，包含 `risk_tags`、`risk_level`、`active_alerts` 等字段。heuristic/hybrid fast path 会把它作为 AIOps-aware critic 的附加安全边际。

## Phase 3 / Scheduling Memory

Phase 3 对应当前 `agent_memory` 包。它把当前状态渲染为检索查询，查找相似历史 episode，并构造 `memory_context_raw` 后委托 `multi_agent.schedule_service(...)`。`multi_agent` 会把 memory 命中写入 metadata 与 trace；只有 structured/sync 路径才把 retrieved episodes 注入 Scheduler prompt。

该 memory 是调度案例检索，不是通用文档问答。

## AIOps Closed Loop

`agent_aiops` 是 Python-first 的监控和策略建议层。常用入口：

```python
from agent_aiops import init_agent, observe_ops_state

init_agent(model_name="heuristic", backend="rule")
insight = observe_ops_state(global_state_raw, tick=tick)
```

它不直接选择服务器，也不修改 NetLogo 策略。闭环效果来自 `multi_agent` 在收到 `aiops_insight_raw` 后收紧 critic 阈值；高风险 placement 返回 `-1`，由 NetLogo 侧 balanced-fit 兜底。

## Benchmark 与 Dataset / Evaluation and Data

`benchmark.runner` 用于生成不同调度策略的指标，重点观察：

- rejection rate
- fallback rate
- average decision latency
- escalation ratio
- memory usage ratio
- SLA/energy proxy metrics

`dataset.build_sft_dataset` 从 trace 构造 OpenAI 风格 tool-call 样本，为后续 SFT 或策略蒸馏提供数据。

## 后续方向 / Roadmap

- 增加更系统的 benchmark 场景矩阵。
- 用 structured trace 生成高质量 SFT 数据。
- 比较 pure heuristic、structured LLM online、hybrid Agent 和 distilled policy。
- 增加更完整的可视化报告，展示 latency、risk、memory 和调度质量之间的权衡。

## 维护提示 / Maintenance Notes

- 修改 NetLogo 集成时同步更新 `tests/test_netlogo_integration.py`。
- 修改 `multi_agent` 决策字段时同步更新 `multi_agent/README.md`、README 和相关 dashboard/export 测试。
- 修改 memory 持久化或检索语义时同步更新 `tests/test_phase3_memory.py`。
- 修改 benchmark 输出字段或默认场景矩阵时同步更新 README headline results。
- 开源文档中的命令应使用相对路径或环境变量，不写入本机绝对路径。
