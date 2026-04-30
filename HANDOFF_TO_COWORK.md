# 技术路线移交 / Technical Handoff

本文是开源仓库中的技术路线说明，帮助新的维护者理解系统目标、当前架构和后续扩展方向。文档不依赖历史对话，也不包含个人背景信息。

## 项目背景 / Project Background

项目以 NetLogo 数据中心服务调度仿真为基础。仿真中服务持续到达，scheduler 需要把服务放置到合适服务器上，并在资源利用率、SLA 风险、迁移成本和能耗代理指标之间做权衡。

仓库新增 Python Agent 层，用于把原有启发式调度扩展为可观测、可评测、可生成 trace 的 Agent-assisted scheduling system。

## 当前交付 / Current State

- `cloud_scheduler_agent.nlogo`：NetLogo 仿真模型，包含 baseline 算法、legacy AI 路径和 `AI-phase1` / `AI-phase2` / `AI-phase3`。
- `agent_phase1`：结构化 tool-calling 调度入口，保持 `schedule_service(servers_raw, service_raw) -> int`。
- `agent_phase2`：hybrid Planner-Scheduler-Critic control plane，支持 `global_state_raw` 和 `memory_context_raw`。
- `agent_phase3`：working memory 与 episodic retrieval，检索相似历史调度案例并委托 Phase 2。
- `benchmark`：批量评测与 metrics 输出。
- `dataset`：将 trace 转换为 tool-call SFT 样本。
- `tests`：覆盖 Python API、NetLogo 集成片段、trace、benchmark 和 memory 行为。

## 架构原则 / Architecture Principles

1. **NetLogo API 稳定**：Python 返回 `>=0`、`-1`、`-2` 三类整数，NetLogo 决定 select、fallback 或 reject。
2. **候选集合正确**：Python 只接收 `the-server-set` 中的候选服务器，不能接收全局 `servers`。
3. **高频路径快速**：NetLogo 实跑默认使用 Phase 2 hybrid fast path，不每次同步调用本地 LLM。
4. **Agent 负责控制层**：Agent escalation、risk policy、structured trace 和 memory context 用于解释、分析和少量复杂 case 接管。
5. **可观测性优先**：关键决策字段写入 trace，并通过 `hybrid_stats()` / `hybrid_stats_summary()` 暴露。

## Phase 1 / Structured Tool Calling

Phase 1 解决自由文本 LLM 调度的脆弱性：

- 用 Pydantic schema 校验 tool args。
- 用 `valid_ids` 校验候选服务器。
- 把 reject 视为合法结构化动作。
- 输出 `last_decision_summary()` 和 `last_decision_dict()` 供调试与后续 memory 使用。

受本地小模型 tool calling 稳定性限制，Phase 1 保留 safety fallback。失败时返回 `-1`，由 NetLogo 侧调用 `find-balanced-fit-server`。

## Phase 2 / Hybrid Control Plane

Phase 2 包含三条路径：

- `backend="auto"`：默认 hybrid fast path，适合 NetLogo 实跑。
- `backend="heuristic"`：纯确定性路径，适合无 LLM 环境。
- `backend="structured"`：LangGraph + Qwen3 structured Agent，适合短 demo 和离线 trace。

`schedule_service` 当前签名：

```python
schedule_service(
    servers_raw,
    service_req_raw,
    global_state_raw=None,
    memory_context_raw=None,
) -> int
```

`global_state_raw` 来自 NetLogo 的全局运行态，包括 active utilization、migration、reschedule 和 SLA violation 信号。Phase 2 会据此计算 `global_risk_score`、`global_risk_level`、`global_risk_tags` 和 `risk_policy`。

## Phase 3 / Scheduling Memory

Phase 3 把当前状态渲染为检索查询，查找相似历史 episode，并构造 `memory_context_raw`。Phase 2 会把 memory 命中写入 metadata 与 trace；只有 structured/sync 路径才把 retrieved episodes 注入 Scheduler prompt。

该 memory 是调度案例检索，不是通用文档问答。

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
- 修改 Phase 2 决策字段时同步更新 `agent_phase2/README.md` 和开发日志。
- 开源文档中的命令应使用相对路径或环境变量，不写入本机绝对路径。
