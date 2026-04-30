# 开发日志 / Phase 2 本地 LLM 调度延迟

## 背景 / Background

Phase 2 的目标是在保持 NetLogo API 稳定的前提下，引入 Planner-Scheduler-Critic control plane：

- Planner 产出调度策略标签。
- Scheduler 生成 `select` / `reject` 提案。
- Critic 校验资源约束，必要时触发 revise 或 fallback。
- NetLogo 侧仍调用 `schedule_service(...) -> int`。

这个方向有利于展示 LangGraph、多节点 Agent、结构化输出和 trace 数据闭环，但本地 LLM 同步接入后暴露了明显延迟问题。

## 问题记录 / Findings

### 原生 Tool Calling 不稳定 / Tool Calling Reliability

Phase 1 使用 `ChatOllama.bind_tools(...)` 调用本地 Qwen3:8b。模型通常能给出合理自然语言判断，但不总是返回 LangChain/Ollama 期望的 `tool_calls`。因此 Phase 2 改用 Pydantic structured output，减少自由文本解析失败。

### Thinking 模式影响结构化输出 / Reasoning Mode

仅在 prompt 中加入 `/no_think` 仍可能出现空输出或 JSON 解析失败。后续改为在 `langchain-ollama` 中显式设置 `reasoning=False`，结构化输出稳定性明显改善。

### NetLogo Nobody 防护 / Nobody Guard

`consolidate-underutilized-servers` 曾假设 `find-server` 一定返回 turtle：

```netlogo
let candidate find-server the-server-set self
set migr-list lput (list who ([who] of candidate)) migr-list
```

但 AI-phase2、fallback 或 staged placement 都可能合法返回 `nobody`。修复后在迁移路径中保留 `candidate != nobody` 防护。

同次修复还把 AI Python reporter 的候选服务器来源从全局 `servers` 改为当前已过滤的 `the-server-set`：

```netlogo
let svrIDs [who] of the-server-set
if not any? the-server-set [ report nobody ]
```

### 同步 LLM 不适合高频仿真 / LLM Latency

本地 8B LLM 的单次结构化决策延迟可达数秒到数十秒，而 NetLogo 的 `py:` 调用是同步阻塞的。若每次服务调度都等待 LLM，仿真会明显卡顿，不适合作为默认在线路径。

## 当前方案 / Current Design

Phase 2 采用 hybrid 设计：

- `backend="auto"`：默认 hybrid scheduler，常规请求走 fast path。
- `backend="heuristic"`：只使用确定性路径，适合测试和无 LLM 环境。
- `backend="structured"`：真实 LangGraph + Qwen3 structured Agent，适合短 demo 或离线 trace。
- `hybrid_agent_mode="record"`：只记录复杂场景升级信号，不同步调用 LLM。
- `hybrid_agent_mode="sync"`：复杂场景同步交给 structured Agent，用于小规模演示。

## 全局风险 / Global Risk

Phase 2 支持第三个可选参数 `global_state_raw`。NetLogo 的 `AI-phase2` 会传入：

- active CPU/MEM/NET utilization
- active server count
- current auto/consolidation migration events
- rescheduled service count
- current CPU/MEM/NET SLA violation signals

Python 侧会计算：

- `global_risk_score`
- `global_risk_level`
- `global_risk_tags`
- `risk_policy`

这些字段进入 trace 和 `hybrid_stats()`。例如网络风险较高时，fast path 会提高 NET headroom 权重。

## Memory 参与度 / Memory-Aware Agent

Phase 3 的 RAG 定位为 retrieval-augmented scheduling memory，而不是文档问答。Phase 3 会检索相似历史 episode，并把结果作为 `memory_context_raw` 传给 Phase 2。

Phase 2 对 memory context 的处理方式：

- 在线 fast path 不因为 memory 命中而阻塞调用 LLM。
- complexity metadata 增加 `retrieved_episode_count`、`memory_used` 和 `memory_confidence`。
- retrieved episodes 会写入 trace 和 `hybrid_stats()`。
- 只有 `hybrid_agent_mode="sync"` 或 `backend="structured"` 时，retrieved episodes 才注入 Scheduler Agent prompt。

## 观测方式 / Observability

NetLogo Command Center 或 Monitor 可使用：

```netlogo
py:runresult "hybrid_stats_summary_phase2()"
```

重点关注：

- `escalation_ratio`
- `global_risk_trigger_ratio`
- `hybrid_agent_call_ratio`
- `fast_path_ratio`
- `memory_usage_ratio`
- `avg_latency_ms`

默认 `hybrid_agent_mode="record"` 下，`hybrid_agent_call_ratio` 应为 `0.0`。

## 结论 / Conclusion

本地 8B LLM 同步参与每次调度不适合 NetLogo 高频仿真。当前实现将 Agent 用作控制层和数据闭环来源：fast path 保证仿真速度，structured Agent 负责少量复杂 case、trace 生成和后续 SFT/蒸馏数据。
