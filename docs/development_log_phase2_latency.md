# 开发日志：Phase 2 本地 LLM 调度延迟问题

## 背景

Phase 2 的目标是把原本的确定性 Planner-Scheduler-Critic 骨架升级为真实本地 Agent：

- Planner 使用 LLM 产出调度策略标签。
- Scheduler 使用 Qwen3:8b + Pydantic 结构化输出产出 `select` / `reject` 提案。
- Critic 使用确定性规则校验资源约束，必要时触发 revise 或 fallback。
- NetLogo 对外 API 保持不变：`schedule_service(...) -> int`。

这个方向有利于展示 LangGraph、多智能体协作、结构化输出和 trace 数据闭环，但实际接入后暴露了运行速度问题。

## 出现的问题

### 1. 原生 tool calling 不稳定

Phase 1 中使用 `ChatOllama.bind_tools(...)` 调用 Qwen3:8b。实测模型能给出合理调度结论，但经常没有返回 LangChain/Ollama 期望的原生 `tool_calls`，导致系统必须进入确定性 safety fallback。

因此 Phase 2 没有继续强依赖原生 tool calling，而是改用 `with_structured_output(PydanticModel)`。

### 2. Qwen3 thinking 模式导致空输出或解析失败

初版 Phase 2 structured backend 仅在 prompt 中加入 `/no_think`，但完整调度 prompt 下仍出现：

- `OutputParserException: Invalid json output`
- response content 为空
- token 被消耗在模型 reasoning / thinking 中

后续确认 `langchain-ollama` 支持 `reasoning=False`，显式关闭后结构化输出稳定性明显改善。

### 3. AI-phase2 consolidation 中出现 `OF ... NOBODY` 运行时错误

运行 NetLogo `AI-phase2` 时曾出现：

```text
OF expected input to be a turtle agentset or turtle but got NOBODY instead.
called by procedure CONSOLIDATE-UNDERUTILIZED-SERVERS
```

根因不是 Python 直接崩溃，而是 NetLogo consolidation 迁移路径中存在一个隐含假设：

```netlogo
let candidate find-server the-server-set self
set migr-list lput (list who ([who] of candidate)) migr-list
```

代码只检查了 `count the-server-set > 0`，但这不等于 `find-server` 一定返回 turtle。AI-phase2 / fallback / staged placement 都可能合法返回 `nobody`。当 `candidate = nobody` 时，`[who] of candidate` 就触发运行时错误。

同次修复还发现 AI Python reporter 原本使用：

```netlogo
let svrIDs [who] of servers
```

这会把全局服务器传给 Python，而不是当前 NetLogo 已过滤好的候选集合。修复后改为：

```netlogo
let svrIDs [who] of the-server-set
```

并在候选集合为空时直接 `report nobody`。

当前处理：

- consolidation 中对 `candidate != nobody` 增加防护。
- AI Python scheduler 只使用 `the-server-set` 中的候选服务器。
- 新增回归测试检查 NetLogo 集成代码包含这些防护。
- 已用 NetLogo headless 100 tick `AI-phase2` smoke 验证通过。

### 4. 同步 LLM 调用速度不符合 NetLogo 高频仿真

即使结构化输出可用，本地 Qwen3:8b 仍然太慢。一次调度需要 Planner 调用 + Scheduler 调用；如果 Critic 要求 revise，还会触发额外 Scheduler 调用。

实测数据：

- `heuristic` fast path：约 `0.13ms` / decision。
- `structured` Qwen3 path：短 smoke 中约 `7.8s` / decision。
- 性能复测中出现约 `29s` / decision。

NetLogo 的 `py:` 调用是同步阻塞的。如果每次服务调度都等待本地 LLM，仿真会明显卡顿，无法满足实际实验和演示需求。

## 当前处理方案

### 1. 调整默认 backend 语义

`agent_phase2.init_agent(...)` 现在支持：

- `backend="auto"`：默认走毫秒级 deterministic fast path。
- `backend="heuristic"`：显式走 fast path。
- `backend="structured"`：显式调用 LangGraph + Qwen3 structured backend。

关键决策：

> NetLogo 实跑默认使用 `auto` / `heuristic`，不自动触发本地 LLM。

即使调用：

```python
init_agent(model_name="qwen3:8b", backend="auto")
```

系统也会走 fast path，而不是偷偷调用 Qwen3。

### 2. 保留真实 LLM Agent 路径作为短 demo / 离线 trace

`backend="structured"` 没有删除，原因是它仍然有展示价值：

- 可以证明项目具备 LangGraph 多节点 Agent 编排。
- 可以生成 Planner/Scheduler/Critic 的结构化 trace。
- 可以用于报告、简历、短视频 demo 中展示真实 LLM 决策。
- 可以为后续 SFT 数据构造提供样本。

但它不再作为高频仿真的默认运行路径。

### 3. 文档中明确区分两条路径

README 和 Phase 2 README 已更新：

- 实跑 / benchmark：使用 `auto` 或 `heuristic`。
- LLM demo / 离线 trace：使用 `structured`。

这样避免后续误把慢路径用于 NetLogo 每 tick 调度。

## 可能的后续解决方案

### 方案 A：Hybrid Agent，LLM 低频规划，启发式高频执行

让 LLM 不再参与每一次服务调度，而是每隔 N tick 或每当负载状态明显变化时运行一次 Planner，产出策略参数：

- 当前偏向 CPU / RAM / NET 哪个资源维度。
- 是否要更激进地提高利用率。
- 是否要更保守地避免 SLA 风险。

之后每次具体 placement 仍由 fast path 执行。

优点：

- 保留 Agent 决策参与感。
- 运行速度接近 heuristic。
- 更符合真实系统中 control plane / data plane 分离的设计。

缺点：

- 需要设计策略缓存和失效条件。

### 方案 B：异步批量调度

把多个待调度服务组成 batch，让 LLM 一次性给出多个 placement proposal，再由 Critic 批量校验。

优点：

- 摊薄 LLM 调用成本。
- 更适合离线 benchmark 或批处理实验。

缺点：

- NetLogo 当前调度逻辑是同步逐服务调用，改造成本较高。
- 实时交互仍可能卡顿。

### 方案 C：离线 Agent Trace，在线 Replay / Distill

用 `backend="structured"` 离线生成高质量 trace，然后训练或蒸馏一个轻量策略：

- 规则参数表。
- 小模型分类器。
- LoRA 微调后的本地模型。
- 甚至直接生成 heuristic policy 的配置。

在线仿真时不调用大模型，只执行蒸馏后的 fast policy。

优点：

- 最符合“Agent 数据飞轮”：trace -> 筛选 -> SFT / policy distillation -> benchmark。
- 可以在简历和报告中形成完整闭环。

缺点：

- 需要额外训练或策略抽取工作。

### 方案 D：更小模型或专门模型

尝试更小、更快、结构化输出更稳定的本地模型，例如 1.5B / 3B 级别模型，或者专门微调后的 scheduler model。

优点：

- 改动较小。
- 可能保留每次调度调用模型的形式。

缺点：

- 小模型稳定性和决策质量需要重新评估。
- 即使变快，也未必能达到毫秒级。

### 方案 E：外部 API 模型只用于对照实验

把 GPT / Claude / DeepSeek API 放在 benchmark 横评中，而不是 NetLogo 实时演示中。

优点：

- 可以展示多模型评测能力。
- 不影响本地实时演示。

缺点：

- 有成本、网络和 API key 管理问题。

## 当前推荐路线

短期推荐：

1. NetLogo 实跑、课堂演示和 benchmark 默认使用 `backend="auto"`。
2. Demo 中单独运行 1-3 个 `backend="structured"` 调度案例，展示真实 LangGraph + Qwen3 Agent trace。
3. 报告中诚实说明：本地同步 LLM 每次调度延迟过高，因此采用 fast path 作为在线 policy，LLM Agent 作为离线 planner / trace generator。

中期推荐：

1. 实现 Hybrid Agent：LLM 低频产出策略，heuristic 高频执行。
2. 用 structured trace 构建 SFT / distillation 数据集。
3. 在 Phase 4 benchmark 中比较：
   - pure heuristic
   - structured LLM online
   - hybrid LLM planner + heuristic executor
   - distilled policy

## 观测方式

为了避免把 NetLogo 的 `ai-usage-count` 误读成真实 LLM Agent 调用次数，Phase 2 增加了运行时统计接口：

- `hybrid_stats()`：返回结构化计数和比例。
- `hybrid_stats_summary()`：返回适合 NetLogo Monitor 显示的一行摘要。

关键字段：

- `agent_escalation_needed` / `escalation_ratio`：复杂场景判定次数和占比。
- `global_risk_agent_triggers` / `global_risk_trigger_ratio`：由全局运行态风险触发升级信号的次数和占比。
- `agent_sync_calls` / `hybrid_agent_call_ratio`：混合模式中真实同步调用 structured Agent 的次数和占比。
- `fast_path_decisions` / `fast_path_ratio`：混合模式中由常规算法直接完成的次数和占比。

默认 `hybrid_agent_mode="record"` 下，系统只记录复杂场景信号，不同步调用本地 LLM，因此 `hybrid_agent_call_ratio` 应为 `0.0`。如果切到 `hybrid_agent_mode="sync"`，复杂场景才会真正进入 structured Agent。

### 全局风险感知优化

原先 Phase 2 只根据单次 placement 判断复杂度，即 `servers_raw + service_raw`。这会漏掉曲线中的全局风险，例如网络利用率长期接近 100%、auto migration 高频、reschedule 增长或 SLA 增加。

现在 Phase 2 支持第三个可选参数 `global_state_raw`，NetLogo 的 `AI-phase2` 会传入：

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

这些字段会进入 trace 和最终 summary。高网络风险会提高 NET headroom 权重，让 fast path 更偏向保留网络余量；高全局风险也会提高 `agent_escalation_needed`，在 `hybrid_agent_mode="sync"` 下允许 structured Agent 接管。

NetLogo 中可在 Command Center 或 Monitor 使用：

```netlogo
py:runresult "hybrid_stats_summary_phase2()"
```

### Memory-aware Agent 参与度

Phase 3 的 RAG 定位为 retrieval-augmented scheduling memory，即调度案例检索，而不是文档问答。Phase 3 会把当前状态渲染成自然语言 summary 和归一化数值特征，检索相似历史 episode，并将 retrieved episodes 作为 `memory_context_raw` 传给 Phase 2。

Phase 2 对 memory context 的处理方式：

- 在线 fast path 不因为 memory 命中而阻塞调用 LLM。
- complexity metadata 增加 `retrieved_episode_count`、`memory_used` 和 `memory_confidence`。
- retrieved episodes 会写入 trace 和 `hybrid_stats()`，用于衡量 Agent control plane 的参与度。
- 只有在 `hybrid_agent_mode="sync"` 或 `backend="structured"` 时，retrieved episodes 才注入 Scheduler Agent prompt，作为 few-shot 调度案例。

因此 Agent 的参与度不再等同于“每 tick 调 LLM”，而是由 `planner_policy_active`、`agent_escalation_needed`、`agent_sync_calls`、`memory_used_decisions`、`retrieved_episode_count` 和 trace 数据共同体现。

## 结论

这次问题不是单纯代码优化问题，而是架构选择问题：

> 本地 8B LLM 同步参与每次调度，不适合 NetLogo 高频仿真。

后续根据“常规算法快速分配，复杂情况再用 Agent”的思路，代码进一步调整为 hybrid 设计：

- `backend="auto"`：默认 hybrid scheduler。
- 常规请求：使用 fast path 直接返回，保证 NetLogo 实跑速度。
- 复杂请求：记录 `agent_escalation_needed=True`、`complexity_score`、`complexity_reasons`。
- `hybrid_agent_mode="record"`：只记录升级信号，不同步调用 LLM，适合在线仿真。
- `hybrid_agent_mode="sync"`：复杂请求同步交给 structured Agent，适合短 demo 和离线分析。

当前代码因此形成三条路径：

- hybrid fast path 负责真实运行速度。
- hybrid sync path 负责少量复杂 case 的 Agent 接管。
- Phase 3 memory path 负责历史调度案例检索和解释上下文。
- structured LLM path 负责 Agent 能力展示、trace 和后续数据飞轮。

这比强行让 LLM 每次在线决策更适合项目落地，也更容易在报告中讲清楚工程权衡。
