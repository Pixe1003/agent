# 开发日志：从 Agent 调度器到 AI Ops 监控层

日期：2026-04-30

## 背景

本阶段工作的目标是评估 Phase 2 / Phase 3 Agent 调度路径在 NetLogo 云数据中心仿真中的实际价值，并与原生 `balanced-fit` 算法进行同等条件对比。前期实验显示，Phase 2 / Phase 3 虽然具备风险分析、记忆检索和可观测性能力，但在直接参与高频服务放置时，SLA、迁移次数和能耗表现并未稳定超过原生启发式算法。

因此，本阶段同时进行了两类调整：

1. 将 `AI-phase2` 和 `AI-phase3` 前 10 tick 的 warm-up 初始分配从 `first-fit` 改为 `balanced-fit`。
2. 将 Phase 2 fast path 的候选服务器打分改为接近 NetLogo `balanced-fit` 的 post-placement resource distance，即最小化放置后的 CPU / RAM / NET 利用率差值。

Phase 3 仍然委托 Phase 2 做实际 placement，因此会自动继承 Phase 2 fast path 的变化。

## 实现记录

### NetLogo warm-up 分配

修改前：

```netlogo
ifelse ticks < 10
[ set candidate find-first-fit-server the-server-set the-service ]
[ set candidate find-AI-phase2-server the-server-set the-service ]
```

修改后：

```netlogo
ifelse ticks < 10
[ set candidate find-balanced-fit-server the-server-set the-service ]
[ set candidate find-AI-phase2-server the-server-set the-service ]
```

`AI-phase3` 采用相同修改。这样可以避免 Phase 2 / Phase 3 在仿真初期形成明显劣于 `balanced-fit` 的资源碎片。

### Phase 2 fast path 打分

修改前，Phase 2 使用 residual spread 和 risk-weighted headroom 对候选服务器排序。该方式在网络风险较高时会偏向保留 NET headroom，但与 NetLogo 原生 `balanced-fit` 的行为并不完全一致。

修改后，Phase 2 fast path 使用以下逻辑：

```python
post_utilization = [
    100.0 - server.cpu_free_pct + ctx.service.cpu_pct,
    100.0 - server.ram_free_pct + ctx.service.ram_pct,
    100.0 - server.net_free_pct + ctx.service.net_pct,
]
resource_distance = max(post_utilization) - min(post_utilization)
return (resource_distance, server.server_id)
```

该逻辑的含义是：选择放置后 CPU / RAM / NET 利用率最均衡的服务器。风险信号仍然被记录到 metadata 和统计中，但不再覆盖 fast path 的主排序。

## 最新实验结果

在 20 台服务器、500 个服务的同等条件下，最新结果如下：

| 指标 | Phase 2 | Phase 3 | Balanced-fit |
|---|---:|---:|---:|
| Total Services | 500 | 500 | 500 |
| Average Lifetime | 29.7269h | 29.7269h | 30.0215h |
| Consolidation Migrations | 178 | 178 | 187 |
| Auto Migrations | 1884 | 1884 | 876 |
| Total Migrations | 2062 | 2062 | 1063 |
| CPU SLA Violations | 1 | 1 | 0 |
| Memory SLA Violations | 4 | 4 | 1 |
| Network SLA Violations | 33 | 33 | 2 |
| Total SLA Violations | 38 | 38 | 3 |
| Rescheduled Services | 3 | 3 | 27 |
| Rejected Services | 0 | 0 | 2 |
| Power Consumption | 144.0681 kWh | 144.0681 kWh | 138.9518 kWh |

Phase 2 Agent usage:

```text
hybrid total=2613 fast=2613 escalate=1582 (60.5%)
risk=1503 (57.5%) agent_sync=0 (0.0%)
avg_latency=0.189ms fallback=0
top reasons=global-reschedule-watch, global-migration-watch, global-sla-risk
```

Phase 3 Agent usage:

```text
phase3 total=2613 fast=2613 escalate=69 (2.6%)
agent_sync=0 (0.0%) memory=2613 (100.0%)
avg_retrieved=3.00 avg_latency=0.244ms fallback=0
```

## 分析结论

### 1. Balanced-fit 仍然是更强的实时执行层

即使 Phase 2 fast path 已经改为 balanced-fit 风格，原生 `balanced-fit` 的 SLA 和迁移表现仍然明显更好：

- 总 SLA 违约：Phase 2 / 3 为 38，Balanced-fit 为 3。
- 网络 SLA 违约：Phase 2 / 3 为 33，Balanced-fit 为 2。
- 总迁移次数：Phase 2 / 3 为 2062，Balanced-fit 为 1063。
- Auto Migration：Phase 2 / 3 为 1884，Balanced-fit 为 876。

主要原因可能是 Python 侧当前只接收有限候选服务器快照和百分比资源信息，而 NetLogo 原生 `balanced-fit` 直接在完整候选集合和真实物理资源容量上计算。因此，直接让 Agent / Python fast path 替代原生启发式调度器，短期内不具备明显生产优势。

### 2. Phase 3 的 memory 更适合作为观测信号

Phase 3 的业务结果与 Phase 2 完全一致，但 Agent 参与指标不同：

- Phase 2 escalation ratio：60.5%。
- Phase 3 escalation ratio：2.6%。
- Phase 3 memory usage：100%。

这说明 memory 检索能够降低复杂度判断或解释压力，但在 `agent_sync=0` 的默认高频仿真路径下，它不会改变实际调度动作。该结果支持一个新的系统定位：Phase 3 memory 更适合作为 AI Ops 监控上下文，而不是每次 placement 的直接决策依据。

### 3. Agent 的实际价值在慢速控制层

在生产环境中，高频 placement 决策需要满足低延迟、稳定、可审计和可回滚。确定性算法在这些方面更可靠。Agent 更适合处理低频、复杂、跨指标的判断，例如：

- 发现网络 SLA 风险持续升高。
- 解释 auto migration 异常增长的原因。
- 识别某些服务器或机架形成资源热点。
- 判断当前策略是否进入高风险区间。
- 输出调参建议、扩容建议或策略切换建议。

因此，后续重点应从 “Agent 直接替代调度算法” 转向 “Agent 作为 AI Ops 监控层和慢速控制层”。

## 后续重点开发方向：AI Ops Monitoring Layer

建议后续架构如下：

```text
Real-time execution layer:
  NetLogo / production scheduler
  deterministic policies: balanced-fit, min-power, migration rules

AI Ops monitoring layer:
  Agent observes metrics, explains risks, retrieves similar episodes,
  generates alerts, recommends policy changes, and writes trace summaries.

Slow control loop:
  Human or safe controller reviews Agent recommendations before applying
  threshold changes, cooldown changes, capacity planning, or policy switching.
```

### 方向一：异常检测与风险分级

Agent 不直接选择服务器，而是持续读取系统指标：

- active CPU / memory / network utilization
- SLA violation counters
- auto migration and consolidation migration
- rescheduled and rejected services
- active server count
- per-run energy consumption

输出统一风险等级：

```text
low / medium / high / critical
```

并给出触发原因，例如：

```text
network-pressure
migration-watch
reschedule-watch
sla-risk
energy-regression
capacity-risk
```

### 方向二：策略建议而非直接动作

Agent 输出建议，不直接执行：

- 当前网络压力持续升高，建议提高网络 headroom 或启用网络保护模式。
- 当前 auto migration 过高，建议增加 migration cooldown。
- 当前 SLA 风险集中在网络资源，建议限制高 NET 服务进入热点区域。
- 当前能耗显著升高但 SLA 未改善，建议回退到 balanced-fit 或 min-power 混合策略。

这类建议更符合生产系统中的 change management 流程。

### 方向三：实验报告自动生成

Agent 可以自动总结每次仿真：

- 和上一轮实验相比的指标变化。
- SLA、迁移、能耗之间的 trade-off。
- 主要瓶颈资源。
- 是否存在回归。
- 下一轮实验建议。

这部分价值已经在当前对比分析中体现出来，适合作为论文或项目展示中的 Agent 能力。

### 方向四：只在低频场景进入 sync Agent

默认保持：

```text
agent_sync=0
```

只有在低频、高风险、非实时路径中才允许 Agent 参与决策，例如：

- 每 N tick 做一次全局健康检查。
- SLA 风险进入 critical 后生成策略建议。
- 仿真结束后做 postmortem。
- 离线 benchmark 后生成下一轮参数搜索方向。

这样可以保留 Agent 的解释和推理价值，同时避免高频 placement 中的延迟和不稳定性。

## 阶段性结论

当前成果表明，Agent 不适合作为每次服务放置的直接替代调度器。更合理、更有生产价值的定位是：

```text
Balanced-fit 等确定性算法负责实时执行；
Agent 负责 AI Ops 监控、风险解释、历史案例检索、实验总结和策略建议。
```

这个方向既保留了原生调度算法的稳定性，也为系统增加了智能化观测、诊断和持续优化能力，是后续开发的重点方向。

## AIOps Agent v1 实现取向

v1 采用 `agent_aiops` 作为 Python-first 的实时监控与策略建议模块。它读取 `global_state`、Phase 2 hybrid stats、recent decisions 和 Phase 3 memory context，输出结构化 insight：风险等级、风险标签、诊断摘要、策略建议、证据和 guardrails。

该模块默认使用规则路径，不调用 LLM，也不返回 `server_id`。所有策略建议都带有 `requires_human_approval=True` 和 `do_not_auto_apply`，用于保持生产式 change management 边界。

实时监测接口为 `observe_ops_state(...)`。NetLogo 在 Phase 2 / Phase 3 每次调度后调用该接口，并把当前 active utilization、migration、reschedule 和 SLA violation 信号传入 AIOps。AIOps 会维护 rolling window、active alerts 和推荐冷却：异常检测可以每次事件运行，策略建议不会每 tick 重复刷屏。

为了让监控过程在分配中可见，NetLogo 侧缓存最新 AIOps 状态到 `aiops-risk-level`、`aiops-risk-score`、`aiops-active-alert-count`、`aiops-last-insight-summary` 和 `aiops-last-stats-summary`。这些字段通过 reporter 暴露给 Monitor，同时 `AIOps Realtime Risk` 图会持续绘制风险分数。

架构上采用“逻辑多角色、单入口”的设计，而不是 v1 就引入真实多 agent 协同：

- Risk Analyzer 负责统一风险打分口径。
- SLA/Migration Analyzer 识别网络 SLA、迁移和 reschedule 异常。
- Memory Context 把历史案例作为 evidence，不直接改变动作。
- Policy Advisor 输出建议，不执行策略。
- Harness Guard 为每条 insight 添加审批、回滚和冷却约束。

这个选择参考 harness engineering 的核心思想：真正决定 Agent 可靠性的不是更自由的推理链，而是外层运行环境、结构化接口、trace、评估和安全边界。当前项目的主要问题是可观测性和策略闭环，不是并行推理吞吐，因此 v1 先做单入口可测、可回归、可审计的监控 Agent。
