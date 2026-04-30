# Phase 2 - 规划器/调度器/审查器

这个包保持与 `agent_phase1` 完全一致的 NetLogo 对外 API，同时增加“规划器-调度器-审查器”调度流水线。

设计定位：Phase 2 是 Agent control plane。在线路径仍由 fast path executor 返回 `server_id`，Agent 信号用于识别复杂场景、解释策略、记录 trace，并在显式开启时接管少量复杂 case。

- `backend="auto"`：hybrid 路径。常规请求用毫秒级 fast path；复杂请求写入 `agent_escalation_needed=True`、`complexity_score` 和 `complexity_reasons`。默认 `hybrid_agent_mode="record"` 不阻塞调用 LLM，适合 NetLogo 实跑。
- `backend="heuristic"`：只使用确定性路径，适合单元测试、benchmark 和无 LLM 环境。
- `backend="structured"`：真实本地 LLM agent 路径，使用 LangGraph 编排 Planner/Scheduler/Critic，并用 Qwen3 + Pydantic 结构化输出生成调度提案。Critic 会校验资源约束，最多触发 2 次 revise，失败则返回 fallback。这个路径会同步调用本地 8B LLM，适合短 demo 或离线 trace，不适合每 tick 高频仿真。
- Phase 2 现在支持可选的 `global_state_raw`，用于把 NetLogo 的全局风险状态传入调度器。全局风险会影响 `agent_escalation_needed`，也会调整 fast path 的资源权重，例如网络高风险时优先保留 NET headroom。
- Phase 2 也支持可选的 `memory_context_raw`，用于接收 Phase 3 retrieval 返回的历史调度案例。memory context 会进入复杂度分析、trace 统计，并在 structured Agent 路径中注入 Scheduler prompt。

使用示例：

```python
from agent_phase2 import init_agent, schedule_service

init_agent(model_name="qwen3:8b", backend="auto")
server_id = schedule_service([[0, 80, 80, 80]], [10, 10, 10])
```

传入全局风险：

```python
server_id = schedule_service(
    [[0, 60, 60, 25], [1, 25, 25, 80]],
    [20, 20, 20],
    {
        "active_net_util": 0.95,
        "current_auto_migrations": 4,
        "net_sla_violations": 1,
    },
)
```

传入历史调度案例：

```python
server_id = schedule_service(
    [[0, 42, 42, 42], [1, 60, 8, 60], [2, 5, 80, 80]],
    [38, 38, 38],
    None,
    {
        "episodic": [
            {
                "episode_id": "similar-high-pressure",
                "action_server_id": 0,
                "reasoning_trace": "A similar large balanced request selected server 0.",
                "reward": 0.9,
            }
        ]
    },
)
```

观察混合模式占比：

```python
from agent_phase2 import hybrid_stats, hybrid_stats_summary

print(hybrid_stats_summary())
print(hybrid_stats())
```

重点看两个比例：

- `escalation_ratio`：复杂场景判定占比，即常规算法认为“应该升级给 Agent 看”的比例。
- `global_risk_trigger_ratio`：由全局风险触发升级信号的比例。
- `hybrid_agent_call_ratio`：混合模式中真实同步调用 Agent 的比例。默认 `hybrid_agent_mode="record"` 时它会是 `0.0`，因为系统只记录升级信号，不阻塞调用 LLM。
- `memory_usage_ratio`：调度时命中 retrieved episodes 的比例。
- `planner_policy_ratio`：决策中存在 Planner/strategy policy 信号的比例。
- `avg_latency_ms`：当前统计窗口内的平均调度延迟。

关键决策字段：

- `local_complexity_score`：单次 placement 的局部复杂度。
- `global_risk_score` / `global_risk_level`：NetLogo 全局运行态风险。
- `global_risk_tags`：风险原因，例如 `network-pressure`、`migration-pressure`、`sla-risk`。
- `risk_policy`：fast path 使用的资源权重策略。
- `risk_aware_fast_path`：本次是否使用了风险感知快速路径。
- `retrieved_episode_count` / `memory_used` / `memory_confidence`：Phase 3 retrieval 传入的历史案例数量和置信度。
- `planner_policy_active`：本次决策是否带有策略标签，可用于衡量 Agent control-plane 参与度。

在 NetLogo Command Center 或 Monitor 中可以用：

```netlogo
py:runresult "hybrid_stats_summary_phase2()"
```

复杂请求同步交给 Agent：

```python
from agent_phase2 import init_agent, schedule_service, last_decision_dict

init_agent(model_name="qwen3:8b", backend="hybrid", hybrid_agent_mode="sync")
server_id = schedule_service([[0, 42, 42, 42]], [38, 38, 38])
print(server_id, last_decision_dict())
```

本地 Qwen3 路径：

```python
from agent_phase2 import init_agent, schedule_service, last_decision_dict

init_agent(model_name="qwen3:8b", backend="structured", temperature=0, num_predict=512)
server_id = schedule_service([[0, 80, 80, 80]], [10, 10, 10])
print(server_id, last_decision_dict())
```
