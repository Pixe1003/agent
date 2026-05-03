# multi_agent / Planner-Scheduler-Critic 控制层

`multi_agent` 在保持 NetLogo 调用语义稳定的同时，提供 Planner-Scheduler-Critic control plane、全局风险感知、hybrid fast path 和可选 structured Agent 路径。

## 定位 / Positioning

`multi_agent` 不是每 tick 都调用 LLM 的在线调度器，而是 Agent control plane：

- 常规 placement 由确定性 fast path 立即返回 `server_id`。
- 复杂场景写入 `agent_escalation_needed`、`complexity_score` 和 `complexity_reasons`。
- `hybrid_agent_mode="sync"` 或 `backend="structured"` 才会同步调用 LangGraph + Qwen3 structured Agent。
- trace 与 `hybrid_stats()` 用于衡量 Agent 参与度，而不是只看 NetLogo 的 `ai-usage-count`。

## 后端模式 / Backends

| 模式 | 用途 |
|---|---|
| `backend="auto"` | 默认 hybrid 路径，适合 NetLogo 实跑和 benchmark |
| `backend="heuristic"` | 纯确定性路径，适合无 LLM 环境和单元测试 |
| `backend="hybrid"` | 显式 hybrid；可通过 `hybrid_agent_mode` 控制 record/sync |
| `backend="structured"` | 真实本地 LLM Agent，适合短 demo 或离线 trace |

## 公共 API / Public API

```python
from multi_agent import init_agent, schedule_service

init_agent(model_name="qwen3:8b", backend="auto")
server_id = schedule_service([[0, 80, 80, 80]], [10, 10, 10])
```

`schedule_service` 支持两个可选上下文：

```python
server_id = schedule_service(
    [[0, 60, 60, 25], [1, 25, 25, 80]],
    [20, 20, 20],
    {
        "active_net_util": 0.95,
        "current_auto_migrations": 4,
        "net_sla_violations": 1,
    },
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

第三个参数是 `global_state_raw`，第四个参数是 `memory_context_raw`。

## 关键观测字段 / Observability

```python
from multi_agent import hybrid_stats, hybrid_stats_summary

print(hybrid_stats_summary())
print(hybrid_stats())
```

重点字段：

- `escalation_ratio`：复杂场景判定占比。
- `global_risk_trigger_ratio`：全局风险触发升级信号的比例。
- `hybrid_agent_call_ratio`：真实同步调用 structured Agent 的比例；默认 record 模式下应为 `0.0`。
- `memory_usage_ratio`：`agent_memory` retrieval 命中并传入 memory context 的比例。
- `planner_policy_ratio`：决策中存在 planner/risk policy 信号的比例。
- `avg_latency_ms`：当前统计窗口内的平均调度延迟。

单次决策中的关键字段：

- `local_complexity_score`
- `global_risk_score`
- `global_risk_level`
- `global_risk_tags`
- `risk_policy`
- `risk_aware_fast_path`
- `retrieved_episode_count`
- `memory_used`
- `memory_confidence`
- `planner_policy_active`

NetLogo Monitor 可使用：

```netlogo
py:runresult "hybrid_stats_summary_phase2()"
```

NetLogo 端把 `multi_agent` 的入口仍以 `init_agent_phase2` / `schedule_service_phase2` 等别名暴露，避免改动 UI 标签。

## Structured Agent Demo / 结构化 Agent 演示

```python
from multi_agent import init_agent, schedule_service, last_decision_dict

init_agent(model_name="qwen3:8b", backend="hybrid", hybrid_agent_mode="sync")
server_id = schedule_service([[0, 42, 42, 42]], [38, 38, 38])
print(server_id, last_decision_dict())
```

或直接使用 structured 后端：

```python
init_agent(model_name="qwen3:8b", backend="structured", temperature=0, num_predict=512)
```

这两种路径会同步调用本地 LLM，不适合 NetLogo 高频仿真的默认运行。
