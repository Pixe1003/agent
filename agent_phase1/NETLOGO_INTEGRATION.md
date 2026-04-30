# NetLogo 集成指南 / NetLogo Integration Guide

本文说明 `cloud_scheduler_agent.nlogo` 如何调用 Python Agent 包。当前模型已经集成 Phase 1、Phase 2 和 Phase 3；后续修改时应保持这些入口函数和哨兵值语义稳定。

## 目录结构 / Expected Layout

```text
project-root/
├── cloud_scheduler_agent.nlogo
├── agent_phase1/
├── agent_phase2/
├── agent_phase3/
├── benchmark/
└── tests/
```

NetLogo 通过 `sys.path.insert(0, os.getcwd())` 导入本仓库中的 Python 包，因此建议从项目根目录打开模型或运行 headless 命令。

## Setup 初始化 / Setup Imports

`setup` 阶段应初始化三个阶段的调度入口：

```netlogo
(py:run
  "import sys, os"
  "sys.path.insert(0, os.getcwd())"
  "from agent_phase1 import init_agent, schedule_service, last_decision_summary"
  "from agent_phase2 import init_agent as init_agent_phase2, schedule_service as schedule_service_phase2, last_decision_summary as last_decision_summary_phase2, hybrid_stats_summary as hybrid_stats_summary_phase2"
  "from agent_phase3 import init_agent as init_agent_phase3, schedule_service as schedule_service_phase3, last_decision_summary as last_decision_summary_phase3"
  "init_agent(model_name='qwen3:8b', temperature=0.1)"
  "init_agent_phase2(model_name='qwen3:8b', backend='auto')"
  "init_agent_phase3(model_name='heuristic')"
)
```

`backend="auto"` 是 Phase 2 的默认线上路径：常规请求走 fast path，复杂请求记录 Agent escalation 信号，但不会每次同步调用本地 LLM。

## 候选服务器数据 / Candidate Data

`find-AI-server` 和 `find-AI-python-server` 必须使用当前 NetLogo 已过滤好的 `the-server-set`，不能使用全局 `servers` 集合：

```netlogo
if not any? the-server-set [ report nobody ]
let svrIDs [who] of the-server-set
```

传给 Python 的服务器格式为：

```text
[server_id cpu_free_pct mem_free_pct net_free_pct]
```

服务请求格式为：

```text
[cpu_req_pct mem_req_pct net_req_pct]
```

## 返回值语义 / Return Semantics

Python 入口统一返回整数：

| 返回值 | NetLogo 行为 |
|---|---|
| `>= 0` | `report server sid` |
| `-1` | `report find-balanced-fit-server the-server-set the-service` |
| `-2` | `report nobody` |

这让 Python Agent 可以失败得很明确，同时保留 NetLogo 侧已有的 baseline fallback。

## Phase 2 全局风险 / Phase 2 Global Risk

Phase 2 会额外接收 `global_state_raw`：

```netlogo
py:set "global_state_raw" global-state
set sid py:runresult "schedule_service_phase2(servers_raw, service_raw, global_state_raw)"
```

当前 global state 包含 active CPU/MEM/NET utilization、active server count、auto/consolidation migration、rescheduled services 和当前 SLA violation 信号。Python 侧会计算 `global_risk_score`、`global_risk_level`、`global_risk_tags` 和 `risk_policy`，并把结果写入 trace 与 `hybrid_stats()`。

NetLogo Monitor 可使用：

```netlogo
py:runresult "hybrid_stats_summary_phase2()"
```

## Phase 3 Memory / Phase 3 记忆路径

`find-AI-phase3-server` 通过 `schedule_service_phase3(servers_raw, service_raw)` 调用 Phase 3。Phase 3 会检索历史 episode，构造 `memory_context_raw`，再委托 Phase 2 完成 fast path、复杂度分析或 structured Agent 调度。

## Headless Smoke / Headless 冒烟测试

```powershell
& "$env:NETLOGO_HOME\netlogo-headless.bat" `
  --model ".\cloud_scheduler_agent.nlogo" `
  --setup-file ".\benchmark\netlogo_100tick_smoke.xml" `
  --experiment "agent-100tick" `
  --table -
```

## 常见问题 / Troubleshooting

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: agent_phase1` | NetLogo 工作目录不是项目根目录 | 从项目根目录启动，或把项目根加入 `sys.path` |
| 每次都走 fallback | 模型没有返回有效 tool call | 先运行 `python -m agent_phase1.test_scheduler` |
| Phase 2 速度很快但没有 LLM 调用 | `backend="auto"` 默认 record 模式 | 这是实跑默认行为；demo 时显式使用 `backend="structured"` 或 `hybrid_agent_mode="sync"` |
| `OF ... NOBODY` | NetLogo 路径未检查 `candidate != nobody` | 在 consolidation/migration 逻辑中保留 nobody 防护 |
