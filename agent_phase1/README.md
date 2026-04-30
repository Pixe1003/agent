# Phase 1 / 结构化 Tool Calling 调度器

Phase 1 将 NetLogo 中的自由文本 LLM 调度逻辑改为 Python 侧的结构化 Agent。NetLogo 仍只传入候选服务器和服务请求，Python 返回一个整数：`server_id`、fallback 哨兵值或 reject 哨兵值。

## 设计目标 / Goals

- 用 Pydantic schema 描述服务器状态、服务请求和调度动作。
- 用 `ChatOllama.bind_tools(...)` 尝试原生 tool calling，并对输出做校验。
- 保持 NetLogo 调用面稳定：`schedule_service(servers_raw, service_raw) -> int`。
- 当 LLM 输出不可用时返回 `-1`，让 NetLogo 侧继续使用 `find-balanced-fit-server` 兜底。

## 目录结构 / Layout

```text
agent_phase1/
├── __init__.py
├── schemas.py
├── prompts.py
├── scheduler.py
├── test_scheduler.py
├── requirements.txt
├── README.md
└── NETLOGO_INTEGRATION.md
```

## 公共 API / Public API

| 函数 | 使用时机 | 返回 |
|---|---|---|
| `init_agent(model_name="qwen3:8b")` | NetLogo `setup` 阶段调用一次 | `None` |
| `schedule_service(servers_raw, service_req_raw)` | 每次 placement 调用 | `>=0` 选择服务器，`-1` fallback，`-2` reject |
| `last_decision_summary()` | NetLogo Monitor 或调试输出 | 人类可读摘要 |
| `last_decision_dict()` | Trace、memory 或测试使用 | 最近一次结构化决策 |

## 快速验证 / Quick Check

```powershell
.\.venv\Scripts\python -m agent_phase1.test_scheduler
py -3.13 -m pytest tests/test_phase1_safety_fallback.py -q
```

## 行为说明 / Behavior

- NetLogo 传入的是百分比形式的资源余量，例如 `[server_id cpu_free_pct mem_free_pct net_free_pct]`。
- Python 端校验候选服务器 ID、资源约束和工具调用参数。
- LLM 合法拒绝时返回 `-2`，由 NetLogo 外层拒绝该服务。
- LLM 未初始化、工具调用缺失或输出不合法时返回 `-1`，由 NetLogo 侧 fallback。

## 限制 / Limitations

- 本地小模型的原生 tool calling 可靠性有限，因此 Phase 2 增加了 structured output 路径。
- NetLogo 的 `py:runresult` 是同步调用，真实 LLM 路径只适合低频 demo 或离线 trace。
