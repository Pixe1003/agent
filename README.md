# 多智能体云调度系统 / Multi-Agent Cloud Scheduler

本项目把 NetLogo 数据中心调度仿真扩展为一个分阶段演进的 LLM Agent 调度系统。核心设计不是让 LLM 替代每一次高频 placement，而是把 Agent 放在控制层：常规请求由毫秒级 fast path 处理，复杂场景、策略解释、Critic 校验、历史案例检索和离线 trace 再交给 Agent 模块参与。

## 项目概览 / Overview

- **NetLogo model**：`cloud_scheduler_agent.nlogo`，保留原有仿真、可视化和调度算法，并新增 `AI-phase1`、`AI-phase2`、`AI-phase3` 路径。
- **Phase 1**：`agent_phase1` 保持 NetLogo API 稳定，通过 Pydantic schema 和 Ollama tool calling 进行结构化调度；非法或缺失 tool call 会进入确定性 fallback。
- **Phase 2**：`agent_phase2` 增加 Planner-Scheduler-Critic control plane。默认 `backend="auto"` 使用 hybrid fast path，只记录复杂度和 Agent escalation 信号；`backend="structured"` 或 `hybrid_agent_mode="sync"` 才同步调用本地 LLM。
- **Phase 3**：`agent_phase3` 增加 working memory 与 episodic retrieval，把相似历史调度案例作为 `memory_context_raw` 传给 Phase 2。
- **AIOps Agent**：`agent_aiops` 作为低频监控与策略建议层，读取运行态、hybrid stats、recent decisions 和 memory evidence，输出风险分级、诊断摘要、策略建议与安全 guardrails；不直接替代实时 placement。
- **Benchmark/SFT**：`benchmark.runner` 输出指标 CSV；`dataset.build_sft_dataset` 将 trace 转换为 OpenAI 风格 tool-call SFT 样本。

## 环境配置 / Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r agent_phase1\requirements.txt
ollama pull qwen3:8b
```

如需运行 NetLogo headless，请设置 `NETLOGO_HOME` 指向本机 NetLogo 安装目录：

```powershell
$env:NETLOGO_HOME = "<path-to-netlogo>"
```

## 验证命令 / Verification

```powershell
py -3.13 -m pytest tests -q
.\.venv\Scripts\python -m agent_phase1.test_scheduler
py -3.13 -m benchmark.runner
py -3.13 -m dataset.build_sft_dataset
```

## AIOps 监控与策略建议 / AIOps Monitoring

`agent_aiops` 是 Python-first 的 realtime monitor + advisory agent。默认 `backend="rule"` 不调用 LLM，适合仿真中的每 tick / 每次调度事件监控：

```python
from agent_aiops import init_agent, observe_ops_state, current_alerts, aiops_stats_summary

init_agent(model_name="heuristic", backend="rule")
insight = observe_ops_state(
    {"active_net_util": 0.94, "current_auto_migrations": 8, "net_sla_violations": 1},
    tick=42,
)
print(current_alerts())
print(aiops_stats_summary())
```

`observe_ops_state(...)` 会维护 rolling window、active alerts 和推荐冷却；`analyze_ops_state(...)` 仍保留为单次分析入口。输出包含 `risk_level`、`risk_tags`、`root_cause_summary`、`recommendations`、`guardrails` 和 `evidence`。所有建议默认 `requires_human_approval=True`，并带有 `do_not_auto_apply` guardrail，确保 Agent 只负责监控、诊断和策略建议，不直接修改调度策略。

NetLogo 运行过程中会同步更新 AIOps 状态字段：`aiops-current-risk-level`、`aiops-current-risk-score`、`aiops-current-alert-count`、`aiops-monitor-summary` 和 `aiops-monitor-stats`。界面中的 `AIOps Realtime Risk` 图会随调度过程绘制当前风险分数。

另外，`dashboard/index.html` 提供一个独立的 AIOps realtime dashboard，用服务器矩阵、风险趋势、资源趋势、建议面板和事件流展示监控过程。页面会优先读取真实导出的 `dashboard/aiops-stream.json`，缺失时回退到 `dashboard/sample-aiops-stream.json`。

从 AIOps trace 导出真实 dashboard 数据：

```powershell
py -3.13 -m dashboard.export_aiops_stream --trace-dir traces --output dashboard/aiops-stream.json
```

Live dashboard sync while NetLogo is running:

```powershell
py -3.13 -m dashboard.live_server --trace-dir traces --port 8000
```

Open `http://localhost:8000/`. The live API reads the newest `traces/aiops-*.jsonl` file and returns the latest 500 AIOps events, so the browser no longer depends on a stale `dashboard/aiops-stream.json` export.
NetLogo writes `service_placement_algorithm` into each new AIOps trace row, and the live server defaults to `--algorithm auto`, so the dashboard label follows `AI-phase2` / `AI-phase3` automatically.

## NetLogo 集成 / NetLogo Integration

模型文件在 `setup` 阶段导入 `agent_phase1`、`agent_phase2` 和 `agent_phase3`，并通过 NetLogo Python extension 调用对应的 `schedule_service(...)` 入口。Phase 2 会接收 `global_state_raw`，用于全局风险感知；Phase 3 会检索历史案例并把 memory context 传入 Phase 2。

100 tick headless 冒烟测试示例：

```powershell
& "$env:NETLOGO_HOME\netlogo-headless.bat" `
  --model ".\cloud_scheduler_agent.nlogo" `
  --setup-file ".\benchmark\netlogo_100tick_smoke.xml" `
  --experiment "agent-100tick" `
  --table -
```

模型使用 `py:setup ".\\.venv\\Scripts\\python.exe"`，因此启动 NetLogo 前需要先创建并安装本地虚拟环境。

## 已知限制 / Known Limitations

- 本地 8B LLM 同步调度延迟较高，因此 NetLogo 实跑默认使用 hybrid fast path。
- `backend="structured"` 保留为短 demo、复杂 case 分析和离线 trace 路径，不适合每 tick 高频仿真。
- Phase 3 的 RAG 定位是 retrieval-augmented scheduling memory / case-based reasoning，不是通用文档问答。
