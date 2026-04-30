# 多智能体云调度系统

本项目把 NetLogo 数据中心调度仿真扩展为一个分阶段演进的 LLM Agent 调度系统。

当前架构把 Agent 定位为云调度系统的智能控制层，而不是每次 placement 的慢速替代算法。在线高频路径使用毫秒级 fast path；复杂场景、策略解释、Critic 校验、历史案例检索和离线 trace 才交给 Agent 相关模块参与。

## 当前里程碑

- Phase 1：`agent_phase1` 保持 NetLogo API 稳定，并通过 Pydantic tool schema 调用本地 Ollama 模型。如果模型没有返回 tool call，会启用确定性安全兜底逻辑进行选择或拒绝，并记录 `tool_call_succeeded=False`。
- Phase 2：`agent_phase2` 增加 Planner-Scheduler-Critic 流水线。默认 `auto` 后端走 hybrid scheduler：常规请求用毫秒级 fast path，复杂请求记录 Agent escalation 信号；显式 `hybrid_agent_mode="sync"` 时复杂请求可交给 LangGraph + Qwen3 structured Agent。`backend="structured"` 保留为短 demo / 离线 trace 路径。
- Phase 3：`agent_phase3` 增加 working memory 和基于历史调度决策的 episodic retrieval，并把 retrieved episodes 作为 Phase 2 的 memory context 写入复杂度分析、trace 和 structured Agent prompt。
- Benchmark/SFT：`benchmark.runner` 输出指标 CSV，覆盖 `AI-phase2` / `AI-phase3` 的 escalation、memory usage 和 latency 指标；`dataset.build_sft_dataset` 将 trace 行转换成 OpenAI 风格的 tool-call SFT 样本。

## 环境配置

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r agent_phase1\requirements.txt
ollama list
```

本地应能看到 `qwen3:8b`。当前机器已确认模型存在。原生 tool calling 仍不完全稳定，因此 Phase 2 的 LLM 演示路径使用 Pydantic 结构化输出而不是原生 tool call。由于本地 8B LLM 单次同步决策可达数秒到数十秒，NetLogo 实跑默认使用 hybrid 的 fast path。

## 验证命令

```powershell
py -3.13 -m pytest tests -q
.\.venv\Scripts\python -m agent_phase1.test_scheduler
py -3.13 -m benchmark.runner
py -3.13 -m dataset.build_sft_dataset
```

## NetLogo 集成

模型文件现在会在 `setup` 阶段导入 `agent_phase1`，并在 `find-AI-server` 内调用 `schedule_service(servers_raw, service_raw)`。旧的自由文本 LLaMA 实现保留为 `find-AI-server-legacy`，用于后续 benchmark 对照。

当前本机 NetLogo 安装目录是 `D:\NETLOGO`。在项目根目录运行下面的 100 tick headless 冒烟测试：

```powershell
& 'D:\NETLOGO\netlogo-headless.bat' `
  --model 'D:\Users\12057\Desktop\agent\2143512_Jiale Miao_2025_Supplementary.nlogo' `
  --setup-file 'D:\Users\12057\Desktop\agent\benchmark\netlogo_100tick_smoke.xml' `
  --experiment 'agent-100tick' `
  --table -
```

模型使用 `py:setup ".\\.venv\\Scripts\\python.exe"`，因此启动 NetLogo 前需要先创建并安装本地虚拟环境。

## 已知限制

- 已使用 `D:\NETLOGO` 跑通 100 tick 的 `AI-phase2` headless 冒烟实验。
- Phase 2 的 `auto` 后端会计算复杂度并默认记录 `agent_escalation_needed`，但不阻塞调用本地 LLM；`heuristic` 后端只走确定性逻辑；`structured` 后端已接入真实 LangGraph + Qwen3 结构化输出，但只建议用于短 demo 或离线 trace。
- Phase 3 的 RAG 定位是 retrieval-augmented scheduling memory / case-based reasoning，不是通用文档问答。检索结果默认进入 trace 和统计字段；只有复杂 case 或显式 sync/demo 路径才注入 structured Scheduler Agent。
