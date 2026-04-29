# 多智能体云调度系统

本项目把 NetLogo 数据中心调度仿真扩展为一个分阶段演进的 LLM Agent 调度系统。

## 当前里程碑

- Phase 1：`agent_phase1` 保持 NetLogo API 稳定，并通过 Pydantic tool schema 调用本地 Ollama 模型。如果模型没有返回 tool call，会启用确定性安全兜底逻辑进行选择或拒绝，并记录 `tool_call_succeeded=False`。
- Phase 2：`agent_phase2` 增加可测试的 Planner-Scheduler-Critic 骨架，并记录 trace。
- Phase 3：`agent_phase3` 增加 working memory 和基于历史调度决策的 episodic retrieval。
- Benchmark/SFT：`benchmark.runner` 输出指标 CSV，`dataset.build_sft_dataset` 将 trace 行转换成 OpenAI 风格的 tool-call SFT 样本。

## 环境配置

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r agent_phase1\requirements.txt
ollama list
```

本地应能看到 `qwen3:8b`。当前机器已确认模型存在，但 tool calling 仍不完全稳定。

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
- Phase 2/3 当前使用确定性调度逻辑。这是有意设计：先让 graph、memory、trace、benchmark 和 SFT 数据流都可测试，再把 scheduler 内部替换为真实的 LangGraph/ReAct LLM 调用。
