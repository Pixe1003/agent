# Demo 脚本

1. 打开 NetLogo 模型，展示旧版 AI 调度器的代码路径。
2. 切换到 Phase 1 agent 初始化逻辑，并运行一段短仿真。
3. 展示 `last_decision_summary()` 如何更新 select/reject/fallback 输出。
4. 切到 `agent_phase2` 的默认 hybrid path，展示常规请求走 fast path，同时复杂请求会记录 `agent_escalation_needed`。
5. 单独用一个短 Python smoke 调用 `backend="hybrid", hybrid_agent_mode="sync"` 或 `backend="structured"`，展示 LangGraph Planner-Scheduler-Critic 如何产生结构化决策，并由 Critic 校验。
6. 运行 benchmark runner，并打开生成的 metrics CSV。
7. 从 trace 行生成 SFT JSONL 文件，并展示其中一条样本。
