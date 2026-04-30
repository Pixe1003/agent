# 项目亮点 / Project Highlights

- 将 NetLogo 云调度仿真器从自由文本 LLM 补全重构为结构化 tool-calling Agent，引入 Pydantic 校验和明确的 select/reject/fallback 语义。
- 设计 hybrid Planner-Scheduler-Critic control plane：在线路径保持毫秒级 fast path，复杂场景记录 Agent escalation 信号，并支持 structured Agent 接管少量复杂 case。
- 引入全局风险感知和 memory-aware scheduling，通过 trace 记录 `global_risk_score`、`risk_policy`、`retrieved_episode_count` 和 Agent 参与度指标。
- 搭建 benchmark 与 SFT 数据构建流程，将调度 trace 转换为可复用的 tool-call 训练样本。
