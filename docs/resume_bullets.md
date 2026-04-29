# 简历 Bullet

- 将 NetLogo 云调度仿真器从自由文本 LLM 补全重构为结构化 tool-calling agent，引入 Pydantic 校验以及明确的 fallback/reject 语义。
- 实现“规划器-调度器-审查器”调度流水线，并记录延迟、tool call、fallback 率和决策结果等 trace。
- 构建 memory-augmented 调度原型，结合 working memory 与基于历史调度决策的 episodic retrieval。
- 搭建 benchmark 和 SFT 数据生成工具，支持从 trace 到监督微调样本的 agent 数据飞轮。
