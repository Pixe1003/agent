# 多智能体云调度系统报告草稿

## 背景

本项目把 NetLogo 数据中心调度仿真扩展为一个 Agent 调度系统，包含 tool calling、规划器-调度器-审查器协作、memory、trace、评测以及 SFT 数据闭环。系统不让 LLM 替代每一次高频 placement，而是把 Agent 放在控制层：常规请求快速调度，复杂请求升级分析，历史经验通过 retrieval-augmented scheduling memory 复用。

## 方法

实现上保留确定性基线算法，并让多个 LLM-agent phase 与基线并存。Phase 1 引入结构化 tool calling。Phase 2 增加 hybrid Planner-Scheduler-Critic control plane：fast path 负责在线执行，复杂度分析记录 Agent escalation 信号，显式 sync/demo 路径才调用 structured Agent。Phase 3 增加 working memory 和 episodic memory，并把 retrieved episodes 作为 Phase 2 的 memory context 注入 trace、复杂度分析和 structured Scheduler prompt。

## 实验计划

对比 First-Fit、Balanced-Fit、Legacy AI、AI-phase1、AI-phase2 和 AI-phase3，并观察 SLA 违反率、能耗代理指标、拒绝率、fallback 率、平均决策延迟、Agent escalation rate 和 memory usage rate。

## 当前限制

Phase 2 已接入 hybrid scheduler：常规请求使用确定性 fast path，复杂请求记录 Agent escalation 信号，并可在显式开启时交给 LangGraph + Qwen3 Pydantic 结构化输出后端。由于本地同步 LLM 决策延迟较高，实验设计上将结构化 LLM 后端用于短 demo、复杂 case 分析和离线 trace，而高频仿真默认使用 hybrid fast path。Phase 3 当前实现的是调度案例检索，不是文档问答式 RAG；检索结果默认进入统计和 trace，复杂 case 才作为 few-shot context 注入 Agent。
