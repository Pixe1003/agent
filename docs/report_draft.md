# 多智能体云调度系统报告草稿

## 背景

本项目把 NetLogo 数据中心调度仿真扩展为一个 Agent 调度系统，包含 tool calling、规划器-调度器-审查器协作、memory、trace、评测以及 SFT 数据闭环。

## 方法

实现上保留确定性基线算法，并让多个 LLM-agent phase 与基线并存。Phase 1 引入结构化 tool calling。Phase 2 增加规划器-调度器-审查器流水线。Phase 3 增加 working memory 和 episodic memory。

## 实验计划

对比 First-Fit、Balanced-Fit、Legacy AI、AI-phase1、AI-phase2 和 AI-phase3，并观察 SLA 违反率、能耗代理指标、拒绝率、fallback 率和平均决策延迟。

## 当前限制

Phase 2/3 第一版实现仍是确定性骨架。这样设计是为了先让数据流可测试，再把 scheduler 步骤替换成真实的 LangGraph/ReAct LLM 后端。
