# 演示脚本 / Demo Script

1. 打开 `cloud_scheduler_agent.nlogo`，展示原有 NetLogo 仿真环境、服务器、服务请求和 baseline 调度算法。
2. 切换到 Phase 1，说明 NetLogo 通过 `schedule_service(servers_raw, service_raw)` 调用 Python，并展示 `last_decision_summary()`。
3. 切换到 Phase 2 默认 hybrid path，展示常规请求走 fast path，同时复杂请求记录 `agent_escalation_needed`、`global_risk_score` 和 `risk_policy`。
4. 在 NetLogo Monitor 中展示 `py:runresult "hybrid_stats_summary_phase2()"`。
5. 单独运行一个短 Python demo，使用 `backend="hybrid", hybrid_agent_mode="sync"` 或 `backend="structured"`，展示 Planner-Scheduler-Critic structured Agent 决策。
6. 运行 `py -3.13 -m benchmark.runner`，查看生成的 metrics CSV。
7. 运行 `py -3.13 -m dataset.build_sft_dataset`，展示 trace 如何转换为 tool-call SFT 样本。
