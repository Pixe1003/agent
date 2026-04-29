# Phase 2 - 规划器/调度器/审查器

这个包保持与 `agent_phase1` 完全一致的 NetLogo 对外 API，同时增加“规划器-调度器-审查器”调度流水线。当前第一版实现是确定性骨架，因此测试、trace、memory 和 benchmark 工具都可以在不依赖本地实时 LLM 的情况下运行。

使用示例：

```python
from agent_phase2 import init_agent, schedule_service

init_agent(model_name="heuristic")
server_id = schedule_service([[0, 80, 80, 80]], [10, 10, 10])
```
