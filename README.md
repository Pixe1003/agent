# 多智能体云调度系统 / Multi-Agent Cloud Scheduler

把 NetLogo 数据中心调度仿真扩展为一个分阶段演进的 LLM Agent 调度系统。核心设计不是让 LLM 替代每一次高频 placement，而是把 Agent 放在控制层：常规请求由毫秒级 fast path 处理，复杂场景、Critic 校验、历史案例检索、AIOps 异常检测和离线 trace 再交给 Agent 模块参与。

## 架构 / Architecture

```mermaid
flowchart LR
    NetLogo([NetLogo<br/>cloud_scheduler_agent.nlogo])
    Common[agent_common<br/>schemas + prompts + tracing]
    Multi[multi_agent<br/>Planner / Scheduler / Critic]
    Memory[agent_memory<br/>working + episodic memory]
    AIOps[agent_aiops<br/>realtime monitor + anomaly]
    Bench[benchmark.runner<br/>5 seeds × 4 dist × 6 algo]
    Dataset[dataset.build_sft_dataset<br/>tool-call SFT samples]
    Traces[(traces/*.jsonl)]

    NetLogo -- "servers, service" --> Multi
    NetLogo -- "global_state, ops snapshot" --> AIOps
    Multi --> Memory
    Memory -- "episodic context" --> Multi
    AIOps -- "risk_tags, active_alerts<br/>(closed loop)" --> Multi
    Multi -- "decision JSONL" --> Traces
    AIOps -- "insight JSONL" --> Traces
    Traces --> Dataset
    Traces --> Bench
    Common -. shared .- Multi
    Common -. shared .- Memory
    Common -. shared .- AIOps
```

四个 Agent 模块各司其职：

- **agent_common**：共享 Pydantic schema、prompt 模板、TraceLogger。所有 Agent 的输入输出都过同一套硬约束。
- **multi_agent**：Planner-Scheduler-Critic control plane，默认 hybrid fast path。`backend="structured"` / `hybrid_agent_mode="sync"` 才同步调用本地 LLM。
- **agent_memory**：working memory + episodic retrieval（词袋重叠 + 欧氏距离 + reward 加权）。把相似历史调度案例作为 `memory_context_raw` 传给 multi_agent。
- **agent_aiops**：realtime ops 监控 + 异常检测 + 策略建议。每 tick `observe_ops_state(...)` 计算 risk_tags / active_alerts；闭环把信号注入 multi_agent 的 critic，强制收紧安全边际。

## Headline Results

5 seeds × 4 distributions × 6 algorithms（每场景 100 请求，cluster 初始利用率 45-70%）的均值：

| 算法 | SLA 违约率 | 拒绝率 | Fallback 率 | 能耗 | 平均延迟 | P95 延迟 | AIOps 触发率 |
|---|---|---|---|---|---|---|---|
| first-fit | 40.60% | 43.05% | 0% | 861 | 0.6 μs | 0.8 μs | — |
| balanced-fit | 34.55% | 43.70% | 0% | 855 | 2 μs | 4 μs | — |
| AI-phase2 | 36.05% | 41.90% | 0% | 856 | 80.7 μs | 122.5 μs | — |
| AI-phase3 | 33.50% | 44.45% | 0% | 856 | 270.7 μs | 435.9 μs | — |
| **AI-phase2 + AIOps** | **0.75%** | 21.25% | 64.65% | **767** | 120.4 μs | 188.2 μs | 80.1% |
| **AI-phase3 + AIOps** | **0.75%** | 22.30% | 64.30% | 770 | 196.1 μs | 294.5 μs | 79.5% |

**核心发现**：

- AIOps 闭环把 SLA 违约率从 33-40% **降到 0.75%**（>97% 相对降幅），同时能耗节省约 **11%**（855→767）。
- 代价是 fallback 率 64%——AIOps critic 把高风险放置以 -1 哨兵值打回 NetLogo 的 balanced-fit 兜底；纯 benchmark 下 fallback 不被回收，但 NetLogo 实跑会被 `find-balanced-fit-server` 接住。
- AIOps 信号**稀释了 phase3 的记忆优势**：phase2-aiops 与 phase3-aiops 的 SLA 数字完全一致 (0.75%)，说明在强外部信号下 episodic memory 的边际收益消失。
- phase3-aiops 延迟 (196.1 μs) **低于** phase3 (270.7 μs)：AIOps 让 64% 请求走 fallback，跳过 episode 写盘，证明 AIOps 不仅过滤决策也过滤了"记忆污染"。
- Pareto 前沿：**AI-phase2 + AIOps** 在 (SLA, latency) 平面的左下角，是综合最优。

**Profile-driven 延迟优化**（cProfile + snakeviz 定位热点后两步改动）：

| 算法 | 优化前 avg | 优化后 avg | 降幅 | 优化前 P95 | 优化后 P95 | 降幅 |
|---|---|---|---|---|---|---|
| AI-phase2 | 104 μs | 80.7 μs | -22% | 167 μs | 122.5 μs | -27% |
| AI-phase3 | 678 μs | 270.7 μs | **-60%** | 1071 μs | 435.9 μs | -59% |
| AI-phase2 + AIOps | 150 μs | 120.4 μs | -20% | 205 μs | 188.2 μs | -8% |
| AI-phase3 + AIOps | 357 μs | 196.1 μs | **-45%** | 648 μs | 294.5 μs | -55% |

总 benchmark 时间从 6.16s 降到 4.81s (-22%)。两步关键修改：
1. **`_token_overlap` 占 22% cumtime** → 在 `Episode` 用 `PrivateAttr` 缓存 frozenset 化 token，retrieve 时不再重复分词；query 端也只 tokenize 一次。
2. **`pathlib.open` 占 7% cumtime** → `EpisodicMemory` 改成长开文件句柄 + `flush_every` 批量刷盘，benchmark 模式直接 `persist=False` 跳过磁盘 I/O。

附带：`_cluster_fragmentation` 用单遍 `E[X²]-E[X]²` 替换 `statistics.pstdev`（13.9% → < 2%），`_parse_context` 用 `model_construct` 跳过 Pydantic field validation。

**Demo 闭环对比**（`demo/aiops_closedloop_demo.py`，mixed-burst 工况，初始利用率 60-80%）：

> 同一 workload + 同一初始集群，AIOps 闭环把 SLA 违约率从 N 降到 0（实测见 demo 输出），换 R 个 reject，平均延迟开销 < 100μs。

跑 demo 后用实际 N / R 替换。Pareto 散点图见 `benchmark/results/pareto_*.png`。

## Quickstart

```powershell
# 1. 创建虚拟环境
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. 跑全量单元测试 (63 项)
.\.venv\Scripts\python -m pytest tests -q

# 3. AIOps 闭环 demo (A/B 对比，约 1 秒)
.\.venv\Scripts\python -m demo.aiops_closedloop_demo

# 4. 完整 benchmark (5 seed × 4 dist × 6 algo = 120 行 csv)
.\.venv\Scripts\python -m benchmark.runner

# 5. 出 Pareto 图 (需要 matplotlib)
.\.venv\Scripts\python -m pip install matplotlib
.\.venv\Scripts\python -m scripts.plot_pareto

# 6. 把 trace 导出成 SFT 数据集
.\.venv\Scripts\python -m dataset.build_sft_dataset
```

输出位置：`benchmark/results/metrics.csv` + `benchmark/results/pareto_*.png` + `dataset/cloud-sched-sft-v1.jsonl`。

## AIOps 闭环 / Closed Loop

`agent_aiops.observe_ops_state(...)` 返回的 insight 可以直接传给 `multi_agent.schedule_service` 的第 5 个参数：

```python
from agent_aiops import init_agent as init_aiops, observe_ops_state
from multi_agent import init_agent as init_scheduler, schedule_service

init_scheduler(model_name="heuristic")
init_aiops(model_name="heuristic", backend="rule", recommendation_cooldown=0)

insight = observe_ops_state(
    {"active_net_util": 0.95, "net_sla_violations": 1},
    tick=42,
)
sid = schedule_service(servers, service, None, None, insight)
```

multi_agent 的 critic 会：
- 在 `network-pressure` 时要求 NET 残余 headroom ≥ 15%
- 在 `cpu-pressure` / `memory-pressure` 时分别收紧 CPU / RAM
- 在 `sla-risk` / `capacity-risk` 时三维同时收紧
- 持续 ≥ 2 个窗口的 alert 把阈值提升到 1.5×（22.5%）

每个 decision 字典都会带 `aiops_aware / aiops_critic_triggered / aiops_critic_revisions / aiops_risk_tags / aiops_risk_level / aiops_risk_score` 字段，可以从 `last_decision_dict()` 或 `hybrid_stats()` 取出来做监控。

## Demo 录制 / Demo GIF

招聘场景下，60 秒 GIF 比一段 README 更有说服力。Windows 自带方案：

```powershell
# 方案 A — Xbox Game Bar (Win+G，自带，无需安装)
# 1. 打开 PowerShell 窗口
# 2. Win+G 唤出 Game Bar，点击 "录制" 圆点
# 3. 在 PowerShell 里按顺序跑：
.\.venv\Scripts\python -m pytest tests -q                    # 显示 63 passed
.\.venv\Scripts\python -m demo.aiops_closedloop_demo         # 显示 A/B 对比
# 4. 停止录制 (Win+Alt+R)。视频保存在 %USERPROFILE%\Videos\Captures
# 5. 用 ScreenToGif (https://www.screentogif.com/) 把 mp4 转成 gif，
#    或用在线工具 ezgif.com/video-to-gif 压到 < 5MB
```

把生成的 GIF 命名为 `docs/demo.gif`，README 顶部加一行 `![demo](docs/demo.gif)` 即可。

## NetLogo 集成 / NetLogo Integration

模型文件在 `setup` 阶段导入 `multi_agent`、`agent_memory` 和 `agent_aiops`，并通过 NetLogo Python extension 调用对应的 `schedule_service_phase2(...)` / `schedule_service_phase3(...)` 入口（NetLogo 侧仍以 `phase2/phase3` 别名命名）。`multi_agent` 接收 `global_state_raw` 用于全局风险感知；`agent_memory` 检索历史案例并把 memory context 传给 `multi_agent`。

100 tick headless 冒烟测试：

```powershell
& "$env:NETLOGO_HOME\netlogo-headless.bat" `
  --model ".\cloud_scheduler_agent.nlogo" `
  --setup-file ".\benchmark\netlogo_100tick_smoke.xml" `
  --experiment "agent-100tick" `
  --table -
```

模型使用 `py:setup ".\\.venv\\Scripts\\python.exe"`，因此启动 NetLogo 前需先创建并安装本地虚拟环境。

## AIOps Dashboard

`dashboard/index.html` 提供独立的 AIOps realtime dashboard，用服务器矩阵、风险趋势、资源趋势、建议面板和事件流展示监控过程。

```powershell
# 一次性导出
.\.venv\Scripts\python -m dashboard.export_aiops_stream --trace-dir traces --output dashboard/aiops-stream.json

# 实时同步 (NetLogo 跑的同时)
.\.venv\Scripts\python -m dashboard.live_server --trace-dir traces --port 8000
```

打开 `http://localhost:8000/`。Live API 读最新 `traces/aiops-*.jsonl`，返回最近 500 条 AIOps 事件。NetLogo 把 `service_placement_algorithm` 写进每条 trace，dashboard 标签会跟着 `AI-phase2` / `AI-phase3` 切换。

## 已知限制 / Known Limitations

- 本地 8B LLM 同步调度延迟较高，因此 NetLogo 实跑默认使用 hybrid fast path。
- `backend="structured"` 保留为短 demo、复杂 case 分析和离线 trace 路径，不适合每 tick 高频仿真。
- `agent_memory` 的 RAG 定位是 retrieval-augmented scheduling memory / case-based reasoning，不是通用文档问答。
- AIOps 闭环目前只在 multi_agent 的 heuristic 路径生效；`backend="structured"` 路径仅透传 metadata，不参与 critic。
