# 毕设拓展项目：Multi-Agent 云调度系统 — 规划移交文档

> **文档用途**：这是一份自包含的项目移交文档。读完这份文档，你（Claude）应该
> 能在不访问历史对话的情况下，接手继续完成 Phase 2–5 的工作。
>
> **使用者背景**：Jiale，本科毕设基于 NetLogo 做了一个数据中心云调度仿真平台，
> 现在在找实习（目标岗位：LLM Agent 工程师 / AI 算法工程师），希望把毕设
> 拓展成一个符合岗位关键词的完整项目。

---

## 1. 项目背景

### 1.1 原毕设简介

- **平台**：NetLogo 6.x + Python extension（`py:`）
- **场景**：数据中心服务调度仿真。服务不断到达，scheduler 把服务放到合适的
  server 上，目标是最大化资源利用率 + 最小化 SLA 违反 + 最小化能耗。
- **原有算法**：`First-Fit`、`Balanced-Fit`、`Max-Utilization`、`Min-Power`，
  外加一个简单的 `AI` 模式（LLaMA 3.2 通过文本补全选 server）。
- **原始文件**：`2143512_Jiale_Miao_2025_Supplementary.nlogo`（~7400 行
  NetLogo 代码，已在本地）。

### 1.2 原 AI 模式的问题

原 `find-AI-server` reporter（NetLogo 1173-1226 行附近）的实现有几个硬伤：

```netlogo
let serverID py:runresult (word "chain.invoke({\"question\": \"if we have 10 servers with specs ..." server-results ...)
ifelse (length serverID) = 1 [ ... ]
```

- 巨大的字符串拼接当 prompt
- 用 `length serverID` 判断输出合法性（脆弱）
- 硬编码 `llama3.2:latest`
- 没有工具调用、没有推理轨迹、没有记忆、没有评测
- 对标 LLM Agent 岗位 JD，关键词命中率极低

### 1.3 拓展目标

把毕设重构成一个完整的 Multi-Agent 系统，命中以下 JD 关键词：

- Multi-Agent / LangGraph / Agent 架构
- ReAct / 任务规划 / 工作流编排
- Tool Calling / Function Calling
- Prompt Engineering
- RAG / Embedding / 向量数据库
- Agent Memory / Continual Learning
- Agent 效果验证 / Eval / Benchmark
- SFT / Reward Model / 数据飞轮
- LangChain / LlamaIndex

**最终交付物**：GitHub repo + 对比报告（5-10 页，可投 ML for Systems workshop）
+ demo 视频 + 中文技术博客。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│              NetLogo 仿真环境                            │
│  (Servers · Services · Schedulers · 可视化)              │
└────────────────┬──────────────────────────┬─────────────┘
        state    │                          │   server_id
        + request│                          │
                 ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph 编排层                            │
│                                                          │
│  Planner Agent ──▶ Scheduler Agent ──▶ Critic Agent     │
│  （任务分解）      （ReAct 推理）      （约束校验）       │
│                   （Tool calling）     （启发式回退）     │
└──┬─────────────────────┬────────────────────┬───────────┘
   │                     │                    │
   ▼                     ▼                    ▼
┌────────────┐   ┌──────────────────┐  ┌──────────────────┐
│ Tool Layer │   │ Memory + RAG     │  │ Eval Harness     │
│            │   │                  │  │                  │
│ 结构化函数  │   │ FAISS 向量库     │  │ Benchmark · Trace│
│ Pydantic   │   │ 历史决策检索      │  │ SFT 数据飞轮     │
└────────────┘   └──────────────────┘  └──────────────────┘
```

---

## 3. 五阶段路线图

| Phase | 主题 | 关键交付 | 状态 | 预估时间 |
|-------|------|---------|------|---------|
| 1 | Tool Calling + 结构化输出 | Pydantic schema、bind_tools、替换 find-AI-server | **草稿已完成** | 1-2 周 |
| 2 | LangGraph Multi-Agent | Planner/Scheduler/Critic 三 agent + StateGraph | 待做 | 1-2 周 |
| 3 | Memory + RAG | FAISS + BGE embedding + episodic memory | 待做 | 1 周 |
| 4 | Eval Harness + 多模型横评 | 50 场景 × 5 算法 × N 模型对比 | 待做 | 2 周 |
| 5 | SFT 数据飞轮 | 轨迹筛选 + 格式化 + LoRA 微调 | 待做 | 1-2 周 |

---

## 4. Phase 1：已完成内容

### 4.1 文件结构

```
agent_phase1/
├── __init__.py              # 对外暴露 init_agent / schedule_service 等
├── schemas.py               # Pydantic 模型：state、tool args、decision
├── prompts.py               # 系统提示 + state 渲染模板
├── scheduler.py             # 主调度逻辑 + NetLogo 入口
├── test_scheduler.py        # 脱离 NetLogo 的冒烟测试
├── requirements.txt
├── README.md
└── NETLOGO_INTEGRATION.md   # NetLogo 侧的 3 处改动清单
```

### 4.2 关键设计决策（新 Claude 必读）

1. **模型选型**：Qwen3:8b（本地 Ollama），走 `ChatOllama.bind_tools()` 原生
   tool calling，不再用 `OllamaLLM` 文本补全。
2. **Qwen3 thinking 模式**：Phase 1 用 `/no_think` 关闭（system prompt 顶部），
   换取低延迟。**Phase 2 Scheduler Agent 做 ReAct 时移除此标签**。
3. **两个 tool**：`select_server`（正常选择）和 `reject_service`（明确拒绝）。
   拒绝是一个一等决策，不是错误。
4. **Fallback 留在 NetLogo 侧**：Python 返回哨兵值 `-1`（fallback）/ `-2`
   （reject）/ `>=0`（正常），NetLogo 端用 `find-balanced-fit-server` 兜底。
   不在 Python 里复写 balanced-fit。
5. **数据格式**：NetLogo 传给 Python 的是**百分比**（0-100），不是绝对值。
   LLM 对百分比的语义感更强，且 threshold 约束（"不超过 90%"）直接成立。
6. **错误处理**：所有 `ValidationError`、LLM 异常、幻觉的 server_id 都坍缩到
   fallback 分支，写入 `SchedulingDecision` 结构化记录（给 Phase 3/4 用）。

### 4.3 NetLogo 侧的 3 处改动

1. `setup` 里删除原 `(py:run "from langchain_ollama.llms import OllamaLLM" ...)`
   代码块，替换为：
   ```netlogo
   (py:run
     "import sys, os"
     "sys.path.insert(0, os.getcwd())"
     "from agent_phase1 import init_agent, schedule_service, last_decision_summary"
     "init_agent(model_name='qwen3:8b', temperature=0.1)"
   )
   ```

2. 重写 `find-AI-server` reporter（详见 `NETLOGO_INTEGRATION.md`）：把服务器
   状态打包成 `[[id, cpu_free%, ram_free%, net_free%], ...]` 用 `py:set` 传给
   Python，Python 返回 `int` 哨兵值。

3. 可选：新增 Monitor 显示 `py:runresult "last_decision_summary()"`。

### 4.4 已知问题 / 待验证

- Qwen3:8b 的 tool calling 成功率不是 100%。Jiale 还没在实机跑过
  `test_scheduler.py`，第一步接手工作应该是帮他跑通这个冒烟测试，如果成功率
  太低（<70%），考虑升级到 `qwen3:14b` 或调 temperature=0。
- NetLogo `py:` 是同步阻塞。LLM 慢的时候仿真会 freeze。Phase 1 不优化，
  Phase 4 做 benchmark 时再考虑异步或批量。

---

## 5. Phase 2：LangGraph Multi-Agent（下一步）

### 5.1 设计目标

把 Phase 1 的单 agent 拆成三个角色，用 LangGraph 的 `StateGraph` 编排：

| Agent | 职责 | 输入 | 输出 |
|-------|------|------|------|
| **Planner** | 分析集群整体负载特征，选调度策略 | 全量 cluster state | 策略 tag（"cpu-pressure" / "balanced" / "memory-pressure" / "bursty"）|
| **Scheduler** | 在策略引导下做 ReAct 推理，选出具体 server | 策略 tag + cluster state + service | tool call（select / reject）|
| **Critic** | 重新拉目标 server 的 metrics，验证不会超阈值 | scheduler 的选择 | approve / revise / fallback |

### 5.2 LangGraph State Schema

```python
class AgentState(TypedDict):
    # 输入
    cluster_state: SchedulingContext
    service_request: ServiceRequest
    
    # Planner 输出
    strategy_tag: str | None
    strategy_reasoning: str | None
    
    # Scheduler 输出（可能多轮）
    react_trajectory: list[dict]  # [{thought, tool_name, tool_args, observation}, ...]
    proposed_server_id: int | None
    proposed_reasoning: str | None
    
    # Critic 输出
    critic_verdict: Literal["approve", "revise", "fallback"] | None
    critic_reasoning: str | None
    revise_count: int  # 防止死循环，max=2
    
    # 最终决策
    final_decision: SchedulingDecision | None
```

### 5.3 Graph 结构

```
START ──▶ Planner ──▶ Scheduler ──▶ Critic
                          ▲             │
                          │             ├──▶ [approve] ──▶ END
                          └─[revise]────┤
                                        └──▶ [fallback] ──▶ END
```

- `revise_count` 达到 2 后强制走 fallback 分支，避免死循环。
- Scheduler 内部是一个 ReAct 循环（单独的小 graph 或 `create_react_agent`）。
- Scheduler 此时**打开 Qwen3 的 thinking 模式**（移除 `/no_think`）。

### 5.4 Phase 2 交付物

1. `agent_phase2/` 目录（与 phase1 并存，interface chooser 可切换）
2. 三个 agent 的 prompt 模板
3. LangGraph 图定义 + 可视化（用 `graph.get_graph().draw_mermaid_png()`）
4. 扩展 `test_scheduler.py` 覆盖 revise 和 fallback 两个分支
5. 更新 `NETLOGO_INTEGRATION.md`

### 5.5 Phase 2 关键风险

- **延迟爆炸**：三个 agent + Scheduler 的 ReAct 循环可能让单次调度从 500ms 飙到
  10s+。应对：Planner 和 Critic 用 Qwen3:4b（如果质量够），只有 Scheduler 用
  8b。或对 Planner 做结果缓存（同样的负载特征 5 秒内不重算）。
- **过度工程化风险**：Critic 如果太严格，fallback 率反而升高。对策：第一版
  Critic 只做硬约束检查（"选中的 server 加上 service 后是否超 90%"），不做
  软性优化建议。

---

## 6. Phase 3：Memory + RAG

### 6.1 设计定位（要诚实）

> 这个场景里的"RAG"严格讲不是文档问答意义上的 retrieval-augmented generation，
> 而是**用 RAG 的技术栈来实现 case-based reasoning / episodic memory**。

简历和面试里说话要注意这一点。关键词命中（embedding / 向量库 / 混合检索）是真的，
但别把它说成"做 RAG 问答系统"。

### 6.2 Episode 数据结构

```python
class Episode(BaseModel):
    episode_id: str                    # uuid
    run_id: str
    tick: int
    
    # Retrieval key
    state_summary_text: str            # 给 embedding 用的自然语言
    state_features: list[float]        # 12 维归一化数值向量，给 rerank 用
    
    # 决策内容
    service_request: dict
    action_server_id: int | Literal["reject", "fallback"]
    reasoning_trace: str
    
    # 延迟填充（K tick 后）
    outcome: dict | None = None        # sla_violated, energy, migrated
    reward: float | None = None
```

### 6.3 两级记忆

| 层级 | 存什么 | 实现 | 生命周期 |
|------|--------|------|---------|
| Working memory | 当前 run 最近 5 次决策 | Python list，塞系统提示 | 每次 run 清空 |
| Episodic memory | 所有 run 累计的高奖励 episode | FAISS + pickle 持久化 | 跨 run 保留 |

两级分开是为了消融实验能分开做：关 working memory 看连续性、关 episodic memory
看跨 run 泛化。

### 6.4 混合检索 pipeline

1. 用 BGE-M3 对 `state_summary_text` 做 embedding
2. FAISS 召回 Top-20
3. 用 `state_features` 的欧氏距离做 rerank
4. 返回 Top-3 作为 few-shot

### 6.5 状态摘要模板（关键）

**不要直接 dump JSON 给 embedding**。数值 embedding 效果差。用模板渲染成自然
语言再 embed：

```
Cluster has 10 active servers. CPU utilization is HIGH (mean 78%, std 12%),
memory MODERATE (mean 45%), network LOW (mean 18%). Incoming service is
CPU-bound: needs 20% CPU, 8% RAM, 3% network, expected lifetime 15 min.
```

### 6.6 写入门槛（Reward Gate）

```python
def should_memorize(episode: Episode) -> bool:
    return (
        episode.outcome["sla_violated"] is False
        and episode.outcome["triggered_migration"] is False
        and episode.reward > running_median_reward
    )
```

只留"好"经验。这份高质量轨迹库到 Phase 5 直接复用为 SFT 数据，形成闭环。

### 6.7 冷启动

前 100 个 tick 向量库是空的。策略：
- **方案 A（推荐）**：前 100 tick 用 Balanced-Fit，决策记录进向量库作为 bootstrap。
- **方案 B**：延用原代码 `ticks < 10` 的逻辑，把 10 改成 100。

### 6.8 Phase 3 交付物

1. `agent_phase3/memory.py`：Episode、VectorStore、retrieve、write
2. 集成到 Phase 2 的 Scheduler agent prompt（在 ReAct 之前注入 few-shot）
3. 消融实验脚本：关/开 working memory、关/开 episodic memory 的 4 组对照

---

## 7. Phase 4：Eval Harness + 多模型横评

### 7.1 Benchmark 矩阵

- **10 个随机种子** × **5 种负载分布**（CPU 密集 / 内存密集 / 混合 / 突发 / 长尾）
  = 50 个场景
- **5 种算法**：First-Fit / Balanced-Fit / Max-Utilization / Phase 1 单 agent /
  Phase 2+3 完整 Multi-Agent
- **4 种模型**（只对 agent 类算法）：Qwen3:8b / DeepSeek-V3 / GPT-4o-mini /
  Claude Haiku

注意：完整跑 50×5×4 = 1000 个 run 成本很高。建议分阶段：先 50×5（单模型
Qwen3），拿到基线；再挑典型场景跑多模型。

### 7.2 核心指标

| 指标 | 口径 |
|------|------|
| SLA 违反率 | 所有 service 生命周期内出现资源超阈值的比例 |
| 总能耗 | NetLogo 原有 `sys-power-consumption-total` |
| 拒绝率 | `sys-service-rejection-counter / total-services` |
| 平均调度延迟 | Agent 决策耗时，ms/次 |
| Fallback 触发率 | Phase 1 对比 Phase 2+3 的关键指标 |
| Token 成本 | Input + output tokens per decision |

### 7.3 Trace 记录

所有 tool call 轨迹用 LangSmith 或自己写 jsonl logger 落盘。字段：
```json
{
  "run_id": "...", "tick": 42, "agent": "scheduler",
  "messages": [...], "tool_calls": [...], "latency_ms": 320, "cost_usd": 0.00012
}
```

### 7.4 Phase 4 交付物

1. `benchmark/` 目录：场景生成器、批量跑 NetLogo 的脚本（`netlogo-headless.sh`）
2. 结果汇总 notebook：pandas + seaborn，输出表格 + 曲线图
3. **5-10 页的对比报告**（作为论文/workshop 投稿候选）

---

## 8. Phase 5：SFT 数据飞轮

### 8.1 数据生成

从 Phase 4 的 trace 里筛：
```python
def is_good_sample(trace: dict) -> bool:
    return (
        trace["critic_verdict"] == "approve"      # Critic 一次通过
        and trace["outcome"]["sla_violated"] is False
        and trace["outcome"]["energy"] < median_energy
    )
```

### 8.2 SFT 格式（OpenAI messages 格式）

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "<rendered cluster state>"},
  {"role": "assistant", "content": "", "tool_calls": [{"name": "select_server", "arguments": "..."}]},
  {"role": "tool", "content": "<tool result>"},
  {"role": "assistant", "content": "<final decision>"}
]}
```

### 8.3 训练

Qwen3:8b + LoRA（r=16, alpha=32），训 2 epoch。用 `axolotl` 或 `LLaMA-Factory`。
对比 base 模型的指标提升——哪怕只有几个点，就能写进简历"构建完整 Agent 数据
飞轮：轨迹采集 → 奖励筛选 → SFT → 微调验证"。

### 8.4 Phase 5 交付物

1. `dataset/cloud-sched-sft-v1.jsonl`（目标 500-2000 条）
2. LoRA checkpoint + 训练 log
3. 微调前后的 A/B 对比（用 Phase 4 的 benchmark）

---

## 9. 简历话术映射

做完 5 个 phase，简历可以这样写（每条都对应一个真实可演示的代码模块）：

| JD 关键词 | 简历 bullet |
|-----------|------------|
| Multi-Agent / LangGraph | 基于 LangGraph 设计并实现 Planner-Scheduler-Critic 三智能体云调度系统 |
| ReAct / Tool Calling | Scheduler Agent 采用 ReAct + 结构化 tool calling（Pydantic schema），幻觉导致的 fallback 率从 X% 降至 Y% |
| RAG / 向量数据库 | 集成 FAISS + BGE-M3 embedding，实现混合检索（语义 + 数值 rerank）的历史决策复用 |
| Agent Memory | 设计两级记忆（working + episodic），消融实验证明跨 run 泛化方差降低 X% |
| Eval / Benchmark | 自建 50 场景 × 5 算法 × 4 模型的评测矩阵，输出 SLA/能耗/延迟三维对比 |
| SFT / 数据飞轮 | 从 trace 中筛选高奖励样本构造 SFT 数据集，用 LoRA 在 Qwen3-8B 上验证 |

---

## 10. 新 Claude 接手后的建议工作流

### 10.1 立刻做（30 分钟）

1. 让 Jiale 把 `agent_phase1/` 文件夹放到 `.nlogo` 同级目录。
2. 让他跑 `python -m agent_phase1.test_scheduler`，**这是 Phase 2 能不能推进的
   前提**。
3. 如果 tool calling 成功率太低（<70%），在动 Phase 2 之前先解决：
   - 确认 Ollama 的 Qwen3:8b 版本支持 tool calling（需 `>= 0.3.x` 的 ollama 版本）
   - 尝试 `temperature=0`、`num_predict=128`
   - 实在不行，降级到手写 JSON 输出 + 正则解析（相当于回到字符串但更结构化）

### 10.2 Phase 2 起手（第一个工作 session）

- 先画 LangGraph state machine 图，和 Jiale 对齐
- 写 Planner agent（最简单，是一个 classification）
- 再写 Critic（纯规则 + LLM 兜底）
- 最后写 Scheduler（ReAct 最复杂，放最后）
- 每个 agent 单独写单元测试，**不要**等三个都写完才联调

### 10.3 跨 Phase 的红线

- **永远保留 NetLogo 侧的 `find-balanced-fit-server` 兜底**。LLM agent 再好也
  会有失败的时候。
- **每个 Phase 都要能单独关掉**。Interface 的 `service-placement-algorithm`
  chooser 至少要有：`first-fit` / `balanced-fit` / `AI-phase1` / `AI-phase2` /
  `AI-phase3` / `AI-phase3-sft`。消融实验靠的就是这个 chooser。
- **所有 LLM 调用都记 trace**。Phase 5 的 SFT 数据来源就是这些 trace，Phase 3
  开始就要把记录基础设施打好。

---

## 11. 环境约束 / 本地栈

- OS：未明（NetLogo 能跑就行）
- Python：3.10+ 预期
- Ollama：本地，`qwen3:8b` 已拉取
- NetLogo：6.x，`py` extension 已装
- Jiale 地点：英国爱丁堡（UTC+0/+1）
- 语言偏好：中文交流，代码英文注释（README/文档中英混合均可）

---

## 12. 对外 API 合约（不要改）

这些接口是 NetLogo 代码依赖的，后续 Phase 只能加不能改：

```python
# agent_phase1.__init__ 导出的 API（Phase 2+ 的包需保持同构）
init_agent(model_name: str = "qwen3:8b", ...) -> None
schedule_service(servers_raw: list, service_req_raw: list) -> int
    # 返回值契约：
    #   >= 0 : chosen server_id
    #   -1   : fallback（NetLogo 应调用 balanced-fit）
    #   -2   : reject（NetLogo 应拒绝该服务）
last_decision_summary() -> str
last_decision_dict() -> dict
```

Phase 2 的包应该叫 `agent_phase2`，导出同名函数。NetLogo 通过 chooser 切换
`from agent_phase1 import ...` 或 `from agent_phase2 import ...`。

---

## 附录 A：对话历史摘要（Phase 1 讨论要点）

Phase 1 草稿已在用户本地，关键设计已落到 `agent_phase1/README.md`。几个在
讨论中明确过但未必写进代码的点：

- **Qwen3:8b 的选择理由**：本地、免费、支持 tool calling、参数量足够不幼稚。
  如果生产化可以无缝升级到 DeepSeek-V3 API。
- **为什么不用 GPT-4o 做主力**：成本 + API key 管理 + 毕设重做的自主性。
  Phase 4 benchmark 时再引入做横向对比。
- **Phase 1.5 彩蛋**：建议保留原 `find-AI-server`（llama3.2 版）重命名为
  `find-AI-server-legacy`，interface 加 chooser，这样 Phase 4 对比时就有了
  "重构前 vs 重构后同模型同场景"的天然对照组。

---

**文档版本**：v1.0  
**最后更新**：Phase 1 草稿完成时  
**下一次更新触发**：Phase 2 骨架搭好后
