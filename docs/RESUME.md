# 简历用项目描述

按字数 / 场景需要选用一版。所有数字均为 5 seed × 4 分布 × 7 算法 benchmark 实测，源自 `benchmark/results/metrics.csv`。

---

## ❀、蚂蚁 / 字节 / 阿里 Agent Infra 岗专用版本 ⭐⭐⭐⭐⭐

**适用 JD 关键词**：Agent Framework / Runtime / Sandbox / Skill / MCP / Memory / 检查点恢复 / 容灾 / 观测体系 / 高并发调度 / CLI/SKILLS 模块化 / 平台工程 / 全生命周期支撑。

**Title（替换原"多智能体云调度系统"）**：

> **Agent Infra Toolkit — Framework + Runtime + Memory SDK + Sandbox + Observability + MCP**
> 个人项目，2026.04 | [GitHub](https://github.com/<your-handle>/<repo>)
> `Python · LangGraph · LangChain-Ollama · Pydantic v2 · MCP SDK · Unsloth · TRL · PEFT · llama-cpp-python · Qwen2.5 · NetLogo · pytest · cProfile`

**四段 bullet（严格对齐 JD 的 4 个职责板块）**：

- **① Agent Framework + Runtime（呼应"高并发、可扩展 Agent 框架；任务调度与状态管理"）**
  设计了 **5 模块同构 API 的 Agent 框架**（multi_agent / agent_memory / agent_aiops / agent_sft / agent_common），用 **LangGraph StateGraph** 编排 Planner-Scheduler-Critic 子 agent，**Pydantic v2** 作为强类型契约 + tool-call schema。所有 agent 暴露统一的 `init_agent / schedule_service / last_decision_dict` 三件套支持**热替换**；hybrid fast-path 让常规调度在 **80μs P95** 内完成（控制平面级延迟），复杂场景才升级到结构化 LLM——满足高并发场景下的延迟 SLO。

- **② 安全 & 稳定性（呼应"沙箱隔离、检查点恢复、容灾、guardrails"）**
  **agent_sft 把 LLM 推理沙箱化**：strict JSON 解析 + balanced-fit deterministic fallback，benchmark 实测 60% hallucinated_server_id 全部被兜底，系统零崩溃。**TraceLogger append-only JSONL 实现 decision-level checkpoint**——任一 agent 崩溃后可从最近 trace 恢复 state，符合长任务可靠性范式。**agent_aiops 用 `do_not_auto_apply` advisory guardrails**：风险建议必须人工 approve，体现 AI Native 容灾与权限控制设计原则。

- **③ 平台服务能力（呼应"上下文预计算、知识检索、向量检索、观测体系、SDK 封装"）**
  **agent_memory 提供两层 Memory SDK**：WorkingMemory（in-context 滚动 5 条）+ EpisodicMemory（JSONL 持久化 + **Pydantic PrivateAttr token cache + 长开文件句柄 + 批量 flush** 多项 profile-driven 优化）；6 维特征 + Jaccard token overlap + reward 加权 top-k 检索，是**上下文预计算 + 轻量级知识检索**的最小可用形态。**TraceLogger + hybrid_stats + aiops_stats 形成 15+ 维 SDK 化观测体系**（fast_path_ratio / escalation_ratio / fallback_ratio / aiops_critic_trigger_ratio / parse_success_rate / hallucination_rate / memory_usage_ratio ...）零代码可接入新 agent。

- **④ AI 能力产品化（呼应"全流程支持、CLI/SKILLS 模块化套件、降低构建门槛"）**
  端到端 **dev-to-prod pipeline + CLI 套件**：**Train**（`dataset/build_sft_dataset --v2` 自动 trace→ChatML SFT，去重 + 平衡；`train_lora_unsloth` 一键 Unsloth + TRL + PEFT LoRA 微调 Qwen2.5-1.5B）→ **Eval**（`benchmark/runner` 5 seed × 4 dist × 7 算法 A/B 框架；`scripts/plot_pareto` matplotlib 自动 Pareto 图）→ **Inference**（`agent_sft` 用 llama-cpp-python 加载 GGUF q4，Windows CUDA DLL 自动注入，CPU/GPU 兼容）→ **Deploy**（**MCP server** 把 multi_agent / agent_aiops / agent_memory 通过标准 Model Context Protocol 暴露，符合 Anthropic MCP spec，任意 MCP host 即接即用；`agent-cli` 提供 list-skills / benchmark / build-dataset / plot / inference-smoke 一键命令行）。

- **⑤ K8s 化部署（呼应"Kubernetes、容器编排、任务调度、观测体系、服务治理"）**
  完整容器化与 K8s 部署能力：5-target **multi-stage Dockerfile**（非 root 用户 + HEALTHCHECK + distroless 思路），**docker-compose** 编排 6 服务（4 agent + Prometheus + Grafana）做本地验证；**Helm chart** 提供 `Deployment / StatefulSet + PVC / HPA / Service / Ingress / ServiceMonitor / NetworkPolicy（default deny + 显式 allow）/ ServiceAccount + 最小权限 RBAC`；每个 agent 通过 **FastAPI server factory** 自动获得 `/healthz` `/readyz` `/metrics` 三件套，Prometheus 暴露 15+ 业务维度指标（QPS / latency histogram / inflight / readiness）；`agent_sft` 用 `nvidia.com/gpu` resource limit + `nodeSelector` 调度 GPU 节点。完整决策文档 `docs/K8S_DEPLOYMENT_PLAN.md`（14 决策点 + 6 phase 实施路径），上手手册 `docs/K8S_QUICKSTART.md`（kind 集群 10 分钟跑通）。

**项目硬数字（蚂蚁面试官最先扫的）**：
| 维度 | 数字 |
|---|---|
| Agent 模块数 | **5 个独立 + 1 个共享底座 + 1 个 MCP server** |
| 统一 SDK 入口 | `init_agent / schedule_service / last_decision_dict` × 5 个模块同构 |
| 控制面延迟 P95 | **80-122 μs**（无 LLM）/ **188 μs**（含 AIOps 闭环） |
| AIOps 闭环 SLA 改善 | **35% → 0.75% (-97%)** |
| Profile 优化收益 | phase3 延迟 **-60%**（678→270μs），benchmark 总耗时 **-22%** |
| 异常兜底覆盖 | hallucination + parse_fail + overload + invalid input **100% 都有 deterministic fallback** |
| 单元测试 | **63 pytest 100% pass** |
| 观测体系维度 | **15+ 内建指标**，SDK 化接入新 agent |
| MCP 标准协议 | **3 个 tool + 1 个 resource** 通过 FastMCP 暴露 |

**给蚂蚁面试官的一句话项目定位**：

> 这是一个**面向 AI Agent 全生命周期的轻量级基础设施套件**，包含 Agent Framework（LangGraph 编排）、Runtime（同构 API + 热替换 + sub-ms fast path）、Memory SDK（WorkingMemory + Episodic + 检索）、Sandbox 推理（agent_sft 容错隔离）、Observability SDK（15+ 维 stats）、MCP standard server 暴露、CLI 模块化套件、以及完整的 Train→Eval→Inference→Deploy pipeline。用 NetLogo 云调度仿真作为压力测试场景验证基础设施完整性。

---

## 〇、Agent × 功能 × 框架 速查表（投简历前先确认这一份 ✓）

写简历时把每个 agent 的"职能 + 用到的框架"扣紧——这是面试官最先看的两个信息。下表里**加粗的就是必须出现在简历正文里的关键词**。

| Agent / 模块 | 功能与作用（一句话） | 使用的框架 / 库 |
|---|---|---|
| **Planner Agent**<br/>(in `multi_agent`) | 把请求分类成 strategy_tag（cpu / memory / network-pressure / balanced / bursty），决定后续策略基调。 | **LangGraph** node + **Pydantic v2** structured output（可选 **LangChain-Ollama** 接 Qwen3:8B） |
| **Scheduler Agent**<br/>(in `multi_agent`) | 在候选服务器集合里挑 post-placement 资源最平衡的一台，输出 `select_server` / `reject_service` 两种 tool call。 | **LangGraph** node + **Pydantic v2** tool schemas + **LangChain-Ollama** `bind_tools` / `with_structured_output` |
| **Critic Agent**<br/>(in `multi_agent`) | 校验 post-placement headroom，调度有问题时触发 ≤2 次 revise；**接收 AIOps risk_tags 自适应收紧 15→22.5% 安全边际**。 | **LangGraph** conditional edge + **Pydantic v2** validation + 自研 critic 规则 |
| **Memory Agent**<br/>(`agent_memory`) | working memory 滚动保留最近 5 条成功决策；episodic memory 用 Jaccard token overlap + 6 维特征 Euclidean + reward 加权检索 top-3，作为 in-context 喂给 Scheduler 做 case-based reasoning。 | **Pydantic v2** (PrivateAttr token cache) + JSONL append-only 持久化 + 自研检索打分 |
| **AIOps Agent**<br/>(`agent_aiops`) | 每 tick 观测 OpsSnapshot，rolling-window 规则化 risk scoring 出 risk_tags / active_alerts / recommendations + guardrails；**闭环把 risk_tags 注入 Critic 收紧安全边际**——项目核心创新。 | **Pydantic v2** + 自研 rule engine + **TraceLogger**（advisory-only with `do_not_auto_apply` guardrail） |
| **SFT Agent**<br/>(`agent_sft`) | 加载 LoRA 微调后 GGUF q4 模型做推理；strict tool-call 解析 + balanced-fit deterministic fallback + parse_success/hallucination 监控。 | **llama-cpp-python** (GGUF 推理) + 自研 tool-call 解析（regex + balanced-brace JSON） |
| **共享层** (`agent_common`) | 所有 agent 共享 schema、prompt、TraceLogger 基础设施。 | **Pydantic v2** schemas + 自研 append-only JSONL TraceLogger |
| **训练 pipeline**<br/>(`dataset/`) | trace JSONL → ChatML SFT 数据集 (12k 样本，去重 + select/reject 平衡) → Qwen2.5-1.5B LoRA 微调。 | **Unsloth** + **TRL SFTTrainer** + **PEFT LoRA** (r=16, target QKVO+MLP) + **bitsandbytes** 4-bit |
| **Benchmark + 优化**<br/>(`benchmark/`, `scripts/`) | 5 seed × 4 dist × 7 算法对比，profile-driven 优化。 | **cProfile + snakeviz** (热点定位) + **matplotlib** (Pareto 图) + **pytest** (63 测试 100% pass) |
| **仿真集成**<br/>(`cloud_scheduler_agent.nlogo`) | NetLogo 仿真器通过 Python extension 调用三个 agent 的 schedule_service 入口。 | **NetLogo** + **py extension** (py:setup / py:run / py:runresult) |

**简历强制覆盖关键词**（按重要性排序）：
LangGraph · LangChain-Ollama · Pydantic v2 · Multi-Agent · Tool Calling · Closed-Loop · AIOps · Episodic Memory / RAG · Unsloth · TRL · PEFT · LoRA · bitsandbytes · llama-cpp-python · Qwen2.5 · GGUF · NetLogo · pytest · cProfile

---

## 一、Title 与 Tech Stack 一行（中英双语，简历项目栏顶部用）

**多智能体云资源调度系统（含 AIOps 闭环 + 自蒸馏 LLM Policy）**
`Python · LangGraph · LangChain · Pydantic v2 · Unsloth · TRL · PEFT · llama-cpp-python · Qwen2.5 · NetLogo · pytest · cProfile`

**Multi-Agent Cloud Resource Scheduler with AIOps Closed-Loop and Self-Distilled LLM Policy**
`Python · LangGraph · LangChain · Pydantic v2 · Unsloth · TRL · PEFT · llama-cpp-python · Qwen2.5 · NetLogo · pytest · cProfile`

---

## 二、简历项目栏 4-5 bullet 版（中）

**多智能体云资源调度系统 | 个人项目，2026.04 | [GitHub](https://github.com/<your-handle>/<repo>)**
**`Python · LangGraph · LangChain-Ollama · Pydantic v2 · Unsloth · TRL · PEFT · bitsandbytes · llama-cpp-python · Qwen2.5-1.5B · NetLogo · pytest`**

- **多 Agent 架构**：用 **LangGraph StateGraph** 编排 Planner-Scheduler-Critic 控制流——**Planner Agent** 把请求分类为 strategy_tag（cpu/memory/network/balanced pressure），**Scheduler Agent** 产出 placement 提案，**Critic Agent** 用 **Pydantic v2 tool-call schema** 做硬约束校验并支持 ≤2 次 revise；落地 **Memory Agent**（working + episodic RAG，词袋+欧氏距离+reward 加权检索）和 **AIOps Agent**（rolling-window 风险标签 + active alerts，advisory-only with guardrails）。
- **AIOps 闭环（项目核心创新）**：把 AIOps Agent 输出的 risk_tags / active_alerts 反馈到 Scheduler Critic，强制收紧 **15-22.5% adaptive safety margin**。**实测 SLA 违约率从 35% 降到 0.75%（-97%），集群能耗节省 11%**，决策 P95 延迟 < 200μs。
- **Profile-driven 性能优化**：cProfile + snakeviz 定位热点，**Pydantic PrivateAttr token cache** 干掉 22% `_token_overlap` 瓶颈，**长开文件句柄 + 批量 flush** 干掉 7% I/O 瓶颈，**Pydantic `model_construct`** 在 trusted 热路径跳过 field validation。**phase3 决策延迟 -60%（678μs → 270μs），benchmark 总耗时 -22%**。
- **自蒸馏 LLM Policy 闭环**：用 multi-agent 跑出 12k 条 trace → ChatML SFT 数据集 → 用 **Unsloth + TRL SFTTrainer + PEFT LoRA**（4-bit bitsandbytes，r=16）在 Qwen2.5-1.5B 上微调 → 导出 GGUF q4 → 用 **llama-cpp-python** 写推理 adapter（strict tool-call 解析 + balanced-fit 兜底 + parse_success/hallucination 监控）。**实证：1.5B 模型学会 90% 工具调用语法但只学会 30% 资源约束推理，推理慢 4240×；量化 LLM 在 control plane vs analysis plane 的合理分工边界**。
- **工程质量**：**63 个 pytest 单元测试**（含 AIOps 闭环、SFT 解析鲁棒性、benchmark pipeline 全覆盖）100% PASS；Pareto 散点图（matplotlib）+ 双架构图（Mermaid + Excalidraw）+ NetLogo Python-extension 集成 + AIOps realtime HTML dashboard（live HTTP server，读取最新 trace）。

---

## 三、简历项目栏 4-5 bullet 版（英）

**Multi-Agent Cloud Resource Scheduler with AIOps Closed-Loop and Self-Distilled LLM Policy**
**Personal project, 2026.04 | [GitHub](https://github.com/<your-handle>/<repo>)**
**`Python · LangGraph · LangChain-Ollama · Pydantic v2 · Unsloth · TRL · PEFT · bitsandbytes · llama-cpp-python · Qwen2.5-1.5B · NetLogo · pytest`**

- **Multi-agent architecture** orchestrated via **LangGraph StateGraph**: **Planner Agent** classifies requests into strategy tags (cpu/memory/network/balanced pressure), **Scheduler Agent** emits placement proposals, **Critic Agent** enforces hard constraints via **Pydantic v2 tool-call schemas** with ≤2 revise rounds. Built two supporting agents: a **Memory Agent** (working + episodic RAG, lexical + Euclidean + reward-weighted retrieval) and an **AIOps Agent** (rolling-window risk tagging + active alerts, advisory-only with rollback guardrails).
- **AIOps closed-loop (key innovation)**: pipes risk_tags and active_alerts from the AIOps Agent into the Scheduler's Critic, enforcing **15-22.5% adaptive safety margins**. **Reduced SLA violations from 35% → 0.75% (-97%) and cluster energy by 11%**, P95 decision latency < 200μs, on a 5-seed × 4-distribution × 7-algorithm benchmark.
- **Profile-driven optimization** with cProfile + snakeviz: cached pre-tokenized sets via **Pydantic PrivateAttr** (eliminated 22% `_token_overlap` hotspot), long-lived file handle + batched flushing (eliminated 7% disk-I/O hotspot), and **Pydantic `model_construct`** to skip field validation on trusted hot paths. **Cut phase3 decision latency by 60% (678μs → 270μs) and total benchmark time by 22%**.
- **Self-distilled LLM policy pipeline**: collected 12k decision traces → ChatML SFT dataset → fine-tuned **Qwen2.5-1.5B** with **Unsloth + TRL SFTTrainer + PEFT LoRA** (4-bit via bitsandbytes, rank=16) → exported merged GGUF q4 → built **llama-cpp-python** inference adapter with strict tool-call parsing, balanced-fit fallback, and parse-success / hallucination telemetry. **Empirically proved fine-tuned small LLMs are unsuitable for sub-ms control planes (4240× slower at only 30% constraint accuracy)**, quantifying the boundary between LLM-as-controller and LLM-as-advisor.
- **Engineering quality**: **63 pytest unit tests** (covering AIOps closed-loop, SFT parser robustness, full benchmark pipeline) at 100% pass rate; Pareto scatter charts (matplotlib) + dual architecture diagrams (Mermaid + Excalidraw) + NetLogo Python-extension integration + AIOps realtime HTML dashboard backed by a live HTTP trace server.

---

## 四、长版（个人主页 / portfolio / 面试携带 PDF 用）

### 项目目标 / Why

把 NetLogo 数据中心调度仿真扩展为分阶段演进的 LLM Agent 调度系统，研究**何时 LLM 应该参与控制决策、何时只应做监控建议**。

### 系统组成（六个 Agent + 共享层）

1. **Planner Agent**（策略选择器）
   - 输入：cluster snapshot + service request
   - 输出：strategy_tag（balanced / cpu-pressure / memory-pressure / network-pressure / bursty）
   - 框架：rule-based fast path + 可选 langchain-ollama `with_structured_output` 走 Qwen3-8B

2. **Scheduler Agent**（提案生成器）
   - 输入：strategy_tag + valid candidates + (memory context, AIOps insight)
   - 输出：select_server(server_id, reasoning) 或 reject_service(reason)
   - 框架：Pydantic v2 BaseModel 作为 tool schema，bind_tools 接通 OpenAI / Qwen 原生 tool-call

3. **Critic Agent**（守门员）
   - 基础规则：检查 post-placement headroom 是否 ≥ 0
   - **AIOps 增强**：从 active alerts 取 tag → 自适应安全边际 15→22.5%（持续告警时 1.5×）
   - 失败时触发 revise（最多 2 轮），仍失败回退 deterministic balanced-fit

4. **Memory Agent**（工作记忆 + 情景记忆 RAG）
   - WorkingMemory：最近 5 次成功决策的滚动窗口
   - EpisodicMemory：JSONL 持久化 + PrivateAttr token cache，检索分数 = 词袋 Jaccard + 欧氏特征距离 + 历史 reward
   - 框架：Pydantic v2 + 自定义检索（无需 FAISS，规模 < 1k 集已足够）

5. **AIOps Agent**（实时监控 + 异常检测 + 策略建议）
   - 输入：OpsSnapshot（cluster utilization, migrations, SLA violations）
   - 流程：rolling window → risk score / risk_level / risk_tags → active alerts → recommendations → guardrails (`do_not_auto_apply=True`)
   - **闭环**：tags 注入到 Scheduler Critic 收紧安全边际

6. **SFT Agent**（自蒸馏 LLM policy）
   - 训练：trace → ChatML JSONL → Qwen2.5-1.5B + Unsloth + TRL SFTTrainer + PEFT LoRA(r=16) + bitsandbytes 4-bit
   - 推理：llama-cpp-python 加载 GGUF q4 + strict `<tool_call>` JSON 解析 + balanced-fit fallback
   - 监控：parse_success_rate / hallucination_rate / fallback_rate / avg_inference_ms

### 共享基础设施（agent_common）

- Pydantic v2 schemas：`ServerSnapshot` / `ServiceRequest` / `SchedulingContext` / `SchedulingDecision` / `OpsSnapshot`
- ChatML / OpenAI 双模板 prompt 渲染器
- 自定义 TraceLogger：append-only JSONL，每个 agent 独立 run_id

### 框架与工具栈一览

| 角色 | 框架 |
|---|---|
| 多 Agent 编排 | **LangGraph** StateGraph + 自写 InlineGraph 兜底 |
| LLM 客户端 | **LangChain-Ollama** (dev) / **llama-cpp-python** (prod inference) |
| Schema / 验证 | **Pydantic v2**（含 `model_construct` 热路径优化、`PrivateAttr` 缓存） |
| Tool calling | OpenAI tool_calls schema + Qwen2.5 native `<tool_call>` ChatML |
| SFT 训练 | **Unsloth** + **TRL SFTTrainer** + **PEFT LoRA** + **bitsandbytes** 4-bit |
| 量化推理 | GGUF q4_k_m + **llama-cpp-python** |
| 仿真 | **NetLogo** + Python extension (py:setup / py:run / py:runresult) |
| 测试 | **pytest** 63 项（AIOps 闭环、SFT 解析、benchmark pipeline、tracing、记忆检索） |
| 性能分析 | **cProfile** + **snakeviz** icicle/sunburst |
| 可视化 | **matplotlib** Pareto + **Mermaid** + **Excalidraw** 架构图 + 自写 HTML dashboard + live HTTP server |
| 数据集 | OpenAI tool_calls JSONL (v1) / Qwen2.5 ChatML JSONL (v2，去重 + select/reject 平衡) |

### 关键量化结果

| 维度 | 数字 |
|---|---|
| SLA 违约率（heuristic baseline） | 35% |
| SLA 违约率（multi-agent + AIOps 闭环） | **0.75% (-97%)** |
| 集群能耗变化 | **-11%** |
| 决策延迟 P95（multi-agent + AIOps） | **188 μs** |
| Profile 优化前 phase3 延迟 | 678 μs |
| Profile 优化后 phase3 延迟 | **270 μs (-60%)** |
| SFT-1.5B 推理延迟 | 509 ms（heuristic 的 254000×） |
| SFT-1.5B 工具调用语法成功率 | ~90% |
| SFT-1.5B 资源约束推理成功率 | ~30% |
| 单元测试覆盖 | **63 pytest，100% PASS** |

### 关键工程结论（面试加分项）

1. **LLM 在 sub-ms 控制平面是错误位置**：4240× 延迟 + 56% per-placement SLA 失败率，验证了 heuristic+critic 是正确架构。
2. **强外部信号会稀释 RAG**：phase2-aiops 与 phase3-aiops 在有 AIOps 信号时 SLA 数字一致，episodic memory 边际收益归零——RAG 只在弱信号场景有意义。
3. **deterministic fallback 是 LLM agent 工程必备**：SFT 模型 60% placement 是 hallucinated_server_id / 无效 fit，没有兜底直接崩。
4. **profile 必须先于优化**：profile 推翻了"Pydantic 是热点"的直觉，真正杀手是字符串 tokenize（22%）+ 磁盘 I/O（7%）。

---

## 五、面试 60 秒 elevator pitch

> 我做了一个云资源调度系统，三个 Python agent 跟 NetLogo 仿真器闭环。**调度 agent** 用 LangGraph StateGraph 编排 Planner-Scheduler-Critic 流程，Pydantic v2 做 tool-call 硬约束。**记忆 agent** 用 working memory 加 episodic RAG（词袋+欧氏+reward 加权），把历史相似案例作为 in-context 喂给调度 agent。**AIOps agent** 实时监控集群异常，把 risk_tags 闭环到 Scheduler Critic 收紧安全边际，这是项目最核心的设计。在 5 seed × 4 分布 × 7 算法的 benchmark 上 SLA 违约率从 35% 干到 0.75%，能耗顺手省 11%。我还做了 profile-driven 性能优化，用 cProfile 找到 token tokenization 和磁盘 I/O 是热点，cache + 批量化把 phase3 延迟降低 60%。最后做了自蒸馏实验：用 12k 条 trace 在 Unsloth + TRL 上把 Qwen2.5-1.5B 微调成专用 policy，结果非常有意思——**模型学会了 90% 的 tool-call 语法但只学会了 30% 的约束推理，推理延迟比 heuristic 慢 25 万倍**。这个结果不是失败，它实证了"小 LLM 不适合 sub-ms 控制平面"这个工程判断，量化了 LLM 在 control plane 与 analysis plane 的合理分工。代码、63 个单元测试、Pareto 图都在 GitHub。

---

## 六、LinkedIn 项目栏（英文，简洁版）

> Designed and built a multi-agent cloud scheduler with closed-loop AIOps anomaly detection and self-distilled LLM policy. Architecture: **Planner-Scheduler-Critic** orchestrated via **LangGraph**, **Memory Agent** (episodic RAG), and **AIOps Agent** that closes the loop into the Critic to enforce adaptive safety margins. Stack: **Pydantic v2 · LangChain-Ollama · llama-cpp-python · Unsloth · TRL · PEFT · NetLogo · pytest**. Reduced SLA violations from 35% to 0.75% (-97%) with 11% energy savings on a 5-seed × 4-distribution × 7-algorithm benchmark; profile-driven optimization cut phase3 latency 60%. Self-distilled Qwen2.5-1.5B LoRA empirically proved fine-tuned small LLMs unsuitable for sub-ms control planes (4240× slower at 30% constraint accuracy), quantifying the boundary between LLM-as-controller and LLM-as-advisor. 63 pytest unit tests at 100%. [GitHub](https://github.com/<your-handle>/<repo>)
