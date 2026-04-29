# Phase 1 — Tool Calling + Structured Output

这是毕设拓展计划的 Phase 1，目标是把你原来基于字符串拼接和字符串长度判断的
LLM 调度逻辑，重构为一个**基于 Pydantic schema + Ollama 原生 tool calling**
的结构化 agent。

## 与原实现的对比

| 维度 | 原实现 | Phase 1 |
|---|---|---|
| LLM 接口 | `OllamaLLM` (text completion) | `ChatOllama.bind_tools()` |
| 输出验证 | `(length serverID) = 1` 判断字符长度 | Pydantic `ValidationError` 捕获 |
| 非法输出处理 | 模糊的 `show "B"` 走 balanced-fit | 明确分三类：select / reject / fallback |
| Prompt 构造 | NetLogo 端 `word` 拼字符串 | Python 端 Jinja-风格模板 |
| 状态传递 | 字符串列表 + 硬编码位数 | `py:set` + Pydantic schema |
| 幻觉防御 | 只检查 `id < 9 and id > 0` | 动态 `valid_ids` 集合校验 |
| 可观测性 | `show` 打印 | 结构化 `SchedulingDecision` 对象 |
| 调度动作种类 | 1（选 server） | 2（选 server / 拒绝） |

## 目录结构

```
agent_phase1/
├── __init__.py              # 对外暴露的 API
├── schemas.py               # Pydantic 模型：state、tool args、decision
├── prompts.py               # 系统提示 + state 渲染模板
├── scheduler.py             # 主调度逻辑 + NetLogo 入口函数
├── test_scheduler.py        # 脱离 NetLogo 的冒烟测试
├── requirements.txt         # Python 依赖
├── README.md                # 本文件
└── NETLOGO_INTEGRATION.md   # NetLogo 侧改动清单
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r agent_phase1/requirements.txt

# 2. 确保 Ollama 服务在运行，并拉取 qwen3:8b
ollama serve  &
ollama pull qwen3:8b

# 3. 跑冒烟测试
python -m agent_phase1.test_scheduler

# 4. 按 NETLOGO_INTEGRATION.md 改 NetLogo 代码，然后用 NetLogo 打开 .nlogo 文件
```

## 对外 API（给 NetLogo 调用）

| 函数 | 时机 | 返回 |
|---|---|---|
| `init_agent(model_name="qwen3:8b")` | 仿真 setup 时调用 1 次 | None |
| `schedule_service(servers, service_req)` | 每次调度调用 | `int`：>=0 为 server_id，-1 走 fallback，-2 拒绝 |
| `last_decision_summary()` | 可选，显示在监视器 | 人类可读字符串 |
| `last_decision_dict()` | 给 Phase 3 memory 模块用 | dict，SchedulingDecision 的 model_dump |

## 为什么这么设计

### 1. 为什么用 Pydantic 而不是 JSON schema 手写
Pydantic v2 是 LangChain 的一等公民。`bind_tools` 直接吃 Pydantic 类，
会自动生成 Ollama 需要的 tool spec。用 JSON schema 手写要多维护一份
文档，还容易和代码漂移。

### 2. 为什么保留两个 tool（select + reject）而不是只一个
LLM 返回"找不到合适的服务器"时，这是一个**正常的结构化决策**，不是错误。
把它提升为一等 tool，Critic agent（Phase 2）就能直接区分"LLM 合理地
认为应该拒绝"和"LLM 搞砸了"。这两种情况在你原代码里都走同一个分支，
Phase 2 之后就分不清了。

### 3. 为什么 fallback 留在 NetLogo 侧
Python 端做 fallback 需要把 balanced-fit 算法在 Python 里重新实现一遍。
你 NetLogo 里已经有 `find-balanced-fit-server`，再复制一份只会带来维护
负担。Python 只返回一个哨兵 `-1`，NetLogo 自己处理 —— 职责清晰。

### 4. 为什么用 `/no_think` 关掉 Qwen3 的 thinking 模式
Qwen3 默认开 thinking，每个 token 前都会先输出一段 `<think>...</think>`，
对调度这种高频短任务来说延迟太大。Phase 1 只做单轮 tool call，不需要
推理链；Phase 2 引入 ReAct 后会在 Scheduler Agent 里打开。

### 5. 为什么 NetLogo 传百分比而不是绝对值
LLM 对"CPU free 72.3%"的语义感比"CPU free 14.6 GHz"强得多 ——
后者需要上下文才知道这到底算多还是少。同时百分比还让 prompt 里的
threshold 约束（"不要超过 90%"）直接成立。

## Phase 1 不做的事（留给后续）

- Multi-agent / Planner-Critic 拆分 → Phase 2
- ReAct 多轮工具调用 → Phase 2
- 历史决策检索（RAG / memory）→ Phase 3
- 多模型横评 + 50 场景 benchmark → Phase 4
- SFT 数据飞轮 → Phase 5

## 已知限制

1. **Qwen3:8b 的 tool calling 可靠性有上限**。8B 模型不如 GPT-4o 那样几乎
   每次都正确调用 tool。如果 `test_scheduler.py` 第一个 case 失败，通常是
   模型输出了自由文本；可以把 temperature 调到 0，或换 `qwen3:14b`。

2. **首次调用延迟大**。Ollama 冷启动要加载模型，第一次 `schedule_service`
   可能要 10+ 秒。之后稳态 200–500ms。

3. **单线程阻塞**。NetLogo 的 `py:runresult` 是同步的，LLM 慢的时候仿真
   会 freeze。生产化需要改成异步队列，但 Phase 1 不优化。
