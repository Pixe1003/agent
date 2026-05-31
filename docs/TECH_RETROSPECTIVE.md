# Technical Retrospective — 项目技术总结

> 把整个项目里**遇到的真实问题、定位过程、修复方案、留下来的工程经验**梳理在一份文档里。每个条目按 **现象 → 诊断 → 修复 → Takeaway** 四段式组织，面试时直接拿来讲。
>
> 这不是炫耀清单，而是工程笔记——很多坑只有亲手踩过才知道，很多决策只有看到对比才能解释清楚。

---

## 一、架构演进：从 phase1/2/3 到 模块化 agent infra

### 1.1 phase1 死代码导致 import 失败

**现象**：项目初版三个 phase 文件夹耦合严重。`agent_phase2/scheduler.py` 引用 `agent_phase1.schemas`，`agent_phase3` 引用 `agent_phase2`，但 phase1 的 `scheduler.py` 源文件被误删（只剩 `.pyc`），整个 import 链断在 lazy-import。

**诊断**：用 `Glob` 扫一遍发现 `agent_phase1/__pycache__/scheduler.cpython-314.pyc` 存在但 `.py` 不见。这是 `git rm --cached` + 没提交源码的典型后果。

**修复**：
- 把 `agent_phase1/schemas.py` 和 `prompts.py` 迁到 `agent_common/`
- phase1 整目录删除
- phase2 / phase3 按功能改名为 `multi_agent` / `agent_memory`
- 同步更新所有 import + NetLogo 集成文件 + 测试断言
- 因为 bash 沙箱不可用，写了 `cleanup_old_phases.ps1` 让用户手动跑

**Takeaway**：
- **Pydantic schema 这种"数据契约"应该放在共享层**，不要让"决策模块"和"数据定义"耦合
- 模块名应该体现**功能**（multi_agent）而不是**演进阶段**（phase2）——阶段名只有写代码的人知道，半年后自己都看不懂
- 没有 bash 也能交付：把删除/重命名操作写成 PowerShell 脚本附在交付里

### 1.2 5 个 agent 同构 API 的设计

**现象**：每加一个 agent（agent_aiops、agent_sft），benchmark、MCP server、CLI 都要重写一遍 init / 调用 / stats 的胶水代码。

**诊断**：缺乏统一的"agent 入口契约"。

**修复**：强制要求每个 agent 暴露三件套：
```
init_agent(model_name, enable_tracing, ...)
schedule_service(servers_raw, service_req_raw, ...)  # 或 observe / infer 等业务方法
last_decision_dict() -> dict
```

之后 benchmark 切换算法、MCP server 包装、Skill 抽象注册全部成为**单点改动**。

**Takeaway**：在多 agent 系统中，**API 同构是最大杠杆**——比单个 agent 的内部优化更重要。设计 framework 的核心工作就是抽象出这种契约。

---

## 二、Performance：profile-driven 优化的方法论

### 2.1 cProfile 推翻直觉：真正的热点不是 Pydantic

**现象**：phase3（memory-augmented scheduler）单次决策延迟 678 μs，远高于 phase2 的 104 μs。直觉认为是 Pydantic schema validation。

**诊断**：跑 `python -m cProfile -o bench.prof -m benchmark.runner` + `snakeviz` 看 sunburst。真热点排序：
1. `agent_memory._token_overlap` (75552 次调用) — **1.34s / 21.7% cumtime** 🔥
2. `agent_memory.retrieve` (4000 次) — 1.74s / 28.2%
3. `pathlib.open / io.open` (1433 次) — 0.42s / 6.8% 🔥
4. `_cluster_fragmentation + statistics.pstdev` (5781 次) — 0.86s / 13.9%

Pydantic 不在前 5。

**修复**：
- **Fix 1（_token_overlap）**：在 `Episode` 类用 `PrivateAttr` 缓存预分词的 `frozenset`，retrieve 时不再 per-episode 重新 tokenize：
  ```python
  class Episode(BaseModel):
      ...
      _cached_tokens: frozenset[str] | None = PrivateAttr(default=None)

      def tokens(self) -> frozenset[str]:
          if self._cached_tokens is None:
              self._cached_tokens = _tokenize(self.state_summary_text)
          return self._cached_tokens
  ```
- **Fix 2（pathlib.open）**：`EpisodicMemory` 改成单次长开文件句柄 + `flush_every` 批量刷盘 + `persist=False` 开关（benchmark 模式直接跳过磁盘）
- **Fix 3（_cluster_fragmentation）**：手写单遍 `E[X²]-E[X]²` 替换 `statistics.pstdev`，跳过类型检查 + list 创建
- **Fix 4（_parse_context）**：用 Pydantic v2 的 `model_construct` 在 trusted hot-path 跳过 field validation（仍保留 int()/float() 类型转换捕获坏输入）

**结果**：phase3 latency 678 → 270 μs (-60%)，benchmark 总耗时 6.16 → 4.81s (-22%)。

**Takeaway**：
- **永远 profile 再优化**——直觉是错的，profile 数据才是对的
- snakeviz 的 icicle/sunburst 视图能帮你看到调用链占比，比单看 tottime 准
- 优化前一定要量化基线（"phase3 678μs"），优化后才能讲故事

### 2.2 statistics.pstdev 比裸 NumPy 慢

**现象**：`_cluster_fragmentation` 用 `statistics.pstdev(values)` 计算标准差，占 13.9% cumtime。

**诊断**：Python 标准库 `statistics` 模块为了支持各种类型（int / float / Decimal / Fraction）做了大量动态类型检查 + list 拷贝。

**修复**：用 `E[X²] - E[X]²` 单遍累加：
```python
n = len(servers) * 3
s = 0.0
sq = 0.0
for server in servers:
    v1, v2, v3 = server.cpu_free_pct, server.ram_free_pct, server.net_free_pct
    s += v1 + v2 + v3
    sq += v1*v1 + v2*v2 + v3*v3
mean = s / n
var = sq / n - mean * mean
return math.sqrt(var) if var > 0.0 else 0.0
```

**Takeaway**：**标准库 ≠ 最优实现**。在 hot path 上手写专用代码经常能比通用库快 5-10×。

### 2.3 Pydantic model_construct 跳过 validation

**现象**：`_parse_context` 每次构造 N+2 个 Pydantic 对象，每个对象都跑 field validation (`ge=0, le=100` 等)，N 台服务器 × 4000 次调度 = 16000+ validations。

**诊断**：输入来自 benchmark 仿真 / NetLogo，类型可信。只需要类型转换（`int(s[0])`, `float(s[1])`）即可，不需要范围校验。

**修复**：trusted hot-path 改用 `model_construct`：
```python
def _parse_context(servers_raw, service_req_raw):
    servers = [
        ServerSnapshot.model_construct(
            server_id=int(s[0]),
            cpu_free_pct=float(s[1]),
            ram_free_pct=float(s[2]),
            net_free_pct=float(s[3]),
        )
        for s in servers_raw
    ]
    ...
```

Bad input 仍会被外层 `try / except (ValueError, TypeError, ...)` 接住，因为 `int()` / `float()` 自己会抛。

**Takeaway**：
- Pydantic v2 的 `model_construct` 是被严重低估的 API
- **可信路径 vs 不可信路径**应该用不同的解析器，混在一起就要付双倍成本

---

## 三、AIOps 闭环：deterministic fallback 的工程哲学

### 3.1 AIOps critic 太严格导致 fallback 率 64%

**现象**：把 AIOps risk_tags 闭环到 multi_agent critic 后，SLA 违约率从 35% 干到 0.75%（-97%，漂亮），但 **fallback 率从 0% 涨到 64%**。

**诊断**：AIOps critic 在 `network-pressure` / `sla-risk` 标签下要求 post-placement headroom ≥ 15%（持续 alert ≥ 22.5%）。但 mixed-burst 工况下大部分候选服务器都达不到这个边际，critic 连续 revise 3 次后退回 fallback。

**修复**：
- 解读上：fallback 在 NetLogo 实跑里会被 `find-balanced-fit-server` 接住，不是真的丢失服务
- benchmark 上：单独统计 fallback 数，简历讲故事时**主动承认 trade-off**："SLA -97% 的代价是 64% 请求降级到 balanced-fit"
- 设计上：保留 `safety_margin` 可调参数，让生产可以根据 SLA budget 收紧 / 放松

**Takeaway**：
- **没有 deterministic fallback 的 LLM agent 一旦上线就是定时炸弹**——hallucination / parse_fail / 越界值 100% 都要有兜底路径
- 简历讲 trade-off 比讲数字漂亮——招聘官见过太多"数字过于完美"的项目，**承认代价反而显得诚实**
- AIOps "advisory only with `do_not_auto_apply` guardrail" 这种设计是 AI Native 容灾的核心范式

### 3.2 RAG 信号稀释效应（论文级别发现）

**现象**：在没有 AIOps 时，phase2 SLA 36%、phase3 SLA 33.5%——记忆模块带来 2.5% 改善；加上 AIOps 后，phase2-aiops 和 phase3-aiops 的 SLA **完全一致都是 0.75%**——episodic memory 的边际收益归零。

**诊断**：AIOps 提供了非常强的外部信号（明确的 risk_tags），critic 已经根据这个信号做了硬决策；此时 episodic memory 的"软建议"被淹没了。

**Takeaway**：
- RAG 不是万能的——**外部 deterministic 信号充足时，RAG 是冗余开销**
- 设计 agent 时要问：**这个 retrieval 在什么场景下真的提供新信息**？
- 这是个面试加分点："我设计了 phase3，跑出来发现它没用——这本身就是个研究结论"

### 3.3 phase3-aiops 反而比 phase3 快

**现象**：phase3 latency 270μs，phase3-aiops 196μs。加了 AIOps 居然更快。

**诊断**：AIOps 让 64% 请求走 fallback，跳过了 EpisodicMemory.add() 的 disk write。fallback 不是成功放置，所以不触发 episode 写盘。

**Takeaway**：AIOps 不仅过滤"决策"还过滤"记忆污染"——只有高质量决策才进 memory 库。这种**反直觉的耦合效应**只有在跑完整 benchmark 才能观察到。

---

## 四、SFT / LoRA：小 LLM 的能力边界

### 4.1 模型学会 90% 语法但只学 30% 约束

**现象**：自蒸馏 Qwen2.5-1.5B（12k SFT 样本 + LoRA r=16 + 3 epoch）在 clean prompt 上 `parse_ok=100% / hallucination=0%`，但在 benchmark 工况下 **per-placement SLA = 56%**（vs critic 的 5%），overload_attempts 达 60%。

**诊断**：模型学会了"输出合法 JSON tool call"（format），但没学会"server 容量必须 ≥ service 需求"（constraint）。这是 imitation learning 的本质局限——trace 是 heuristic 跑出来的，模型只能拟合数据分布，无法超越 ground truth。

**修复**（建议方向，未实施）：
- **Hard-case mining**：从 trace 里按难度（critic revise / fallback / AIOps trigger）加权采样
- **复合 reward**：基于 post-placement headroom + SLA 违约 + 后续 K tick 的迁移率定义 reward，用 reward-weighted SFT 或 DPO 训练
- **加 reject 样本**：当前 dataset 7800 select + 546 reject（**reject 占比 7%**），明显过少；建议 30%+

**Takeaway**：
- **小 LLM (1-3B) 在硬约束推理任务上能力有限**——它擅长格式、上下文理解、模式匹配，不擅长 constraint satisfaction
- 简历讲这个发现的方式："**实证了 LLM 在 sub-ms control plane 不可用，应只用于 analysis plane**"——这是工程判断而不是失败
- 真正的解决方向是 RL，不是更大的 SFT 数据集

### 4.2 SFT 推理 5500 ms vs heuristic 2 μs

**现象**：SFT 单次推理 5.5 秒（CPU），即使加 GPU offload (`n_gpu_layers=20`) 也要 5.0 秒。比 heuristic 慢 250 万倍。

**诊断**：1.5B 模型在 q4 GGUF 下，128 max_tokens 的生成需要 ~100-200 token，每个 token ~50ms。这是 LLM 自回归生成的天花板。

**Takeaway**：
- **LLM 在控制平面延迟代价不可接受是结构性问题**，不是工程问题——再优化都难突破 100ms 级
- 合理用法：把 LLM 放在 **analysis plane / offline pipeline / human-in-loop assistant**，不要塞进 sub-ms control loop
- 这条结论 → 简历"反直觉发现"段，面试加分项

### 4.3 SFT dataset 平衡问题

**现象**：`build_sft_dataset --v2` 跑出 8346 个样本，其中 7800 select + 546 reject。说明 trace 里 reject case 太少。

**诊断**：benchmark 默认 workload (`init_util_range=(45, 70)`) 太宽松，大部分请求都能放下，trace 里 reject 占比天然很低。即使 dataset builder 设了 `max_reject_ratio=0.35`，源数据池里 reject 不够也填不满。

**修复**（建议）：
- benchmark 里加 `--high-pressure` 模式（init_util 60-85% + 大请求），生成更多 reject case
- 或合成数据：人为构造"明显塞不下"的 case 作为 hard negative

**Takeaway**：**SFT 数据集的分布偏差直接限制模型能力**。Reject 路径占 0% 意味着模型永远不会拒绝——这是 1.5B 上跑出来 rejection_rate=0 的根因。

---

## 五、跨平台 / Shell 兼容性坑

### 5.1 PowerShell 单引号也吃 JSON 内 "

**现象**：用 `--inline '{"servers":[[0,80,80,80]]...}'` 调 CLI，Python 报 `json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes`。

**诊断**：PowerShell 5.1 / 7 在某些版本下会把单引号内的 `"` 也剥掉（known issue: `PowerShell argument transmission to native exe is broken`）。Python 实际收到的是 `{servers:[[0,80,80,80]]...}` 没有双引号的非法 JSON。

**修复**：在 CLI 加三个 PowerShell-friendly 入参：
```python
@click.option("--stdin", is_flag=True, ...)         # 优先用
@click.option("--inline-base64", ...)               # 完全绕过 shell quoting
@click.option("--input", ..., encoding="utf-8-sig") # 文件 BOM 兼容
```

PowerShell 用法变成：
```powershell
@'
{"servers":[[0,80,80,80]],...}
'@ | python -m cli.main run --skill X --stdin
```

**Takeaway**：
- **CLI 工具必须把 PowerShell 当一等公民**——尤其在 Windows 开发者占比大的国内
- `--stdin` + here-string 是最稳的传 JSON 方式
- BOM 这种隐形坑要在 `encoding="utf-8-sig"` 一次性解决

### 5.2 PowerShell Out-File 默认带 UTF-8 BOM

**现象**：`@'...'@ | Out-File tmp.json -Encoding utf8` 写出来的文件 Python `json.loads(... encoding='utf-8')` 报 `Unexpected UTF-8 BOM`。

**诊断**：PowerShell 5.1 的 `-Encoding utf8` 默认带 BOM（PowerShell 7 才有 `utf8NoBOM`）。

**修复**：CLI 改用 `encoding="utf-8-sig"` 读文件，自动剥 BOM。

**Takeaway**：跨平台 IO 永远用 `utf-8-sig` 读、`utf-8` 写。

### 5.3 Windows bash workspace 不可用时如何交付

**现象**：迁移过程中需要批量删除/重命名目录，但开发环境的 bash 沙箱启动失败。

**修复**：写 PowerShell 清理脚本（`cleanup_old_phases.ps1`）随项目交付：
```powershell
Remove-Item -Recurse -Force agent_phase1
Remove-Item -Recurse -Force agent_phase2
...
Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
```

**Takeaway**：**工具不可用时不要硬等**——把操作步骤写成可重放的脚本，工程交付的可复现性反而更好。

---

## 六、Benchmark 设计：让数据说真话

### 6.1 初版 benchmark 所有指标都是 0

**现象**：初版 `benchmark/runner.py` 用 `_make_scenario` 生成无状态 workload（每次请求重置 cluster），跑出来所有算法 100% 成功率、0% SLA 违约、能耗都是 20。**指标完全饱和，看不出算法差异**。

**诊断**：
- 集群没有状态累积——每个请求都对着一个干净集群
- `total_energy = selected * 1.0 + rejects * 0.1 + fallbacks * 0.5` 是个 placeholder，实际是 placement 计数
- 请求强度不够 (cpu 5-25% 之类)，永远塞得下

**修复**：
- **有状态集群**：用 `_Cluster` 类持久化 free_pct，每次放置真实扣减；churn (`prob=0.05`) 模拟服务完成释放资源
- **真实能耗模型**：`active_servers × 0.4 + 0.6 × Σutil + 0.5 × stddev`，每 tick 采样累加
- **加压**：`init_util_range=(45, 70)` 初始就有压力；加 `mixed-burst` 分布 30% 突发尖峰
- **默认 100 请求/场景**（初版 20）让 phase3 记忆库见到 workload drift

之后再跑：SLA 违约率从 0% 涨到 35-40%，AIOps 闭环把它干到 0.75%——故事就出来了。

**Takeaway**：
- **Benchmark 不会自动告诉你算法好坏**——压力配错就什么都看不到
- "indicators saturate at 0" 是 benchmark 设计的最常见失败模式
- 加 churn / 状态累积 / 真实 cost model 是基本功

### 6.2 fallback 在 benchmark 被丢，在 NetLogo 被接住

详见 §3.1。Benchmark 里 sid=-1 直接被丢，相当于"放置失败但不扣分"，所以 fallback 率涨到 64% 也不会反映在 SLA 数字上——容易让人误读"AIOps 完全压制 SLA"。

**Takeaway**：报告 benchmark 数字时**必须同时报 fallback 率**，不然结论会被反向解释。

---

## 七、K8s 化的取舍

### 7.1 同进程 vs gRPC 的延迟代价

**现象**：把 multi_agent / memory / aiops 拆成 K8s pod 之后，最简单的方案是每次跨 service 调用走 gRPC——但 gRPC overhead 50-200μs，**multi_agent fast-path 本身才 80μs**。

**诊断**：control plane 的延迟预算被 RPC 吃掉。

**修复（MVP 选择）**：
- 暂时**不拆 service 间通信**——multi_agent 在同进程里直接 import memory 和 aiops
- 用 FastAPI 暴露**对外接口**，内部仍是 Python in-process call
- 留出 `agent_common/runtime.py` 抽象点，将来切 gRPC 时只改一处
- 加 environment variable `MODE=local|grpc|mcp` 让运维可配置

**Takeaway**：
- **设计可扩展性 ≠ 立刻实现可扩展性**——MVP 阶段先用简单方案，留出抽象点
- gRPC 这种重型方案在 sub-ms control plane 里要慎用
- 决策时先看延迟预算，再选通信方式

### 7.2 选 FastAPI 而不是 gRPC 的原因

**理由**：
- MVP 阶段需要快速 demo，FastAPI 5 行起 server，gRPC 要 proto + 代码生成
- 客户端用 `curl` / Postman 即可调试，gRPC 要 grpcurl / BloomRPC
- Prometheus 抓 `/metrics` HTTP endpoint 是约定俗成，gRPC 抓还要 grpc-gateway
- 招聘 JD 写"熟悉 gRPC 优先"但实际工作中 FastAPI 占比高

**长期方向**：核心高频路径切 gRPC + bidirectional streaming，外部 gateway 仍保留 HTTP

**Takeaway**：**框架选型先看场景再看潮流**。FastAPI 在 control plane 不一定输 gRPC——Pydantic 集成、async 原生、自动 OpenAPI 都是加分。

### 7.3 NetworkPolicy default-deny 的工程价值

**现象**：刚写完 Helm chart 时所有 pod 间能互相访问，看不出隔离效果。

**修复**：加 `networkpolicy.yaml`：
- 一条 `default-deny` 拒绝所有 ingress / egress
- 每个 agent 一条 `allow` policy，显式列出谁能调用它（multi_agent 只能被 mcp_server 调，等等）

**Takeaway**：
- **deny-by-default + explicit allow 是 zero-trust 的本质**——而不是某个高大上框架
- 在 K8s 部署里这是 must-have 不是 nice-to-have，**漏配 NetworkPolicy 就是默认全开**

### 7.4 Docker 多 stage 共享 base 减小镜像

**现象**：每个 agent 一个独立 Dockerfile 的话，每个 stage 都要装一遍 Pydantic + FastAPI + LangChain，镜像各自 700+ MB。

**修复**：单个 Dockerfile + 多 target：
- `builder`: 装所有依赖到 venv
- `base`: 共享 venv + agent_common
- `multi_agent` / `agent_memory` / `agent_aiops` / `mcp_server`: 在 base 上加自己的源码
- `sft_builder` + `agent_sft`: 单独 stage（含 llama-cpp-python，镜像 800 MB）

**结果**：base 镜像 200 MB，普通 agent 镜像 220 MB（增量只有自己源码）。

**Takeaway**：multi-stage Docker 的真实价值不是"小"，是**共享层 + 显式依赖边界**。

---

## 八、Testing 基础设施

### 8.1 测试覆盖从 63 涨到 98

| 阶段 | tests count | 新增覆盖 |
|---|---|---|
| 初版 | 63 | schema / memory / aiops / closed-loop / SFT parser |
| + B 档 | 71 | Skill registry + MCP server module + CLI commands |
| + K8s MVP | 98 | FastAPI factory + Prometheus metrics + 4 个 serve.py |

### 8.2 关键 test pattern

**Pattern 1：模块独立可 import**（`test_imports.py`）：
```python
def test_agent_common_schemas_import_without_scheduler_dependencies():
    import agent_common.schemas as schemas
    assert schemas.ServerSnapshot(...).server_id == 1
```
**为什么重要**：保证 schema 层和决策层解耦，防止循环依赖回来。

**Pattern 2：parser 鲁棒性**（`test_sft_parser.py`）：
覆盖 9 种 model output edge case，包括"没有 close tag"、"裸 JSON"、"嵌套大括号"、"未知 tool name"、"arguments 不是 dict"。

**Pattern 3：closed-loop 集成**（`test_aiops_closedloop.py`）：
不只测 component，测**信号穿透**：AIOps observe → risk_tags → schedule_service → critic 收紧 → 最终选不同 server。

**Pattern 4：HTTP server 不依赖 docker**（`test_http_server.py`）：
用 `fastapi.testclient.TestClient` 起 in-process 服务，不真启 uvicorn，速度快且隔离。

**Takeaway**：
- 不同 pattern 覆盖不同 risk：依赖图、parser、集成、网络
- **CI 友好的测试 = 没有外部依赖 + 可并行 + 失败信息明确**

---

## 九、可观测性：从 stats dict 到 Prometheus

### 9.1 hybrid_stats 的演进

**v1**：`hybrid_stats()` 返回 15+ 字段的 dict，让外部代码自己解读

**v2**：CLI 加 `stats` subcommand 一行命令 dump 全部状态

**v3**：FastAPI server 暴露 `/stats` endpoint，外部 HTTP 拉

**v4 (现在)**：`agent_common/server.py` 自动加 prometheus_client exporter：
- `agent_requests_total{agent, endpoint, status}` (Counter)
- `agent_request_latency_seconds{agent, endpoint}` (Histogram)
- `agent_inflight_requests{agent, endpoint}` (Gauge)
- `agent_ready{agent}` (Gauge)

**Takeaway**：observability **不是部署阶段才加，是设计阶段就抽象的 SDK**。每个新 agent 通过 `build_app(agent_name="xxx")` 一行得到完整指标——零额外代码。

### 9.2 LiveProbe / ReadyProbe 的语义差异

**坑**：早期 readinessProbe 直接返回 `200 OK` 不检查模型加载状态，K8s 把流量路过来时 agent_sft 还在加载 GGUF，请求全 500。

**修复**：`build_app(readiness_check=lambda: _LLM is not None)`——readyz 等 GGUF 真正加载完才返回 200。Liveness 只看进程活着，readiness 看业务 ready。

**Takeaway**：
- liveness = "进程还在跑"，readiness = "可以接业务流量"——两个语义不能混
- 慢启动服务（LLM / 大模型 / DB connection pool）必须有独立 readiness check

---

## 十、改进方向（投递后想做的事）

按优先级排：

### 10.1 数据飞轮闭环（高 ROI）

详见 `docs/K8S_DEPLOYMENT_PLAN.md` 的"数据飞轮"段落。三个维度：

**A. Hard-case mining**：`build_sft_dataset` 加 `difficulty_score(record)`，按 `critic_revise / fallback / aiops_trigger` 加权采样难样本

**B. Delayed reward backfill**：benchmark.runner 在 placement 之后 K tick 回看是否引发 SLA 违约，把 reward 回填到 trace；用 reward-weighted SFT 或 DPO 训练

**C. Drift detection auto-retrain**：agent_aiops 加 `distribution_monitor`，KS-test / PSI 比对当前 workload vs 训练集分布，漂移 > 阈值触发 retrain trigger

预期：跑 3 轮循环把 SFT per-placement SLA 从 56% 降到 25%。

### 10.2 K8s 完整 6 phase（中 ROI）

P4 Argo Workflows + P5 Istio + P6 gVisor sandbox。详见 `docs/K8S_DEPLOYMENT_PLAN.md`。

### 10.3 RLHF / DPO（中 ROI）

`dashboard/index.html` 加 Approve / Reject button，操作员对 AIOps recommendation 打标，攒 200-500 对 preference pair → TRL DPOTrainer 微调。

### 10.4 NamedTuple Pydantic 替换（低 ROI）

profile 显示 `_parse_context` 还占 14.5%，进一步用 `typing.NamedTuple` 替换 Pydantic 模型可再降 20-30μs。但**当前 80μs 已远低于 1ms 控制延迟预算**，ROI 不够，**主动放弃**。

### 10.5 Multi-cluster / federation（低 ROI）

跨地域调度、Karmada 多集群——超出个人项目 scope，暂不动。

---

## 十一、整个项目的 5 条核心 takeaways

1. **Profile-driven 优化 > 直觉优化**：用 cProfile + snakeviz 找到的热点经常推翻"我以为是 Pydantic 慢"这种直觉

2. **同形 API 是多 agent 集成的最大杠杆**：5 个 agent 都暴露 `init / call / last_decision`，benchmark / MCP / CLI / Skill / K8s 全部成为单点改动

3. **Deterministic fallback 是 LLM agent 工程必备**：60% hallucination 也不会让系统崩，是因为 strict parser + balanced-fit 兜底——**没有这层就是定时炸弹**

4. **LLM 在 sub-ms 控制平面是错误位置**：1.5B fine-tuned 模型 5500ms 推理慢 25 万倍，per-placement SLA 56%——**实证后再做架构决策**比读论文可靠

5. **强外部信号下 RAG 边际收益归零**：phase2-aiops 与 phase3-aiops SLA 完全一致——**RAG 不是万能的，要看场景**

---

## 十二、给后来人的建议

如果你拿到这个项目想接着做：

1. **先跑 `pytest tests -q` 确认 98/98 pass**——这是 baseline，任何改动必须保持
2. **跑 `python -m cli.main benchmark --algos all` 重新生成 metrics.csv**——确认你的硬件跑出来数字和文档接近
3. **看 `docs/RESUME.md` 的"❀蚂蚁专用版本"** 找简历能写的话术
4. **看 `docs/K8S_DEPLOYMENT_PLAN.md` 的 phase 路径** 决定下一步做哪个
5. **不要再优化 phase2 延迟了**——80μs 已经够好，再榨没意义；优化对象应该是 SLA / 能耗 / fallback 率这种业务指标

最后，**这个项目的所有数字都是真实可复现的**：5 seed × 4 dist × 7 algo benchmark 跑出来的 SLA、profile 跑出来的延迟、SFT 模型跑出来的 parse 率——都在 `benchmark/results/` 和 `dataset/` 里。简历写数字之前先跑一遍验证，是工程素养。
