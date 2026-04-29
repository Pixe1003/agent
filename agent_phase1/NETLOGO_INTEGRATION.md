# NetLogo 侧改动清单

Phase 1 需要改三处 NetLogo 代码。所有 Python 逻辑都已经下沉到 `agent_phase1/`
这个包，NetLogo 只负责传数据、收 int。

---

## 改动 1：`setup` 里替换 LLM 初始化

找到你原 setup 里这一段：

```netlogo
(py:run
  "from langchain_core.prompts import ChatPromptTemplate"
  "from langchain_ollama.llms import OllamaLLM"
  "template = \"Question: {question} Answer: value only.\""
  "prompt = ChatPromptTemplate.from_template(template)"
  "model = OllamaLLM(model=\"llama3.2:latest\")"
  "chain = prompt | model"
)
```

**全部删掉**，替换为：

```netlogo
;; Phase 1: 初始化 tool-calling agent (Qwen3:8b + Pydantic)
(py:run
  "import sys, os"
  "sys.path.insert(0, os.getcwd())"   ;; 让 Python 能找到 agent_phase1 包
  "from agent_phase1 import init_agent, schedule_service, last_decision_summary"
  "init_agent(model_name='qwen3:8b', temperature=0.1)"
)
```

> 注意：`os.getcwd()` 默认是 NetLogo 启动目录。如果你的 `.nlogo` 文件和
> `agent_phase1/` 文件夹不在同一级，就把 `os.getcwd()` 替换成 agent_phase1
> 父目录的绝对路径，例如 `"/Users/you/project"`。

---

## 改动 2：重写 `find-AI-server`

你原来的这个 reporter（约 1173–1226 行）替换为下面这版。核心变化：

1. 不再手动循环拼字符串，用 `py:set` 把结构化数据直接传过去。
2. 把资源从"剩余绝对值"换成"剩余百分比"——LLM 看百分比比看绝对数字准得多。
3. 根据 Python 返回的哨兵值分三种情况处理：选中、拒绝、fallback。

```netlogo
to-report find-AI-server [ the-server-set the-service ]
  ;; 总调用次数 +1
  set total-usage-count total-usage-count + 1

  ;; ---- 1. 把前 10 台服务器的状态打包成 [id cpu% ram% net%] 列表 ----
  let svrIDs [who] of servers
  let servers-data []
  let n min list 10 length svrIDs

  let i 0
  while [ i < n ] [
    let svr server (item i svrIDs)
    let sid [who] of svr
    ;; 百分比空闲 = (1 - 使用率) * 100
    let cpu-free-pct (1 - ([ops-now] of svr / [ops-phy] of svr)) * 100
    let ram-free-pct (1 - ([mem-now] of svr / [mem-phy] of svr)) * 100
    let net-free-pct (1 - ([net-now] of svr / [net-phy] of svr)) * 100
    set servers-data lput (list sid cpu-free-pct ram-free-pct net-free-pct) servers-data
    set i i + 1
  ]

  ;; ---- 2. 把服务请求转成百分比形式 ----
  ;; 假设所有服务器容量相同，拿第一台做基准（你原代码的假设）
  let ref-svr server (item 0 svrIDs)
  let cpu-req-pct ([ops-cnf] of the-service / [ops-phy] of ref-svr) * 100
  let ram-req-pct ([mem-cnf] of the-service / [mem-phy] of ref-svr) * 100
  let net-req-pct ([net-cnf] of the-service / [net-phy] of ref-svr) * 100

  let service-data (list cpu-req-pct ram-req-pct net-req-pct)

  ;; ---- 3. 把数据喂给 Python，取回 server_id ----
  py:set "servers_raw" servers-data
  py:set "service_raw" service-data
  let sid py:runresult "schedule_service(servers_raw, service_raw)"

  ;; ---- 4. 根据返回值分派 ----
  ;; sid >= 0 : agent 正常选择
  ;; sid = -1 : fallback 到 balanced-fit
  ;; sid = -2 : agent 拒绝该服务
  (ifelse
    sid >= 0 [
      set ai-usage-count ai-usage-count + 1
      ;; 可选：把决策摘要打到日志
      ;; show py:runresult "last_decision_summary()"
      report server sid
    ]
    sid = -2 [
      ;; agent 明确拒绝 —— 走 NetLogo 原本的拒绝流程（由 find-server 外层处理）
      report nobody
    ]
    ;; else: sid = -1，走 fallback
    [
      report find-balanced-fit-server the-server-set the-service
    ]
  )
end
```

---

## 改动 3（可选）：监视器显示最近一次决策

在你 NetLogo 界面上新建一个 Monitor，Reporter 填：

```netlogo
py:runresult "last_decision_summary()"
```

就能实时看到 `[select] server=5 ok=True 320ms | server 5 has ample ...` 这样的摘要，
调试和演示时非常直观。

---

## 文件结构

最终应该长这样：

```
your-project/
├── 2143512_Jiale_Miao_2025_Supplementary.nlogo   (按上面改 3 处)
└── agent_phase1/
    ├── __init__.py
    ├── schemas.py
    ├── prompts.py
    ├── scheduler.py
    ├── test_scheduler.py
    ├── requirements.txt
    └── README.md
```

---

## 启动顺序

1. 终端 1：`ollama serve`
2. 终端 2：`ollama pull qwen3:8b`（第一次用需要拉模型）
3. 终端 3：`python -m agent_phase1.test_scheduler` —— 先确认 Python 侧跑得通
4. 打开 NetLogo，Interface 选 `service-placement-algorithm = "AI"`，点 setup → go

---

## 可能踩的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: agent_phase1` | NetLogo 工作目录不对 | 改动 1 里 `os.getcwd()` 换成绝对路径 |
| 调用很慢（>5s） | Qwen3 默认开了 thinking | 已用 `/no_think` 关了；若仍慢，检查 Ollama 日志 |
| 每次都走 fallback | Qwen3:8b 没返回 tool call | 跑 `test_scheduler.py` 看 `raw_llm_response`，多半是模型输出自由文本 |
| Tool call 参数类型错 | 模型把 int 写成 str | Pydantic 会自动 coerce，若仍失败看 `last_decision_summary()` |
