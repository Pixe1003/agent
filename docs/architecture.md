# Architecture Diagram

## How to view / export

打开 [excalidraw.com](https://excalidraw.com)（不需要登录）：

1. 左上角菜单 → **Open** → 选 `docs/architecture.excalidraw`
2. 或者：用文本编辑器打开 `docs/architecture.excalidraw` → 全选复制 → Excalidraw 画布上 `Ctrl+V`
3. 编辑完后：**File → Save as image** → PNG 或 SVG，导出到 `docs/architecture.png`，README 顶部用 `![architecture](docs/architecture.png)` 引用

也可以用本地 VSCode 插件 [Excalidraw](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor) 直接在编辑器里打开 `.excalidraw` 文件。

## Layout 说明

图纸 1000 × 760，分四层：

**第 1 层（顶部）**：NetLogo Cluster Simulation（蓝色长条）—— 数据源，输出 servers / service requests / global state。

**第 2 层（中间）**：三个 Agent 模块并排，颜色编码方便面试时口头指：
- **绿色 multi_agent**（左）：内嵌 Planner → Scheduler → Critic 三个小盒子，Critic 是黄色突出色（含 AIOps safety margin 注解）
- **紫色 agent_memory**（中）：Working memory + Episodic memory
- **红色 agent_aiops**（右）：Observe → Risk analyzer → Active alerts

**第 3 层（中下）**：灰色长条 agent_common —— 共享 schema / prompt / tracing 基础设施。

**第 4 层（底部）**：四个虚线小盒子的 SFT pipeline 横向流：
traces → build_sft_dataset --v2 → Unsloth LoRA → agent_sft (llama.cpp + fallback)。

## 关键箭头

- 蓝色 ↓：NetLogo 喂数据给三个 agent
- **粗红色 ←**：AIOps closed loop —— 从 agent_aiops 横向射回 multi_agent 的 Critic（这是项目最强的卖点，箭头特意做粗加红）
- 紫色 ↔：multi_agent 与 agent_memory 双向交换 episodic context
- 灰色虚线 ↓：multi_agent 写 trace
- 灰色 →：traces 一路串到 SFT pipeline

## 配色（来自 Excalidraw 默认 palette，导出后底色仍是白）

| 模块 | stroke | fill |
|---|---|---|
| NetLogo | #1971c2 (blue) | #a5d8ff |
| multi_agent | #2f9e44 (green) | #b2f2bb |
| Critic 高亮 | #e67700 (amber) | #ffec99 |
| agent_memory | #7048e8 (violet) | #d0bfff |
| agent_aiops | #c92a2a (red) | #ffc9c9 |
| agent_common | #495057 (slate) | #dee2e6 |
| SFT pipeline | #495057 dashed | #f8f9fa |

## README 集成方式

`README.md` 已经合并本文件的核心架构说明，并保留 Mermaid 版本用于 GitHub 直接渲染。`docs/architecture.excalidraw` 是更适合演示、简历、PPT 和 LinkedIn 的可编辑图源。

如果需要在 README 顶部改用导出的静态图片，可以把架构段落中的 Mermaid 代码块替换为：

```markdown
![architecture](docs/architecture.png)

> 源文件：[`docs/architecture.excalidraw`](docs/architecture.excalidraw) — 用 [excalidraw.com](https://excalidraw.com) 打开可编辑。
```

Mermaid 适合仓库首页的快速渲染；Excalidraw 导出的 PNG/SVG 更适合外部展示。同一项目保留两套图，按场景切换即可。

## 备选：手动重画的 5 分钟版本

如果对 excalidraw JSON 有任何渲染问题，按下面文字稿在白板风格工具（excalidraw / draw.io / Miro）里 5 分钟手画：

```
[NetLogo Cluster Simulation] (top, full width, blue)
        │
        ▼
┌─────────────┬─────────────┬─────────────┐
│ multi_agent │agent_memory │ agent_aiops │
│  (green)    │  (violet)   │  (red)      │
│             │             │             │
│  Planner    │ Working mem │  Observe    │
│     ↓       │ Episodic    │  Risk tags  │
│ Scheduler   │  retrieval  │  Alerts     │
│     ↓       │             │             │
│  Critic ←───────────────── (closed loop, thick red arrow)
│ +safety mgn │             │             │
└─────────────┴─────────────┴─────────────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
        [agent_common: schemas + prompts + tracing] (gray bar)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
[traces.jsonl] → [build_sft_dataset --v2] → [Unsloth LoRA] → [agent_sft]
                  (dashed boxes, gray)

Footer: "AIOps SLA 35%→0.75% • energy -11% • SFT 4240× slower"
```

记住的口诀：**蓝顶（输入）→ 三色并排（控制层）→ 灰底（共享层）→ 虚线尾巴（离线 pipeline）→ 红色闭环箭头压在 Critic 上**。
