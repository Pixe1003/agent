"""Expose the Agent Infra Toolkit as an MCP (Model Context Protocol) server.

把 multi_agent / agent_aiops / agent_memory 三个 agent 的核心能力通过标准
MCP protocol 暴露，使本项目可以被任意 MCP host（Claude Desktop / VSCode /
Cursor / 任何符合 Anthropic MCP spec 的 LLM 客户端）直接调用。

依赖：
    pip install mcp

跑法（stdio transport，给 Claude Desktop / Cursor 使用）：
    python mcp_server.py

在 Claude Desktop 中注册：编辑 `claude_desktop_config.json` 加上：
    {
      "mcpServers": {
        "cloud-scheduler": {
          "command": "python",
          "args": ["D:\\\\Users\\\\12057\\\\Desktop\\\\agent\\\\mcp_server.py"]
        }
      }
    }

暴露的能力：
    Tools:
        schedule_placement   — 调用 multi_agent 做调度决策（支持 AIOps 闭环）
        aiops_observe        — 调用 agent_aiops 观测 ops state 拿 risk_tags
        memory_retrieve      — 从 episodic memory 检索历史相似 case
    Resources:
        scheduler://stats    — multi_agent.hybrid_stats() 当前快照
        aiops://summary      — agent_aiops 当前 risk level / alerts 摘要
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 让 mcp_server.py 直接跑也能 import 项目里的 agent 包
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise SystemExit(
        "MCP SDK 未安装。先跑：\n"
        "    pip install mcp\n"
        "或：\n"
        "    pip install 'mcp[cli]'"
    ) from e


# ============================================================================
# Server 初始化（lazy）
# ============================================================================

mcp = FastMCP("cloud-scheduler")

_MULTI_AGENT_INITIALIZED = False
_AIOPS_INITIALIZED = False


def _ensure_multi_agent() -> None:
    global _MULTI_AGENT_INITIALIZED
    if _MULTI_AGENT_INITIALIZED:
        return
    from multi_agent import init_agent
    init_agent(model_name="heuristic", enable_tracing=False)
    _MULTI_AGENT_INITIALIZED = True


def _ensure_aiops() -> None:
    global _AIOPS_INITIALIZED
    if _AIOPS_INITIALIZED:
        return
    from agent_aiops import init_agent as init_aiops
    init_aiops(
        model_name="heuristic",
        backend="rule",
        enable_tracing=False,
        window_size=8,
        recommendation_cooldown=0,
    )
    _AIOPS_INITIALIZED = True


# ============================================================================
# Tools
# ============================================================================

@mcp.tool()
def schedule_placement(
    servers: list[list[float]],
    service: list[float],
    aiops_risk_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Place an incoming service onto a cluster server using the multi_agent
    Planner-Scheduler-Critic graph.

    Args:
        servers: List of [server_id, cpu_free_pct, ram_free_pct, net_free_pct].
                 Example: [[0, 80.0, 70.0, 90.0], [1, 50.0, 60.0, 40.0]]
        service: [cpu_pct, ram_pct, net_pct] demanded by the incoming service.
                 Example: [25.0, 20.0, 10.0]
        aiops_risk_tags: Optional list of active AIOps risk tags. Supported tags:
                 "network-pressure" | "cpu-pressure" | "memory-pressure" |
                 "sla-risk" | "capacity-risk" | "migration-pressure" | ...
                 When provided, the critic tightens the post-placement safety
                 margin to 15-22.5% on the matching dimensions.

    Returns:
        Decision dict with action / server_id / reasoning / latency_ms and full
        aiops_* + critic_* metadata.
    """
    _ensure_multi_agent()
    from multi_agent import schedule_service, last_decision_dict

    aiops_insight: dict[str, Any] | None = None
    if aiops_risk_tags:
        aiops_insight = {
            "risk_tags": list(aiops_risk_tags),
            "risk_level": "high",
            "risk_score": 0.7,
            "active_alerts": [
                {"tag": t, "occurrence_count": 1, "risk_score": 0.7}
                for t in aiops_risk_tags
            ],
        }

    sid = schedule_service(servers, service, None, None, aiops_insight)
    decision = last_decision_dict()
    decision["server_id_returned"] = sid
    return decision


@mcp.tool()
def aiops_observe(
    active_cpu_util: float = 0.0,
    active_mem_util: float = 0.0,
    active_net_util: float = 0.0,
    rejected_services: int = 0,
    ops_sla_violations: float = 0.0,
    mem_sla_violations: float = 0.0,
    net_sla_violations: float = 0.0,
    current_auto_migrations: int = 0,
    tick: int | None = None,
) -> dict[str, Any]:
    """Observe an OpsSnapshot and return the AIOps risk insight (risk_tags +
    active_alerts + recommendations + guardrails).

    通常在调用 schedule_placement 之前先调用本工具拿到 risk_tags，再把
    risk_tags 作为参数传给 schedule_placement 实现 closed-loop 调度。

    Args:
        active_cpu_util: 0-1 fraction or 0-100 percentage.
        active_mem_util: same convention as cpu.
        active_net_util: same convention.
        rejected_services: 累计拒绝服务数（capacity-risk 信号）。
        ops_sla_violations / mem_sla_violations / net_sla_violations: SLA 违约累计计数。
        current_auto_migrations: 当前自动迁移数（migration-pressure 信号）。
        tick: 可选，rolling window 的时间戳。
    """
    _ensure_aiops()
    from agent_aiops import observe_ops_state

    insight = observe_ops_state(
        {
            "active_cpu_util": active_cpu_util,
            "active_mem_util": active_mem_util,
            "active_net_util": active_net_util,
            "rejected_services": rejected_services,
            "ops_sla_violations": ops_sla_violations,
            "mem_sla_violations": mem_sla_violations,
            "net_sla_violations": net_sla_violations,
            "current_auto_migrations": current_auto_migrations,
        },
        tick=tick,
    )
    # 把 evidence 这个比较大的 nested dict 简化掉，节省 MCP 传输
    insight = {k: v for k, v in insight.items() if k != "evidence"}
    return insight


@mcp.tool()
def memory_retrieve(
    query_text: str,
    query_features: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve similar past scheduling episodes from EpisodicMemory.

    适合给 LLM 做 in-context evidence：先用 aiops_observe 拿到当前状态描述，
    再调本工具拿 top-k 相似历史决策。

    Args:
        query_text: cluster + service 的自然语言摘要。
        query_features: 6 维特征向量 [cpu_free_mean, ram_free_mean, net_free_mean,
                        service_cpu, service_ram, service_net]，每维 0-1。
        top_k: 返回的相似 episode 数量。
    """
    from agent_memory.memory import EpisodicMemory

    store = EpisodicMemory(persist=False)  # 只读模式，不写盘
    matches = store.retrieve(query_text, query_features, top_k=top_k)
    return [m.model_dump() for m in matches]


# ============================================================================
# Resources
# ============================================================================

@mcp.resource("scheduler://stats")
def scheduler_stats() -> str:
    """Current scheduler statistics — multi_agent.hybrid_stats() 快照。

    给 LLM host 一次性看到 fast_path_ratio / escalation_ratio /
    aiops_critic_trigger_ratio 等 15+ 维 SDK 化指标。
    """
    _ensure_multi_agent()
    from multi_agent import hybrid_stats
    return json.dumps(hybrid_stats(), indent=2, ensure_ascii=False, default=str)


@mcp.resource("aiops://summary")
def aiops_summary() -> str:
    """AIOps current risk_level / active_alerts / last insight 摘要。"""
    _ensure_aiops()
    from agent_aiops import aiops_stats, last_insight_dict
    return json.dumps(
        {
            "stats": aiops_stats(),
            "last_insight": last_insight_dict(),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    # stdio transport 是 MCP 默认配置，Claude Desktop / Cursor 都用这种
    mcp.run()
