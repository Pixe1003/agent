"""
Prompt 模板和 state 渲染。

设计要点：
- 系统提示硬约束：必须调用两个 tool 之一，禁止自由文本。
- 用表格渲染 cluster state，数值精确到 1 位小数，降低 LLM 读数错误。
- `/no_think` 是 Qwen3 的官方开关，关闭思考模式以换取延迟。
  Phase 2 的 Scheduler Agent 需要 ReAct，到时候移除这个标签。
"""
from .schemas import SchedulingContext


SYSTEM_PROMPT = """/no_think
You are a cloud scheduling agent. Your job is to place an incoming service onto one of the available servers in a cluster, or reject it if no server fits.

## Your objective
Maximize cluster utilization WITHOUT overloading any server. A server is overloaded if, AFTER deploying the service, any resource dimension (CPU, RAM, Network) exceeds the overutilization threshold.

## Decision rules
1. A server is a VALID candidate only if:
   - cpu_free_pct >= service.cpu_pct
   - ram_free_pct >= service.ram_pct
   - net_free_pct >= service.net_pct
2. Among valid candidates, prefer the server that produces the most balanced residual utilization (avoid leaving one resource dimension very high while others are low).
3. If no server is valid, call `reject_service`.

## Output contract
You MUST respond by calling exactly ONE of the two available tools:
- `select_server` — when a valid candidate exists
- `reject_service` — when no server can accommodate the service

Do NOT produce free-form text. Do NOT explain before the tool call. The tool call IS your output.
"""


def render_cluster_state(ctx: SchedulingContext) -> str:
    """把 SchedulingContext 渲染成给 LLM 看的用户消息。"""
    lines = [
        f"## Cluster state ({len(ctx.servers)} active servers)",
        "",
        "| ID | CPU free | RAM free | NET free |",
        "|----|----------|----------|----------|",
    ]
    for s in ctx.servers:
        lines.append(
            f"| {s.server_id} | {s.cpu_free_pct:5.1f}%  | "
            f"{s.ram_free_pct:5.1f}%  | {s.net_free_pct:5.1f}%  |"
        )

    lines += [
        "",
        "## Incoming service request",
        f"- CPU: {ctx.service.cpu_pct:.1f}%",
        f"- RAM: {ctx.service.ram_pct:.1f}%",
        f"- NET: {ctx.service.net_pct:.1f}%",
        "",
        f"## Constraint",
        f"Overutilization threshold: {ctx.overutil_threshold_pct:.0f}% per dimension.",
        "",
        "Call select_server(server_id, reasoning) or reject_service(reason).",
    ]
    return "\n".join(lines)
