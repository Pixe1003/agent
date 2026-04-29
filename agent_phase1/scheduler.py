"""
Scheduler agent —— Phase 1 主模块。

对 NetLogo 暴露的唯一接口：schedule_service(servers, service_req) -> int

返回语义：
    >= 0 : chosen server_id
    -1   : fallback（NetLogo 应调用 balanced-fit）
    -2   : reject（NetLogo 应拒绝服务）

所有 LLM 异常、Pydantic 验证失败、非法 server_id 都坍缩成 -1。
NetLogo 只需判断返回值符号，不需要了解 Python 端的任何细节。
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from .schemas import (
    ServerSnapshot,
    ServiceRequest,
    SchedulingContext,
    SchedulingDecision,
    select_server,
    reject_service,
)
from .prompts import SYSTEM_PROMPT, render_cluster_state


# =============================================================================
# 全局 LLM 实例 —— 只在 setup 时构建一次，避免每个 tick 重新连接 Ollama
# =============================================================================

_LLM: Any = None  # bound LLM with tools, ChatOllama with bind_tools applied
_LAST_DECISION: SchedulingDecision | None = None  # 供调试和 memory 模块读取


def init_agent(
    model_name: str = "qwen3:8b",
    temperature: float = 0.1,
    num_predict: int = 256,
    base_url: str | None = None,
) -> None:
    """由 NetLogo 的 setup 调用一次。

    参数:
        model_name: Ollama 模型标识。默认 qwen3:8b。
        temperature: 采样温度。调度是决定性任务，保持低。
        num_predict: 最大输出 token 数。tool call 很短，256 够用。
        base_url: Ollama server 地址，默认 http://localhost:11434。
    """
    global _LLM

    kwargs = dict(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
    )
    if base_url:
        kwargs["base_url"] = base_url

    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise RuntimeError(
            "LangChain/Ollama dependencies are missing. "
            "Run: pip install -r agent_phase1/requirements.txt"
        ) from e

    base_llm = ChatOllama(**kwargs)
    # bind_tools: 把两个 Pydantic 类注册为可调用工具
    # LangChain 会把类名和 docstring 翻译成 Ollama 原生的 tool spec
    _LLM = base_llm.bind_tools([select_server, reject_service])

    print(f"[agent] Initialized with model={model_name}, temp={temperature}")


# =============================================================================
# 核心调度函数 —— NetLogo 每次调度调用一次
# =============================================================================

def schedule_service(servers_raw: list, service_req_raw: list) -> int:
    """NetLogo 入口。

    参数格式（NetLogo 侧用 py:set 传入 Python 列表）:
        servers_raw: [[id, cpu_free_pct, ram_free_pct, net_free_pct], ...]
        service_req_raw: [cpu_pct, ram_pct, net_pct]

    返回:
        server_id >= 0 : 正常选择
        -1             : fallback 到 balanced-fit
        -2             : 拒绝该服务
    """
    global _LAST_DECISION
    t0 = time.perf_counter()

    # ---- 1. 解析 NetLogo 传入的原始数据 ----
    try:
        ctx = SchedulingContext(
            servers=[
                ServerSnapshot(
                    server_id=int(s[0]),
                    cpu_free_pct=float(s[1]),
                    ram_free_pct=float(s[2]),
                    net_free_pct=float(s[3]),
                )
                for s in servers_raw
            ],
            service=ServiceRequest(
                cpu_pct=float(service_req_raw[0]),
                ram_pct=float(service_req_raw[1]),
                net_pct=float(service_req_raw[2]),
            ),
        )
    except (ValidationError, ValueError, IndexError, TypeError) as e:
        # 输入数据畸形 —— 不是 LLM 的锅，但无法继续，走 fallback
        _LAST_DECISION = SchedulingDecision(
            action="fallback",
            reasoning=f"Input validation error: {e}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        return -1

    valid_ids = {s.server_id for s in ctx.servers}

    # ---- 2. LLM 调用 ----
    if _LLM is None:
        _LAST_DECISION = SchedulingDecision(
            action="fallback",
            reasoning="Agent not initialized. Call init_agent() in NetLogo setup.",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        return -1

    messages = _build_messages(ctx)

    try:
        response = _LLM.invoke(messages)
    except Exception as e:
        # 网络错误、Ollama 宕机、超时等
        _LAST_DECISION = SchedulingDecision(
            action="fallback",
            reasoning=f"LLM call failed: {type(e).__name__}: {e}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        return -1

    # ---- 3. 解析 tool call ----
    tool_calls = getattr(response, "tool_calls", None) or []

    if not tool_calls:
        # 模型输出了自由文本，没走 tool。为了实机演示可用，启用确定性安全兜底；
        # telemetry 仍标记 tool_call_succeeded=False，benchmark 能看出不是 LLM 成功。
        return _deterministic_safety_fallback(
            ctx,
            t0=t0,
            reason="No tool call in response (model produced free text); deterministic safety fallback used.",
            raw_llm_response=str(getattr(response, "content", ""))[:500],
        )

    tc = tool_calls[0]  # 只取第一个 tool call
    tool_name = tc.get("name", "")
    tool_args = tc.get("args", {}) or {}

    # ---- 4. 分发到具体 tool handler ----
    try:
        if tool_name == "select_server":
            decision = select_server(**tool_args)  # Pydantic 二次验证
            if decision.server_id not in valid_ids:
                # LLM 幻觉了不存在的 server_id
                _LAST_DECISION = SchedulingDecision(
                    action="fallback",
                    reasoning=f"Hallucinated server_id={decision.server_id} not in {sorted(valid_ids)}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    tool_call_succeeded=False,
                )
                return -1

            _LAST_DECISION = SchedulingDecision(
                action="select",
                server_id=decision.server_id,
                reasoning=decision.reasoning,
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            return decision.server_id

        elif tool_name == "reject_service":
            decision = reject_service(**tool_args)
            _LAST_DECISION = SchedulingDecision(
                action="reject",
                reasoning=decision.reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=True,
            )
            return -2

        else:
            # LLM 调了一个不存在的 tool
            _LAST_DECISION = SchedulingDecision(
                action="fallback",
                reasoning=f"Unknown tool name: {tool_name}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_call_succeeded=False,
            )
            return -1

    except ValidationError as e:
        # tool 参数类型/范围不符合 Pydantic 约束
        _LAST_DECISION = SchedulingDecision(
            action="fallback",
            reasoning=f"Tool args validation failed: {e}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        return -1


# =============================================================================
# 调试 / telemetry 接口
# =============================================================================

def last_decision_summary() -> str:
    """NetLogo 可用 py:runresult 调用，用来在监视器上看最近一次决策。"""
    if _LAST_DECISION is None:
        return "no decision yet"
    d = _LAST_DECISION
    return (
        f"[{d.action}] server={d.server_id} "
        f"ok={d.tool_call_succeeded} "
        f"{d.latency_ms:.0f}ms | {d.reasoning[:80]}"
    )


def last_decision_dict() -> dict:
    """Phase 3 memory 模块从这里拉 episode 数据。"""
    if _LAST_DECISION is None:
        return {}
    return _LAST_DECISION.model_dump()


def _build_messages(ctx: SchedulingContext) -> list[Any]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=render_cluster_state(ctx)),
        ]
    except ImportError:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_cluster_state(ctx)},
        ]


def _deterministic_safety_fallback(
    ctx: SchedulingContext,
    *,
    t0: float,
    reason: str,
    raw_llm_response: str | None = None,
) -> int:
    global _LAST_DECISION
    candidates = [
        server
        for server in ctx.servers
        if server.cpu_free_pct >= ctx.service.cpu_pct
        and server.ram_free_pct >= ctx.service.ram_pct
        and server.net_free_pct >= ctx.service.net_pct
    ]
    if not candidates:
        _LAST_DECISION = SchedulingDecision(
            action="reject",
            reasoning=f"{reason} No valid server exists.",
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
            raw_llm_response=raw_llm_response,
        )
        return -2

    def score(server: ServerSnapshot) -> tuple[float, float, int]:
        residual = [
            server.cpu_free_pct - ctx.service.cpu_pct,
            server.ram_free_pct - ctx.service.ram_pct,
            server.net_free_pct - ctx.service.net_pct,
        ]
        spread = max(residual) - min(residual)
        return (spread, sum(residual), server.server_id)

    chosen = min(candidates, key=score)
    _LAST_DECISION = SchedulingDecision(
        action="select",
        server_id=chosen.server_id,
        reasoning=f"{reason} Selected server {chosen.server_id}.",
        latency_ms=(time.perf_counter() - t0) * 1000,
        tool_call_succeeded=False,
        raw_llm_response=raw_llm_response,
    )
    return chosen.server_id
