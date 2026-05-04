"""SFT/LoRA scheduler 推理适配器。

设计原则：
- 跟 multi_agent 完全同形 API，benchmark / NetLogo 切换零成本
- llama-cpp-python 加载 GGUF q4，CPU 也能跑（10-50ms/decision），有 GPU 更快
- 模型输出走 strict JSON 解析，任何 parse fail / hallucinated server_id / 越界值
  都坍缩到 balanced-fit fallback，保证 benchmark 不崩
- 每次决策都写 stats：parse_success / hallucination / fallback / inference_latency
- 模型输出走 system prompt + user prompt，与 dataset/build_sft_dataset.py v2 一致
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from agent_common.tracing import TraceLogger
from agent_common.prompts import SYSTEM_PROMPT, render_cluster_state
from agent_common.schemas import (
    SchedulingContext,
    SchedulingDecision,
    ServerSnapshot,
    ServiceRequest,
)


# ============================================================================
# 全局
# ============================================================================

_LLM: Any = None
_MODEL_PATH: str | None = None
_MAX_TOKENS = 128
_TEMPERATURE = 0.0
_TRACE_LOGGER: TraceLogger | None = None
_LAST_DECISION: dict[str, Any] = {}
_STATS: dict[str, int | float] = {}
_DLL_DIR_HANDLES: list[Any] = []


def _add_nvidia_cuda_dll_dirs() -> None:
    """Make CUDA runtime wheels discoverable on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    global _DLL_DIR_HANDLES
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for rel in (
        ("nvidia", "cuda_runtime", "bin"),
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
    ):
        dll_dir = site_packages.joinpath(*rel)
        if dll_dir.exists():
            dll_dir_str = str(dll_dir)
            if dll_dir_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{dll_dir_str}{os.pathsep}{os.environ.get('PATH', '')}"
            _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir_str))


# ============================================================================
# 公共 API
# ============================================================================

def init_agent(
    model_path: str | Path = "dataset/qwen25-1p5b-sched-merged-q4.gguf",
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
    n_ctx: int = 2048,
    max_tokens: int = 128,
    temperature: float = 0.0,
    enable_tracing: bool = False,
    trace_dir: str = "traces",
    run_id: str | None = None,
    **_: Any,
) -> None:
    """加载 GGUF + 重置 stats。第一次调用会触发 llama-cpp-python 导入。"""
    global _LLM, _MODEL_PATH, _MAX_TOKENS, _TEMPERATURE, _TRACE_LOGGER, _LAST_DECISION, _STATS

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SFT model not found: {model_path}\n"
            "训练完毕后把 dataset/qwen25-1p5b-sched-merged-q4.gguf 放到该路径。"
        )

    _add_nvidia_cuda_dll_dirs()
    try:
        from llama_cpp import Llama
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            "llama-cpp-python 未安装。pip install llama-cpp-python  "
            "(GPU 加速版：CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-cache-dir)"
        ) from e

    _MODEL_PATH = str(model_path)
    _MAX_TOKENS = max_tokens
    _TEMPERATURE = temperature
    _LLM = Llama(
        model_path=_MODEL_PATH,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    _LAST_DECISION = {}
    _STATS = {
        "calls": 0,
        "parse_success": 0,
        "parse_fail": 0,
        "hallucinated_server_id": 0,
        "balanced_fit_fallback": 0,
        "select_returned": 0,
        "reject_returned": 0,
        "total_inference_ms": 0.0,
    }
    _TRACE_LOGGER = TraceLogger(
        trace_dir=trace_dir,
        run_id=run_id or f"sft-{uuid.uuid4().hex[:10]}",
        phase="sft",
        model=Path(_MODEL_PATH).name,
        enabled=enable_tracing,
    )


def schedule_service(
    servers_raw: list,
    service_req_raw: list,
    global_state_raw: Any | None = None,   # 兼容签名，本 agent 暂不使用
    memory_context_raw: Any | None = None,
    aiops_insight_raw: Any | None = None,
) -> int:
    """生产 API。返回 server_id (>=0) / -1 fallback / -2 reject。"""
    global _LAST_DECISION
    if _LLM is None:
        raise RuntimeError("init_agent() 未调用。")

    t0 = time.perf_counter()
    _STATS["calls"] += 1

    # 解析 ctx 用于 fallback 路径
    try:
        ctx = _parse_context(servers_raw, service_req_raw)
    except (ValueError, IndexError, TypeError) as e:
        _LAST_DECISION = _record(
            "fallback", None,
            f"input validation error: {e}",
            (time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
        )
        _STATS["balanced_fit_fallback"] += 1
        return -1

    valid_ids = {int(s.server_id) for s in ctx.servers}

    # 推理
    user_prompt = render_cluster_state(ctx)
    chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inf_t0 = time.perf_counter()
    output = _LLM(
        chatml,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        top_p=1.0,
        stop=["<|im_end|>", "</tool_call>"],
        echo=False,
    )
    inf_ms = (time.perf_counter() - inf_t0) * 1000
    _STATS["total_inference_ms"] += inf_ms
    raw_text = output["choices"][0]["text"]

    # 解析 tool call
    tool = _parse_tool_call(raw_text)
    if tool is None:
        _STATS["parse_fail"] += 1
        sid = _balanced_fit(ctx, valid_ids)
        _STATS["balanced_fit_fallback"] += 1
        _LAST_DECISION = _record(
            "fallback", sid if sid >= 0 else None,
            "parse_fail; balanced-fit fallback",
            (time.perf_counter() - t0) * 1000,
            tool_call_succeeded=False,
            inference_ms=inf_ms,
            raw_response=raw_text[:200],
        )
        return sid if sid >= 0 else -1

    _STATS["parse_success"] += 1
    name = tool["name"]
    args = tool["arguments"]

    if name == "reject_service":
        _STATS["reject_returned"] += 1
        _LAST_DECISION = _record(
            "reject", None,
            args.get("reason", "model rejected"),
            (time.perf_counter() - t0) * 1000,
            tool_call_succeeded=True,
            inference_ms=inf_ms,
            raw_response=raw_text[:200],
        )
        return -2

    if name == "select_server":
        sid = args.get("server_id")
        if not isinstance(sid, int) or sid not in valid_ids:
            _STATS["hallucinated_server_id"] += 1
            recovery = _balanced_fit(ctx, valid_ids)
            _STATS["balanced_fit_fallback"] += 1
            _LAST_DECISION = _record(
                "fallback", recovery if recovery >= 0 else None,
                f"hallucinated server_id={sid}; balanced-fit fallback to {recovery}",
                (time.perf_counter() - t0) * 1000,
                tool_call_succeeded=False,
                inference_ms=inf_ms,
                raw_response=raw_text[:200],
            )
            return recovery if recovery >= 0 else -1
        _STATS["select_returned"] += 1
        _LAST_DECISION = _record(
            "select", sid,
            args.get("reasoning", ""),
            (time.perf_counter() - t0) * 1000,
            tool_call_succeeded=True,
            inference_ms=inf_ms,
            raw_response=raw_text[:200],
        )
        return sid

    # 未知 tool
    _STATS["parse_fail"] += 1
    sid = _balanced_fit(ctx, valid_ids)
    _STATS["balanced_fit_fallback"] += 1
    _LAST_DECISION = _record(
        "fallback", sid if sid >= 0 else None,
        f"unknown tool {name}; balanced-fit fallback",
        (time.perf_counter() - t0) * 1000,
        tool_call_succeeded=False,
        inference_ms=inf_ms,
        raw_response=raw_text[:200],
    )
    return sid if sid >= 0 else -1


def last_decision_dict() -> dict[str, Any]:
    return dict(_LAST_DECISION)


def last_decision_summary() -> str:
    if not _LAST_DECISION:
        return "sft no decision yet"
    return (
        f"[{_LAST_DECISION['action']}] server={_LAST_DECISION.get('server_id')} "
        f"inf={_LAST_DECISION.get('inference_ms', 0):.1f}ms | "
        f"{_LAST_DECISION.get('reasoning', '')[:80]}"
    )


def sft_stats() -> dict[str, Any]:
    calls = max(1, _STATS["calls"])
    return {
        **dict(_STATS),
        "parse_success_rate": _STATS["parse_success"] / calls,
        "hallucination_rate": _STATS["hallucinated_server_id"] / calls,
        "fallback_rate": _STATS["balanced_fit_fallback"] / calls,
        "avg_inference_ms": _STATS["total_inference_ms"] / calls,
    }


def sft_stats_summary() -> str:
    s = sft_stats()
    return (
        f"sft calls={s['calls']} parse_ok={s['parse_success_rate']*100:.1f}% "
        f"hallucinate={s['hallucination_rate']*100:.1f}% "
        f"fallback={s['fallback_rate']*100:.1f}% "
        f"avg_inf={s['avg_inference_ms']:.1f}ms"
    )


# ============================================================================
# Helpers
# ============================================================================

def _parse_context(servers_raw: list, service_req_raw: list) -> SchedulingContext:
    servers = [
        ServerSnapshot.model_construct(
            server_id=int(s[0]),
            cpu_free_pct=float(s[1]),
            ram_free_pct=float(s[2]),
            net_free_pct=float(s[3]),
        )
        for s in servers_raw
    ]
    service = ServiceRequest.model_construct(
        cpu_pct=float(service_req_raw[0]),
        ram_pct=float(service_req_raw[1]),
        net_pct=float(service_req_raw[2]),
    )
    return SchedulingContext.model_construct(
        servers=servers, service=service, overutil_threshold_pct=90.0,
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>)?", re.DOTALL)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """从模型自由文本里挖出 <tool_call>{...}</tool_call> 的 JSON 体。

    宽松：允许末尾没有 </tool_call> (我们用 stop token 截断了)；允许 JSON 里有
    多余空白；name 必须是 select_server 或 reject_service。
    """
    if not text:
        return None
    text = text.strip()
    # 尝试最直接的 <tool_call>{...} 模式
    candidate: str | None = None
    tag_pos = text.find("<tool_call>")
    search_from = tag_pos + len("<tool_call>") if tag_pos != -1 else 0
    first_brace = text.find("{", search_from)
    if first_brace != -1:
        candidate = _extract_balanced_json(text, first_brace)
    # 退化：如果模型直接吐 JSON 没有 tag
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments")
    if name not in {"select_server", "reject_service"}:
        return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def _extract_balanced_json(text: str, start: int) -> str | None:
    """从 start (一个 '{') 开始抠出第一个完整的 JSON 对象 (括号配平)。"""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _balanced_fit(ctx: SchedulingContext, valid_ids: set[int]) -> int:
    """与 multi_agent / benchmark 的 balanced-fit 一致：选 post-placement 资源
    分布最平衡的 server。无候选返回 -2。"""
    cands = [
        s for s in ctx.servers
        if s.cpu_free_pct >= ctx.service.cpu_pct
        and s.ram_free_pct >= ctx.service.ram_pct
        and s.net_free_pct >= ctx.service.net_pct
    ]
    if not cands:
        return -2

    def score(s: ServerSnapshot):
        residuals = (
            s.cpu_free_pct - ctx.service.cpu_pct,
            s.ram_free_pct - ctx.service.ram_pct,
            s.net_free_pct - ctx.service.net_pct,
        )
        return (max(residuals) - min(residuals), sum(residuals), s.server_id)
    return int(min(cands, key=score).server_id)


def _record(
    action: str,
    server_id: int | None,
    reasoning: str,
    latency_ms: float,
    *,
    tool_call_succeeded: bool,
    inference_ms: float | None = None,
    raw_response: str | None = None,
) -> dict[str, Any]:
    decision = SchedulingDecision(
        action=action,  # type: ignore[arg-type]
        server_id=server_id,
        reasoning=reasoning,
        latency_ms=latency_ms,
        tool_call_succeeded=tool_call_succeeded,
        raw_llm_response=raw_response,
    )
    data = decision.model_dump()
    data["phase"] = "sft"
    data["model"] = Path(_MODEL_PATH).name if _MODEL_PATH else "unknown"
    if inference_ms is not None:
        data["inference_ms"] = inference_ms
    if _TRACE_LOGGER is not None:
        _TRACE_LOGGER.write(decision=data, latency_ms=latency_ms)
    return data
