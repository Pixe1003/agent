"""SFT 数据集 builder。

v1：每行一个 OpenAI 风格 messages 样本（user → assistant tool_call → tool → assistant final）
v2：上面 + Qwen2.5 ChatML pretokenized text，含 system prompt，去重 / 加噪音 / 不平衡场景增强。

跑法：
    .\.venv\Scripts\python.exe -m dataset.build_sft_dataset
    .\.venv\Scripts\python.exe -m dataset.build_sft_dataset --v2 --max-samples 15000

输出：
    dataset/cloud-sched-sft-v1.jsonl  (默认；OpenAI tool_calls 格式)
    dataset/cloud-sched-sft-v2.jsonl  (--v2；ChatML 格式 + system prompt + 增强)

为什么有两份：v1 给 LLaMA-Factory / TRL 的 SFTTrainer 直接吃；v2 给
Unsloth 的 train_on_responses_only 走的 ChatML pretokenized 路径。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a cloud scheduling agent. Your job is to place an incoming service onto one of the available servers in a cluster, or reject it if no server fits.

You MUST respond by calling exactly ONE tool:
- select_server(server_id, reasoning) — when a valid candidate exists
- reject_service(reason) — when no server can accommodate the service

A server is a VALID candidate only if cpu_free_pct >= service.cpu_pct AND ram_free_pct >= service.ram_pct AND net_free_pct >= service.net_pct. Among valid candidates prefer the one that produces the most balanced residual utilization. Do NOT produce free-form text. The tool call IS your output."""


# =============================================================================
# v1：原 OpenAI 风格 (兼容)
# =============================================================================

def build_sft_dataset(
    *,
    trace_dir: str | Path = "traces",
    output_path: str | Path = "dataset/cloud-sched-sft-v1.jsonl",
    max_samples: int = 500,
) -> int:
    trace_path = Path(trace_dir)
    rows: list[dict[str, Any]] = []
    for file in sorted(trace_path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sample = _trace_to_sample(record)
            if sample is not None:
                rows.append(sample)
            if len(rows) >= max_samples:
                break
        if len(rows) >= max_samples:
            break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _trace_to_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    decision = record.get("decision") or {}
    outcome = record.get("outcome") or {}
    if decision.get("action") not in {"select", "reject"}:
        return None
    if outcome and outcome.get("sla_violated") is True:
        return None

    source_messages = record.get("messages") or [{"role": "user", "content": ""}]
    user_content = source_messages[-1].get("content", "")
    tool_call = _format_tool_call(record.get("tool_calls") or [], decision)
    final = {
        "role": "assistant",
        "content": decision.get("reasoning", "Decision recorded."),
    }
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(decision, ensure_ascii=False)},
            final,
        ]
    }


def _format_tool_call(tool_calls: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    raw = tool_calls[0] if tool_calls else {}
    name = raw.get("name")
    args = raw.get("args")
    if not name:
        name = "reject_service" if decision.get("action") == "reject" else "select_server"
    if not args:
        if name == "select_server":
            args = {"server_id": decision.get("server_id"), "reasoning": decision.get("reasoning", "")}
        else:
            args = {"reason": decision.get("reasoning", "")}
    return {
        "id": "call_0",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


# =============================================================================
# v2：ChatML 格式 + 去重 + 平衡 select/reject
# =============================================================================

def build_sft_dataset_v2(
    *,
    trace_dir: str | Path = "traces",
    output_path: str | Path = "dataset/cloud-sched-sft-v2.jsonl",
    max_samples: int = 15000,
    max_reject_ratio: float = 0.35,
    seed: int = 13,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, int]:
    """生成 ChatML pretokenized 训练样本。

    每行 schema：
        {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n
                  <|im_start|>assistant\\n<tool_call>{...}</tool_call><|im_end|>"}

    去重 + 平衡：
    - 按 (cluster_state_hash, action, server_id) 去重，避免高频重复决策淹没数据
    - reject 类样本最多占 max_reject_ratio (默认 35%)，剩下都是 select
    """
    trace_path = Path(trace_dir)
    rng = random.Random(seed)

    select_samples: list[dict[str, Any]] = []
    reject_samples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    stats = {
        "scanned": 0,
        "kept_select": 0,
        "kept_reject": 0,
        "skipped_dup": 0,
        "skipped_invalid": 0,
        "skipped_sla_violated": 0,
    }

    for file in sorted(trace_path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            stats["scanned"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                continue

            decision = record.get("decision") or {}
            action = decision.get("action")
            if action not in {"select", "reject"}:
                stats["skipped_invalid"] += 1
                continue
            outcome = record.get("outcome") or {}
            if outcome.get("sla_violated") is True:
                stats["skipped_sla_violated"] += 1
                continue

            source_messages = record.get("messages") or []
            user_content = ""
            for msg in source_messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            if not user_content:
                stats["skipped_invalid"] += 1
                continue

            # 去重 key：cluster state + action + server_id
            key_blob = f"{user_content}|{action}|{decision.get('server_id')}"
            key = hashlib.sha1(key_blob.encode("utf-8")).hexdigest()[:16]
            if key in seen_keys:
                stats["skipped_dup"] += 1
                continue
            seen_keys.add(key)

            tool_call = _format_tool_call(record.get("tool_calls") or [], decision)
            assistant_text = _render_assistant_tool_call(tool_call)
            chatml = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
            )

            sample = {
                "text": chatml,
                "meta": {
                    "action": action,
                    "server_id": decision.get("server_id"),
                    "trace_phase": record.get("phase"),
                },
            }
            if action == "reject":
                reject_samples.append(sample)
                stats["kept_reject"] += 1
            else:
                select_samples.append(sample)
                stats["kept_select"] += 1

    # 平衡 reject / select 比例
    rng.shuffle(select_samples)
    rng.shuffle(reject_samples)
    select_target = max_samples - int(max_samples * max_reject_ratio)
    reject_target = max_samples - select_target
    selects = select_samples[:select_target]
    rejects = reject_samples[:reject_target]
    combined = selects + rejects
    rng.shuffle(combined)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["written"] = len(combined)
    stats["written_select"] = len(selects)
    stats["written_reject"] = len(rejects)
    return stats


def _render_assistant_tool_call(tool_call: dict[str, Any]) -> str:
    """把 OpenAI tool_call 渲染成 Qwen2.5 native tool-call 文本格式。"""
    fn = tool_call.get("function") or {}
    name = fn.get("name", "select_server")
    args = fn.get("arguments", "{}")
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    payload = json.dumps({"name": name, "arguments": json.loads(args)}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="output ChatML format with system prompt + balancing")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.v2:
        stats = build_sft_dataset_v2(
            trace_dir=args.trace_dir,
            output_path=args.output or "dataset/cloud-sched-sft-v2.jsonl",
            max_samples=args.max_samples or 15000,
        )
        print(json.dumps(stats, indent=2))
    else:
        count = build_sft_dataset(
            trace_dir=args.trace_dir,
            output_path=args.output or "dataset/cloud-sched-sft-v1.jsonl",
            max_samples=args.max_samples or 500,
        )
        print(f"wrote {count} SFT samples (v1 / OpenAI format)")
