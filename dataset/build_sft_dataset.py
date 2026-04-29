from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


if __name__ == "__main__":
    count = build_sft_dataset()
    print(f"wrote {count} SFT samples")

