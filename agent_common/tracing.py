from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class TraceLogger:
    """Append-only JSONL trace writer for scheduler decisions."""

    def __init__(
        self,
        trace_dir: str | Path = "traces",
        run_id: str | None = None,
        phase: str = "phase2",
        model: str = "heuristic",
        enabled: bool = True,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.phase = phase
        self.model = model
        self.enabled = enabled

    @property
    def path(self) -> Path:
        return self.trace_dir / f"{self.run_id}.jsonl"

    def write(
        self,
        *,
        tick: int | None = None,
        messages: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        decision: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        fallback_reason: str | None = None,
        outcome: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trace_id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "run_id": self.run_id,
            "tick": tick,
            "phase": self.phase,
            "model": self.model,
            "messages": messages or [],
            "tool_calls": tool_calls or [],
            "decision": decision or {},
            "latency_ms": latency_ms,
            "fallback_reason": fallback_reason,
            "outcome": outcome,
        }
        if extra:
            record.update(extra)

        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

