from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

from agent_phase1.schemas import SchedulingContext, SchedulingDecision, ServerSnapshot, ServiceRequest
from pydantic import BaseModel, Field


class Episode(BaseModel):
    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    tick: int
    state_summary_text: str
    state_features: list[float]
    service_request: dict[str, Any]
    action_server_id: int | str
    reasoning_trace: str
    outcome: dict[str, Any] | None = None
    reward: float | None = None


class WorkingMemory:
    def __init__(self, max_items: int = 5) -> None:
        self.max_items = max_items
        self._items: list[SchedulingDecision] = []

    def add(self, decision: SchedulingDecision) -> None:
        if decision.action != "select" or not decision.tool_call_succeeded or decision.server_id is None:
            return
        self._items.append(decision)
        self._items = self._items[-self.max_items :]

    def render(self) -> str:
        if not self._items:
            return "No recent successful decisions."
        lines = ["Recent successful decisions:"]
        for item in self._items:
            lines.append(f"- selected server {item.server_id}: {item.reasoning}")
        return "\n".join(lines)


class EpisodicMemory:
    def __init__(self, path: str | Path = "traces/episodes.jsonl") -> None:
        self.path = Path(path)
        self._episodes: list[Episode] = []
        self._loaded = False

    def add(self, episode: Episode) -> None:
        self._load()
        self._episodes.append(episode)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(episode.model_dump_json() + "\n")

    def retrieve(self, query_text: str, query_features: list[float], top_k: int = 3) -> list[Episode]:
        self._load()
        if not self._episodes:
            return []

        def score(episode: Episode) -> float:
            lexical = _token_overlap(query_text, episode.state_summary_text)
            distance = _euclidean(query_features, episode.state_features)
            reward = episode.reward or 0.0
            return lexical + reward - distance

        return sorted(self._episodes, key=score, reverse=True)[:top_k]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._episodes.append(Episode.model_validate_json(line))


def summarize_context(servers_raw: list, service_req_raw: list) -> tuple[str, list[float]]:
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
    cpu_mean = sum(s.cpu_free_pct for s in ctx.servers) / len(ctx.servers)
    ram_mean = sum(s.ram_free_pct for s in ctx.servers) / len(ctx.servers)
    net_mean = sum(s.net_free_pct for s in ctx.servers) / len(ctx.servers)
    summary = (
        f"Cluster has {len(ctx.servers)} active servers. "
        f"Mean free resources are CPU {cpu_mean:.1f}%, RAM {ram_mean:.1f}%, NET {net_mean:.1f}%. "
        f"Incoming service needs CPU {ctx.service.cpu_pct:.1f}%, RAM {ctx.service.ram_pct:.1f}%, "
        f"NET {ctx.service.net_pct:.1f}%."
    )
    features = [
        cpu_mean / 100,
        ram_mean / 100,
        net_mean / 100,
        ctx.service.cpu_pct / 100,
        ctx.service.ram_pct / 100,
        ctx.service.net_pct / 100,
    ]
    return summary, features


def _token_overlap(a: str, b: str) -> float:
    left = {token.lower() for token in a.replace(",", " ").split()}
    right = {token.lower() for token in b.replace(",", " ").split()}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _euclidean(a: list[float], b: list[float]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 1.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(width)))

