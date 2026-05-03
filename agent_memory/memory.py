from __future__ import annotations

import io
import math
import uuid
from pathlib import Path
from typing import Any

from agent_common.schemas import SchedulingContext, SchedulingDecision, ServerSnapshot, ServiceRequest
from pydantic import BaseModel, Field, PrivateAttr


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
    # 缓存预分词的 token set，retrieve 时不再每次重切字符串。
    # PrivateAttr 不会影响序列化 / 反序列化。
    _cached_tokens: frozenset[str] | None = PrivateAttr(default=None)

    def tokens(self) -> frozenset[str]:
        if self._cached_tokens is None:
            self._cached_tokens = _tokenize(self.state_summary_text)
        return self._cached_tokens


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
    """Episodic memory store with optional persistence and batched I/O.

    Performance notes (driven by profiling):
    - tokens are cached on each Episode at insertion (avoid per-retrieve re-tokenization)
    - the writer keeps a single open file handle and flushes every `flush_every` adds
    - `persist=False` skips the file entirely (recommended for benchmarks where
      cross-run persistence is not needed)

    Direct store usage defaults to durable writes so a fresh EpisodicMemory
    instance can immediately read newly-added episodes. Hot-path schedulers can
    still opt into batching by passing a larger `flush_every`.
    """

    def __init__(
        self,
        path: str | Path = "traces/episodes.jsonl",
        *,
        persist: bool = True,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.persist = persist
        self.flush_every = max(1, int(flush_every))
        self._episodes: list[Episode] = []
        self._loaded = False
        self._writer: io.TextIOBase | None = None
        self._unflushed = 0

    def add(self, episode: Episode) -> None:
        self._load()
        episode.tokens()  # warm cache up-front, retrieve won't pay the cost
        self._episodes.append(episode)
        if not self.persist:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 单次打开，长开文件句柄；析构 / close() 时 flush
            self._writer = self.path.open("a", encoding="utf-8")
        self._writer.write(episode.model_dump_json() + "\n")
        self._unflushed += 1
        if self._unflushed >= self.flush_every:
            self._writer.flush()
            self._unflushed = 0

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
            finally:
                self._writer.close()
            self._writer = None
            self._unflushed = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def retrieve(self, query_text: str, query_features: list[float], top_k: int = 3) -> list[Episode]:
        self._load()
        if not self._episodes:
            return []

        query_tokens = _tokenize(query_text)
        # 把 query feature 转为 tuple 一次，避免 _euclidean 内部反复 len()
        qf = tuple(query_features)

        # 内联 score 计算，避免函数调用开销 + 重复 tokenize
        scored: list[tuple[float, Episode]] = []
        for episode in self._episodes:
            ep_tokens = episode.tokens()
            if query_tokens and ep_tokens:
                inter = len(query_tokens & ep_tokens)
                union = len(query_tokens | ep_tokens)
                lexical = inter / union if union else 0.0
            else:
                lexical = 0.0
            distance = _euclidean_tuple(qf, episode.state_features)
            reward = episode.reward or 0.0
            scored.append((lexical + reward - distance, episode))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = Episode.model_validate_json(line)
                ep.tokens()  # warm cache
                self._episodes.append(ep)


def summarize_context(servers_raw: list, service_req_raw: list) -> tuple[str, list[float]]:
    """Build a short natural-language summary + 6-dim feature vector.

    Optimized: avoids creating heavy Pydantic objects in the hot path.
    Inputs are trusted (they come from NetLogo / benchmark sim).
    """
    n = len(servers_raw)
    if n == 0:
        return ("Cluster has 0 active servers.", [0.0] * 6)

    cpu_sum = ram_sum = net_sum = 0.0
    for s in servers_raw:
        cpu_sum += float(s[1])
        ram_sum += float(s[2])
        net_sum += float(s[3])

    cpu_mean = cpu_sum / n
    ram_mean = ram_sum / n
    net_mean = net_sum / n
    svc_cpu = float(service_req_raw[0])
    svc_ram = float(service_req_raw[1])
    svc_net = float(service_req_raw[2])

    summary = (
        f"Cluster has {n} active servers. "
        f"Mean free resources are CPU {cpu_mean:.1f}%, RAM {ram_mean:.1f}%, NET {net_mean:.1f}%. "
        f"Incoming service needs CPU {svc_cpu:.1f}%, RAM {svc_ram:.1f}%, NET {svc_net:.1f}%."
    )
    features = [cpu_mean / 100, ram_mean / 100, net_mean / 100,
                svc_cpu / 100, svc_ram / 100, svc_net / 100]
    return summary, features


def _tokenize(text: str) -> frozenset[str]:
    """Cheap, stable tokenizer used by both query and stored episodes."""
    if not text:
        return frozenset()
    return frozenset(token.lower() for token in text.replace(",", " ").split())


def _token_overlap(a: str, b: str) -> float:
    """Public-facing helper kept for backward compat (tests may import it)."""
    left = _tokenize(a)
    right = _tokenize(b)
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


def _euclidean(a: list[float], b: list[float]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 1.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(width)))


def _euclidean_tuple(a: tuple[float, ...], b: list[float]) -> float:
    """Hot-path variant that skips the len-based slice and uses a precomputed tuple."""
    width = min(len(a), len(b))
    if width == 0:
        return 1.0
    s = 0.0
    for i in range(width):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)
