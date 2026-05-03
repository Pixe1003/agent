from __future__ import annotations

import csv
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 让 `python benchmark/runner.py` 直接当脚本跑也能 import 到项目根的包
# (multi_agent / agent_memory / agent_common)。用 `-m benchmark.runner`
# 时这一段是 no-op。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


ServerRows = list[list[float]]
ServiceRow = list[float]


# =============================================================================
# 公共参数 / Defaults
# =============================================================================

# 默认请求量提到 100，让 phase3 的记忆库见到真实的 workload drift
DEFAULT_REQUESTS_PER_SCENARIO = 100

# 集群规模
DEFAULT_N_SERVERS = 10

# 初始利用率区间 (单位 %)。45-70 让一开始就有适度压力，
# burst 请求触发拒绝/SLA 违约。
DEFAULT_INIT_UTIL_RANGE = (45.0, 70.0)

# 每次调度前以此概率把每条历史 placement 释放掉一部分资源
# (模拟服务完成 / 缩容)。0.05 ≈ 服务平均寿命 ~20 ticks。
DEFAULT_CHURN_PROB = 0.05

# 放置后任意维度 free_pct 低于此阈值视为 SLA 违约
DEFAULT_SLA_HEADROOM = 10.0

# 能耗模型权重
ENERGY_IDLE_PER_ACTIVE = 0.4   # 每台活跃服务器的待机基线
ENERGY_DYNAMIC_WEIGHT = 0.6    # 动态项权重 (按利用率分数 0-1)
ENERGY_HOTSPOT_WEIGHT = 0.5    # 利用率方差惩罚


# =============================================================================
# 集群状态 / Stateful cluster
# =============================================================================

@dataclass
class _Cluster:
    """带状态的集群快照。每个算法跑前都会从一个共同的 initial cluster 深拷贝出来。"""
    servers: list[list[float]]
    placements: list[tuple[int, list[float], int]] = field(default_factory=list)
    # placements: (server_id, service_demand, tick_placed)

    def snapshot(self) -> ServerRows:
        return [list(s) for s in self.servers]

    def fits(self, sid: int, service: ServiceRow) -> bool:
        for s in self.servers:
            if int(s[0]) == sid:
                return s[1] >= service[0] and s[2] >= service[1] and s[3] >= service[2]
        return False

    def place(self, sid: int, service: ServiceRow, tick: int) -> bool:
        for s in self.servers:
            if int(s[0]) == sid:
                if s[1] < service[0] or s[2] < service[1] or s[3] < service[2]:
                    return False
                s[1] -= service[0]
                s[2] -= service[1]
                s[3] -= service[2]
                self.placements.append((sid, list(service), tick))
                return True
        return False

    def churn(self, rng: random.Random, prob: float) -> None:
        """以 prob 概率独立释放每条历史 placement (模拟服务完成)。"""
        if not self.placements:
            return
        remaining: list[tuple[int, list[float], int]] = []
        for sid, svc, t in self.placements:
            if rng.random() < prob:
                for s in self.servers:
                    if int(s[0]) == sid:
                        s[1] = min(100.0, s[1] + svc[0])
                        s[2] = min(100.0, s[2] + svc[1])
                        s[3] = min(100.0, s[3] + svc[2])
                        break
            else:
                remaining.append((sid, svc, t))
        self.placements = remaining


def _initial_cluster(seed: int, distribution: str, *, n_servers: int, init_util_range: tuple[float, float]) -> _Cluster:
    rng = random.Random(f"{seed}:{distribution}:cluster")
    lo, hi = init_util_range
    servers: list[list[float]] = []
    for i in range(n_servers):
        # 三个维度独立采样，避免全部一致带来的对称性
        cpu_used = rng.uniform(lo, hi)
        ram_used = rng.uniform(lo, hi)
        net_used = rng.uniform(lo, hi)
        servers.append([
            float(i),
            max(0.0, min(100.0, 100.0 - cpu_used)),
            max(0.0, min(100.0, 100.0 - ram_used)),
            max(0.0, min(100.0, 100.0 - net_used)),
        ])
    return _Cluster(servers=servers)


def _clone_cluster(c: _Cluster) -> _Cluster:
    return _Cluster(servers=[list(s) for s in c.servers], placements=[])


# =============================================================================
# 请求生成 / Workload generation
# =============================================================================

def _make_requests(seed: int, distribution: str, requests: int) -> list[ServiceRow]:
    rng = random.Random(f"{seed}:{distribution}:reqs")
    out: list[ServiceRow] = []
    for _ in range(requests):
        if distribution == "cpu-heavy":
            svc = [rng.uniform(30, 55), rng.uniform(5, 20), rng.uniform(5, 20)]
        elif distribution == "memory-heavy":
            svc = [rng.uniform(5, 20), rng.uniform(30, 55), rng.uniform(5, 20)]
        elif distribution == "mixed-burst":
            # 30% 突发尖峰，70% 常规小请求
            if rng.random() < 0.30:
                svc = [rng.uniform(40, 60), rng.uniform(40, 60), rng.uniform(40, 60)]
            else:
                svc = [rng.uniform(8, 22), rng.uniform(8, 22), rng.uniform(8, 22)]
        else:  # "mixed"
            svc = [rng.uniform(10, 35), rng.uniform(10, 35), rng.uniform(10, 35)]
        out.append(svc)
    return out


# =============================================================================
# 能耗 / Energy model
# =============================================================================

def _tick_telemetry(servers: ServerRows) -> dict[str, float]:
    """单 tick 的集群遥测：能耗、活跃数、利用率方差。"""
    utils = []  # 每台服务器的利用率 (fraction 0-1, 三维取均值)
    for s in servers:
        u = (300.0 - float(s[1]) - float(s[2]) - float(s[3])) / 300.0
        utils.append(max(0.0, min(1.0, u)))

    active = sum(1 for u in utils if u > 0.05)  # 利用率 > 5% 视为活跃
    mean_u = sum(utils) / len(utils) if utils else 0.0
    if len(utils) > 1:
        var = sum((u - mean_u) ** 2 for u in utils) / len(utils)
    else:
        var = 0.0
    stddev = math.sqrt(var)

    # 简化能耗：活跃服务器的 idle 基线 + 动态项 (cluster 总利用率) + 热点惩罚
    energy = (
        ENERGY_IDLE_PER_ACTIVE * active
        + ENERGY_DYNAMIC_WEIGHT * sum(u for u in utils if u > 0.05)
        + ENERGY_HOTSPOT_WEIGHT * stddev
    )
    return {
        "energy": energy,
        "active": float(active),
        "mean_util": mean_u,
        "stddev": stddev,
    }


# =============================================================================
# 主入口 / Public API
# =============================================================================

def run_benchmark(
    *,
    seeds: list[int],
    distributions: list[str],
    algorithms: list[str],
    output_path: str | Path = "benchmark/results/metrics.csv",
    requests_per_scenario: int = DEFAULT_REQUESTS_PER_SCENARIO,
    n_servers: int = DEFAULT_N_SERVERS,
    init_util_range: tuple[float, float] = DEFAULT_INIT_UTIL_RANGE,
    churn_prob: float = DEFAULT_CHURN_PROB,
    sla_headroom: float = DEFAULT_SLA_HEADROOM,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for distribution in distributions:
            initial_cluster = _initial_cluster(
                seed, distribution,
                n_servers=n_servers,
                init_util_range=init_util_range,
            )
            requests = _make_requests(seed, distribution, requests_per_scenario)
            for algorithm in algorithms:
                rows.append(_run_algorithm(
                    seed=seed,
                    distribution=distribution,
                    algorithm=algorithm,
                    initial_cluster=initial_cluster,
                    requests=requests,
                    churn_prob=churn_prob,
                    sla_headroom=sla_headroom,
                    init_util_range=init_util_range,
                ))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _is_aiops_aware(algorithm: str) -> bool:
    return algorithm.endswith("-aiops")


def _build_aiops_global_state(
    cluster: _Cluster,
    placement_algorithm: str,
    cumulative_rejects: int,
    cumulative_sla_per_dim: dict[str, int],
) -> dict:
    """从 cluster 当前状态合成 OpsSnapshot，喂给 agent_aiops.observe_ops_state。"""
    cpu_util = sum(100.0 - s[1] for s in cluster.servers) / len(cluster.servers) / 100
    mem_util = sum(100.0 - s[2] for s in cluster.servers) / len(cluster.servers) / 100
    net_util = sum(100.0 - s[3] for s in cluster.servers) / len(cluster.servers) / 100
    utils = [
        max(0.0, min(1.0, (300.0 - s[1] - s[2] - s[3]) / 300.0))
        for s in cluster.servers
    ]
    active = sum(1 for u in utils if u > 0.05)
    return {
        "service_placement_algorithm": placement_algorithm,
        "active_cpu_util": cpu_util,
        "active_mem_util": mem_util,
        "active_net_util": net_util,
        "active_servers": active,
        "current_auto_migrations": 0,
        "current_consolidation_migrations": 0,
        "rescheduled_services": 0,
        "rejected_services": cumulative_rejects,
        "ops_sla_violations": float(cumulative_sla_per_dim.get("cpu", 0)),
        "mem_sla_violations": float(cumulative_sla_per_dim.get("mem", 0)),
        "net_sla_violations": float(cumulative_sla_per_dim.get("net", 0)),
    }


def _run_algorithm(
    *,
    seed: int,
    distribution: str,
    algorithm: str,
    initial_cluster: _Cluster,
    requests: list[ServiceRow],
    churn_prob: float,
    sla_headroom: float,
    init_util_range: tuple[float, float],
) -> dict[str, float | int | str]:
    selector = _algorithm(algorithm, seed=seed, distribution=distribution)
    cluster = _clone_cluster(initial_cluster)
    churn_rng = random.Random(f"{seed}:{distribution}:{algorithm}:churn")

    aiops_aware = _is_aiops_aware(algorithm)
    observe_ops_state = None
    if aiops_aware:
        from agent_aiops import (
            init_agent as _init_aiops,
            observe_ops_state as _observe_ops_state,
        )
        _init_aiops(
            model_name="heuristic",
            backend="rule",
            enable_tracing=False,
            window_size=8,
            recommendation_cooldown=0,
        )
        observe_ops_state = _observe_ops_state

    rejects = 0
    fallbacks = 0
    selected = 0
    sla_violations = 0
    sla_per_dim = {"cpu": 0, "mem": 0, "net": 0}
    overload_attempts = 0     # 算法返回了 sid 但实际放不下 (理论上不应出现)
    agent_escalations = 0
    agent_sync_calls = 0
    memory_used_count = 0
    retrieved_episode_count = 0
    aiops_critic_triggered_count = 0
    aiops_aware_count = 0
    latencies: list[float] = []
    energies: list[float] = []
    actives: list[float] = []
    stddevs: list[float] = []

    for tick, service in enumerate(requests):
        cluster.churn(churn_rng, churn_prob)

        snapshot = cluster.snapshot()

        # 如果是 AIOps-aware 算法，先观察当前 ops state 拿到 insight
        aiops_insight = None
        if aiops_aware and observe_ops_state is not None:
            global_state = _build_aiops_global_state(cluster, algorithm, rejects, sla_per_dim)
            aiops_insight = observe_ops_state(
                global_state,
                server_snapshots_raw=snapshot,
                tick=tick,
            )

        t0 = time.perf_counter()
        if aiops_aware:
            sid = selector(snapshot, service, aiops_insight_raw=aiops_insight)
        else:
            sid = selector(snapshot, service)
        latencies.append((time.perf_counter() - t0) * 1000)

        decision = _last_decision(algorithm)
        if decision.get("agent_escalation_needed") is True:
            agent_escalations += 1
        if decision.get("backend") == "hybrid" and decision.get("fast_path_used") is False:
            agent_sync_calls += 1
        if decision.get("memory_used") is True:
            memory_used_count += 1
        retrieved_episode_count += int(decision.get("retrieved_episode_count") or 0)
        if decision.get("aiops_aware") is True:
            aiops_aware_count += 1
        if decision.get("aiops_critic_triggered") is True:
            aiops_critic_triggered_count += 1

        if sid == -1:
            fallbacks += 1
        elif sid == -2:
            rejects += 1
        elif sid >= 0:
            placed = cluster.place(int(sid), service, tick)
            if not placed:
                # scheduler 给了一个塞不下的 sid -- 计入 overload，请求当成 fallback
                overload_attempts += 1
                fallbacks += 1
            else:
                selected += 1
                # 检查放置后的剩余 headroom
                for s in cluster.servers:
                    if int(s[0]) == int(sid):
                        if s[1] < sla_headroom:
                            sla_violations += 1
                            sla_per_dim["cpu"] += 1
                        elif s[2] < sla_headroom:
                            sla_violations += 1
                            sla_per_dim["mem"] += 1
                        elif s[3] < sla_headroom:
                            sla_violations += 1
                            sla_per_dim["net"] += 1
                        break
        else:
            # 未知返回值，统一兜底
            fallbacks += 1

        telem = _tick_telemetry(cluster.snapshot())
        energies.append(telem["energy"])
        actives.append(telem["active"])
        stddevs.append(telem["stddev"])

    total = len(requests)
    return {
        "seed": seed,
        "distribution": distribution,
        "algorithm": algorithm,
        "requests": total,
        "init_util_lo": init_util_range[0],
        "init_util_hi": init_util_range[1],
        "selected": selected,
        "rejected": rejects,
        "fallbacks": fallbacks,
        "overload_attempts": overload_attempts,
        "rejection_rate": round(rejects / total, 4),
        "fallback_rate": round(fallbacks / total, 4),
        "sla_violations": sla_violations,
        "sla_violation_rate": round(sla_violations / total, 4),
        "sla_violations_cpu": sla_per_dim["cpu"],
        "sla_violations_mem": sla_per_dim["mem"],
        "sla_violations_net": sla_per_dim["net"],
        "total_energy": round(sum(energies), 3),
        "avg_active_servers": round(sum(actives) / len(actives), 3),
        "avg_util_stddev": round(sum(stddevs) / len(stddevs), 4),
        "peak_util_stddev": round(max(stddevs), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 4),
        "p95_latency_ms": round(_percentile(latencies, 95), 4),
        "agent_escalations": agent_escalations,
        "agent_escalation_rate": round(agent_escalations / total, 4),
        "agent_sync_calls": agent_sync_calls,
        "memory_used_count": memory_used_count,
        "memory_usage_rate": round(memory_used_count / total, 4),
        "retrieved_episode_count": retrieved_episode_count,
        "avg_retrieved_episode_count": round(retrieved_episode_count / total, 4),
        "aiops_aware_count": aiops_aware_count,
        "aiops_critic_triggered_count": aiops_critic_triggered_count,
        "aiops_critic_trigger_rate": round(aiops_critic_triggered_count / total, 4),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


# =============================================================================
# 算法工厂 / Algorithm registry
# =============================================================================

def _algorithm(name: str, *, seed: int, distribution: str) -> Callable[[ServerRows, ServiceRow], int]:
    if name == "first-fit":
        return _first_fit
    if name == "balanced-fit":
        return _balanced_fit
    if name in ("AI-phase2", "AI-phase2-aiops"):
        from multi_agent import init_agent, schedule_service

        init_agent(model_name="heuristic", enable_tracing=False)
        return schedule_service
    if name in ("AI-phase3", "AI-phase3-aiops"):
        from agent_memory import init_agent, schedule_service

        memory_path = Path("traces") / f"benchmark-{seed}-{distribution}-{time.time_ns()}.episodes.jsonl"
        # benchmark 内不需要跨 run 持久化 episode，关掉磁盘 I/O。
        # 之前 profile 显示 1433 次 file open 占 6.8% (~420ms)，关掉直接归零。
        init_agent(
            model_name="heuristic",
            backend="hybrid",
            enable_tracing=False,
            memory_path=memory_path,
            persist_episodes=False,
        )
        return schedule_service
    raise ValueError(f"Unknown algorithm: {name}")


def _last_decision(algorithm: str) -> dict:
    if algorithm in ("AI-phase2", "AI-phase2-aiops"):
        from multi_agent import last_decision_dict

        return last_decision_dict()
    if algorithm in ("AI-phase3", "AI-phase3-aiops"):
        from agent_memory import last_decision_dict

        return last_decision_dict()
    return {}


def _first_fit(servers: ServerRows, service: ServiceRow) -> int:
    for server in servers:
        if _fits(server, service):
            return int(server[0])
    return -2


def _balanced_fit(servers: ServerRows, service: ServiceRow) -> int:
    candidates = [server for server in servers if _fits(server, service)]
    if not candidates:
        return -2
    return int(
        min(
            candidates,
            key=lambda s: (
                max(s[1] - service[0], s[2] - service[1], s[3] - service[2])
                - min(s[1] - service[0], s[2] - service[1], s[3] - service[2]),
                sum(s[1:4]),
                s[0],
            ),
        )[0]
    )


def _fits(server: list[float], service: ServiceRow) -> bool:
    return server[1] >= service[0] and server[2] >= service[1] and server[3] >= service[2]


if __name__ == "__main__":
    run_benchmark(
        seeds=[1, 2, 3, 4, 5],
        distributions=["mixed", "cpu-heavy", "memory-heavy", "mixed-burst"],
        algorithms=[
            "first-fit",
            "balanced-fit",
            "AI-phase2",
            "AI-phase3",
            "AI-phase2-aiops",
            "AI-phase3-aiops",
        ],
    )
