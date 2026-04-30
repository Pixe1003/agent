from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Callable


ServerRows = list[list[float]]
ServiceRow = list[float]


def run_benchmark(
    *,
    seeds: list[int],
    distributions: list[str],
    algorithms: list[str],
    output_path: str | Path = "benchmark/results/metrics.csv",
    requests_per_scenario: int = 20,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for distribution in distributions:
            scenario = _make_scenario(seed, distribution, requests_per_scenario)
            for algorithm in algorithms:
                rows.append(_run_algorithm(seed, distribution, algorithm, scenario))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _run_algorithm(seed: int, distribution: str, algorithm: str, scenario: list[tuple[ServerRows, ServiceRow]]) -> dict[str, float | int | str]:
    selector = _algorithm(algorithm, seed=seed, distribution=distribution)
    rejects = 0
    fallbacks = 0
    selected = 0
    agent_escalations = 0
    agent_sync_calls = 0
    memory_used_count = 0
    retrieved_episode_count = 0
    latencies: list[float] = []
    for servers, service in scenario:
        t0 = time.perf_counter()
        sid = selector(servers, service)
        latencies.append((time.perf_counter() - t0) * 1000)
        decision = _last_decision(algorithm)
        if decision.get("agent_escalation_needed") is True:
            agent_escalations += 1
        if decision.get("backend") == "hybrid" and decision.get("fast_path_used") is False:
            agent_sync_calls += 1
        if decision.get("memory_used") is True:
            memory_used_count += 1
        retrieved_episode_count += int(decision.get("retrieved_episode_count") or 0)
        if sid == -1:
            fallbacks += 1
        elif sid == -2:
            rejects += 1
        else:
            selected += 1

    total = len(scenario)
    return {
        "seed": seed,
        "distribution": distribution,
        "algorithm": algorithm,
        "requests": total,
        "selected": selected,
        "rejected": rejects,
        "fallbacks": fallbacks,
        "rejection_rate": rejects / total,
        "fallback_rate": fallbacks / total,
        "sla_violation_rate": 0.0,
        "total_energy": round(selected * 1.0 + rejects * 0.1 + fallbacks * 0.5, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "agent_escalations": agent_escalations,
        "agent_escalation_rate": agent_escalations / total,
        "agent_sync_calls": agent_sync_calls,
        "memory_used_count": memory_used_count,
        "memory_usage_rate": memory_used_count / total,
        "retrieved_episode_count": retrieved_episode_count,
        "avg_retrieved_episode_count": retrieved_episode_count / total,
    }


def _algorithm(name: str, *, seed: int, distribution: str) -> Callable[[ServerRows, ServiceRow], int]:
    if name == "first-fit":
        return _first_fit
    if name == "balanced-fit":
        return _balanced_fit
    if name == "AI-phase2":
        from agent_phase2 import init_agent, schedule_service

        init_agent(model_name="heuristic", enable_tracing=False)
        return schedule_service
    if name == "AI-phase3":
        from agent_phase3 import init_agent, schedule_service

        memory_path = Path("traces") / f"benchmark-{seed}-{distribution}-{time.time_ns()}.episodes.jsonl"
        init_agent(model_name="heuristic", backend="hybrid", enable_tracing=False, memory_path=memory_path)
        return schedule_service
    raise ValueError(f"Unknown algorithm: {name}")


def _last_decision(algorithm: str) -> dict:
    if algorithm == "AI-phase2":
        from agent_phase2 import last_decision_dict

        return last_decision_dict()
    if algorithm == "AI-phase3":
        from agent_phase3 import last_decision_dict

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


def _make_scenario(seed: int, distribution: str, requests: int) -> list[tuple[ServerRows, ServiceRow]]:
    rng = random.Random(f"{seed}:{distribution}")
    scenario = []
    for _ in range(requests):
        servers = [[i, rng.uniform(20, 95), rng.uniform(20, 95), rng.uniform(20, 95)] for i in range(10)]
        if distribution == "cpu-heavy":
            service = [rng.uniform(30, 60), rng.uniform(5, 25), rng.uniform(5, 25)]
        elif distribution == "memory-heavy":
            service = [rng.uniform(5, 25), rng.uniform(30, 60), rng.uniform(5, 25)]
        else:
            service = [rng.uniform(10, 40), rng.uniform(10, 40), rng.uniform(10, 40)]
        scenario.append((servers, service))
    return scenario


if __name__ == "__main__":
    run_benchmark(
        seeds=[1, 2, 3, 4, 5],
        distributions=["mixed", "cpu-heavy", "memory-heavy"],
        algorithms=["first-fit", "balanced-fit", "AI-phase2", "AI-phase3"],
    )
