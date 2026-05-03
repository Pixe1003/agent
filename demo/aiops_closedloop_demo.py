"""端到端 demo：实时监控 + 异常检测 → 调度器 critic 闭环。

跑法：
    python -m demo.aiops_closedloop_demo
或:
    .\.venv\Scripts\python.exe demo\aiops_closedloop_demo.py

输出对比同一 workload / 同一 cluster 初始状态下，
   Run A: 调度器单独工作  (multi_agent only)
   Run B: AIOps 信号闭环到 critic  (multi_agent + observe_ops_state)
两者的 SLA 违约率、拒绝率、能耗、延迟和 critic 触发统计。

设计要点：
- 集群初始利用率 60-80%，已经处于"亚饱和"区间；
- 工作负载使用 mixed-burst，30% 突发请求 (40-60% 资源)；
- 每个 tick 都 observe_ops_state，把 risk_tags + active_alerts 喂给 schedule_service
  的新参数 aiops_insight_raw；
- multi_agent 的 _aiops_critic_check 在基础 critic 通过后再加一道安全边际检查，
  当 AIOps 报警时强制要求残余 headroom >= 15%。
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent_aiops import (
    init_agent as init_aiops,
    observe_ops_state,
    last_insight_dict,
    aiops_stats,
)
from multi_agent import (
    init_agent as init_scheduler,
    schedule_service,
    last_decision_dict,
    hybrid_stats,
)

# 复用 benchmark.runner 里的 cluster 仿真和工作负载生成器，确保两次跑法一致
from benchmark.runner import (
    _Cluster,
    _initial_cluster,
    _make_requests,
    _tick_telemetry,
    _clone_cluster,
    DEFAULT_SLA_HEADROOM,
)


# ----- Demo 配置 -----
N_SERVERS = 10
INIT_UTIL_RANGE = (60.0, 80.0)        # 高初始压力，AIOps 容易触发警报
REQUESTS = 200
CHURN_PROB = 0.05
DISTRIBUTION = "mixed-burst"
SEED = 7
AIOPS_OBSERVE_EVERY = 1               # 每个 tick 都观察 (recommendation_cooldown=0)


def _build_global_state(
    cluster: _Cluster,
    placement_algorithm: str,
    cumulative_rejects: int,
    cumulative_sla: dict[str, int],
) -> dict:
    """把当前 cluster 状态和累计统计组合成 OpsSnapshot 形态，喂给 AIOps。"""
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
        "ops_sla_violations": float(cumulative_sla.get("cpu", 0)),
        "mem_sla_violations": float(cumulative_sla.get("mem", 0)),
        "net_sla_violations": float(cumulative_sla.get("net", 0)),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100)))
    return sorted_values[idx]


def _run(label: str, *, use_aiops: bool, base_cluster: _Cluster, requests: list[list[float]]) -> dict:
    print(f"\n========== {label} ==========")
    init_scheduler(model_name="heuristic", enable_tracing=False)
    init_aiops(
        model_name="heuristic",
        backend="rule",
        enable_tracing=False,
        window_size=8,
        recommendation_cooldown=0,  # demo 里不抑制 recommendation
    )

    cluster = _clone_cluster(base_cluster)
    churn_rng = random.Random(f"{SEED}:{DISTRIBUTION}:demo")  # 两次跑用同一 churn 序列以保证公平

    rejects = 0
    fallbacks = 0
    selected = 0
    sla_violations = 0
    sla_per_dim = {"cpu": 0, "mem": 0, "net": 0}
    aiops_critic_triggered_count = 0
    energies: list[float] = []
    latencies: list[float] = []
    risk_level_history: list[str] = []

    for tick, service in enumerate(requests):
        cluster.churn(churn_rng, CHURN_PROB)
        snapshot = cluster.snapshot()

        aiops_insight = None
        if use_aiops and tick % AIOPS_OBSERVE_EVERY == 0:
            global_state = _build_global_state(cluster, "AI-multi", rejects, sla_per_dim)
            aiops_insight = observe_ops_state(
                global_state,
                server_snapshots_raw=snapshot,
                tick=tick,
            )
            risk_level_history.append(aiops_insight.get("risk_level", "low"))

        t0 = time.perf_counter()
        sid = schedule_service(snapshot, service, None, None, aiops_insight)
        latencies.append((time.perf_counter() - t0) * 1000)

        decision = last_decision_dict()
        if decision.get("aiops_critic_triggered") is True:
            aiops_critic_triggered_count += 1

        if sid == -1:
            fallbacks += 1
        elif sid == -2:
            rejects += 1
        elif sid >= 0:
            placed = cluster.place(int(sid), service, tick)
            if placed:
                selected += 1
                # 检查 placed server 的 post-placement headroom
                for s in cluster.servers:
                    if int(s[0]) == int(sid):
                        if s[1] < DEFAULT_SLA_HEADROOM:
                            sla_per_dim["cpu"] += 1
                            sla_violations += 1
                        elif s[2] < DEFAULT_SLA_HEADROOM:
                            sla_per_dim["mem"] += 1
                            sla_violations += 1
                        elif s[3] < DEFAULT_SLA_HEADROOM:
                            sla_per_dim["net"] += 1
                            sla_violations += 1
                        break
            else:
                fallbacks += 1

        telem = _tick_telemetry(cluster.snapshot())
        energies.append(telem["energy"])

    total = len(requests)
    avg_lat = sum(latencies) / len(latencies)
    p95_lat = _percentile(latencies, 95)

    print(f"  Placed:           {selected}/{total} ({selected / total * 100:.1f}%)")
    print(f"  Rejected:         {rejects}/{total} ({rejects / total * 100:.1f}%)")
    print(f"  Fallbacks:        {fallbacks}/{total}")
    print(f"  SLA violations:   {sla_violations}/{total} ({sla_violations / total * 100:.1f}%)")
    print(f"  SLA by dim:       cpu={sla_per_dim['cpu']} mem={sla_per_dim['mem']} net={sla_per_dim['net']}")
    print(f"  Total energy:     {sum(energies):.2f}")
    print(f"  Avg latency:      {avg_lat:.4f} ms")
    print(f"  P95 latency:      {p95_lat:.4f} ms")
    if use_aiops:
        print(f"  AIOps observations:        {len(risk_level_history)}")
        print(f"  AIOps critic triggered:    {aiops_critic_triggered_count}/{total} "
              f"({aiops_critic_triggered_count / total * 100:.1f}%)")
        risk_counts: dict[str, int] = {}
        for level in risk_level_history:
            risk_counts[level] = risk_counts.get(level, 0) + 1
        print(f"  Risk level distribution:   {risk_counts}")
        final_insight = last_insight_dict()
        print(f"  Final risk:                {final_insight.get('risk_level')} "
              f"score={final_insight.get('risk_score'):.3f} "
              f"tags={final_insight.get('risk_tags')}")
        astats = aiops_stats()
        print(f"  Final active alerts:       {astats['active_alert_tags']}")

    return {
        "selected": selected,
        "rejects": rejects,
        "fallbacks": fallbacks,
        "sla_violations": sla_violations,
        "energy": sum(energies),
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "aiops_triggered": aiops_critic_triggered_count if use_aiops else 0,
    }


def main() -> None:
    print("=" * 64)
    print(" AIOps Closed-Loop Demo")
    print("=" * 64)
    print(f" Setup: {N_SERVERS} servers, init utilization {INIT_UTIL_RANGE[0]}%-{INIT_UTIL_RANGE[1]}%")
    print(f"        {DISTRIBUTION} workload, {REQUESTS} requests, churn={CHURN_PROB}")
    print(f"        SLA threshold: post-placement headroom < {DEFAULT_SLA_HEADROOM}% on any dim")
    print(f"        AIOps safety margin: 15% (1.5x for persistent alerts)")

    # 共享同一份初始 cluster + workload，确保 A/B 公平
    base_cluster = _initial_cluster(
        SEED, DISTRIBUTION,
        n_servers=N_SERVERS,
        init_util_range=INIT_UTIL_RANGE,
    )
    requests = _make_requests(SEED, DISTRIBUTION, REQUESTS)

    blind = _run("Run A: scheduler ONLY (no AIOps signal)",
                 use_aiops=False, base_cluster=base_cluster, requests=requests)
    aware = _run("Run B: scheduler + AIOps closed loop",
                 use_aiops=True, base_cluster=base_cluster, requests=requests)

    print("\n" + "=" * 64)
    print(" Comparison")
    print("=" * 64)
    sla_delta = blind["sla_violations"] - aware["sla_violations"]
    sla_pct = (sla_delta / blind["sla_violations"] * 100) if blind["sla_violations"] else 0.0
    energy_delta = blind["energy"] - aware["energy"]
    rej_delta = aware["rejects"] - blind["rejects"]
    lat_overhead_us = (aware["avg_latency_ms"] - blind["avg_latency_ms"]) * 1000

    print(f"  SLA violations:   {blind['sla_violations']:>3} → {aware['sla_violations']:<3}"
          f"   (Δ {sla_delta:+d}, {sla_pct:+.1f}%)")
    print(f"  Rejected:         {blind['rejects']:>3} → {aware['rejects']:<3}"
          f"   (Δ {rej_delta:+d})")
    print(f"  Selected:         {blind['selected']:>3} → {aware['selected']:<3}")
    print(f"  Total energy:     {blind['energy']:>7.2f} → {aware['energy']:<7.2f}"
          f"   (Δ {energy_delta:+.2f})")
    print(f"  Avg latency:      {blind['avg_latency_ms']:.4f} → {aware['avg_latency_ms']:.4f} ms"
          f"   (overhead {lat_overhead_us:+.1f} μs)")
    print(f"  AIOps interventions: {aware['aiops_triggered']}")
    print()
    if sla_delta > 0:
        tradeoff = (rej_delta / sla_delta) if sla_delta else 0
        print(f"  → AIOps closed loop reduced SLA violations by {sla_pct:.1f}% "
              f"(saved {sla_delta} violations, costs {rej_delta} extra rejects, "
              f"tradeoff {tradeoff:.2f} reject per saved violation).")
    elif sla_delta < 0:
        print(f"  → AIOps closed loop INCREASED SLA violations by {-sla_pct:.1f}% — "
              f"safety_margin may be too aggressive for this workload.")
    else:
        print("  → No SLA difference under this workload — try higher init utilization "
              "or larger burst ratio.")


if __name__ == "__main__":
    main()
