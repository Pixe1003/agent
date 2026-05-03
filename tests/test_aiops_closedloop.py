"""验证 agent_aiops 与 multi_agent.critic 的闭环：
observe_ops_state(...) → schedule_service(..., aiops_insight_raw=insight)
→ multi_agent 的 _aiops_critic_check 在压力下收紧 critic 安全边际。
"""
from agent_aiops import init_agent as init_aiops, observe_ops_state
from multi_agent import (
    init_agent as init_scheduler,
    schedule_service,
    last_decision_dict,
    hybrid_stats,
)


def _fresh_init():
    init_scheduler(model_name="heuristic", enable_tracing=False)
    init_aiops(
        model_name="heuristic",
        backend="rule",
        enable_tracing=False,
        recommendation_cooldown=0,
    )


def test_aiops_signal_reaches_decision_metadata_when_no_pressure():
    """无压力时 aiops_aware=True 但 aiops_critic_triggered=False。"""
    _fresh_init()
    insight = observe_ops_state({
        "active_cpu_util": 0.30,
        "active_mem_util": 0.30,
        "active_net_util": 0.30,
    })
    sid = schedule_service(
        [[0, 80.0, 80.0, 80.0]],
        [10.0, 10.0, 10.0],
        None,
        None,
        insight,
    )
    decision = last_decision_dict()
    assert sid == 0
    assert decision["aiops_aware"] is True
    assert decision["aiops_critic_triggered"] is False
    assert decision["aiops_critic_revisions"] == 0
    assert decision["aiops_risk_level"] == "low"


def test_aiops_network_pressure_triggers_critic_revise():
    """AIOps 报 network-pressure 且 NET headroom 不足时，critic 被收紧。

    balanced-fit 默认会按 'min(post_util_max - post_util_min)' 评分，
    server 0 的三维资源分布更平衡，会被 balanced-fit 优先选中；
    但 server 0 的 NET 余量在放置后只剩 5%，AIOps critic 应该把它排除，
    转向 NET 充裕的 server 1。
    """
    _fresh_init()
    insight = observe_ops_state({
        "active_net_util": 0.95,           # 报 network-pressure
        "current_auto_migrations": 4,      # 加权 migration-watch
        "net_sla_violations": 1,           # sla-risk 同时报
    })
    assert "network-pressure" in insight["risk_tags"]

    # server 0：三维分布平衡 (20/20/25)，balanced-fit 优选；放置后 NET=5%（紧张）
    # server 1：三维都充裕 (70/70/70)，但 spread 更大；放置后 NET=50%（安全）
    sid = schedule_service(
        [[0, 20.0, 20.0, 25.0],
         [1, 70.0, 70.0, 70.0]],
        [10.0, 10.0, 20.0],
        None,
        None,
        insight,
    )
    decision = last_decision_dict()
    assert decision["aiops_aware"] is True
    assert "network-pressure" in decision["aiops_risk_tags"]
    # critic 应该至少 revise 一次（把 server 0 排除掉）
    assert decision["aiops_critic_triggered"] is True
    assert decision["aiops_critic_revisions"] >= 1
    # 最终应该选到 NET 更充裕的 server 1
    assert sid == 1


def test_aiops_pressure_can_force_reject_when_no_safe_candidate():
    """所有候选都不满足 AIOps 收紧的安全边际时，应 reject 而不是硬塞。"""
    _fresh_init()
    insight = observe_ops_state({
        "active_net_util": 0.95,
        "net_sla_violations": 1,
    })
    sid = schedule_service(
        [[0, 80.0, 80.0, 25.0]],   # 只有一台，NET 余量 25%；放置后 5% < 15% margin
        [10.0, 10.0, 20.0],
        None, None, insight,
    )
    decision = last_decision_dict()
    assert sid == -2  # reject
    assert decision["aiops_aware"] is True
    assert decision["aiops_critic_triggered"] is True
    assert decision["aiops_critic_revisions"] >= 1


def test_aiops_insight_omitted_keeps_legacy_behavior():
    """不传 aiops_insight_raw 时，行为退化到 legacy (aiops_aware=False)。"""
    _fresh_init()
    sid = schedule_service([[0, 80.0, 80.0, 80.0]], [10.0, 10.0, 10.0])
    decision = last_decision_dict()
    assert sid == 0
    assert decision["aiops_aware"] is False
    assert decision["aiops_critic_triggered"] is False


def test_hybrid_stats_track_aiops_aware_decisions():
    """hybrid_stats 应该记录 AIOps-aware 决策数和触发率。"""
    _fresh_init()

    # 第 1 次：无压力 — aiops_aware=True 但 not triggered
    light = observe_ops_state({"active_cpu_util": 0.20})
    schedule_service([[0, 80.0, 80.0, 80.0]], [10.0, 10.0, 10.0], None, None, light)

    # 第 2 次：网络压力 — 触发
    # server 0: net_free=23 → resource_distance=7 (最优先选中), NET residual=13 < 15% margin → AIOps revise
    # server 1: net_free=80 → resource_distance=30 (次选), NET residual=70 ≥ 15% → 通过
    heavy = observe_ops_state({"active_net_util": 0.95, "net_sla_violations": 1})
    schedule_service(
        [[0, 30.0, 30.0, 23.0], [1, 50.0, 50.0, 80.0]],
        [10.0, 10.0, 10.0],
        None, None, heavy,
    )

    stats = hybrid_stats()
    assert stats["aiops_aware_decisions"] == 2
    assert stats["aiops_critic_triggered_decisions"] >= 1
    assert stats["aiops_aware_ratio"] == 1.0
    assert stats["aiops_critic_trigger_ratio"] >= 0.5
    assert "network-pressure" in stats["aiops_risk_tag_counts"]
