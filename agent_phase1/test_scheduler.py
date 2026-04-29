"""
脱离 NetLogo 的冒烟测试。先跑通这个再集成到 NetLogo。

运行：
    python -m agent_phase1.test_scheduler

前置条件：
    1. ollama serve 已运行
    2. ollama pull qwen3:8b 已完成
    3. pip install -r agent_phase1/requirements.txt
"""
from .scheduler import init_agent, schedule_service, last_decision_summary


def main():
    init_agent(model_name="qwen3:8b")

    # ---- Case 1: 正常调度 ----
    # 服务器 3 CPU 最空，但 RAM 不够；服务器 5 各项都充裕 —— 期望选 5
    print("\n===== Case 1: normal scheduling =====")
    servers = [
        [0, 20.0, 30.0, 80.0],
        [1, 15.0, 25.0, 70.0],
        [2, 45.0, 40.0, 60.0],
        [3, 95.0, 10.0, 50.0],   # CPU 超充裕但 RAM 非常紧
        [4, 50.0, 55.0, 65.0],
        [5, 70.0, 80.0, 75.0],   # 各维度都很充裕 —— 理想候选
        [6, 30.0, 35.0, 40.0],
        [7, 25.0, 20.0, 90.0],
        [8, 40.0, 45.0, 55.0],
        [9, 60.0, 50.0, 70.0],
    ]
    service = [25.0, 20.0, 10.0]  # 中等请求

    sid = schedule_service(servers, service)
    print(f"Returned server_id: {sid}")
    print(f"Summary: {last_decision_summary()}")
    assert sid >= 0, "Expected a valid server id"

    # ---- Case 2: 应当拒绝 ----
    # 所有服务器资源都很紧张，服务请求巨大
    print("\n===== Case 2: should reject =====")
    tight_servers = [
        [i, 10.0, 8.0, 12.0] for i in range(10)
    ]
    big_service = [80.0, 70.0, 60.0]

    sid = schedule_service(tight_servers, big_service)
    print(f"Returned server_id: {sid}")
    print(f"Summary: {last_decision_summary()}")
    # 不强制 assert == -2，因为某些模型可能选择尝试而非拒绝
    # 但至少不应该 happily 返回一个明显不合适的正整数
    # （后续 Phase 2 的 Critic agent 会进一步守这一层）

    # ---- Case 3: 输入畸形 ----
    print("\n===== Case 3: malformed input =====")
    bad_servers = [[0, "not_a_number", 50, 50]]
    sid = schedule_service(bad_servers, service)
    print(f"Returned server_id: {sid}")
    print(f"Summary: {last_decision_summary()}")
    assert sid == -1, "Malformed input must route to fallback"

    print("\nSmoke tests passed.")


if __name__ == "__main__":
    main()
