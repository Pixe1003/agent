"""验证 FastAPI server factory + Prometheus metrics + 4 个 agent 的 serve.py。

不依赖真实 K8s / Docker，只测 in-process FastAPI app。
"""
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi 未安装，跳过 HTTP 测试", allow_module_level=True)

from agent_common.server import build_app


# =============================================================================
# server factory
# =============================================================================

def test_build_app_exposes_health_ready_metrics():
    app = build_app(agent_name="test_agent")
    client = TestClient(app)
    # startup 触发 ready
    with client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "agent_ready" in r.text
        assert 'agent="test_agent"' in r.text


def test_build_app_readiness_check_can_block_ready():
    state = {"ok": False}
    app = build_app(agent_name="test", readiness_check=lambda: state["ok"])
    client = TestClient(app)
    with client:
        assert client.get("/readyz").status_code == 503
        state["ok"] = True
        assert client.get("/readyz").status_code == 200


def test_metrics_record_request_count_and_latency():
    app = build_app(agent_name="test_count")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    with client:
        for _ in range(3):
            assert client.get("/ping").status_code == 200
        metrics_text = client.get("/metrics").text

    assert 'agent_requests_total{agent="test_count",endpoint="/ping",status="200"} 3' in metrics_text
    assert 'agent_request_latency_seconds_count{agent="test_count",endpoint="/ping"} 3' in metrics_text


# =============================================================================
# multi_agent.serve
# =============================================================================

def test_multi_agent_serve_schedule_endpoint():
    from multi_agent.serve import app
    client = TestClient(app)
    with client:
        r = client.post("/schedule", json={
            "servers": [[0, 80.0, 80.0, 80.0], [1, 70.0, 70.0, 70.0]],
            "service": [10.0, 10.0, 10.0],
            "aiops_risk_tags": [],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["action"] == "select"
        assert body["server_id_returned"] in (0, 1)
        assert body["aiops_critic_triggered"] is False


def test_multi_agent_serve_schedule_with_aiops_pressure_triggers_critic():
    from multi_agent.serve import app
    client = TestClient(app)
    with client:
        r = client.post("/schedule", json={
            "servers": [[0, 20.0, 20.0, 25.0], [1, 70.0, 70.0, 70.0]],
            "service": [10.0, 10.0, 20.0],
            "aiops_risk_tags": ["network-pressure", "sla-risk"],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["aiops_critic_triggered"] is True
        # 收紧后应该转向 net 充裕的 server 1
        assert body["server_id_returned"] == 1


def test_multi_agent_stats_endpoint():
    from multi_agent.serve import app
    client = TestClient(app)
    with client:
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total_decisions" in body
        assert "aiops_aware_decisions" in body


# =============================================================================
# agent_aiops.serve
# =============================================================================

def test_agent_aiops_observe_returns_risk_tags():
    from agent_aiops.serve import app
    client = TestClient(app)
    with client:
        r = client.post("/observe", json={
            "active_net_util": 0.95,
            "net_sla_violations": 1.0,
        })
        assert r.status_code == 200, r.text
        insight = r.json()
        assert "risk_tags" in insight
        assert "network-pressure" in insight["risk_tags"]
        assert insight["risk_level"] in {"medium", "high", "critical"}


def test_agent_aiops_alerts_endpoint():
    from agent_aiops.serve import app
    client = TestClient(app)
    with client:
        # 先 observe 一次让 window 里有数据
        client.post("/observe", json={"active_net_util": 0.95, "net_sla_violations": 1.0})
        r = client.get("/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()
