import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"


def test_dashboard_static_files_exist():
    assert (DASHBOARD / "index.html").exists()
    assert (DASHBOARD / "styles.css").exists()
    assert (DASHBOARD / "app.js").exists()
    assert (DASHBOARD / "sample-aiops-stream.json").exists()


def test_dashboard_html_exposes_realtime_aiops_panels():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")

    assert "AIOps Realtime Dashboard" in html
    assert 'id="server-grid"' in html
    assert 'id="risk-canvas"' in html
    assert 'id="utilization-canvas"' in html
    assert 'id="aiops-tags"' in html
    assert 'id="recommendations"' in html
    assert 'id="event-stream"' in html
    assert "styles.css" in html
    assert "app.js" in html


def test_dashboard_js_loads_sample_stream_and_renders_expected_sections():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")

    assert "/api/aiops-stream.json?limit=500" in js
    assert "aiops-stream.json" in js
    assert "sample-aiops-stream.json" in js
    assert "pollLiveStream" in js
    assert "loadJson" in js
    assert "renderServerGrid" in js
    assert "renderAIOpsPanel" in js
    assert "drawRiskChart" in js
    assert "drawUtilizationChart" in js
    assert "requestAnimationFrame" in js


def test_sample_aiops_stream_has_realtime_shape():
    data = json.loads((DASHBOARD / "sample-aiops-stream.json").read_text(encoding="utf-8"))

    assert len(data["events"]) >= 6
    first = data["events"][0]
    assert {"tick", "global_state", "aiops", "servers", "events"} <= set(first)
    assert {"risk_level", "risk_score", "risk_tags", "active_alerts", "recommendations"} <= set(first["aiops"])
    assert {"id", "status", "cpu", "mem", "net"} <= set(first["servers"][0])
