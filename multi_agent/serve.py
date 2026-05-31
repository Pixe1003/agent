"""HTTP server entrypoint for multi_agent.

    python -m multi_agent.serve   # 启动 0.0.0.0:8080
    或在 Docker 里通过 ENTRYPOINT 启动。

暴露：
    POST /schedule  → SchedulerOutput
    GET  /stats     → hybrid_stats()
    GET  /healthz, /readyz, /metrics  (来自 agent_common.server)
"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from agent_common.server import build_app


class ScheduleRequest(BaseModel):
    servers: list[list[float]]
    service: list[float]
    aiops_risk_tags: list[str] = []


class ScheduleResponse(BaseModel):
    action: str
    server_id: int | None = None
    reasoning: str = ""
    latency_ms: float = 0.0
    server_id_returned: int = -1
    aiops_critic_triggered: bool = False
    aiops_risk_tags: list[str] = []


def _init() -> None:
    from multi_agent import init_agent
    init_agent(
        model_name=os.environ.get("MODEL_NAME", "heuristic"),
        enable_tracing=os.environ.get("ENABLE_TRACING", "false").lower() == "true",
    )


app = build_app(agent_name="multi_agent", init_callback=_init)


@app.post("/schedule", response_model=ScheduleResponse)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    from multi_agent import schedule_service, last_decision_dict
    aiops_insight: dict[str, Any] | None = None
    if req.aiops_risk_tags:
        aiops_insight = {
            "risk_tags": list(req.aiops_risk_tags),
            "risk_level": "high",
            "risk_score": 0.7,
            "active_alerts": [
                {"tag": t, "occurrence_count": 1, "risk_score": 0.7}
                for t in req.aiops_risk_tags
            ],
        }
    sid = schedule_service(req.servers, req.service, None, None, aiops_insight)
    d = last_decision_dict()
    return ScheduleResponse(
        action=str(d.get("action", "fallback")),
        server_id=d.get("server_id"),
        reasoning=str(d.get("reasoning", "")),
        latency_ms=float(d.get("latency_ms", 0.0)),
        server_id_returned=sid,
        aiops_critic_triggered=bool(d.get("aiops_critic_triggered", False)),
        aiops_risk_tags=list(d.get("aiops_risk_tags") or []),
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    from multi_agent import hybrid_stats
    return hybrid_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "multi_agent.serve:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("WORKERS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
