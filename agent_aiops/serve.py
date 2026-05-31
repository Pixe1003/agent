"""HTTP server entrypoint for agent_aiops."""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from agent_common.server import build_app


class ObserveRequest(BaseModel):
    active_cpu_util: float = 0.0
    active_mem_util: float = 0.0
    active_net_util: float = 0.0
    rejected_services: int = 0
    ops_sla_violations: float = 0.0
    mem_sla_violations: float = 0.0
    net_sla_violations: float = 0.0
    current_auto_migrations: int = 0
    tick: int | None = None


def _init() -> None:
    from agent_aiops import init_agent as init_aiops
    init_aiops(
        model_name=os.environ.get("MODEL_NAME", "heuristic"),
        backend=os.environ.get("AIOPS_BACKEND", "rule"),
        enable_tracing=os.environ.get("ENABLE_TRACING", "false").lower() == "true",
        window_size=int(os.environ.get("AIOPS_WINDOW", "8")),
        recommendation_cooldown=int(os.environ.get("AIOPS_COOLDOWN", "0")),
    )


app = build_app(agent_name="agent_aiops", init_callback=_init)


@app.post("/observe")
def observe(req: ObserveRequest) -> dict[str, Any]:
    from agent_aiops import observe_ops_state
    insight = observe_ops_state(req.model_dump(exclude={"tick"}), tick=req.tick)
    # drop evidence (太大) 节省传输
    insight.pop("evidence", None)
    return insight


@app.get("/stats")
def stats() -> dict[str, Any]:
    from agent_aiops import aiops_stats
    return aiops_stats()


@app.get("/alerts")
def alerts() -> dict[str, Any]:
    from agent_aiops import current_alerts
    return {"alerts": current_alerts()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent_aiops.serve:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
