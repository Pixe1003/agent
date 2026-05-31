"""HTTP server entrypoint for agent_sft (LLM 推理服务，GPU 节点专用)."""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from agent_common.server import build_app


class InferRequest(BaseModel):
    servers: list[list[float]]
    service: list[float]


def _init() -> None:
    from agent_sft import init_agent
    init_agent(
        model_path=os.environ.get("MODEL_PATH", "/var/lib/agent/qwen25-1p5b-sched-merged-q4.gguf"),
        n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),
        n_threads=int(os.environ.get("N_THREADS", "0")) or None,
        n_ctx=int(os.environ.get("N_CTX", "2048")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "128")),
        temperature=float(os.environ.get("TEMPERATURE", "0.0")),
        enable_tracing=os.environ.get("ENABLE_TRACING", "false").lower() == "true",
    )


def _ready() -> bool:
    from agent_sft.scheduler import _LLM
    return _LLM is not None


app = build_app(agent_name="agent_sft", init_callback=_init, readiness_check=_ready)


@app.post("/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    from agent_sft import schedule_service, last_decision_dict
    sid = schedule_service(req.servers, req.service)
    d = last_decision_dict()
    d["server_id_returned"] = sid
    return d


@app.get("/stats")
def stats() -> dict[str, Any]:
    from agent_sft import sft_stats
    return sft_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent_sft.serve:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
