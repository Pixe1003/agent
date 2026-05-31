"""HTTP server entrypoint for agent_memory."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from agent_common.server import build_app


class RetrieveRequest(BaseModel):
    query_text: str
    query_features: list[float]
    top_k: int = 3


def _init() -> None:
    # 在 K8s 里 EpisodicMemory path 指向 PVC mount 点
    memory_path = os.environ.get("MEMORY_PATH", "/var/lib/agent/episodes.jsonl")
    Path(memory_path).parent.mkdir(parents=True, exist_ok=True)


app = build_app(agent_name="agent_memory", init_callback=_init)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    from agent_memory.memory import EpisodicMemory
    memory_path = os.environ.get("MEMORY_PATH", "/var/lib/agent/episodes.jsonl")
    persist = os.environ.get("PERSIST_EPISODES", "true").lower() == "true"
    store = EpisodicMemory(path=memory_path, persist=persist)
    matches = store.retrieve(req.query_text, req.query_features, top_k=req.top_k)
    return {"episodes": [m.model_dump() for m in matches]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent_memory.serve:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
