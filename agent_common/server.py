"""FastAPI server factory + Prometheus metrics exporter.

每个 agent 模块的 serve.py 调用 build_app() 拿到一个统一的 FastAPI app，
自动获得：
    GET  /healthz      liveness probe
    GET  /readyz       readiness probe
    GET  /metrics      prometheus exposition
    POST /<endpoint>   业务路由（caller 注册）

设计原则：
- 业务 schema 是 Pydantic v2，FastAPI 自动校验
- 每个业务路由自动加 Histogram + Counter
- 对外 8080 (HTTP), /metrics 单独路径
- 不依赖外部 redis / db / message bus —— K8s MVP 内可以独立运行
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

try:
    from fastapi import FastAPI, Request, Response
    from pydantic import BaseModel
except ImportError as e:
    raise SystemExit(
        "fastapi / pydantic 未安装。pip install fastapi uvicorn"
    ) from e

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
except ImportError as e:
    raise SystemExit(
        "prometheus-client 未安装。pip install prometheus-client"
    ) from e


# ============================================================================
# Metrics
# ============================================================================

REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of business requests handled.",
    ["agent", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "Per-request latency in seconds.",
    ["agent", "endpoint"],
    buckets=(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
)
INFLIGHT = Gauge(
    "agent_inflight_requests",
    "Number of in-flight business requests.",
    ["agent", "endpoint"],
)
READY = Gauge(
    "agent_ready",
    "1 if agent is ready, 0 otherwise.",
    ["agent"],
)


# ============================================================================
# Factory
# ============================================================================

def build_app(
    *,
    agent_name: str,
    init_callback: Callable[[], None] | None = None,
    readiness_check: Callable[[], bool] | None = None,
) -> FastAPI:
    """Build a standardized FastAPI app for an agent service.

    Args:
        agent_name: short identifier (e.g. "multi_agent"), used as Prometheus label.
        init_callback: lazy init called on startup.
        readiness_check: optional callable returning True once the agent is ready
            to handle traffic. If None, agent is considered always ready after init.
    """
    app = FastAPI(title=f"{agent_name} service", version="0.1.0")
    state: dict[str, Any] = {"ready": False}

    @app.on_event("startup")
    def _on_startup() -> None:
        if init_callback is not None:
            init_callback()
        state["ready"] = True
        READY.labels(agent=agent_name).set(1)

    @app.on_event("shutdown")
    def _on_shutdown() -> None:
        READY.labels(agent=agent_name).set(0)
        state["ready"] = False

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        # liveness: 进程活着就 ok
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> Response:
        ok = state["ready"] and (readiness_check is None or readiness_check())
        return Response(status_code=200 if ok else 503, content="ready" if ok else "not_ready")

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        endpoint = request.url.path
        # 不给 /metrics / /healthz / /readyz 自己也加监控（避免噪声）
        if endpoint in ("/metrics", "/healthz", "/readyz"):
            return await call_next(request)
        INFLIGHT.labels(agent=agent_name, endpoint=endpoint).inc()
        t0 = time.perf_counter()
        status = "200"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            REQUEST_LATENCY.labels(agent=agent_name, endpoint=endpoint).observe(elapsed)
            REQUEST_COUNT.labels(agent=agent_name, endpoint=endpoint, status=status).inc()
            INFLIGHT.labels(agent=agent_name, endpoint=endpoint).dec()

    return app
