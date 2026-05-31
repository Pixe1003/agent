# Agent Infra Toolkit — multi-stage Dockerfile
#
# 提供 6 个构建 target，共享同一个 base 层减少镜像体积：
#   base           - 共享 Python deps (Pydantic / FastAPI / LangChain / prometheus)
#   multi_agent    - 调度核心
#   agent_memory   - episodic memory
#   agent_aiops    - 实时监控
#   agent_sft      - LLM 推理 (单独大镜像，含 llama-cpp-python)
#   mcp_server     - MCP gateway
#
# 用法：
#   docker build -f docker/Dockerfile --target multi_agent -t agent/multi_agent:0.1 .
#   docker build -f docker/Dockerfile --target agent_memory -t agent/memory:0.1 .
#   docker build -f docker/Dockerfile --target agent_sft   -t agent/sft:0.1 .
#
# 镜像大小（实测）：base ≈ 200 MB，普通 agent ≈ 220 MB，sft ≈ 800 MB。

# =============================================================================
# Stage 0: builder
# =============================================================================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# 把核心轻量依赖装到 /opt/venv
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install fastapi uvicorn[standard] prometheus-client


# =============================================================================
# Stage 1: base — 共享给所有非-sft agent
# =============================================================================
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# 复制 venv (含全部 deps)
COPY --from=builder /opt/venv /opt/venv

# 创建非 root 用户
RUN groupadd -r agent && useradd -r -g agent -d /app -s /sbin/nologin agent

WORKDIR /app

# 共用源码层
COPY agent_common ./agent_common
COPY pyproject.toml setup.cfg* /app/

ENV PORT=8080 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info \
    PYTHONPATH=/app

USER agent
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1


# =============================================================================
# Stage 2: multi_agent
# =============================================================================
FROM base AS multi_agent

COPY --chown=agent:agent multi_agent ./multi_agent

ENTRYPOINT ["python", "-m", "multi_agent.serve"]


# =============================================================================
# Stage 3: agent_memory
# =============================================================================
FROM base AS agent_memory

COPY --chown=agent:agent agent_memory ./agent_memory

# Memory 用 PVC 持久化 episodic JSONL
VOLUME ["/var/lib/agent"]
ENV MEMORY_PATH=/var/lib/agent/episodes.jsonl

ENTRYPOINT ["python", "-m", "agent_memory.serve"]


# =============================================================================
# Stage 4: agent_aiops
# =============================================================================
FROM base AS agent_aiops

COPY --chown=agent:agent agent_aiops ./agent_aiops

ENTRYPOINT ["python", "-m", "agent_aiops.serve"]


# =============================================================================
# Stage 5: mcp_server (gateway)
# =============================================================================
FROM base AS mcp_server

COPY --chown=agent:agent multi_agent ./multi_agent
COPY --chown=agent:agent agent_memory ./agent_memory
COPY --chown=agent:agent agent_aiops ./agent_aiops
COPY --chown=agent:agent mcp_server.py /app/mcp_server.py

USER root
RUN /opt/venv/bin/pip install mcp
USER agent

ENTRYPOINT ["python", "mcp_server.py"]


# =============================================================================
# Stage 6: agent_sft (大镜像，单独 stage，含 llama-cpp-python)
# =============================================================================
FROM python:3.13-slim AS sft_builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install fastapi uvicorn[standard] prometheus-client pydantic
# llama-cpp-python (CPU build for portability; GPU build needs CUDA base image)
RUN /opt/venv/bin/pip install llama-cpp-python


FROM python:3.13-slim AS agent_sft

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PORT=8080 \
    HOST=0.0.0.0

COPY --from=sft_builder /opt/venv /opt/venv

RUN groupadd -r agent && useradd -r -g agent -d /app -s /sbin/nologin agent

WORKDIR /app
COPY --chown=agent:agent agent_common ./agent_common
COPY --chown=agent:agent agent_sft ./agent_sft

ENV PYTHONPATH=/app \
    MODEL_PATH=/var/lib/agent/qwen25-1p5b-sched-merged-q4.gguf

VOLUME ["/var/lib/agent"]

USER agent
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/readyz')" || exit 1

ENTRYPOINT ["python", "-m", "agent_sft.serve"]
