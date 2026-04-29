"""
Pydantic schemas for scheduler agent.

约束原则：
1. 所有 LLM 输出必须能通过 Pydantic 验证，否则走 fallback。
2. 所有字段都有 description —— bind_tools 会把 description 喂给 LLM。
3. 用 Field(ge=..., le=...) 做数值范围硬约束，幻觉数字直接拦截。
"""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# State schema —— NetLogo 传过来的数据结构
# =============================================================================

class ServerSnapshot(BaseModel):
    """单台服务器的当前可用资源快照（百分比形式）。"""
    model_config = ConfigDict(extra="forbid")

    server_id: int = Field(..., ge=0, description="Server identifier")
    cpu_free_pct: float = Field(..., ge=0, le=100, description="CPU headroom, 0-100")
    ram_free_pct: float = Field(..., ge=0, le=100, description="RAM headroom, 0-100")
    net_free_pct: float = Field(..., ge=0, le=100, description="Network headroom, 0-100")


class ServiceRequest(BaseModel):
    """待调度的服务请求（占单台服务器容量的百分比）。"""
    model_config = ConfigDict(extra="forbid")

    cpu_pct: float = Field(..., ge=0, le=100, description="CPU demand, 0-100")
    ram_pct: float = Field(..., ge=0, le=100, description="RAM demand, 0-100")
    net_pct: float = Field(..., ge=0, le=100, description="Network demand, 0-100")


class SchedulingContext(BaseModel):
    """Agent 收到的完整上下文。"""
    servers: list[ServerSnapshot]
    service: ServiceRequest
    overutil_threshold_pct: float = Field(
        default=90.0,
        description="A server is considered overloaded if usage exceeds this after deployment"
    )


# =============================================================================
# Tool schemas —— 给 bind_tools 用
# 类名即 tool 名，docstring 即 tool description
# =============================================================================

class select_server(BaseModel):
    """Deploy the incoming service on the chosen server.
    
    Call this tool when you have identified a server that can accommodate
    the service without exceeding the overutilization threshold on any
    resource dimension (CPU, RAM, Network).
    """
    server_id: int = Field(
        ...,
        ge=0,
        description="The ID of the server chosen for deployment. Must be one of the server IDs present in the cluster state.",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="A brief one-sentence justification. Mention the dominant resource dimension that drove the choice (e.g. 'server 3 has the highest CPU headroom and fits the service').",
    )


class reject_service(BaseModel):
    """Reject the incoming service because no server in the cluster can
    accommodate it without exceeding the overutilization threshold.
    
    Only call this tool if you have checked all servers and none can host
    the service on every resource dimension.
    """
    reason: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Explain which resource bottleneck caused the rejection across all candidate servers.",
    )


# =============================================================================
# Internal decision record —— 供 Phase 3 memory 模块消费
# =============================================================================

class SchedulingDecision(BaseModel):
    """Agent 的最终决策，也是写入 memory 的基础数据结构。"""
    action: Literal["select", "reject", "fallback"]
    server_id: int | None = None         # None for reject / fallback
    reasoning: str
    latency_ms: float
    tool_call_succeeded: bool
    raw_llm_response: str | None = None  # 调试用，生产环境可关
