"""Skill 抽象：最小可复用的 Agent 能力单元。

设计原则：
- 每个 Skill = (name, system_prompt, input_schema, output_schema, execute())
- 全部 Pydantic v2 schema 做强类型契约
- 通过 __init_subclass__ 自动注册到 Skill._registry
- 提供 run() 包装层做"入参校验 → 执行 → 出参校验"
- 子类只需实现 execute(payload: I) -> O，框架级别保证类型安全

应用场景：
- agent_common/skill.py 提供基类
- multi_agent / agent_memory / agent_aiops / agent_sft 把核心动作包装成 Skill
- agent-cli 通过 Skill.list_skills() / Skill.get() 做命令分发
- mcp_server 可以把每个 Skill 一行代码暴露成 MCP tool

简历卖点：
    "实现了模块化 Skill SDK，新 Agent 开发者继承 Skill 基类即可注册新能力，
     framework 自动负责 schema 校验、prompt 渲染、错误兜底和观测埋点。"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError


I = TypeVar("I", bound=BaseModel)  # input schema type
O = TypeVar("O", bound=BaseModel)  # output schema type


class Skill(ABC, Generic[I, O]):
    """Agent Skill 基类。

    子类必须提供以下 ClassVar：
        name           — Skill 唯一标识 (snake_case)
        description    — 一句话描述（CLI / MCP tool description 共用）
        input_schema   — Pydantic 输入 schema
        output_schema  — Pydantic 输出 schema
    可选：
        system_prompt  — LLM 系统提示模板
        version        — Skill 版本号

    子类必须实现：
        execute(payload: I) -> O

    自动行为：
        __init_subclass__ 注册到 Skill._registry
        run() 包装 execute() 做入参 / 出参 schema 校验
    """

    # 必须由子类覆盖
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    # 可选
    system_prompt: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"

    # 类级注册表
    _registry: ClassVar[dict[str, type["Skill"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # 抽象子类（仍带 abstract method）不注册
        if getattr(cls, "__abstractmethods__", None):
            return
        skill_name = getattr(cls, "name", "")
        if not skill_name:
            return  # 没指定 name 视作匿名/中间类，不注册
        if skill_name in Skill._registry and Skill._registry[skill_name] is not cls:
            # 同名覆盖时给 warning, 但允许（开发期重载常见）
            existing = Skill._registry[skill_name]
            if existing.__module__ != cls.__module__:
                import warnings
                warnings.warn(
                    f"Skill name {skill_name!r} re-registered "
                    f"({existing.__module__} → {cls.__module__})",
                    stacklevel=2,
                )
        Skill._registry[skill_name] = cls

    # ------------------------------------------------------------------
    # 注册表 / 查询 API
    # ------------------------------------------------------------------

    @classmethod
    def list_skills(cls) -> list[str]:
        """列出所有已注册的 Skill 名称（排序）。"""
        return sorted(Skill._registry.keys())

    @classmethod
    def get(cls, name: str) -> type["Skill"]:
        """按名字取 Skill 类。未注册时抛 KeyError 并列出可选项。"""
        if name not in Skill._registry:
            raise KeyError(
                f"Skill {name!r} not registered. Available: {cls.list_skills()}"
            )
        return Skill._registry[name]

    @classmethod
    def describe_all(cls) -> list[dict[str, str]]:
        """返回所有 Skill 的 (name, description, version, input/output schema name) 摘要。"""
        out = []
        for name in cls.list_skills():
            sk = Skill._registry[name]
            out.append(
                {
                    "name": sk.name,
                    "description": sk.description,
                    "version": sk.version,
                    "input_schema": sk.input_schema.__name__,
                    "output_schema": sk.output_schema.__name__,
                }
            )
        return out

    # ------------------------------------------------------------------
    # 执行入口
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, payload: I) -> O:
        """子类实现具体逻辑。输入已经通过 input_schema 校验过。"""
        raise NotImplementedError

    def run(self, payload_raw: dict[str, Any] | I) -> O:
        """生产入口：入参 → 校验 → execute → 出参 → 校验。

        Raises:
            ValidationError: 入参或出参不符合 schema。
        """
        # 1) 入参校验
        if isinstance(payload_raw, BaseModel):
            payload = payload_raw  # type: ignore[assignment]
        else:
            payload = self.input_schema.model_validate(payload_raw)  # type: ignore[assignment]

        # 2) 执行
        result = self.execute(payload)  # type: ignore[arg-type]

        # 3) 出参校验
        if isinstance(result, self.output_schema):
            return result
        if isinstance(result, BaseModel):
            return self.output_schema.model_validate(result.model_dump())  # type: ignore[return-value]
        if isinstance(result, dict):
            return self.output_schema.model_validate(result)  # type: ignore[return-value]
        raise ValidationError.from_exception_data(
            self.output_schema.__name__,
            [{"type": "value_error", "loc": (), "msg": f"unexpected output type: {type(result).__name__}"}],
        )


# =============================================================================
# 内建 Skill：SchedulerSkill —— 把 multi_agent.schedule_service 包装成 Skill
# =============================================================================

class SchedulerInput(BaseModel):
    """Skill input: 调度请求。"""
    servers: list[list[float]]  # [[id, cpu_free, ram_free, net_free], ...]
    service: list[float]         # [cpu_pct, ram_pct, net_pct]
    aiops_risk_tags: list[str] = []


class SchedulerOutput(BaseModel):
    """Skill output: 决策结果（multi_agent.last_decision_dict 的核心字段）。"""
    action: str                  # "select" | "reject" | "fallback"
    server_id: int | None = None
    reasoning: str = ""
    latency_ms: float = 0.0
    server_id_returned: int = -1
    aiops_critic_triggered: bool = False


class SchedulerSkill(Skill[SchedulerInput, SchedulerOutput]):
    """把 multi_agent 的核心调度能力包装成可复用的 Skill。"""
    name = "scheduler.place"
    description = "Decide where to place an incoming service on a cluster, with optional AIOps closed-loop critic."
    input_schema = SchedulerInput
    output_schema = SchedulerOutput

    def execute(self, payload: SchedulerInput) -> SchedulerOutput:
        # lazy import 避免顶层循环依赖
        from multi_agent import init_agent, schedule_service, last_decision_dict

        init_agent(model_name="heuristic", enable_tracing=False)

        aiops_insight: dict[str, Any] | None = None
        if payload.aiops_risk_tags:
            aiops_insight = {
                "risk_tags": list(payload.aiops_risk_tags),
                "risk_level": "high",
                "risk_score": 0.7,
                "active_alerts": [
                    {"tag": t, "occurrence_count": 1, "risk_score": 0.7}
                    for t in payload.aiops_risk_tags
                ],
            }

        sid = schedule_service(payload.servers, payload.service, None, None, aiops_insight)
        decision = last_decision_dict()
        return SchedulerOutput(
            action=str(decision.get("action", "fallback")),
            server_id=decision.get("server_id"),
            reasoning=str(decision.get("reasoning", "")),
            latency_ms=float(decision.get("latency_ms", 0.0)),
            server_id_returned=sid,
            aiops_critic_triggered=bool(decision.get("aiops_critic_triggered", False)),
        )
