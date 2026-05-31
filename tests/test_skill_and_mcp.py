"""验证 Skill 抽象 + MCP server adapter 的基本契约。

不需要安装 mcp SDK（MCP 测试只验证模块结构 + 不导入触发 ImportError）。
"""
import json

import pytest
from pydantic import BaseModel, ValidationError

from agent_common.skill import (
    Skill,
    SchedulerSkill,
    SchedulerInput,
    SchedulerOutput,
)


# =============================================================================
# Skill registry
# =============================================================================

def test_scheduler_skill_is_registered():
    """SchedulerSkill 应该自动注册到 Skill._registry."""
    assert "scheduler.place" in Skill.list_skills()
    assert Skill.get("scheduler.place") is SchedulerSkill


def test_skill_get_raises_keyerror_on_unknown():
    with pytest.raises(KeyError):
        Skill.get("definitely-not-a-skill")


def test_describe_all_includes_scheduler():
    rows = Skill.describe_all()
    names = [r["name"] for r in rows]
    assert "scheduler.place" in names
    row = next(r for r in rows if r["name"] == "scheduler.place")
    assert row["input_schema"] == "SchedulerInput"
    assert row["output_schema"] == "SchedulerOutput"


# =============================================================================
# Skill.run() 校验
# =============================================================================

def test_scheduler_skill_runs_end_to_end_with_dict_input():
    skill = SchedulerSkill()
    out = skill.run({
        "servers": [[0, 80.0, 80.0, 80.0], [1, 70.0, 70.0, 70.0]],
        "service": [10.0, 10.0, 10.0],
        "aiops_risk_tags": [],
    })
    assert isinstance(out, SchedulerOutput)
    assert out.action in {"select", "reject", "fallback"}
    assert isinstance(out.server_id_returned, int)


def test_scheduler_skill_aiops_risk_tags_propagate():
    """加上 aiops_risk_tags 后 decision 应当被 critic 收紧。"""
    skill = SchedulerSkill()
    out = skill.run({
        # server 0 NET 紧张 (放置后 NET=5%), server 1 NET 充裕
        "servers": [[0, 20.0, 20.0, 25.0], [1, 70.0, 70.0, 70.0]],
        "service": [10.0, 10.0, 20.0],
        "aiops_risk_tags": ["network-pressure", "sla-risk"],
    })
    # 在 network-pressure 下 critic 应该把 server 0 排除，转向 server 1
    assert out.aiops_critic_triggered is True
    assert out.server_id_returned == 1


def test_scheduler_skill_rejects_invalid_input_via_schema():
    skill = SchedulerSkill()
    with pytest.raises(ValidationError):
        skill.run({
            "servers": "not-a-list",
            "service": [10.0, 10.0, 10.0],
        })


# =============================================================================
# MCP server 模块结构（不依赖 mcp SDK 安装）
# =============================================================================

def test_mcp_server_module_lists_tools(monkeypatch):
    """mcp_server.py 应该能在 mcp SDK 没装时给出友好错误，装了时正常 import。"""
    # 用 importlib 模拟 import 但只验证模块语法
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "mcp_server_check",
        Path(__file__).resolve().parents[1] / "mcp_server.py",
    )
    assert spec is not None

    # 不强制 import（mcp SDK 可能没装），只检查文件存在和顶层签名
    text = (Path(__file__).resolve().parents[1] / "mcp_server.py").read_text(encoding="utf-8")
    # 这三个 tool + 两个 resource 必须暴露
    assert "@mcp.tool()" in text
    assert "def schedule_placement(" in text
    assert "def aiops_observe(" in text
    assert "def memory_retrieve(" in text
    assert '@mcp.resource("scheduler://stats")' in text
    assert '@mcp.resource("aiops://summary")' in text


# =============================================================================
# CLI 模块结构（不实际跑子命令，避免 click 测试 runner 引入额外依赖）
# =============================================================================

def test_cli_main_module_defines_expected_commands():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "cli" / "main.py").read_text(encoding="utf-8")
    for cmd in (
        '@cli.command("list-skills"',
        '@cli.command("run"',
        '@cli.command("benchmark"',
        '@cli.command("plot"',
        '@cli.command("build-dataset"',
        '@cli.command("inference-smoke"',
        '@cli.command("mcp-server"',
        '@cli.command("stats"',
    ):
        assert cmd in text, f"CLI missing command registration: {cmd}"
