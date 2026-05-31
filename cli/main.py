"""agent-cli — Agent Infra Toolkit 模块化命令行套件。

跑法：
    python -m cli.main --help
    python -m cli.main list-skills
    python -m cli.main run --skill scheduler.place --input data.json
    python -m cli.main benchmark --algos first-fit,balanced-fit,AI-phase2-aiops
    python -m cli.main plot
    python -m cli.main build-dataset --max-samples 12000
    python -m cli.main inference-smoke
    python -m cli.main mcp-server

设计原则（呼应蚂蚁 JD 的"CLI/SKILLS 模块化套件，降低 Agent 构建与调优门槛"）：
- 一个 entry point 覆盖 train / eval / inference / observe / serve 全流程
- 每个 subcommand 是独立模块的薄包装，没有重复逻辑
- 通过 Skill registry 自动 list / run，新增 Skill 无需改 CLI 代码
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 让 `python cli/main.py` 直接当脚本跑也能 import 项目里的包
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import click
except ImportError:
    raise SystemExit("click 未安装。先跑：pip install click")


# =============================================================================
# Click group
# =============================================================================

@click.group(help="Agent Infra Toolkit CLI — train / eval / inference / observe / serve.")
@click.version_option("0.1.0")
def cli() -> None:
    pass


# =============================================================================
# Skill 子命令
# =============================================================================

@cli.command("list-skills", help="List all registered Skills.")
@click.option("--detail/--no-detail", default=False, help="Show description + schemas.")
def list_skills(detail: bool) -> None:
    # 触发 skill 注册：导入 agent_common.skill 会自动注册内建 Skill
    from agent_common.skill import Skill  # noqa: F401
    _import_extra_skills()

    if detail:
        rows = Skill.describe_all()
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for name in Skill.list_skills():
            click.echo(name)


@cli.command("run", help="Run a registered Skill with JSON input.")
@click.option("--skill", "skill_name", required=True, help="Skill name (see list-skills)")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False),
              help="Path to JSON file containing input payload (UTF-8, BOM tolerant).")
@click.option("--inline", "inline_json", default=None,
              help="Inline JSON. WARN: PowerShell eats inner double-quotes — prefer --stdin or --input.")
@click.option("--stdin", "use_stdin", is_flag=True, default=False,
              help="Read JSON payload from stdin (PowerShell-friendly).")
@click.option("--inline-base64", "inline_b64", default=None,
              help="Inline JSON encoded as base64 (no shell quoting issues).")
def run_skill(
    skill_name: str,
    input_path: str | None,
    inline_json: str | None,
    use_stdin: bool,
    inline_b64: str | None,
) -> None:
    from agent_common.skill import Skill
    _import_extra_skills()

    if use_stdin:
        payload = json.loads(sys.stdin.read())
    elif inline_b64:
        import base64
        payload = json.loads(base64.b64decode(inline_b64).decode("utf-8"))
    elif inline_json:
        payload = json.loads(inline_json)
    elif input_path:
        # utf-8-sig 兼容 PowerShell `Out-File -Encoding utf8` 默认带的 BOM
        payload = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    else:
        raise click.UsageError(
            "provide one of: --stdin (recommended on PowerShell), "
            "--input <path>, --inline <json>, --inline-base64 <b64>"
        )

    skill_cls = Skill.get(skill_name)
    skill = skill_cls()  # type: ignore[abstract]
    result = skill.run(payload)
    click.echo(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))


# =============================================================================
# Benchmark 子命令
# =============================================================================

@cli.command("benchmark", help="Run the 5-seed × 4-dist × N-algo benchmark.")
@click.option("--seeds", default="1,2,3,4,5", help="Comma-separated seeds.")
@click.option("--distributions", default="mixed,cpu-heavy,memory-heavy,mixed-burst",
              help="Comma-separated distributions.")
@click.option("--algos", default="all", help="Comma-separated algorithms or 'all'.")
@click.option("--requests-per-scenario", default=100, type=int)
@click.option("--output", default="benchmark/results/metrics.csv", type=click.Path())
def benchmark(seeds: str, distributions: str, algos: str, requests_per_scenario: int, output: str) -> None:
    from benchmark.runner import run_benchmark

    all_algos = [
        "first-fit", "balanced-fit",
        "AI-phase2", "AI-phase3",
        "AI-phase2-aiops", "AI-phase3-aiops",
    ]
    chosen = all_algos if algos == "all" else [a.strip() for a in algos.split(",") if a.strip()]

    rows = run_benchmark(
        seeds=[int(s) for s in seeds.split(",")],
        distributions=[d.strip() for d in distributions.split(",") if d.strip()],
        algorithms=chosen,
        output_path=output,
        requests_per_scenario=requests_per_scenario,
    )
    click.echo(f"wrote {len(rows)} rows → {output}")


# =============================================================================
# Plot 子命令
# =============================================================================

@cli.command("plot", help="Generate Pareto charts from benchmark/results/metrics.csv.")
def plot() -> None:
    try:
        from scripts.plot_pareto import main as plot_main
    except ImportError:
        raise click.ClickException("matplotlib not installed. pip install matplotlib")
    plot_main()


# =============================================================================
# Build dataset 子命令
# =============================================================================

@cli.command("build-dataset", help="Convert trace JSONL → SFT dataset.")
@click.option("--version", type=click.Choice(["v1", "v2"]), default="v2")
@click.option("--max-samples", default=12000, type=int)
@click.option("--trace-dir", default="traces", type=click.Path())
@click.option("--output", default=None, type=click.Path())
def build_dataset(version: str, max_samples: int, trace_dir: str, output: str | None) -> None:
    if version == "v1":
        from dataset.build_sft_dataset import build_sft_dataset
        count = build_sft_dataset(
            trace_dir=trace_dir,
            output_path=output or "dataset/cloud-sched-sft-v1.jsonl",
            max_samples=max_samples,
        )
        click.echo(f"wrote {count} v1 SFT samples")
    else:
        from dataset.build_sft_dataset import build_sft_dataset_v2
        stats = build_sft_dataset_v2(
            trace_dir=trace_dir,
            output_path=output or "dataset/cloud-sched-sft-v2.jsonl",
            max_samples=max_samples,
        )
        click.echo(json.dumps(stats, indent=2, ensure_ascii=False))


# =============================================================================
# Inference smoke 子命令
# =============================================================================

@cli.command("inference-smoke", help="Single-call SFT inference smoke test.")
@click.option("--model-path", default="dataset/qwen25-1p5b-sched-merged-q4.gguf",
              type=click.Path())
@click.option("--n-gpu-layers", default=0, type=int)
def inference_smoke(model_path: str, n_gpu_layers: int) -> None:
    from agent_sft import init_agent, schedule_service, last_decision_dict, sft_stats_summary
    init_agent(model_path=model_path, n_gpu_layers=n_gpu_layers, enable_tracing=False)
    sid = schedule_service(
        [[0, 80.0, 80.0, 80.0], [1, 50.0, 60.0, 40.0], [2, 30.0, 70.0, 70.0]],
        [10.0, 15.0, 20.0],
    )
    click.echo(json.dumps({
        "sid_returned": sid,
        "decision": last_decision_dict(),
        "stats": sft_stats_summary(),
    }, indent=2, ensure_ascii=False, default=str))


# =============================================================================
# MCP server 子命令
# =============================================================================

@cli.command("mcp-server", help="Start the MCP server (stdio transport).")
def mcp_server() -> None:
    """启动 MCP server，供 Claude Desktop / Cursor / VSCode 接入。"""
    try:
        import mcp_server as _server  # 项目根目录的 mcp_server.py
    except ImportError as e:
        raise click.ClickException(
            f"Failed to import mcp_server: {e}. "
            f"Make sure 'pip install mcp' has been run."
        )
    _server.mcp.run()


# =============================================================================
# Hybrid stats 子命令（一行命令看 multi_agent / aiops 实时状态）
# =============================================================================

@cli.command("stats", help="Snapshot of multi_agent.hybrid_stats and agent_aiops.aiops_stats.")
def stats() -> None:
    from multi_agent import init_agent as init_ma, hybrid_stats
    from agent_aiops import init_agent as init_aiops, aiops_stats
    init_ma(model_name="heuristic", enable_tracing=False)
    init_aiops(model_name="heuristic", backend="rule", enable_tracing=False)
    click.echo(json.dumps({
        "multi_agent": hybrid_stats(),
        "agent_aiops": aiops_stats(),
    }, indent=2, ensure_ascii=False, default=str))


# =============================================================================
# Helper: 导入所有项目内已知的额外 Skill 实现
# =============================================================================

def _import_extra_skills() -> None:
    """触发 Skill 子类导入 → 自动注册到 Skill._registry。

    目前 SchedulerSkill 已经在 agent_common.skill 模块里，所以只需 import 它。
    将来新增 Skill (如 MemoryRetrieveSkill / AIOpsObserveSkill) 时，在此处补 import。
    """
    import agent_common.skill  # noqa: F401  triggers SchedulerSkill registration


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    cli()
