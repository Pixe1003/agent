"""SLA-vs-Latency Pareto 散点图。

读取 benchmark/results/metrics.csv，按算法聚合 5-seed 均值，
为每个 distribution 出一张图，再合成一张总览。

跑法：
    .\.venv\Scripts\python.exe -m pip install matplotlib   # 首次安装
    .\.venv\Scripts\python.exe -m scripts.plot_pareto
或直接：
    .\.venv\Scripts\python.exe scripts\plot_pareto.py

输出：
    benchmark/results/pareto_<distribution>.png       (每个 distribution 一张)
    benchmark/results/pareto_overview.png             (汇总)
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib 未安装。先跑：")
    print("    .\\.venv\\Scripts\\python.exe -m pip install matplotlib")
    raise SystemExit(1)


CSV_PATH = _PROJECT_ROOT / "benchmark" / "results" / "metrics.csv"
OUTPUT_DIR = _PROJECT_ROOT / "benchmark" / "results"


# 每个算法的展示样式：颜色 + 形状
_STYLE = {
    "first-fit":          {"color": "#999999", "marker": "o", "label": "first-fit"},
    "balanced-fit":       {"color": "#1f77b4", "marker": "s", "label": "balanced-fit"},
    "AI-phase2":          {"color": "#2ca02c", "marker": "^", "label": "AI-phase2 (multi-agent)"},
    "AI-phase3":          {"color": "#9467bd", "marker": "D", "label": "AI-phase3 (+memory)"},
    "AI-phase2-aiops":    {"color": "#d62728", "marker": "*", "label": "AI-phase2 + AIOps"},
    "AI-phase3-aiops":    {"color": "#ff7f0e", "marker": "P", "label": "AI-phase3 + AIOps"},
}


def _load_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        print(f"ERROR: 找不到 {CSV_PATH}。先跑：")
        print("    .\\.venv\\Scripts\\python.exe -m benchmark.runner")
        raise SystemExit(1)
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    """按 (distribution, algorithm) 聚合 5 seed 均值。"""
    groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (row["distribution"], row["algorithm"])
        groups[key].append(
            {
                "sla_rate": float(row["sla_violation_rate"]),
                "rejection_rate": float(row["rejection_rate"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
                "p95_latency_ms": float(row["p95_latency_ms"]),
                "total_energy": float(row["total_energy"]),
                "avg_util_stddev": float(row["avg_util_stddev"]),
                "aiops_critic_trigger_rate": float(row.get("aiops_critic_trigger_rate") or 0.0),
            }
        )

    aggregated: dict[tuple[str, str], dict[str, float]] = {}
    for key, items in groups.items():
        n = len(items)
        means = {field: sum(item[field] for item in items) / n for field in items[0].keys()}
        aggregated[key] = means
    return aggregated


def _plot_one(
    title: str,
    points: dict[str, dict[str, float]],
    output_path: Path,
    *,
    x_field: str = "avg_latency_ms",
    y_field: str = "sla_rate",
    x_label: str = "Avg latency (ms, log scale)",
    y_label: str = "SLA violation rate",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for algo, metrics in points.items():
        style = _STYLE.get(algo, {"color": "black", "marker": "x", "label": algo})
        x = metrics[x_field]
        y = metrics[y_field]
        ax.scatter(
            [x],
            [y],
            color=style["color"],
            marker=style["marker"],
            s=180,
            edgecolors="black",
            linewidths=0.6,
            label=style["label"],
            zorder=3,
        )
        # 用细字标注算法
        ax.annotate(
            algo,
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#444444",
        )

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # 标注 Pareto 前沿（更低 SLA + 更低延迟更好；Pareto 最优在左下）
    ax.text(
        0.02, 0.02,
        "Pareto-better: ↙",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"),
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {output_path.relative_to(_PROJECT_ROOT)}")


def _plot_overview(aggregated: dict[tuple[str, str], dict[str, float]], output_path: Path) -> None:
    """所有 distribution 的算法均值聚成一张总览图。"""
    points: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (_, algo), metrics in aggregated.items():
        points[algo].append(metrics)
    overall: dict[str, dict[str, float]] = {}
    for algo, items in points.items():
        n = len(items)
        overall[algo] = {field: sum(item[field] for item in items) / n for field in items[0].keys()}
    _plot_one(
        title="SLA vs Latency — overall (mean over distributions × seeds)",
        points=overall,
        output_path=output_path,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    aggregated = _aggregate(rows)

    distributions = sorted({key[0] for key in aggregated})
    algorithms = sorted({key[1] for key in aggregated})
    print(f"loaded {len(rows)} rows: {len(distributions)} distributions × {len(algorithms)} algorithms")
    print()

    for dist in distributions:
        points = {algo: aggregated[(dist, algo)] for algo in algorithms if (dist, algo) in aggregated}
        _plot_one(
            title=f"SLA vs Latency — {dist} (mean over seeds)",
            points=points,
            output_path=OUTPUT_DIR / f"pareto_{dist}.png",
        )

    _plot_overview(aggregated, OUTPUT_DIR / "pareto_overview.png")
    print()
    print("Done. Open the PNG files in benchmark/results/ to see the Pareto frontier.")
    print("Pareto-best algorithms are toward the lower-left (less SLA, less latency).")


if __name__ == "__main__":
    main()
