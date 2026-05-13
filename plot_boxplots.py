"""
plot_boxplots.py  —  生成箱线图展示原始数据分布
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

STRATEGIES  = ["CoT", "few_shot", "role_based", "zero_shot"]
LABELS      = ["CoT", "Few-shot", "Role-based", "Zero-shot"]
COLORS      = ["#534AB7", "#1D9E75", "#D85A30", "#888780"]
OUT         = "/Users/kacy/PycharmProjects/PromptStratageComparationOnTestcase/ResultsImages/"

# ── 加载数据 ──────────────────────────────────────────────────────
with open("/Users/kacy/PycharmProjects/PromptStratageComparationOnTestcase/metrics_results_new.json") as f:
    metrics = json.load(f)
with open("pipeline_results.json") as f:
    pipeline = json.load(f)

# 按策略收集原始值
raw = {s: {
    "line_rate": [], "branch_rate": [],
    "avg_complexity": [], "avg_assertions": []
} for s in STRATEGIES}

for r in metrics:
    s = r["strategy"]
    if s not in raw:
        continue
    cov  = r.get("coverage",   {})
    comp = r.get("complexity", {})
    assr = r.get("assertions", {})
    raw[s]["line_rate"].append(cov.get("line_rate"))
    raw[s]["branch_rate"].append(cov.get("branch_rate"))
    raw[s]["avg_complexity"].append(comp.get("avg_complexity"))
    raw[s]["avg_assertions"].append(assr.get("avg_assertions"))

# 执行通过率（最优 trial）
pass_tmp = defaultdict(lambda: defaultdict(list))
for r in pipeline:
    s   = r["strategy"]
    fid = r["function_id"]
    exe = r.get("execution", {})
    tot = exe.get("total", 0)
    pas = exe.get("passed", 0)
    pass_tmp[s][fid].append(pas / tot if tot > 0 else 0.0)

best_pass = {s: [max(v) for v in pass_tmp[s].values()]
             for s in STRATEGIES}

def clean(lst):
    return np.array([v for v in lst if v is not None], dtype=float)

# ═══════════════════════════════════════════════════════════════════
# 图6：RQ1 覆盖率箱线图
# ═══════════════════════════════════════════════════════════════════
def fig6_coverage_box():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.35}
    )

    for ax, key, title, ylabel in [
        (ax1, "line_rate",
         "(a) Line coverage distribution",
         "Line coverage (function-level)"),
        (ax2, "branch_rate",
         "(b) Branch coverage distribution",
         "Branch coverage (function-level)"),
    ]:
        data_list = [clean(raw[s][key]) for s in STRATEGIES]

        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(
                marker="o", markersize=3,
                alpha=0.4, linestyle="none"
            ),
            widths=0.5,
        )

        for patch, color, flier in zip(
            bp["boxes"], COLORS, bp["fliers"]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        # 叠加均值点
        means = [clean(raw[s][key]).mean() for s in STRATEGIES]
        ax.scatter(range(1, 5), means,
                   marker="D", s=40, zorder=5,
                   color="white", edgecolor="black",
                   linewidth=1.0, label="Mean")

        # 标注中位数和均值
        for i, (s, m) in enumerate(zip(STRATEGIES, means)):
            arr    = clean(raw[s][key])
            median = float(np.median(arr))
            ax.text(i+1, median + 0.03,
                    f"Med={median:.2f}",
                    ha="center", fontsize=7, color="white",
                    fontweight="bold")
            ax.text(i+1, -0.10,
                    f"Mean={m:.3f}",
                    ha="center", fontsize=7.5, color=COLORS[i])

        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(LABELS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.15, 1.15)
        ax.set_title(title, pad=8)
        ax.legend(loc="upper right", fontsize=8)

        # 无分支函数的说明
        if key == "branch_rate":
            ax.text(0.02, 0.97,
                    "Note: 30 functions with no branches excluded\n"
                    "Median = 0.00 for all strategies",
                    transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.85))

    fig.suptitle(
        "RQ1: Distribution of function-level coverage across strategies\n"
        "(boxes = IQR, whiskers = 1.5×IQR, diamonds = mean)",
        fontsize=11, y=1.03)
    plt.savefig(OUT+"fig6_coverage_box.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig6] saved")


# ═══════════════════════════════════════════════════════════════════
# 图7：RQ3 可读性指标箱线图
# ═══════════════════════════════════════════════════════════════════
def fig7_quality_box():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.35}
    )

    for ax, key, title, ylabel, note in [
        (ax1, "avg_complexity",
         "(a) Cyclomatic complexity distribution",
         "Mean cyclomatic complexity per test file",
         "KW: H=15.58, p=0.0014**\nFew-shot significantly lower (p<0.001)"),
        (ax2, "avg_assertions",
         "(b) Assertion density distribution",
         "Mean assertions per test function",
         "KW: H=15.10, p=0.0017**\nFew-shot significantly lower (p<0.001)"),
    ]:
        data_list = [clean(raw[s][key]) for s in STRATEGIES]

        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(
                marker="o", markersize=3,
                alpha=0.4, linestyle="none"
            ),
            widths=0.5,
        )

        for patch, color, flier in zip(
            bp["boxes"], COLORS, bp["fliers"]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        means = [clean(raw[s][key]).mean() for s in STRATEGIES]
        ax.scatter(range(1, 5), means,
                   marker="D", s=40, zorder=5,
                   color="white", edgecolor="black",
                   linewidth=1.0, label="Mean")

        for i, (s, m) in enumerate(zip(STRATEGIES, means)):
            arr    = clean(raw[s][key])
            median = float(np.median(arr))
            ax.text(i+1, median + 0.08,
                    f"Med={median:.2f}",
                    ha="center", fontsize=7, color="white",
                    fontweight="bold")
            ax.text(i+1, -0.8,
                    f"Mean={m:.2f}",
                    ha="center", fontsize=7.5, color=COLORS[i])

        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(LABELS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.2, ax.get_ylim()[1] * 1.05)
        ax.set_title(title, pad=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(0.02, 0.97, note,
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.85))

    fig.suptitle(
        "RQ3: Distribution of readability metrics across strategies\n"
        "(boxes = IQR, whiskers = 1.5×IQR, diamonds = mean)",
        fontsize=11, y=1.03)
    plt.savefig(OUT+"fig7_quality_box.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig7] saved")


# ═══════════════════════════════════════════════════════════════════
# 图8：RQ2 执行通过率箱线图（最优 trial）
# ═══════════════════════════════════════════════════════════════════
def fig8_pass_box():
    fig, ax = plt.subplots(figsize=(7, 5))

    data_list = [np.array(best_pass[s]) for s in STRATEGIES]

    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(
            marker="o", markersize=3,
            alpha=0.4, linestyle="none"
        ),
        widths=0.5,
    )

    for patch, color, flier in zip(
        bp["boxes"], COLORS, bp["fliers"]
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    means = [np.array(best_pass[s]).mean() for s in STRATEGIES]
    ax.scatter(range(1, 5), means,
               marker="D", s=50, zorder=5,
               color="white", edgecolor="black",
               linewidth=1.0, label="Mean")

    for i, (s, m) in enumerate(zip(STRATEGIES, means)):
        arr    = np.array(best_pass[s])
        median = float(np.median(arr))
        ax.text(i+1, median + 0.02,
                f"Med={median:.2f}",
                ha="center", fontsize=7.5, color="white",
                fontweight="bold")
        ax.text(i+1, -0.08,
                f"Mean={m:.3f}",
                ha="center", fontsize=8, color=COLORS[i])

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Execution pass rate (best trial per function)")
    ax.set_ylim(-0.12, 1.18)
    ax.set_title(
        "RQ2: Distribution of execution pass rate by strategy\n"
        "(best trial selected per function)",
        pad=8)
    # 图例移到右上角
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    # 统计注释放左上角，独占空间
    ax.text(0.02, 0.99,
            "KW: H=23.21, p<0.001***\n"
            "Few-shot vs Role-based: p_adj<0.001***\n"
            "Few-shot vs Zero-shot:  p_adj<0.001***",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="gray", alpha=0.92))

    fig.tight_layout()
    plt.savefig(OUT+"fig8_pass_box.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig8] saved")


# ═══════════════════════════════════════════════════════════════════
# 图11：圈复杂度复合图（柱状图 + 箱线图）
# ═══════════════════════════════════════════════════════════════════
def fig11_complexity_combined():
    cc_mean = [3.1953, 2.0931, 3.1544, 3.0060]
    data_list = [clean(raw[s]["avg_complexity"]) for s in STRATEGIES]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.38}
    )

    # ── 左：柱状图 ────────────────────────────────
    x = np.arange(4)
    bar_w = 0.52
    bars = ax1.bar(x, cc_mean, width=bar_w,
                   color=COLORS, alpha=0.88,
                   edgecolor="white", linewidth=0.5,
                   zorder=3)

    for bar, v in zip(bars, cc_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.06,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Mean cyclomatic complexity")
    ax1.set_ylim(0, 4.5)
    ax1.set_title("(a) Mean cyclomatic complexity by strategy", pad=8)
    ax1.text(0.02, 0.97,
             "KW: H=12.48, p=0.0059**\nLower = more readable",
             transform=ax1.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.85))

    # ── 右：箱线图 ────────────────────────────────
    bp = ax2.boxplot(
        data_list,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3,
                        alpha=0.4, linestyle="none"),
        widths=0.5,
    )

    for patch, color, flier in zip(bp["boxes"], COLORS, bp["fliers"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    means = [clean(raw[s]["avg_complexity"]).mean() for s in STRATEGIES]
    ax2.scatter(range(1, 5), means,
                marker="D", s=40, zorder=5,
                color="white", edgecolor="black",
                linewidth=1.0, label="Mean")

    for i, (s, m) in enumerate(zip(STRATEGIES, means)):
        arr = clean(raw[s]["avg_complexity"])
        median = float(np.median(arr))
        ax2.text(i + 1, median + 0.12,
                 f"Med={median:.2f}",
                 ha="center", fontsize=7, color="white",
                 fontweight="bold")
        ax2.text(i + 1, 0.2,
                 f"Mean={m:.2f}",
                 ha="center", fontsize=7.5, color=COLORS[i])

    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Cyclomatic complexity")
    ax2.set_title("(b) Distribution of cyclomatic complexity", pad=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Cyclomatic complexity of generated test functions\n"
        "(boxes = IQR, whiskers = 1.5×IQR, diamonds = mean)",
        fontsize=11, y=1.03)
    plt.savefig(OUT + "fig11_complexity.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig11] saved")


# ═══════════════════════════════════════════════════════════════════
# 图12：断言密度复合图（柱状图 + 箱线图）
# ═══════════════════════════════════════════════════════════════════
def fig12_assertion_combined():
    ad_mean = [2.1790, 1.0867, 2.1445, 1.9966]
    data_list = [clean(raw[s]["avg_assertions"]) for s in STRATEGIES]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.38}
    )

    # ── 左：柱状图 ────────────────────────────────
    x = np.arange(4)
    bar_w = 0.52
    bars = ax1.bar(x, ad_mean, width=bar_w,
                   color=COLORS, alpha=0.88,
                   edgecolor="white", linewidth=0.5,
                   zorder=3)

    for bar, v in zip(bars, ad_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("Mean assertions per test function")
    ax1.set_ylim(0, 3.5)
    ax1.set_title("(a) Mean assertion density by strategy", pad=8)
    ax1.text(0.02, 0.97,
             "KW: H=12.00, p=0.0074**\nLower = more focused",
             transform=ax1.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.85))

    # ── 右：箱线图 ────────────────────────────────
    bp = ax2.boxplot(
        data_list,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3,
                        alpha=0.4, linestyle="none"),
        widths=0.5,
    )

    for patch, color, flier in zip(bp["boxes"], COLORS, bp["fliers"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    means = [clean(raw[s]["avg_assertions"]).mean() for s in STRATEGIES]
    ax2.scatter(range(1, 5), means,
                marker="D", s=40, zorder=5,
                color="white", edgecolor="black",
                linewidth=1.0, label="Mean")

    for i, (s, m) in enumerate(zip(STRATEGIES, means)):
        arr = clean(raw[s]["avg_assertions"])
        median = float(np.median(arr))
        ax2.text(i + 1, median + 0.08,
                 f"Med={median:.2f}",
                 ha="center", fontsize=7, color="white",
                 fontweight="bold")
        ax2.text(i + 1, -0.6,
                 f"Mean={m:.2f}",
                 ha="center", fontsize=7.5, color=COLORS[i])

    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Assertions per test function")
    ax2.set_title("(b) Distribution of assertion density", pad=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Assertion density of generated test functions\n"
        "(boxes = IQR, whiskers = 1.5×IQR, diamonds = mean)",
        fontsize=11, y=1.03)
    plt.savefig(OUT + "fig12_assertion.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig12] saved")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # fig6_coverage_box()
    # fig7_quality_box()
    # fig8_pass_box()
    # fig11_complexity_combined()
    fig12_assertion_combined()
    print("\n所有箱线图生成完毕。")