"""
plot_results.py
===============
生成论文 Results 章节所需的所有图表。

输出：
  fig1_rq2_execution.pdf   RQ2 执行通过率（分组柱状图 + 折线趋势）
  fig2_rq1_coverage.pdf    RQ1 覆盖率（并排柱状图 + 误差棒）
  fig3_rq3_quality.pdf     RQ3 可读性指标（并排柱状图）
  fig4_passk.pdf           Pass@1 vs Pass@3（分组柱状图）
  fig5_summary.pdf         六指标汇总热力图
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 全局样式 ──────────────────────────────────────────────────────────
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

STRATEGIES   = ["CoT", "Few-shot", "Role-based", "Zero-shot"]
KEYS         = ["CoT", "few_shot", "role_based", "zero_shot"]
COLORS       = ["#534AB7", "#1D9E75", "#D85A30", "#888780"]
COLORS_LIGHT = ["#AFA9EC", "#9FE1CB", "#F0997B", "#B4B2A9"]

OUT = "/Users/kacy/PycharmProjects/PromptStratageComparationOnTestcase/ResultsImages"


def add_sig_bracket(ax, x1, x2, y, h, text, color="black"):
    """在两个柱子之间画显著性括号"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color=color)
    ax.text((x1+x2)/2, y+h*1.2, text, ha="center", va="bottom",
            fontsize=8, color=color)


# ═══════════════════════════════════════════════════════════════════
# 图1：RQ2 — 执行通过率
#   上方：按策略×trial 的分组柱状图
#   下方：三次 trial 折线趋势图
# ═══════════════════════════════════════════════════════════════════
def fig1_rq2():
    # 数据
    by_trial = {
        "CoT":        [65.6, 70.0, 68.9],
        "few_shot":   [67.8, 63.3, 65.6],
        "role_based": [56.7, 61.1, 60.0],
        "zero_shot":  [58.9, 60.0, 58.9],
    }
    overall = {
        "CoT": 68.1, "few_shot": 65.6,
        "role_based": 59.3, "zero_shot": 59.3,
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.35}
    )

    # ── 左图：分组柱状图（按 trial）──────────────
    n_strategies = 4
    n_trials     = 3
    bar_w        = 0.22
    x            = np.arange(n_strategies)
    trial_shades = [0.6, 0.8, 1.0]

    for t in range(n_trials):
        vals = [by_trial[k][t] for k in KEYS]
        bars = ax1.bar(
            x + (t - 1) * bar_w, vals,
            width=bar_w,
            color=[matplotlib.colors.to_rgba(c, trial_shades[t])
                   for c in COLORS],
            edgecolor="white", linewidth=0.5,
            label=f"Trial {t+1}",
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.8,
                     f"{v:.1f}", ha="center", va="bottom",
                     fontsize=7, color="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("Execution pass rate (%)")
    ax1.set_ylim(45, 80)
    ax1.set_title("(a) Execution pass rate by strategy and trial",
                  pad=8)
    ax1.legend(loc="upper right", framealpha=0.9)

    # 添加 Kruskal-Wallis 注释
    ax1.text(0.02, 0.96,
             "Kruskal–Wallis: H=23.21, p<0.001***",
             transform=ax1.transAxes, fontsize=7.5,
             va="top", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.8))

    # 添加显著性括号（few_shot vs role_based, few_shot vs zero_shot）
    ymax = 78
    add_sig_bracket(ax1, 1-bar_w/2, 2+bar_w/2, ymax,     1.0, "*", "#333")
    add_sig_bracket(ax1, 1-bar_w/2, 3+bar_w/2, ymax+3.5, 1.0, "*", "#333")

    # ── 右图：折线趋势图 ──────────────────────────
    trials = [1, 2, 3]
    for i, (key, label) in enumerate(zip(KEYS, STRATEGIES)):
        vals = by_trial[key]
        ax2.plot(trials, vals,
                 marker="o", markersize=6,
                 color=COLORS[i], linewidth=1.8,
                 label=label, zorder=3)
        ax2.fill_between(trials, vals,
                         alpha=0.07, color=COLORS[i])
        # 标注最终值
        ax2.annotate(
            f"{vals[-1]:.1f}%",
            xy=(3, vals[-1]),
            xytext=(3.08, vals[-1]),
            fontsize=7.5, color=COLORS[i],
            va="center",
        )

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Trial 1", "Trial 2", "Trial 3"])
    ax2.set_ylabel("Execution pass rate (%)")
    ax2.set_ylim(45, 80)
    ax2.set_xlim(0.7, 3.5)
    ax2.set_title("(b) Trial-to-trial stability", pad=8)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("RQ2: Test execution pass rate across strategies",
                 fontsize=12, y=1.01)
    plt.savefig(OUT + "fig1_rq2_execution.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig1] saved")


# ═══════════════════════════════════════════════════════════════════
# 图2：RQ1 — 行覆盖率 & 分支覆盖率
#   左：Mean ± IQR 误差棒柱状图
#   右：散点图（各函数覆盖率分布）
# ═══════════════════════════════════════════════════════════════════
def fig2_rq1():
    line_mean   = [0.3637, 0.5372, 0.4190, 0.4764]
    line_iqr    = [0.8069, 0.9091, 0.9000, 0.9091]
    branch_mean = [0.2692, 0.4718, 0.3217, 0.4013]
    branch_iqr  = [0.8125, 1.0000, 0.8750, 1.0000]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.35}
    )

    x     = np.arange(4)
    bar_w = 0.35

    # ── 左图：行覆盖率 vs 分支覆盖率 柱状图 ──────
    b1 = ax1.bar(x - bar_w/2, line_mean,
                 width=bar_w, color=COLORS, alpha=0.85,
                 edgecolor="white", linewidth=0.5,
                 label="Line coverage", zorder=3)
    b2 = ax1.bar(x + bar_w/2,
                 branch_mean, width=bar_w,
                 color=COLORS_LIGHT, alpha=0.85,
                 edgecolor="white", linewidth=0.5,
                 label="Branch coverage", zorder=3,
                 hatch="///")

    # 误差棒（IQR/2）
    for i in range(4):
        ax1.errorbar(x[i] - bar_w/2, line_mean[i],
                     yerr=line_iqr[i]/2,
                     fmt="none", color="black",
                     capsize=3, linewidth=1, zorder=4)
        ax1.errorbar(x[i] + bar_w/2, branch_mean[i],
                     yerr=branch_iqr[i]/2,
                     fmt="none", color="black",
                     capsize=3, linewidth=1, zorder=4)

    # 数值标注
    for bar, v in zip(b1, line_mean):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#333")
    for bar, v in zip(b2, branch_mean):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("Mean coverage (function-level)")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("(a) Line and branch coverage by strategy",
                  pad=8)
    ax1.legend(loc="upper left", framealpha=0.9)

    # KW 注释
    ax1.text(0.02, 0.97,
             "Line:   KW H=3.73,  p=0.292 ns\n"
             "Branch: KW H=7.36,  p=0.061 ns",
             transform=ax1.transAxes, fontsize=7,
             va="top", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.8))

    # 显著性括号（branch: CoT vs few_shot, p_adj=0.0366）
    add_sig_bracket(ax1, 0+bar_w/2, 1+bar_w/2, 0.72, 0.02, "*", "#333")

    # ── 右图：折线图 — 两项覆盖率趋势对比 ──────────
    ax2.plot(STRATEGIES, line_mean,
             marker="o", markersize=7,
             color="#3266ad", linewidth=2,
             label="Line coverage", zorder=3)
    ax2.plot(STRATEGIES, branch_mean,
             marker="s", markersize=7,
             color="#d45f3c", linewidth=2,
             linestyle="--",
             label="Branch coverage", zorder=3)

    for i, (lv, bv) in enumerate(zip(line_mean, branch_mean)):
        ax2.text(i, lv + 0.025, f"{lv:.3f}",
                 ha="center", fontsize=7.5, color="#3266ad")
        ax2.text(i, bv - 0.045, f"{bv:.3f}",
                 ha="center", fontsize=7.5, color="#d45f3c")

    # 填充两者之间的差距区域
    ax2.fill_between(range(4), line_mean, branch_mean,
                     alpha=0.08, color="gray",
                     label="Line–branch gap")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(STRATEGIES)
    ax2.set_ylabel("Mean coverage (function-level)")
    ax2.set_ylim(0, 0.75)
    ax2.set_title("(b) Coverage gap between line and branch",
                  pad=8)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle(
        "RQ1: Function-level structural coverage by strategy",
        fontsize=12, y=1.01)
    plt.savefig(OUT + "fig2_rq1_coverage.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig2] saved")


# ═══════════════════════════════════════════════════════════════════
# 图3：RQ3 — 圈复杂度 & 断言密度
# ═══════════════════════════════════════════════════════════════════
def fig3_rq3():
    cc_mean   = [3.3238, 2.1129, 3.3022, 3.1242]
    cc_iqr    = [2.5350, 0.3125, 1.6700, 1.8150]
    ad_mean   = [2.3073, 1.1056, 2.2910, 2.1136]
    ad_iqr    = [2.5450, 0.3250, 1.6800, 1.8150]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.35}
    )

    x     = np.arange(4)
    bar_w = 0.5

    # ── 左图：圈复杂度 ────────────────────────────
    bars = ax1.bar(x, cc_mean, width=bar_w,
                   color=COLORS, alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   zorder=3)
    ax1.errorbar(x, cc_mean, yerr=[q/2 for q in cc_iqr],
                 fmt="none", color="black",
                 capsize=4, linewidth=1, zorder=4)

    for bar, v in zip(bars, cc_mean):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.08,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("Mean cyclomatic complexity")
    ax1.set_ylim(0, 5.5)
    ax1.set_title("(a) Cyclomatic complexity of test functions",
                  pad=8)
    ax1.axhline(y=1, color="gray", linestyle=":", alpha=0.5,
                label="Minimum (no branching)")
    ax1.legend(loc="upper right", fontsize=8)

    ax1.text(0.02, 0.97,
             "Kruskal–Wallis: H=15.58, p=0.0014**",
             transform=ax1.transAxes, fontsize=7.5,
             va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.8))

    # 显著性括号
    add_sig_bracket(ax1, 0, 1, 4.6, 0.15, "*", "#333")   # CoT vs few_shot
    add_sig_bracket(ax1, 1, 2, 5.0, 0.15, "*", "#333")   # few_shot vs role_based
    add_sig_bracket(ax1, 1, 3, 5.3, 0.15, "*", "#333")   # few_shot vs zero_shot

    # ── 右图：断言密度 ────────────────────────────
    bars2 = ax2.bar(x, ad_mean, width=bar_w,
                    color=COLORS, alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    zorder=3)
    ax2.errorbar(x, ad_mean, yerr=[q/2 for q in ad_iqr],
                 fmt="none", color="black",
                 capsize=4, linewidth=1, zorder=4)

    for bar, v in zip(bars2, ad_mean):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.08,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(STRATEGIES)
    ax2.set_ylabel("Mean assertions per test function")
    ax2.set_ylim(0, 5.0)
    ax2.set_title("(b) Assertion density per test function",
                  pad=8)
    ax2.text(0.02, 0.97,
             "Kruskal–Wallis: H=15.10, p=0.0017**",
             transform=ax2.transAxes, fontsize=7.5,
             va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.8))

    add_sig_bracket(ax2, 0, 1, 4.1, 0.15, "*", "#333")
    add_sig_bracket(ax2, 1, 2, 4.5, 0.15, "*", "#333")
    add_sig_bracket(ax2, 1, 3, 4.8, 0.15, "*", "#333")

    fig.suptitle(
        "RQ3: Readability metrics of generated test suites",
        fontsize=12, y=1.01)
    plt.savefig(OUT + "fig3_rq3_quality.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig3] saved")


# ═══════════════════════════════════════════════════════════════════
# 图4：Pass@1 vs Pass@3
# ═══════════════════════════════════════════════════════════════════
def fig4_passk():
    pass1 = [0.6815, 0.6556, 0.5926, 0.5926]
    pass3 = [0.9685, 0.9600, 0.9335, 0.9335]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x     = np.arange(4)
    bar_w = 0.35

    b1 = ax.bar(x - bar_w/2, [v*100 for v in pass1],
                width=bar_w, color=COLORS, alpha=0.85,
                edgecolor="white", linewidth=0.5,
                label="Pass@1", zorder=3)
    b2 = ax.bar(x + bar_w/2, [v*100 for v in pass3],
                width=bar_w, color=COLORS_LIGHT, alpha=0.85,
                edgecolor="white", linewidth=0.5,
                label="Pass@3", hatch="///", zorder=3)

    for bar, v in zip(b1, pass1):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    for bar, v in zip(b2, pass3):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    # 连线显示 Pass@1 → Pass@3 的提升
    for i in range(4):
        ax.annotate(
            "",
            xy=(x[i]+bar_w/2, pass3[i]*100 - 1),
            xytext=(x[i]-bar_w/2, pass1[i]*100 + 1),
            arrowprops=dict(arrowstyle="->",
                            color=COLORS[i],
                            lw=1.2, alpha=0.6),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(40, 108)
    ax.set_title(
        "RQ2: Pass@1 and Pass@3 by prompt strategy\n"
        r"(Pass@$k$: probability of success in $k$ attempts)",
        pad=8)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    plt.savefig(OUT + "fig4_passk.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig4] saved")


# ═══════════════════════════════════════════════════════════════════
# 图5：六指标汇总热力图（归一化排名）
# ═══════════════════════════════════════════════════════════════════
def fig5_summary():
    # 数据矩阵：行=指标，列=策略
    # 对于"越低越好"的指标（复杂度、断言数），取反后归一化
    raw = {
        "Compilation\nrate (↑)":    [0.9926, 1.0000, 0.9889, 0.9963],
        "Execution\npass rate (↑)": [0.4474, 0.4969, 0.3711, 0.3659],
        "Line\ncoverage (↑)":       [0.3637, 0.5372, 0.4190, 0.4764],
        "Branch\ncoverage (↑)":     [0.2692, 0.4718, 0.3217, 0.4013],
        "Cyclomatic\ncomplexity (↓)":[3.3238, 2.1129, 3.3022, 3.1242],
        "Assertions\nper func (↓)": [2.3073, 1.1056, 2.2910, 2.1136],
    }

    metrics  = list(raw.keys())
    n_metrics = len(metrics)
    matrix   = np.zeros((n_metrics, 4))

    for i, (k, vals) in enumerate(raw.items()):
        v = np.array(vals, dtype=float)
        if "↓" in k:
            v = -v          # 反转：越小越好变成越大越好
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            matrix[i] = (v - vmin) / (vmax - vmin)
        else:
            matrix[i] = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn",
                   vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(STRATEGIES, fontsize=10)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metrics, fontsize=9)

    # 在每个格子里显示原始值
    raw_fmt = {
        "Compilation\nrate (↑)":
            [f"{v*100:.1f}%" for v in raw["Compilation\nrate (↑)"]],
        "Execution\npass rate (↑)":
            [f"{v*100:.1f}%" for v in raw["Execution\npass rate (↑)"]],
        "Line\ncoverage (↑)":
            [f"{v:.3f}" for v in raw["Line\ncoverage (↑)"]],
        "Branch\ncoverage (↑)":
            [f"{v:.3f}" for v in raw["Branch\ncoverage (↑)"]],
        "Cyclomatic\ncomplexity (↓)":
            [f"{v:.2f}" for v in raw["Cyclomatic\ncomplexity (↓)"]],
        "Assertions\nper func (↓)":
            [f"{v:.2f}" for v in raw["Assertions\nper func (↓)"]],
    }

    for i, k in enumerate(metrics):
        for j in range(4):
            val = matrix[i, j]
            txt_color = "white" if val < 0.25 or val > 0.85 else "black"
            ax.text(j, i, raw_fmt[k][j],
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color,
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised performance\n(green = better)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Worst", "Mid", "Best"])

    ax.set_title(
        "Summary: normalised performance across all six metrics\n"
        "(↑ = higher is better, ↓ = lower is better)",
        pad=10, fontsize=11)

    # 加边框突出最优列
    best_col = np.argmax(matrix.mean(axis=0))
    rect = plt.Rectangle(
        (best_col - 0.5, -0.5), 1, n_metrics,
        linewidth=2, edgecolor="#1D9E75",
        facecolor="none", zorder=5,
    )
    ax.add_patch(rect)
    ax.text(best_col, n_metrics - 0.1,
            "* best overall",
            ha="center", va="bottom",
            fontsize=8, color="#1D9E75")

    fig.tight_layout()
    plt.savefig(OUT + "fig5_summary.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig5] saved")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    fig1_rq2()
    fig2_rq1()
    fig3_rq3()
    fig4_passk()
    fig5_summary()
    print("\n所有图表已生成完毕。")
