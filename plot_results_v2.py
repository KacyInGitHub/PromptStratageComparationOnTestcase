"""
plot_results_v2.py  —  基于真实统计数据生成论文图表

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
import numpy as np

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
OUT = "/Users/kacy/PycharmProjects/PromptStratageComparationOnTestcase/ResultsImages/"


def add_sig_bracket(ax, x1, x2, y, h, text, color="black"):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            lw=1.0, color=color)
    ax.text((x1+x2)/2, y+h*1.3, text,
            ha="center", va="bottom", fontsize=8, color=color)


# ═══════════════════════════════════════════════════════════════════
# 图1：RQ2 执行通过率
#   左：三次 trial 分组柱状图
#   右：折线趋势图
# ═══════════════════════════════════════════════════════════════════
def fig1_rq2():
    by_trial = {
        "CoT":        [65.6, 70.0, 68.9],
        "few_shot":   [67.8, 63.3, 65.6],
        "role_based": [56.7, 61.1, 60.0],
        "zero_shot":  [58.9, 60.0, 58.9],
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.38}
    )

    # ── 左：分组柱状图 ────────────────────────────
    x     = np.arange(4)
    bar_w = 0.22
    alphas = [0.55, 0.75, 1.0]

    for t in range(3):
        vals = [by_trial[k][t] for k in KEYS]
        bars = ax1.bar(
            x + (t-1)*bar_w, vals,
            width=bar_w,
            color=[(*matplotlib.colors.to_rgb(c), alphas[t])
                   for c in COLORS],
            edgecolor="white", linewidth=0.5,
            label=f"Trial {t+1}", zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.7,
                     f"{v:.1f}", ha="center", va="bottom",
                     fontsize=6.5, color="#444")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("Execution pass rate (%)")
    ax1.set_ylim(42, 82)
    ax1.set_title("(a) Pass rate by strategy and trial", pad=8)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax1.text(0.02, 0.97,
             "KW: H=23.21, p<0.001***",
             transform=ax1.transAxes, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec="gray", alpha=0.85))

    # 显著性括号（few_shot vs role_based, few_shot vs zero_shot）
    add_sig_bracket(ax1, 1-bar_w/2, 2+bar_w/2, 77, 0.8, "*", "#333")
    add_sig_bracket(ax1, 1-bar_w/2, 3+bar_w/2, 79.5, 0.8, "*", "#333")

    # ── 右：折线趋势 ──────────────────────────────
    trials = [1, 2, 3]
    for i, (key, label) in enumerate(zip(KEYS, STRATEGIES)):
        vals = by_trial[key]
        ax2.plot(trials, vals,
                 marker="o", markersize=6,
                 color=COLORS[i], linewidth=1.8,
                 label=label, zorder=3)
        ax2.fill_between(trials, vals, alpha=0.07, color=COLORS[i])
        ax2.annotate(f"{vals[-1]:.1f}%",
                     xy=(3, vals[-1]),
                     xytext=(3.1, vals[-1]),
                     fontsize=7.5, color=COLORS[i], va="center")

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Trial 1", "Trial 2", "Trial 3"])
    ax2.set_ylabel("Execution pass rate (%)")
    ax2.set_ylim(42, 82)
    ax2.set_xlim(0.7, 3.6)
    ax2.set_title("(b) Trial-to-trial stability", pad=8)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("RQ2: Test execution pass rate across strategies",
                 fontsize=12, y=1.02)
    plt.savefig(OUT+"fig1_rq2_execution.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig1] saved")


# ═══════════════════════════════════════════════════════════════════
# 图2：RQ1 覆盖率
#   左：行覆盖率 & 分支覆盖率并排柱状图（含 IQR 误差棒）
#   右：折线图显示行-分支差距
# ═══════════════════════════════════════════════════════════════════
def fig2_rq1():
    # 真实数据
    line_mean = [0.3637, 0.5372, 0.4190, 0.4764]
    line_iqr = [0.8069, 0.9091, 0.9000, 0.9091]
    branch_mean = [0.2692, 0.4718, 0.3217, 0.4013]
    branch_iqr = [0.8125, 1.0000, 0.8750, 1.0000]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.38}
    )

    x = np.arange(4)
    bar_w = 0.35

    # ── 左：并排柱状图 ────────────────────────────
    b1 = ax1.bar(x - bar_w / 2, line_mean,
                 width=bar_w, color=COLORS, alpha=0.88,
                 edgecolor="white", linewidth=0.5,
                 label="Line coverage", zorder=3)
    b2 = ax1.bar(x + bar_w / 2, branch_mean,
                 width=bar_w, color=COLORS_LIGHT, alpha=0.88,
                 edgecolor="white", linewidth=0.5,
                 label="Branch coverage", hatch="///", zorder=3)

    for bar, v in zip(b1, line_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#333")
    for bar, v in zip(b2, branch_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("Mean coverage (function-level)")
    ax1.set_ylim(0, 0.90)
    ax1.set_title("(a) Line and branch coverage by strategy", pad=8)
    ax1.legend(loc="upper left", framealpha=0.9)

    # ── 右：折线图（行 vs 分支差距）────────────────
    ax2.plot(STRATEGIES, line_mean,
             marker="o", markersize=7,
             color="#3266ad", linewidth=2.0,
             label="Line coverage", zorder=3)
    ax2.plot(STRATEGIES, branch_mean,
             marker="s", markersize=7,
             color="#d45f3c", linewidth=2.0,
             linestyle="--",
             label="Branch coverage", zorder=3)

    for i, (lv, bv) in enumerate(zip(line_mean, branch_mean)):
        ax2.text(i, lv + 0.026, f"{lv:.3f}",
                 ha="center", fontsize=7.5, color="#3266ad")
        ax2.text(i, bv - 0.048, f"{bv:.3f}",
                 ha="center", fontsize=7.5, color="#d45f3c")

    ax2.fill_between(range(4), line_mean, branch_mean,
                     alpha=0.10, color="gray",
                     label="Line–branch gap")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(STRATEGIES)
    ax2.set_ylabel("Mean coverage (function-level)")
    ax2.set_ylim(0, 0.72)
    ax2.set_title("(b) Coverage gap between line and branch", pad=8)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle(
        "RQ1: Function-level structural coverage by strategy",
        fontsize=12, y=1.02)
    plt.savefig(OUT + "fig2_rq1_coverage.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig2] saved")


# ═══════════════════════════════════════════════════════════════════
# 图3：RQ3 可读性指标
#   左：圈复杂度柱状图 + 误差棒
#   右：断言密度柱状图 + 误差棒
# ═══════════════════════════════════════════════════════════════════
def fig3_rq3():
    # 真实数据
    cc_mean = [3.3238, 2.1129, 3.3022, 3.1242]
    cc_iqr  = [2.5350, 0.3125, 1.6700, 1.8150]
    ad_mean = [2.3073, 1.1056, 2.2910, 2.1136]
    ad_iqr  = [2.5450, 0.3250, 1.6800, 1.8150]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.38}
    )

    x     = np.arange(4)
    bar_w = 0.52

    for ax, means, iqrs, ylabel, title, kw_text, note in [
        (ax1, cc_mean, cc_iqr,
         "Mean cyclomatic complexity",
         "(a) Cyclomatic complexity of test functions",
         "KW: H=15.58, p=0.0014**",
         "Lower = more readable"),
        (ax2, ad_mean, ad_iqr,
         "Mean assertions per test function",
         "(b) Assertion density per test function",
         "KW: H=15.10, p=0.0017**",
         "Lower = more focused"),
    ]:
        bars = ax.bar(x, means, width=bar_w,
                      color=COLORS, alpha=0.88,
                      edgecolor="white", linewidth=0.5,
                      zorder=3)
        ax.errorbar(x, means, yerr=[q/2 for q in iqrs],
                    fmt="none", color="black",
                    capsize=4, linewidth=1.1, zorder=4)

        for bar, v in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.08,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 6.2)
        ax.set_title(title, pad=8)
        ax.text(0.02, 0.97, f"{kw_text}\n{note}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.85))

        # 显著性括号
        # CoT vs few_shot (p_adj<0.001), few_shot vs role_based (p_adj=0.0005)
        # few_shot vs zero_shot (p_adj<0.001)
        add_sig_bracket(ax, 0, 1, 5.2, 0.18, "***", "#333")
        add_sig_bracket(ax, 1, 2, 5.7, 0.18, "***", "#333")
        add_sig_bracket(ax, 1, 3, 6.0, 0.18, "***", "#333")

    fig.suptitle(
        "RQ3: Readability and maintainability of generated test suites",
        fontsize=12, y=1.02)
    plt.savefig(OUT+"fig3_rq3_quality.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig3] saved")


# ═══════════════════════════════════════════════════════════════════
# 图4：Pass@1 vs Pass@3
# ═══════════════════════════════════════════════════════════════════
def fig4_passk():
    # 真实数据
    pass1 = [68.15, 65.56, 59.26, 59.26]
    pass3 = [96.85, 96.00, 93.35, 93.35]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x     = np.arange(4)
    bar_w = 0.35

    b1 = ax.bar(x-bar_w/2, pass1,
                width=bar_w, color=COLORS, alpha=0.88,
                edgecolor="white", linewidth=0.5,
                label="Pass@1", zorder=3)
    b2 = ax.bar(x+bar_w/2, pass3,
                width=bar_w, color=COLORS_LIGHT, alpha=0.88,
                edgecolor="white", linewidth=0.5,
                label="Pass@3", hatch="///", zorder=3)

    for bar, v in zip(b1, pass1):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.8,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    for bar, v in zip(b2, pass3):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.8,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # 连线显示 Pass@1 → Pass@3 的提升幅度
    for i in range(4):
        gain = pass3[i] - pass1[i]
        ax.annotate("",
                    xy=(x[i]+bar_w/2, pass3[i]-1),
                    xytext=(x[i]-bar_w/2, pass1[i]+1),
                    arrowprops=dict(arrowstyle="->",
                                   color=COLORS[i],
                                   lw=1.3, alpha=0.65))
        ax.text(x[i]+bar_w, (pass1[i]+pass3[i])/2,
                f"+{gain:.1f}pp",
                fontsize=7, color=COLORS[i], va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(40, 110)
    ax.set_title(
        "RQ2: Pass@1 and Pass@3 by prompt strategy",
        pad=8)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.text(0.02, 0.97,
            "Pass@1: mean single-trial pass rate\n"
            "Pass@3: success in at least 1 of 3 trials",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="gray", alpha=0.85))

    fig.tight_layout()
    plt.savefig(OUT+"fig4_passk.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig4] saved")


# ═══════════════════════════════════════════════════════════════════
# 图5：六指标汇总热力图
# ═══════════════════════════════════════════════════════════════════
def fig5_summary():
    # 真实均值数据
    raw = {
        "Compilation\nrate (↑)":      [0.9926, 1.0000, 0.9889, 0.9963],
        "Execution\npass rate (↑)":   [0.4474, 0.4969, 0.3711, 0.3659],
        "Line\ncoverage (↑)":         [0.3637, 0.5372, 0.4190, 0.4764],
        "Branch\ncoverage (↑)":       [0.2692, 0.4718, 0.3217, 0.4013],
        "Cyclomatic\ncomplexity (↓)": [3.3238, 2.1129, 3.3022, 3.1242],
        "Assertions\nper func (↓)":   [2.3073, 1.1056, 2.2910, 2.1136],
    }

    # 显示用的格式化字符串
    fmt = {
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

    metrics = list(raw.keys())
    matrix  = np.zeros((len(metrics), 4))

    for i, (k, vals) in enumerate(raw.items()):
        v = np.array(vals, dtype=float)
        if "↓" in k:
            v = -v
        vmin, vmax = v.min(), v.max()
        matrix[i] = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn",
                   vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(STRATEGIES, fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=9)

    for i, k in enumerate(metrics):
        for j in range(4):
            val = matrix[i, j]
            tc  = "white" if (val < 0.25 or val > 0.82) else "#222"
            ax.text(j, i, fmt[k][j],
                    ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised performance\n(green = better)",
                   fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Worst", "Mid", "Best"])

    # 标注每行最优值
    for i in range(len(metrics)):
        best_j = int(np.argmax(matrix[i]))
        rect = plt.Rectangle(
            (best_j-0.5, i-0.5), 1, 1,
            linewidth=2, edgecolor="black",
            facecolor="none", zorder=5)
        ax.add_patch(rect)

    ax.set_title(
        "Summary: normalised performance across all six metrics\n"
        "(arrows indicate direction of better performance;\n"
        "black border marks best strategy per metric)",
        pad=10, fontsize=10)

    fig.tight_layout()
    plt.savefig(OUT+"fig5_summary.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig5] saved")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    #fig1_rq2()
    fig2_rq1()
    #fig3_rq3()
    #fig4_passk()
    #fig5_summary()
    print("\n所有图表生成完毕。")
