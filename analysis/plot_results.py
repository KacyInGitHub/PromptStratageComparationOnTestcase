"""
plot_results.py  —  generate thesis figures from real statistical data

Output:
  fig1_rq1_execution.pdf   RQ2 execution pass rate (grouped bar + trend line)
  fig2_rq2_coverage.pdf    RQ1 coverage (side-by-side bars + error bars)
  fig3_rq3_quality.pdf     RQ3 readability metrics (side-by-side bars)
  fig4_passk.pdf           Pass@1 vs Pass@3 (grouped bar)
  fig5_summary.pdf         Six-metric summary heatmap
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
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

STRATEGIES   = ["CoT", "Few-shot", "Role-based", "Zero-shot"]
KEYS         = ["CoT", "few_shot", "role_based", "zero_shot"]
LABELS      = ["CoT", "Few-shot", "Role-based", "Zero-shot"]
COLORS       = ["#534AB7", "#1D9E75", "#D85A30", "#888780"]
COLORS_LIGHT = ["#AFA9EC", "#9FE1CB", "#F0997B", "#B4B2A9"]
OUT = str(Path(__file__).parent.parent / "ResultsImages") + "/"

def add_sig_bracket(ax, x1, x2, y, h, text, color="black"):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            lw=1.0, color=color)
    ax.text((x1+x2)/2, y+h*1.3, text,
            ha="center", va="bottom", fontsize=8, color=color)


# ═══════════════════════════════════════════════════════════════════
# Fig 1: RQ2 execution pass rate
#   Left:  grouped bar chart for three trials
#   Right: trend line chart
# ═══════════════════════════════════════════════════════════════════
def fig1_rq1():
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

    # ── Left: grouped bar chart ───────────────────
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

    # Significance brackets (few_shot vs role_based, few_shot vs zero_shot)
    add_sig_bracket(ax1, 1-bar_w/2, 2+bar_w/2, 77, 0.8, "*", "#333")
    add_sig_bracket(ax1, 1-bar_w/2, 3+bar_w/2, 79.5, 0.8, "*", "#333")

    # ── Right: trend line ─────────────────────────
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

    fig.suptitle("RQ1: Test execution pass rate across strategies",
                 fontsize=12, y=1.02)
    plt.savefig(OUT+"fig1_rq1_execution.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig1] saved")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: RQ1 coverage
#   Left:  side-by-side bar chart for line & branch coverage (with IQR error bars)
#   Right: line chart showing line–branch gap
# ═══════════════════════════════════════════════════════════════════
def fig2_rq2():
    # Real data
    line_mean = [0.3222, 0.4682, 0.3840, 0.4242]
    line_iqr = [0.1091, 0.9091, 0.9000, 0.9091]
    branch_mean = [0.2256, 0.3853, 0.2699, 0.3350]
    branch_iqr = [0.0000, 1.0000, 0.7500, 1.0000]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"wspace": 0.38}
    )

    x = np.arange(4)
    bar_w = 0.35

    # ── Left: side-by-side bar chart ─────────────
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

    # ── Right: line chart (line vs branch gap) ────
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
        "RQ2: Function-level structural coverage by strategy",
        fontsize=12, y=1.02)
    plt.savefig(OUT + "fig2_rq2_coverage.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig2] saved")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: RQ3 readability metrics
#   Left:  cyclomatic complexity bar chart + error bars
#   Right: assertion density bar chart + error bars
# ═══════════════════════════════════════════════════════════════════
def fig3_rq3():
    # Real data
    cc_mean = [3.1953, 2.0931, 3.1544, 3.0060]
    cc_iqr  = [2.2200, 0.2175, 1.4300, 1.6800]
    ad_mean = [2.1790, 1.0867, 2.1445, 1.9966]
    ad_iqr  = [2.2300, 0.2150, 1.4300, 1.6550]

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
        ax.set_ylim(0, 5.3)
        ax.set_title(title, pad=8)
        ax.text(0.02, 0.97, f"{kw_text}\n{note}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.85))

        # Significance brackets
        # CoT vs few_shot (p_adj<0.001), few_shot vs role_based (p_adj=0.0005)
        # few_shot vs zero_shot (p_adj<0.001)
        # add_sig_bracket(ax, 0, 1, 5.2, 0.18, "***", "#333")
        # add_sig_bracket(ax, 1, 2, 5.7, 0.18, "***", "#333")
        # add_sig_bracket(ax, 1, 3, 6.0, 0.18, "***", "#333")

    fig.suptitle(
        "RQ3: Readability and maintainability of generated test suites",
        fontsize=12, y=1.02)
    plt.savefig(OUT+"fig3_rq3_quality.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig3] saved")


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Pass@1 vs Pass@3
# ═══════════════════════════════════════════════════════════════════
def fig4_passk():
    # Real data
    pass1 = [68.15, 65.56, 59.26, 59.26]
    pass3 = [84.44, 73.33, 68.89, 68.89]

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

    # Arrows showing gain from Pass@1 to Pass@3
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
        "RQ1: Pass@1 and Pass@3 by prompt strategy",
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
# Fig 5: Six-metric summary heatmap
# ═══════════════════════════════════════════════════════════════════
def fig5_summary():
    # Real mean values
    raw = {
        "Compilation\nrate (↑)":      [0.993, 1.0000, 0.9889, 0.9963],
        "Execution\npass rate (↑)":   [0.682, 0.656, 0.593, 0.593],
        "Line\ncoverage (↑)":         [0.322, 0.468, 0.384, 0.424],
        "Branch\ncoverage (↑)":       [0.226, 0.385, 0.270, 0.335],
        "Cyclomatic\ncomplexity (↓)": [3.20, 2.09, 3.15, 3.01],
        "Assertions\nper func (↓)":   [2.18, 1.09, 2.14, 2.00],
    }

    # Display format strings
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

    # Highlight best value in each row
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


# ── Load data ─────────────────────────────────────────────────────
with open("../results/metrics_results.json") as f:
    metrics = json.load(f)
with open("../results/pipeline_results.json") as f:
    pipeline = json.load(f)

# Collect raw values per strategy
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

# Execution pass rate (best trial per function)
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
# Fig 6: RQ1 coverage box plots
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

        # Overlay mean diamonds
        means = [clean(raw[s][key]).mean() for s in STRATEGIES]
        ax.scatter(range(1, 5), means,
                   marker="D", s=40, zorder=5,
                   color="white", edgecolor="black",
                   linewidth=1.0, label="Mean")

        # Annotate median and mean
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

        # Note about branch-less functions
        if key == "branch_rate":
            ax.text(0.02, 0.97,
                    "Note: 30 functions with no branches excluded\n"
                    "Median = 0.00 for all strategies",
                    transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.85))

    fig.suptitle(
        "RQ2: Distribution of function-level coverage across strategies\n"
        "(boxes = IQR, whiskers = 1.5×IQR, diamonds = mean)",
        fontsize=11, y=1.03)
    plt.savefig(OUT+"fig6_coverage_box.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("[fig6] saved")


# ═══════════════════════════════════════════════════════════════════
# Fig 7: RQ3 readability metrics box plots
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
# Fig 8: RQ2 execution pass rate box plot (best trial)
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
        "RQ1: Distribution of execution pass rate by strategy\n"
        "(best trial selected per function)",
        pad=8)
    # Legend in upper right; stats annotation in upper left
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
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
# Fig 11: Cyclomatic complexity combined (bar + box)
# ═══════════════════════════════════════════════════════════════════
def fig11_complexity_combined():
    cc_mean = [3.1953, 2.0931, 3.1544, 3.0060]
    data_list = [clean(raw[s]["avg_complexity"]) for s in STRATEGIES]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.38}
    )

    # ── Left: bar chart ───────────────────────────
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

    # ── Right: box plot ───────────────────────────
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
# Fig 12: Assertion density combined (bar + box)
# ═══════════════════════════════════════════════════════════════════
def fig12_assertion_combined():
    ad_mean = [2.1790, 1.0867, 2.1445, 1.9966]
    data_list = [clean(raw[s]["avg_assertions"]) for s in STRATEGIES]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.38}
    )

    # ── Left: bar chart ───────────────────────────
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

    # ── Right: box plot ───────────────────────────
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
    fig1_rq1()
    fig2_rq2()
    fig3_rq3()
    fig4_passk()
    fig5_summary()
    fig6_coverage_box()
    fig7_quality_box()
    fig8_pass_box()
    fig11_complexity_combined()
    fig12_assertion_combined()
    print("\nAll figures generated.")
