"""
statistical_analysis.py
=======================
Statistical analysis of six evaluation metrics for thesis reporting.

Input:
    - pipeline_results.json    (RQ2: compilation rate, execution pass rate, three trials)
    - metrics_results.json     (RQ1+RQ3: coverage, complexity, assertion count, best trial)

Output:
    - statistical_results.json  Full statistical results
    - statistical_report.txt    Human-readable report for direct citation in the thesis

Usage:
    python statistical_analysis.py \
        --pipeline ../results/pipeline_results.json \
        --metrics  ../results/metrics_results.json \
        --output   ../results/statistical_results.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import shapiro, kruskal, wilcoxon

# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
STRATEGIES = ["CoT", "few_shot", "role_based", "zero_shot"]
STRATEGY_PAIRS = [
    ("CoT",        "few_shot"),
    ("CoT",        "role_based"),
    ("CoT",        "zero_shot"),
    ("few_shot",   "role_based"),
    ("few_shot",   "zero_shot"),
    ("role_based", "zero_shot"),
]
BONFERRONI_N = 6          # 6 pairwise comparisons
ALPHA        = 0.05
ALPHA_ADJ    = ALPHA / BONFERRONI_N   # 0.0083


def rank_biserial(x, y):
    """Compute rank-biserial correlation coefficient r for two groups."""
    nx, ny = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * u_stat) / (nx * ny)
    return round(r, 4)


def effect_size_label(r):
    r = abs(r)
    if r >= 0.5:
        return "large"
    elif r >= 0.3:
        return "medium"
    elif r >= 0.1:
        return "small"
    return "negligible"


def descriptive(values):
    """Compute descriptive statistics."""
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n":      int(len(arr)),
        "mean":   round(float(np.mean(arr)),   4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(np.std(arr)),    4),
        "q1":     round(float(q1),             4),
        "q3":     round(float(q3),             4),
        "iqr":    round(float(q3 - q1),        4),
        "min":    round(float(np.min(arr)),     4),
        "max":    round(float(np.max(arr)),     4),
    }


def normality_test(values):
    """Shapiro-Wilk normality test."""
    arr = np.array([v for v in values if v is not None])
    if len(arr) < 3:
        return {"stat": None, "p": None, "normal": False}
    stat, p = shapiro(arr)
    return {
        "stat":   round(float(stat), 4),
        "p":      round(float(p),    4),
        "normal": bool(p >= ALPHA),
    }


def kruskal_wallis_test(groups: dict):
    """Kruskal-Wallis test. groups = {strategy: [values]}"""
    arrays = [
        np.array([v for v in groups[s] if v is not None])
        for s in STRATEGIES
        if s in groups
    ]
    arrays = [a for a in arrays if len(a) > 0]
    if len(arrays) < 2:
        return {"H": None, "p": None, "significant": False}
    H, p = kruskal(*arrays)
    return {
        "H":           round(float(H), 4),
        "p":           round(float(p), 4),
        "significant": bool(p < ALPHA),
    }


def posthoc_tests(groups: dict):
    """Pairwise Wilcoxon post-hoc tests with Bonferroni correction."""
    results = []
    for s1, s2 in STRATEGY_PAIRS:
        if s1 not in groups or s2 not in groups:
            continue
        x = np.array([v for v in groups[s1] if v is not None])
        y = np.array([v for v in groups[s2] if v is not None])

        # Align sample sizes (use smaller length)
        min_n = min(len(x), len(y))
        if min_n < 5:
            continue
        x, y = x[:min_n], y[:min_n]

        try:
            stat, p_raw = wilcoxon(x, y, alternative="two-sided")
        except ValueError:
            continue

        p_adj = min(p_raw * BONFERRONI_N, 1.0)
        r     = rank_biserial(x, y)

        results.append({
            "pair":        f"{s1} vs {s2}",
            "s1":          s1,
            "s2":          s2,
            "stat":        round(float(stat), 4),
            "p_raw":       round(float(p_raw), 4),
            "p_adjusted":  round(float(p_adj), 4),
            "significant": bool(p_adj < ALPHA),
            "r":           r,
            "effect_size": effect_size_label(r),
        })
    return results


def pass_at_k(n: int, c: int, k: int) -> float:
    """Pass@k estimator (Chen et al., 2021)."""
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
# def load_pipeline(path: str) -> dict:
#     """Load RQ2 data from pipeline_results.json.
#     Each record contains strategy, trial, static, compile, execution fields.
#     """
#     with open(path, encoding="utf-8") as f:
#         records = json.load(f)
#
#     # Group by strategy, keep all trials
#     data = defaultdict(list)
#     for r in records:
#         s = r.get("strategy", "unknown")
#         compile_ok = int(r.get("compile", {}).get("compile_ok", False))
#         passed     = r.get("execution", {}).get("passed", 0)
#         total      = r.get("execution", {}).get("total",  0)
#         pass_rate  = passed / total if total > 0 else 0.0
#         data[s].append({
#             "trial":       r.get("trial"),
#             "compile_ok":  compile_ok,
#             "pass_rate":   pass_rate,
#             "passed":      passed,
#             "total":       total,
#         })
#     return dict(data)

def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    data = defaultdict(list)
    for r in records:
        s          = r.get("strategy", "unknown")
        compile_ok = int(r.get("compile", {}).get("compile_ok", False))
        passed     = r.get("execution", {}).get("passed", 0)
        total      = r.get("execution", {}).get("total",  0)
        pass_rate  = passed / total if total > 0 else 0.0
        data[s].append({
            "trial":       r.get("trial"),
            "compile_ok":  compile_ok,
            "pass_rate":   pass_rate,
            "passed":      passed,
            "total":       total,
            "function_id": r.get("function_id"),
            "unique_fid":  r.get("unique_fid"),
        })
    return dict(data)


def load_metrics(path: str) -> dict:
    """Load RQ1 + RQ3 data from metrics_results.json.
    Each record contains strategy, coverage, complexity, assertions fields.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    data = defaultdict(lambda: {
        "line_rate":       [],
        "branch_rate":     [],
        "avg_complexity":  [],
        "avg_assertions":  [],
    })

    for r in records:
        s = r.get("strategy", "unknown")
        cov  = r.get("coverage",   {})
        comp = r.get("complexity", {})
        assr = r.get("assertions", {})

        data[s]["line_rate"].append(cov.get("line_rate"))
        data[s]["branch_rate"].append(cov.get("branch_rate"))
        data[s]["avg_complexity"].append(comp.get("avg_complexity"))
        data[s]["avg_assertions"].append(assr.get("avg_assertions"))

    return dict(data)


# ─────────────────────────────────────────────
# RQ2 analysis
# ─────────────────────────────────────────────
def analyse_rq2(pipeline_data: dict) -> dict:
    results = {}

    # ── Compilation rate ───────────────────────
    compile_groups = {}
    for s, records in pipeline_data.items():
        compile_groups[s] = [r["compile_ok"] for r in records]

    results["compilation_rate"] = {
        "descriptive":  {s: descriptive(compile_groups[s])
                         for s in STRATEGIES if s in compile_groups},
        "normality":    {s: normality_test(compile_groups[s])
                         for s in STRATEGIES if s in compile_groups},
        "kruskal":      kruskal_wallis_test(compile_groups),
        "posthoc":      posthoc_tests(compile_groups),
        "by_trial":     {},
    }

    # Compilation rate by trial
    for s, records in pipeline_data.items():
        by_trial = defaultdict(list)
        for r in records:
            by_trial[r["trial"]].append(r["compile_ok"])
        results["compilation_rate"]["by_trial"][s] = {
            t: round(sum(v) / len(v) * 100, 1)
            for t, v in sorted(by_trial.items())
        }

    # ── Execution pass rate ───────────────────
    pass_groups = {}
    for s, records in pipeline_data.items():
        pass_groups[s] = [r["pass_rate"] for r in records]

    results["execution_pass_rate"] = {
        "descriptive":  {s: descriptive(pass_groups[s])
                         for s in STRATEGIES if s in pass_groups},
        "normality":    {s: normality_test(pass_groups[s])
                         for s in STRATEGIES if s in pass_groups},
        "kruskal":      kruskal_wallis_test(pass_groups),
        "posthoc":      posthoc_tests(pass_groups),
        "by_trial":     {},
    }

    for s, records in pipeline_data.items():
        by_trial = defaultdict(list)
        for r in records:
            by_trial[r["trial"]].append(r["pass_rate"])
        results["execution_pass_rate"]["by_trial"][s] = {
            t: round(sum(v) / len(v) * 100, 1)
            for t, v in sorted(by_trial.items())
        }

    # ── Pass@k ───────────────────────────────
    passk_results = {}
    for s, records in pipeline_data.items():
        # Group by function_id
        func_trials = defaultdict(list)
        for r in records:
            # fid = r.get("function_id", "unknown")

            # start_line = r.get("start_line", 0)
            # fid = f"{r.get('function_id', 'unknown')}.L{start_line}"
            fid = r.get("unique_fid") or r.get("function_id", "unknown")

            func_trials[fid].append(1 if r["pass_rate"] > 0 else 0)

        pass1_list, pass3_list = [], []
        for fid, trials in func_trials.items():
            n = len(trials)
            c = sum(trials)
            pass1_list.append(pass_at_k(n, c, 1))
            pass3_list.append(pass_at_k(n, c, 3))

        passk_results[s] = {
            "pass_at_1": round(sum(pass1_list) / len(pass1_list), 4)
                         if pass1_list else None,
            "pass_at_3": round(sum(pass3_list) / len(pass3_list), 4)
                         if pass3_list else None,
        }

    results["pass_at_k"] = passk_results
    return results


# ─────────────────────────────────────────────
# RQ1 analysis
# ─────────────────────────────────────────────
def analyse_rq1(metrics_data: dict) -> dict:
    results = {}

    for metric_key, label in [
        ("line_rate",   "line_coverage"),
        ("branch_rate", "branch_coverage"),
    ]:
        groups = {s: metrics_data[s][metric_key]
                  for s in STRATEGIES if s in metrics_data}

        results[label] = {
            "descriptive": {s: descriptive(groups[s])
                            for s in STRATEGIES if s in groups},
            "normality":   {s: normality_test(groups[s])
                            for s in STRATEGIES if s in groups},
            "kruskal":     kruskal_wallis_test(groups),
            "posthoc":     posthoc_tests(groups),
        }

        if metric_key == "branch_rate":
            n_excluded = sum(
                1 for s in STRATEGIES if s in groups
                for v in groups[s] if v is None
            )
            results[label]["n_excluded_no_branch"] = n_excluded

    return results


# ─────────────────────────────────────────────
# RQ3 analysis
# ─────────────────────────────────────────────
def analyse_rq3(metrics_data: dict) -> dict:
    results = {}

    for metric_key, label in [
        ("avg_complexity",  "cyclomatic_complexity"),
        ("avg_assertions",  "assertion_density"),
    ]:
        groups = {s: metrics_data[s][metric_key]
                  for s in STRATEGIES if s in metrics_data}

        results[label] = {
            "descriptive": {s: descriptive(groups[s])
                            for s in STRATEGIES if s in groups},
            "normality":   {s: normality_test(groups[s])
                            for s in STRATEGIES if s in groups},
            "kruskal":     kruskal_wallis_test(groups),
            "posthoc":     posthoc_tests(groups),
        }

    return results


# ─────────────────────────────────────────────
# Text report generation
# ─────────────────────────────────────────────
def generate_report(all_results: dict) -> str:
    lines = []
    sep = "=" * 65

    def sig_mark(p):
        if p is None:
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    lines.append(sep)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append(f"Bonferroni-adjusted threshold: p < {ALPHA_ADJ:.4f}")
    lines.append(sep)

    for rq_label, rq_data in all_results.items():
        lines.append(f"\n{'─'*65}")
        lines.append(f"  {rq_label.upper()}")
        lines.append(f"{'─'*65}")

        for metric_name, mdata in rq_data.items():
            if metric_name == "pass_at_k":
                lines.append(f"\n  Pass@k:")
                for s, v in mdata.items():
                    lines.append(
                        f"    {s:<14}  Pass@1={v['pass_at_1']:.4f}"
                        f"  Pass@3={v['pass_at_3']:.4f}"
                    )
                continue

            lines.append(f"\n  [{metric_name}]")

            # Descriptive statistics
            if "descriptive" in mdata:
                lines.append(
                    f"  {'Strategy':<14} {'Mean':>8} {'Median':>8}"
                    f" {'IQR':>8} {'n':>5}"
                )
                lines.append(f"  {'-'*46}")
                for s in STRATEGIES:
                    d = mdata["descriptive"].get(s, {})
                    if d:
                        lines.append(
                            f"  {s:<14} {d['mean']:>8.4f}"
                            f" {d['median']:>8.4f}"
                            f" {d['iqr']:>8.4f}"
                            f" {d['n']:>5}"
                        )

            # Normality test
            if "normality" in mdata:
                lines.append(f"\n  Shapiro-Wilk normality test:")
                for s in STRATEGIES:
                    n = mdata["normality"].get(s, {})
                    if n:
                        lines.append(
                            f"    {s:<14}  W={n['stat']:.4f}"
                            f"  p={n['p']:.4f}"
                            f"  {'normal' if n['normal'] else 'non-normal'}"
                        )

            # Kruskal-Wallis
            if "kruskal" in mdata:
                kw = mdata["kruskal"]
                lines.append(
                    f"\n  Kruskal-Wallis:  H={kw['H']}  p={kw['p']}"
                    f"  {sig_mark(kw['p'])}"
                    f"  {'SIGNIFICANT' if kw['significant'] else 'not significant'}"
                )

            # Post-hoc comparisons
            if "posthoc" in mdata and mdata["posthoc"]:
                lines.append(f"\n  Post-hoc Wilcoxon (Bonferroni-corrected):")
                lines.append(
                    f"  {'Pair':<28} {'W':>8} {'p_raw':>8}"
                    f" {'p_adj':>8} {'r':>7} {'effect':<10} sig"
                )
                lines.append(f"  {'-'*72}")
                for ph in mdata["posthoc"]:
                    lines.append(
                        f"  {ph['pair']:<28} {ph['stat']:>8.2f}"
                        f" {ph['p_raw']:>8.4f}"
                        f" {ph['p_adjusted']:>8.4f}"
                        f" {ph['r']:>7.4f}"
                        f" {ph['effect_size']:<10}"
                        f" {'*' if ph['significant'] else ''}"
                    )

            if "n_excluded_no_branch" in mdata:
                lines.append(
                    f"\n  Note: {mdata['n_excluded_no_branch']} "
                    f"function-strategy pairs excluded (no conditional branches)."
                )

    lines.append(f"\n{sep}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis script: compute all metrics for the thesis"
    )
    parser.add_argument(
        "--pipeline", required=True,
        help="Path to pipeline_results.json (compilation/execution data for all three trials)"
    )
    parser.add_argument(
        "--metrics", required=True,
        help="Path to metrics_results.json (coverage/complexity/assertion data for best trial)"
    )
    parser.add_argument(
        "--output", default="../results/statistical_results.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    print("[load] loading pipeline data...")
    pipeline_data = load_pipeline(args.pipeline)
    print(f"       strategies: {list(pipeline_data.keys())}")

    print("[load] loading metrics data...")
    metrics_data = load_metrics(args.metrics)
    print(f"       strategies: {list(metrics_data.keys())}")

    print("\n[analyse] RQ2 correctness metrics...")
    rq2 = analyse_rq2(pipeline_data)

    print("[analyse] RQ1 coverage metrics...")
    rq1 = analyse_rq1(metrics_data)

    print("[analyse] RQ3 readability metrics...")
    rq3 = analyse_rq3(metrics_data)

    all_results = {
        "RQ2_correctness": rq2,
        "RQ1_coverage":    rq1,
        "RQ3_readability": rq3,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] JSON results saved to: {args.output}")

    report = generate_report(all_results)
    report_path = Path(args.output).with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[save] text report saved to: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
