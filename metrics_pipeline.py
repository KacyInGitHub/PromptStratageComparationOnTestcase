"""
metrics_pipeline.py
===================
模块化指标采集脚本，每个指标可单独运行。

使用方式：
  # 单独运行某个指标
  python metrics_pipeline.py --metric complexity  --data_dir tests_for_coverage/ --output results_complexity.json
  python metrics_pipeline.py --metric assertions  --data_dir tests_for_coverage/ --output results_assertions.json
  python metrics_pipeline.py --metric coverage    --data_dir tests_for_coverage/ --output results_coverage.json
  python metrics_pipeline.py --metric mutation    --data_dir tests_for_coverage/ --output results_mutation.json

  # 运行全部指标
  python metrics_pipeline.py --metric all         --data_dir tests_for_coverage/ --output results_all.json

  # 合并多个已有结果文件
  python metrics_pipeline.py --metric merge \
      --merge_files results_complexity.json results_coverage.json results_mutation.json \
      --output results_merged.json
"""

import ast
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# 配置：导入修复映射
# ─────────────────────────────────────────────
MODULE_IMPORT_CFG = {
    "requests/utils.py": {
        "module": "requests.utils",
        "funcs": [
            "address_in_network", "dotted_netmask", "is_ipv4_address",
            "is_valid_cidr", "requote_uri", "unquote_unreserved",
            "parse_list_header", "parse_dict_header", "unquote_header_value",
            "parse_header_links", "iter_slices", "from_key_val_list",
            "to_key_val_list", "get_encoding_from_headers",
            "resolve_proxies", "urldefragauth", "_validate_header_part",
            "get_auth_from_url", "prepend_scheme_if_needed",
            "select_proxy", "should_bypass_proxies", "get_environ_proxies",
        ],
        "extras": [
            "from requests.exceptions import InvalidURL, InvalidHeader, UnrewindableBodyError",
        ],
    },
    "requests/models.py": {
        "module": "requests.models",
        "funcs": ["PreparedRequest", "Request", "Response"],
        "extras": [
            "from requests.exceptions import HTTPError, JSONDecodeError",
            "from requests.auth import HTTPBasicAuth",
            "from requests.cookies import RequestsCookieJar",
        ],
    },
    "locales.py": {
        "module": "arrow.locales",
        "funcs": [
            "get_locale", "get_locale_by_class_name",
            "EnglishLocale", "FinnishLocale", "GermanLocale",
            "TamilLocale", "BengaliLocale", "Locale",
        ],
        "extras": [],
    },
    "arrow.py": {
        "module": "arrow",
        "funcs": ["Arrow"],
        "extras": [
            "import arrow",
            "from arrow.arrow import Arrow",
            "from datetime import datetime, timezone, timedelta",
        ],
    },
    "parser.py": {
        "module": "arrow.parser",
        "funcs": ["DateTimeParser"],
        "extras": [
            "from arrow.parser import DateTimeParser, ParserError, ParserMatchError",
        ],
    },
    "util.py": {
        "module": "arrow.util",
        "funcs": ["is_timestamp", "normalize_timestamp"],
        "extras": [],
    },
    "more.py": {
        "module": "more_itertools.more",
        "funcs": [],
        "extras": [
            "from more_itertools import *",
            "from more_itertools.more import *",
        ],
    },
    "recipes.py": {
        "module": "more_itertools.recipes",
        "funcs": [],
        "extras": [
            "from more_itertools import *",
            "from more_itertools.recipes import *",
        ],
    },
}

CLASS_MAP = {
    ("requests", "requests/models.py"): "PreparedRequest",
    ("arrow",    "locales.py"):          "EnglishLocale",
    ("arrow",    "arrow.py"):            "Arrow",
    ("arrow",    "parser.py"):           "DateTimeParser",
}

# 模块点路径映射（用于 importlib 定位实际文件）
MODULE_DOTTED = {
    "requests/models.py": "requests.models",
    "requests/utils.py":  "requests.utils",
    "arrow.py":           "arrow.arrow",
    "locales.py":         "arrow.locales",
    "parser.py":          "arrow.parser",
    "util.py":            "arrow.util",
    "more.py":            "more_itertools.more",
    "recipes.py":         "more_itertools.recipes",
}


# ─────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────
def parse_strategy(filename: str) -> str:
    match = re.search(r"generated_tests_(.+)\.json", filename)
    return match.group(1) if match else "unknown"


def load_all(data_dir: str) -> list:
    all_records = []
    for path in sorted(Path(data_dir).glob("generated_tests_*.json")):
        strategy = parse_strategy(path.name)
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["strategy"] = strategy
        all_records.extend(records)
    print(f"[load] 共加载 {len(all_records)} 条记录，来自 {data_dir}")
    return all_records


def fix_imports(source: str, project: str, module: str) -> str:
    # 移除占位符
    source = re.sub(r"^.*your_module.*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"#\s*Replace.*$",    "", source, flags=re.MULTILINE)

    # 替换 YourClass
    real_class = CLASS_MAP.get((project, module))
    if real_class:
        source = source.replace("YourClass()", f"{real_class}()")
        source = source.replace("YourClass",    real_class)

    # 注入正确 import
    cfg = MODULE_IMPORT_CFG.get(module, {})
    inject_lines = []
    if cfg.get("funcs"):
        inject_lines.append(
            f"from {cfg['module']} import {', '.join(cfg['funcs'])}"
        )
    elif cfg.get("module"):
        inject_lines.append(f"import {cfg['module']}")
    inject_lines.extend(cfg.get("extras", []))

    if inject_lines:
        lines = source.split("\n")
        insert_pos = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                insert_pos = i + 1
        lines.insert(insert_pos, "\n".join(inject_lines))
        source = "\n".join(lines)

    return re.sub(r"\n{3,}", "\n\n", source).strip()


def base_record(r: dict) -> dict:
    """生成结果记录的基础字段"""
    return {
        "function_id": f"{r['project']}.{r['name']}",
        "project":     r["project"],
        "module":      r["module"],
        "name":        r["name"],
        "strategy":    r.get("strategy"),
    }


def save(results: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] 结果已保存至: {output_path}  ({len(results)} 条)")


# ─────────────────────────────────────────────
# 指标一：圈复杂度
# ─────────────────────────────────────────────
def metric_complexity(records: list) -> list:
    """
    使用 radon 计算测试代码中每个 test_ 函数的平均圈复杂度。
    依赖：pip install radon
    纯静态分析，无需执行，速度快。
    """
    try:
        from radon.complexity import cc_visit
    except ImportError:
        print("[error] 请先安装: pip install radon")
        sys.exit(1)

    results = []
    for r in records:
        out = base_record(r)
        source = r.get("tests_source", "")
        try:
            blocks = cc_visit(source)
            test_blocks = [b for b in blocks if b.name.startswith("test_")]
            if test_blocks:
                avg = sum(b.complexity for b in test_blocks) / len(test_blocks)
                out["complexity"] = {
                    "avg_complexity":  round(avg, 2),
                    "test_func_count": len(test_blocks),
                    "detail": [
                        {"name": b.name, "complexity": b.complexity}
                        for b in test_blocks
                    ],
                }
            else:
                out["complexity"] = {"avg_complexity": None, "test_func_count": 0}
        except Exception as e:
            out["complexity"] = {"avg_complexity": None, "error": str(e)}
        results.append(out)
    return results


# ─────────────────────────────────────────────
# 指标二：每函数断言数
# ─────────────────────────────────────────────
def metric_assertions(records: list) -> list:
    """
    使用 AST 统计每个 test_ 函数的断言数量，计算平均值。
    纯静态分析，无需执行，速度快。
    """
    results = []
    for r in records:
        out = base_record(r)
        source = r.get("tests_source", "")
        try:
            tree = ast.parse(source)
            counts = []
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef) and
                        node.name.startswith("test_")):
                    n = sum(
                        1 for n in ast.walk(node)
                        if isinstance(n, ast.Assert)
                    )
                    counts.append({"func": node.name, "assertions": n})

            if counts:
                vals = [c["assertions"] for c in counts]
                out["assertions"] = {
                    "avg_assertions":   round(sum(vals) / len(vals), 2),
                    "total_assertions": sum(vals),
                    "test_func_count":  len(vals),
                    "detail":           counts,
                }
            else:
                out["assertions"] = {
                    "avg_assertions": None,
                    "test_func_count": 0,
                }
        except SyntaxError as e:
            out["assertions"] = {"avg_assertions": None, "error": str(e)}
        results.append(out)
    return results


# ─────────────────────────────────────────────
# 指标三：行覆盖率 & 分支覆盖率（函数级）
# ─────────────────────────────────────────────
def get_func_lines(module: str, func_name: str) -> tuple:
    """
    用 importlib 定位模块文件，再用 AST 找到函数的起止行号。
    返回 (文件绝对路径, 起始行, 结束行)，找不到返回 (None, None, None)。
    """
    import importlib.util

    dotted = MODULE_DOTTED.get(module)
    if not dotted:
        return None, None, None

    spec = importlib.util.find_spec(dotted)
    if not spec or not spec.origin:
        return None, None, None

    file_path = spec.origin
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == func_name):
                return file_path, node.lineno, node.end_lineno
    except Exception:
        pass

    return file_path, None, None


def metric_coverage(records: list, timeout: int = 60) -> list:
    """
    函数级行覆盖率 & 分支覆盖率。
    使用 `coverage run --source=<dotted> -m pytest` 方案，
    可正确追踪 site-packages 中已安装的包，解决 pytest --cov
    对绝对路径无法识别的问题。
    分母只统计被测函数内的可执行行/分支。
    依赖：pip install coverage pytest requests arrow more-itertools
    """
    results = []
    total   = len(records)

    for i, r in enumerate(records):
        out       = base_record(r)
        project   = r["project"]
        module    = r["module"]
        func_name = r["name"]
        test_src  = fix_imports(r.get("tests_source", ""), project, module)

        print(
            f"  [coverage {i+1:03d}/{total}] "
            f"{r['strategy']:<12} {project}.{func_name}",
            end=" ... ", flush=True,
        )

        # 1. 定位被测函数的行号范围
        file_path, start_line, end_line = get_func_lines(module, func_name)
        dotted = MODULE_DOTTED.get(module)

        if not file_path or not dotted:
            out["coverage"] = {
                "line_rate": None, "branch_rate": None,
                "error": f"cannot locate module: {module}",
            }
            print("module not found")
            results.append(out)
            continue

        if not start_line:
            out["coverage"] = {
                "line_rate": None, "branch_rate": None,
                "error": f"function '{func_name}' not found",
            }
            print("func not found")
            results.append(out)
            continue

        uid           = uuid.uuid4().hex
        tmp_test      = f"/tmp/test_{uid}.py"
        cov_data_file = f"/tmp/.coverage_{uid}"
        cov_json      = f"/tmp/cov_{uid}.json"

        with open(tmp_test, "w", encoding="utf-8") as f:
            f.write(test_src)

        try:
            # 2. coverage run --source=<dotted.module> --branch -m pytest
            subprocess.run(
                [
                    sys.executable, "-m", "coverage", "run",
                    f"--data-file={cov_data_file}",
                    f"--source={dotted}",
                    "--branch",
                    "-m", "pytest", tmp_test,
                    "-q", "--tb=no", "--no-header",
                ],
                capture_output=True, text=True, timeout=timeout,
            )

            # 3. 导出 JSON 报告
            subprocess.run(
                [
                    sys.executable, "-m", "coverage", "json",
                    f"--data-file={cov_data_file}",
                    "-o", cov_json,
                ],
                capture_output=True, text=True, timeout=30,
            )

            if not os.path.exists(cov_json):
                out["coverage"] = {
                    "line_rate": None, "branch_rate": None,
                    "error": "no coverage output",
                }
                print("no output")
                results.append(out)
                continue

            with open(cov_json, encoding="utf-8") as f:
                cov_data = json.load(f)

            # 4. 找到对应文件的数据（先绝对路径匹配，再文件名兜底）
            file_data = None
            for fname, fdata in cov_data.get("files", {}).items():
                if os.path.abspath(fname) == os.path.abspath(file_path):
                    file_data = fdata
                    break
            if not file_data:
                base = os.path.basename(file_path)
                for fname, fdata in cov_data.get("files", {}).items():
                    if os.path.basename(fname) == base:
                        file_data = fdata
                        break

            if not file_data:
                out["coverage"] = {
                    "line_rate": None, "branch_rate": None,
                    "error": "file not found in coverage report",
                }
                print("not in report")
                results.append(out)
                continue

            # 5. 只统计函数行号范围内的行和分支
            func_line_set  = set(range(start_line, end_line + 1))
            executed_lines = set(file_data.get("executed_lines", []))
            missing_lines  = set(file_data.get("missing_lines",  []))
            func_all       = (executed_lines | missing_lines) & func_line_set
            func_covered   = executed_lines & func_line_set

            line_rate = (
                len(func_covered) / len(func_all)
                if func_all else None
            )

            exec_br      = file_data.get("executed_branches", [])
            miss_br      = file_data.get("missing_branches",  [])
            func_exec_br = [b for b in exec_br if start_line <= b[0] <= end_line]
            func_miss_br = [b for b in miss_br if start_line <= b[0] <= end_line]
            total_br     = len(func_exec_br) + len(func_miss_br)

            branch_rate = (
                len(func_exec_br) / total_br
                if total_br else None
            )

            out["coverage"] = {
                "line_rate":           round(line_rate,   4) if line_rate   is not None else None,
                "branch_rate":         round(branch_rate, 4) if branch_rate is not None else None,
                "covered_lines":       len(func_covered),
                "total_func_lines":    len(func_all),
                "covered_branches":    len(func_exec_br),
                "total_func_branches": total_br,
                "func_line_range":     [start_line, end_line],
            }
            lr = out["coverage"]["line_rate"]
            br = out["coverage"]["branch_rate"]
            print(f"line={lr}  branch={br}")

        except subprocess.TimeoutExpired:
            out["coverage"] = {
                "line_rate": None, "branch_rate": None,
                "timeout": True,
            }
            print("timeout")
        finally:
            for p in [tmp_test, cov_json, cov_data_file]:
                if os.path.exists(p):
                    os.unlink(p)

        results.append(out)
    return results


# ─────────────────────────────────────────────
# 指标四：变异得分
# ─────────────────────────────────────────────
def metric_mutation(records: list, timeout: int = 180) -> list:
    """
    使用 mutmut 对每个函数进行变异测试，计算变异得分。
    依赖：pip install mutmut
    耗时最长，建议最后单独运行。
    """
    results = []
    total = len(records)

    for i, r in enumerate(records):
        out       = base_record(r)
        project   = r["project"]
        module    = r["module"]
        func_src  = r.get("source", "")
        test_src  = fix_imports(r.get("tests_source", ""), project, module)

        print(
            f"  [mutation {i+1:03d}/{total}] "
            f"{r['strategy']:<12} {r['project']}.{r['name']}",
            end=" ... ", flush=True,
        )

        uid      = uuid.uuid4().hex
        work_dir = f"/tmp/mutmut_{uid}"
        os.makedirs(work_dir, exist_ok=True)

        func_path = os.path.join(work_dir, "target.py")
        test_path = os.path.join(work_dir, "test_target.py")

        # 测试代码中的原始 import 替换为 from target import *
        adjusted = re.sub(
            r"^(from\s+\S+\s+import|import\s+(?!pytest|unittest)\S+)",
            r"# \1",
            test_src,
            flags=re.MULTILINE,
        )
        test_content = f"from target import *\n{adjusted}"

        with open(func_path, "w", encoding="utf-8") as f:
            f.write(func_src)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_content)

        try:
            subprocess.run(
                [
                    sys.executable, "-m", "mutmut", "run",
                    "--paths-to-mutate", func_path,
                    "--runner",
                    f"{sys.executable} -m pytest {test_path} -x -q --tb=no",
                ],
                capture_output=True, text=True,
                timeout=timeout, cwd=work_dir,
            )

            res_proc = subprocess.run(
                [sys.executable, "-m", "mutmut", "results"],
                capture_output=True, text=True, cwd=work_dir,
            )
            stdout = res_proc.stdout
            killed   = len(re.findall(r"Killed",   stdout))
            survived = len(re.findall(r"Survived", stdout))
            total_m  = killed + survived

            out["mutation"] = {
                "mutation_score": round(killed / total_m, 4) if total_m else None,
                "killed":         killed,
                "survived":       survived,
                "total_mutants":  total_m,
            }
            print(
                f"score={out['mutation']['mutation_score']}  "
                f"killed={killed}  survived={survived}"
            )

        except subprocess.TimeoutExpired:
            out["mutation"] = {"mutation_score": None, "timeout": True}
            print("timeout")
        except FileNotFoundError:
            out["mutation"] = {"mutation_score": None,
                               "error": "mutmut not installed"}
            print("mutmut not found")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        results.append(out)
    return results


# ─────────────────────────────────────────────
# 合并多个结果文件
# ─────────────────────────────────────────────
def merge_results(file_paths: list) -> list:
    """
    按 (function_id, strategy) 合并多个指标文件。
    每个文件包含不同指标，合并后每条记录包含所有指标。
    """
    merged = {}

    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        print(f"[merge] 读取 {path}  ({len(records)} 条)")

        for r in records:
            key = (r["function_id"], r.get("strategy"))
            if key not in merged:
                merged[key] = {
                    "function_id": r["function_id"],
                    "project":     r["project"],
                    "module":      r["module"],
                    "name":        r["name"],
                    "strategy":    r.get("strategy"),
                }
            # 合并各指标字段
            for metric in ["complexity", "assertions", "coverage", "mutation"]:
                if metric in r:
                    merged[key][metric] = r[metric]

    result_list = list(merged.values())
    print(f"[merge] 合并完成，共 {len(result_list)} 条记录")
    return result_list


# ─────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────
def print_summary(results: list, metric: str):
    print(f"\n{'='*70}")
    print(f"SUMMARY  —  metric: {metric}")
    print(f"{'='*70}")

    grouped = defaultdict(list)
    for r in results:
        grouped[r.get("strategy", "unknown")].append(r)

    def safe_avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    if metric in ("complexity", "all", "merge"):
        print(f"\n{'策略':<20} {'平均圈复杂度':>14} {'函数数':>8}")
        print("-" * 45)
        for s, recs in sorted(grouped.items()):
            cc  = safe_avg([r.get("complexity", {}).get("avg_complexity") for r in recs])
            cnt = safe_avg([r.get("complexity", {}).get("test_func_count") for r in recs])
            print(f"  {s:<18} {fmt(cc):>14} {fmt(cnt):>8}")

    if metric in ("assertions", "all", "merge"):
        print(f"\n{'策略':<20} {'平均断言数/函数':>16} {'平均总断言数':>14}")
        print("-" * 55)
        for s, recs in sorted(grouped.items()):
            avg_a = safe_avg([r.get("assertions", {}).get("avg_assertions")   for r in recs])
            tot_a = safe_avg([r.get("assertions", {}).get("total_assertions")  for r in recs])
            print(f"  {s:<18} {fmt(avg_a):>16} {fmt(tot_a):>14}")

    if metric in ("coverage", "all", "merge"):
        print(f"\n{'策略':<20} {'行覆盖率':>12} {'分支覆盖率':>12}")
        print("-" * 48)
        for s, recs in sorted(grouped.items()):
            lr = safe_avg([r.get("coverage", {}).get("line_rate")   for r in recs])
            br = safe_avg([r.get("coverage", {}).get("branch_rate") for r in recs])
            print(f"  {s:<18} {fmt(lr):>12} {fmt(br):>12}")

        no_branch_count = sum(
            1 for r in recs
            if r.get("coverage", {}).get("total_func_branches", 0) == 0
        )
        print(f"    (其中 {no_branch_count} 个函数无分支，分支覆盖率不适用)")

    if metric in ("mutation", "all", "merge"):
        print(f"\n{'策略':<20} {'变异得分':>12} {'平均killed':>12} {'平均survived':>14}")
        print("-" * 62)
        for s, recs in sorted(grouped.items()):
            ms = safe_avg([r.get("mutation", {}).get("mutation_score") for r in recs])
            ki = safe_avg([r.get("mutation", {}).get("killed")         for r in recs])
            su = safe_avg([r.get("mutation", {}).get("survived")       for r in recs])
            print(f"  {s:<18} {fmt(ms):>12} {fmt(ki):>12} {fmt(su):>14}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
METRICS = ["complexity", "assertions", "coverage", "mutation", "all", "merge"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="模块化测试指标采集脚本"
    )
    parser.add_argument(
        "--metric", required=True, choices=METRICS,
        help="要计算的指标，或 all / merge"
    )
    parser.add_argument(
        "--data_dir", default="tests_for_coverage/",
        help="JSON 数据目录（merge 模式下不需要）"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出 JSON 文件路径"
    )
    parser.add_argument(
        "--merge_files", nargs="+",
        help="merge 模式：要合并的结果文件列表"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="单条记录执行超时秒数（coverage/mutation 有效）"
    )
    args = parser.parse_args()

    # merge 模式
    if args.metric == "merge":
        if not args.merge_files:
            print("[error] merge 模式需要 --merge_files 参数")
            sys.exit(1)
        results = merge_results(args.merge_files)
        save(results, args.output)
        print_summary(results, "merge")
        sys.exit(0)

    # 加载数据
    records = load_all(args.data_dir)

    # 按指标分发
    if args.metric == "complexity":
        results = metric_complexity(records)

    elif args.metric == "assertions":
        results = metric_assertions(records)

    elif args.metric == "coverage":
        results = metric_coverage(records, timeout=args.timeout)

    elif args.metric == "mutation":
        results = metric_mutation(records, timeout=args.timeout)

    elif args.metric == "all":
        print("\n[all] 依次计算所有指标...")
        comp = metric_complexity(records)
        assr = metric_assertions(records)
        cov  = metric_coverage(records,  timeout=args.timeout)
        mut  = metric_mutation(records,   timeout=args.timeout * 3)

        # 合并四个结果
        results = merge_results([])   # 借用 merge 逻辑
        # 直接在内存中合并
        merged = {}
        for lst in [comp, assr, cov, mut]:
            for r in lst:
                key = (r["function_id"], r.get("strategy"))
                if key not in merged:
                    merged[key] = {k: v for k, v in r.items()}
                else:
                    merged[key].update(r)
        results = list(merged.values())

    save(results, args.output)
    print_summary(results, args.metric)
