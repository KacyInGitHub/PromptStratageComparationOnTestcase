"""
4_compute_metrics.py
====================
Modular metric collection script; each metric can be run independently.

Usage:
  # Run a single metric
  python 4_compute_metrics.py --metric complexity  --data_dir ../tests_for_coverage/ --output results_complexity.json
  python 4_compute_metrics.py --metric assertions  --data_dir ../tests_for_coverage/ --output results_assertions.json
  python 4_compute_metrics.py --metric coverage    --data_dir ../tests_for_coverage/ --output results_coverage.json
  python 4_compute_metrics.py --metric mutation    --data_dir ../tests_for_coverage/ --output results_mutation.json

  # Run all metrics
  python 4_compute_metrics.py --metric all         --data_dir ../tests_for_coverage/ --output results_all.json

  # Merge multiple existing result files
  python 4_compute_metrics.py --metric merge \
      --merge_files results_complexity.json results_coverage.json results_assertions.json \
      --output ../results/metrics_results.json
"""

import ast
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# Import fix configuration
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

# Dotted module path mapping (used by importlib to locate source files)
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
# Candidate function loader: source hash -> lineno map
# ─────────────────────────────────────────────
def load_candidate_source_map(
        candidate_path: str = "../experiment_data/candidate_functions.json") -> dict:
    """
    Build a (project, name, source_hash) -> lineno lookup table.
    Uses the source content to uniquely identify each candidate function,
    correctly distinguishing same-named functions (e.g. multiple _format_relative in arrow locales).
    """
    try:
        with open(candidate_path, encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print(f"[warn] {candidate_path} not found; same-named functions cannot be distinguished")
        return {}

    source_map = {}
    for c in candidates:
        source_hash = hashlib.md5(
            c["source"].strip().encode("utf-8")
        ).hexdigest()
        key = (c["project"], c["name"], source_hash)
        source_map[key] = c["lineno"]

    print(f"[candidate] loaded {len(source_map)} candidate functions")
    return source_map


# ─────────────────────────────────────────────
# Common utilities
# ─────────────────────────────────────────────
def parse_strategy(filename: str) -> str:
    match = re.search(r"generated_tests_(.+)\.json", filename)
    return match.group(1) if match else "unknown"


def load_all(data_dir: str,
             candidate_path: str = "../experiment_data/candidate_functions.json") -> list:
    """
    Load all generated test files and match each record to candidate_functions.json
    via source hash, writing lineno and unique_fid fields.
    """
    source_map = load_candidate_source_map(candidate_path)
    unmatched  = []
    all_records = []

    for path in sorted(Path(data_dir).glob("generated_tests_*.json")):
        strategy = parse_strategy(path.name)
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            r["strategy"] = strategy

            project = r.get("project", "")
            name    = r.get("name", "")
            source  = r.get("source", "").strip()

            source_hash = hashlib.md5(
                source.encode("utf-8")
            ).hexdigest()
            key    = (project, name, source_hash)
            lineno = source_map.get(key)

            if lineno is not None:
                r["lineno"]     = lineno
                r["unique_fid"] = f"{project}.{name}.L{lineno}"
            else:
                r["lineno"]     = 0
                r["unique_fid"] = f"{project}.{name}.L0"
                unmatched.append(f"{project}.{name}")

        all_records.extend(records)

    if unmatched:
        print(f"[warn] {len(unmatched)} records could not be matched to a candidate function")
        for u in sorted(set(unmatched)):
            print(f"       {u}")
    else:
        print("[load] all records matched successfully to candidate functions")

    print(f"[load] loaded {len(all_records)} records from {data_dir}")
    return all_records


def fix_imports(source: str, project: str, module: str) -> str:
    # Remove placeholder imports
    source = re.sub(r"^.*your_module.*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"#\s*Replace.*$",    "", source, flags=re.MULTILINE)

    # Replace YourClass with real class name
    real_class = CLASS_MAP.get((project, module))
    if real_class:
        source = source.replace("YourClass()", f"{real_class}()")
        source = source.replace("YourClass",    real_class)

    # Inject correct imports
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
    """
    Build the base fields for a result record.
    unique_fid includes the line number to distinguish same-named functions;
    function_id keeps the original format for cross-referencing with other data sources.
    """
    return {
        "function_id": f"{r['project']}.{r['name']}",
        "unique_fid":  r.get("unique_fid",
                       f"{r['project']}.{r['name']}.L0"),
        "project":     r["project"],
        "module":      r["module"],
        "name":        r["name"],
        "lineno":      r.get("lineno", 0),
        "strategy":    r.get("strategy"),
    }


def save(results: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] results saved to: {output_path}  ({len(results)} records)")


# ─────────────────────────────────────────────
# Metric 1: Cyclomatic complexity
# ─────────────────────────────────────────────
def metric_complexity(records: list) -> list:
    """
    Compute mean cyclomatic complexity of each test_ function using radon.
    Requires: pip install radon. Pure static analysis, no execution needed.
    """
    try:
        from radon.complexity import cc_visit
    except ImportError:
        print("[error] please install: pip install radon")
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
# Metric 2: Assertions per test function
# ─────────────────────────────────────────────
def metric_assertions(records: list) -> list:
    """
    Count assertions in each test_ function via AST and compute the mean.
    Pure static analysis, no execution needed.
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
# Metric 3: Line coverage & branch coverage (function-level)
# ─────────────────────────────────────────────
def get_func_lines(module: str, func_name: str,
                   target_lineno: int = 0) -> tuple:
    """
    Locate the module file via importlib and find the function's start/end lines via AST.
    If target_lineno > 0 and multiple same-named functions exist,
    select the one whose start line is closest to target_lineno;
    otherwise return the first match (backwards-compatible behaviour).
    Returns (absolute_file_path, start_line, end_line), or (None, None, None) if not found.
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

        # Collect all line ranges for same-named functions
        candidates = []
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == func_name):
                candidates.append((node.lineno, node.end_lineno))

        if not candidates:
            return file_path, None, None

        if target_lineno > 0 and len(candidates) > 1:
            # Multiple matches: pick the one closest to target_lineno
            best = min(candidates,
                       key=lambda x: abs(x[0] - target_lineno))
            return file_path, best[0], best[1]
        else:
            # Single match or no target_lineno provided: return first
            return file_path, candidates[0][0], candidates[0][1]

    except Exception:
        pass

    return file_path, None, None


def metric_coverage(records: list, timeout: int = 60) -> list:
    """
    Function-level line and branch coverage.
    Uses `coverage run --source=<dotted> -m pytest` to correctly track installed packages.
    Denominators count only executable lines/branches within the target function.
    Requires: pip install coverage pytest requests arrow more-itertools
    """
    results = []
    total   = len(records)

    for i, r in enumerate(records):
        out       = base_record(r)
        project   = r["project"]
        module    = r["module"]
        func_name = r["name"]
        lineno    = r.get("lineno", 0)
        test_src  = fix_imports(r.get("tests_source", ""), project, module)

        print(
            f"  [coverage {i+1:03d}/{total}] "
            f"{r['strategy']:<12} {project}.{func_name} (L{lineno})",
            end=" ... ", flush=True,
        )

        # 1. Locate the target function's line range (lineno disambiguates same-named functions)
        file_path, start_line, end_line = get_func_lines(
            module, func_name, target_lineno=lineno
        )
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
            # 2. Run coverage with branch tracking
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

            # 3. Export JSON coverage report
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

            # 4. Find file data (absolute path first, filename fallback)
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

            # 5. Count only lines and branches within the function's line range
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
# Metric 4: Mutation score
# ─────────────────────────────────────────────
def metric_mutation(records: list, timeout: int = 180) -> list:
    """
    Run mutation testing with mutmut and compute mutation score.
    Requires: pip install mutmut. Most time-consuming; run last as a standalone step.
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
            f"{r['strategy']:<12} {r['project']}.{r['name']} "
            f"(L{r.get('lineno', 0)})",
            end=" ... ", flush=True,
        )

        uid      = uuid.uuid4().hex
        work_dir = f"/tmp/mutmut_{uid}"
        os.makedirs(work_dir, exist_ok=True)

        func_path = os.path.join(work_dir, "target.py")
        test_path = os.path.join(work_dir, "test_target.py")

        # Replace original imports in test code with from target import *
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
# Merge multiple result files
# ─────────────────────────────────────────────
def merge_results(file_paths: list) -> list:
    """
    Merge multiple metric files by (unique_fid, strategy).
    Uses unique_fid (with line number) as the merge key so same-named functions are not conflated.
    Each file contains different metrics; the merged records contain all metrics.
    """
    merged = {}

    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        print(f"[merge] read {path}  ({len(records)} records)")

        for r in records:
            # Prefer unique_fid; fall back to function_id for older formats without line numbers
            unique_fid = r.get("unique_fid", r["function_id"])
            key = (unique_fid, r.get("strategy"))
            if key not in merged:
                merged[key] = {
                    "function_id": r["function_id"],
                    "unique_fid":  unique_fid,
                    "project":     r["project"],
                    "module":      r["module"],
                    "name":        r["name"],
                    "lineno":      r.get("lineno", 0),
                    "strategy":    r.get("strategy"),
                }
            # Merge metric fields
            for metric in ["complexity", "assertions", "coverage", "mutation"]:
                if metric in r:
                    merged[key][metric] = r[metric]

    result_list = list(merged.values())
    print(f"[merge] done, {len(result_list)} records total")
    return result_list


# ─────────────────────────────────────────────
# Summary report
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
        print(f"\n{'Strategy':<20} {'Avg complexity':>14} {'Test funcs':>10}")
        print("-" * 48)
        for s, recs in sorted(grouped.items()):
            cc  = safe_avg([r.get("complexity", {}).get("avg_complexity") for r in recs])
            cnt = safe_avg([r.get("complexity", {}).get("test_func_count") for r in recs])
            print(f"  {s:<18} {fmt(cc):>14} {fmt(cnt):>10}")

    if metric in ("assertions", "all", "merge"):
        print(f"\n{'Strategy':<20} {'Avg assertions/func':>20} {'Avg total assertions':>20}")
        print("-" * 63)
        for s, recs in sorted(grouped.items()):
            avg_a = safe_avg([r.get("assertions", {}).get("avg_assertions")   for r in recs])
            tot_a = safe_avg([r.get("assertions", {}).get("total_assertions")  for r in recs])
            print(f"  {s:<18} {fmt(avg_a):>20} {fmt(tot_a):>20}")

    if metric in ("coverage", "all", "merge"):
        print(f"\n{'Strategy':<20} {'Line rate':>12} {'Branch rate':>12}")
        print("-" * 48)
        for s, recs in sorted(grouped.items()):
            lr = safe_avg([r.get("coverage", {}).get("line_rate")   for r in recs])
            br = safe_avg([r.get("coverage", {}).get("branch_rate") for r in recs])
            print(f"  {s:<18} {fmt(lr):>12} {fmt(br):>12}")

        no_branch_count = sum(
            1 for r in results
            if r.get("coverage", {}).get("total_func_branches", 0) == 0
        )
        print(f"    ({no_branch_count} functions have no branches; branch coverage N/A)")

    if metric in ("mutation", "all", "merge"):
        print(f"\n{'Strategy':<20} {'Mutation score':>14} {'Avg killed':>12} {'Avg survived':>14}")
        print("-" * 64)
        for s, recs in sorted(grouped.items()):
            ms = safe_avg([r.get("mutation", {}).get("mutation_score") for r in recs])
            ki = safe_avg([r.get("mutation", {}).get("killed")         for r in recs])
            su = safe_avg([r.get("mutation", {}).get("survived")       for r in recs])
            print(f"  {s:<18} {fmt(ms):>14} {fmt(ki):>12} {fmt(su):>14}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
METRICS = ["complexity", "assertions", "coverage", "mutation", "all", "merge"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modular test metric collection script"
    )
    parser.add_argument(
        "--metric", required=True, choices=METRICS,
        help="Metric to compute, or 'all' / 'merge'"
    )
    parser.add_argument(
        "--data_dir", default="../tests_for_coverage/",
        help="Directory of JSON data files (not needed in merge mode)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--merge_files", nargs="+",
        help="merge mode: list of result files to merge"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-record execution timeout in seconds (applies to coverage/mutation)"
    )
    parser.add_argument(
        "--candidate_path", default="../experiment_data/candidate_functions.json",
        help="Path to candidate_functions.json"
    )
    args = parser.parse_args()

    if args.metric == "merge":
        if not args.merge_files:
            print("[error] merge mode requires --merge_files")
            sys.exit(1)
        results = merge_results(args.merge_files)
        save(results, args.output)
        print_summary(results, "merge")
        sys.exit(0)

    # Load data
    records = load_all(args.data_dir, args.candidate_path)

    # Dispatch to the selected metric
    if args.metric == "complexity":
        results = metric_complexity(records)

    elif args.metric == "assertions":
        results = metric_assertions(records)

    elif args.metric == "coverage":
        results = metric_coverage(records, timeout=args.timeout)

    elif args.metric == "mutation":
        results = metric_mutation(records, timeout=args.timeout)

    elif args.metric == "all":
        print("\n[all] computing all metrics in sequence...")
        comp = metric_complexity(records)
        assr = metric_assertions(records)
        cov  = metric_coverage(records,  timeout=args.timeout)
        mut  = metric_mutation(records,   timeout=args.timeout * 3)

        # Merge four result sets in memory using unique_fid as key
        merged = {}
        for lst in [comp, assr, cov, mut]:
            for r in lst:
                unique_fid = r.get("unique_fid", r["function_id"])
                key = (unique_fid, r.get("strategy"))
                if key not in merged:
                    merged[key] = {k: v for k, v in r.items()}
                else:
                    merged[key].update(r)
        results = list(merged.values())

    save(results, args.output)
    print_summary(results, args.metric)
