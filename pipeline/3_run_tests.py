"""
3_run_tests.py
==============
Automated test case processing pipeline.
Input:  generated_tests_*.json
Output: pipeline_results.json
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import hashlib
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# Import path fix mappings
# ─────────────────────────────────────────────
IMPORT_FIX = {
    "requests": {
        "requests/models.py": [
            "from requests.models import PreparedRequest, Request, Response",
            "from requests.exceptions import HTTPError, JSONDecodeError",
            "from requests.cookies import RequestsCookieJar",
            "from requests.auth import HTTPBasicAuth",
        ],
        "requests/utils.py": [
            "from requests.utils import ("
            "address_in_network, dotted_netmask, is_ipv4_address, "
            "is_valid_cidr, requote_uri, unquote_unreserved, "
            "parse_list_header, parse_dict_header, unquote_header_value, "
            "parse_header_links, iter_slices, from_key_val_list, "
            "to_key_val_list, get_encoding_from_headers, "
            "resolve_proxies, urldefragauth, _validate_header_part)",
            "from requests.exceptions import InvalidURL, InvalidHeader",
        ],
    },
    "arrow": {
        "arrow.py": [
            "import arrow",
            "from arrow.arrow import Arrow",
            "from datetime import datetime, timezone, timedelta",
        ],
        "locales.py": [
            "from arrow.locales import (",
            "    get_locale, get_locale_by_class_name,",
            "    EnglishLocale, FinnishLocale, GermanLocale,",
            "    TamilLocale, BengaliLocale,",
            ")",
        ],
        "parser.py": [
            "from arrow.parser import DateTimeParser, ParserError, ParserMatchError",
            "from arrow import parser",
        ],
        "util.py": [
            "from arrow.util import is_timestamp, normalize_timestamp",
        ],
    },
    "more_itertools": {
        "more.py": [
            "from more_itertools import *",
            "from more_itertools.more import *",
        ],
        "recipes.py": [
            "from more_itertools import *",
            "from more_itertools.recipes import *",
        ],
    },
}

# Placeholder replacement patterns
PLACEHOLDER_PATTERNS = [
    (r"from your_module import.*\n", ""),
    (r"from your_module import.*", ""),
    (r"import your_module.*\n", ""),
    (r"YourClass\(\)", "{real_class}()"),
    (r"# Replace.*\n", "\n"),
    (r"# Replace.*", ""),
]

# Real class name mapping
CLASS_MAP = {
    ("requests", "requests/models.py"): "PreparedRequest",
    ("requests", "requests/utils.py"): None,
    ("arrow", "arrow.py"): "Arrow",
    ("arrow", "locales.py"): "EnglishLocale",
    ("arrow", "parser.py"): "DateTimeParser",
    ("arrow", "util.py"): None,
    ("more_itertools", "more.py"): None,
    ("more_itertools", "recipes.py"): None,
}


def load_candidate_source_map(candidate_path: str) -> dict:
    """Build a (project, name, source_hash) -> lineno lookup table using source content as key."""
    with open(candidate_path, encoding="utf-8") as f:
        candidates = json.load(f)

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
# Filename parsing
# ─────────────────────────────────────────────
def parse_filename(filename: str) -> dict:
    pattern = r"generated_tests_(.+)_trial(\d+)\.json"
    match = re.search(pattern, filename)
    if match:
        return {
            "strategy": match.group(1),
            "trial": int(match.group(2)),
        }
    return {"strategy": "unknown", "trial": 0}


# ─────────────────────────────────────────────
# Load all JSON files
# ─────────────────────────────────────────────
# def load_all(data_dir: str) -> list:
#     all_records = []
#     for path in sorted(Path(data_dir).glob("generated_tests_*.json")):
#         meta = parse_filename(path.name)
#         with open(path, encoding="utf-8") as f:
#             records = json.load(f)
#         for record in records:
#             record.update(meta)
#             all_records.append(record)
#     print(f"[load] loaded {len(all_records)} records")
#     return all_records


def load_all(data_dir: str,
             candidate_path: str = "../experiment_data/candidate_functions.json") -> list:
    source_map = load_candidate_source_map(candidate_path)
    unmatched  = []

    all_records = []
    for path in sorted(Path(data_dir).glob("generated_tests_*.json")):
        meta = parse_filename(path.name)
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            record.update(meta)

            # Match lineno precisely via source hash
            project = record.get("project", "")
            name    = record.get("name", "")
            source  = record.get("source", "").strip()

            source_hash = hashlib.md5(
                source.encode("utf-8")
            ).hexdigest()
            key    = (project, name, source_hash)
            lineno = source_map.get(key)

            if lineno is not None:
                record["lineno"]     = lineno
                record["unique_fid"] = f"{project}.{name}.L{lineno}"
            else:
                record["lineno"]     = 0
                record["unique_fid"] = f"{project}.{name}.L0"
                unmatched.append({
                    "project": project,
                    "name":    name,
                    "file":    path.name,
                })

            all_records.append(record)

    if unmatched:
        print(f"[warn] {len(unmatched)} records could not be matched to a candidate function")
        for u in unmatched:
            print(f"       {u['project']}.{u['name']} in {u['file']}")
    else:
        print("[load] all records matched successfully to candidate functions")

    print(f"[load] loaded {len(all_records)} records total")
    return all_records


# ─────────────────────────────────────────────
# Step 1: Static analysis
# ─────────────────────────────────────────────
def static_check(source: str) -> dict:
    result = {
        "syntax_valid": False,
        "has_test_func": False,
        "test_count": 0,
        "has_assert": False,
        "has_placeholder": False,
    }

    # Check for placeholder imports
    if "your_module" in source.lower() or "yourclass" in source.lower():
        result["has_placeholder"] = True

    # Syntax check
    try:
        tree = ast.parse(source)
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["syntax_error"] = str(e)
        return result

    # Count test_ functions
    funcs = [
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    test_funcs = [f for f in funcs if f.startswith("test_")]
    result["has_test_func"] = len(test_funcs) > 0
    result["test_count"] = len(test_funcs)

    # Check for assert statements
    result["has_assert"] = any(
        isinstance(n, ast.Assert) for n in ast.walk(tree)
    )

    return result


# ─────────────────────────────────────────────
# Step 2: Import fixing
# ─────────────────────────────────────────────
def fix_imports(source: str, project: str, module: str, func_name: str) -> str:
    # Remove placeholder imports
    source = re.sub(
        r"^from your_module import.*$", "",
        source, flags=re.MULTILINE
    )
    source = re.sub(
        r"^import your_module.*$", "",
        source, flags=re.MULTILINE
    )

    # Replace YourClass with real class name
    real_class = CLASS_MAP.get((project, module))
    if real_class:
        source = source.replace("YourClass()", f"{real_class}()")
        source = source.replace("YourClass", real_class)

    # Remove replace-hint comments
    source = re.sub(r"#\s*Replace.*", "", source)

    # Inject real imports; insert after the last import line
    imports = IMPORT_FIX.get(project, {}).get(module, [])
    if imports:
        import_block = "\n".join(imports) + "\n"
        lines = source.split("\n")
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith("import") or line.startswith("from"):
                insert_pos = i + 1
        lines.insert(insert_pos, import_block)
        source = "\n".join(lines)

    return source


# ─────────────────────────────────────────────
# Step 3: Compilation check
# ─────────────────────────────────────────────
def compile_check(source: str) -> dict:
    import py_compile
    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(source)
        tmp_path = f.name
    try:
        py_compile.compile(tmp_path, doraise=True)
        return {"compile_ok": True}
    except py_compile.PyCompileError as e:
        return {"compile_ok": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Step 4: Execution
# ─────────────────────────────────────────────
def run_tests(source: str, timeout: int = 30) -> dict:
    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(source)
        tmp_path = f.name

    report_path = tmp_path + ".json"

    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", tmp_path,
                "--tb=short", "-q",
                f"--json-report",
                f"--json-report-file={report_path}",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as r:
                report = json.load(r)
            summary = report.get("summary", {})
            return {
                "passed":   summary.get("passed", 0),
                "failed":   summary.get("failed", 0),
                "error":    summary.get("error", 0),
                "total":    summary.get("total", 0),
                "duration": report.get("duration", 0),
                "timeout":  False,
            }
        else:
            return {
                "passed": 0, "failed": 0, "error": 0, "total": 0,
                "timeout": False,
                "stderr": proc.stderr[:500],
            }

    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "error": 0, "total": 0, "timeout": True}
    finally:
        os.unlink(tmp_path)
        if os.path.exists(report_path):
            os.unlink(report_path)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
# def process_record(record: dict) -> dict:
#     result = {
#         "function_id": f"{record['project']}.{record['name']}",
#         "project":     record["project"],
#         "module":      record["module"],
#         "name":        record["name"],
#         "strategy":    record.get("strategy"),
#         "trial":       record.get("trial"),
#         "status":      "pending",
#         "static":      {},
#         "compile":     {},
#         "execution":   {},
#     }
#
#     source = record.get("tests_source", "")
#     if not source.strip():
#         result["status"] = "empty"
#         return result
#
#     # Step 1: static analysis
#     result["static"] = static_check(source)
#     if not result["static"]["syntax_valid"]:
#         result["status"] = "syntax_error"
#         return result
#     if not result["static"]["has_test_func"]:
#         result["status"] = "no_test_func"
#         return result
#
#     # Step 2: fix imports
#     source = fix_imports(
#         source,
#         record["project"],
#         record["module"],
#         record["name"],
#     )
#
#     # Step 3: compile check
#     result["compile"] = compile_check(source)
#     if not result["compile"]["compile_ok"]:
#         result["status"] = "compile_error"
#         return result
#
#     # Step 4: execute
#     result["execution"] = run_tests(source)
#     result["status"] = "completed"
#     result["fixed_source"] = source
#
#     return result

def process_record(record: dict) -> dict:
    result = {
        "function_id": f"{record['project']}.{record['name']}",
        "unique_fid":  record.get("unique_fid",
                       f"{record['project']}.{record['name']}.L0"),
        "project":     record["project"],
        "module":      record["module"],
        "name":        record["name"],
        "lineno":      record.get("lineno", 0),
        "strategy":    record.get("strategy"),
        "trial":       record.get("trial"),
        "status":      "pending",
        "static":      {},
        "compile":     {},
        "execution":   {},
    }

    source = record.get("tests_source", "")
    if not source.strip():
        result["status"] = "empty"
        return result

    # Step 1: static analysis
    result["static"] = static_check(source)
    if not result["static"]["syntax_valid"]:
        result["status"] = "syntax_error"
        return result
    if not result["static"]["has_test_func"]:
        result["status"] = "no_test_func"
        return result

    # Step 2: fix imports
    source = fix_imports(
        source,
        record["project"],
        record["module"],
        record["name"],
    )

    # Step 3: compile check
    result["compile"] = compile_check(source)
    if not result["compile"]["compile_ok"]:
        result["status"] = "compile_error"
        return result

    # Step 4: execute
    result["execution"] = run_tests(source)
    result["status"]    = "completed"
    result["fixed_source"] = source

    return result


# def run_pipeline(data_dir: str, output_path: str):
#     records = load_all(data_dir)
#     results = []
#
#     for i, record in enumerate(records):
#         print(
#             f"[{i+1:03d}/{len(records)}] "
#             f"{record.get('strategy')}/trial{record.get('trial')} "
#             f"- {record['project']}.{record['name']}",
#             end=" ... "
#         )
#         result = process_record(record)
#         results.append(result)
#         print(result["status"])
#
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     print_summary_all(results)
#     print(f"\nResults saved to: {output_path}")

def run_pipeline(data_dir: str, output_path: str,
                 candidate_path: str = "../experiment_data/candidate_functions.json"):
    records = load_all(data_dir, candidate_path)
    results = []

    for i, record in enumerate(records):
        print(
            f"[{i+1:03d}/{len(records)}] "
            f"{record.get('strategy')}/trial{record.get('trial')} "
            f"- {record['project']}.{record['name']} "
            f"(L{record.get('lineno', '?')})",
            end=" ... "
        )
        result = process_record(record)
        results.append(result)
        print(result["status"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary_all(results)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: list):
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)

    from collections import Counter
    status_counter = Counter(r["status"] for r in results)
    for status, count in sorted(status_counter.items()):
        print(f"  {status:20s}: {count}")

    print()
    # Pass rate by strategy (at least 1 test passed)
    by_strategy = defaultdict(lambda: {"total": 0, "passed_any": 0})
    for r in results:
        s = r.get("strategy", "unknown")
        by_strategy[s]["total"] += 1
        exec_info = r.get("execution", {})
        if exec_info.get("passed", 0) > 0:
            by_strategy[s]["passed_any"] += 1

    print("By strategy (at least 1 test passed):")
    for strategy, counts in sorted(by_strategy.items()):
        rate = counts["passed_any"] / counts["total"] * 100 if counts["total"] else 0
        print(f"  {strategy:20s}: {counts['passed_any']}/{counts['total']} ({rate:.1f}%)")

def print_summary_all(results: list):
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    # Overall status distribution
    from collections import Counter, defaultdict
    status_counter = Counter(r["status"] for r in results)
    print("\n[Overall status distribution]")
    for status, count in sorted(status_counter.items()):
        print(f"  {status:20s}: {count}")

    # ─────────────────────────────────────────────
    # Pass rates by strategy × trial
    # ─────────────────────────────────────────────
    print("\n[By strategy × Trial]")
    print(f"\n{'Strategy':<20} {'Trial':<8} {'Syntax OK':>10} {'Compile OK':>10} {'Exec passed':>12} {'Total':>6}")
    print("-" * 70)

    grouped = defaultdict(lambda: {
        "total": 0,
        "syntax_ok": 0,
        "compile_ok": 0,
        "exec_passed": 0,
    })

    for r in results:
        key = (r.get("strategy", "unknown"), r.get("trial", 0))
        g = grouped[key]
        g["total"] += 1

        # Syntax OK: syntax_valid = True and has_test_func = True
        if (r.get("static", {}).get("syntax_valid") and
                r.get("static", {}).get("has_test_func")):
            g["syntax_ok"] += 1

        # Compile OK
        if r.get("compile", {}).get("compile_ok"):
            g["compile_ok"] += 1

        # Exec passed: at least 1 test passed
        if r.get("execution", {}).get("passed", 0) > 0:
            g["exec_passed"] += 1

    # Print sorted by strategy and trial
    for (strategy, trial), g in sorted(grouped.items()):
        n = g["total"]
        def pct(x): return f"{x}/{n} ({x/n*100:.1f}%)" if n else "0/0"
        print(
            f"  {strategy:<18} {f'Trial {trial}':<8}"
            f" {pct(g['syntax_ok']):>18}"
            f" {pct(g['compile_ok']):>18}"
            f" {pct(g['exec_passed']):>18}"
            f" {n:>6}"
        )

    # ─────────────────────────────────────────────
    # Totals by strategy (across all trials)
    # ─────────────────────────────────────────────
    print("\n[Totals by strategy (all trials combined)]")
    print(f"\n{'Strategy':<20} {'Syntax OK':>18} {'Compile OK':>18} {'Exec passed':>18} {'Total':>6}")
    print("-" * 70)

    strategy_total = defaultdict(lambda: {
        "total": 0, "syntax_ok": 0, "compile_ok": 0, "exec_passed": 0
    })
    for (strategy, trial), g in grouped.items():
        s = strategy_total[strategy]
        s["total"]       += g["total"]
        s["syntax_ok"]   += g["syntax_ok"]
        s["compile_ok"]  += g["compile_ok"]
        s["exec_passed"] += g["exec_passed"]

    for strategy, s in sorted(strategy_total.items()):
        n = s["total"]
        def pct(x): return f"{x}/{n} ({x/n*100:.1f}%)" if n else "0/0"
        print(
            f"  {strategy:<18}"
            f" {pct(s['syntax_ok']):>18}"
            f" {pct(s['compile_ok']):>18}"
            f" {pct(s['exec_passed']):>18}"
            f" {n:>6}"
        )

    # ─────────────────────────────────────────────
    # Exec pass rate by project
    # ─────────────────────────────────────────────
    print("\n[Exec pass rate by project]")
    print(f"\n{'Project':<20} {'Exec passed':>18} {'Total':>6}")
    print("-" * 45)

    proj_total = defaultdict(lambda: {"total": 0, "exec_passed": 0})
    for r in results:
        p = proj_total[r.get("project", "unknown")]
        p["total"] += 1
        if r.get("execution", {}).get("passed", 0) > 0:
            p["exec_passed"] += 1

    for project, p in sorted(proj_total.items()):
        n = p["total"]
        def pct(x): return f"{x}/{n} ({x/n*100:.1f}%)" if n else "0/0"
        print(f"  {project:<18} {pct(p['exec_passed']):>18} {n:>6}")

# if __name__ == "__main__":
#     DATA_DIR = "generated_tests/"
#     OUTPUT   = "pipeline_results.json"
#     run_pipeline(DATA_DIR, OUTPUT)
if __name__ == "__main__":
    DATA_DIR       = "../generated_tests/"
    OUTPUT         = "../results/pipeline_results.json"
    CANDIDATE_PATH = "../experiment_data/candidate_functions.json"
    run_pipeline(DATA_DIR, OUTPUT, CANDIDATE_PATH)