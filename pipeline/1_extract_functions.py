"""
1_extract_functions.py
Usage:
    python 1_extract_functions.py

Output:
    - candidate_functions.json   Detailed information for all candidate functions
    - extraction_report.txt      Filtering process statistics report
"""

import ast
import json
import random
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Configuration: update to your local project paths
# ─────────────────────────────────────────────
PROJECTS = {
    "requests": {
        "path": Path.home() / "PycharmProjects" / "requests" / "src",
        # Sample only from these modules (excludes network-dependent modules)
        "include_modules": ["utils.py", "models.py", "structures.py"],
        "sample_size": 30,
    },
    "arrow": {
        "path": Path.home() / "PycharmProjects" / "arrow" / "arrow",
        "include_modules": None,   # None means scan all modules
        "sample_size": 30,
    },
    "more_itertools": {
        "path": Path.home() / "PycharmProjects" / "more-itertools" / "more_itertools",
        "include_modules": None,
        "sample_size": 30,
    },
}

# Filtering rules (see section 3.2.2)
MIN_LINES = 5          # Minimum executable lines
MAX_LINES = 50         # Maximum executable lines
RANDOM_SEED = 42       # Random seed for reproducibility

# Complexity tiers (uses radon; skips tiering if not installed)
COMPLEXITY_TIERS = {
    "low":    (1, 3),
    "medium": (4, 7),
    "high":   (8, 999),
}


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class FunctionRecord:
    project: str
    module: str           # Relative path, e.g. requests/utils.py
    name: str             # Function name
    qualname: str         # Qualified name, e.g. MyClass.my_method
    lineno: int           # Starting line number
    source: str           # Function source code
    lines: int            # Executable line count
    complexity: int       # Cyclomatic complexity (-1 if not computed)
    complexity_tier: str  # low / medium / high / unknown
    is_method: bool       # Whether it is a class method
    docstring: Optional[str]  # Docstring


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def count_executable_lines(node: ast.FunctionDef) -> int:
    """Count executable statement lines in a function (excludes comments and blank lines)."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (
            ast.Assign, ast.AugAssign, ast.AnnAssign,
            ast.Return, ast.Raise, ast.Assert,
            ast.Expr, ast.If, ast.For, ast.While,
            ast.With, ast.Try, ast.Delete,
            ast.Import, ast.ImportFrom,
        )):
            count += 1
    return count


def get_complexity(source: str, func_name: str) -> int:
    """Compute cyclomatic complexity via radon; returns -1 if radon is not installed."""
    try:
        import radon.complexity as rc
        results = rc.cc_visit(source)
        for block in results:
            if block.name == func_name:
                return block.complexity
        return 1  # Default complexity for branch-free functions
    except ImportError:
        return -1


def get_tier(complexity: int) -> str:
    if complexity == -1:
        return "unknown"
    for tier, (lo, hi) in COMPLEXITY_TIERS.items():
        if lo <= complexity <= hi:
            return tier
    return "high"


def is_excluded(node: ast.FunctionDef, source_lines: list[str]) -> tuple[bool, str]:
    """
    Determine whether a function should be excluded.
    Returns (should_exclude, reason).
    """
    name = node.name

    # Rule 1: constructors
    if name == "__init__":
        return True, "constructor"

    # Rule 2: dunder methods (__str__, __repr__, etc.)
    if name.startswith("__") and name.endswith("__"):
        return True, "dunder_method"

    # Rule 3: private methods (optional, enable if needed)
    # if name.startswith("_"):
    #     return True, "private_method"

    # Rule 4: pure getter/setter (body contains only a return or assignment)
    body = node.body
    # Skip docstring
    real_body = body[1:] if (body and isinstance(body[0], ast.Expr)
                             and isinstance(body[0].value, ast.Constant)) else body
    if len(real_body) == 1 and isinstance(real_body[0], ast.Return):
        return True, "pure_getter"

    # Rule 5: too few lines
    exec_lines = count_executable_lines(node)
    if exec_lines < MIN_LINES:
        return True, f"too_short({exec_lines}lines)"

    # Rule 6: too many lines (estimated from start/end line numbers)
    start = node.lineno - 1
    end = node.end_lineno
    total_lines = end - start
    if total_lines > MAX_LINES:
        return True, f"too_long({total_lines}lines)"

    return False, ""


def extract_source(source_lines: list[str], node: ast.FunctionDef) -> str:
    """Extract function source code and strip common indentation."""
    start = node.lineno - 1
    end = node.end_lineno
    raw = "".join(source_lines[start:end])
    return textwrap.dedent(raw)


def get_docstring(node: ast.FunctionDef) -> Optional[str]:
    """Extract the function's docstring."""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        return str(node.body[0].value.value)
    return None


# ─────────────────────────────────────────────
# Core extraction logic
# ─────────────────────────────────────────────
def scan_file(
    filepath: Path,
    project_name: str,
    project_root: Path,
) -> tuple[list[FunctionRecord], dict]:
    """
    Scan a single Python file and extract filtered functions.
    Returns (list of functions that passed filtering, stats dict).
    """
    stats = {
        "total": 0,
        "excluded_constructor": 0,
        "excluded_dunder": 0,
        "excluded_getter": 0,
        "excluded_too_short": 0,
        "excluded_too_long": 0,
        "passed": 0,
    }

    try:
        source_text = filepath.read_text(encoding="utf-8")
        source_lines = source_text.splitlines(keepends=True)
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  [SKIP] {filepath.name}: {e}")
        return [], stats

    module_rel = str(filepath.relative_to(project_root))
    records = []

    # Collect all function definitions (including class methods)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        stats["total"] += 1
        is_method = False

        # Determine whether the function is a class method
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in ast.walk(parent):
                    is_method = True
                    break

        # Apply filtering rules
        # excluded, reason = is_excluded(node, source_lines)
        # if excluded:
        #     key = f"excluded_{reason.split('(')[0]}"
        #     if key in stats:
        #         stats[key] += 1
        #     continue

        # Map reason string to the correct stats key
        REASON_TO_KEY = {
            "constructor": "excluded_constructor",
            "dunder_method": "excluded_dunder",
            "pure_getter": "excluded_getter",
        }

        excluded, reason = is_excluded(node, source_lines)
        if excluded:
            base = reason.split("(")[0]
            key = REASON_TO_KEY.get(base, f"excluded_{base}")
            if key in stats:
                stats[key] += 1
            continue



        # Extract source code
        source = extract_source(source_lines, node)

        # Compute cyclomatic complexity
        complexity = get_complexity(source, node.name)
        tier = get_tier(complexity)

        record = FunctionRecord(
            project=project_name,
            module=module_rel,
            name=node.name,
            qualname=node.name,  # Simplified: does not track class name
            lineno=node.lineno,
            source=source,
            lines=count_executable_lines(node),
            complexity=complexity,
            complexity_tier=tier,
            is_method=is_method,
            docstring=get_docstring(node),
        )
        records.append(record)
        stats["passed"] += 1

    return records, stats


def scan_project(
    project_name: str,
    config: dict,
) -> tuple[list[FunctionRecord], dict]:
    """Scan an entire project."""
    project_path = config["path"]
    include_modules = config["include_modules"]

    all_records = []
    total_stats = {
        "files_scanned": 0,
        "total": 0,
        "excluded_constructor": 0,
        "excluded_dunder": 0,
        "excluded_getter": 0,
        "excluded_too_short": 0,
        "excluded_too_long": 0,
        "passed": 0,
    }

    # Determine scan scope
    if include_modules:
        # Scan only the specified modules
        py_files = []
        for module_name in include_modules:
            matches = list(project_path.rglob(module_name))
            py_files.extend(matches)
    else:
        # Scan all Python files, excluding test and config files
        py_files = [
            f for f in project_path.rglob("*.py")
            if not any(part.startswith("test") for part in f.parts)
            and "setup.py" not in f.name
            and "conf.py" not in f.name
            and "__pycache__" not in str(f)
            and ".egg" not in str(f)
            and "venv" not in str(f)
            and ".venv" not in str(f)
        ]

    print(f"\n{'='*50}")
    print(f"Project: {project_name}")
    print(f"Files to scan: {len(py_files)}")
    print(f"{'='*50}")

    for filepath in sorted(py_files):
        records, stats = scan_file(filepath, project_name, project_path)
        all_records.extend(records)
        total_stats["files_scanned"] += 1
        for key in stats:
            if key in total_stats:
                total_stats[key] += stats[key]

        if records:
            print(f"  {filepath.name}: {stats['passed']} functions passed")

    return all_records, total_stats


# ─────────────────────────────────────────────
# Stratified sampling
# ─────────────────────────────────────────────
def stratified_sample(
    records: list[FunctionRecord],
    sample_size: int,
    seed: int = RANDOM_SEED,
) -> list[FunctionRecord]:
    """Sample functions with stratification by complexity tier."""
    random.seed(seed)

    # Group by tier
    tiers: dict[str, list[FunctionRecord]] = {
        "low": [], "medium": [], "high": [], "unknown": []
    }
    for r in records:
        tiers[r.complexity_tier].append(r)

    print(f"\n  Complexity distribution before sampling:")
    for tier, funcs in tiers.items():
        print(f"    {tier:8s}: {len(funcs)} functions")

    # Fall back to random sampling if no complexity info is available
    if all(len(v) == 0 for k, v in tiers.items() if k != "unknown"):
        return random.sample(records, min(sample_size, len(records)))

    # Allocate sample quota proportionally
    total = len(records)
    if total == 0:
        return []

    sampled = []
    remaining = sample_size

    tier_order = ["low", "medium", "high", "unknown"]
    for i, tier in enumerate(tier_order):
        funcs = tiers[tier]
        if not funcs:
            continue

        # Last tier gets all remaining quota
        if i == len(tier_order) - 1:
            n = remaining
        else:
            proportion = len(funcs) / total
            n = round(sample_size * proportion)

        n = min(n, len(funcs), remaining)
        sampled.extend(random.sample(funcs, n))
        remaining -= n

        if remaining <= 0:
            break

    # Top up with remaining functions if sample is still short
    if len(sampled) < sample_size:
        sampled_set = set(id(r) for r in sampled)
        remaining_pool = [r for r in records if id(r) not in sampled_set]
        extra = random.sample(
            remaining_pool,
            min(sample_size - len(sampled), len(remaining_pool))
        )
        sampled.extend(extra)

    return sampled[:sample_size]


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────
def generate_report(
    all_results: dict,
    output_path: Path,
) -> None:
    """Generate a plain-text filtering report."""
    lines = []
    lines.append("=" * 60)
    lines.append("FUNCTION EXTRACTION REPORT")
    lines.append("=" * 60)

    total_passed = 0
    total_sampled = 0

    for project_name, data in all_results.items():
        stats = data["stats"]
        sampled = data["sampled"]
        lines.append(f"\nProject: {project_name}")
        lines.append(f"  Files scanned:        {stats['files_scanned']}")
        lines.append(f"  Total functions:      {stats['total']}")
        lines.append(f"  Excluded (constructor):{stats['excluded_constructor']}")
        lines.append(f"  Excluded (dunder):    {stats['excluded_dunder']}")
        lines.append(f"  Excluded (getter):    {stats['excluded_getter']}")
        lines.append(f"  Excluded (too short): {stats['excluded_too_short']}")
        lines.append(f"  Excluded (too long):  {stats['excluded_too_long']}")
        lines.append(f"  Passed filtering:     {stats['passed']}")
        lines.append(f"  Final sample:         {len(sampled)}")

        # Complexity distribution
        tier_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
        for r in sampled:
            tier_counts[r.complexity_tier] += 1
        lines.append(f"  Complexity (sampled): "
                     f"low={tier_counts['low']}, "
                     f"medium={tier_counts['medium']}, "
                     f"high={tier_counts['high']}")

        total_passed += stats["passed"]
        total_sampled += len(sampled)

    lines.append(f"\n{'='*60}")
    lines.append(f"TOTAL: {total_sampled} functions sampled for experiment")
    lines.append(f"Total generation tasks: {total_sampled * 4} (×4 strategies)")
    lines.append(f"Total API calls: {total_sampled * 4 * 3} (×3 trials)")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    output_dir = Path("../experiment_data")
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    all_sampled = []

    for project_name, config in PROJECTS.items():
        if not config["path"].exists():
            print(f"[WARNING] Project path not found: {config['path']}")
            print(f"  Please update the path in PROJECTS config.")
            continue

        # Scan project
        records, stats = scan_project(project_name, config)

        print(f"\n  Total passed: {stats['passed']} functions")

        # Stratified sampling
        sampled = stratified_sample(records, config["sample_size"])
        print(f"  Sampled: {len(sampled)} functions")

        all_results[project_name] = {
            "stats": stats,
            "all_records": records,
            "sampled": sampled,
        }
        all_sampled.extend(sampled)

    # Save candidate function list (JSON format)
    output_json = output_dir / "candidate_functions.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in all_sampled],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nCandidate functions saved to: {output_json}")
    print(f"Total functions: {len(all_sampled)}")

    # Generate report
    generate_report(all_results, output_dir / "extraction_report.txt")

    # Print sample preview
    print("\n" + "=" * 50)
    print("SAMPLE PREVIEW (first 3 functions)")
    print("=" * 50)
    for r in all_sampled[:3]:
        print(f"\n[{r.project}] {r.module} :: {r.name}")
        print(f"  Lines: {r.lines}, Complexity: {r.complexity} ({r.complexity_tier})")
        print(f"  Source preview:")
        preview = r.source[:200] + "..." if len(r.source) > 200 else r.source
        for line in preview.split("\n")[:5]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
