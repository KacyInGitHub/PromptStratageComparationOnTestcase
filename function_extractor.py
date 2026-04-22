"""
function_extractor.py
=====================
从三个实验项目中提取候选函数，应用3.2.2节定义的筛选规则。

使用方法：
    python function_extractor.py

输出：
    - candidate_functions.json   所有候选函数的详细信息
    - extraction_report.txt      筛选过程统计报告
"""

import ast
import json
import random
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 配置：修改为你本地的项目路径
# ─────────────────────────────────────────────
PROJECTS = {
    "requests": {
        "path": Path.home() / "PycharmProjects" / "requests" / "src",
        # 仅从这些模块采样（排除网络依赖模块）
        "include_modules": ["utils.py", "models.py", "structures.py"],
        "sample_size": 30,
    },
    "arrow": {
        "path": Path.home() / "PycharmProjects" / "arrow" / "arrow",
        "include_modules": None,   # None表示扫描全部模块
        "sample_size": 30,
    },
    "more_itertools": {
        "path": Path.home() / "PycharmProjects" / "more-itertools" / "more_itertools",
        "include_modules": None,
        "sample_size": 30,
    },
}

# 筛选规则（对应3.2.2节）
MIN_LINES = 5          # 最少可执行行数
MAX_LINES = 50         # 最多可执行行数
RANDOM_SEED = 42       # 随机种子，保证可重现

# 复杂度分层（使用radon，若未安装则跳过分层）
COMPLEXITY_TIERS = {
    "low":    (1, 3),
    "medium": (4, 7),
    "high":   (8, 999),
}


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class FunctionRecord:
    project: str
    module: str           # 相对路径，如 requests/utils.py
    name: str             # 函数名
    qualname: str         # 限定名，如 MyClass.my_method
    lineno: int           # 起始行号
    source: str           # 函数源代码
    lines: int            # 可执行行数
    complexity: int       # 圈复杂度（-1表示未计算）
    complexity_tier: str  # low / medium / high / unknown
    is_method: bool       # 是否为类方法
    docstring: Optional[str]  # 文档字符串


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def count_executable_lines(node: ast.FunctionDef) -> int:
    """统计函数中可执行语句行数（排除注释和空行）"""
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
    """使用radon计算圈复杂度，若未安装返回-1"""
    try:
        import radon.complexity as rc
        results = rc.cc_visit(source)
        for block in results:
            if block.name == func_name:
                return block.complexity
        return 1  # 无分支函数默认复杂度为1
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
    判断函数是否应被排除，返回 (是否排除, 排除原因)
    """
    name = node.name

    # 规则1：构造函数
    if name == "__init__":
        return True, "constructor"

    # 规则2：魔术方法（__str__, __repr__等）
    if name.startswith("__") and name.endswith("__"):
        return True, "dunder_method"

    # 规则3：私有方法（可选，视情况开启）
    # if name.startswith("_"):
    #     return True, "private_method"

    # 规则4：纯getter/setter（函数体只有return或赋值）
    body = node.body
    # 跳过docstring
    real_body = body[1:] if (body and isinstance(body[0], ast.Expr)
                             and isinstance(body[0].value, ast.Constant)) else body
    if len(real_body) == 1 and isinstance(real_body[0], ast.Return):
        return True, "pure_getter"

    # 规则5：行数过少
    exec_lines = count_executable_lines(node)
    if exec_lines < MIN_LINES:
        return True, f"too_short({exec_lines}lines)"

    # 规则6：行数过多
    # 通过起止行号估算总行数
    start = node.lineno - 1
    end = node.end_lineno
    total_lines = end - start
    if total_lines > MAX_LINES:
        return True, f"too_long({total_lines}lines)"

    return False, ""


def extract_source(source_lines: list[str], node: ast.FunctionDef) -> str:
    """提取函数源代码并去除公共缩进"""
    start = node.lineno - 1
    end = node.end_lineno
    raw = "".join(source_lines[start:end])
    return textwrap.dedent(raw)


def get_docstring(node: ast.FunctionDef) -> Optional[str]:
    """提取函数文档字符串"""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        return str(node.body[0].value.value)
    return None


# ─────────────────────────────────────────────
# 主提取逻辑
# ─────────────────────────────────────────────
def scan_file(
    filepath: Path,
    project_name: str,
    project_root: Path,
) -> tuple[list[FunctionRecord], dict]:
    """
    扫描单个Python文件，提取并筛选函数。
    返回：(通过筛选的函数列表, 统计信息字典)
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

    # 收集所有函数定义（包括类方法）
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        stats["total"] += 1
        is_method = False

        # 判断是否为类方法
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in ast.walk(parent):
                    is_method = True
                    break

        # 应用筛选规则
        # excluded, reason = is_excluded(node, source_lines)
        # if excluded:
        #     key = f"excluded_{reason.split('(')[0]}"
        #     if key in stats:
        #         stats[key] += 1
        #     continue

        # 将 reason 映射到正确的 stats key
        REASON_TO_KEY = {
            "constructor": "excluded_constructor",
            "dunder_method": "excluded_dunder",
            "pure_getter": "excluded_getter",  # 修复这里
        }

        excluded, reason = is_excluded(node, source_lines)
        if excluded:
            base = reason.split("(")[0]
            key = REASON_TO_KEY.get(base, f"excluded_{base}")
            if key in stats:
                stats[key] += 1
            continue



        # 提取源代码
        source = extract_source(source_lines, node)

        # 计算圈复杂度
        complexity = get_complexity(source, node.name)
        tier = get_tier(complexity)

        record = FunctionRecord(
            project=project_name,
            module=module_rel,
            name=node.name,
            qualname=node.name,  # 简化版，不追踪类名
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
    """扫描整个项目"""
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

    # 确定扫描范围
    if include_modules:
        # 仅扫描指定模块
        py_files = []
        for module_name in include_modules:
            matches = list(project_path.rglob(module_name))
            py_files.extend(matches)
    else:
        # 扫描全部Python文件，排除测试文件和配置文件
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
# 分层采样
# ─────────────────────────────────────────────
def stratified_sample(
    records: list[FunctionRecord],
    sample_size: int,
    seed: int = RANDOM_SEED,
) -> list[FunctionRecord]:
    """按复杂度分层采样"""
    random.seed(seed)

    # 按tier分组
    tiers: dict[str, list[FunctionRecord]] = {
        "low": [], "medium": [], "high": [], "unknown": []
    }
    for r in records:
        tiers[r.complexity_tier].append(r)

    print(f"\n  Complexity distribution before sampling:")
    for tier, funcs in tiers.items():
        print(f"    {tier:8s}: {len(funcs)} functions")

    # 如果没有复杂度信息，直接随机采样
    if all(len(v) == 0 for k, v in tiers.items() if k != "unknown"):
        return random.sample(records, min(sample_size, len(records)))

    # 按比例分配采样数量
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

        # 最后一个tier拿剩余所有配额
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

    # 如果采样不足，从剩余函数补充
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
# 报告生成
# ─────────────────────────────────────────────
def generate_report(
    all_results: dict,
    output_path: Path,
) -> None:
    """生成文本格式的筛选报告"""
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

        # 复杂度分布
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
# 主程序
# ─────────────────────────────────────────────
def main():
    output_dir = Path("experiment_data")
    output_dir.mkdir(exist_ok=True)

    all_results = {}
    all_sampled = []

    for project_name, config in PROJECTS.items():
        if not config["path"].exists():
            print(f"[WARNING] Project path not found: {config['path']}")
            print(f"  Please update the path in PROJECTS config.")
            continue

        # 扫描项目
        records, stats = scan_project(project_name, config)

        print(f"\n  Total passed: {stats['passed']} functions")

        # 分层采样
        sampled = stratified_sample(records, config["sample_size"])
        print(f"  Sampled: {len(sampled)} functions")

        all_results[project_name] = {
            "stats": stats,
            "all_records": records,
            "sampled": sampled,
        }
        all_sampled.extend(sampled)

    # 保存候选函数列表（JSON格式）
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

    # 生成报告
    generate_report(all_results, output_dir / "extraction_report.txt")

    # 打印样本预览
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
