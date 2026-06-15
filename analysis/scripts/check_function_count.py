import json
from collections import defaultdict

with open("../../results/metrics_results.json") as f:
    records = json.load(f)

counts = defaultdict(list)
for r in records:
    s = r.get("strategy")
    # 从complexity或assertions字段取test_func_count
    count = r.get("complexity", {}).get("test_func_count") or \
            r.get("assertions", {}).get("test_func_count")
    if count is not None:
        counts[s].append(count)

for s, vals in sorted(counts.items()):
    avg = sum(vals) / len(vals)
    print(f"{s}: avg test_func_count = {avg:.2f}  n={len(vals)}")