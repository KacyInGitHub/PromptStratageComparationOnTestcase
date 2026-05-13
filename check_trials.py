import json
from collections import defaultdict
from math import comb

# ── 加载文件 ──
with open("pipeline_results.json", encoding="utf-8") as f:
    records = json.load(f)

with open("experiment_data/candidate_functions.json", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"pipeline记录总数: {len(records)}")

# ── 检查lineno和unique_fid字段是否存在 ──
print("\n【字段检查】")
sample = records[0]
print(f"  lineno字段存在:     {'lineno' in sample}")
print(f"  unique_fid字段存在: {'unique_fid' in sample}")
print(f"  样本unique_fid: {sample.get('unique_fid')}")
print(f"  样本lineno:     {sample.get('lineno')}")

# ── 检查未匹配记录（lineno=0）──
print("\n【未匹配记录检查】")
unmatched = [r for r in records if r.get("lineno") == 0]
if unmatched:
    print(f"⚠️  未匹配记录数: {len(unmatched)}")
    for r in unmatched:
        print(f"   {r.get('unique_fid')}  "
              f"strategy={r.get('strategy')}  "
              f"trial={r.get('trial')}")
else:
    print("✅ 无未匹配记录，所有函数均成功通过源码匹配")

# ── 按策略检查函数数量和trial分布 ──
print("\n【各策略函数数量与trial分布】")
grouped = defaultdict(lambda: defaultdict(list))
for r in records:
    s   = r.get("strategy")
    fid = r.get("unique_fid")
    grouped[s][fid].append(r)

all_strategies_ok = True
for strategy in sorted(grouped.keys()):
    funcs = grouped[strategy]
    print(f"\n{'='*55}")
    print(f"策略: {strategy}")
    print(f"唯一函数数: {len(funcs)}")

    # trial数量异常检查
    abnormal = {fid: trials for fid, trials in funcs.items()
                if len(trials) != 3}
    if abnormal:
        all_strategies_ok = False
        print(f"⚠️  trial数量不等于3的函数（{len(abnormal)}个）:")
        for fid, trials in sorted(abnormal.items()):
            print(f"   {fid}: n={len(trials)}")
    else:
        print("✅ 所有函数均有3条trial记录")

    # 三次全败统计
    all_fail = [
        fid for fid, trials in funcs.items()
        if all(
            (t.get("execution", {}).get("passed", 0) /
             max(t.get("execution", {}).get("total", 1), 1)) == 0
            for t in trials
        )
    ]
    print(f"三次全败函数数: {len(all_fail)}")
    print(f"至少一次成功:   {len(funcs) - len(all_fail)}")

# ── 跨策略一致性检查 ──
print(f"\n{'='*55}")
print("【跨策略函数集合一致性】")
strategy_fids = {s: set(funcs.keys()) for s, funcs in grouped.items()}
strategies    = sorted(strategy_fids.keys())
reference     = strategy_fids[strategies[0]]
consistent    = True
for s in strategies[1:]:
    diff = reference.symmetric_difference(strategy_fids[s])
    if diff:
        consistent = False
        print(f"⚠️  {strategies[0]} vs {s} 存在差异:")
        for fid in sorted(diff):
            print(f"   {fid}")
if consistent:
    print(f"✅ 所有策略包含相同的函数集合")
    n = len(reference)
    print(f"✅ 有效函数总数: {n}")
    if n == 90:
        print("✅ n=90 正确")
    else:
        print(f"❌ n={n}，论文中n=90的表述需要修正")

# ── 验证同名函数的lineno分配 ──
print(f"\n{'='*55}")
print("【同名函数lineno分配验证】")
name_to_fids = defaultdict(set)
for r in records:
    if r.get("strategy") == sorted(grouped.keys())[0] \
    and r.get("trial") == 1:
        name_to_fids[r.get("function_id")].add(r.get("unique_fid"))

has_duplicate = False
for base_fid, unique_fids in sorted(name_to_fids.items()):
    if len(unique_fids) > 1:
        has_duplicate = True
        print(f"✅ {base_fid}:")
        for ufid in sorted(unique_fids):
            print(f"   {ufid}")
if not has_duplicate:
    print("（无同名函数，或同名函数已正确区分）")

# ── 重新计算Pass@k ──
print(f"\n{'='*55}")
print("【修正后Pass@k计算结果】\n")

def pass_at_k(n, c, k):
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

paper_values = {
    "CoT":        {"pass_at_1": 68.15, "pass_at_3": 96.85},
    "few_shot":   {"pass_at_1": 65.56, "pass_at_3": 96.00},
    "role_based": {"pass_at_1": 59.26, "pass_at_3": 93.35},
    "zero_shot":  {"pass_at_1": 59.26, "pass_at_3": 93.35},
}

print(f"{'策略':<15} {'函数数':>6} {'Pass@1(新)':>12} {'Pass@1(原)':>12}"
      f" {'Pass@3(新)':>12} {'Pass@3(原)':>12} {'全败数':>6}")
print("-" * 80)

corrected = {}
for s in sorted(grouped.keys()):
    funcs = grouped[s]
    pass1_list, pass3_list = [], []
    all_fail_count = 0

    for fid, trials in funcs.items():
        n = len(trials)
        c = sum(
            1 if (t.get("execution", {}).get("passed", 0) /
                  max(t.get("execution", {}).get("total", 1), 1)) > 0
            else 0
            for t in trials
        )
        if c == 0:
            all_fail_count += 1
        pass1_list.append(pass_at_k(n, c, 1))
        pass3_list.append(pass_at_k(n, c, 3))

    pass1 = sum(pass1_list) / len(pass1_list) * 100 if pass1_list else 0
    pass3 = sum(pass3_list) / len(pass3_list) * 100 if pass3_list else 0
    corrected[s] = {"pass_at_1": pass1, "pass_at_3": pass3}

    orig  = paper_values.get(s, {})
    print(f"{s:<15} {len(funcs):>6} {pass1:>11.2f}%"
          f" {orig.get('pass_at_1', 0):>11.2f}%"
          f" {pass3:>11.2f}%"
          f" {orig.get('pass_at_3', 0):>11.2f}%"
          f" {all_fail_count:>6}")

# ── 保存结果 ──
with open("passk_corrected.json", "w", encoding="utf-8") as f:
    json.dump(corrected, f, indent=2, ensure_ascii=False)
print("\n修正后结果已保存至 passk_corrected.json")