import json
from collections import defaultdict
from math import comb

# ── Load files ──
with open("../../results/pipeline_results.json", encoding="utf-8") as f:
    records = json.load(f)

with open("../../experiment_data/candidate_functions.json", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total pipeline records: {len(records)}")

# ── Check for lineno and unique_fid fields ──
print("\n[Field check]")
sample = records[0]
print(f"  lineno field present:     {'lineno' in sample}")
print(f"  unique_fid field present: {'unique_fid' in sample}")
print(f"  sample unique_fid: {sample.get('unique_fid')}")
print(f"  sample lineno:     {sample.get('lineno')}")

# ── Check unmatched records (lineno=0) ──
print("\n[Unmatched record check]")
unmatched = [r for r in records if r.get("lineno") == 0]
if unmatched:
    print(f"WARNING: {len(unmatched)} unmatched records")
    for r in unmatched:
        print(f"   {r.get('unique_fid')}  "
              f"strategy={r.get('strategy')}  "
              f"trial={r.get('trial')}")
else:
    print("OK: no unmatched records; all functions matched via source hash")

# ── Check function count and trial distribution per strategy ──
print("\n[Function count and trial distribution by strategy]")
grouped = defaultdict(lambda: defaultdict(list))
for r in records:
    s   = r.get("strategy")
    fid = r.get("unique_fid")
    grouped[s][fid].append(r)

all_strategies_ok = True
for strategy in sorted(grouped.keys()):
    funcs = grouped[strategy]
    print(f"\n{'='*55}")
    print(f"Strategy: {strategy}")
    print(f"Unique functions: {len(funcs)}")

    # Check for functions that don't have exactly 3 trials
    abnormal = {fid: trials for fid, trials in funcs.items()
                if len(trials) != 3}
    if abnormal:
        all_strategies_ok = False
        print(f"WARNING: {len(abnormal)} functions without exactly 3 trials:")
        for fid, trials in sorted(abnormal.items()):
            print(f"   {fid}: n={len(trials)}")
    else:
        print("OK: all functions have exactly 3 trial records")

    # Count functions that failed all 3 trials
    all_fail = [
        fid for fid, trials in funcs.items()
        if all(
            (t.get("execution", {}).get("passed", 0) /
             max(t.get("execution", {}).get("total", 1), 1)) == 0
            for t in trials
        )
    ]
    print(f"All-fail functions: {len(all_fail)}")
    print(f"At least one pass:  {len(funcs) - len(all_fail)}")

# ── Cross-strategy consistency check ──
print(f"\n{'='*55}")
print("[Cross-strategy function set consistency]")
strategy_fids = {s: set(funcs.keys()) for s, funcs in grouped.items()}
strategies    = sorted(strategy_fids.keys())
reference     = strategy_fids[strategies[0]]
consistent    = True
for s in strategies[1:]:
    diff = reference.symmetric_difference(strategy_fids[s])
    if diff:
        consistent = False
        print(f"WARNING: {strategies[0]} vs {s} differ:")
        for fid in sorted(diff):
            print(f"   {fid}")
if consistent:
    print(f"OK: all strategies contain the same function set")
    n = len(reference)
    print(f"OK: total valid functions: {n}")
    if n == 90:
        print("OK: n=90 is correct")
    else:
        print(f"ERROR: n={n}; the thesis statement of n=90 needs correction")

# ── Verify lineno assignment for same-named functions ──
print(f"\n{'='*55}")
print("[Lineno assignment verification for same-named functions]")
name_to_fids = defaultdict(set)
for r in records:
    if r.get("strategy") == sorted(grouped.keys())[0] \
    and r.get("trial") == 1:
        name_to_fids[r.get("function_id")].add(r.get("unique_fid"))

has_duplicate = False
for base_fid, unique_fids in sorted(name_to_fids.items()):
    if len(unique_fids) > 1:
        has_duplicate = True
        print(f"OK (disambiguated) {base_fid}:")
        for ufid in sorted(unique_fids):
            print(f"   {ufid}")
if not has_duplicate:
    print("(no same-named functions, or all already disambiguated)")

# ── Recompute Pass@k ──
print(f"\n{'='*55}")
print("[Corrected Pass@k results]\n")

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

print(f"{'Strategy':<15} {'Funcs':>6} {'Pass@1(new)':>12} {'Pass@1(orig)':>12}"
      f" {'Pass@3(new)':>12} {'Pass@3(orig)':>12} {'All-fail':>8}")
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

# ── Save results ──
with open("../../results/passk_corrected.json", "w", encoding="utf-8") as f:
    json.dump(corrected, f, indent=2, ensure_ascii=False)
print("\nCorrected results saved to passk_corrected.json")