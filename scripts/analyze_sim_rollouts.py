#!/usr/bin/env python3
"""Per-condition and paired analysis of sim rollouts (stdlib only; Python >= 3.8).

    python3 scripts/analyze_sim_rollouts.py runs/sim_eval/<run>                    # a sim/eval.py run dir
    python3 scripts/analyze_sim_rollouts.py experiments/E001_act_orange_2cam_b8/metrics/rollouts_sim.jsonl \
        --ref ck70k_h100

Reports, per condition (n, Wilson 95% CI everywhere):
  success          the record's own `success` (protocol_v1 if `success_def` present, else LeIsaac task_done)
  success_leisaac  LeIsaac task_done (= `success` for pre-ADR-0004 records)
  all3_end         all oranges inside the plate box at the last check (= LeIsaac minus the rest-pose clause;
                   no release check). For pre-ADR-0004 records the closest available proxy to protocol_v1.
  max_placed       mean (bootstrap 95% CI) and 0/1/2/3 histogram
Paired: same `trial` index = same initial layout (one env reset per episode, fixed seed; verified for new
records through `init_layout`). Differences vs --ref: mean diff of max_placed (bootstrap CI), exact sign
test, and an exact McNemar test on all3_end / success.
Aggregates are printed, never written (docs/05).
"""

from __future__ import annotations

import sys as _sys

_sys.dont_write_bytecode = True  # repo rule: never create __pycache__ (CLAUDE.md §3.9)

import argparse
import collections
import json
import math
import random
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def boot_mean(xs, rng, b=4000):
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(b))
    return m[int(0.025 * b)], m[int(0.975 * b)]


def exact_two_sided(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path: Path):
    if path.is_dir():
        plan = json.loads((path / "plan.json").read_text())
        order = [c.get("name") or f"ck{c['ckpt'] // 1000}k_h{c['horizon']}" for c in plan["conditions"]]
        recs = []
        for f in sorted((path / "rollouts").glob("*.jsonl")):
            recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    else:
        recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        order = list(dict.fromkeys(r["condition"] for r in recs))
    by = collections.defaultdict(dict)
    for r in recs:
        by[r["condition"]][r["trial"]] = r
    return order, by


def all3_end(r):
    if r.get("success_leisaac") or (r.get("success") and "success_def" not in r):
        return True
    return r.get("n_objects_placed") == 3


def leisaac(r):
    return bool(r.get("success_leisaac", r.get("success") if "success_def" not in r else False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--ref", help="reference condition for paired tests (default: first)")
    a = ap.parse_args()
    order, by = load(a.src)
    rng = random.Random(0)
    ref = a.ref or order[0]
    print(f"{'cond':<14} {'n':>3}  {'success':<22} {'success_leisaac':<22} {'all3_end':<22} "
          f"{'max_placed mean [boot95]':<26} hist0123")
    for c in order:
        rs = list(by[c].values())
        n = len(rs)
        cols = []
        for f in (lambda r: bool(r.get("success")), leisaac, all3_end):
            k = sum(f(r) for r in rs)
            lo, hi = wilson(k, n)
            cols.append(f"{k:>2}/{n} [{lo:.2f},{hi:.2f}]")
        pl = [r.get("max_objects_placed") or 0 for r in rs]
        lo, hi = boot_mean(pl, rng)
        hist = [collections.Counter(pl).get(i, 0) for i in range(4)]
        print(f"{c:<14} {n:>3}  {cols[0]:<22} {cols[1]:<22} {cols[2]:<22} "
              f"{sum(pl) / max(n, 1):.2f} [{lo:.2f},{hi:.2f}]{'':<10} {hist}")
    # layout pairing check (records that carry init_layout)
    lay = collections.defaultdict(set)
    for c in order:
        for t, r in by[c].items():
            if r.get("init_layout"):
                lay[t].add(json.dumps(r["init_layout"]["oranges"]))
    if lay:
        bad = [t for t, s in lay.items() if len(s) > 1]
        print(f"\nlayout pairing: {len(lay) - len(bad)}/{len(lay)} trials identical across conditions"
              + (f"  ⚠️ mismatched trials {sorted(bad)}" if bad else ""))
    print(f"\npaired vs {ref} (same trial index):")
    for c in order:
        if c == ref:
            continue
        ts = sorted(set(by[c]) & set(by[ref]))
        d = [(by[c][t].get("max_objects_placed") or 0) - (by[ref][t].get("max_objects_placed") or 0) for t in ts]
        lo, hi = boot_mean(d, rng)
        pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
        line = (f"  {c:<14} max_placed diff {sum(d) / max(len(d), 1):+.2f} [{lo:+.2f},{hi:+.2f}] "
                f"better/worse/tie {pos}/{neg}/{len(d) - pos - neg} sign p={exact_two_sided(pos, neg):.3f}")
        for nm, f in (("success", lambda r: bool(r.get("success"))), ("all3_end", all3_end)):
            b = sum(f(by[c][t]) and not f(by[ref][t]) for t in ts)
            w = sum(f(by[ref][t]) and not f(by[c][t]) for t in ts)
            line += f" | {nm} +{b}/-{w} McNemar p={exact_two_sided(b, w):.3f}"
        print(line)


if __name__ == "__main__":
    main()
