#!/usr/bin/env python
"""experiments/<ID>/metrics/*.jsonl -> experiments/<ID>/figures/*.{png,pdf}; optionally refresh README table.

    python scripts/make_figures.py E001_act_orange_2cam_b8
    python scripts/make_figures.py --all
    python scripts/make_figures.py --update-readme
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from aiku_hands import REPO_ROOT, registry
from aiku_hands.logging.jsonl import read_jsonl
from aiku_hands.viz import plots, style

BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"


def figures_for(exp_dir: Path) -> list[Path]:
    m = exp_dir / "metrics"
    exp_id = exp_dir.name.split("_")[0]
    out: list[Path] = []
    tj = m / "train.jsonl"
    if tj.exists():
        df = pd.DataFrame(list(read_jsonl(tj)))
        if not df.empty:
            df = df.sort_values("step")
            out += style.save(plots.loss_curves(df, exp_id), exp_dir / "figures" / "loss")
            out += style.save(plots.grad_lr(df, exp_id), exp_dir / "figures" / "grad_norm_lr")
            out += style.save(plots.throughput(df, exp_id), exp_dir / "figures" / "throughput")
    sj = m / "system.jsonl"
    if sj.exists():
        df = pd.DataFrame(list(read_jsonl(sj)))
        if len(df) > 1:
            out += style.save(plots.system(df, exp_id), exp_dir / "figures" / "system")
    return out


def results_table() -> str:
    rows = registry.read()
    head = "| ID | date | status | model | dataset | steps | batch | cams | best metric | report |\n|---|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for r in rows:
        rep = f"[README]({r['report']})" if r.get("report") else ""
        lines.append(f"| {r['id']} | {r['date']} | {r['status']} | {r['model']} | {r['dataset']} | "
                     f"{r['steps']} | {r['batch']} | {r['cams']} | {r['best_metric']} | {rep} |")
    return "\n".join(lines)


def update_readme() -> None:
    p = REPO_ROOT / "README.md"
    s = p.read_text()
    if BEGIN not in s or END not in s:
        raise SystemExit("README.md has no results markers")
    pre, rest = s.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    p.write_text(f"{pre}{BEGIN}\n{results_table()}\n{END}{post}")
    print("README results table updated")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="*", help="experiment dir names under experiments/")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--update-readme", action="store_true")
    a = ap.parse_args()
    style.apply()
    exps = sorted(p for p in (REPO_ROOT / "experiments").glob("E*_*") if p.is_dir()) if a.all else \
        [REPO_ROOT / "experiments" / e for e in a.exp]
    for e in exps:
        written = figures_for(e)
        print(f"{e.name}: {len(written)} files" + "".join(f"\n  {p.relative_to(REPO_ROOT)}" for p in written))
    if a.update_readme:
        update_readme()
    return 0


if __name__ == "__main__":
    sys.exit(main())
