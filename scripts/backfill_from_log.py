#!/usr/bin/env python
"""Convert a raw lerobot-train stdout log into experiments/<ID>/metrics/{train,events,offline_eval}.jsonl.

    python scripts/backfill_from_log.py --config configs/train/act_orange_2cam_b8.yaml \
        --log logs/act_orange_2cam_b8.log [--follow] [--force]

ts / wall_s come from the lerobot log header (1 s resolution), not from the backfill wall clock.
"""

from __future__ import annotations

import sys as _sys

_sys.dont_write_bytecode = True  # repo rule: never create __pycache__ (CLAUDE.md §3.9)

import argparse
import hashlib
import sys
import time
from pathlib import Path

from aiku_hands import REPO_ROOT
from aiku_hands.config import load_train_config
from aiku_hands.logging.ingest import LineBuffer, TrainLogIngestor
from aiku_hands.logging.jsonl import JsonlWriter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--follow", action="store_true", help="keep tailing the log until 'End of training'")
    ap.add_argument("--force", action="store_true", help="truncate existing metrics jsonl first")
    ap.add_argument("--poll-s", type=float, default=2.0)
    a = ap.parse_args()

    cfg = load_train_config(a.config)
    mdir = REPO_ROOT / "experiments" / cfg.exp_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    targets = [mdir / f for f in ("train.jsonl", "events.jsonl", "offline_eval.jsonl")]
    existing = [p for p in targets if p.exists() and p.stat().st_size > 0]
    if existing and not a.force:
        print(f"refusing: {', '.join(map(str, existing))} already non-empty (append-only). use --force", file=sys.stderr)
        return 2
    if a.force:
        for p in existing:
            p.write_text("")

    tw = JsonlWriter(targets[0], cfg.exp_id, cfg.run_id, fsync=False)
    ew = JsonlWriter(targets[1], cfg.exp_id, cfg.run_id, fsync=False)
    vw = JsonlWriter(targets[2], cfg.exp_id, cfg.run_id, fsync=False)
    ing = TrainLogIngestor(tw, ew, vw, batch_size=cfg.batch_size, log_freq=cfg.log_freq,
                           total_steps=cfg.steps, run_dir=cfg.output_dir, use_log_time=True)
    sha = hashlib.sha256(a.log.read_bytes()).hexdigest() if not a.follow else None
    ew.write({"type": "backfill_start", "source_log": str(a.log), "source_sha256": sha,
              "config": str(a.config), "batch_size": cfg.batch_size, "log_freq": cfg.log_freq})

    lb = LineBuffer()
    with open(a.log, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if chunk:
                for line in lb.push(chunk):
                    ing.feed(line)
                continue
            if not a.follow or ing.saw_end:
                break
            time.sleep(a.poll_s)
    for line in lb.flush():
        ing.feed(line)
    ing.finalize(None)
    ew.write({"type": "backfill_end", "counts": ing.counts, "last_step": ing.last_step})
    for w in (tw, ew, vw):
        w.close()
    print(f"backfill done: {ing.counts} last_step={ing.last_step} -> {mdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
