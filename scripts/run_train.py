#!/usr/bin/env python
"""lerobot-train wrapper (logging v0): manifest + live JSONL + sysmon + raw log tee.

    caffeinate -is python scripts/run_train.py --config configs/train/act_orange_2cam_b8.yaml
    python scripts/run_train.py --config ... --dry-run      # no training; validates manifest + parser

Safety (CLAUDE.md §3):
- refuses to start if another `lerobot-train` is running (MPS 1-job rule)
- refuses if disk free < 5 GB, if output_dir exists, or if metrics/train.jsonl is non-empty
- never kills the child; Ctrl-C is forwarded by the terminal to the child's process group, we just wait
"""

from __future__ import annotations

import sys as _sys

_sys.dont_write_bytecode = True  # repo rule: never create __pycache__ (CLAUDE.md §3.9)

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import yaml

from aiku_hands import REPO_ROOT, registry
from aiku_hands.config import load_train_config
from aiku_hands.logging.ingest import LineBuffer, TrainLogIngestor
from aiku_hands.logging.jsonl import JsonlWriter, now_iso, read_jsonl
from aiku_hands.logging.manifest import build_manifest
from aiku_hands.logging.sysmon import DISK_WARN_GB, SysMon, disk_free_gb

SAMPLE_LOG = REPO_ROOT / "tests" / "fixtures" / "lerobot_smoke_stdout.log"


def other_training_pids() -> list[int]:
    me = os.getpid()
    pids = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if p.info["pid"] != me and "lerobot-train" in cmd and "run_train.py" not in cmd:
            pids.append(p.info["pid"])
    return pids


def resolve_entry(entry: str) -> str:
    venv_bin = Path(sys.executable).parent / entry
    return str(venv_bin) if venv_bin.exists() else (shutil.which(entry) or entry)


def preflight(cfg, exp_dir: Path, dry: bool) -> list[str]:
    problems = []
    if pids := other_training_pids():
        problems.append(f"another lerobot-train is running (pids {pids}) — MPS 1-job rule")
    free = disk_free_gb()
    if free < DISK_WARN_GB:
        problems.append(f"disk free {free} GB < {DISK_WARN_GB} GB")
    if cfg.output_dir.exists():
        problems.append(f"output_dir exists: {cfg.output_dir} (lerobot would refuse)")
    tj = exp_dir / "metrics" / "train.jsonl"
    if tj.exists() and tj.stat().st_size > 0:
        problems.append(f"{tj} is non-empty (append-only; new run needs a new exp)")
    if cfg.steps and cfg.get("save_freq"):
        n_ckpt = -(-cfg.steps // int(cfg.get("save_freq")))
        print(f"[preflight] expected checkpoints: {n_ckpt} (≈{n_ckpt * 0.6:.1f} GB at 0.6 GB/ckpt 🔮), disk free {free} GB")
    return problems


def validate_parser_on_sample(cfg) -> dict:
    """Dry-run check: run the real ingestor over the measured smoke log into a temp dir."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = [JsonlWriter(td / f, cfg.exp_id, cfg.run_id, fsync=False) for f in ("t.jsonl", "e.jsonl", "v.jsonl")]
        ing = TrainLogIngestor(*ws, batch_size=8, log_freq=20, total_steps=200, run_dir=None, use_log_time=True)
        lb = LineBuffer()
        for line in lb.push(SAMPLE_LOG.read_text()) + lb.flush():
            ing.feed(line)
        ing.finalize(0)
        for w in ws:
            w.close()
        rows = list(read_jsonl(td / "t.jsonl"))
        return {"counts": ing.counts, "first": rows[0] if rows else None, "last_step": ing.last_step}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--exp", help="sanity check: must match exp_id in the config (e.g. E002)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sysmon-interval-s", type=float, default=30.0)
    a = ap.parse_args()

    cfg = load_train_config(a.config)
    if a.exp and a.exp != cfg.exp_id:
        print(f"--exp {a.exp} != config exp_id {cfg.exp_id}", file=sys.stderr)
        return 2
    exp_dir = REPO_ROOT / "experiments" / cfg.exp_dir
    entry = resolve_entry(cfg.entrypoint)
    argv = [entry, *cfg.resolved_args()]
    print("[cmd]", " ".join(shlex.quote(x) for x in argv))

    problems = preflight(cfg, exp_dir, a.dry_run)
    if problems and not a.dry_run:
        for p in problems:
            print("[preflight] REFUSE:", p, file=sys.stderr)
        return 3
    for p in problems:
        print("[preflight] (dry-run, would refuse):", p)

    ds_root = cfg.get("dataset.root")
    ds_root = Path(os.path.expanduser(os.path.expandvars(ds_root))) if ds_root else None

    if a.dry_run:
        with tempfile.TemporaryDirectory() as td:
            m = build_manifest(exp_id=cfg.exp_id, run_id=cfg.run_id, cmdline=argv, repo=REPO_ROOT,
                               dataset_root=ds_root, dataset_repo_id=cfg.get("dataset.repo_id"),
                               seed=cfg.seed, env_dir=Path(td) / "env")
            m["pip_freeze_lines"] = len((Path(td) / "env" / "pip_freeze.txt").read_text().splitlines())
        print("[dry-run] manifest:", json.dumps(m, indent=1, ensure_ascii=False)[:3000])
        print("[dry-run] parser on sample:", json.dumps(validate_parser_on_sample(cfg), ensure_ascii=False))
        ok = m["dataset"].get("info_json_sha256") is not None and m["lerobot"] is not None
        print("[dry-run] OK" if ok else "[dry-run] manifest incomplete")
        return 0 if ok else 1

    # ---------------------------------------------------------------- real run
    mdir = exp_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    raw_log = REPO_ROOT / "logs" / f"{cfg.run_id}.log"
    t0 = time.time()
    manifest = build_manifest(exp_id=cfg.exp_id, run_id=cfg.run_id, cmdline=argv, repo=REPO_ROOT,
                              dataset_root=ds_root, dataset_repo_id=cfg.get("dataset.repo_id"),
                              seed=cfg.seed, env_dir=exp_dir / "env")
    manifest["wrapper_cmdline"] = sys.argv
    manifest["raw_log"] = str(raw_log.relative_to(REPO_ROOT))
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(
        {"exp_id": cfg.exp_id, "run_id": cfg.run_id, "entrypoint": cfg.entrypoint,
         "source_config": str(a.config), "args": cfg.args, "resolved_args": cfg.resolved_args()},
        sort_keys=False, allow_unicode=True))

    tw = JsonlWriter(mdir / "train.jsonl", cfg.exp_id, cfg.run_id, t0)
    ew = JsonlWriter(mdir / "events.jsonl", cfg.exp_id, cfg.run_id, t0)
    vw = JsonlWriter(mdir / "offline_eval.jsonl", cfg.exp_id, cfg.run_id, t0)
    sw = JsonlWriter(mdir / "system.jsonl", cfg.exp_id, cfg.run_id, t0)
    ew.write({"type": "run_start", "cmdline": argv, "manifest": str((exp_dir / "manifest.json").relative_to(REPO_ROOT)),
              "raw_log": str(raw_log.relative_to(REPO_ROOT))})
    registry.upsert({"id": cfg.exp_dir, "date": now_iso()[:10], "status": "running",
                     "steps": str(cfg.steps or ""), "batch": str(cfg.batch_size)})

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)
    ing = TrainLogIngestor(tw, ew, vw, batch_size=cfg.batch_size, log_freq=cfg.log_freq,
                           total_steps=cfg.steps, run_dir=cfg.output_dir)
    mon = SysMon(sw, proc.pid, a.sysmon_interval_s,
                 on_disk_low=lambda gb: ew.write({"type": "disk_warning", "disk_free_gb": gb}))
    mon.start()
    lb = LineBuffer()
    exit_code = None
    try:
        with open(raw_log, "ab") as raw:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read1(1 << 14) if hasattr(proc.stdout, "read1") else proc.stdout.read(1 << 14)
                if not chunk:
                    break
                raw.write(chunk)
                raw.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                for line in lb.push(chunk.decode("utf-8", errors="replace")):
                    ing.feed(line)
        for line in lb.flush():
            ing.feed(line)
        exit_code = proc.wait()
    except KeyboardInterrupt:
        print("\n[run_train] KeyboardInterrupt: waiting for lerobot-train to exit (not killing it)", file=sys.stderr)
        exit_code = proc.wait()
    finally:
        mon.stop()
        ing.finalize(exit_code)
        wall = round(time.time() - t0, 1)
        ew.write({"type": "run_end", "exit_code": exit_code, "total_wall_s": wall,
                  "last_step": ing.last_step, "counts": ing.counts})
        status = "done" if exit_code == 0 and ing.saw_end else ("aborted" if exit_code in (-2, 130) else "failed")
        registry.upsert({"id": cfg.exp_dir, "status": status})
        for w in (tw, ew, vw, sw):
            w.close()
        print(f"[run_train] exit={exit_code} wall={wall}s status={status} counts={ing.counts}")
    return int(exit_code or 0)


if __name__ == "__main__":
    sys.exit(main())
