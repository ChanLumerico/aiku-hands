"""Stand-in for sim/leisaac_act_rollout.py used by `sim/eval.py run --mock`.

Emits the same progress events and per-episode records, with fake timing and random success.
MOCK_FAIL_COND=<condition> makes that condition crash mid-run (to test crash handling / --resume).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
from progress import Progress  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("--task", "--out", "--progress", "--exp_id", "--condition", "--model", "--video_dir", "--dump_dir"):
    ap.add_argument(k)
for k in ("--eval_rounds", "--seed", "--policy_port", "--action_horizon", "--warmup_steps", "--step_hz",
          "--ckpt_step", "--video_eps", "--video_every", "--dump_eps"):
    ap.add_argument(k, type=int, default=0)
ap.add_argument("--episode_length_s", type=float, default=60.0)
a, _ = ap.parse_known_args()

prog = Progress(a.progress, cond=a.condition)
rng = random.Random(a.seed * 7 + a.ckpt_step + a.action_horizon)
time.sleep(1.5)  # "Isaac Sim load"
max_steps = int(a.episode_length_s * 60)
prog.emit("start", n_eps=a.eval_rounds, max_steps=max_steps, horizon=a.action_horizon, warmup_steps=a.warmup_steps)
p_succ = min(0.9, a.ckpt_step / 100000)
succ = 0
with open(a.out, "a") as out:
    for ep in range(1, a.eval_rounds + 1):
        if os.environ.get("MOCK_FAIL_COND") == a.condition and ep == 2:
            prog.emit("error", msg="mock crash")
            sys.exit(3)
        prog.emit("ep_start", ep=ep)
        ok = rng.random() < p_succ
        n = rng.randint(max_steps // 4, max_steps // 2) if ok else max_steps
        t0 = time.time()
        for s in range(0, n + 1, 60):
            time.sleep(0.01)
            prog.tick(every_s=0.1, ep=ep, steps=s, n_infer=s // max(a.action_horizon, 1) + 1,
                      sps=round(s / max(time.time() - t0, 1e-6), 1), infer_ms=20.0)
        succ += ok
        dur = round(time.time() - t0, 2)
        out.write(json.dumps({"schema_version": 1, "exp_id": a.exp_id, "ckpt_step": a.ckpt_step,
                              "condition": a.condition, "trial": ep, "seed": a.seed, "success": ok,
                              "episode_len_steps": n, "duration_s": dur, "notes": "" if ok else "timeout",
                              "action_horizon": a.action_horizon, "mean_infer_ms": 20.0, "mock": True}) + "\n")
        out.flush()
        prog.emit("ep_end", ep=ep, success=ok, steps=n, duration_s=dur, successes=succ, infer_ms=20.0)
prog.emit("done", successes=succ, n_eps=a.eval_rounds)
