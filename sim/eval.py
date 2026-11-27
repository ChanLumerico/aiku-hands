#!/usr/bin/env python3
"""One-command sim evaluation on the HANDS PC, with live progress bars (stdlib only; Python >= 3.8).

    python3 sim/eval.py run configs/eval/E001_ckpt_sweep.json      # start (run inside tmux; `scripts/ah eval`)
    python3 sim/eval.py watch [RUN_DIR|latest]                     # live progress from any terminal / ssh
    python3 sim/eval.py summary [RUN_DIR|latest] [--md]            # success rate + Wilson 95% CI per condition
    python3 sim/eval.py run --resume latest                        # re-run conditions that did not finish
    python3 sim/eval.py list
    python3 sim/eval.py run configs/eval/E001_ckpt_sweep.json --mock --episodes 3   # pipeline test, no Isaac

For each checkpoint: start sim/act_policy_server.py (policy env), then for each horizon start
sim/leisaac_act_rollout.py (sim env, headless) and follow its progress events. Everything goes to
runs/sim_eval/<stamp>_<config>/ (gitignored):

    plan.json            resolved config + git commit + host       (provenance)
    progress.jsonl       UI events (orchestrator + rollout)
    rollouts/<cond>.jsonl  per-episode records (docs/05 schema)   <- research data
    videos/<cond>/ep###.mp4, logs/{server,sim}_<cond>.log, dumps/<cond>/

Aggregates are never written to disk (docs/05): `summary` recomputes them from rollouts/*.jsonl.
"""

from __future__ import annotations

import sys as _sys

_sys.dont_write_bytecode = True  # repo rule: never create __pycache__ (CLAUDE.md §3.9)

import argparse
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "sim"
MOCK = REPO / "tests" / "mock_sim"
RUNS = REPO / "runs" / "sim_eval"

DEFAULTS = {
    "exp_id": "E001",
    "run": "act_orange_2cam_b8",
    "ckpt_root": None,            # default: runs/<run>/checkpoints
    "task": "LeIsaac-SO101-PickOrange-v0",
    "episodes": 20,
    "seed": 1000,
    "episode_length_s": 60.0,
    "warmup_steps": 120,
    "step_hz": 0,
    "video_eps": 2,
    "video_every": 2,
    "dump_eps": 0,
    "model": "A",
    "port": 5765,
    "device": "cuda",
    "envs": {"policy": "lerobot070", "sim": "leisaac"},
    "server_timeout_s": 300,
    "sim_close_grace_s": 90,
    # success (ADR-0004): protocol_v1 = oranges in plate + released, held success_hold_steps, within max_policy_steps
    "success_def": "protocol_v1",
    "success_hold_steps": 60,
    "release_dist": 0.10,
    "max_policy_steps": 1800,
    # per condition also: name, episodes, seed, warmup_steps, rest [6], policy_mode chunk|step,
    #                     temporal_ensemble <coeff> (server flag; implies policy_mode step), extra_args
    "conditions": [{"ckpt": 70000, "horizon": 100}],
}


# ----------------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def cond_name(c: dict) -> str:
    ck = c["ckpt"]
    ck_s = f"{ck // 1000}k" if ck % 1000 == 0 else str(ck)
    return c.get("name") or f"ck{ck_s}_h{c['horizon']}"


def hms(s: Optional[float]) -> str:
    if s is None or s != s or s < 0:
        return "--:--"
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def ckpt_dir(cfg: dict, ckpt: int) -> Path:
    root = Path(os.path.expanduser(cfg["ckpt_root"])) if cfg.get("ckpt_root") else REPO / "runs" / cfg["run"] / "checkpoints"
    return root / f"{ckpt:06d}" / "pretrained_model"


def find_conda() -> str:
    cands = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    cands += [os.path.expanduser(f"~/{d}/bin/conda") for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge")]
    cands += ["/opt/conda/bin/conda"]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit("conda not found (set CONDA_EXE=/path/to/conda)")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(start: int) -> int:
    for p in range(start, start + 50):
        if port_free(p):
            return p
    sys.exit(f"no free port in {start}..{start + 49}")


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def resolve_run(arg: Optional[str]) -> Path:
    if not arg or arg == "latest":
        p = RUNS / "latest"
        if not p.exists():
            sys.exit(f"no runs yet in {RUNS}")
        return p.resolve()
    p = Path(arg)
    if not p.exists() and (RUNS / arg).exists():
        p = RUNS / arg
    if not p.exists():
        sys.exit(f"no such run: {arg}")
    return p.resolve()


def read_jsonl(path: Path) -> List[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def gpu_line() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        u, m, t = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        return f"GPU {u}% {float(m) / 1024:.1f}/{float(t) / 1024:.1f} GB"
    except Exception:
        return ""


# ----------------------------------------------------------------------------- progress state + rendering
class Tail:
    """Incrementally read new complete lines of a JSONL file."""

    def __init__(self, path: Path):
        self.path, self.pos, self.buf = path, 0, ""

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        self.buf += data
        *lines, self.buf = self.buf.split("\n")
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


class State:
    def __init__(self, plan: dict):
        self.plan = plan
        self.order = [cond_name(c) for c in plan["conditions"]]
        self.c: Dict[str, dict] = {n: {"stage": "pending", "ep_done": 0, "succ": 0, "n_eps": plan["episodes"],
                                       "ep": 0, "steps": 0, "max_steps": None, "sps": None, "infer_ms": None,
                                       "t_stage": None, "load_s": None, "ep_s": [], "rc": None, "placed": []}
                                   for n in self.order}
        self.t_start = None
        self.ended = None
        self.notes: List[str] = []

    def apply(self, e: dict) -> None:
        ev, t, cond = e.get("ev"), e.get("t", time.time()), e.get("cond")
        s = self.c.get(cond) if cond else None
        if ev == "run_start":
            self.t_start, self.ended = t, None
            for n in e.get("skip", []):
                if n in self.c:
                    self.c[n]["stage"] = "done"
        elif ev == "run_end":
            self.ended = e.get("status", "done")
        elif ev == "note":
            self.notes = (self.notes + [e.get("msg", "")])[-3:]
        elif s is None:
            return
        elif ev == "server_start":
            s.update(stage="server", t_stage=t)
        elif ev == "sim_start":
            s.update(stage="loading", t_stage=t, ep_done=0, succ=0, ep=0, steps=0, ep_s=[], placed=[])
        elif ev == "start":
            s.update(stage="running", max_steps=e.get("max_steps"), n_eps=e.get("n_eps", s["n_eps"]))
            if s["t_stage"]:
                s["load_s"] = t - s["t_stage"]
        elif ev == "ep_start":
            s.update(ep=e["ep"], steps=0)
        elif ev == "step":
            s.update(steps=e.get("steps", 0), sps=e.get("sps"), infer_ms=e.get("infer_ms"))
        elif ev == "ep_end":
            s["ep_done"] = e["ep"]
            s["succ"] = e.get("successes", s["succ"])
            s["ep_s"].append(e.get("duration_s", 0.0))
            if e.get("max_placed") is not None:
                s["placed"].append(e["max_placed"])
            s["infer_ms"] = e.get("infer_ms", s["infer_ms"])
        elif ev == "done":
            s.update(stage="closing")
        elif ev == "error":
            self.notes = (self.notes + [f"{cond}: {e.get('msg', '')}"])[-3:]
        elif ev == "cond_done":
            s.update(stage=e.get("status", "done"), rc=e.get("rc"))

    def done_seen(self, name: str) -> bool:
        return self.c[name]["stage"] in ("closing", "done")

    def eta(self) -> Optional[float]:
        ep_s = [x for s in self.c.values() for x in s["ep_s"]]
        if not ep_s:
            return None
        ep_avg = sum(ep_s) / len(ep_s)
        loads = [s["load_s"] for s in self.c.values() if s["load_s"]]
        load = sum(loads) / len(loads) if loads else 120.0
        rem = 0.0
        for s in self.c.values():
            if s["stage"] in ("done", "crashed", "aborted"):
                continue
            if s["stage"] in ("running", "closing"):
                rem += max(0, s["n_eps"] - s["ep_done"]) * ep_avg
            else:
                rem += load + s["n_eps"] * ep_avg + 15
        return rem


def bar(frac: float, width: int) -> str:
    frac = 0.0 if frac != frac else max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def render(st: State, run_dir: Path, gpu: str = "") -> List[str]:
    p = st.plan
    now = time.time()
    n_done = sum(1 for s in st.c.values() if s["stage"] in ("done", "crashed", "aborted"))
    el = now - st.t_start if st.t_start else None
    head = f"aiku-hands sim eval · {p['exp_id']} · {run_dir.name}"
    sub = f"conds {n_done}/{len(st.order)} · elapsed {hms(el)} · ETA {hms(st.eta())}" + (f" · {gpu}" if gpu else "")
    L = [head, sub, "─" * min(78, max(len(head), len(sub)))]
    for n in st.order:
        s = st.c[n]
        k, m = s["succ"], s["ep_done"]
        rate = f"{k}/{m}" + (f" ({100 * k / m:.0f}%)" if m else "")
        if s["placed"]:
            rate += f" · placed≤ {sum(s['placed']) / len(s['placed']):.1f}"

        if s["stage"] == "pending":
            L.append(f" ·  {n:<14} pending")
        elif s["stage"] in ("done",):
            lo, hi = wilson(k, m)
            ci = f"CI95 [{lo:.2f},{hi:.2f}]" if m else "(skipped / earlier run)"
            L.append(f" ✓  {n:<14} succ {rate:<12} {ci}")
        elif s["stage"] in ("crashed", "aborted"):
            L.append(f" ✗  {n:<14} {s['stage']} (rc={s['rc']}) after {m} eps, succ {rate}  → logs/sim_{n}.log")
        else:
            L.append(f" ▶  {n:<14} {bar(m / max(s['n_eps'], 1), 20)} {m:>3}/{s['n_eps']} eps · succ {rate}")
            if s["stage"] == "server":
                L.append(f"      starting policy server … {hms(now - (s['t_stage'] or now))}")
            elif s["stage"] == "loading":
                L.append(f"      loading Isaac Sim (≈2 min) … {hms(now - (s['t_stage'] or now))}")
            elif s["stage"] == "closing":
                L.append("      closing sim …")
            else:
                ms = s["max_steps"] or 0
                extra = []
                if s["sps"]:
                    extra.append(f"{s['sps']:.0f} st/s")
                if s["infer_ms"]:
                    extra.append(f"infer {s['infer_ms']:.0f} ms")
                L.append(f"      ep {s['ep']:>3} {bar(s['steps'] / ms if ms else 0, 20)} {s['steps']}/{ms or '?'} steps"
                         + (" · " + " · ".join(extra) if extra else ""))
    for note in st.notes:
        L.append(f" ! {note[:100]}")
    if st.ended:
        L.append(f"── run {st.ended} ──  python3 sim/eval.py summary {run_dir.name}")
    return L


class Display:
    def __init__(self, run_dir: Path, plan: dict):
        self.run_dir, self.st, self.tail = run_dir, State(plan), Tail(run_dir / "progress.jsonl")
        self.tty = sys.stdout.isatty()
        self.n_prev = 0
        self.t_draw = 0.0
        self.t_gpu, self.gpu = 0.0, ""

    def pump(self, force: bool = False) -> None:
        evs = self.tail.read()
        for e in evs:
            self.st.apply(e)
            if not self.tty:  # plain log lines for non-interactive output
                if e.get("ev") in ("ep_end", "cond_done", "server_ready", "sim_start", "run_end", "error", "note"):
                    print(json.dumps({k: e[k] for k in e if k not in ("policy", "tb")}), flush=True)
        if not self.tty:
            return
        now = time.time()
        if not force and now - self.t_draw < 0.5:
            return
        if now - self.t_gpu > 5:
            self.gpu, self.t_gpu = gpu_line(), now
        self.t_draw = now
        cols = shutil.get_terminal_size((100, 30)).columns
        lines = [ln[: cols - 1] for ln in render(self.st, self.run_dir, self.gpu)]
        buf = (f"\x1b[{self.n_prev}F" if self.n_prev else "") + "\x1b[J" + "\n".join(lines) + "\n"
        sys.stdout.write(buf)
        sys.stdout.flush()
        self.n_prev = len(lines)


# ----------------------------------------------------------------------------- run
class Runner:
    def __init__(self, cfg: dict, run_dir: Path, mock: bool):
        self.cfg, self.rd, self.mock = cfg, run_dir, mock
        self.conda = None if mock else find_conda()
        self.procs: List[subprocess.Popen] = []
        for d in ("logs", "rollouts", "videos", "dumps"):
            (run_dir / d).mkdir(parents=True, exist_ok=True)
        self.ev_f = open(run_dir / "progress.jsonl", "a", encoding="utf-8", buffering=1)
        self.disp = Display(run_dir, cfg)

    def ev(self, ev: str, **kw) -> None:
        rec = {"ev": ev, "t": round(time.time(), 3)}
        rec.update(kw)
        self.ev_f.write(json.dumps(rec) + "\n")
        self.ev_f.flush()

    def py(self, env_key: str) -> List[str]:
        if self.mock:
            return [sys.executable, "-u"]
        return [self.conda, "run", "--no-capture-output", "-n", self.cfg["envs"][env_key], "python", "-u"]

    def launch(self, cmd: List[str], log: Path) -> subprocess.Popen:
        f = open(log, "ab")
        f.write(("\n$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env,
                             start_new_session=True)
        self.procs.append(p)
        return p

    def kill(self, p: Optional[subprocess.Popen], grace: float = 10.0) -> None:
        if p is None or p.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                return
            t0 = time.time()
            while time.time() - t0 < wait:
                if p.poll() is not None:
                    return
                time.sleep(0.2)

    def wait(self, pred, timeout: Optional[float], alive: List[subprocess.Popen]) -> str:
        t0 = time.time()
        while True:
            self.disp.pump()
            if pred():
                return "ok"
            if any(p.poll() is not None for p in alive):
                return "died"
            if timeout is not None and time.time() - t0 > timeout:
                return "timeout"
            time.sleep(0.25)

    def run(self, todo: List[dict]) -> str:
        cfg = self.cfg
        groups: List[List[dict]] = []
        def skey(c):  # server identity: checkpoint + temporal-ensemble setting
            return (c["ckpt"], c.get("temporal_ensemble"))
        for c in todo:  # consecutive conditions sharing a server config share one policy server
            if groups and skey(groups[-1][0]) == skey(c):
                groups[-1].append(c)
            else:
                groups.append([c])
        status = "done"
        for grp in groups:
            first = cond_name(grp[0])
            ck = grp[0]["ckpt"]
            port = pick_port(cfg["port"])
            te = grp[0].get("temporal_ensemble")
            slog = self.rd / "logs" / (f"server_ck{ck}" + (f"_te{te}" if te is not None else "") + ".log")
            server_script = MOCK / "mock_policy_server.py" if self.mock else SIM / "act_policy_server.py"
            self.ev("server_start", cond=first, ckpt=ck, port=port)
            srv = self.launch(self.py("policy") + [str(server_script), "--ckpt", str(ckpt_dir(cfg, ck)),
                                                  "--device", cfg["device"], "--port", str(port)]
                              + (["--temporal_ensemble", str(te)] if te is not None else []), slog)

            def ready():
                return slog.exists() and "listening on" in slog.read_text(errors="replace")
            r = self.wait(ready, cfg["server_timeout_s"], [srv])
            if r != "ok":
                self.ev("note", msg=f"policy server ck{ck} {r} → {slog.relative_to(self.rd)}")
                for c in grp:
                    self.ev("cond_done", cond=cond_name(c), status="crashed", rc=srv.poll())
                self.kill(srv)
                status = "partial"
                continue
            self.ev("server_ready", cond=first, ckpt=ck, port=port)
            for c in grp:
                if self.run_cond(c, port) != "done":
                    status = "partial"
            self.kill(srv)
        return status

    def run_cond(self, c: dict, port: int) -> str:
        cfg, name = self.cfg, cond_name(c)
        out = self.rd / "rollouts" / f"{name}.jsonl"
        if out.exists():  # earlier crashed attempt: keep it, but out of the summary
            k = 1
            while (self.rd / "rollouts" / f"{name}.crashed-{k}.jsonl.bak").exists():
                k += 1
            out.rename(self.rd / "rollouts" / f"{name}.crashed-{k}.jsonl.bak")
        script = MOCK / "mock_rollout.py" if self.mock else SIM / "leisaac_act_rollout.py"
        cmd = self.py("sim") + [
            str(script), "--task", cfg["task"], "--eval_rounds", str(c.get("episodes", cfg["episodes"])),
            "--seed", str(c.get("seed", cfg["seed"])), "--episode_length_s", str(cfg["episode_length_s"]),
            "--policy_port", str(port), "--action_horizon", str(c.get("horizon", 100)),
            "--warmup_steps", str(c.get("warmup_steps", cfg["warmup_steps"])), "--step_hz", str(cfg["step_hz"]),
            "--out", str(out), "--progress", str(self.rd / "progress.jsonl"),
            "--exp_id", cfg["exp_id"], "--ckpt_step", str(c["ckpt"]), "--model", cfg["model"],
            "--condition", name, "--headless", "--enable_cameras",
            "--success_def", c.get("success_def", cfg["success_def"]),
            "--success_hold_steps", str(c.get("success_hold_steps", cfg["success_hold_steps"])),
            "--release_dist", str(c.get("release_dist", cfg["release_dist"])),
            "--max_policy_steps", str(c.get("max_policy_steps", cfg["max_policy_steps"])),
            "--policy_mode", c.get("policy_mode", "step" if c.get("temporal_ensemble") is not None else "chunk"),
        ]
        if c.get("rest"):
            cmd += ["--rest"] + [str(v) for v in c["rest"]]
        if cfg["video_eps"] > 0:
            cmd += ["--video_dir", str(self.rd / "videos" / name), "--video_eps", str(cfg["video_eps"]),
                    "--video_every", str(cfg["video_every"])]
        if cfg["dump_eps"] > 0:
            cmd += ["--dump_dir", str(self.rd / "dumps" / name), "--dump_eps", str(cfg["dump_eps"])]
        cmd += c.get("extra_args", [])
        self.ev("sim_start", cond=name)
        p = self.launch(cmd, self.rd / "logs" / f"sim_{name}.log")
        self.wait(lambda: self.disp.st.done_seen(name), None, [p])
        if self.disp.st.done_seen(name) and p.poll() is None:  # Isaac Sim sometimes hangs on close
            if self.wait(lambda: p.poll() is not None, cfg["sim_close_grace_s"], []) != "ok":
                self.kill(p)
        rc = p.wait()
        self.disp.pump()
        status = "done" if self.disp.st.done_seen(name) else "crashed"
        self.ev("cond_done", cond=name, status=status, rc=rc)
        if status != "done":
            log = self.rd / "logs" / f"sim_{name}.log"
            tail = log.read_text(errors="replace").splitlines()[-1:] if log.exists() else [""]
            self.ev("note", msg=f"{name} crashed rc={rc}: {tail[0][:80]}")
        self.disp.pump(force=True)
        return status

    def shutdown(self) -> None:
        for p in self.procs:
            self.kill(p, grace=5.0)


def load_cfg(path: str) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    envs = dict(cfg["envs"], **user.pop("envs", {}))
    cfg.update(user)
    cfg["envs"] = envs
    return cfg


def cmd_run(a) -> int:
    if a.resume:
        rd = resolve_run(a.resume)
        cfg = json.loads((rd / "plan.json").read_text())
        mock = cfg.get("_mock", False)
        st = State(cfg)
        for e in read_jsonl(rd / "progress.jsonl"):
            st.apply(e)
        skip = [n for n in st.order if st.c[n]["stage"] == "done"]
    else:
        if not a.config:
            sys.exit("usage: eval.py run CONFIG.json  (or --resume latest)")
        cfg = load_cfg(a.config)
        if a.episodes:
            cfg["episodes"] = a.episodes
        if a.video_eps is not None:
            cfg["video_eps"] = a.video_eps
        mock = a.mock
        cfg["_mock"] = mock
        try:
            cfg["_config_file"] = str(Path(a.config).resolve().relative_to(REPO))
        except ValueError:
            cfg["_config_file"] = a.config
        if not mock:
            missing = [str(ckpt_dir(cfg, c["ckpt"])) for c in cfg["conditions"] if not ckpt_dir(cfg, c["ckpt"]).is_dir()]
            if missing:
                print("missing checkpoints:\n  " + "\n  ".join(missing))
                print("→ on the MacBook: scripts/ah sync " + cfg["run"] + " " +
                      " ".join(str(c["ckpt"]) for c in cfg["conditions"]))
                return 2
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rd = RUNS / f"{stamp}_{Path(a.config).stem}{'_mock' if mock else ''}"
        rd.mkdir(parents=True)
        cfg["_created"] = datetime.now().astimezone().isoformat(timespec="seconds")
        cfg["_host"] = socket.gethostname()
        cfg["_git"] = git_info()
        (rd / "plan.json").write_text(json.dumps(cfg, indent=2))
        skip = []
    latest = RUNS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(rd.name)
    (rd / "orchestrator.pid").write_text(str(os.getpid()))
    todo = [c for c in cfg["conditions"] if cond_name(c) not in skip]
    runner = Runner(cfg, rd, mock)

    def on_sig(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGHUP, on_sig)
    runner.ev("run_start", skip=skip, n_todo=len(todo), pid=os.getpid())
    status = "aborted"
    try:
        status = runner.run(todo)
    except KeyboardInterrupt:
        runner.ev("note", msg="interrupted → stopping children (resume: eval.py run --resume latest)")
    finally:
        runner.shutdown()
        runner.ev("run_end", status=status)
        runner.disp.pump(force=True)
        try:
            (rd / "orchestrator.pid").unlink()
        except FileNotFoundError:
            pass
    print()
    summarize(rd, md=False)
    return 0 if status == "done" else 1


def orchestrator_alive(rd: Path) -> bool:
    pf = rd / "orchestrator.pid"
    if not pf.exists():
        return False
    try:
        os.kill(int(pf.read_text().strip()), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def cmd_watch(a) -> int:
    rd = resolve_run(a.run)
    t0 = time.time()
    while not (rd / "plan.json").exists():
        if time.time() - t0 > 30:
            sys.exit(f"{rd}: no plan.json")
        time.sleep(0.5)
    disp = Display(rd, json.loads((rd / "plan.json").read_text()))
    if not disp.tty:
        print("(not a TTY: printing events; use `ssh -t` for live bars)")
    try:
        while True:
            disp.pump()
            if disp.st.ended:
                disp.pump(force=True)
                print()
                summarize(rd, md=False)
                return 0
            if not orchestrator_alive(rd) and time.time() - t0 > 5:
                disp.pump(force=True)
                print("\n(orchestrator is not running — finished, crashed, or killed. "
                      "resume: python3 sim/eval.py run --resume " + rd.name + ")")
                return 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n(detached; the run keeps going)")
        return 0


def summarize(rd: Path, md: bool) -> None:
    plan = json.loads((rd / "plan.json").read_text())
    rows = []
    for c in plan["conditions"]:
        n_ = cond_name(c)
        recs = read_jsonl(rd / "rollouts" / f"{n_}.jsonl")
        n, k = len(recs), sum(bool(r.get("success")) for r in recs)
        lo, hi = wilson(k, n)
        to = sum(1 for r in recs if r.get("notes") == "timeout")
        ls = [r["episode_len_steps"] for r in recs if r.get("success")]
        inf = [r["mean_infer_ms"] for r in recs if r.get("mean_infer_ms") is not None]
        pl = [r["max_objects_placed"] for r in recs if r.get("max_objects_placed") is not None]
        kl = sum(bool(r.get("success_leisaac", r.get("success") if "success_def" not in r else False)) for r in recs)
        defs = sorted({r.get("success_def", "leisaac") for r in recs})
        h_lbl = "TE" if c.get("temporal_ensemble") is not None else str(c.get("horizon", "-"))
        rows.append((n_, c["ckpt"], h_lbl, n, k, (k / n if n else float("nan")), lo, hi, to,
                     (sum(ls) / len(ls) if ls else None), (sum(inf) / len(inf) if inf else None), (sum(pl) / len(pl) if pl else None), kl, "+".join(defs) or "-"))
    print(f"run {rd.name} · {plan['exp_id']} · task {plan['task']} · seed {plan['seed']} · "
          f"warmup {plan['warmup_steps']} · git {plan.get('_git', {}).get('commit', '')[:8]}"
          + (" (MOCK)" if plan.get("_mock") else ""))
    hdr = ["cond", "ckpt", "h", "n", "succ", "rate", "Wilson 95% CI", "timeouts", "steps|succ", "infer ms",
           "max placed (mean)", "succ LeIsaac", "success_def"]
    fmt = [(r[0], str(r[1]), str(r[2]), str(r[3]), str(r[4]), "-" if r[3] == 0 else f"{r[5]:.2f}",
            "-" if r[3] == 0 else f"[{r[6]:.2f}, {r[7]:.2f}]", str(r[8]),
            "-" if r[9] is None else f"{r[9]:.0f}", "-" if r[10] is None else f"{r[10]:.1f}",
            "-" if r[11] is None else f"{r[11]:.2f}", str(r[12]), r[13]) for r in rows]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for f in fmt:
            print("| " + " | ".join(f) + " |")
    else:
        w = [max(len(h), *(len(f[i]) for f in fmt)) for i, h in enumerate(hdr)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for f in fmt:
            print("  ".join(x.ljust(w[i]) for i, x in enumerate(f)))
    print("(n is small: compare conditions by CI overlap, not point estimates; docs/04)")


def cmd_summary(a) -> int:
    summarize(resolve_run(a.run), md=a.md)
    return 0


def cmd_list(a) -> int:
    if not RUNS.exists():
        print("no runs")
        return 0
    for rd in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink()):
        if not (rd / "plan.json").exists():
            continue
        st = State(json.loads((rd / "plan.json").read_text()))
        for e in read_jsonl(rd / "progress.jsonl"):
            st.apply(e)
        n_done = sum(1 for s in st.c.values() if s["stage"] == "done")
        live = " (running)" if orchestrator_alive(rd) else ""
        print(f"{rd.name:<50} {n_done}/{len(st.order)} conds{live}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("config", nargs="?")
    r.add_argument("--resume", metavar="RUN")
    r.add_argument("--episodes", type=int, help="override episodes per condition")
    r.add_argument("--video_eps", type=int, help="override number of recorded episodes per condition")
    r.add_argument("--mock", action="store_true", help="fake server + fake sim (pipeline test, no Isaac)")
    w = sp.add_parser("watch")
    w.add_argument("run", nargs="?", default="latest")
    s = sp.add_parser("summary")
    s.add_argument("run", nargs="?", default="latest")
    s.add_argument("--md", action="store_true")
    sp.add_parser("list")
    a = ap.parse_args()
    return {"run": cmd_run, "watch": cmd_watch, "summary": cmd_summary, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
