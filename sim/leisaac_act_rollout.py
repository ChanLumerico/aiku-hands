"""Roll out a lerobot 0.7.0 ACT policy in LeIsaac (Isaac Sim 5.1 / Isaac Lab 2.3 / LeIsaac 0.4.0).

Runs in the **leisaac** conda env on the HANDS PC, from the repo (never inside ~/lerobot_ws/).
The policy itself runs in a separate process: sim/act_policy_server.py (lerobot 0.7.0 env).
Normally launched by sim/eval.py (one command, progress bars); manual use:

    python sim/leisaac_act_rollout.py --task LeIsaac-SO101-PickOrange-v0 --eval_rounds 10 \
        --policy_port 5765 --action_horizon 100 --warmup_steps 120 --seed 1000 \
        --out rollouts.jsonl --ckpt_step 70000 --exp_id E001 --headless --enable_cameras

Adapted from LeIsaac v0.4.0 scripts/evaluation/policy_inference.py (Apache-2.0), replacing the
lerobot-v0.3.3 gRPC client with sim/act_bridge.py.

2026-11-27 changes (HANDS GUI pilot, docs/07):
  * --warmup_steps/--rest: after every reset, hold the dataset's folded rest pose before the policy starts.
    The sim resets to all-zero joints, which is out of distribution (dataset episodes start folded), and
    the very first rendered frame after reset has no textures.
  * --progress: JSONL progress events for sim/eval.py; --video_dir: per-episode mp4 of the policy's
    camera inputs (front | wrist); --dump_dir: PNG of the frames fed to each inference.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

REST_LEROBOT = [8.0, -99.9, 96.0, 64.0, 15.0, 13.0]  # typical dataset action0 (lerobot units)

parser = argparse.ArgumentParser(description="ACT (lerobot 0.7.0) rollout in LeIsaac via TCP bridge.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-PickOrange-v0")
parser.add_argument("--step_hz", type=int, default=60, help="real-time pacing; 0 = run as fast as possible")
parser.add_argument("--seed", type=int, default=1000, help="env seed (logged per rollout)")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="timeout per episode (sim seconds)")
parser.add_argument("--eval_rounds", type=int, default=10, help="number of episodes")
parser.add_argument("--policy_host", type=str, default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=5765)
parser.add_argument("--action_horizon", type=int, default=100,
                    help="actions executed per inference (<= chunk_size=100; training used n_action_steps=100)")
parser.add_argument("--warmup_steps", type=int, default=0,
                    help="sim steps holding --rest after each reset before the policy starts (0 = off)")
parser.add_argument("--rest", type=float, nargs=6, default=REST_LEROBOT, help="warmup pose, lerobot units")
parser.add_argument("--out", type=str, default="rollouts_sim_pilot.jsonl")
parser.add_argument("--progress", type=str, default=None, help="append progress events (JSONL) here")
parser.add_argument("--video_dir", type=str, default=None, help="write ep###.mp4 (front|wrist) here")
parser.add_argument("--video_eps", type=int, default=2, help="record the first N episodes (with --video_dir)")
parser.add_argument("--video_every", type=int, default=2, help="keep every k-th sim step (60/k fps = real time)")
parser.add_argument("--dump_dir", type=str, default=None, help="PNG of each inference's input frames")
parser.add_argument("--dump_eps", type=int, default=1, help="dump the first N episodes (with --dump_dir)")
parser.add_argument("--exp_id", type=str, default="E001")
parser.add_argument("--ckpt_step", type=int, default=70000)
parser.add_argument("--model", type=str, default="A")
parser.add_argument("--condition", type=str, default="nominal_pilot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type  # noqa: E402
from leisaac.utils.robot_utils import (  # noqa: E402
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

import leisaac  # noqa: F401,E402

try:  # registers LeIsaac-SO101-PickOrange-v0 (CLAUDE.md §2.2: needed before gym.make)
    import leisaac.tasks.pick_orange  # noqa: F401,E402
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from act_bridge import BridgeClient  # noqa: E402
from progress import Progress  # noqa: E402

KST = timezone(timedelta(hours=9))


class RateLimiter:
    """Verbatim from LeIsaac v0.4.0 policy_inference.py, plus hz <= 0 = no pacing."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz if hz > 0 else 0.0
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        if self.hz <= 0:
            return
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def compose(images: dict, cams: list) -> np.ndarray:
    """Side-by-side RGB frame (cams in policy order), padded to even H/W for yuv420p."""
    frames = [images[c] for c in cams]
    h = max(f.shape[0] for f in frames)
    frames = [np.pad(f, ((0, h - f.shape[0]), (0, 0), (0, 0))) for f in frames]
    out = np.concatenate(frames, axis=1)
    return np.pad(out, ((0, out.shape[0] % 2), (0, out.shape[1] % 2), (0, 0)))


class VideoWriter:
    """Pipe raw RGB frames into ffmpeg (no temp files). No-op if ffmpeg is missing."""

    def __init__(self, path: str, fps: float):
        self.path, self.fps, self.p = path, fps, None

    def add(self, frame: np.ndarray) -> None:
        if self.p is None:
            h, w = frame.shape[:2]
            self.p = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", f"{self.fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", self.path],
                stdin=subprocess.PIPE)
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()
            self.p = None


def save_png(path: str, img: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, img[:, :, ::-1])


class PlacedCounter:
    """Best-effort partial-progress metric: #oranges inside the plate box (x,y ±10 cm, z ±7 cm; the LeIsaac
    task_done thresholds, docs/07 §5) — WITHOUT the 'robot back at rest' part of success. Finds rigid objects
    by name ('orange*', 'plate*'); if not found, returns None and prints the available names once."""

    def __init__(self, env):
        self.oranges, self.plate = [], None
        rigid = getattr(getattr(env, "scene", None), "rigid_objects", None) or {}
        names = list(rigid.keys())
        self.oranges = [rigid[k] for k in names if "orange" in k.lower()]
        plates = [rigid[k] for k in names if "plate" in k.lower()]
        self.plate = plates[0] if plates else None
        ok = bool(self.oranges) and self.plate is not None
        print(f"[placed] rigid objects: {names} -> {'tracking ' + str(len(self.oranges)) + ' oranges' if ok else 'disabled'}")
        self.ok = ok

    def __call__(self):
        if not self.ok:
            return None
        p = self.plate.data.root_pos_w[0]
        n = 0
        for o in self.oranges:
            d = (o.data.root_pos_w[0] - p).abs()
            n += int(bool(d[0] < 0.10) and bool(d[1] < 0.10) and bool(d[2] < 0.07))
        return n


def main(prog: Progress):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    if task_type != "so101leader":
        raise ValueError(f"only single-arm so101leader tasks supported, got {task_type}")
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    client = BridgeClient(args_cli.policy_host, args_cli.policy_port)
    info = client.info
    print(f"[bridge] connected: {info}")
    cams = [k.split(".")[-1] for k in info["image_keys"]]
    horizon = min(args_cli.action_horizon, info["chunk_size"])
    rate = RateLimiter(args_cli.step_hz)
    max_steps = int(getattr(env, "max_episode_length", round(args_cli.episode_length_s * 60)))
    rest = torch.from_numpy(
        convert_lerobot_action_to_leisaac(np.asarray([args_cli.rest], dtype=np.float32))).float().to(env.device)
    try:
        placed = PlacedCounter(env)
    except Exception as e:  # never let the diagnostic metric break a rollout
        print(f"[placed] disabled: {type(e).__name__}: {e}")
        placed = lambda: None  # noqa: E731
    use_video = bool(args_cli.video_dir) and args_cli.video_eps > 0
    if use_video and shutil.which("ffmpeg") is None:
        print("[video] ffmpeg not found -> video disabled")
        use_video = False
    for d in (args_cli.video_dir if use_video else None, args_cli.dump_dir):
        if d:
            os.makedirs(d, exist_ok=True)

    def step_env(action):
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        return env.step(action)

    prog.emit("start", n_eps=args_cli.eval_rounds, max_steps=max_steps, horizon=horizon,
              warmup_steps=args_cli.warmup_steps, cams=cams, policy=info)
    out = open(args_cli.out, "a", encoding="utf-8")
    t_run = time.time()
    successes = 0
    obs_dict, _ = env.reset()
    for ep in range(1, args_cli.eval_rounds + 1):
        client.reset()
        prog.emit("ep_start", ep=ep)
        # --- warmup: hold the dataset rest pose (also flushes the untextured first frame) ---
        for _ in range(args_cli.warmup_steps):
            obs_dict, _, _, _, _ = step_env(rest)
            rate.sleep(env)
        state0 = convert_leisaac_action_to_lerobot(obs_dict["policy"]["joint_pos"])[0]
        print(f"[dbg] ep {ep} state0 (lerobot units) = {np.round(to_numpy(state0), 1).tolist()}")

        vw = None
        if use_video and ep <= args_cli.video_eps:
            vw = VideoWriter(os.path.join(args_cli.video_dir, f"ep{ep:03d}.mp4"), 60.0 / args_cli.video_every)
        success = time_out = False
        steps, n_infer, infer_s = 0, 0, 0.0
        n_placed, max_placed = None, None
        t_ep = time.time()
        while simulation_app.is_running() and not (success or time_out):
            with torch.inference_mode():
                pol = obs_dict["policy"]
                state = convert_leisaac_action_to_lerobot(pol["joint_pos"])[0]  # rad -> lerobot units
                images = {c: to_numpy(pol[c])[0].astype(np.uint8) for c in cams}  # (H,W,3)
                if args_cli.dump_dir and ep <= args_cli.dump_eps:
                    for c in cams:
                        save_png(os.path.join(args_cli.dump_dir, f"ep{ep}_{n_infer:03d}_{c}.png"), images[c])
                t0 = time.time()
                chunk = client.infer(state, images)  # (100, 6) lerobot units
                infer_s += time.time() - t0
                n_infer += 1
                actions = torch.from_numpy(convert_lerobot_action_to_leisaac(chunk)).float().to(env.device)
                for i in range(horizon):
                    obs_dict, _, terminated, time_outs, _ = step_env(actions[i : i + 1])
                    steps += 1
                    if steps % 15 == 0 and not (terminated[0] or time_outs[0]):  # before auto-reset moves objects
                        n_placed = placed()
                        if n_placed is not None:
                            max_placed = max(max_placed or 0, n_placed)
                    if vw is not None and steps % args_cli.video_every == 0:
                        vw.add(compose({c: to_numpy(obs_dict["policy"][c])[0].astype(np.uint8) for c in cams}, cams))
                    if terminated[0]:
                        success = True
                        break
                    if time_outs[0]:
                        time_out = True
                        break
                    rate.sleep(env)
                el = time.time() - t_ep
                prog.tick(ep=ep, steps=steps, n_infer=n_infer, sps=round(steps / max(el, 1e-6), 1),
                          infer_ms=round(1000 * infer_s / n_infer, 1))
        if vw is not None:
            vw.close()
        if not simulation_app.is_running():
            prog.emit("error", msg="simulation app stopped")
            break
        successes += int(success)
        dur = time.time() - t_ep
        rec = {
            "schema_version": 1, "exp_id": args_cli.exp_id, "run_id": f"sim_{args_cli.exp_id}_{args_cli.ckpt_step}",
            "ts": datetime.now(KST).isoformat(timespec="milliseconds"), "wall_s": round(time.time() - t_run, 3),
            "ckpt_step": args_cli.ckpt_step, "model": args_cli.model, "condition": args_cli.condition,
            "position_id": None, "position_mode": "leisaac_default_reset_randomization",
            "trial": ep, "seed": args_cli.seed, "success": success,
            "n_objects_placed": n_placed, "max_objects_placed": max_placed,  # 🔮 best-effort, see PlacedCounter
            "episode_len_steps": steps, "duration_s": round(dur, 2),
            "failure_tag": None, "video": f"ep{ep:03d}.mp4" if vw is not None else None,
            "notes": "timeout" if time_out else "",
            "action_horizon": horizon, "warmup_steps": args_cli.warmup_steps, "n_inferences": n_infer,
            "mean_infer_ms": round(1000 * infer_s / max(n_infer, 1), 1), "task": args_cli.task,
            "state0": [round(float(v), 2) for v in to_numpy(state0)],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        prog.emit("ep_end", ep=ep, success=success, timeout=time_out, steps=steps, duration_s=round(dur, 2),
                  successes=successes, n_infer=n_infer, infer_ms=rec["mean_infer_ms"],
                  placed=n_placed, max_placed=max_placed)
        print(f"[rollout] ep {ep}: success={success} steps={steps} placed={n_placed} (max {max_placed}) -> {successes}/{ep}")
        # terminated/timed-out envs are auto-reset by Isaac Lab; obs_dict is already the new episode's obs
    out.close()
    client.close()
    prog.emit("done", successes=successes, n_eps=args_cli.eval_rounds)
    print(f"[rollout] done: {successes}/{args_cli.eval_rounds} (report with Wilson CI, see docs/04)")
    env.close()


if __name__ == "__main__":
    _prog = Progress(args_cli.progress, cond=args_cli.condition)
    try:
        main(_prog)
    except Exception as e:
        _prog.emit("error", msg=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
        raise
    finally:
        _prog.close()
        simulation_app.close()
