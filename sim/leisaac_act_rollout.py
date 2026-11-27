"""Roll out a lerobot 0.7.0 ACT policy in LeIsaac (Isaac Sim 5.1 / Isaac Lab 2.3 / LeIsaac 0.4.0).

Runs in the **leisaac** conda env on the HANDS PC, from ~/sim_tools/ (never inside ~/lerobot_ws/).
The policy itself runs in a separate process: sim/act_policy_server.py (lerobot 0.7.0 venv).

    python leisaac_act_rollout.py --task LeIsaac-SO101-PickOrange-v0 --eval_rounds 10 \
        --policy_port 5765 --action_horizon 100 --seed 1000 --out rollouts_sim_pilot.jsonl \
        --ckpt_step 70000 --exp_id E001 --enable_cameras

Adapted from LeIsaac v0.4.0 scripts/evaluation/policy_inference.py (Apache-2.0), replacing the
lerobot-v0.3.3 gRPC client with sim/act_bridge.py. Env setup, obs keys, unit conversions and the
success/timeout logic follow that script; this file has NOT been run on the HANDS PC yet.
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

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
parser.add_argument("--out", type=str, default="rollouts_sim_pilot.jsonl")
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
import sys  # noqa: E402
import time  # noqa: E402
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from act_bridge import BridgeClient  # noqa: E402

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


def main():
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

    out = open(args_cli.out, "a", encoding="utf-8")
    t_run = time.time()
    successes = 0
    obs_dict, _ = env.reset()
    for ep in range(1, args_cli.eval_rounds + 1):
        client.reset()
        success = time_out = False
        steps, n_infer, infer_s = 0, 0, 0.0
        t_ep = time.time()
        while simulation_app.is_running() and not (success or time_out):
            with torch.inference_mode():
                pol = obs_dict["policy"]
                state = convert_leisaac_action_to_lerobot(pol["joint_pos"])[0]  # rad -> lerobot units
                images = {c: to_numpy(pol[c])[0].astype(np.uint8) for c in cams}  # (H,W,3)
                t0 = time.time()
                chunk = client.infer(state, images)  # (100, 6) lerobot units
                infer_s += time.time() - t0
                n_infer += 1
                actions = torch.from_numpy(convert_lerobot_action_to_leisaac(chunk)).float().to(env.device)
                for i in range(horizon):
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, terminated, time_outs, _ = env.step(actions[i : i + 1])
                    steps += 1
                    if terminated[0]:
                        success = True
                        break
                    if time_outs[0]:
                        time_out = True
                        break
                    rate.sleep(env)
        if not simulation_app.is_running():
            break
        successes += int(success)
        rec = {
            "schema_version": 1, "exp_id": args_cli.exp_id, "run_id": f"sim_{args_cli.exp_id}_{args_cli.ckpt_step}",
            "ts": datetime.now(KST).isoformat(timespec="milliseconds"), "wall_s": round(time.time() - t_run, 3),
            "ckpt_step": args_cli.ckpt_step, "model": args_cli.model, "condition": args_cli.condition,
            "position_id": None, "position_mode": "leisaac_default_reset_randomization",
            "trial": ep, "seed": args_cli.seed, "success": success, "n_objects_placed": None,
            "episode_len_steps": steps, "duration_s": round(time.time() - t_ep, 2),
            "failure_tag": None, "video": None, "notes": "timeout" if time_out else "",
            "action_horizon": horizon, "n_inferences": n_infer,
            "mean_infer_ms": round(1000 * infer_s / max(n_infer, 1), 1), "task": args_cli.task,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[rollout] ep {ep}: success={success} steps={steps} -> {successes}/{ep}")
        # terminated/timed-out envs are auto-reset by Isaac Lab; obs_dict is already the new episode's obs
    out.close()
    client.close()
    print(f"[rollout] done: {successes}/{args_cli.eval_rounds} (report with Wilson CI, see docs/04)")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
