#!/usr/bin/env python
"""Serve a lerobot 0.7.0 ACT checkpoint to the LeIsaac sim over TCP (see sim/act_bridge.py).

Run in a Python >= 3.12 env with lerobot==0.7.0 (NOT the leisaac conda env):

    python sim/act_policy_server.py --ckpt ~/policies/E001/070000/pretrained_model --device cuda

Inference per request: one observation -> full action chunk (chunk_size x 6, lerobot units).
The sim decides how many of those actions to execute before asking again.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from act_bridge import DEFAULT_PORT, PROTOCOL_VERSION, listen, pack, unpack  # noqa: E402


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ActServer:
    def __init__(self, ckpt: Path, device: str):
        from lerobot import __version__ as lerobot_version
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.device = device  # checkpoint was trained on mps; override for this host
        self.policy = ACTPolicy.from_pretrained(ckpt, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device = device
        self.image_keys = [k for k, f in cfg.input_features.items() if f.type.name == "VISUAL"]
        self.image_shapes = {k: tuple(cfg.input_features[k].shape) for k in self.image_keys}  # (C,H,W)
        self.info = {
            "protocol": PROTOCOL_VERSION, "ckpt": str(ckpt), "lerobot": lerobot_version, "device": device,
            "image_keys": self.image_keys, "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
            "state_dim": cfg.input_features["observation.state"].shape[0],
        }

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @torch.no_grad()
    def infer(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        obs = {"observation.state": torch.from_numpy(state.astype(np.float32))}
        for key in self.image_keys:
            cam = key.split(".")[-1]  # observation.images.front -> front
            if cam not in images:
                raise KeyError(f"missing camera {cam!r}; got {sorted(images)}")
            img = images[cam]
            c, h, w = self.image_shapes[key]
            if img.shape != (h, w, c):
                raise ValueError(f"{cam}: expected HWC {(h, w, c)}, got {img.shape}")
            # training saw float32 CHW in [0,1] (decoded video); normalizer applies ImageNet mean/std
            obs[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        batch = self.pre(obs)
        chunk = self.policy.predict_action_chunk(batch)  # (1, chunk, 6), normalized
        chunk = self.post(chunk)  # unnormalized, cpu
        return chunk[0].float().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path, help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only on a trusted network (Tailscale)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    srv = ActServer(a.ckpt.expanduser(), pick_device(a.device))
    logging.info("loaded %s", srv.info)
    with listen(a.host, a.port) as lst:
        logging.info("listening on %s:%d", a.host, a.port)
        while True:
            conn = lst.accept()
            logging.info("client connected: %s", lst.last_accepted)
            n, t_sum = 0, 0.0
            try:
                while True:
                    msg = conn.recv()
                    cmd = msg.get("cmd")
                    try:
                        if cmd == "hello":
                            conn.send(srv.info)
                        elif cmd == "reset":
                            srv.reset()
                            conn.send({"ok": True})
                        elif cmd == "infer":
                            t0 = time.perf_counter()
                            act = srv.infer(unpack(msg["state"]), {k: unpack(v) for k, v in msg["images"].items()})
                            dt = (time.perf_counter() - t0) * 1000
                            n, t_sum = n + 1, t_sum + dt
                            conn.send({"actions": pack(act.astype(np.float32)), "infer_ms": dt})
                        elif cmd == "bye":
                            break
                        else:
                            conn.send({"error": f"unknown cmd {cmd!r}"})
                    except Exception as e:  # report to client, keep serving
                        logging.exception("request failed")
                        conn.send({"error": f"{type(e).__name__}: {e}"})
            except (EOFError, ConnectionResetError):
                pass
            finally:
                conn.close()
                logging.info("client disconnected (%d inferences, mean %.1f ms)", n, t_sum / max(n, 1))


if __name__ == "__main__":
    sys.exit(main())
