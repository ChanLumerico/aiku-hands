"""Tiny policy<->sim bridge protocol (stdlib + numpy only; works in both Python 3.11 and 3.12 envs).

Why: LeIsaac v0.4.0's LeRobot client speaks lerobot v0.3.3's gRPC/pickle protocol, while our checkpoints are
lerobot 0.7.0 (Zenoh server, processor-based normalization, Python >= 3.12). So the policy runs in its own
process (lerobot 0.7.0 venv) and the sim (leisaac conda env) talks to it over TCP.

Messages are dicts of builtins only. numpy arrays are sent as raw bytes + dtype + shape, because numpy 1.x
and 2.x pickles are not interchangeable.
"""

from __future__ import annotations

import os
from multiprocessing.connection import Client, Listener
from typing import Any

import numpy as np

DEFAULT_PORT = 5765
PROTOCOL_VERSION = 1


def authkey() -> bytes:
    return os.environ.get("AIKU_BRIDGE_KEY", "aiku-hands-bridge").encode()


def pack(a: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tobytes()}


def unpack(d: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def listen(host: str, port: int) -> Listener:
    return Listener((host, port), authkey=authkey())


class BridgeClient:
    """Used inside the sim process."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.conn = Client((host, port), authkey=authkey())
        self.info = self.call({"cmd": "hello", "protocol": PROTOCOL_VERSION})

    def call(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.conn.send(msg)
        rep = self.conn.recv()
        if rep.get("error"):
            raise RuntimeError(f"policy server error: {rep['error']}")
        return rep

    def reset(self) -> None:
        self.call({"cmd": "reset"})

    def infer(self, state_lerobot: np.ndarray, images_hwc_uint8: dict[str, np.ndarray]) -> np.ndarray:
        """state: (6,) in lerobot units (-100..100, gripper 0..100). images: {cam: (H,W,3) uint8}.
        Returns action chunk (chunk_size, 6) in lerobot units."""
        rep = self.call({
            "cmd": "infer",
            "state": pack(np.asarray(state_lerobot, dtype=np.float32).reshape(-1)),
            "images": {k: pack(np.asarray(v, dtype=np.uint8)) for k, v in images_hwc_uint8.items()},
        })
        return unpack(rep["actions"])

    def close(self) -> None:
        try:
            self.conn.send({"cmd": "bye"})
        except Exception:
            pass
        self.conn.close()
