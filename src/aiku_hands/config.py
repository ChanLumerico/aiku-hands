"""Train config = ordered list of lerobot-train CLI args (the single source of truth).

configs/train/<run>.yaml::

    exp_id: E001
    exp_dir: E001_act_orange_2cam_b8
    run_id: act_orange_2cam_b8
    base: _base_act.yaml          # optional; args merged by key, child wins
    entrypoint: lerobot-train
    args:
      - --dataset.repo_id=...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import REPO_ROOT


def _key(arg: str) -> str:
    return arg.split("=", 1)[0]


def merge_args(base: list[str], child: list[str]) -> list[str]:
    out: dict[str, str] = {}
    for a in base + child:
        out[_key(a)] = a
    return list(out.values())


@dataclass
class TrainConfig:
    exp_id: str
    exp_dir: str
    run_id: str
    entrypoint: str
    args: list[str]
    path: Path
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        k = key if key.startswith("--") else f"--{key}"
        for a in self.args:
            if _key(a) == k:
                return a.split("=", 1)[1] if "=" in a else "true"
        return default

    def resolved_args(self) -> list[str]:
        """Expand $HOME / ~ and make --output_dir absolute (relative paths = repo root)."""
        out = []
        for a in self.args:
            if "=" in a:
                k, v = a.split("=", 1)
                v = os.path.expanduser(os.path.expandvars(v))
                if k == "--output_dir" and not os.path.isabs(v):
                    v = str(REPO_ROOT / v)
                a = f"{k}={v}"
            out.append(a)
        return out

    @property
    def batch_size(self) -> int:
        return int(self.get("batch_size", "8"))

    @property
    def log_freq(self) -> int:
        return int(self.get("log_freq", "200"))  # lerobot 0.7.0 default

    @property
    def steps(self) -> int | None:
        s = self.get("steps")
        return int(s) if s else None

    @property
    def seed(self) -> int:
        return int(self.get("seed", "1000"))  # lerobot 0.7.0 default

    @property
    def output_dir(self) -> Path:
        for a in self.resolved_args():
            if _key(a) == "--output_dir":
                return Path(a.split("=", 1)[1])
        raise ValueError("config has no --output_dir")


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    args = list(d.get("args", []))
    if d.get("base"):
        base = yaml.safe_load((path.parent / d["base"]).read_text())
        args = merge_args(list(base.get("args", [])), args)
    known = {"exp_id", "exp_dir", "run_id", "entrypoint", "args", "base", "notes"}
    return TrainConfig(exp_id=d["exp_id"], exp_dir=d["exp_dir"], run_id=d["run_id"],
                       entrypoint=d.get("entrypoint", "lerobot-train"), args=args, path=path,
                       notes=d.get("notes", ""), extra={k: v for k, v in d.items() if k not in known})
