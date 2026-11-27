#!/usr/bin/env python3
"""Do the dataset's demos start/end inside LeIsaac's rest-pose box? (evidence for ADR-0004)

    ~/venvs/lerobot/bin/python scripts/analyze_dataset_rest_pose.py [--root ~/datasets/cohab_orange]

lerobot units -> degrees uses LeIsaac 0.4.0 `convert_leisaac_action_to_lerobot` inverted (linear map of the
motor range [-100,100] (gripper [0,100]) onto SO101_FOLLOWER_USD_JOINT_LIMLITS). Rest box =
SO101_FOLLOWER_REST_POSE_RANGE (center ±30°): pan 0, lift −100, elbow 90, wflex 50, wroll 0, gripper −10.
Both tables copied from ~/lerobot_ws/leisaac/source/leisaac/leisaac/assets/robots/lerobot.py (HANDS).
"""

from __future__ import annotations

import sys as _sys

_sys.dont_write_bytecode = True  # repo rule: never create __pycache__ (CLAUDE.md §3.9)

import argparse
import glob
import os

import numpy as np
import pandas as pd

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
USD_LIMITS = [(-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160), (-10, 100)]
MOTOR_LIMITS = [(-100, 100)] * 5 + [(0, 100)]
REST_CENTER = [0.0, -100.0, 90.0, 50.0, 0.0, -10.0]
REST_TOL = 30.0


def to_deg(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for j, ((a, b), (m0, m1)) in enumerate(zip(USD_LIMITS, MOTOR_LIMITS)):
        out[..., j] = (x[..., j] - m0) / (m1 - m0) * (b - a) + a
    return out


def in_rest(deg: np.ndarray) -> np.ndarray:
    c = np.asarray(REST_CENTER)
    return np.all((deg > c - REST_TOL) & (deg < c + REST_TOL), axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/datasets/cohab_orange"))
    a = ap.parse_args()
    fs = sorted(glob.glob(f"{a.root}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in fs])
    df = df.sort_values(["episode_index", "frame_index"])
    S, ep = np.stack(df["observation.state"].values), df["episode_index"].values
    eps = np.unique(ep)
    first = np.array([np.where(ep == e)[0][0] for e in eps])
    last = np.array([np.where(ep == e)[0][-1] for e in eps])
    np.set_printoptions(precision=1, suppress=True, linewidth=140)
    print(f"episodes {len(eps)}, frames {len(df)}; joints {JOINTS}")
    for nm, idx in (("first", first), ("last", last)):
        d = to_deg(S[idx])
        viol = {j: int((np.abs(d[:, i] - REST_CENTER[i]) >= REST_TOL).sum()) for i, j in enumerate(JOINTS)}
        print(f"{nm:>5} frame: median deg {np.median(d, 0)} | inside rest box {int(in_rest(d).sum())}/{len(eps)}"
              f" | per-joint violations {viol}")
    tail = []
    for e in eps:
        r = in_rest(to_deg(S[ep == e]))
        k = len(r) - 1
        while k >= 0 and r[k]:
            k -= 1
        tail.append(len(r) - 1 - k)
    print("final in-rest run (frames) pctl[0,10,50,90,100]:", np.percentile(tail, [0, 10, 50, 90, 100]))


if __name__ == "__main__":
    main()
