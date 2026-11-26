"""Figure builders. Every function takes raw records (from JSONL) and returns a Figure."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ..eval.wilson import wilson_interval
from . import style


def _xlabel(ax, df: pd.DataFrame) -> None:
    ax.set_xlabel("step")


def _caption(fig, exp_id: str, df: pd.DataFrame) -> None:
    n = len(df)
    inexact = int((~df["step_exact"]).sum()) if "step_exact" in df else 0
    txt = f"{exp_id} · n_log_points={n}"
    if inexact:
        txt += f" · step_inexact={inexact}"
    fig.text(0.99, 0.005, txt, ha="right", va="bottom", fontsize=6, color="0.5")


def loss_curves(df: pd.DataFrame, exp_id: str):
    keys = [k for k in ("loss", "l1_loss", "kld_loss") if k in df]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot(df["step"], df[k], marker="o" if len(df) < 40 else None, ms=2.5)
        ax.set_yscale("log")
        ax.set_title(k)
        _xlabel(ax, df)
    fig.suptitle(f"{exp_id} · training loss (log y; per-log_freq window average)", fontsize=9)
    _caption(fig, exp_id, df)
    fig.tight_layout()
    return fig


def grad_lr(df: pd.DataFrame, exp_id: str):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    if "grad_norm" in df:
        axes[0].plot(df["step"], df["grad_norm"], color=style.PALETTE[1])
        axes[0].set_yscale("log")
        axes[0].set_title("grad_norm (pre-clip, log y)")
    if "lr" in df:
        axes[1].plot(df["step"], df["lr"], color=style.PALETTE[2])
        axes[1].set_title("learning rate")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        _xlabel(ax, df)
    _caption(fig, exp_id, df)
    fig.tight_layout()
    return fig


def throughput(df: pd.DataFrame, exp_id: str):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for k, c in (("step_s", 0), ("update_s", 1), ("data_s", 2)):
        if k in df:
            axes[0].plot(df["step"], df[k], label=k, color=style.PALETTE[c])
    axes[0].set_title("time per step [s]")
    axes[0].legend(frameon=False)
    if "samples_per_s" in df:
        axes[1].plot(df["step"], df["samples_per_s"], color=style.PALETTE[3])
    axes[1].set_title("samples/s (lerobot, rounded)")
    for ax in axes:
        _xlabel(ax, df)
    _caption(fig, exp_id, df)
    fig.tight_layout()
    return fig


def system(df: pd.DataFrame, exp_id: str):
    cols = [("cpu_pct", "CPU [% of 1 core]"), ("rss_mb", "RSS proc tree [MB]"),
            ("sys_mem_used_pct", "system mem used [%]"), ("disk_free_gb", "disk free [GB]")]
    cols = [c for c in cols if c[0] in df and df[c[0]].notna().any()]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 2.4), squeeze=False)
    for ax, (k, t) in zip(axes[0], cols):
        ax.plot(df["wall_s"] / 3600, df[k])
        ax.set_title(t)
        ax.set_xlabel("wall [h]")
    fig.tight_layout()
    return fig


def success_bars(labels: Sequence[str], ks: Sequence[int], ns: Sequence[int], title: str = ""):
    """Success rate bars with Wilson 95% CI and n annotated (never report a rate without both)."""
    fig, ax = plt.subplots(figsize=(max(3.0, 0.8 * len(labels)), 2.8))
    ps, los, his = zip(*(wilson_interval(k, n) for k, n in zip(ks, ns)))
    x = range(len(labels))
    ax.bar(x, ps, color=style.PALETTE[: len(labels)])
    ax.errorbar(x, ps, yerr=[[p - l for p, l in zip(ps, los)], [h - p for p, h in zip(ps, his)]],
                fmt="none", ecolor="k", capsize=3, lw=1)
    for i, (k, n) in enumerate(zip(ks, ns)):
        ax.text(i, 1.02, f"{k}/{n}", ha="center", fontsize=7)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
