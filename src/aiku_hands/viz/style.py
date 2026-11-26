"""Shared matplotlib style: one look for every figure in the repo."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito (colour-blind safe), fixed order
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MODEL_COLORS = {"A": PALETTE[0], "B": PALETTE[1], "C": PALETTE[2]}
DPI = 300


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out_stem: Path) -> list[Path]:
    """Save PNG (300 dpi) + PDF next to each other; returns written paths."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".png"), out_stem.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths
