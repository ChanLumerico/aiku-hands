"""Wilson score interval for a binomial proportion (default 95%)."""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (p_hat, lo, hi). n == 0 -> (nan, 0.0, 1.0).

    $\\tilde p = \\frac{\\hat p + z^2/2n}{1 + z^2/n}$, half-width $\\frac{z}{1+z^2/n}\\sqrt{\\hat p(1-\\hat p)/n + z^2/4n^2}$.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid k={k}, n={n}")
    if n == 0:
        return float("nan"), 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def fmt_rate(k: int, n: int, conf: float = 0.95) -> str:
    """'0.667 [0.354, 0.879] (n=9)' -- the only allowed way to report a success rate."""
    p, lo, hi = wilson_interval(k, n, conf)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"
