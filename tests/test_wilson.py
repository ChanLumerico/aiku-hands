import math

import pytest

from aiku_hands.eval.wilson import fmt_rate, wilson_interval


def test_known_values():
    # reference: k=6, n=9 -> [0.3542, 0.8794] (Wilson 95%)
    p, lo, hi = wilson_interval(6, 9)
    assert p == pytest.approx(2 / 3)
    assert lo == pytest.approx(0.3542, abs=1e-3) and hi == pytest.approx(0.8794, abs=1e-3)


def test_edges():
    p, lo, hi = wilson_interval(0, 18)
    assert p == 0 and lo == 0 and 0 < hi < 0.2
    p, lo, hi = wilson_interval(50, 50)
    assert hi == 1 and 0.9 < lo < 1
    p, lo, hi = wilson_interval(0, 0)
    assert math.isnan(p) and (lo, hi) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_fmt():
    assert fmt_rate(6, 9).endswith("(n=9)")
