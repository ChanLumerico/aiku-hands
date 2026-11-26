"""Parse lerobot 0.7.0 ``lerobot-train`` stdout lines into dicts.

Line format (measured, 2026-11-27)::

    INFO 2026-11-27 01:45:31 ot_train.py:894 step:20 smpl:160 ep:0 epch:0.00 loss:25.027 grdn:444.021 \
lr:1.0e-05 data_s:0.149 prep_s:0.019 updt_s:0.533 step_s:0.701 smp/s:11 l1_loss:0.827 kld_loss:2.420

Notes (verified in lerobot/utils/logging_utils.py::MetricsTracker.__str__):
- ``step``, ``smpl`` and ``ep`` go through ``format_big_number(precision=0)`` -> ``14K`` etc.
  So ``step`` is *not exact* above 1000. ``aiku_hands.logging.ingest`` reconstructs it from log_freq.
- tqdm output can be glued to the INFO line (``...2.93step/s]INFO 2026-...``), so we anchor on the
  `` step:`` token, not on the line start.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .jsonl import TZ

ALIASES = {
    "grdn": "grad_norm",
    "updt_s": "update_s",
    "smp/s": "samples_per_s",
    "epch": "epoch",
    "ep": "episodes_seen",
    "smpl": "samples_reported",
    "step": "step_reported",
}
# keys rendered with format_big_number (lossy)
BIG_NUMBER_KEYS = {"step", "smpl", "ep"}
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}

_STEP_ANCHOR = re.compile(r"(?:^|(?<=[\s\]]))step:\S")
_TOKEN = re.compile(r"(\S+?):(\S+)")
_LOG_HEAD = re.compile(r"(INFO|WARNING|ERROR|DEBUG|CRITICAL) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+?):(\d+) ")
_CKPT = re.compile(r"Checkpoint policy after step (\d+)")
_EVAL = re.compile(r"step (\d+): eval_loss=(\S+)")
_END = re.compile(r"End of training")
_META = {
    "total_steps": re.compile(r"cfg\.steps=(\d+)"),
    "num_frames": re.compile(r"dataset\.num_frames=(\d+)"),
    "num_episodes": re.compile(r"dataset\.num_episodes=(\d+)"),
    "num_learnable_params": re.compile(r"num_learnable_params=(\d+)"),
    "num_total_params": re.compile(r"num_total_params=(\d+)"),
    "effective_batch_size": re.compile(r"Effective batch size: .*?= (\d+)"),
}
_SPLIT = re.compile(r"Train/eval split: (\d+) train, (\d+) eval")
_IGNORE = (
    re.compile(r"^objc\["),
    re.compile(r"redirects\.py:\d+\] NOTE: Redirects"),
    re.compile(r"^\s*Training:\s+\d+%\|"),  # bare tqdm frame
)
_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):")


def parse_number(s: str) -> float | int | None:
    """'160' -> 160, '1K' -> 1000, '1.0e-05' -> 1e-05, 'nan' -> nan; None if not numeric."""
    s = s.strip().rstrip(",")
    mult = 1.0
    if s and s[-1] in SUFFIX and s[:-1].replace(".", "", 1).lstrip("-").isdigit():
        mult = SUFFIX[s[-1]]
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    v *= mult
    if v.is_integer() and "e" not in s.lower() and "." not in s:
        return int(v)
    return v


def format_big_number(num: float, precision: int = 0) -> str:
    """Exact replica of lerobot.utils.utils.format_big_number (used to check step reconstruction)."""
    for suffix in ["", "K", "M", "B", "T", "Q"]:
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{suffix}"
        num /= 1000.0
    return str(num)


def parse_log_ts(line: str) -> tuple[str | None, float | None]:
    """Return (iso_ts, epoch_seconds) of the *last* lerobot log header in the line."""
    heads = list(_LOG_HEAD.finditer(line))
    if not heads:
        return None, None
    dt = datetime.strptime(heads[-1].group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
    return dt.isoformat(), dt.timestamp()


def parse_train_metrics(line: str) -> dict[str, Any] | None:
    """Parse the ``step:... smpl:...`` part. Returns None if the line is not a metrics line.

    Raises ValueError if it *looks* like a metrics line but cannot be parsed.
    """
    m = _STEP_ANCHOR.search(line)
    if not m:
        return None
    out: dict[str, Any] = {}
    for key, raw in _TOKEN.findall(line[m.start():]):
        name = ALIASES.get(key, key)
        val = parse_number(raw)
        if val is None:
            raise ValueError(f"non-numeric value for {key!r}: {raw!r}")
        out[name] = val
        if key in BIG_NUMBER_KEYS:
            out[f"{name}_raw"] = raw
    if "step_reported" not in out or "loss" not in out:
        raise ValueError("metrics line without step/loss")
    return out


def classify(line: str) -> tuple[str, dict[str, Any]]:
    """Classify one stdout line.

    Returns (kind, payload) with kind in:
      train | ckpt | eval_loss | end | meta | split | traceback | warning | ignore | other
    Raises ValueError for metrics-looking lines that fail to parse (caller logs parse_error).
    """
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return "ignore", {}
    iso, epoch = parse_log_ts(stripped)
    base = {"log_ts": iso, "log_epoch": epoch}

    # metrics first: tqdm frames may be glued in front of an INFO line on the same line
    metrics = parse_train_metrics(stripped)
    if metrics is not None:
        return "train", {**base, **metrics}
    if iso is None:  # pure noise lines only when no lerobot log header is present
        for pat in _IGNORE:
            if pat.search(stripped):
                return "ignore", {}
    elif _IGNORE[1].search(stripped):
        return "ignore", {}
    if (m := _CKPT.search(stripped)):
        return "ckpt", {**base, "step": int(m.group(1))}
    if (m := _EVAL.search(stripped)):
        return "eval_loss", {**base, "step": int(m.group(1)), "eval_loss": parse_number(m.group(2))}
    if _END.search(stripped):
        return "end", base
    if (m := _SPLIT.search(stripped)):
        return "split", {**base, "n_train_episodes": int(m.group(1)), "n_eval_episodes": int(m.group(2))}
    for key, pat in _META.items():
        if (m := pat.search(stripped)):
            return "meta", {**base, key: int(m.group(1))}
    if _TRACEBACK.search(stripped):
        return "traceback", base
    if stripped.startswith(("WARNING", "ERROR", "CRITICAL")):
        return "warning", {**base, "raw": stripped[:2000]}
    return "other", base


def split_stream(text: str) -> list[str]:
    """tqdm uses '\\r'; treat both '\\r' and '\\n' as line separators."""
    return [s for s in re.split(r"[\r\n]", text) if s]
