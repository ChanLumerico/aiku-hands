"""Append-only JSON Lines writer with the common fields required by docs/05_logging_spec.md."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

SCHEMA_VERSION = 1
TZ = ZoneInfo("Asia/Seoul")


def now_iso(ts: float | None = None) -> str:
    """ISO-8601 timestamp with Asia/Seoul offset (e.g. 2026-11-27T01:45:31.123+09:00)."""
    dt = datetime.fromtimestamp(time.time() if ts is None else ts, tz=TZ)
    return dt.isoformat(timespec="milliseconds")


def _clean(v: Any) -> Any:
    """Make values JSON-safe: NaN/inf -> None (strict JSON has no NaN)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


class JsonlWriter:
    """Append-only writer. Every line is flushed (and fsync'ed) immediately.

    Common fields (schema_version, exp_id, run_id, ts, wall_s) are injected unless the
    caller already provides them (backfill passes the original log timestamp as ``ts``/``wall_s``).
    """

    def __init__(self, path: str | Path, exp_id: str, run_id: str, t0: float | None = None,
                 fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.exp_id = exp_id
        self.run_id = run_id
        self.t0 = time.time() if t0 is None else t0
        self.fsync = fsync
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        base = {
            "schema_version": SCHEMA_VERSION,
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "ts": now_iso(now),
            "wall_s": round(now - self.t0, 3),
        }
        base.update(record)
        line = json.dumps(_clean(base), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
        return base

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records; tolerates a truncated last line (e.g. read while the writer is live)."""
    p = Path(path)
    opener = open
    if p.suffix == ".gz":
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
