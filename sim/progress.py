"""Progress events shared by the sim rollout and sim/eval.py (stdlib only; Python >= 3.8).

The rollout appends one JSON object per line to a progress file; sim/eval.py tails that file to draw
progress bars (locally or over ssh). Events are UI telemetry, not research data: the research record is
the per-episode rollouts JSONL (docs/05_logging_spec.md).

Event names (field `ev`):
  orchestrator: run_start, server_start, server_ready, sim_start, cond_done, run_end, note
  rollout     : start, ep_start, step (throttled), ep_end, done, error
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional


class Progress:
    def __init__(self, path: Optional[str], **base: Any):
        self.f = open(path, "a", encoding="utf-8", buffering=1) if path else None
        self.base = base
        self._last = 0.0

    def emit(self, ev: str, **kw: Any) -> None:
        if self.f is None:
            return
        rec = {"ev": ev, "t": round(time.time(), 3)}
        rec.update(self.base)
        rec.update(kw)
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()

    def tick(self, every_s: float = 0.5, **kw: Any) -> None:
        """Throttled `step` event."""
        now = time.time()
        if now - self._last >= every_s:
            self._last = now
            self.emit("step", **kw)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None
