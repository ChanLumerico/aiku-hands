"""Turn a stream of lerobot stdout lines into train.jsonl / events.jsonl / offline_eval.jsonl records.

Shared by ``scripts/run_train.py`` (live) and ``scripts/backfill_from_log.py`` (offline).

Step reconstruction: lerobot prints ``step`` via format_big_number (``14K``), so above 1000 the
printed value is lossy. We infer ``step = prev_step + log_freq`` and accept it only if
``format_big_number(inferred) == printed``; otherwise we search a few multiples ahead (lost lines),
and as a last resort keep the rounded value with ``step_exact=false``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from .jsonl import JsonlWriter, now_iso
from .lerobot_log_parser import classify, format_big_number

CKPT_DIRNAME = "checkpoints"


def step_identifier(step: int, total_steps: int) -> str:
    """Replica of lerobot.common.train_utils.get_step_identifier."""
    return f"{step:0{max(6, len(str(total_steps)))}d}"


def dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return round(total / 2**20, 2)


class TrainLogIngestor:
    def __init__(
        self,
        train_w: JsonlWriter,
        events_w: JsonlWriter,
        eval_w: JsonlWriter | None,
        batch_size: int,
        log_freq: int,
        total_steps: int | None = None,
        run_dir: Path | None = None,
        use_log_time: bool = False,
    ):
        self.train_w, self.events_w, self.eval_w = train_w, events_w, eval_w
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.total_steps = total_steps
        self.run_dir = Path(run_dir) if run_dir else None
        self.use_log_time = use_log_time  # backfill: ts/wall_s come from the log header
        self.first_log_epoch: float | None = None
        self.last_step: int | None = None
        self.last_log_epoch: float | None = None
        self.num_frames: int | None = None
        self.num_episodes: int | None = None
        self.pending_ckpt: dict[str, Any] | None = None
        self.tail: deque[str] = deque(maxlen=200)
        self.traceback: list[str] | None = None
        self.counts = {"train": 0, "ckpt": 0, "eval_loss": 0, "parse_error": 0, "warning": 0, "step_inexact": 0}
        self.saw_end = False

    # ------------------------------------------------------------------ helpers
    def _time_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        epoch = payload.get("log_epoch")
        if epoch is not None and self.first_log_epoch is None:
            self.first_log_epoch = epoch
        if self.use_log_time and epoch is not None:
            return {"ts": now_iso(epoch), "wall_s": round(epoch - self.first_log_epoch, 3)}
        return {}

    def _event(self, etype: str, payload: dict[str, Any] | None = None, time_fields: dict | None = None) -> None:
        rec = {"type": etype, **(time_fields or {}), **(payload or {})}
        self.events_w.write(rec)

    def _resolve_step(self, reported: int, raw: str, log_epoch: float | None = None,
                      step_s: float | None = None) -> tuple[int, bool]:
        if reported < 1000 and raw.isdigit():
            return int(raw), True
        if self.last_step is not None:
            # format_big_number uses banker's rounding, so e.g. 1500/2000/2500 all print '2K'.
            # Labels alone cannot detect a lost line -> order candidates by a timing hint:
            # k_hint = elapsed_s / (log_freq * step_s).
            ks = list(range(1, 11))
            if log_epoch is not None and self.last_log_epoch is not None and step_s:
                k_hint = (log_epoch - self.last_log_epoch) / (self.log_freq * step_s)
                ks.sort(key=lambda k: (abs(k - k_hint), k))
            for k in ks:
                cand = self.last_step + k * self.log_freq
                if format_big_number(cand) == raw:
                    return cand, True
        # no anchor: pick the multiple of log_freq nearest to the rounded value
        cand = int(round(reported / self.log_freq) * self.log_freq)
        return cand, False

    def _flush_ckpt(self) -> None:
        if not self.pending_ckpt:
            return
        ck = self.pending_ckpt
        self.pending_ckpt = None
        path = Path(ck["path"]) if ck.get("path") else None
        size = dir_size_mb(path) if path else None
        self._event("ckpt_saved", {"step": ck["step"], "path": str(path) if path else None,
                                   "size_mb": size}, ck["time"])
        self.counts["ckpt"] += 1

    # ------------------------------------------------------------------ main
    def feed(self, line: str) -> str:
        self.tail.append(line)
        if self.traceback is not None:
            self.traceback.append(line)
        try:
            kind, p = classify(line)
        except ValueError as e:
            self.counts["parse_error"] += 1
            self._event("parse_error", {"raw": line[:4000], "error": str(e)})
            return "parse_error"

        tf = self._time_fields(p)
        if kind == "train":
            self._flush_ckpt()
            step, exact = self._resolve_step(int(p["step_reported"]), str(p["step_reported_raw"]),
                                             p.get("log_epoch"), p.get("step_s"))
            if not exact:
                self.counts["step_inexact"] += 1
            self.last_step = step
            self.last_log_epoch = p.get("log_epoch")
            samples = step * self.batch_size
            rec: dict[str, Any] = {**tf, "step": step, "step_exact": exact, "samples": samples}
            if self.num_frames:
                rec["epoch_exact"] = round(samples / self.num_frames, 6)
            for k in ("episodes_seen", "epoch", "loss", "l1_loss", "kld_loss", "grad_norm", "lr",
                      "data_s", "prep_s", "update_s", "step_s", "samples_per_s"):
                if k in p:
                    rec[k] = p[k]
            # any extra metric lerobot adds later is kept, not dropped
            known = set(rec) | {"log_ts", "log_epoch", "step_reported", "step_reported_raw"}
            for k, v in p.items():
                if k not in known:
                    rec[k] = v
            rec["log_ts"] = p.get("log_ts")
            self.train_w.write(rec)
            self.counts["train"] += 1
        elif kind == "ckpt":
            self._flush_ckpt()
            path = None
            if self.run_dir is not None:
                total = self.total_steps or p["step"]
                path = self.run_dir / CKPT_DIRNAME / step_identifier(p["step"], total)
            self.pending_ckpt = {"step": p["step"], "path": path, "time": tf}
            if self.last_step is not None and p["step"] != self.last_step and p["step"] % self.log_freq == 0:
                self.last_step = p["step"]  # exact anchor
        elif kind == "eval_loss":
            if self.eval_w is not None:
                self.eval_w.write({**tf, "source": "lerobot_eval_loss", "split": "heldout",
                                   "step": p["step"], "eval_loss": p["eval_loss"]})
            self.counts["eval_loss"] += 1
        elif kind == "meta":
            if "num_frames" in p:
                self.num_frames = p["num_frames"]
            if "num_episodes" in p:
                self.num_episodes = p["num_episodes"]
            if "total_steps" in p and not self.total_steps:
                self.total_steps = p["total_steps"]
            self._event("run_info", {k: v for k, v in p.items() if k not in ("log_ts", "log_epoch")}, tf)
        elif kind == "split":
            self._event("run_info", {k: v for k, v in p.items() if k not in ("log_ts", "log_epoch")}, tf)
        elif kind == "end":
            self._flush_ckpt()
            self.saw_end = True
            self._event("training_end_logged", {"last_step": self.last_step}, tf)
        elif kind == "traceback":
            self.traceback = [line]
        elif kind == "warning":
            self.counts["warning"] += 1
            self._event("log_warning", {"raw": p["raw"]}, tf)
        return kind

    def finalize(self, exit_code: int | None = None) -> None:
        self._flush_ckpt()
        if self.traceback is not None or (exit_code not in (None, 0)):
            lines = self.traceback if self.traceback is not None else list(self.tail)
            self._event("error", {"exit_code": exit_code, "traceback_tail": lines[-30:]})


class LineBuffer:
    """Incremental splitter on '\\r' and '\\n' for byte/str chunks (tqdm writes '\\r')."""

    def __init__(self) -> None:
        self.buf = ""

    def push(self, chunk: str) -> list[str]:
        self.buf += chunk
        out: list[str] = []
        start = 0
        for i, ch in enumerate(self.buf):
            if ch in "\r\n":
                if i > start:
                    out.append(self.buf[start:i])
                start = i + 1
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        return [rest] if rest else []
