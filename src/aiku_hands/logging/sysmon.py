"""System monitor thread -> system.jsonl (every `interval_s`, default 30 s).

v0 (out-of-process): psutil metrics of the training PID *and its children* (DataLoader workers).
MPS allocator stats (torch.mps.*) are only readable in-process -> null in v0, filled by v1 hooks.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

import psutil

from .jsonl import JsonlWriter

DISK_WARN_GB = 5.0


def thermal_state() -> str | None:
    """macOS `pmset -g therm`: 'nominal' if no warning recorded, else the raw warning text."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if all("No " in l and "recorded" in l for l in lines):
        return "nominal"
    return "; ".join(lines)[:200] or None


def disk_free_gb(path: str | Path = Path.home()) -> float:
    return round(shutil.disk_usage(path).free / 1e9, 2)


class SysMon(threading.Thread):
    def __init__(self, writer: JsonlWriter, pid: int | None, interval_s: float = 30.0,
                 on_disk_low: Callable[[float], None] | None = None, disk_path: Path = Path.home()):
        super().__init__(daemon=True, name="sysmon")
        self.w = writer
        self.pid = pid
        self.interval_s = interval_s
        self.on_disk_low = on_disk_low
        self.disk_path = disk_path
        self._stop_evt = threading.Event()
        self._procs: dict[int, psutil.Process] = {}
        self._warned = False

    def _tree(self) -> list[psutil.Process]:
        if self.pid is None:
            return []
        try:
            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            return []
        # keep Process objects so cpu_percent() has a previous sample to diff against
        out = []
        for p in procs:
            out.append(self._procs.setdefault(p.pid, p))
        return out

    def sample(self) -> dict:
        procs = self._tree()
        cpu = rss = 0.0
        n = 0
        for p in procs:
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
                n += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        vm = psutil.virtual_memory()
        free = disk_free_gb(self.disk_path)
        rec = {
            "pid": self.pid,
            "n_procs": n,
            "cpu_pct": round(cpu, 1),
            "rss_mb": round(rss / 2**20, 1),
            "sys_mem_used_pct": vm.percent,
            "swap_used_mb": round(psutil.swap_memory().used / 2**20, 1),
            "mps_alloc_mb": None,   # v1 (in-process) only
            "mps_driver_mb": None,  # v1 (in-process) only
            "disk_free_gb": free,
            "thermal_state": thermal_state(),
        }
        if free < DISK_WARN_GB and self.on_disk_low and not self._warned:
            self._warned = True
            self.on_disk_low(free)
        return rec

    def run(self) -> None:
        self._tree()  # prime cpu_percent
        for p in self._procs.values():
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        while not self._stop_evt.wait(self.interval_s):
            self.w.write(self.sample())

    def stop(self) -> None:
        self._stop_evt.set()
