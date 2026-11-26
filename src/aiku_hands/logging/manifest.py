"""Reproducibility manifest (docs/05_logging_spec.md §5.5)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .jsonl import now_iso


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], repo)
    status = _run(["git", "status", "--porcelain"], repo)
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None,
            "git_dirty_files": status.splitlines()[:50] if status else [],
            "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)}


def _ver(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def host_info() -> dict[str, Any]:
    mem = _run(["sysctl", "-n", "hw.memsize"])
    return {
        "hostname": platform.node(),
        "hw_model": _run(["sysctl", "-n", "hw.model"]),
        "cpu_brand": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mem_gb": round(int(mem) / 2**30, 1) if mem and mem.isdigit() else None,
        "os": f"{platform.system()} {platform.release()}",
        "macos": _run(["sw_vers", "-productVersion"]),
    }


def dataset_info(root: Path | None, repo_id: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"repo_id": repo_id, "root": str(root) if root else None}
    if root is None:
        return out
    info_p = Path(root) / "meta" / "info.json"
    out["info_json_sha256"] = sha256_file(info_p)
    out["stats_json_sha256"] = sha256_file(Path(root) / "meta" / "stats.json")
    if info_p.exists():
        info = json.loads(info_p.read_text())
        out.update(n_episodes=info.get("total_episodes"), n_frames=info.get("total_frames"),
                   fps=info.get("fps"), codebase_version=info.get("codebase_version"))
    return out


def build_manifest(*, exp_id: str, run_id: str, cmdline: list[str], repo: Path,
                   dataset_root: Path | None, dataset_repo_id: str | None, seed: int | None,
                   env_dir: Path, python: str = sys.executable) -> dict[str, Any]:
    git = git_info(repo)  # BEFORE writing anything into the repo (else our own files make it dirty)
    env_dir.mkdir(parents=True, exist_ok=True)
    freeze = _run([python, "-m", "pip", "freeze"]) or ""
    freeze_p = env_dir / "pip_freeze.txt"
    freeze_p.write_text(freeze + "\n")
    mps = _run([python, "-c", "import torch;print(torch.backends.mps.is_available())"])
    return {
        "schema_version": 1,
        "exp_id": exp_id,
        "run_id": run_id,
        "start_ts": now_iso(),
        **git,
        "cmdline": cmdline,
        "seed": seed,
        "python": _run([python, "-c", "import sys;print(sys.version.split()[0])"]),
        "python_executable": python,
        "torch": _ver("torch"),
        "lerobot": _ver("lerobot"),
        "mps_available": (mps == "True") if mps is not None else None,
        "pip_freeze": str(freeze_p.relative_to(repo)) if freeze_p.is_relative_to(repo) else str(freeze_p),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "host": host_info(),
        "dataset": dataset_info(dataset_root, dataset_repo_id),
    }
