"""experiments/registry.csv helpers (one row per experiment)."""

from __future__ import annotations

import csv
from pathlib import Path

from . import REPO_ROOT

FIELDS = ["id", "date", "status", "model", "dataset", "steps", "batch", "cams", "best_metric", "report"]
STATUSES = {"planned", "running", "done", "failed", "aborted"}
REGISTRY = REPO_ROOT / "experiments" / "registry.csv"


def read(path: Path = REGISTRY) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write(rows: list[dict[str, str]], path: Path = REGISTRY) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def upsert(row: dict[str, str], path: Path = REGISTRY) -> None:
    if row.get("status") and row["status"] not in STATUSES:
        raise ValueError(f"bad status {row['status']!r}; expected one of {sorted(STATUSES)}")
    rows = read(path)
    for r in rows:
        if r["id"] == row["id"]:
            r.update({k: v for k, v in row.items() if v is not None})
            break
    else:
        rows.append(row)
    write(rows, path)
