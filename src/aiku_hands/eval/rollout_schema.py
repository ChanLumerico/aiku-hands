"""Schema for rollouts_sim.jsonl / rollouts_real.jsonl (docs/05_logging_spec.md, docs/04_evaluation_protocol.md)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FAILURE_TAGS = {
    "perception", "grasp_pos", "gripper_timing", "object_drop",
    "place_pos", "joint_limit_collision", "stall_jitter",
}
MODELS = {"A", "B", "C10", "C20", "C30"}
GRID_POSITIONS = range(1, 10)  # 3x3 grid, row-major, 1..9


@dataclass
class RolloutRecord:
    ckpt_step: int
    model: str
    condition: str
    position_id: int
    trial: int
    seed: int | None
    success: bool
    n_objects_placed: int | None = None
    episode_len_steps: int | None = None
    duration_s: float | None = None
    failure_tag: str | None = None
    video: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "RolloutRecord":
        errs = []
        if self.model not in MODELS:
            errs.append(f"model {self.model!r} not in {sorted(MODELS)}")
        if self.position_id not in GRID_POSITIONS:
            errs.append(f"position_id {self.position_id} not in 1..9")
        if self.trial < 1:
            errs.append("trial must be >= 1")
        if self.success and self.failure_tag is not None:
            errs.append("success=True must have failure_tag=None")
        if not self.success and self.failure_tag not in FAILURE_TAGS:
            errs.append(f"failed rollout needs failure_tag in {sorted(FAILURE_TAGS)}")
        if errs:
            raise ValueError("; ".join(errs))
        return self

    def to_record(self) -> dict[str, Any]:
        d = asdict(self.validate())
        extra = d.pop("extra")
        d.update(extra)
        return d
