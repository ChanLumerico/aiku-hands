#!/usr/bin/env python
"""Create experiments/E###_slug/ with README template + registry row (status=planned).

    python scripts/new_experiment.py E002_act_orange_wrist_b8 --purpose "wrist-only + state, held-out 90/10" \
        --model ACT --dataset cohab_orange --cams wrist --steps 70000 --batch 8
"""

from __future__ import annotations

import argparse
import re
import sys

from aiku_hands import REPO_ROOT, registry
from aiku_hands.logging.jsonl import now_iso

TEMPLATE = """# {exp_id} · {title}
> TL;DR (EN): (planned) {purpose}

| 항목 | 값 |
|---|---|
| 상태 / Status | planned |
| 목적 / Purpose | {purpose} |
| 데이터 | {dataset} |
| 모델 | {model}, cams={cams} |
| 하드웨어 | Mac Studio M4 Max 36GB, MPS fp32 |
| 시간 | start – / end – / wall – |
| 커밋 | – |
| Config | [configs/train/{run_id}.yaml](../../configs/train/{run_id}.yaml) |

## 1. 설정 / Setup
## 2. 학습 dynamics / Training dynamics
## 3. 평가 / Evaluation
## 4. 관찰 / Observations
## 5. 한계 · 주의 / Caveats
## 6. 다음 단계 / Next
## 7. 재현 / Reproduce
```bash
caffeinate -is python scripts/run_train.py --exp {exp_id} --config configs/train/{run_id}.yaml
```

## Open questions / 미해결
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="E###_slug")
    ap.add_argument("--purpose", default="")
    ap.add_argument("--model", default="ACT")
    ap.add_argument("--dataset", default="cohab_orange")
    ap.add_argument("--cams", default="")
    ap.add_argument("--steps", default="")
    ap.add_argument("--batch", default="")
    a = ap.parse_args()
    if not re.fullmatch(r"E\d{3}_[a-z0-9_]+", a.name):
        print("name must look like E002_act_orange_wrist_b8", file=sys.stderr)
        return 2
    d = REPO_ROOT / "experiments" / a.name
    if d.exists():
        print(f"{d} exists", file=sys.stderr)
        return 2
    exp_id, run_id = a.name.split("_", 1)
    (d / "metrics").mkdir(parents=True)
    (d / "figures").mkdir()
    (d / "README.md").write_text(TEMPLATE.format(exp_id=exp_id, title=run_id.replace("_", " "), run_id=run_id,
                                                 purpose=a.purpose, dataset=a.dataset, model=a.model, cams=a.cams))
    registry.upsert({"id": a.name, "date": now_iso()[:10], "status": "planned", "model": a.model,
                     "dataset": a.dataset, "steps": a.steps, "batch": a.batch, "cams": a.cams,
                     "best_metric": "", "report": f"experiments/{a.name}/README.md"})
    print(f"created {d.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
