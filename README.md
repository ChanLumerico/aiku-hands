# aiku-hands — Sim-to-Real ACT on SO-101 with Domain Randomization

> **TL;DR (EN)**: AIKU × HANDS research repo. We train ACT purely on Isaac Lab (LeIsaac) simulation demos and
> deploy it on a real SO-101 arm, testing whether sim-side Domain Randomization improves real-world robustness.
> Every run is logged as append-only JSONL with a reproducibility manifest. Every report gives success rates
> with Wilson 95% CIs.

## 1. 목표 / Goal
- sim 데이터만으로 학습한 ACT가 실제 SO-101에서 pick-and-place를 수행하는지 확인함
- DR이 실제 환경 변화에 대한 robustness를 높이는지 검증함 → [docs/00_project_overview.md](docs/00_project_overview.md)

## 2. Research Questions
| ID | 질문 | 지표 |
|---|---|---|
| RQ1 | sim-only ACT가 실제 SO-101에서 동작하는가 | $S_{\text{sim}} - S_{\text{real}}$ |
| RQ2 | sim DR이 실제 성능·robustness를 높이는가 | 조건별 $S$, $S_{\text{normal}} - S_{\text{changed}}$ |
| RQ3 (선택) | real demo 10/20/30 ep 추가 효과 | fine-tune 비교 |

## 3. 현재 상태 / Status (2026-11-27)
| 영역 | 상태 |
|---|---|
| 학습 pipeline (Mac Studio, MPS) | ✅ 스모크 통과 ([E000](experiments/E000_act_orange_smoke/README.md)): 2.94 step/s, 70k ≈ 6.6 h |
| 로깅 v0 (`run_train.py`) | ✅ 구현, dry-run과 wrapper self-test 통과, 28 tests |
| E001 (ACT 2cam, 70k) | ✅ done (6.66 h): train L1 0.140, KL collapse 관찰 ([E001](experiments/E001_act_orange_2cam_b8/README.md)) |
| sim rollout (HANDS PC) | ⬜ 미착수 |
| real wrist camera | ⚠️ intrinsics 측정 완료, **mount offset 미측정** |
| real SO-101 | ⚠️ 모터 보드 USB `error -71`, 현장 수리 필요 |

## 4. 결과 요약 / Results
<!-- results:begin -->
| ID | date | status | model | dataset | steps | batch | cams | best metric | report |
|---|---|---|---|---|---|---|---|---|---|
| E000_act_orange_smoke | 2026-11-27 | done | ACT | cohab_orange | 200 | 8 | front+wrist | loss@200=3.545 (train) | [README](experiments/E000_act_orange_smoke/README.md) |
| E001_act_orange_2cam_b8 | 2026-11-27 | done | ACT | cohab_orange | 70000 | 8 | front+wrist | l1@70k=0.140 (train) | [README](experiments/E001_act_orange_2cam_b8/README.md) |
<!-- results:end -->
- 이 표는 `python scripts/make_figures.py --update-readme`로 [experiments/registry.csv](experiments/registry.csv)에서 다시 생성함

## 5. 빠른 시작 / Quick start (Mac Studio)
```bash
source ~/venvs/lerobot/bin/activate           # Python 3.12.16, lerobot 0.7.0
pip install -e '.[dev]' && pytest -q          # 28 tests

# 학습 (tmux 안에서 실행)
python scripts/run_train.py --exp E001 --config configs/train/act_orange_2cam_b8.yaml --dry-run
caffeinate -is python scripts/run_train.py --exp E001 --config configs/train/act_orange_2cam_b8.yaml

# 이미 있는 stdout 로그 → JSONL, figure
python scripts/backfill_from_log.py --config configs/train/<run>.yaml --log logs/<run>.log [--follow]
python scripts/make_figures.py E001_act_orange_2cam_b8

# 기타
python scripts/new_experiment.py E002_act_orange_wrist_b8 --purpose "..."
bash scripts/disk_guard.sh                     # 디스크 여유 확인 (삭제하지 않음)
python -m aiku_hands.data.inspect ~/datasets/cohab_orange
```

## 6. 디렉터리 / Layout
| 경로 | 내용 |
|---|---|
| `configs/train/` | lerobot-train 인자 목록 (YAML, 유일한 원천) |
| `experiments/<ID>/` | 실험별 README, config, manifest, `metrics/*.jsonl`, `figures/` |
| `src/aiku_hands/` | logging (jsonl, parser, ingest, sysmon, manifest), eval (wilson, rollout schema), viz, data |
| `scripts/` | run_train, backfill, make_figures, new_experiment, sync_ckpt_to_hands, disk_guard |
| `docs/` | 환경·데이터·평가·로깅·규칙 문서, camera, ADR |
| `reports/` | 주간 보고 (HANDS 미팅), 마일스톤 |
| `runs/ logs/ data/` | git 제외 (checkpoint, raw stdout, dataset) |

## 7. 문서 / Docs
- [00 개요](docs/00_project_overview.md) · [01 Mac Studio](docs/01_setup_macstudio.md) · [02 HANDS PC](docs/02_setup_hands_pc.md) · [03 데이터](docs/03_data.md)
- [04 평가 프로토콜](docs/04_evaluation_protocol.md) · [05 로깅 사양](docs/05_logging_spec.md) · [06 규칙](docs/06_conventions.md)
- 카메라: [intrinsics](docs/camera/wrist_intrinsics.md) · [wrist offset (미해결)](docs/camera/wrist_offset.md)
- 결정 기록: [ADR-0001](docs/decisions/ADR-0001-sim-to-real-act-dr.md) · [ADR-0002](docs/decisions/ADR-0002-logging-v0-wrapper.md) · [ADR-0003](docs/decisions/ADR-0003-lerobot070-investigation.md)
