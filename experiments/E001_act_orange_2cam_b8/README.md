# E001 · ACT on cohab orange (2 cams, bs=8, 70k)
> TL;DR (EN): First full ACT run on the cohab orange sim dataset (front+wrist, bs=8, 70k steps, all 100 episodes).
> **Status: running** — launched 2026-11-27 02:14:37 KST via `scripts/run_train.py` in tmux `e001`
> (the bootstrap note assumed it had started at 01:50 in tmux `act`, but it had not).
> Expected wall time ≈ 6.6 h based on E000 throughput → ETA ≈ 08:55 KST.

| 항목 | 값 |
|---|---|
| 상태 / Status | running (tmux `e001`) |
| 목적 / Purpose | Model A 후보 — sim pipeline 검증 (M1 사전 단계), 이후 sim rollout(RQ1의 $S_{\text{sim}}$) |
| 데이터 | edge-inference/cohab-so101-pick-orange-dataset (100 ep 전체, teleop, Isaac Sim 5.1) |
| 모델 | ACT 52M, cams=front+wrist, chunk=100, n_action_steps=100, kl_weight=10, lr=1e-5 (기본값) |
| 하드웨어 | Mac Studio M4 Max 36GB, MPS fp32 |
| 시간 | start 2026-11-27 02:14:37 KST / end – / wall 예상 6.6 h (📏 E000 기반 추정) |
| 커밋 | 실행 시 manifest.json에 기록됨 |
| Config | [configs/train/act_orange_2cam_b8.yaml](../../configs/train/act_orange_2cam_b8.yaml) |

## 1. 설정 / Setup
```
--dataset.repo_id=edge-inference/cohab-so101-pick-orange-dataset
--dataset.root=$HOME/datasets/cohab_orange
--dataset.video_backend=pyav
--policy.type=act --policy.device=mps --policy.push_to_hub=false
--batch_size=8 --num_workers=8 --steps=70000 --log_freq=500 --save_freq=14000
--trackio.enable=false
--output_dir=runs/act_orange_2cam_b8 --job_name=act_orange_2cam_b8
```
- seed는 지정하지 않았으므로 lerobot 기본값 1000을 씀
- checkpoint 5개 (14k/28k/42k/56k/70k) × 약 0.6 GB (🔮) ≈ 3 GB. 현재 디스크 여유 약 16 GB
- 📏 스모크 참고값 ([E000](../E000_act_orange_smoke/README.md)): 2.94 step/s, `data_s` 0.003, `updt_s` 0.337, loss 25.0→3.55 (200 step)

## 2. 학습 dynamics / Training dynamics
- 실행 후 `python scripts/make_figures.py E001_act_orange_2cam_b8`로 생성함

## 3. 평가 / Evaluation
- offline action L1: **train-set 수치임** (100 ep 전체로 학습했으므로 held-out이 없음)
- sim rollout: HANDS PC, 9-grid × 50 rollouts ([04](../../docs/04_evaluation_protocol.md)) — 계획 단계

## 4. 관찰 / Observations
- (실행 후 작성)

## 5. 한계 · 주의 / Caveats
- `manifest.json`의 `git_dirty: true`는 wrapper 버그 때문임. `env/pip_freeze.txt`를 먼저 쓴 뒤 `git status`를 실행해서 생긴 값임. 실행 직전에 `git status --short`가 0줄이었음을 확인함 (커밋 `28bfe37`). 코드는 이후 수정함
- held-out split이 없음 → 과적합 여부를 판단할 수 없음. E002부터 `--dataset.eval_split=0.1`을 씀
- long-tail episode(ep33: 2564 frames, ep40: 1912 frames)가 포함됨 ([03_data](../../docs/03_data.md))

## 6. 다음 단계 / Next
- 실행 (사용자 승인 후):
```bash
tmux new -s e001   # 또는 기존 act 세션
cd ~/Desktop/Programming/aiku-hands && source ~/venvs/lerobot/bin/activate
caffeinate -is python scripts/run_train.py --exp E001 --config configs/train/act_orange_2cam_b8.yaml
```
- 계획 변경 조건
  - `step_s`가 0.34 s보다 20% 이상 느려지면 thermal/swap을 확인함 (system.jsonl)
  - 첫 checkpoint 크기가 1 GB를 넘으면 save_freq를 늘리거나 중간 ckpt 정리를 사용자에게 제안함

## 7. 재현 / Reproduce
```bash
caffeinate -is python scripts/run_train.py --exp E001 --config configs/train/act_orange_2cam_b8.yaml
python scripts/make_figures.py E001_act_orange_2cam_b8
```

## Open questions / 미해결
- 원래 계획대로 E001을 wrist+front로 돌릴지, 실제 로봇에 바로 이식할 수 있는 E002(wrist-only + held-out)를 먼저 돌릴지
