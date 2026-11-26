# 02. HANDS PC 환경 / HANDS PC Setup
> TL;DR (EN): Sim/Isaac host reached via `ssh hands`. Ubuntu 22.04, NVIDIA 580.178.04 (pinned; 595 breaks
> Isaac Sim 5.1 RTX), conda env `leisaac` (Isaac Sim 5.1.0 / Isaac Lab 2.3.0 / LeIsaac 0.4.0).
> Run sim scripts from `~/sim_tools/`, never from inside `~/lerobot_ws/`.

## 1. 사양 / Spec (2026-11-27 기준, 사용자 제공)
| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04, kernel 6.8 |
| GPU driver | NVIDIA **580.178.04** (apt pin 고정). 595 계열은 Isaac Sim 5.1 RTX와 충돌 |
| conda env | `leisaac`: Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeIsaac 0.4.0 |
| LeIsaac 소스 | `~/lerobot_ws/leisaac` |
| 정책 저장 위치 | `~/policies/<EXP_ID>/<step>/` (`scripts/sync_ckpt_to_hands.sh`) |

## 2. 규칙 / Rules
- **sim 스크립트는 `~/sim_tools/`에 두고 거기서 실행함**
  - `~/lerobot_ws/` 안에서 실행하면 repo 루트가 namespace package로 잡혀 `import leisaac`가 깨짐
- task `LeIsaac-SO101-PickOrange-v0` 생성 전 필수 호출:
```python
import leisaac.tasks.pick_orange  # registers the task
cfg = ...  # env cfg
cfg.use_teleop_device("so101leader")
cfg.recorders = None
env = gym.make("LeIsaac-SO101-PickOrange-v0", cfg=cfg)
```

## 3. Wrist camera (sim template)
- `pos=(-0.001, 0.1, -0.04)`, `rot_wxyz=(-0.404379, -0.912179, -0.0451242, 0.0486914)`, ROS convention, parent `Robot/gripper`, HFOV 53.5°
- cohab dataset은 이 template offset을 그대로 쓴 것으로 보임 (렌더 비교 육안 일치, 정량 검증 아님 🔮)
- 실제 카메라와의 차이 → [camera/wrist_offset.md](camera/wrist_offset.md)

## 4. 하드웨어 이슈 / Hardware issues
- ⚠️ 모터 보드 USB 인식 실패 (`error -71`) → 현장 수리 필요. 해결 전에는 RQ1 real 평가 불가

## Open questions / 미해결
- Mac Studio → `ssh hands` 직접 접속 가능 여부 미확인 (안 되면 MacBook 경유)
- LeIsaac `scripts/evaluation/`의 rollout 출력 형식 → `rollouts_sim.jsonl` 변환 방법
