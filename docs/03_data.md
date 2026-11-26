# 03. 데이터 / Data
> TL;DR (EN): Current training data is `edge-inference/cohab-so101-pick-orange-dataset` (LeRobot v3.0,
> 100 teleop episodes, 60,541 frames @ 30 fps, front+wrist 640×480 AV1, 6-D action/state). Episode length median
> 522.5 frames (17.4 s), with a long-tail outlier of 2564 frames. Hashes below pin the exact version.

## 1. Dataset card (📏 `python -m aiku_hands.data.inspect ~/datasets/cohab_orange`)
| 항목 | 값 |
|---|---|
| HF repo | `edge-inference/cohab-so101-pick-orange-dataset` |
| 로컬 경로 | `~/datasets/cohab_orange` (1.1 GB) |
| format | LeRobot `codebase_version=v3.0`, `robot_type=so101_follower` |
| 수집 | Isaac Sim 5.1 kitchen scene, teleop (SO-101 leader) |
| episodes / frames | 100 / 60,541 @ 30 fps |
| task (1개) | "Pick three oranges and put them into the plate, …" |
| `meta/info.json` sha256 | `94574dc76d8f9262594023b3af1a7578e11857d1693f1cde894b0a23b12f690d` |
| `meta/stats.json` sha256 | `9e083ac6636888380e8fcb8eba8e45695ad4c1b61e930df33a9bdab6d525fe41` |

## 2. Features
| key | dtype | shape | 비고 |
|---|---|---|---|
| `observation.images.front` | video (AV1) | 480×640×3 | |
| `observation.images.wrist` | video (AV1) | 480×640×3 | sim template offset 사용 추정 🔮 |
| `observation.state` | float32 | 6 | 아래 joint 순서 |
| `action` | float32 | 6 | 아래 joint 순서 |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | — | 1 | |

- joint 순서: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`.pos`)
- 값 범위: joint −100..100, gripper 0..100 (SO-101 lerobot 정규화 단위)

## 3. Action 통계 / Action stats (📏 `meta/stats.json`)
| joint | min | max | mean | std |
|---|---|---|---|---|
| shoulder_pan | −25.7 | 100.0 | 22.7 | 23.9 |
| shoulder_lift | −100.0 | 67.6 | 3.0 | 49.2 |
| elbow_flex | −100.0 | 100.0 | −3.4 | 53.4 |
| wrist_flex | −0.2 | 99.5 | 83.4 | 14.0 |
| wrist_roll | −92.4 | 100.0 | 18.2 | 20.1 |
| gripper | 0.0 | 90.6 | 28.5 | 23.8 |

- `shoulder_pan`, `elbow_flex`, `wrist_roll`이 ±100 경계에 닿음 → sim joint limit 또는 정규화 포화 가능성 (🔮 확인 필요)

## 4. Episode 길이 / Episode length (frames)
| min | p10 | median | mean | p90 | max |
|---|---|---|---|---|---|
| 354 | 415.8 | 522.5 | 605.4 | 793.9 | 2564 |

- 평균 20.2 s. 가장 긴 5개: ep33 (2564), ep40 (1912), ep32 (1287), ep30 (1164), ep11 (1015) frames
- ep33 (85 s)·ep40 (64 s)은 30 s 제한 평가 프로토콜보다 훨씬 김 → 재시도·실패 회복이 섞인 demo일 가능성 (🔮 영상 확인 필요)

## 5. Split
- E001: 100 ep 전체 학습 (held-out 없음)
- E002~: `--dataset.eval_split=0.1` → task별 **마지막** ceil(n·0.1)개 episode 홀드아웃 (task 1개이므로 ep 90–99) — 📏 소스 확인 [ADR-0003](decisions/ADR-0003-lerobot070-investigation.md)

## Open questions / 미해결
- 원본 HF revision(commit hash) 미기록 — 다운로드 경로 확인 후 `meta/info.json` 해시와 함께 고정할 것
- ep33·ep40 영상 내용 확인 후 학습에서 제외(`--dataset.exclude_episodes`)할지 결정
- sim 9-grid와 이 dataset의 물체 초기 위치 분포가 겹치는지
